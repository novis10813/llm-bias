"""E3 compact suppression mechanics.

The module keeps intervention tensors in the forward pass.  Public helpers
return scalar diagnostics, dose metadata, and exclusion reasons only.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


from .attention_attribution import (
    SOURCE_GROUPS,
    _apply_rope,
    _attention_module,
    _norm,
    _positions,
    _query_mask,
    _weight,
)

E3_ALPHA_GRID = (1.0, 0.5, 0.0, -1.0, -2.0, -3.0)
E3_BETA_GRID = (1.0, 0.75, 0.5, 0.25, 0.0)
SOURCE_NORM_EPSILON = 1e-6
DECISION_PREFIX = '{\n  "decision": "'
POSITIVE_CANDIDATE = "buy"
NEGATIVE_CANDIDATE = "sell"


def _finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite tensor")
    return value.float()


def _norm_value(value: torch.Tensor) -> float:
    return float(_finite(value, "source vector").norm().detach().cpu())


def validate_dose_grid(values: Sequence[float], *, name: str, lower: float | None = None, upper: float | None = None) -> tuple[float, ...]:
    """Validate and normalize a frozen finite dose grid."""
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite doses")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate doses")
    if lower is not None and any(value < lower for value in result):
        raise ValueError(f"{name} contains a dose below {lower}")
    if upper is not None and any(value > upper for value in result):
        raise ValueError(f"{name} contains a dose above {upper}")
    return result


def norm_matched_source_delta(
    target_delta: torch.Tensor,
    source_direction: torch.Tensor,
    *,
    epsilon: float = SOURCE_NORM_EPSILON,
) -> tuple[torch.Tensor | None, bool, str | None]:
    """Return the frozen norm-matched control perturbation.

    The caller must keep a low-norm source excluded.  It must not select a
    replacement source after seeing the norm.
    """
    target_delta = _finite(target_delta, "target_delta")
    source_direction = _finite(source_direction, "source_direction")
    if target_delta.shape != source_direction.shape:
        raise ValueError("target and source directions must have the same shape")
    if epsilon <= 0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be a positive finite number")
    source_norm = source_direction.norm()
    if float(source_norm) < epsilon:
        return None, False, "source_direction_norm_below_epsilon"
    delta = -target_delta.norm() * source_direction / (source_norm + epsilon)
    return delta, True, None


def matched_perturbation_norm(target_delta: torch.Tensor, source_direction: torch.Tensor, *, epsilon: float = SOURCE_NORM_EPSILON) -> dict[str, Any]:
    delta, eligible, reason = norm_matched_source_delta(target_delta, source_direction, epsilon=epsilon)
    return {
        "eligible": eligible,
        "exclusion_reason": reason,
        "target_norm": _norm_value(target_delta),
        "source_norm": _norm_value(source_direction),
        "matched_norm": None if delta is None else _norm_value(delta),
        "formula": f"-||target_delta|| source_direction/(||source_direction||+{float(epsilon):g})",
    }


def deterministic_source_subset(token_count: int, source_positions: Sequence[int], *, seed: int = 0) -> tuple[int, ...]:
    """Select a deterministic same-count subset without replacement."""
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    values = sorted({int(value) for value in source_positions})
    if token_count > len(values):
        raise ValueError("source positions cannot supply the requested token count")
    digest = hashlib.sha256(str(int(seed)).encode()).digest()
    ranked = sorted(values, key=lambda value: (hashlib.sha256(digest + str(value).encode()).digest(), value))
    return tuple(sorted(ranked[:token_count]))


def _attention_dimensions(attention: Any) -> tuple[int, int, int]:
    q_weight = _weight(attention.q_proj)
    k_weight = _weight(attention.k_proj)
    v_weight = _weight(attention.v_proj)
    head_dim = int(getattr(attention, "head_dim", 0) or getattr(getattr(attention, "config", None), "head_dim", 0) or q_weight.shape[0] // 32)
    heads = int(getattr(attention, "num_heads", 0) or getattr(getattr(attention, "config", None), "num_attention_heads", 0) or q_weight.shape[0] // (2 * head_dim))
    kv = int(getattr(attention, "num_key_value_heads", 0) or getattr(getattr(attention, "config", None), "num_key_value_heads", 0) or k_weight.shape[0] // head_dim)
    if heads < 1 or kv < 1 or heads % kv:
        raise ValueError("attention heads must be divisible by key/value heads")
    if q_weight.shape[0] != 2 * heads * head_dim or k_weight.shape[0] != kv * head_dim or v_weight.shape[0] != kv * head_dim:
        raise ValueError("attention projections do not match declared dimensions")
    return head_dim, heads, kv


def source_resolved_head_inputs(
    attention: Any,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: Any = None,
    attention_mask: torch.Tensor | None = None,
    source_groups: Mapping[str, Any],
    query_position: int = -1,
) -> dict[int, dict[str, torch.Tensor]]:
    """Reconstruct transient gated per-head source vectors before ``o_proj``."""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("source reconstruction expects one [batch, sequence, hidden] input")
    batch, sequence, _ = hidden_states.shape
    query = query_position if query_position >= 0 else sequence + query_position
    if not 0 <= query < sequence:
        raise ValueError("query position is outside the sequence")
    head_dim, heads, kv = _attention_dimensions(attention)
    q, gate = torch.chunk(attention.q_proj(hidden_states).view(batch, sequence, heads, 2 * head_dim), 2, dim=-1)
    k = attention.k_proj(hidden_states).view(batch, sequence, kv, head_dim)
    v = attention.v_proj(hidden_states).view(batch, sequence, kv, head_dim)
    q = _norm(getattr(attention, "q_norm", None), q).transpose(1, 2)
    k = _norm(getattr(attention, "k_norm", None), k).transpose(1, 2)
    v = v.transpose(1, 2)
    q, k = _apply_rope(q, k, position_embeddings)
    repeat = heads // kv
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(q[:, :, query:query + 1].float(), k.float().transpose(-1, -2))
    scores *= float(getattr(attention, "scaling", head_dim ** -0.5))
    scores += _query_mask(attention_mask, batch=batch, query=query, sequence=sequence, device=scores.device).unsqueeze(1)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)[:, :, 0, :]
    gates = torch.sigmoid(gate[:, query]).to(v.dtype)
    positions = _positions(source_groups, sequence, query)
    result: dict[int, dict[str, torch.Tensor]] = {}
    for head in range(heads):
        result[head] = {}
        for group in SOURCE_GROUPS:
            indices = list(positions[group])
            result[head][group] = (weights[:, head, indices].unsqueeze(-1) * v[:, head, indices]).sum(dim=1)[0] * gates[0, head]
    return result


# Descriptive aliases used by callers that use the E2 terminology.
reconstruct_source_components = source_resolved_head_inputs
_raw_source_contributions = source_resolved_head_inputs


def _concat_vectors(vectors: Mapping[int, torch.Tensor], heads: Sequence[int]) -> torch.Tensor:
    return torch.cat([_finite(vectors[head], "source vector") for head in heads])


def compute_source_attenuation_deltas(
    source_vectors: Mapping[int, Mapping[str, torch.Tensor]],
    heads: Iterable[int],
    *,
    beta: float,
    mode: str = "identity",
    grouping: str = "single",
    source_positions: Sequence[int] = (),
    identity_token_count: int | None = None,
    seed: int = 0,
    epsilon: float = SOURCE_NORM_EPSILON,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Compute identity, source-control, or whole-head deltas without hooks."""
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must be between zero and one")
    if mode not in {"identity", "evidence", "random_subset", "whole_head"}:
        raise ValueError("unsupported source attenuation mode")
    if grouping not in {"single", "group"}:
        raise ValueError("grouping must be single or group")
    selected = tuple(dict.fromkeys(int(head) for head in heads))
    if not selected:
        raise ValueError("at least one attention head is required")
    if any(head not in source_vectors for head in selected):
        raise ValueError("source vectors are missing a selected head")
    identity = {head: _finite(source_vectors[head]["identity_header"], "identity source") for head in selected}
    target = {head: -(1.0 - float(beta)) * identity[head] for head in selected}
    controls: dict[str, Any] = {"mode": mode, "grouping": grouping, "eligible": True, "exclusion_reason": None}
    if mode == "identity":
        return target, controls
    if mode == "whole_head":
        return {
            head: -(1.0 - float(beta)) * sum((_finite(source_vectors[head][name], name) for name in SOURCE_GROUPS), torch.zeros_like(identity[head]))
            for head in selected
        }, controls
    if mode == "evidence":
        directions = {head: _finite(source_vectors[head]["evidence"], "evidence source") for head in selected}
    else:
        counts = identity_token_count if identity_token_count is not None else len(tuple(source_positions))
        if counts < 0:
            raise ValueError("identity_token_count must be non-negative")
        subset = deterministic_source_subset(counts, source_positions, seed=seed)
        controls["random_subset_token_count"] = len(subset)
        directions = {}
        # ``other_prefix`` carries the subset temporarily.  Source vectors are
        # already reconstructed per group, so the random direction needs a
        # caller-provided source vector for arbitrary positions.
        if "random_subset" not in source_vectors:
            raise ValueError("random_subset requires source_vectors['random_subset']")
        random_vectors = source_vectors["random_subset"]
        directions = {head: _finite(random_vectors[head], "random source") for head in selected}
    if grouping == "single":
        result: dict[int, torch.Tensor] = {}
        eligibility: dict[str, Any] = {}
        for head in selected:
            delta, eligible, reason = norm_matched_source_delta(target[head], directions[head], epsilon=epsilon)
            eligibility[str(head)] = {"eligible": eligible, "exclusion_reason": reason, "target_norm": _norm_value(target[head]), "source_norm": _norm_value(directions[head]), "matched_norm": None if delta is None else _norm_value(delta)}
            result[head] = torch.zeros_like(target[head]) if delta is None else delta
        controls["heads"] = eligibility
        controls["eligible"] = all(item["eligible"] for item in eligibility.values())
        if not controls["eligible"]:
            controls["exclusion_reason"] = "one_or_more_selected_heads_ineligible"
        return result, controls
    target_cat = _concat_vectors(target, selected)
    direction_cat = _concat_vectors(directions, selected)
    delta, eligible, reason = norm_matched_source_delta(target_cat, direction_cat, epsilon=epsilon)
    controls.update({"eligible": eligible, "exclusion_reason": reason, "target_norm": _norm_value(target_cat), "source_norm": _norm_value(direction_cat), "matched_norm": None if delta is None else _norm_value(delta)})
    if delta is None:
        return {head: torch.zeros_like(target[head]) for head in selected}, controls
    result = {}
    offset = 0
    for head in selected:
        width = target[head].numel()
        result[head] = delta[offset:offset + width].reshape_as(target[head])
        offset += width
    return result, controls


@dataclass
class AttenuationControl:
    mode: str
    eligible: bool
    exclusion_reason: str | None
    target_norm: float
    source_norm: float | None
    matched_norm: float | None
    grouping: str = "single"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "eligible": self.eligible, "exclusion_reason": self.exclusion_reason, "target_norm": self.target_norm, "source_norm": self.source_norm, "matched_norm": self.matched_norm, "grouping": self.grouping}


class SourceAttenuationSession:
    """Temporarily attenuate selected source groups at selected heads."""

    def __init__(self, model: Any, source_groups_by_layer: Mapping[int, Mapping[str, Any]], heads: Iterable[tuple[int, int]], *, beta: float = 1.0, mode: str = "identity", grouping: str = "single", seed: int = 0, epsilon: float = SOURCE_NORM_EPSILON, query_position: int = -1):
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be between zero and one")
        if mode not in {"identity", "evidence", "random_subset", "whole_head"}:
            raise ValueError("unsupported source attenuation mode")
        if grouping not in {"single", "group"}:
            raise ValueError("grouping must be single or group")
        self.model = model
        self.source_groups_by_layer = {int(layer): groups for layer, groups in source_groups_by_layer.items()}
        self.heads = tuple((int(layer), int(head)) for layer, head in heads)
        if not self.heads:
            raise ValueError("at least one attention head is required")
        self.beta, self.mode, self.grouping, self.seed, self.epsilon = float(beta), mode, grouping, int(seed), float(epsilon)
        self.query_position = int(query_position)
        self.margin_direction = None
        self.handles: list[Any] = []
        self.controls: dict[str, Any] = {}
        self.deltas: dict[tuple[int, int], float] = {}
        self.contribution_deltas: dict[tuple[int, int], dict[str, float]] = {}

    def __enter__(self) -> "SourceAttenuationSession":
        layers = getattr(self.model, "layers", None) or getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise TypeError("model does not expose decoder layers")
        by_layer: defaultdict[int, list[int]] = defaultdict(list)
        for layer, head in self.heads:
            if layer not in self.source_groups_by_layer:
                raise ValueError(f"source groups are missing for layer {layer}")
            if head not in by_layer[layer]:
                by_layer[layer].append(head)
        try:
            for layer, selected_heads in by_layer.items():
                attention = _attention_module(layers[layer])
                capture: dict[str, Any] = {}

                def before(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None, *, capture=capture) -> None:
                    values = kwargs or {}
                    hidden = args[0] if args else values.get("hidden_states")
                    if not torch.is_tensor(hidden):
                        raise ValueError("attention forward received no hidden states")
                    capture["hidden"] = hidden
                    capture["position_embeddings"] = values.get("position_embeddings", args[1] if len(args) > 1 else None)
                    capture["attention_mask"] = values.get("attention_mask", args[2] if len(args) > 2 else None)

                def alter(_module: Any, args: tuple[Any, ...], *, layer=layer, selected_heads=tuple(selected_heads), attention=attention, capture=capture) -> tuple[torch.Tensor]:
                    if not args or not torch.is_tensor(args[0]) or args[0].ndim != 3:
                        raise ValueError("o_proj input must be [batch, sequence, heads*head_dim]")
                    if "hidden" not in capture:
                        return args
                    value = args[0].clone()
                    query = self.query_position if self.query_position >= 0 else value.shape[1] - 1
                    raw = source_resolved_head_inputs(attention, capture["hidden"], position_embeddings=capture["position_embeddings"], attention_mask=capture["attention_mask"], source_groups=self.source_groups_by_layer[layer], query_position=query)
                    selected = tuple(selected_heads)
                    random_vectors: dict[int, torch.Tensor] = {}
                    if self.mode == "random_subset":
                        groups = _positions(self.source_groups_by_layer[layer], value.shape[1], query)
                        identity_positions = set(groups["identity_header"])
                        pool = tuple(position for position in range(query) if position not in identity_positions)
                        identity_count = len(groups["identity_header"])
                        subset = deterministic_source_subset(identity_count, pool, seed=self.seed + layer)
                        # Reconstruct a same-token-count non-identity source
                        # subset while keeping the four groups disjoint.
                        subset_set = set(subset)
                        subset_groups = {name: {"ranges": []} for name in SOURCE_GROUPS}
                        subset_groups["identity_header"]["ranges"] = [[pos, pos + 1] for pos in subset]
                        for name in ("evidence", "instruction_context", "other_prefix"):
                            remaining = [pos for pos in groups[name] if pos < query and pos not in subset_set]
                            subset_groups[name]["ranges"] = [[pos, pos + 1] for pos in remaining]
                        subset_raw = source_resolved_head_inputs(attention, capture["hidden"], position_embeddings=capture["position_embeddings"], attention_mask=capture["attention_mask"], source_groups=subset_groups, query_position=query)
                        random_vectors = {head: subset_raw[head]["identity_header"] for head in selected}
                    vectors = {head: raw[head] for head in selected}
                    if self.mode == "random_subset":
                        vectors["random_subset"] = random_vectors
                    deltas, diagnostics = compute_source_attenuation_deltas(vectors, selected, beta=self.beta, mode=self.mode, grouping=self.grouping, seed=self.seed, epsilon=self.epsilon, source_positions=tuple(range(query)), identity_token_count=len(_positions(self.source_groups_by_layer[layer], value.shape[1], query)["identity_header"]))
                    self.controls[f"L{layer}"] = diagnostics
                    head_dim, num_heads, _ = _attention_dimensions(attention)
                    if self.margin_direction is not None:
                        direction = self.margin_direction.to(device=value.device, dtype=torch.float32).flatten()
                        weight = _weight(attention.o_proj).float()
                        for head, delta in deltas.items():
                            start, end = head * head_dim, (head + 1) * head_dim
                            projected = F.linear(delta.float().unsqueeze(0), weight[:, start:end], None)[0]
                            if projected.numel() != direction.numel():
                                raise ValueError("margin direction width differs from projected head output")
                            self.deltas[(layer, head)] = float(torch.dot(projected, direction).detach().cpu())
                            group_deltas: dict[str, float] = {}
                            for group in SOURCE_GROUPS:
                                source = raw[head][group]
                                if self.mode == "identity":
                                    change = deltas[head] if group == "identity_header" else torch.zeros_like(source)
                                elif self.mode == "evidence":
                                    change = deltas[head] if group == "evidence" else torch.zeros_like(source)
                                elif self.mode == "whole_head":
                                    change = -(1.0 - self.beta) * source
                                else:
                                    change = torch.zeros_like(source)
                                source_projected = F.linear(change.float().unsqueeze(0), weight[:, start:end], None)[0]
                                group_deltas[group] = float(torch.dot(source_projected, direction).detach().cpu())
                            if self.mode == "random_subset":
                                random_projected = F.linear(deltas[head].float().unsqueeze(0), weight[:, start:end], None)[0]
                                group_deltas["random_subset"] = float(torch.dot(random_projected, direction).detach().cpu())
                            self.contribution_deltas[(layer, head)] = group_deltas
                    for head, delta in deltas.items():
                        start, end = head * head_dim, (head + 1) * head_dim
                        if end > value.shape[-1]:
                            raise ValueError("selected head slice is outside o_proj input width")
                        value[:, query, start:end] = value[:, query, start:end] + delta.to(device=value.device, dtype=value.dtype)
                    return (value,)

                try:
                    self.handles.append(attention.register_forward_pre_hook(before, with_kwargs=True))
                except TypeError:
                    self.handles.append(attention.register_forward_pre_hook(lambda module, args, callback=before: callback(module, args, {})))
                self.handles.append(attention.o_proj.register_forward_pre_hook(alter))
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


@contextmanager
def attenuate_identity_sources(model: Any, source_groups_by_layer: Mapping[int, Mapping[str, Any]], heads: Iterable[tuple[int, int]], *, beta: float, mode: str = "identity", grouping: str = "single", seed: int = 0, epsilon: float = SOURCE_NORM_EPSILON, query_position: int = -1, margin_direction: torch.Tensor | None = None):
    session = SourceAttenuationSession(model, source_groups_by_layer, heads, beta=beta, mode=mode, grouping=grouping, seed=seed, epsilon=epsilon, query_position=query_position)
    session.margin_direction = margin_direction
    with session:
        yield session


def whole_head_upper_bound_beta(beta: float) -> float:
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must be between zero and one")
    return float(beta)


def e3_progress(margin: float, clean_margin: float, anonymous_margin: float, *, epsilon: float = 1e-6) -> float:
    gap = float(anonymous_margin) - float(clean_margin)
    return (float(margin) - float(clean_margin)) * gap / (gap * gap + epsilon)


_RAW_FIELDS = {"activation", "activations", "gradient", "gradients", "residual", "residuals", "hidden_state", "hidden_states", "jacobian", "jacobians", "kv_cache", "recurrent_state"}


def _compact_value(value: Any, *, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in _RAW_FIELDS or any(part in normalized.split("_") for part in _RAW_FIELDS):
        raise ValueError(f"raw runtime payloads are not supported: {key}")
    if torch.is_tensor(value) or hasattr(value, "detach"):
        raise ValueError("tensor payloads are not supported")
    if value.__class__.__module__.startswith("numpy"):
        raise ValueError("ndarray payloads are not supported")
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite artifact values are not supported")
        return value
    if isinstance(value, Mapping):
        return {str(name): _compact_value(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_value(item, key=key) for item in value]
    raise TypeError(f"unsupported compact value type: {type(value).__name__}")


def e3_compact_record(*, ticker: str, prompt_id: str, phase: str, scope: str, dose: float, margin: float, clean_margin: float, anonymous_margin: float, flip: bool, contributions: Mapping[str, Mapping[str, float]], controls: Mapping[str, Any] | None = None, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    values = (dose, margin, clean_margin, anonymous_margin)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("E3 metrics must be finite")
    compact_contributions = _compact_value(contributions, key="contributions")
    if not isinstance(compact_contributions, dict):
        raise TypeError("contributions must be a mapping")
    return {
        "schema_version": 1,
        "artifact_type": "entity_cell_e3_suppression",
        "ticker": str(ticker), "prompt_id": str(prompt_id), "phase": str(phase), "scope": str(scope), "dose": float(dose),
        "margin": float(margin), "clean_margin": float(clean_margin), "anonymous_margin": float(anonymous_margin), "anonymous_progress": float(e3_progress(margin, clean_margin, anonymous_margin)), "flip": bool(flip),
        "contributions": compact_contributions,
        "controls": _compact_value(dict(controls or {}), key="controls"), "provenance": {**_compact_value(dict(provenance or {}), key="provenance"), "raw_runtime_payloads": False},
        "fixed_continuation": {"positive": POSITIVE_CANDIDATE, "negative": NEGATIVE_CANDIDATE},
    }


def preservation_metrics(clean: Mapping[str, float], intervened: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in SOURCE_GROUPS:
        before, after = float(clean.get(group, 0.0)), float(intervened.get(group, 0.0))
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("preservation metrics require finite values")
        result[group] = {"clean": before, "intervened": after, "delta": after - before, "absolute_delta": abs(after - before)}
    result["identity_vs_evidence_delta_ratio"] = abs(result["identity_header"]["delta"]) / max(abs(result["evidence"]["delta"]), SOURCE_NORM_EPSILON)
    result["identity_vs_instruction_delta_ratio"] = abs(result["identity_header"]["delta"]) / max(abs(result["instruction_context"]["delta"]), SOURCE_NORM_EPSILON)
    return result


__all__ = [
    "E3_ALPHA_GRID", "E3_BETA_GRID", "SOURCE_NORM_EPSILON", "AttenuationControl", "SourceAttenuationSession", "attenuate_identity_sources", "compute_source_attenuation_deltas", "deterministic_source_subset", "e3_compact_record", "e3_progress", "matched_perturbation_norm", "norm_matched_source_delta", "preservation_metrics", "reconstruct_source_components", "source_resolved_head_inputs", "validate_dose_grid", "whole_head_upper_bound_beta",
]
