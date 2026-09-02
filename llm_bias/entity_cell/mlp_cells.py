"""Qwen-compatible early MLP entity-cell hooks and compact E1 calculations.

The hook point is the input to ``down_proj``.  Qwen's SwiGLU block computes
that input after its gate and up projections, so a pre-hook observes the
post-SwiGLU intermediate without retaining the sequence activation.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from llm_bias.core.continuation_scoring import score_single_token_margin_fp32
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids

CANDIDATE_LAYERS = tuple(range(6))
EPSILON = 1e-6
ALPHA_GRID = (1.0, 0.0, -1.0, -2.0, -3.0)
DECISION_PREFIX = '{\n  "decision": "'
POSITIVE_CANDIDATE = "buy"
NEGATIVE_CANDIDATE = "sell"


def _validate_layers(layers: Iterable[int]) -> tuple[int, ...]:
    selected = tuple(sorted(set(int(layer) for layer in layers)))
    if any(layer not in CANDIDATE_LAYERS for layer in selected):
        raise ValueError("E1 MLP localization is restricted to layers L0-L5")
    return selected


def _layer_mlp(layer: Any) -> Any:
    for owner in (layer, getattr(layer, "_hf_layer", None)):
        if owner is None:
            continue
        for name in ("mlp", "feed_forward", "ffn"):
            value = getattr(owner, name, None)
            if value is not None:
                return value
    raise TypeError("decoder layer does not expose a Qwen-compatible MLP")


def _down_proj(layer: Any) -> Any:
    mlp = _layer_mlp(layer)
    for name in ("down_proj", "down", "w2"):
        value = getattr(mlp, name, None)
        if value is not None and hasattr(value, "register_forward_pre_hook"):
            return value
    # Qwen3.5-MoE puts the dense post-SwiGLU path under shared_expert; the
    # routed expert ``down_proj`` is a packed Parameter and has no hook point.
    shared = getattr(mlp, "shared_expert", None)
    if shared is not None:
        value = getattr(shared, "down_proj", None)
        if value is not None and hasattr(value, "register_forward_pre_hook"):
            return value
    raise TypeError("MLP does not expose a hookable dense down_proj")


def _as_sequence_vector(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim not in (2, 3):
        raise ValueError("down_proj input must have shape [sequence, width] or [batch, sequence, width]")
    return value if value.ndim == 3 else value.unsqueeze(0)


def qwen_mlp_down_projection(layer: Any) -> Any:
    """Return the dense Qwen SwiGLU ``down_proj`` module for one layer."""
    return _down_proj(layer)


def _register_pre_hook(module: Any, hook: Callable[..., Any]) -> Any:
    """Register the portable positional-input form used by HF and fake modules."""
    # Qwen calls down_proj(hidden_states) positionally.  Avoid the kwargs hook
    # signature here: torch versions and small test doubles disagree about the
    # return contract of ``with_kwargs=True``.
    return module.register_forward_pre_hook(hook)


@dataclass
class MLPHookSession:
    """Temporary recorder/scaler session; handles are removed on every exit."""

    model: Any
    layers: tuple[int, ...]
    position: int | None = None
    channel_scales: Mapping[int, Mapping[int, float]] | None = None
    scope: str = "all_positions"
    scope_positions: Mapping[int, Sequence[int]] | Sequence[int] | None = None

    def __post_init__(self) -> None:
        self.handles: list[Any] = []
        self.records: dict[int, torch.Tensor] = {}
        if self.scope not in {"all_positions", "header_only"}:
            raise ValueError("scope must be all_positions or header_only")
        self.layers = _validate_layers(self.layers)
        available = getattr(self.model, "layers", None)
        if available is None:
            raise TypeError("model does not expose decoder layers")
        for layer in self.layers:
            if int(layer) < 0 or int(layer) >= len(available):
                raise ValueError(f"MLP layer {layer} is out of range")

    def __enter__(self) -> "MLPHookSession":
        try:
            for layer_number in self.layers:
                layer = int(layer_number)
                module = _down_proj(self.model.layers[layer])

                def hook(_module: Any, args: tuple[Any, ...], *, layer=layer) -> Any:
                    if not args:
                        raise ValueError("down_proj pre-hook received no intermediate tensor")
                    intermediate = args[0]
                    values = _as_sequence_vector(intermediate)
                    if self.position is not None:
                        position = self.position if self.position >= 0 else values.shape[1] + self.position
                        if position < 0 or position >= values.shape[1]:
                            raise ValueError(f"selected token position {self.position} is outside the sequence")
                        selected = values[:, position, :]
                        self.records[layer] = selected[0].detach().to(device="cpu", dtype=torch.float32).clone()
                    scales = (self.channel_scales or {}).get(layer, {})
                    if not scales:
                        return None
                    result = intermediate.clone()
                    if self.scope == "all_positions":
                        positions = range(values.shape[1])
                    else:
                        requested = self.scope_positions
                        if isinstance(requested, Mapping):
                            requested = requested.get(layer, ())
                        positions = tuple(int(p) for p in (requested or ()))
                    for neuron, factor in scales.items():
                        neuron = int(neuron)
                        if neuron < 0 or neuron >= values.shape[-1]:
                            raise ValueError(f"neuron {neuron} is outside the intermediate width")
                        for pos in positions:
                            if 0 <= pos < values.shape[1]:
                                if intermediate.ndim == 2:
                                    result[pos, neuron] = values[0, pos, neuron] * float(factor)
                                else:
                                    result[:, pos, neuron] = values[:, pos, neuron] * float(factor)
                    return (result.squeeze(0),) if intermediate.ndim == 2 else (result,)

                self.handles.append(_register_pre_hook(module, hook))
        except BaseException:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


@contextmanager
def mlp_hooks(
    model: Any,
    layers: Iterable[int] = CANDIDATE_LAYERS,
    *,
    position: int | None = None,
    channel_scales: Mapping[int, Mapping[int, float]] | None = None,
    scope: str = "all_positions",
    scope_positions: Mapping[int, Sequence[int]] | Sequence[int] | None = None,
):
    """Install pre-``down_proj`` hooks and remove them on normal/error exit."""
    session = MLPHookSession(
        model, _validate_layers(layers), position,
        channel_scales, scope, scope_positions,
    )
    with session:
        yield session


def record_post_swiglu(
    model: Any,
    input_ids_tensor: torch.Tensor,
    *,
    layers: Iterable[int] = CANDIDATE_LAYERS,
    position: int = -1,
    forward: Callable[[torch.Tensor], Any] | None = None,
) -> dict[int, torch.Tensor]:
    """Run one prompt and return only selected post-SwiGLU vectors on CPU FP32."""
    if input_ids_tensor.ndim != 2 or input_ids_tensor.shape[0] != 1:
        raise ValueError("E1 selected-position recording accepts one prompt at a time")
    with torch.no_grad(), mlp_hooks(model, layers, position=position) as session:
        (forward or model.forward)(input_ids_tensor)
        return dict(session.records)


@dataclass
class OnlineVectorStats:
    """Bounded-memory per-neuron Welford statistics."""

    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def update(self, vector: torch.Tensor) -> None:
        value = vector.detach().to(device="cpu", dtype=torch.float64).flatten()
        if value.numel() == 0 or not torch.isfinite(value).all():
            raise ValueError("online statistics require a finite non-empty vector")
        if self.mean is None:
            self.mean = torch.zeros_like(value)
            self.m2 = torch.zeros_like(value)
        if value.shape != self.mean.shape:
            raise ValueError("online vectors have inconsistent widths")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> torch.Tensor:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("statistics are empty")
        return torch.sqrt(torch.clamp(self.m2 / self.count, min=0.0))

    def compact(self, *, include_moments: bool = False) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "width": 0}
        result: dict[str, Any] = {"count": self.count, "width": int(self.mean.numel())}
        if include_moments:
            result["mean"] = [float(value) for value in self.mean.tolist()]
            result["std"] = [float(value) for value in self.std.tolist()]
        return result

    @classmethod
    def from_compact(cls, value: Mapping[str, Any]) -> "OnlineVectorStats":
        mean = value.get("mean")
        std = value.get("std")
        count = int(value.get("count", 0))
        if count < 1 or not isinstance(mean, list) or not isinstance(std, list) or len(mean) != len(std):
            raise ValueError("compact statistics must contain count, mean, and std")
        result = cls(count=count, mean=torch.tensor(mean, dtype=torch.float64))
        result.m2 = torch.tensor(std, dtype=torch.float64).square() * count
        return result


def collect_generic_stats(
    model: Any,
    tokenizer: Any,
    prompts: Iterable[str],
    *,
    layers: Iterable[int] = CANDIDATE_LAYERS,
    position: int = -1,
    device: Any | None = None,
) -> dict[int, OnlineVectorStats]:
    """Accumulate baseline moments while retaining no prompt vectors."""
    selected_layers = _validate_layers(layers)
    stats = {layer: OnlineVectorStats() for layer in selected_layers}
    target = torch.device(device) if device is not None else getattr(model, "input_device", None)
    if target is None:
        try:
            target = next(model.parameters()).device
        except (AttributeError, StopIteration):
            target = torch.device("cpu")
    for prompt in prompts:
        ids = torch.tensor([input_ids(tokenizer, prompt, add_special_tokens=True)], dtype=torch.long, device=target)
        values = record_post_swiglu(model, ids, layers=selected_layers, position=position)
        for layer in selected_layers:
            stats[layer].update(values[layer])
    if any(item.count == 0 for item in stats.values()):
        raise ValueError("generic baseline is empty")
    return stats


def _candidate_key(layer: int, neuron: int) -> tuple[int, int]:
    return int(layer), int(neuron)


def rank_stability_scores(
    vectors_by_layer: Mapping[int, Iterable[torch.Tensor]],
    baseline_stats: Mapping[int, OnlineVectorStats],
    *,
    top_k: int = 5,
    epsilon: float = EPSILON,
    include_scores: bool = True,
) -> list[dict[str, float | int]]:
    """Rank all layer/neuron pairs and return only the deterministic top-k."""
    if top_k < 1 or epsilon <= 0:
        raise ValueError("top_k and epsilon must be positive")
    selected_layers = _validate_layers(vectors_by_layer.keys())
    if set(selected_layers) != set(int(layer) for layer in baseline_stats):
        raise ValueError("baseline statistics and localization layers must match")
    accumulators: dict[int, tuple[torch.Tensor | None, torch.Tensor | None, int]] = {}
    for layer in selected_layers:
        vectors = vectors_by_layer[layer]
        stats = baseline_stats[int(layer)]
        if stats.mean is None:
            raise ValueError("baseline statistics are empty")
        sums = torch.zeros_like(stats.mean, dtype=torch.float64)
        squares = torch.zeros_like(stats.mean, dtype=torch.float64)
        count = 0
        for vector in vectors:
            value = vector.detach().to("cpu", torch.float64).flatten()
            if value.shape != stats.mean.shape:
                raise ValueError("localization vector width differs from baseline")
            z = (value - stats.mean) / (stats.std + epsilon)
            sums += z
            squares += z.square()
            count += 1
        if not count:
            raise ValueError(f"no localization vectors for layer {layer}")
        accumulators[int(layer)] = (sums, squares, count)
    ranked: list[tuple[float, int, int]] = []
    for layer, (sums, squares, count) in accumulators.items():
        mean = sums / count
        std = torch.sqrt(torch.clamp(squares / count - mean.square(), min=0.0))
        score = mean.square() / (std + epsilon)
        for neuron, value in enumerate(score.tolist()):
            if math.isfinite(value):
                ranked.append((float(value), layer, neuron))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            "layer": layer,
            "neuron": neuron,
            **({"score": score} if include_scores else {}),
            "rank": rank,
        }
        for rank, (score, layer, neuron) in enumerate(ranked[:top_k], start=1)
    ]


def rank_absolute_activations(
    vectors_by_layer: Mapping[int, torch.Tensor],
    baseline_stats: Mapping[int, OnlineVectorStats],
    *,
    top_k: int = 5,
    epsilon: float = EPSILON,
) -> list[dict[str, float | int]]:
    """Rank layer/neuron pairs by |z| for a single prompt (template signature)."""
    if top_k < 1 or epsilon <= 0:
        raise ValueError("top_k and epsilon must be positive")
    selected_layers = _validate_layers(vectors_by_layer.keys())
    if set(selected_layers) != set(int(layer) for layer in baseline_stats):
        raise ValueError("baseline statistics and layers must match")
    ranked: list[tuple[float, int, int]] = []
    for layer in selected_layers:
        vector = vectors_by_layer[int(layer)].detach().to("cpu", torch.float64).flatten()
        stats = baseline_stats[int(layer)]
        if stats.mean is None:
            raise ValueError("baseline statistics are empty")
        if vector.shape != stats.mean.shape:
            raise ValueError("template vector width differs from baseline")
        z = (vector - stats.mean) / (stats.std + epsilon)
        for neuron, value in enumerate(z.abs().tolist()):
            if math.isfinite(value):
                ranked.append((float(value), int(layer), neuron))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {"layer": layer, "neuron": neuron, "abs_z": value, "rank": rank}
        for rank, (value, layer, neuron) in enumerate(ranked[:top_k], start=1)
    ]


def select_matched_random_neuron(
    stats: Mapping[int, OnlineVectorStats],
    *,
    layer: int,
    target_neuron: int,
    seed: int = 0,
) -> int:
    """Choose a same-layer neuron with the closest generic mean/std pair."""
    reference = stats[layer]
    if reference.mean is None:
        raise ValueError("statistics are empty")
    width = reference.mean.numel()
    if not 0 <= target_neuron < width:
        raise ValueError("target neuron is out of range")
    distance = (reference.mean - reference.mean[target_neuron]).square() + (
        reference.std - reference.std[target_neuron]
    ).square()
    candidates = [index for index in range(width) if index != target_neuron]
    if not candidates:
        raise ValueError("same-layer matched random control needs at least two neurons")
    digest = int(hashlib.sha256(f"{seed}:{layer}:{target_neuron}".encode()).hexdigest()[:16], 16)
    candidates.sort(key=lambda index: (float(distance[index]), (index + digest) % width))
    return candidates[0]


def _replace_header(prompt: str, ticker: str, name: str) -> str:
    import re
    prompt = re.sub(r"(?m)^(Stock Ticker: \[)[^\]\r\n]+(\])$", rf"\g<1>{ticker}\g<2>", prompt, count=1)
    return re.sub(r"(?m)^(Stock Name: \[)[^\]\r\n]+(\])$", rf"\g<1>{name}\g<2>", prompt, count=1)


def rot13_surface_form(value: str) -> str:
    import codecs
    return codecs.decode(value, "rot_13")


def surface_form_controls(prompt: str, ticker: str, name: str) -> dict[str, str]:
    """Render the three frozen header-only surface controls."""
    return {
        "anonymous_ticker": _replace_header(prompt, "ANON", name),
        "anonymous_name": _replace_header(prompt, ticker, "Anonymous Company"),
        "name_form_control": _replace_header(prompt, rot13_surface_form(ticker), rot13_surface_form(name)),
    }


# Explicit spelling for callers that distinguish prompt construction from the
# later ranking summary.  Both names return fresh strings and mutate only the
# two header fields.
build_surface_form_controls = surface_form_controls


def _default_margin(model: Any, tokenizer: Any, prompt: str, *, device: Any | None = None) -> float:
    formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False) + DECISION_PREFIX
    return float(score_single_token_margin_fp32(model, tokenizer, formatted, POSITIVE_CANDIDATE, NEGATIVE_CANDIDATE, device=device).value)


def anonymous_progress(margin: float, clean_margin: float, anonymous_margin: float, *, epsilon: float = EPSILON) -> float:
    gap = anonymous_margin - clean_margin
    return (margin - clean_margin) * gap / (gap * gap + epsilon)


def run_amnesia_curve(
    model: Any,
    tokenizer: Any,
    *,
    prompt_rows: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    wrong_candidate: Mapping[str, Any],
    random_neuron: int,
    wrong_neuron: int | None = None,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    device: Any | None = None,
    score_fn: Callable[[str], float] | None = None,
    wrong_degenerate: bool = False,
) -> list[dict[str, Any]]:
    """Run target, wrong-cell, and matched random-neuron compact dose curves."""
    score = score_fn or (lambda prompt: _default_margin(model, tokenizer, prompt, device=device))
    layer, neuron = int(candidate["layer"]), int(candidate["neuron"])
    wrong_layer = int(wrong_candidate["layer"])
    wrong_neuron = int(wrong_neuron if wrong_neuron is not None else wrong_candidate["neuron"])
    output: list[dict[str, Any]] = []
    for row in prompt_rows:
        prompt = str(row["prompt"])
        controls = surface_form_controls(prompt, str(row["ticker"]), str(row["name"]))
        clean = score(prompt)
        anonymous = score(_replace_header(prompt, "ANON", "Anonymous Company"))
        gap = anonymous - clean
        eligible = abs(gap) >= 0.1
        exclusion_reason = None if eligible else "anonymous_gap_below_0.1"
        scopes = {"all_positions": None, "header_only": ()}
        identity_groups = row.get("source_groups", {}).get("identity_header", {})
        header_positions = tuple(
            position
            for start, end in identity_groups.get("ranges", ())
            for position in range(int(start), int(end))
        )
        for scope, positions in scopes.items():
            if scope == "header_only":
                positions = header_positions
            for alpha in alpha_grid:
                for label, cell_layer, cell_neuron in (
                    (("target", layer, neuron),)
                    + (() if wrong_degenerate else (("wrong_entity", wrong_layer, wrong_neuron),))
                    + (("matched_random", layer, random_neuron),)
                ):
                    with mlp_hooks(
                        model,
                        [cell_layer],
                        channel_scales={cell_layer: {cell_neuron: float(alpha)}},
                        scope=scope,
                        scope_positions=positions,
                    ):
                        margin = score(prompt)
                    output.append({
                        "ticker": row["ticker"], "prompt_id": row.get("prompt_id"),
                        "candidate": label, "scope": scope, "alpha": float(alpha),
                        "margin": float(margin), "clean_margin": float(clean),
                        "anonymous_margin": float(anonymous),
                        "anonymous_gap": float(gap), "eligible": bool(eligible),
                        "exclusion_reason": exclusion_reason,
                        "anonymous_progress": float(anonymous_progress(margin, clean, anonymous)),
                        "surface_controls": {name: float(score(value)) for name, value in controls.items()},
                        "wrong_entity_layer": wrong_layer,
                        "wrong_entity_neuron": wrong_neuron,
                        "wrong_entity_degenerate": bool(wrong_degenerate),
                        "matched_random_neuron": int(random_neuron),
                    })
    return output


record_pre_down_proj = record_post_swiglu
scale_mlp_channels = mlp_hooks

compute_online_stats = collect_generic_stats
compute_stability_scores = rank_stability_scores
amnesia_progress = anonymous_progress
render_surface_controls = surface_form_controls
install_mlp_hooks = mlp_hooks
record_mlp_intermediates = record_post_swiglu

__all__ = [
    "ALPHA_GRID", "CANDIDATE_LAYERS", "DECISION_PREFIX", "EPSILON",
    "MLPHookSession", "OnlineVectorStats", "anonymous_progress",
    "qwen_mlp_down_projection", "record_pre_down_proj", "scale_mlp_channels",
    "collect_generic_stats", "mlp_hooks", "rank_absolute_activations", "rank_stability_scores",
    "record_post_swiglu", "rot13_surface_form", "run_amnesia_curve",
    "select_matched_random_neuron", "surface_form_controls", "build_surface_form_controls",
    "compute_online_stats", "compute_stability_scores", "amnesia_progress",
    "render_surface_controls", "install_mlp_hooks", "record_mlp_intermediates",
]
