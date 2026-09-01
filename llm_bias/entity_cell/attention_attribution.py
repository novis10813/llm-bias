"""Qwen3.5 full-attention source attribution and causal head controls."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from llm_bias.core.continuation_scoring import continuation_token_ids, fp32_next_token_logits

FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
PRIMARY_ATTENTION_LAYERS = (11, 15, 19)
SOURCE_GROUPS = ("identity_header", "evidence", "instruction_context", "other_prefix")
ROUTING_EPSILON_FLOOR = 1e-4
RECONSTRUCTION_ATOL = 2e-4
RECONSTRUCTION_RTOL = 2e-4
SELECTION_TOP_K = 5


def _attention_module(layer: Any) -> Any:
    if getattr(layer, "block_type", None) is not None and str(layer.block_type) != "full_attention":
        raise ValueError("E2 attribution accepts full-attention layers only")
    for owner in (layer, getattr(layer, "_hf_layer", None)):
        module = getattr(owner, "self_attn", None) if owner is not None else None
        if module is not None:
            return module
    raise ValueError("requested layer is not a Qwen3.5 full-attention layer")


def validate_attention_layers(layers: Iterable[int]) -> tuple[int, ...]:
    selected = tuple(sorted(set(int(layer) for layer in layers)))
    if not selected or any(layer not in FULL_ATTENTION_LAYERS for layer in selected):
        raise ValueError("E2 layers must be one of L3/L7/L11/L15/L19/L23/L27/L31")
    return selected


def _weight(module: Any) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise TypeError("attention projection must expose a rank-2 weight")
    return weight


def _norm(module: Any, value: torch.Tensor) -> torch.Tensor:
    if module is None:
        return value
    output = module(value)
    if not torch.is_tensor(output) or output.shape != value.shape:
        raise ValueError("Q/K norm changed the expected tensor shape")
    return output


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    left, right = value[..., : value.shape[-1] // 2], value[..., value.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, position_embeddings: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if position_embeddings is None:
        return q, k
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        raise ValueError("position_embeddings must be a (cos, sin) pair")
    cos, sin = position_embeddings
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.ndim != 3 or sin.shape != cos.shape or cos.shape[0] not in (1, q.shape[0]) or cos.shape[1] < q.shape[2]:
        raise ValueError("position embeddings do not cover the attention sequence")
    cos, sin = cos[:, : q.shape[2]].to(q), sin[:, : q.shape[2]].to(q)
    rotary_dim = min(cos.shape[-1], q.shape[-1])
    if rotary_dim % 2:
        raise ValueError("rotary dimension must be even")
    cos, sin = cos[..., :rotary_dim].unsqueeze(1), sin[..., :rotary_dim].unsqueeze(1)
    q_rot, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_tail), dim=-1)
    k = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_tail), dim=-1)
    return q, k


def _query_mask(mask: torch.Tensor | None, *, batch: int, query: int, sequence: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        result = torch.full((batch, 1, sequence), float("-inf"), device=device, dtype=torch.float32)
        result[..., : query + 1] = 0
        return result
    value = mask.to(device=device)
    if value.ndim == 2:
        if value.shape[0] != batch or value.shape[1] < sequence:
            raise ValueError("attention mask has incompatible shape")
        # Public 2-D masks use one for valid tokens, not additive values.
        result = torch.where(value[:, None, :sequence].bool(), torch.zeros(1, device=device), torch.full((1,), float("-inf"), device=device))
        causal = torch.arange(sequence, device=device) <= query
        return result.masked_fill(~causal[None, None, :], float("-inf"))
    if value.ndim == 4:
        value = value[:, :, query, :sequence]
        if value.shape[1] != 1:
            value = value[:, :1]
    elif value.ndim == 3:
        value = value[:, query, :sequence]
        if value.ndim == 2:
            value = value[:, None, :]
    if value.ndim != 3 or value.shape[0] != batch or value.shape[-1] != sequence:
        raise ValueError("unsupported Qwen3.5 attention-mask shape")
    return value.float()


def _positions(source_groups: Mapping[str, Any], sequence: int, query: int) -> dict[str, tuple[int, ...]]:
    if set(source_groups) != set(SOURCE_GROUPS):
        raise ValueError("source groups must contain the four frozen E2 groups")
    result: dict[str, tuple[int, ...]] = {}
    used: set[int] = set()
    for name in SOURCE_GROUPS:
        group = source_groups[name]
        ranges = group.get("ranges") if isinstance(group, Mapping) else None
        if not isinstance(ranges, Sequence):
            raise ValueError(f"source group {name} is missing ranges")
        values: list[int] = []
        for pair in ranges:
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("source ranges must be [start, end] pairs")
            start, end = int(pair[0]), int(pair[1])
            if start < 0 or end <= start or end > query or end > sequence:
                raise ValueError("source range is outside the prefix before the final query")
            values.extend(range(start, end))
        if len(set(values)) != len(values) or used.intersection(values):
            raise ValueError("source groups must be disjoint")
        used.update(values)
        result[name] = tuple(values)
    if used != set(range(query)):
        raise ValueError("source groups must cover every position before the final query")
    # HF's causal attention includes the query token itself.  T1 ranges cover
    # source positions strictly before the query, so keep the self position in
    # the residual "other_prefix" component without changing the prepared
    # range contract.
    result["other_prefix"] = (*result["other_prefix"], query)
    return result


def _project(o_proj: Any, head: int, value: torch.Tensor, head_dim: int) -> torch.Tensor:
    weight = _weight(o_proj).float()
    start, end = head * head_dim, (head + 1) * head_dim
    if end > weight.shape[1]:
        raise ValueError("head slice is outside o_proj input width")
    return F.linear(value.float(), weight[:, start:end], None)


@dataclass
class AttentionReconstruction:
    head_group_vectors: dict[int, dict[str, torch.Tensor]]
    head_group_margins: dict[int, dict[str, float]]
    reconstructed_output: torch.Tensor
    clean_output: torch.Tensor | None
    max_abs_error: float
    relative_error: float
    additive: bool


def reconstruct_attention_components(
    attention: Any,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: Any = None,
    attention_mask: torch.Tensor | None = None,
    source_groups: Mapping[str, Any],
    query_position: int = -1,
    clean_output: torch.Tensor | None = None,
    atol: float = RECONSTRUCTION_ATOL,
    rtol: float = RECONSTRUCTION_RTOL,
) -> AttentionReconstruction:
    """Reconstruct final-position per-head vectors grouped by prepared source ranges."""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("reconstruction expects one [batch, sequence, hidden] input")
    batch, sequence, _ = hidden_states.shape
    query = query_position if query_position >= 0 else sequence + query_position
    if not 0 <= query < sequence:
        raise ValueError("query position is outside the sequence")
    q_proj, k_proj, v_proj, o_proj = (getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if any(module is None for module in (q_proj, k_proj, v_proj, o_proj)):
        raise TypeError("attention module does not expose Qwen3.5 projections")
    q_weight, k_weight, v_weight = (_weight(module) for module in (q_proj, k_proj, v_proj))
    config = getattr(attention, "config", None)
    head_dim = int(getattr(attention, "head_dim", 0) or getattr(config, "head_dim", 0) or q_weight.shape[0] // 32)
    num_heads = int(getattr(attention, "num_heads", 0) or getattr(config, "num_attention_heads", 0) or q_weight.shape[0] // (2 * head_dim))
    num_kv = int(getattr(attention, "num_key_value_heads", 0) or getattr(config, "num_key_value_heads", 0) or k_weight.shape[0] // head_dim)
    if q_weight.shape[0] != 2 * num_heads * head_dim or k_weight.shape[0] != num_kv * head_dim or v_weight.shape[0] != num_kv * head_dim:
        raise ValueError("attention projections do not match Qwen3.5 dimensions")
    if num_heads % num_kv:
        raise ValueError("Qwen3.5 GQA requires attention heads divisible by KV heads")
    x = hidden_states
    query_states, gate = torch.chunk(q_proj(x).view(batch, sequence, num_heads, 2 * head_dim), 2, dim=-1)
    key_states = k_proj(x).view(batch, sequence, num_kv, head_dim)
    value_states = v_proj(x).view(batch, sequence, num_kv, head_dim)
    query_states = _norm(getattr(attention, "q_norm", None), query_states).transpose(1, 2)
    key_states = _norm(getattr(attention, "k_norm", None), key_states).transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    query_states, key_states = _apply_rope(query_states, key_states, position_embeddings)
    repeat = num_heads // num_kv
    key_states = key_states.repeat_interleave(repeat, dim=1)
    value_states = value_states.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(query_states[:, :, query : query + 1].float(), key_states.float().transpose(-1, -2))
    scores *= float(getattr(attention, "scaling", head_dim ** -0.5))
    scores = scores + _query_mask(attention_mask, batch=batch, query=query, sequence=sequence, device=scores.device).unsqueeze(1)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value_states.dtype)[:, :, 0, :]
    gates = torch.sigmoid(gate[:, query]).to(value_states.dtype)
    source_positions = _positions(source_groups, sequence, query)
    vectors: dict[int, dict[str, torch.Tensor]] = {}
    margins: dict[int, dict[str, float]] = {}
    reconstructed = torch.zeros(_weight(o_proj).shape[0], device=x.device, dtype=torch.float32)
    for head in range(num_heads):
        vectors[head], margins[head] = {}, {}
        for group, positions in source_positions.items():
            indices = list(positions)
            head_value = (weights[:, head, indices].unsqueeze(-1) * value_states[:, head, indices]).sum(dim=1)
            # Qwen3.5 applies the query-dependent sigmoid gate to each head
            # after attention aggregation and before concatenation/o_proj.
            gated_value = head_value * gates[:, head]
            projected = _project(o_proj, head, gated_value, head_dim)[0]
            # A projection bias is not source-position dependent.  Assign it to
            # the first head's other-prefix component so the four component
            # vectors remain exactly additive, including bias-enabled fakes.
            if head == 0 and group == "other_prefix" and getattr(o_proj, "bias", None) is not None:
                projected = projected + o_proj.bias.float().to(projected)
            vectors[head][group] = projected
            margins[head][group] = 0.0
            reconstructed += projected
    clean = None if clean_output is None else clean_output.float().reshape(-1)
    if clean is not None and clean.shape != reconstructed.shape:
        raise ValueError("clean attention output has an incompatible shape")
    error = torch.zeros_like(reconstructed) if clean is None else reconstructed - clean
    maximum = float(error.abs().max().detach().cpu()) if clean is not None else 0.0
    scale = float(clean.abs().max().detach().cpu()) if clean is not None else 1.0
    relative = maximum / max(scale, 1e-8)
    return AttentionReconstruction(vectors, margins, reconstructed, clean, maximum, relative, clean is None or torch.allclose(reconstructed, clean, atol=atol, rtol=rtol))


def frozen_margin_direction(clean_final_residual: torch.Tensor, final_norm: Any, lm_head: Any, positive_token_id: int, negative_token_id: int) -> torch.Tensor:
    """Return the exact FP32 frozen-scale Buy-minus-Sell residual direction."""
    residual = clean_final_residual.float()
    if residual.ndim == 2:
        residual = residual[-1]
    if residual.ndim != 1:
        raise ValueError("clean final residual must be [hidden] or [sequence, hidden]")
    weight = final_norm.weight.float().to(residual.device)
    epsilon = float(getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6)))
    scale = torch.rsqrt(residual.square().mean() + epsilon)
    # Match core.continuation_scoring.fp32_next_token_logits exactly: the
    # frozen scale contracts with the final norm weight used by that helper.
    return (lm_head.weight[int(positive_token_id)].float().to(residual.device) - lm_head.weight[int(negative_token_id)].float().to(residual.device)) * weight * scale


def resolve_single_token_pair(tokenizer: Any, prompt: str, positive: str = "buy", negative: str = "sell") -> tuple[int, int]:
    positive_ids = continuation_token_ids(tokenizer, prompt, positive)
    negative_ids = continuation_token_ids(tokenizer, prompt, negative)
    if len(positive_ids) != 1 or len(negative_ids) != 1:
        raise ValueError("E2 DLA requires single-token Buy/Sell continuations")
    return int(positive_ids[0]), int(negative_ids[0])


def direct_logit_attribution(
    reconstruction: AttentionReconstruction,
    direction: torch.Tensor,
    *,
    final_norm: Any | None = None,
    lm_head: Any | None = None,
    positive_token_id: int | None = None,
    negative_token_id: int | None = None,
) -> dict[int, dict[str, dict[str, float | None]]]:
    """Contract components with the frozen FP32 margin direction.

    The optional final norm/unembedding path reports the component-alone
    diagnostic. It never replaces the additive frozen-scale value.
    """
    direction = direction.float().flatten()
    if direction.numel() != reconstruction.reconstructed_output.numel():
        raise ValueError("DLA direction width does not match attention output")
    result: dict[int, dict[str, dict[str, float | None]]] = {}
    for head, groups in reconstruction.head_group_vectors.items():
        result[head] = {}
        for group, vector in groups.items():
            item: dict[str, float | None] = {"frozen_scale_margin": float(torch.dot(vector.float(), direction).detach().cpu()), "component_alone_margin": None}
            if final_norm is not None and lm_head is not None and positive_token_id is not None and negative_token_id is not None:
                logits = fp32_next_token_logits(type("_Model", (), {"_final_norm": final_norm, "_lm_head": lm_head})(), vector.float().unsqueeze(0))[0]
                item["component_alone_margin"] = float((logits[int(positive_token_id)] - logits[int(negative_token_id)]).detach().cpu())
            result[head][group] = item
    return result


contract_dla = direct_logit_attribution


def routing_label(identity: float, instruction: float, *, epsilon_floor: float = ROUTING_EPSILON_FLOOR) -> str:
    if epsilon_floor <= 0:
        raise ValueError("epsilon floor must be positive")
    identity, instruction = abs(float(identity)), abs(float(instruction))
    if identity / max(instruction, epsilon_floor) > 10:
        return "identity-dominant"
    if instruction / max(identity, epsilon_floor) > 10:
        return "instruction-dominant"
    return "mixed"


def _group_margin(row: Mapping[str, Any], group: str) -> float:
    groups = row.get("groups", {})
    value = groups.get(group, 0.0) if isinstance(groups, Mapping) else 0.0
    if isinstance(value, Mapping):
        value = value.get("frozen_scale_margin", value.get("margin", 0.0))
    return float(value)


def rank_attention_heads(records: Iterable[Mapping[str, Any]], *, top_k: int = SELECTION_TOP_K, epsilon_floor: float = ROUTING_EPSILON_FLOOR) -> list[dict[str, Any]]:
    """Apply the frozen equal-ticker top-five and sign-consistency rules."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    grouped: defaultdict[tuple[int, int], defaultdict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[(int(row["layer"]), int(row["head"]))][str(row.get("ticker", row.get("entity", "")))].append(row)
    ranked: list[dict[str, Any]] = []
    for (layer, head), by_ticker in grouped.items():
        identity_means = {ticker: sum(_group_margin(row, "identity_header") for row in rows) / len(rows) for ticker, rows in by_ticker.items()}
        instruction_mean = sum(sum(_group_margin(row, "instruction_context") for row in rows) / len(rows) for rows in by_ticker.values()) / max(len(by_ticker), 1)
        identity_mean = sum(identity_means.values()) / max(len(identity_means), 1)
        consistency = sum(
            1 for rows in by_ticker.values()
            if len(rows) >= 3 and (sum(_group_margin(row, "identity_header") > 0 for row in rows[:3]) >= 2 or sum(_group_margin(row, "identity_header") < 0 for row in rows[:3]) >= 2)
        )
        ticker_count = len(by_ticker)
        ranked.append({"layer": layer, "head": head, "equal_ticker_mean_abs_identity": sum(abs(value) for value in identity_means.values()) / max(ticker_count, 1), "identity_mean": identity_mean, "instruction_mean": instruction_mean, "ticker_count": ticker_count, "consistency_count": consistency, "consistency_fraction": consistency / max(ticker_count, 1), "routing_label": routing_label(identity_mean, instruction_mean, epsilon_floor=epsilon_floor)})
    ranked.sort(key=lambda row: (-row["equal_ticker_mean_abs_identity"], row["layer"], row["head"]))
    for rank, row in enumerate(ranked, 1):
        row.update(rank=rank, selected_top_five=rank <= top_k, selection_eligible=rank <= top_k and row["consistency_fraction"] >= 0.5)
    return ranked


def compact_attribution_record(*, ticker: str, prompt_id: str, layer: int, head: int, dla: Mapping[int, Mapping[str, Mapping[str, float | None]]], additivity: AttentionReconstruction, routing: str) -> dict[str, Any]:
    groups = dla[int(head)]
    return {"schema_version": 1, "artifact_type": "entity_cell_e2_head_attribution", "ticker": ticker, "prompt_id": prompt_id, "layer": int(layer), "head": int(head), "groups": {name: {"frozen_scale_margin": float(values["frozen_scale_margin"]), "component_alone_margin": None if values.get("component_alone_margin") is None else float(values["component_alone_margin"])} for name, values in groups.items()}, "routing_label": routing, "additivity_max_abs_error": float(additivity.max_abs_error), "additivity_relative_error": float(additivity.relative_error), "additivity_pass": bool(additivity.additive), "raw_runtime_payloads": False}


@dataclass
class AttentionCapture:
    hidden_states: torch.Tensor | None = None
    position_embeddings: Any = None
    attention_mask: torch.Tensor | None = None
    output: torch.Tensor | None = None


def _register_pre_hook(module: Any, callback: Callable[..., Any]) -> Any:
    try:
        return module.register_forward_pre_hook(callback, with_kwargs=True)
    except TypeError:
        return module.register_forward_pre_hook(lambda mod, args: callback(mod, args, {}))


def capture_attention_forward(attention: Any) -> tuple[AttentionCapture, tuple[Any, Any]]:
    capture = AttentionCapture()
    def before(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> None:
        kwargs = kwargs or {}
        if not args:
            raise ValueError("attention forward received no hidden states")
        capture.hidden_states = args[0]
        capture.position_embeddings = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        capture.attention_mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
    def after(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        capture.output = output[0] if isinstance(output, (tuple, list)) else output
    handles: list[Any] = []
    try:
        handles.append(_register_pre_hook(attention, before))
        handles.append(attention.register_forward_hook(after))
    except BaseException:
        remove_hooks(handles)
        raise
    return capture, (handles[0], handles[1])


def remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def reconstruct_captured_attention(layer: Any, capture: AttentionCapture, source_groups: Mapping[str, Any], *, query_position: int = -1) -> AttentionReconstruction:
    if capture.hidden_states is None:
        raise ValueError("attention capture is empty")
    query = query_position if query_position >= 0 else capture.hidden_states.shape[1] + query_position
    clean = None if capture.output is None else capture.output[:, query]
    return reconstruct_attention_components(_attention_module(layer), capture.hidden_states, position_embeddings=capture.position_embeddings, attention_mask=capture.attention_mask, source_groups=source_groups, query_position=query, clean_output=clean)


@dataclass
class OProjectionCapture:
    value: torch.Tensor | None = None


def capture_o_projection_input(o_proj: Any) -> tuple[OProjectionCapture, Any]:
    capture = OProjectionCapture()
    def before(_module: Any, args: tuple[Any, ...]) -> None:
        if not args:
            raise ValueError("o_proj received no concatenated head output")
        capture.value = args[0]
    return capture, o_proj.register_forward_pre_hook(before)


@contextmanager
def selected_head_output_patch(attention: Any, *, head: int, donor_output: torch.Tensor, query_position: int = -1):
    """Patch one head's gated concatenated output immediately before o_proj."""
    o_proj = getattr(attention, "o_proj", None)
    if o_proj is None:
        raise TypeError("attention module has no o_proj")
    head_dim = int(getattr(attention, "head_dim", 0) or donor_output.shape[-1])
    num_heads = int(getattr(attention, "num_heads", 0) or _weight(o_proj).shape[1] // head_dim)
    if not 0 <= int(head) < num_heads:
        raise ValueError("head index is outside the attention head range")
    donor = donor_output.float().reshape(-1)
    if donor.numel() != head_dim:
        raise ValueError("donor head output width differs from head_dim")
    def patch(_module: Any, args: tuple[Any, ...]) -> tuple[torch.Tensor]:
        if not args or args[0].ndim != 3:
            raise ValueError("Qwen3.5 o_proj input must be [batch, sequence, heads*head_dim]")
        value = args[0].clone()
        query = query_position if query_position >= 0 else value.shape[1] + query_position
        if not 0 <= query < value.shape[1]:
            raise ValueError("query position is outside o_proj input")
        start, end = int(head) * head_dim, (int(head) + 1) * head_dim
        if end > value.shape[-1]:
            raise ValueError("head slice is outside o_proj input")
        value[:, query, start:end] = donor.to(device=value.device, dtype=value.dtype)
        return (value,)
    handle = o_proj.register_forward_pre_hook(patch)
    try:
        yield handle
    finally:
        handle.remove()


def patch_head_output(model: Any, input_ids: torch.Tensor, donor_input_ids: torch.Tensor, *, layer: int, head: int, forward: Callable[[torch.Tensor], Any] | None = None, query_position: int = -1) -> Any:
    """Run target with one selected head's final-position output copied from donor."""
    attention = _attention_module(model.layers[int(layer)])
    o_proj = attention.o_proj
    donor_capture, donor_handle = capture_o_projection_input(o_proj)
    runner = forward or model.forward
    try:
        with torch.no_grad():
            runner(donor_input_ids)
        if donor_capture.value is None:
            raise ValueError("donor o_proj capture is empty")
        query = query_position if query_position >= 0 else donor_capture.value.shape[1] + query_position
        head_dim = int(getattr(attention, "head_dim", donor_capture.value.shape[-1] // int(getattr(attention, "num_heads", 1))))
        start, end = int(head) * head_dim, (int(head) + 1) * head_dim
        donor = donor_capture.value[:, query, start:end].detach()
    finally:
        donor_handle.remove()
    with selected_head_output_patch(attention, head=head, donor_output=donor[0], query_position=query_position):
        with torch.no_grad():
            return runner(input_ids)


__all__ = ["FULL_ATTENTION_LAYERS", "PRIMARY_ATTENTION_LAYERS", "SOURCE_GROUPS", "ROUTING_EPSILON_FLOOR", "RECONSTRUCTION_ATOL", "RECONSTRUCTION_RTOL", "SELECTION_TOP_K", "AttentionReconstruction", "validate_attention_layers", "reconstruct_attention_components", "frozen_margin_direction", "resolve_single_token_pair", "direct_logit_attribution", "contract_dla", "routing_label", "rank_attention_heads", "compact_attribution_record", "AttentionCapture", "capture_attention_forward", "remove_hooks", "reconstruct_captured_attention", "OProjectionCapture", "capture_o_projection_input", "selected_head_output_patch", "patch_head_output"]
