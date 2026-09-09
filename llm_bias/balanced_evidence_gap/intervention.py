"""Model-level intervention mechanics for the Phase 2 operators.

Reuses the shared core mechanics (record_residuals, residual_interventions,
score_single_token_margin_fp32, mlp_coordinates, fp32 tail scoring). All
tensors are transient; nothing raw is persisted.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs, score_single_token_margin_fp32
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.inference.mlp import dense_down_projection, mlp_coordinates

from .analysis import FULL_ATTENTION_LAYERS


# ── scoring ──────────────────────────────────────────────────────────────────

def clean_margin(model: Any, tokenizer: Any, scoring_text: str, *, device: Any = None) -> float:
    """FP32 tail-logit margin for one formatted prompt + decision prefix."""
    margin = score_single_token_margin_fp32(
        model, tokenizer, scoring_text, "buy", "sell", device=device
    )
    return float(margin.value)


def answer_token_ids(tokenizer: Any, scoring_text: str) -> tuple[list[int], int, int]:
    """(prompt_ids, buy_id, sell_id) for the fixed JSON decision prefix."""
    from llm_bias.core.prompt_input.encoding import continuation_token_ids, input_ids

    prompt_ids = input_ids(tokenizer, scoring_text, add_special_tokens=True)
    buy = continuation_token_ids(tokenizer, scoring_text, "buy")
    sell = continuation_token_ids(tokenizer, scoring_text, "sell")
    if len(buy) != 1 or len(sell) != 1 or buy == sell:
        raise ValueError("buy/sell must be distinct single-token continuations")
    return prompt_ids, buy[0], sell[0]


def margin_from_log_probs(log_probs: torch.Tensor, buy_id: int, sell_id: int) -> float:
    value = float(log_probs[0, buy_id].detach().cpu() - log_probs[0, sell_id].detach().cpu())
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("non-finite margin")
    return value


def patched_final_margin(
    model: Any,
    input_tensor: torch.Tensor,
    transforms: Mapping[int, Any],
) -> torch.Tensor:
    """Final-position FP32 log probabilities under residual interventions.

    ``transforms`` follows the residual_interventions contract: layer index
    to a [batch, sequence, d_model] transform applied after that block.
    """
    final_layer = int(model.n_layers) - 1
    with residual_interventions(model, transforms):
        residual = record_residuals(model, input_tensor, [final_layer])[final_layer]
    return fp32_next_token_log_probs(model, residual[:, -1, :])


def capture_residuals(
    model: Any, input_tensor: torch.Tensor, layers: Sequence[int]
) -> dict[int, torch.Tensor]:
    """Full-sequence residuals for the requested layers (GPU memory only)."""
    return record_residuals(model, input_tensor, list(layers))


def make_span_transform(source_residual: torch.Tensor, mapping: Mapping[int, int]) -> Any:
    """Residual transform replacing target positions with mapped source values.

    ``mapping`` is target_position -> source_position. The transform clones
    and assigns, so a self-source patch is an exact no-op (|ΔM| = 0).
    """
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        for target_pos, source_pos in mapping.items():
            patched[:, target_pos, :] = source_residual[:, source_pos, :].to(patched.dtype)
        return patched

    return transform


def nearest_position_mapping(
    source_span: tuple[int, int], target_span: tuple[int, int]
) -> dict[int, int]:
    """Target-complete nearest-normalized mapping (causal-tracing contract).

    Every target span position maps to the nearest normalized source
    position; equal-length spans reduce to identity-by-offset.
    """
    src_start, src_end = source_span
    tgt_start, tgt_end = target_span
    n_src = src_end - src_start
    n_tgt = tgt_end - tgt_start
    if n_src <= 0 or n_tgt <= 0:
        return {}
    mapping: dict[int, int] = {}
    for target_offset in range(n_tgt):
        source_offset = 0 if n_tgt == 1 else round(target_offset * (n_src - 1) / (n_tgt - 1))
        mapping[tgt_start + target_offset] = src_start + source_offset
    return mapping


# ── H4 dial readout ──────────────────────────────────────────────────────────

def capture_mlp_channel(model: Any, input_tensor: torch.Tensor, layer: int, neuron: int, position: int) -> float:
    """One down-projection channel at one absolute position (transient).

    Runs one no-grad forward under the coordinate hook, mirroring the
    financial-soundness capture convention.
    """
    from llm_bias.core.inference.forward import record_residuals

    with mlp_coordinates(model, [layer], position) as records:
        with torch.no_grad():
            record_residuals(model, input_tensor, [int(model.n_layers) - 1])
    if layer not in records:
        raise RuntimeError("MLP hook did not fire for the requested layer")
    vector = records[layer]
    if neuron < 0 or neuron >= vector.shape[0]:
        raise ValueError("neuron out of range")
    return float(vector[neuron])


# ── 2C: attention-edge zeroing ───────────────────────────────────────────────

def _attention_module(model: Any, layer: int) -> Any:
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("model does not expose decoder layers")
    block = layers[layer]
    if getattr(block, "block_type", None) is not None and str(block.block_type) != "full_attention":
        raise ValueError(f"layer {layer} is not a full-attention layer")
    for owner in (block, getattr(block, "_hf_layer", None)):
        module = getattr(owner, "self_attn", None) if owner is not None else None
        if module is not None:
            return module
    raise ValueError(f"layer {layer} does not expose a full-attention self_attn module")


def _rope_rotate_half(value: torch.Tensor) -> torch.Tensor:
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
    if cos.ndim != 3 or sin.shape != cos.shape or cos.shape[0] not in (1, q.shape[0]):
        raise ValueError("position embeddings do not cover the attention sequence")
    cos, sin = cos[:, : q.shape[2]].to(q), sin[:, : q.shape[2]].to(q)
    rotary_dim = min(cos.shape[-1], q.shape[-1])
    if rotary_dim % 2:
        raise ValueError("rotary dimension must be even")
    cos, sin = cos[..., :rotary_dim].unsqueeze(1), sin[..., :rotary_dim].unsqueeze(1)
    q_rot, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat((q_rot * cos + _rope_rotate_half(q_rot) * sin, q_tail), dim=-1)
    k = torch.cat((k_rot * cos + _rope_rotate_half(k_rot) * sin, k_tail), dim=-1)
    return q, k


def _rms_norm(attention: Any, name: str, value: torch.Tensor) -> torch.Tensor:
    module = getattr(attention, name, None)
    if module is None:
        return value
    output = module(value)
    if not torch.is_tensor(output) or output.shape != value.shape:
        raise ValueError(f"{name} changed the expected tensor shape")
    return output


class AttentionEdgeZeroing:
    """Zero one head's attention weights from the query position to a set of
    source positions, applied as an additive delta on the o_proj input.

    Before the delta is applied, the internal reconstruction of the head's
    gated attention value must match the actual o_proj input within the
    bf16 tolerance; otherwise the forward fails closed.
    """

    def __init__(
        self,
        model: Any,
        layer: int,
        head: int,
        zero_positions: Sequence[int],
        *,
        query_position: int,
        absolute_floor: float = 1e-3,
        relative_tolerance: float = 2e-2,
    ) -> None:
        if layer not in FULL_ATTENTION_LAYERS:
            raise ValueError(f"layer {layer} is outside the full-attention set {FULL_ATTENTION_LAYERS}")
        if not zero_positions:
            raise ValueError("at least one zero position is required")
        if query_position < 0:
            raise ValueError("query_position must be nonnegative")
        self.model = model
        self.layer = int(layer)
        self.head = int(head)
        self.zero_positions = tuple(int(p) for p in zero_positions)
        if any(p >= query_position for p in self.zero_positions):
            raise ValueError("zero positions must be strictly before the query position")
        self.query_position = int(query_position)
        self.absolute_floor = float(absolute_floor)
        self.relative_tolerance = float(relative_tolerance)
        self.reconstruction_error: float | None = None
        self._handles: list[Any] = []
        self._capture: dict[str, Any] = {}

    def __enter__(self) -> "AttentionEdgeZeroing":
        attention = _attention_module(self.model, self.layer)
        o_proj = getattr(attention, "o_proj", None)
        if o_proj is None:
            raise TypeError("attention module does not expose o_proj")
        head_dim = int(getattr(attention, "head_dim", 0) or attention.q_proj.weight.shape[0] // 32)
        num_heads = attention.q_proj.weight.shape[0] // (2 * head_dim)
        if not 0 <= self.head < num_heads:
            raise ValueError("head out of range for this layer")

        def before(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> None:
            values = kwargs or {}
            hidden = args[0] if args else values.get("hidden_states")
            if not torch.is_tensor(hidden):
                raise ValueError("attention forward received no hidden states")
            self._capture["hidden"] = hidden
            self._capture["position_embeddings"] = values.get(
                "position_embeddings", args[1] if len(args) > 1 else None
            )

        def alter(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            value = args[0]
            if not torch.is_tensor(value) or value.ndim != 3 or value.shape[0] != 1:
                raise ValueError("o_proj input must be [1, sequence, heads*head_dim]")
            if "hidden" not in self._capture:
                raise ValueError("attention inputs were not captured")
            hidden = self._capture["hidden"]
            batch, sequence, _ = value.shape
            if hidden.shape[0] != batch or hidden.shape[1] != sequence:
                raise ValueError("captured hidden does not match o_proj input")
            query = self.query_position
            if query >= sequence:
                raise ValueError("query position is outside the sequence")

            q_flat = attention.q_proj(hidden)
            q, gate = torch.chunk(q_flat.view(batch, sequence, num_heads, 2 * head_dim), 2, dim=-1)
            num_kv = attention.k_proj.weight.shape[0] // head_dim
            k = attention.k_proj(hidden).view(batch, sequence, num_kv, head_dim)
            v = attention.v_proj(hidden).view(batch, sequence, num_kv, head_dim)
            q = _rms_norm(attention, "q_norm", q).transpose(1, 2)
            k = _rms_norm(attention, "k_norm", k).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = _apply_rope(q, k, self._capture["position_embeddings"])
            repeat = num_heads // num_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            scaling = float(getattr(attention, "scaling", head_dim ** -0.5))
            scores = torch.matmul(
                q[:, :, query : query + 1].float(), k.float().transpose(-1, -2)
            ) * scaling
            causal = torch.arange(sequence, device=scores.device) <= query
            scores = scores.masked_fill(~causal[None, None, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            gates = torch.sigmoid(gate[:, query]).to(v.dtype)
            w = weights[0, self.head, 0, :].float()
            v_head = v[0, self.head].float()
            gate_h = gates[0, self.head].float()
            original = (w[:, None] * v_head).sum(dim=0) * gate_h
            w_zeroed = w.clone()
            w_zeroed[list(self.zero_positions)] = 0.0
            total = w_zeroed.sum()
            if total <= 0:
                raise ValueError("zeroing removes all attendable keys")
            w_zeroed = w_zeroed / total
            zeroed = (w_zeroed[:, None] * v_head).sum(dim=0) * gate_h
            actual = value[0, query, self.head * head_dim : (self.head + 1) * head_dim].float()
            max_error = float((original - actual).abs().max().detach().cpu())
            scale = float(actual.abs().max().detach().cpu())
            if max_error > max(self.absolute_floor, self.relative_tolerance * scale):
                raise ValueError(
                    f"attention reconstruction mismatch at L{self.layer}H{self.head}: "
                    f"max_abs_error={max_error:.6f} exceeds tolerance "
                    f"{max(self.absolute_floor, self.relative_tolerance * scale):.6f}"
                )
            self.reconstruction_error = max_error
            delta = (zeroed - original).to(value.dtype)
            patched = value.clone()
            start, end = self.head * head_dim, (self.head + 1) * head_dim
            patched[0, query, start:end] = patched[0, query, start:end] + delta
            return (patched, *args[1:])

        try:
            try:
                self._handles.append(attention.register_forward_pre_hook(before, with_kwargs=True))
            except TypeError:
                self._handles.append(
                    attention.register_forward_pre_hook(
                        lambda module, args, callback=before: callback(module, args, {})
                    )
                )
            self._handles.append(o_proj.register_forward_pre_hook(alter))
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


def random_match_positions(
    query_position: int,
    entity_positions: Sequence[int],
    *,
    n_samples: int = 10,
    seed: int = 42,
) -> list[tuple[int, ...]]:
    """Position-matched controls: random same-count non-entity key sets.

    Each sample is a sorted tuple of positions strictly before or at the
    query position, excluding the entity positions (E3 random_subset
    semantics).
    """
    pool = tuple(p for p in range(query_position + 1) if p not in set(entity_positions))
    count = len(entity_positions)
    if count <= 0 or len(pool) < count:
        raise ValueError("not enough non-entity positions for matched controls")
    rng = random.Random(seed)
    return [tuple(sorted(rng.sample(pool, count))) for _ in range(n_samples)]


# ── 2C: MLP margin attribution ───────────────────────────────────────────────

def mlp_margin_attribution(
    model: Any,
    input_tensor: torch.Tensor,
    *,
    layer: int,
    position: int,
    buy_id: int,
    sell_id: int,
) -> dict[str, Any]:
    """|dM/da|·|a| over MLP down-projection channels at one position.

    M = logit(buy) − logit(sell) is linear in the final normalized residual
    ((w_buy − w_sell)·h), so the backward pass is exact and needs one
    differentiable forward. Parameters stay frozen; only the hooked MLP input
    enters the graph.
    """
    if not 0 <= layer < len(model.layers):
        raise ValueError("layer out of range")
    grad_box: dict[str, torch.Tensor] = {}
    leaf_box: dict[str, torch.Tensor] = {}
    final_box: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        values = args[0]
        if values.ndim != 3 or values.shape[0] != 1 or position >= values.shape[1]:
            raise ValueError("MLP input must be [1, sequence, width] with a valid position")
        if not values.requires_grad:
            values = values.detach().requires_grad_(True)
        leaf_box["values"] = values

        def collect(gradient: torch.Tensor) -> None:
            selected = gradient[0, position].detach().float()
            if not torch.isfinite(selected).all():
                raise ValueError("non-finite MLP gradient at entity position")
            grad_box["gradient"] = selected.cpu()

        handles.append(values.register_hook(collect))
        return (values, *args[1:])

    handles.append(
        dense_down_projection(model.layers[layer]).register_forward_pre_hook(hook)
    )
    final_layer = int(model.n_layers) - 1
    handles.append(
        model.layers[final_layer].register_forward_hook(
            lambda _module, _inputs, output: final_box.update(
                residual=(output if torch.is_tensor(output) else output[0])
            )
        )
    )

    try:
        attention_mask = torch.ones_like(input_tensor)
        with torch.enable_grad():
            try:
                model.forward(input_tensor, attention_mask=attention_mask)
            except TypeError:
                model.forward(input_tensor)
            residual = final_box["residual"]
            h = residual[0, -1, :].float()
            h_norm = model._final_norm(h)
            weight_delta = (
                model._lm_head.weight[buy_id].float()
                - model._lm_head.weight[sell_id].float()
            )
            margin = torch.dot(h_norm, weight_delta.to(h_norm.device))
            margin.backward()
    finally:
        for handle in handles:
            handle.remove()

    if "gradient" not in grad_box or "values" not in leaf_box:
        raise ValueError("attribution did not capture gradient and activation")
    activation = leaf_box["values"][0, position].detach().float().cpu()
    gradient = grad_box["gradient"]
    if gradient.shape != activation.shape:
        raise ValueError("gradient/activation shape mismatch")
    attribution = (gradient * activation).abs()
    if not torch.isfinite(attribution).all():
        raise ValueError("non-finite attribution")
    return {
        "layer": layer,
        "position": position,
        "margin": float(margin.detach().cpu()),
        "attribution": attribution,
        "input_norm": float(activation.norm().cpu()),
        "derivative_norm": float(gradient.norm().cpu()),
    }
