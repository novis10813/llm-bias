"""Phase D signed attribution: per-position MLP derivatives + top-channel stats.

Protocol: docs/entity-to-dial/details/proposal-phase-d.md §4.3 (Rev 1).

``mlp_position_derivative`` mirrors the core ``mlp_summed_derivatives``
hook lifecycle (pre-hook on the dense MLP down-projection input,
``detach().requires_grad_(True)`` for frozen models, exception-safe
handle removal) but collects the signed gradient at ONE absolute
position instead of summing over all positions. ``differentiable_margin``
is the D2 objective: the same FP32-tail margin as the no-grad pipeline
paths, kept in the autograd graph. ``top_channel_stats`` is a pure
function that reduces the transient per-company, per-layer gradient
vectors to compact per-layer scalar statistics (top channel, sector
agreement, matched controls); the gradient vectors themselves are never
persisted.
"""
from __future__ import annotations

import random
import statistics
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.mlp import dense_down_projection

from .analysis import spearman


@contextmanager
def mlp_position_derivative(
    model: Any, layers: list[int], position: int
) -> Iterator[dict[int, torch.Tensor]]:
    """Capture signed d(objective)/d(down-proj input) at one position.

    Run one batch-one differentiable forward and call ``backward()``
    once inside this context. ``records[layer]`` is the CPU float32
    vector of shape ``[down_proj input width]`` for the requested
    absolute position. Each layer hook must fire exactly once (single
    teacher-forced forward); a second firing fails closed.
    """
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("require unique nonempty layers")
    if any(not 0 <= layer < len(model.layers) for layer in layers):
        raise ValueError("layer out of range")
    if position < 0:
        raise ValueError("position must be nonnegative")
    records: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    try:
        for layer in layers:
            def hook(_module: Any, args: tuple[Any, ...], layer: int = layer) -> Any:
                values = args[0]
                if values.ndim != 3 or values.shape[0] != 1:
                    raise ValueError("require batch-one MLP input")
                if position >= values.shape[1]:
                    raise ValueError("position outside the sequence")
                if not torch.is_grad_enabled():
                    raise RuntimeError("gradient recording requires grad mode")
                # Frozen models still need a graph starting at the first hook.
                if not values.requires_grad:
                    values = values.detach().requires_grad_(True)

                def collect(gradient: torch.Tensor) -> None:
                    if layer in records:
                        raise RuntimeError(f"derivative hook fired twice at layer {layer}")
                    vector = gradient.detach().float()[0, position].cpu()
                    if not torch.isfinite(vector).all():
                        raise ValueError(f"non-finite MLP derivative at layer {layer}")
                    records[layer] = vector

                handles.append(values.register_hook(collect))
                return (values, *args[1:])
            handles.append(
                dense_down_projection(model.layers[layer]).register_forward_pre_hook(hook)
            )
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def mlp_all_positions_derivative(
    model: Any, layer: int
) -> Iterator[dict[int, torch.Tensor]]:
    """Capture signed d(objective)/d(down-proj input) at ALL positions.

    Same lifecycle as :func:`mlp_position_derivative`; ``records[layer]``
    is the CPU float32 tensor ``[sequence, down_proj input width]``. The
    fp32 sum over the position axis must reproduce the core
    ``mlp_summed_derivatives`` output within bf16 accumulation tolerance
    (linearity of the gradient). Used by the smoke acceptance to validate
    the per-position slicing mechanic.
    """
    if not 0 <= layer < len(model.layers):
        raise ValueError("layer out of range")
    records: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    try:
        def hook(_module: Any, args: tuple[Any, ...]) -> Any:
            values = args[0]
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError("require batch-one MLP input")
            if not torch.is_grad_enabled():
                raise RuntimeError("gradient recording requires grad mode")
            if not values.requires_grad:
                values = values.detach().requires_grad_(True)

            def collect(gradient: torch.Tensor) -> None:
                if layer in records:
                    raise RuntimeError(f"derivative hook fired twice at layer {layer}")
                tensor = gradient.detach().float()[0].cpu()
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"non-finite MLP derivative at layer {layer}")
                records[layer] = tensor

            handles.append(values.register_hook(collect))
            return (values, *args[1:])
        handles.append(
            dense_down_projection(model.layers[layer]).register_forward_pre_hook(hook)
        )
        yield records
    finally:
        for handle in handles:
            handle.remove()


def differentiable_margin(
    model: Any, input_tensor: torch.Tensor, buy_id: int, sell_id: int
) -> torch.Tensor:
    """In-graph FP32-tail decision margin (the D2 objective).

    Identical tail to the no-grad pipeline paths (final-layer residual
    -> FP32 final norm -> unembedding -> log-softmax, buy minus sell at
    the final position), but without detaching, so
    ``margin.backward()`` propagates through the layer stack.
    """
    if input_tensor.ndim != 2 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor must be [1, sequence]")
    final_layer = int(model.n_layers) - 1
    box: dict[str, torch.Tensor] = {}

    def final_hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = output if torch.is_tensor(output) else output[0]
        box["residual"] = tensor

    handle = model.layers[final_layer].register_forward_hook(final_hook)
    try:
        attention_mask = torch.ones_like(input_tensor)
        try:
            model.forward(input_tensor, attention_mask=attention_mask)
        except TypeError:
            model.forward(input_tensor)
        if "residual" not in box:
            raise RuntimeError("final-layer hook did not fire")
        log_probs = fp32_next_token_log_probs(model, box["residual"][:, -1, :])[0]
        return log_probs[buy_id] - log_probs[sell_id]
    finally:
        handle.remove()


def _channel_rho(values: Sequence[float], margins: Sequence[float]) -> float:
    """Spearman rho of one channel across companies; 0.0 when degenerate."""
    try:
        return spearman(list(values), list(margins))
    except ValueError:
        # Degenerate (constant) series: rank correlation undefined.
        return 0.0


def top_channel_stats(
    gradients: Mapping[int, Mapping[str, Sequence[float]]],
    pure_entity_margins: Mapping[str, float],
    sector_of: Mapping[str, str],
    *,
    controls_n: int = 10,
    controls_seed_base: int = 42,
) -> dict[int, dict]:
    """Per-layer top-channel statistics (protocol §2/§4.3; descriptive only).

    ``gradients`` maps layer -> ticker -> signed dM/d(down-proj input)
    vector (identical width for all tickers). Per layer, the per-channel
    Spearman rho against the 2A pure entity margin is computed across
    companies; the top channel is argmax |rho| (ties: lowest index).
    Sector agreement counts the sectors whose mean top-channel value
    (over the sector's companies) carries the same sign as the top
    channel's rho. Matched controls are
    ``random.Random(controls_seed_base + layer).sample(range(width), controls_n)``.
    """
    per_layer: dict[int, dict] = {}
    for layer in sorted(gradients):
        by_ticker = gradients[layer]
        tickers = sorted(by_ticker)
        if not tickers:
            raise ValueError(f"layer {layer} has no company gradients")
        width = len(by_ticker[tickers[0]])
        for ticker in tickers:
            if len(by_ticker[ticker]) != width:
                raise ValueError(f"gradient width mismatch for {ticker} at L{layer}")
            if ticker not in pure_entity_margins or ticker not in sector_of:
                raise ValueError(f"missing margin/sector metadata for {ticker} at L{layer}")
        margins = [float(pure_entity_margins[t]) for t in tickers]

        def channel_values(index: int) -> list[float]:
            return [float(by_ticker[t][index]) for t in tickers]

        rhos = [_channel_rho(channel_values(index), margins) for index in range(width)]
        top_idx = 0
        for index in range(1, width):
            if abs(rhos[index]) > abs(rhos[top_idx]):
                top_idx = index
        top_rho = rhos[top_idx]

        sign = 1 if top_rho > 0 else (-1 if top_rho < 0 else 0)
        sectors = sorted({sector_of[t] for t in tickers})
        agreement = 0
        if sign:
            for sector in sectors:
                mean = statistics.fmean(
                    float(by_ticker[t][top_idx])
                    for t in tickers
                    if sector_of[t] == sector
                )
                if mean * sign > 0:
                    agreement += 1

        control_k = min(controls_n, width)
        control_idxs = sorted(
            random.Random(controls_seed_base + layer).sample(range(width), control_k)
        )
        max_control_rho = max((abs(rhos[i]) for i in control_idxs), default=0.0)
        per_layer[layer] = {
            "top_channel_idx": int(top_idx),
            "top_channel_rho": float(top_rho),
            "top_channel_sector_agreement": int(agreement),
            "n_sectors": len(sectors),
            "control_channel_idxs": [int(i) for i in control_idxs],
            "max_control_rho": float(max_control_rho),
            "n_companies": len(tickers),
        }
    return per_layer
