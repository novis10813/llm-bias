"""Single-forward decision margin + dial activation probe.

Protocol: docs/entity-to-dial/proposal.md §4.4. One no-grad forward
yields the FP32-tail decision margin and the L15/n8490 down-projection
input channel (dial activation, native units) at the requested absolute
positions. Values are transient; callers reduce to scalars immediately.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.core.prompt_input.encoding import continuation_token_ids, input_ids

from .template import DECISION_PREFIX, DIAL_LAYER, DIAL_NEURON


def scoring_ids(tokenizer: Any, formatted: str) -> list[int]:
    """Token ids of the formatted prompt plus the fixed decision prefix."""
    return input_ids(tokenizer, formatted + DECISION_PREFIX, add_special_tokens=True)


def answer_token_ids(tokenizer: Any, scoring_text: str) -> tuple[int, int]:
    """(buy_id, sell_id) for the fixed JSON decision prefix."""
    buy = continuation_token_ids(tokenizer, scoring_text, "buy")
    sell = continuation_token_ids(tokenizer, scoring_text, "sell")
    if len(buy) != 1 or len(sell) != 1 or buy == sell:
        raise ValueError("buy/sell must be distinct single-token continuations")
    return buy[0], sell[0]


def margin_from_log_probs(log_probs: torch.Tensor, buy_id: int, sell_id: int) -> float:
    value = float(log_probs[0, buy_id].detach().cpu() - log_probs[0, sell_id].detach().cpu())
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("non-finite margin")
    return value


@contextmanager
def dial_value_capture(
    model: Any,
    layer: int,
    channel: int,
    positions: Sequence[int],
):
    """Capture one down-projection input channel at positions (Phase E).

    Yields a dict ``{position: value}`` (FP32, native units) filled during
    the caller's single forward. Fails closed: every requested position
    must be captured with a finite value. The hook is removed on all exit
    paths.
    """
    if not 0 <= layer < len(model.layers):
        raise ValueError(f"dial layer {layer} out of range")
    if channel < 0:
        raise ValueError("dial channel must be nonnegative")
    positions = tuple(int(p) for p in positions)
    if not positions:
        raise ValueError("dial positions must be nonempty")

    box: dict[int, float] = {}

    def hook(_module: Any, args: tuple[Any, ...]) -> None:
        values = args[0]
        if not torch.is_tensor(values) or values.ndim != 3 or values.shape[0] != 1:
            raise ValueError("MLP input must be [1, sequence, width]")
        if channel >= values.shape[-1]:
            raise ValueError("dial channel out of range")
        for pos in positions:
            if pos < 0 or pos >= values.shape[1]:
                raise ValueError(f"dial position {pos} outside the sequence")
            value = values[0, pos, channel].detach().float()
            if not torch.isfinite(value):
                raise ValueError(f"non-finite dial activation at position {pos}")
            box[pos] = float(value.cpu())

    handle = dense_down_projection(model.layers[layer]).register_forward_pre_hook(hook)
    try:
        yield box
        missing = [p for p in positions if p not in box]
        if missing:
            raise RuntimeError(f"dial capture incomplete: missing positions {missing}")
    finally:
        handle.remove()


def probe_forward(
    model: Any,
    input_tensor: torch.Tensor,
    *,
    buy_id: int,
    sell_id: int,
    dial_positions: Sequence[int] = (),
    dial_layer: int = DIAL_LAYER,
    dial_neuron: int = DIAL_NEURON,
) -> tuple[float, dict[int, float]]:
    """One no-grad forward: FP32-tail margin + dial channel at positions.

    Returns ``(margin, {position: dial_activation})``. The margin is the
    log-probability difference buy minus sell at the final position,
    computed through the model's own FP32 final-norm + unembedding tail.
    """
    if input_tensor.ndim != 2 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor must be [1, sequence]")
    final_layer = int(model.n_layers) - 1
    dial_positions = tuple(int(p) for p in dial_positions)
    if any(pos < 0 or pos >= input_tensor.shape[1] for pos in dial_positions):
        raise ValueError("dial position outside the scoring sequence")

    dial_box: dict[int, float] = {}
    final_box: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    if dial_positions:
        if dial_neuron < 0:
            raise ValueError("dial neuron must be nonnegative")

        def dial_hook(_module: Any, args: tuple[Any, ...]) -> None:
            values = args[0]
            if not torch.is_tensor(values) or values.ndim != 3 or values.shape[0] != 1:
                raise ValueError("MLP input must be [1, sequence, width]")
            if dial_neuron >= values.shape[-1]:
                raise ValueError("dial neuron out of range")
            for pos in dial_positions:
                value = values[0, pos, dial_neuron].detach().float().cpu()
                if not torch.isfinite(value):
                    raise ValueError(f"non-finite dial activation at position {pos}")
                dial_box[pos] = float(value)

        handles.append(
            dense_down_projection(model.layers[dial_layer]).register_forward_pre_hook(dial_hook)
        )

    def final_hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = output if torch.is_tensor(output) else output[0]
        final_box["residual"] = tensor

    handles.append(model.layers[final_layer].register_forward_hook(final_hook))
    try:
        with torch.no_grad():
            attention_mask = torch.ones_like(input_tensor)
            try:
                model.forward(input_tensor, attention_mask=attention_mask)
            except TypeError:
                model.forward(input_tensor)
        if "residual" not in final_box:
            raise RuntimeError("final-layer hook did not fire")
        log_probs = fp32_next_token_log_probs(model, final_box["residual"][:, -1, :])
    finally:
        for handle in handles:
            handle.remove()

    if dial_positions and sorted(dial_box) != sorted(set(dial_positions)):
        raise RuntimeError("dial hook did not capture every requested position")
    return margin_from_log_probs(log_probs, buy_id, sell_id), dial_box
