"""Single-forward decision-margin scoring under residual interventions.

One no-grad forward yields the FP32-tail fixed answer-token margin
(``log p("buy") − log p("sell")`` at the final position) with optional
post-block residual transforms and an optional MLP-channel addition
(dial probe). No generation; no raw states retained.

The mechanics mirror the entity-to-dial / balanced-evidence-gap scoring
path (``record_residuals`` + ``fp32_next_token_log_probs``), which has
been verified bit-exact against the 2A/2B archives in those lines.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import ResidualTransform, residual_interventions
from llm_bias.core.inference.mlp_addition import mlp_addition
from llm_bias.core.prompt_input.encoding import continuation_token_ids, input_ids

from .template import DECISION_PREFIX


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
    """Fixed answer-token margin (nats, buy positive); fail-closed on non-finite.

    The subtraction happens in Python float64 (convert-to-float first),
    matching the canonical 2A reference mechanism
    (``score_single_token_margin_fp32``) bit-for-bit; a float32 tensor
    subtraction would round the difference by up to one float32 ulp and
    break the bit-exact clean-vs-archive pre-check.
    """
    value = float(log_probs[0, buy_id].detach().cpu()) - float(log_probs[0, sell_id].detach().cpu())
    if not math.isfinite(value):
        raise ValueError("non-finite margin")
    return value


def margin_forward(
    model: Any,
    input_tensor: torch.Tensor,
    transforms: Mapping[int, ResidualTransform] | None,
    buy_id: int,
    sell_id: int,
    *,
    mlp_delta: float | None = None,
    mlp_layer: int | None = None,
    mlp_neuron: int | None = None,
) -> float:
    """Final-position margin under post-block transforms (+ optional MLP addition).

    ``mlp_delta`` composes the frozen investment-dial coordinate push with
    the residual transforms in one forward (dial probe). Hooks from both
    mechanics are removed on all exit paths.
    """
    if input_tensor.ndim != 2 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor must be [1, sequence]")
    if mlp_delta is None:
        ctx: Any = None
    elif mlp_layer is None or mlp_neuron is None:
        raise ValueError("mlp_delta requires mlp_layer and mlp_neuron")
    else:
        ctx = mlp_addition(model, mlp_layer, mlp_neuron, mlp_delta)

    final_layer = int(model.n_layers) - 1
    if ctx is None:
        with residual_interventions(model, dict(transforms or {})):
            residual = record_residuals(model, input_tensor, [final_layer])[final_layer]
    else:
        with ctx, residual_interventions(model, dict(transforms or {})):
            residual = record_residuals(model, input_tensor, [final_layer])[final_layer]
    return margin_from_log_probs(
        fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
    )
