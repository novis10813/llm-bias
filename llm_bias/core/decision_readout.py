"""Generation-path readouts: output path class, finish kind, and the realized-path decision margin.

The realized-path margin is log p(chosen) - log p(counterfactual) at the generation step that emits
the decision value, read from the raw logits of that same greedy generation (no extra forward).
The counterfactual token is the chosen token with "buy" and "sell" swapped in its raw vocabulary
string; when the value is split across tokens or the swapped string is not one token, the margin is
None with an explicit status instead of a guess.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from llm_bias.core.decision_parsing import HARMONY_FINAL_MARKER

PATH_CLASSES = ("direct_json", "brace_newline", "fenced", "thought", "harmony_final",
                "harmony_analysis", "other")
DECISION_KEY = re.compile(r'"decision"\s*:\s*"')
OTHER = {"buy": "sell", "sell": "buy"}


def path_class(text: str) -> str:
    body = text.strip()
    if body.startswith('{"decision"'):
        return "direct_json"
    if body.startswith("{"):
        return "brace_newline"
    if body.startswith("```"):
        return "fenced"
    if body.startswith("thought\n"):
        return "thought"
    if HARMONY_FINAL_MARKER in body:
        return "harmony_final"
    if body.startswith("analysis"):
        return "harmony_analysis"
    return "other"


def unparsed_kind(decision: str, finish: str) -> str | None:
    """None for parsed rows; truncated rows hit the token budget, collapsed rows ended malformed."""
    if decision in ("buy", "sell"):
        return None
    return "truncated" if finish == "max_new_tokens" else "collapsed"


def generation_finish(new_ids: Sequence[int], eos_ids: set[int], max_new_tokens: int) -> str:
    if not new_ids:
        return "empty"
    if new_ids[-1] in eos_ids:
        return "eos"
    return "max_new_tokens" if len(new_ids) >= max_new_tokens else "model_stop"


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def realized_margin(tokenizer: Any, new_ids: Sequence[int], step_logits: Sequence[torch.Tensor],
                    decision: str) -> dict[str, Any]:
    """Margin at the token carrying the parsed decision value, with its step and (buy, sell) token pair.

    Sign convention matches the fixed-prefix margin: log p(buy-token) - log p(sell-token).
    """
    def result(status: str, step: int | None = None, margin: float | None = None,
               pair: list[int] | None = None) -> dict[str, Any]:
        return {"realized_margin": margin, "realized_status": status, "realized_step": step,
                "realized_token_ids": pair}

    if decision not in OTHER:
        return result("unparsed")
    if len(step_logits) != len(new_ids):
        raise ValueError("generation logits do not cover every new token")
    text = _decode(tokenizer, new_ids)
    matches = [m for m in DECISION_KEY.finditer(text) if text[m.end():m.end() + len(decision)] == decision]
    if not matches:
        return result("decision_value_not_found")
    value_at = matches[-1].end()
    previous = ""
    for step in range(len(new_ids)):
        current = _decode(tokenizer, new_ids[:step + 1])
        if len(current) > value_at:
            break
        previous = current
    else:
        return result("decision_value_not_found")
    offset = value_at - len(previous)
    piece = current[len(previous):]
    chosen = int(new_ids[step])
    raw = tokenizer.convert_ids_to_tokens(chosen)
    if offset < 0 or piece[offset:offset + len(decision)] != decision or not isinstance(raw, str) or decision not in raw:
        return result("value_split_across_tokens", step)
    swapped = tokenizer.convert_tokens_to_ids(raw.replace(decision, OTHER[decision], 1))
    if swapped is None or swapped == getattr(tokenizer, "unk_token_id", None) or swapped == chosen:
        return result("no_counterfactual_token", step)
    pair = [chosen, int(swapped)] if decision == "buy" else [int(swapped), chosen]
    log_probs = F.log_softmax(step_logits[step].float().reshape(-1), dim=-1)
    return result("ok", step, float(log_probs[pair[0]] - log_probs[pair[1]]), pair)


__all__ = ["PATH_CLASSES", "generation_finish", "path_class", "realized_margin", "unparsed_kind"]
