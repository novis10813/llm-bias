"""Financial-decision prompt preparation and evidence-span mapping."""
from __future__ import annotations

from typing import Any

from llm_bias.core.prompt_input.encoding import format_prompt, token_span

EVIDENCE_START = "--- Evidence ---"
EVIDENCE_END = "\n---\nRespond"


def prepare_scoring_prompt(
    tokenizer: Any,
    raw_prompt: str,
    *,
    decision_prefix: str,
) -> tuple[str, tuple[int, int]]:
    """Format one user prompt and locate evidence tokens in the scored string."""
    evidence_marker = raw_prompt.find(EVIDENCE_START)
    if evidence_marker < 0:
        raise ValueError("prompt is missing the evidence start marker")
    evidence_start = evidence_marker + len(EVIDENCE_START)
    evidence_end = raw_prompt.find(EVIDENCE_END, evidence_start)
    if evidence_end < 0:
        raise ValueError("prompt is missing the evidence end marker")
    formatted = format_prompt(
        tokenizer, raw_prompt, use_chat_template=True, enable_thinking=False
    )
    raw_offset = formatted.find(raw_prompt)
    if raw_offset < 0:
        raise ValueError("formatted chat prompt does not contain the raw user prompt")
    scoring_prompt = formatted + decision_prefix
    span = token_span(
        tokenizer,
        scoring_prompt,
        raw_offset + evidence_start,
        raw_offset + evidence_end,
        add_special_tokens=True,
    )
    if span is None:
        raise ValueError("could not map evidence text to token positions")
    return scoring_prompt, span


__all__ = ["prepare_scoring_prompt"]
