"""Span resolution for cross-entity probe prompts.

Maps the frozen prompt template's character regions to token ranges inside
the formatted (chat-templated) prompt, using the shared token_span mechanic.
"""
from __future__ import annotations

from typing import Any

from llm_bias.core.prompt_input.encoding import input_ids, token_span

from .template import (
    EVIDENCE_CLOSE,
    EVIDENCE_MARKER,
    NAME_LINE_PREFIX,
    TICKER_LINE_PREFIX,
    build_prompt,
    build_prompt_v2,
    variant_id_v2,
)


def prompt_char_spans(prompt: str) -> dict[str, tuple[int, int]]:
    """Character ranges for the three patchable regions of a probe prompt."""
    # entity region: from "Stock Ticker: [" through the end of the name line.
    ticker_start = prompt.find(TICKER_LINE_PREFIX)
    if ticker_start < 0:
        raise ValueError("ticker line not found")
    name_start = prompt.find(NAME_LINE_PREFIX)
    if name_start < 0 or name_start <= ticker_start:
        raise ValueError("name line not found after ticker line")
    name_line_end = prompt.find("\n", name_start)
    if name_line_end < 0:
        raise ValueError("name line unterminated")

    # evidence region: the four bullet lines between the markers.
    marker_end = prompt.find(EVIDENCE_MARKER)
    if marker_end < 0:
        raise ValueError("evidence marker not found")
    evidence_start = prompt.find("\n\n", marker_end) + 2
    close = prompt.find(EVIDENCE_CLOSE, evidence_start)
    if close < 0:
        raise ValueError("evidence close marker not found")
    evidence_end = close + 2  # include the "\n\n" before the closing dash

    # instruction region: after the closing "—\n\n" line, to end of prompt.
    instruction_start = close + len(EVIDENCE_CLOSE)
    return {
        "entity": (ticker_start, name_line_end),
        "evidence": (evidence_start, evidence_end),
        "instruction": (instruction_start, len(prompt)),
    }


def resolve_row(
    tokenizer: Any,
    ticker: str,
    name: str,
    sector: str,
    reverse: bool,
    order: int,
    *,
    format_fn: Any,
) -> "dict[str, Any]":
    """Build and resolve one probe row (prompt + formatted + token spans)."""
    from .template import ProbeRow, variant_id

    prompt = build_prompt(ticker, name, order, reverse)
    formatted = format_fn(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
    ids = input_ids(tokenizer, formatted, add_special_tokens=True)
    char_spans = prompt_char_spans(prompt)

    resolved: dict[str, tuple[int, int]] = {}
    for region, (char_start, char_end) in char_spans.items():
        # The prompt body sits inside the formatted text; locate it first.
        body_start = formatted.find(prompt)
        if body_start < 0:
            raise ValueError("prompt body not found in formatted text")
        span = token_span(
            tokenizer, formatted, body_start + char_start, body_start + char_end,
            add_special_tokens=True,
        )
        if span is None:
            raise ValueError(f"could not map {region} character span to tokens")
        resolved[region] = span

    final_position = len(ids) - 1
    entity_position = resolved["entity"][1] - 1
    row = ProbeRow(
        id=variant_id(ticker, reverse, order),
        ticker=ticker,
        name=name,
        sector=sector,
        reverse=reverse,
        order=order,
        prompt=prompt,
        formatted=formatted,
        prompt_ids=ids,
        entity_span=resolved["entity"],
        evidence_span=resolved["evidence"],
        instruction_span=resolved["instruction"],
        final_position=final_position,
        entity_position=entity_position,
    )
    return row.to_dict()


def resolve_row_v2(
    tokenizer: Any,
    ticker: str,
    name: str,
    sector: str,
    condition: str,
    reverse: bool,
    *,
    format_fn: Any,
) -> "dict[str, Any]":
    """Build and resolve one v2 condition probe row (two-sentence evidence).

    Row schema matches v1 (``order`` is 0; the sentence-order axis does not
    exist for a same-sign pair) plus ``condition`` and ``key``
    (``<ticker>:<condition>``, the canonical row identifier used by the 2B
    condition-flip directions).
    """
    from .template import ProbeRow

    prompt = build_prompt_v2(ticker, name, condition, reverse)
    formatted = format_fn(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
    ids = input_ids(tokenizer, formatted, add_special_tokens=True)
    char_spans = prompt_char_spans(prompt)

    resolved: dict[str, tuple[int, int]] = {}
    for region, (char_start, char_end) in char_spans.items():
        body_start = formatted.find(prompt)
        if body_start < 0:
            raise ValueError("prompt body not found in formatted text")
        span = token_span(
            tokenizer, formatted, body_start + char_start, body_start + char_end,
            add_special_tokens=True,
        )
        if span is None:
            raise ValueError(f"could not map {region} character span to tokens")
        resolved[region] = span

    row = ProbeRow(
        id=variant_id_v2(ticker, condition, reverse),
        ticker=ticker,
        name=name,
        sector=sector,
        reverse=reverse,
        order=0,
        prompt=prompt,
        formatted=formatted,
        prompt_ids=ids,
        entity_span=resolved["entity"],
        evidence_span=resolved["evidence"],
        instruction_span=resolved["instruction"],
        final_position=len(ids) - 1,
        entity_position=resolved["entity"][1] - 1,
    )
    return row.to_dict() | {"condition": condition, "key": f"{ticker}:{condition}"}
