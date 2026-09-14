"""Anonymous prompt construction and instruction-region token spans.

Works on the stored Phase 2A prompt text (the frozen template is never
re-derived here). The character-span → token-span mechanic is the shared
``core/prompt_input/encoding.token_span``. The byte-level header
substitution is copied from the entity-to-dial line (same frozen template
constants; promotion to a shared location is a future consolidation).
"""
from __future__ import annotations

from typing import Any

from llm_bias.core.prompt_input.encoding import format_prompt, input_ids, token_span

from .template import ANON_NAME, ANON_TICKER, EVIDENCE_CLOSE, EVIDENCE_MARKER, NAME_LINE_PREFIX, TICKER_LINE_PREFIX


def anonymous_prompt(named_prompt: str, ticker: str, name: str) -> str:
    """Byte-level single replacement of the entity header (layout preserved)."""
    header_named = (
        f"{TICKER_LINE_PREFIX}{ticker}]\n\n{NAME_LINE_PREFIX}{name}]"
    )
    header_anon = (
        f"{TICKER_LINE_PREFIX}{ANON_TICKER}]\n\n{NAME_LINE_PREFIX}{ANON_NAME}]"
    )
    if named_prompt.count(header_named) != 1:
        raise ValueError("anonymous header substitution must hit exactly once")
    return named_prompt.replace(header_named, header_anon)


def instruction_char_span(prompt: str) -> tuple[int, int]:
    """Character range of the instruction region (Phase 2A span rule).

    From after the closing ``"—\\n\\n"`` evidence line to the end of the
    prompt — same semantics as the stored 2A ``instruction_span``.
    """
    marker_end = prompt.find(EVIDENCE_MARKER)
    if marker_end < 0:
        raise ValueError("evidence marker not found")
    evidence_start = prompt.find("\n\n", marker_end) + 2
    close = prompt.find(EVIDENCE_CLOSE, evidence_start)
    if close < 0:
        raise ValueError("evidence close marker not found")
    start = close + len(EVIDENCE_CLOSE)
    if start >= len(prompt):
        raise ValueError("instruction region is empty")
    return (start, len(prompt))


def instruction_token_span(
    tokenizer: Any, formatted: str, prompt: str
) -> tuple[int, int]:
    """Token range of the instruction region for one formatted prompt."""
    body_start = formatted.find(prompt)
    if body_start < 0:
        raise ValueError("prompt body not found in formatted text")
    char_start, char_end = instruction_char_span(prompt)
    span = token_span(
        tokenizer,
        formatted,
        body_start + char_start,
        body_start + char_end,
        add_special_tokens=True,
    )
    if span is None:
        raise ValueError("could not map instruction character span to tokens")
    if span[1] <= span[0]:
        raise ValueError("instruction token span is empty")
    return span


def _anonymous_row(tokenizer: Any, named_row: dict[str, Any]) -> dict:
    """Anonymous probe row: header substitution + instruction span (fail-closed).

    Mirrors the entity-to-dial Phase F anonymous-row construction; the
    stored 2A row provides the prompt body and the frozen header layout.
    """
    prompt = anonymous_prompt(named_row["prompt"], named_row["ticker"], named_row["name"])
    formatted = format_prompt(
        tokenizer, prompt, use_chat_template=True, enable_thinking=False
    )
    prompt_ids = input_ids(tokenizer, formatted, add_special_tokens=True)
    char_start, char_end = instruction_token_span(tokenizer, formatted, prompt)
    body_start = formatted.find(prompt)
    if body_start < 0:
        raise ValueError("anonymous prompt body not found in formatted text")
    span = token_span(
        tokenizer,
        formatted,
        body_start + char_start,
        body_start + char_end,
        add_special_tokens=True,
    )
    if span is None or span[1] <= span[0]:
        raise ValueError("could not map anonymous instruction span to tokens")
    return {
        "id": "anonymous",
        "prompt_type": "anon",
        "prompt": prompt,
        "formatted": formatted,
        "prompt_ids": prompt_ids,
        "ticker": None,
        "name": None,
        "sector": None,
        "reverse": False,
        "order": 0,
        "instruction_span": list(span),
    }
