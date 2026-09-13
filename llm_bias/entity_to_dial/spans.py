"""Token-group spans and anonymous prompt construction.

Works on the stored Phase 2A prompt text (the frozen template is never
re-derived here); character ranges map to tokens via the shared
token_span mechanic. Protocol: docs/entity-to-dial/details/proposal-phase-abc.md §4.1–4.2.
"""
from __future__ import annotations

from typing import Any

from llm_bias.core.prompt_input.encoding import token_span

from .template import (
    ANON_NAME,
    ANON_TICKER,
    EVIDENCE_CLOSE,
    EVIDENCE_MARKER,
    NAME_LINE_PREFIX,
    TICKER_LINE_PREFIX,
)


def _bracket_symbol_range(prompt: str, line_prefix: str) -> tuple[int, int]:
    """Character range of the bracketed symbol on one header line."""
    line_start = prompt.find(line_prefix)
    if line_start < 0:
        raise ValueError(f"header line not found: {line_prefix!r}")
    open_bracket = prompt.find("[", line_start)
    if open_bracket < 0:
        raise ValueError(f"header symbol not bracketed: {line_prefix!r}")
    close_bracket = prompt.find("]", open_bracket)
    if close_bracket <= open_bracket:
        raise ValueError(f"header symbol not closed: {line_prefix!r}")
    return (open_bracket + 1, close_bracket)


def _char_span(
    tokenizer: Any, formatted: str, prompt: str, char_start: int, char_end: int
) -> tuple[int, int]:
    """Token range for a character range inside the formatted prompt."""
    body_start = formatted.find(prompt)
    if body_start < 0:
        raise ValueError("prompt body not found in formatted text")
    span = token_span(
        tokenizer,
        formatted,
        body_start + char_start,
        body_start + char_end,
        add_special_tokens=True,
    )
    if span is None:
        raise ValueError("could not map character span to tokens")
    return span


def resolve_token_groups(
    tokenizer: Any, row: dict[str, Any]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """(ticker_span, name_span) token ranges for one stored 2A row.

    Invariants (fail-closed): both groups nonempty, both inside the
    stored entity span, and disjoint. Token granularity is the tokenizer
    unit (a bracket token merged with the symbol belongs to the group).
    """
    prompt: str = row["prompt"]
    formatted: str = row["formatted"]
    entity_span = (int(row["entity_span"][0]), int(row["entity_span"][1]))
    ticker_span = _char_span(
        tokenizer, formatted, prompt, *_bracket_symbol_range(prompt, TICKER_LINE_PREFIX)
    )
    name_span = _char_span(
        tokenizer, formatted, prompt, *_bracket_symbol_range(prompt, NAME_LINE_PREFIX)
    )
    for group, span in (("ticker", ticker_span), ("name", name_span)):
        if span[1] <= span[0]:
            raise ValueError(f"{group} token span is empty")
        if span[0] < entity_span[0] or span[1] > entity_span[1]:
            raise ValueError(
                f"{group} token span {span} outside entity span {entity_span}"
            )
    if not (ticker_span[1] <= name_span[0] or name_span[1] <= ticker_span[0]):
        raise ValueError(
            f"ticker and name token spans overlap: {ticker_span} vs {name_span}"
        )
    return ticker_span, name_span


def entity_char_span(prompt: str) -> tuple[int, int]:
    """Character range of the full entity header (2A entity-span rule)."""
    ticker_start = prompt.find(TICKER_LINE_PREFIX)
    if ticker_start < 0:
        raise ValueError("ticker line not found")
    name_start = prompt.find(NAME_LINE_PREFIX)
    if name_start < 0 or name_start <= ticker_start:
        raise ValueError("name line not found after ticker line")
    name_line_end = prompt.find("\n", name_start)
    if name_line_end < 0:
        raise ValueError("name line unterminated")
    return (ticker_start, name_line_end)


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
    """Character range of the instruction region (Phase F §4.1).

    Same semantics as the balanced-evidence-gap Phase 2
    ``prompt_char_spans["instruction"]``: from after the closing
    ``"—\n\n"`` evidence line to the end of the prompt.
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
