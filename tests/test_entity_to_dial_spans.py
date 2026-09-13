"""Regression tests for entity-to-dial span resolution and anonymous prompts.

Character-level fake tokenizer with offset_mapping; no checkpoint is loaded.
Prompt text comes from the frozen Phase 2A template (test scaffolding only).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_bias.balanced_evidence_gap.template import build_prompt
from llm_bias.core.prompt_input.encoding import token_span
from llm_bias.entity_to_dial.spans import (
    anonymous_prompt,
    entity_char_span,
    resolve_token_groups,
)
from llm_bias.entity_to_dial.template import (
    ANON_NAME,
    ANON_TICKER,
    NAME_LINE_PREFIX,
    TICKER_LINE_PREFIX,
)


class _CharTokenizer:
    """Character tokenizer; each character is one token with a 1-char offset."""

    chat_template = "fake"

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        return_special_tokens_mask: bool = False,
        **_kwargs,
    ) -> SimpleNamespace:
        del add_special_tokens
        text = str(text)
        values = [ord(char) for char in text]
        if return_offsets_mapping:
            return SimpleNamespace(
                input_ids=values,
                offset_mapping=[(i, i + 1) for i in range(len(text))],
                special_tokens_mask=[False] * len(values),
            )
        return SimpleNamespace(input_ids=values)

    def apply_chat_template(self, messages, **_kwargs):
        return "«" + messages[0]["content"] + "»"


TICKER = "NSC"
NAME = "Norfolk Southern"
PROMPT = build_prompt(TICKER, NAME, order=0, reverse=False)


def _tokenizer() -> _CharTokenizer:
    return _CharTokenizer()


def _row(tokenizer: _CharTokenizer, prompt: str = PROMPT, **overrides) -> dict:
    formatted = "«" + prompt + "»"
    char_span = entity_char_span(prompt)
    span = token_span(
        tokenizer, formatted, char_span[0], char_span[1], add_special_tokens=True
    )
    row = {
        "prompt": prompt,
        "formatted": formatted,
        "entity_span": list(span),
    }
    row.update(overrides)
    return row


# ── entity_char_span ─────────────────────────────────────────────────────────


def test_entity_char_span_covers_both_header_lines():
    span = entity_char_span(PROMPT)
    text = PROMPT[span[0] : span[1]]
    assert text.startswith(f"Stock Ticker: [{TICKER}]")
    assert text.endswith(f"{NAME}]")
    assert "\n\nStock Name: [" in text


def test_entity_char_span_fails_closed():
    with pytest.raises(ValueError, match="ticker line not found"):
        entity_char_span("no header at all")
    prompt_no_name = "Stock Ticker: [NSC]\n\nBody."
    with pytest.raises(ValueError, match="name line"):
        entity_char_span(prompt_no_name)
    with pytest.raises(ValueError, match="name line"):
        entity_char_span("Stock Name: [X]\n\nStock Ticker: [Y]\n\nBody.")


# ── resolve_token_groups ─────────────────────────────────────────────────────


def test_token_groups_split_the_entity_span():
    tokenizer = _tokenizer()
    row = _row(tokenizer)
    ticker_span, name_span = resolve_token_groups(tokenizer, row)

    prompt_body = row["prompt"]
    ticker_chars = prompt_body.index(f"[{TICKER}]") + 1
    name_chars = prompt_body.index(f"[{NAME}]") + 1
    # Character tokenizer: token index in the formatted text is the prompt
    # character index plus one (« prefix shifts every body position).
    assert ticker_span == (ticker_chars + 1, ticker_chars + len(TICKER) + 1)
    assert name_span == (name_chars + 1, name_chars + len(NAME) + 1)

    entity = row["entity_span"]
    for span in (ticker_span, name_span):
        assert span[1] > span[0]
        assert entity[0] <= span[0] < span[1] <= entity[1]
    assert ticker_span[1] <= name_span[0]


def test_token_groups_disjoint_across_entities():
    tokenizer = _tokenizer()
    a, _ = resolve_token_groups(tokenizer, _row(tokenizer))
    b, _ = resolve_token_groups(
        tokenizer,
        _row(tokenizer, build_prompt("DE", "Deere & Co", order=0, reverse=False)),
    )
    # Different symbols, same header layout: spans are at the same token
    # positions but carry different text.
    assert a[0] == b[0]
    assert a[1] - a[0] != b[1] - b[0]


def test_token_groups_fail_closed_outside_entity_span():
    tokenizer = _tokenizer()
    row = _row(tokenizer)
    # Entity span truncated before the name line ends: name span must fail.
    truncated = dict(row, entity_span=[row["entity_span"][0], row["entity_span"][1] - 3])
    with pytest.raises(ValueError, match="outside entity span"):
        resolve_token_groups(tokenizer, truncated)
    # Unterminated ticker bracket: no "]" anywhere after the open bracket
    # (name line has no bracket either; row built by hand, since the helper
    # would fail on the entity char span first).
    broken_prompt = "Stock Ticker: [NSC\n\nStock Name: X\n\nBody."
    broken = {
        "prompt": broken_prompt,
        "formatted": "«" + broken_prompt + "»",
        "entity_span": [0, 10],
    }
    with pytest.raises(ValueError, match="not closed"):
        resolve_token_groups(tokenizer, broken)
    # Ticker line entirely missing.
    no_ticker_prompt = "Stock Name: [X]\n\nBody."
    no_ticker = {
        "prompt": no_ticker_prompt,
        "formatted": "«" + no_ticker_prompt + "»",
        "entity_span": [0, 10],
    }
    with pytest.raises(ValueError, match="header line not found"):
        resolve_token_groups(tokenizer, no_ticker)


# ── anonymous_prompt ─────────────────────────────────────────────────────────


def test_anonymous_prompt_single_byte_level_replacement():
    anon = anonymous_prompt(PROMPT, TICKER, NAME)
    # The full anonymous header appears exactly once, with the blank line
    # between the two lines preserved (layout inheritance).
    header = f"{TICKER_LINE_PREFIX}{ANON_TICKER}]\n\n{NAME_LINE_PREFIX}{ANON_NAME}]"
    assert anon.count(header) == 1
    # Preamble before the ticker line and everything after the name line
    # are byte-identical to the named prompt.
    ticker_at = PROMPT.index(TICKER_LINE_PREFIX)
    assert anon[: anon.index(TICKER_LINE_PREFIX)] == PROMPT[:ticker_at]
    named_tail = PROMPT[PROMPT.index("\n\n", PROMPT.index(NAME)) :]
    anon_tail = anon[anon.index("\n\n", anon.index(ANON_NAME)) :]
    assert named_tail == anon_tail


def test_anonymous_prompt_idempotent_second_call_fails():
    anon = anonymous_prompt(PROMPT, TICKER, NAME)
    with pytest.raises(ValueError, match="exactly once"):
        anonymous_prompt(anon, TICKER, NAME)


def test_anonymous_prompt_fail_closed_on_missing_or_ambiguous_header():
    with pytest.raises(ValueError, match="exactly once"):
        anonymous_prompt("no header at all", "NSC", "X")
    duplicated = PROMPT + "\n\n" + f"{TICKER_LINE_PREFIX}{TICKER}]\n\n{NAME_LINE_PREFIX}{NAME}]"
    with pytest.raises(ValueError, match="exactly once"):
        anonymous_prompt(duplicated, TICKER, NAME)
