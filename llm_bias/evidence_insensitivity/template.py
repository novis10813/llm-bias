"""Frozen Phase 1 prompt family and tokenizer span construction."""
from __future__ import annotations

import hashlib
from typing import Any

from llm_bias.core.prompt_input.encoding import input_ids, token_span

POS_ITEM = (
    "According to analysts, the company recently secured a major long-term partnership "
    "that is expected to significantly expand its revenue base, providing a catalyst "
    "for a {p}% increase in the stock price over the coming quarter."
)
NEG_ITEM = (
    "Recent operational challenges in the company's main business division have raised "
    "concerns about execution timelines, with multiple initiatives experiencing delays "
    "and cost overruns that could compress margins by approximately {n}% over the coming "
    "quarters, potentially driving the stock price lower."
)
DECISION_PREFIX = '{\n  "decision": "'
CONDITIONS = ("zero", "N6", "N8", "N10", "N15", "P6", "P8", "P10", "P15")
POLARITY_SCORES = {"zero": 0, "N6": -1, "N8": -2, "N10": -3, "N15": -4, "P6": 1, "P8": 2, "P10": 3, "P15": 4}
CONDITION_MATRIX = {
    "zero": (None, None, 0),
    "N6": (5, 6, -1), "N8": (5, 8, -2), "N10": (5, 10, -3), "N15": (5, 15, -4),
    "P6": (6, 5, 1), "P8": (8, 5, 2), "P10": (10, 5, 3), "P15": (15, 5, 4),
}

PROMPT_HEAD = "Refer to the evidence below to make a final investment decision.\n"
PROMPT_TAIL = (
    'Respond with one valid JSON object containing only the keys "decision" '
    '(buy | sell) and "reason" (brief justification). Do not choose hold.'
)


def evidence_lines(condition: str, *, reverse: bool = False) -> tuple[str, str]:
    if condition not in CONDITION_MATRIX:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "zero":
        return ("1. No evidence provided.",)
    positive, negative, _ = CONDITION_MATRIX[condition]
    items = (POS_ITEM.format(p=positive), NEG_ITEM.format(n=negative))
    if reverse:
        items = items[::-1]
    return tuple(f"{index}. {item}" for index, item in enumerate(items, 1))


def build_prompt(ticker: str, company_name: str, condition: str, *, anonymous: bool = False, reverse: bool = False) -> str:
    """Build the exact frozen raw prompt, including the anonymous header option."""
    if condition not in CONDITION_MATRIX:
        raise ValueError(f"unknown condition: {condition}")
    shown_ticker = "[TICKER]" if anonymous else f"[{ticker}]"
    shown_name = "[Company X]" if anonymous else f"[{company_name}]"
    evidence = "\n".join(evidence_lines(condition, reverse=reverse))
    return (
        f"{PROMPT_HEAD}Stock Ticker: {shown_ticker}\n"
        f"Stock Name: {shown_name}\n"
        "--- Evidence ---\n"
        f"{evidence}\n"
        "---\n"
        f"{PROMPT_TAIL}"
    )


def prompt_char_spans(prompt: str) -> dict[str, tuple[int, int]]:
    header_end = prompt.index("--- Evidence ---")
    evidence_start = header_end + len("--- Evidence ---\n")
    evidence_end = prompt.index("\n---\n", evidence_start)
    instruction_start = evidence_end + len("\n---\n")
    return {
        # Delimiters belong to the adjacent region so the three ranges are
        # mutually exclusive and cover the complete prompt.
        "header": (0, evidence_start),
        "evidence": (evidence_start, instruction_start),
        "instruction": (instruction_start, len(prompt)),
    }


def _token_span(tokenizer: Any, text: str, chars: tuple[int, int]) -> list[int]:
    span = token_span(tokenizer, text, chars[0], chars[1], add_special_tokens=True)
    if span is None:
        raise ValueError(f"could not map span {chars}")
    return [int(span[0]), int(span[1])]


def build_prompt_record(tokenizer: Any, ticker: str, company_name: str, sector: str, condition: str, *, split: str, arm: str = "primary", anonymous: bool = False, reverse: bool = False, format_fn: Any | None = None) -> dict[str, Any]:
    prompt = build_prompt(ticker, company_name, condition, anonymous=anonymous, reverse=reverse)
    formatted = format_fn(tokenizer, prompt, use_chat_template=True, enable_thinking=False) if format_fn else prompt
    body_start = formatted.find(prompt)
    if body_start < 0:
        raise ValueError("prompt body not found in formatted text")
    spans = {name: _token_span(tokenizer, formatted, (body_start + start, body_start + end)) for name, (start, end) in prompt_char_spans(prompt).items()}
    ids = input_ids(tokenizer, formatted, add_special_tokens=True)
    if not (spans["header"][0] == 0 and spans["instruction"][1] == len(ids)):
        # Chat-template special tokens may precede the body; coverage is checked below.
        pass
    if set(range(spans["header"][0], spans["header"][1])) & set(range(spans["evidence"][0], spans["evidence"][1])):
        raise ValueError("prompt spans overlap")
    score = POLARITY_SCORES[condition]
    return {
        "prompt_id": hashlib.sha256(formatted.encode()).hexdigest()[:12],
        "ticker": ticker, "company_name": company_name, "gics_sector": sector,
        "split": split, "arm": arm, "condition": condition,
        "polarity_score": score, "prompt_text": formatted,
        "evidence_sha": hashlib.sha256("\n".join(evidence_lines(condition, reverse=reverse)).encode()).hexdigest(),
        "header_span": spans["header"], "evidence_span": spans["evidence"], "instruction_span": spans["instruction"],
    }


def condition_polarity(condition: str) -> int:
    return POLARITY_SCORES[condition]


def resolve_spans(tokenizer: Any, formatted: str, prompt: str) -> dict[str, list[int]]:
    """Resolve the three frozen regions in a formatted prompt."""
    body_start = formatted.find(prompt)
    if body_start < 0:
        raise ValueError("prompt body not found in formatted text")
    spans = {
        name: _token_span(tokenizer, formatted, (body_start + start, body_start + end))
        for name, (start, end) in prompt_char_spans(prompt).items()
    }
    return spans
