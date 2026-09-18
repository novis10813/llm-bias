"""Shared analysis of paired generated Buy/Sell decisions."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

DECISIONS = frozenset({"buy", "sell"})


def parse_buy_sell_decision(generated_text: Any) -> str | None:
    """Parse a Buy/Sell decision from a generated JSON continuation.

    The first JSON object is authoritative. Text before that object and trailing
    stop-marker text are allowed, matching the existing JSON decision protocols.
    Invalid or non-Buy/Sell decisions return ``None``.
    """
    if not isinstance(generated_text, str):
        return None
    start = generated_text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(generated_text[start:])
    except json.JSONDecodeError:
        return None
    decision = value.get("decision") if isinstance(value, dict) else None
    return decision if decision in DECISIONS else None


def decision_flip_summary(
    clean_decisions: Mapping[str, Any],
    intervened_decisions: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize paired Buy/Sell decision flips.

    Invalid pairs are retained in the coverage counts but excluded from the
    flip-rate denominator. The returned rates therefore describe valid paired
    generated decisions, while parse coverage remains explicit.
    """
    if set(clean_decisions) != set(intervened_decisions):
        raise ValueError("clean and intervened decision keys mismatch")

    valid_pairs = 0
    flips = 0
    sell_to_buy = 0
    buy_to_sell = 0
    for key in clean_decisions:
        clean = clean_decisions[key]
        intervened = intervened_decisions[key]
        if clean not in DECISIONS or intervened not in DECISIONS:
            continue
        valid_pairs += 1
        if clean == intervened:
            continue
        flips += 1
        if clean == "sell":
            sell_to_buy += 1
        else:
            buy_to_sell += 1

    def rate(count: int) -> float | None:
        return count / valid_pairs if valid_pairs else None

    return {
        "pair_count": len(clean_decisions),
        "valid_pair_count": valid_pairs,
        "invalid_pair_count": len(clean_decisions) - valid_pairs,
        "flip_count": flips,
        "flip_rate": rate(flips),
        "sell_to_buy_count": sell_to_buy,
        "sell_to_buy_rate": rate(sell_to_buy),
        "buy_to_sell_count": buy_to_sell,
        "buy_to_sell_rate": rate(buy_to_sell),
    }


def generated_decision_flip_summary(
    pairs: Iterable[Mapping[str, Any]],
    *,
    pair_id_field: str = "pair_id",
    clean_text_field: str = "clean_generated_text",
    intervened_text_field: str = "intervened_generated_text",
) -> dict[str, Any]:
    """Parse and summarize paired generated-output records.

    Each record must contain one stable pair identifier and the clean and
    intervened generated texts. Duplicate identifiers and missing fields fail
    closed so a partial or ambiguous pairing cannot produce a rate.
    """
    clean: dict[str, str | None] = {}
    intervened: dict[str, str | None] = {}
    for index, row in enumerate(pairs, 1):
        if pair_id_field not in row:
            raise ValueError(f"pair {index}: missing {pair_id_field!r}")
        if clean_text_field not in row or intervened_text_field not in row:
            raise ValueError(f"pair {index}: missing generated text field")
        pair_id = str(row[pair_id_field])
        if not pair_id:
            raise ValueError(f"pair {index}: empty pair identifier")
        if pair_id in clean:
            raise ValueError(f"duplicate pair identifier: {pair_id}")
        clean[pair_id] = parse_buy_sell_decision(row[clean_text_field])
        intervened[pair_id] = parse_buy_sell_decision(row[intervened_text_field])
    return decision_flip_summary(clean, intervened)


__all__ = [
    "DECISIONS",
    "decision_flip_summary",
    "generated_decision_flip_summary",
    "parse_buy_sell_decision",
]
