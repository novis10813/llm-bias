"""Pre-registered Phase 1 grouping, gates, and compact descriptive summaries."""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

GATE_THRESHOLDS = {"parse": 0.95, "sign": 0.90, "power_responsive": 10, "power_insensitive": 10}
DECISIONS = ("buy", "sell")
EXCLUDED = "excluded"
# Protocol §8 item 8: first JSON key is "decision", tolerant of whitespace
# formatting (the model pretty-prints the object; pilot-verified).
DECISION_KEY_PREFIX = re.compile(r'^\s*\{\s*"decision"\s*:\s*"')


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite analysis value")
    return value


def _iqr(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return _finite(values[int(0.75 * (len(values) - 1))] - values[int(0.25 * (len(values) - 1))])


def classify_group(n15: str | None, p15: str | None) -> str | None:
    """Pre-registered grouping rule (protocol §5).

    Returns None when either polarity-extreme decision is unparsed; such
    companies are excluded from grouping (not "mixed") and reported
    descriptively, while still counting toward the G-P1 denominator.
    """
    if n15 not in DECISIONS or p15 not in DECISIONS:
        return None
    if n15 == "sell" and p15 == "buy":
        return "evidence-responsive"
    if n15 == "sell" and p15 == "sell":
        return "fixed-sell"
    if n15 == "buy" and p15 == "buy":
        return "fixed-buy"
    return "mixed"


def group_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ticker: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row.get("arm") == "primary":
            by_ticker[row["ticker"]][row["condition"]] = row
    output = []
    for ticker in sorted(by_ticker):
        rows = by_ticker[ticker]
        n15_row = rows.get("N15", {})
        p15_row = rows.get("P15", {})
        group = classify_group(n15_row.get("decision"), p15_row.get("decision"))
        d0 = rows.get("zero", {}).get("decision")
        m0 = rows.get("zero", {}).get("margin")
        prior_label = None
        if group in {"fixed-sell", "fixed-buy"} and d0 in DECISIONS:
            direction = "sell" if group == "fixed-sell" else "buy"
            prior_label = "prior-consistent" if d0 == direction else "prior-reversed"
        contrast = None
        n15_margin = n15_row.get("margin")
        p15_margin = p15_row.get("margin")
        if n15_margin is not None and p15_margin is not None:
            contrast = _finite(float(p15_margin) - float(n15_margin))
        output.append(
            {
                "ticker": ticker,
                "gics_sector": n15_row.get("gics_sector"),
                "split": n15_row.get("split"),
                "group": group,
                "prior_label": prior_label,
                "d0": d0,
                "m0": m0,
                "contrast_c": contrast,
            }
        )
    return output


def evaluate_gates(records: list[dict[str, Any]], groups: list[dict[str, Any]], determinism: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    parsed = [row for row in records if row.get("decision") in DECISIONS]
    parse_rate = len(parsed) / total if total else 0.0
    # Protocol §6 G-P2: M = 0 is a mismatch for both decisions.
    sign_agree = sum(
        (row["margin"] > 0.0) if row["decision"] == "buy" else (row["margin"] < 0.0)
        for row in parsed
    )
    sign_rate = sign_agree / len(parsed) if parsed else 0.0
    responsive = sum(row["group"] == "evidence-responsive" for row in groups)
    insensitive = sum(row["group"] in {"fixed-sell", "fixed-buy"} for row in groups)
    text_mismatch = int(determinism["text_mismatch_count"])
    max_delta = float(determinism["max_abs_delta_margin"])
    gate = {
        "G-P1": {"value": _finite(parse_rate), "pass": parse_rate >= 0.95},
        "G-P2": {"value": _finite(sign_rate), "pass": sign_rate >= 0.90},
        "G-P3": {
            "value": {"text_mismatch_count": text_mismatch, "max_abs_delta_margin": _finite(max_delta)},
            "pass": text_mismatch == 0 and max_delta == 0.0,
        },
        "G-P4": {
            "value": {"responsive": responsive, "insensitive": insensitive},
            "pass": responsive >= 10 and insensitive >= 10,
        },
    }
    return {"gates": gate, "fallback": not gate["G-P4"]["pass"]}


def _order_swap_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary: dict[tuple[str, str], dict[str, Any]] = {}
    swapped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (row["ticker"], row["condition"])
        if row.get("arm") == "primary":
            primary[key] = row
        elif row.get("arm") == "order_swap":
            swapped[key] = row
    pairs = sorted(set(primary) & set(swapped))
    if not pairs:
        return {"n_pairs": 0, "decision_flip_rate": None, "mean_abs_delta_margin": None}
    flips = 0
    deltas: list[float] = []
    for key in pairs:
        base, reverse = primary[key], swapped[key]
        if (
            base.get("decision") in DECISIONS
            and reverse.get("decision") in DECISIONS
            and base["decision"] != reverse["decision"]
        ):
            flips += 1
        deltas.append(abs(float(base["margin"]) - float(reverse["margin"])))
    return {
        "n_pairs": len(pairs),
        "decision_flip_rate": _finite(flips / len(pairs)),
        "mean_abs_delta_margin": _finite(statistics.mean(deltas)),
    }


def descriptive_stats(records: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in records if row.get("arm") == "primary"]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        by_condition[row["condition"]].append(row)
    curves: dict[str, dict[str, Any]] = {}
    for condition, rows in sorted(by_condition.items()):
        margins = [_finite(row["margin"]) for row in rows]
        decisions = [row.get("decision") for row in rows if row.get("decision") in DECISIONS]
        curves[condition] = {
            "mean_margin": _finite(statistics.mean(margins)),
            "iqr_margin": _iqr(margins),
            "buy_rate": _finite(sum(d == "buy" for d in decisions) / len(decisions)) if decisions else None,
        }
    sectors: dict[str, Counter] = defaultdict(Counter)
    for row in groups:
        sectors[row["gics_sector"]][row["group"] or EXCLUDED] += 1
    contrasts = sorted(row["contrast_c"] for row in groups if row["contrast_c"] is not None)
    generated = [row["generated_text"] for row in records if isinstance(row.get("generated_text"), str)]
    prefix_match = None
    if generated:
        prefix_match = _finite(
            sum(bool(DECISION_KEY_PREFIX.match(text)) for text in generated) / len(generated)
        )
    return {
        "anonymous_base_rate": {
            row["condition"]: {"decision": row.get("decision"), "margin": row["margin"]}
            for row in records
            if row.get("arm") == "anon"
        },
        "population_polarity": curves,
        "group_by_sector": {sector: dict(counts) for sector, counts in sorted(sectors.items())},
        "prior_distribution": {
            "d0_count": sum(row["d0"] is not None for row in groups),
            "margin_iqr": _iqr([row["m0"] for row in groups if row["m0"] is not None]),
            "prior_consistent": sum(row["prior_label"] == "prior-consistent" for row in groups),
            "prior_reversed": sum(row["prior_label"] == "prior-reversed" for row in groups),
        },
        "order_swap": _order_swap_stats(records),
        "generation_prefix_match": prefix_match,
        "response_contrast_c": {
            "count": len(contrasts),
            "minimum": contrasts[0] if contrasts else None,
            "maximum": contrasts[-1] if contrasts else None,
        },
    }
