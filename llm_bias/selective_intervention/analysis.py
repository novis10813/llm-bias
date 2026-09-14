"""Gap / spread metrics and pre-registered gates for selective-intervention V1.

Protocol: docs/selective-intervention/proposal-v1.md §7. All inputs are
reduced margin statistics (nats) computed by the pipeline from the
per-prompt forward records.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .template import (
    BOTTOM_GROUP,
    GATE_ANON_SHIFT,
    GATE_GAP_FRACTION,
    GATE_MIN_CLEAN_GAP,
    GATE_MEAN_SHIFT,
    GATE_SPECIFICITY_FRACTION,
    TOP_GROUP,
)


def group_gap(per_ticker: Mapping[str, float]) -> float:
    """TOP-group mean minus BOTTOM-group mean (nats). Fail-closed on missing tickers."""
    for ticker in (*TOP_GROUP, *BOTTOM_GROUP):
        if ticker not in per_ticker:
            raise ValueError(f"group gap missing ticker {ticker}")
    top = statistics.fmean(per_ticker[t] for t in TOP_GROUP)
    bottom = statistics.fmean(per_ticker[t] for t in BOTTOM_GROUP)
    return top - bottom


def iqr(values: Sequence[float]) -> float:
    """Interquartile range (inclusive linear interpolation). Fail-closed on n < 4."""
    if len(values) < 4:
        raise ValueError(f"IQR needs at least 4 values, got {len(values)}")
    ordered = sorted(values)

    def quantile(q: float) -> float:
        index = (len(ordered) - 1) * q
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        fraction = index - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction

    return quantile(0.75) - quantile(0.25)


def per_ticker_margins(margins: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Per-company mean margin over its variants; fail-closed on empty variants."""
    per_ticker: dict[str, float] = {}
    for ticker, values in margins.items():
        if not values:
            raise ValueError(f"no variant margins for {ticker}")
        per_ticker[ticker] = statistics.fmean(values)
    return per_ticker


def _criterion(name: str, value: float, limit: float, *, inclusive_le: bool = True) -> dict[str, Any]:
    return {
        "gate": name,
        "value": value,
        "limit": limit,
        "pass": value <= limit if inclusive_le else value >= limit,
    }


def evaluate_gates(
    clean: dict[str, Any],
    intervention: dict[str, Any],
    random_arm: dict[str, Any],
    anon_clean: float,
    anon_intervention: float,
) -> dict[str, Any]:
    """G1a/G1b/G2/G3/G4 on reduced margin statistics (proposal §7).

    ``clean`` / ``intervention`` / ``random_arm`` each carry
    ``per_ticker`` (company means) and ``all_margins`` (every prompt).
    Returns the per-criterion dict + combined pass. G2 is not evaluable
    (``pass: null``) when the main-arm reduction is non-positive.
    """
    g_clean = group_gap(clean["per_ticker"])
    if abs(g_clean) < GATE_MIN_CLEAN_GAP:
        raise ValueError(
            f"clean group gap {g_clean:.4f} is degenerate (|gap| < {GATE_MIN_CLEAN_GAP}); "
            "gates are not evaluable"
        )
    s_clean = iqr(clean["per_ticker"].values())
    if s_clean <= 0:
        raise ValueError("clean IQR is degenerate (<= 0); G1b is not evaluable")

    g_int = group_gap(intervention["per_ticker"])
    s_int = iqr(intervention["per_ticker"].values())
    g_rand = group_gap(random_arm["per_ticker"])

    g1a = _criterion("G1a", abs(g_int), GATE_GAP_FRACTION * abs(g_clean))
    g1b = _criterion("G1b", s_int, GATE_GAP_FRACTION * s_clean)

    reduction_main = 1.0 - abs(g_int) / abs(g_clean)
    reduction_random = 1.0 - abs(g_rand) / abs(g_clean)
    if reduction_main > 0:
        g2 = _criterion("G2", reduction_random, GATE_SPECIFICITY_FRACTION * reduction_main)
        g2["reduction_main"] = reduction_main
        g2["reduction_random"] = reduction_random
    else:
        g2 = {
            "gate": "G2",
            "pass": None,
            "reason": "main-arm reduction is non-positive; G2 not evaluable",
            "reduction_main": reduction_main,
            "reduction_random": reduction_random,
        }

    mean_clean = statistics.fmean(clean["all_margins"])
    mean_int = statistics.fmean(intervention["all_margins"])
    g3 = _criterion("G3", abs(mean_int - mean_clean), GATE_MEAN_SHIFT)
    g3["mean_clean"] = mean_clean
    g3["mean_intervention"] = mean_int

    g4 = _criterion("G4", abs(anon_intervention - anon_clean), GATE_ANON_SHIFT)
    g4["anon_clean"] = anon_clean
    g4["anon_intervention"] = anon_intervention

    hard = [g1a["pass"], g1b["pass"], g3["pass"], g4["pass"]]
    soft_ok = g2["pass"] is not False
    return {
        "g1a": g1a,
        "g1b": g1b,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "group_gap_clean": g_clean,
        "group_gap_intervention": g_int,
        "group_gap_random": g_rand,
        "iqr_clean": s_clean,
        "iqr_intervention": s_int,
        "pass": all(hard) and soft_ok,
    }


def decision_flips(clean_decisions: Mapping[str, str], int_decisions: Mapping[str, str]) -> int:
    """Count of sell -> buy decision flips (prompt-level)."""
    if set(clean_decisions) != set(int_decisions):
        raise ValueError("decision keys mismatch")
    return sum(
        1
        for key, clean in clean_decisions.items()
        if clean == "sell" and int_decisions[key] == "buy"
    )
