"""Unit tests for selective-intervention V1 metrics and gates (no model).

Covers group gap / IQR / per-ticker reductions, the pre-registered gate
boundaries (G1a/G1b/G2/G3/G4), the G2 not-evaluable path, degenerate
clean-gap fail-closed, and decision flip counting.
"""
from __future__ import annotations

import pytest

from llm_bias.selective_intervention.analysis import (
    decision_flips,
    evaluate_gates,
    group_gap,
    iqr,
    per_ticker_margins,
)


# ── primitive metrics ─────────────────────────────────────────────────────────


def test_group_gap_known():
    per_ticker = {"NSC": 2.0, "BLK": 1.0, "IT": -2.0, "BDX": -1.0}
    assert group_gap(per_ticker) == pytest.approx(3.0)


def test_group_gap_missing_ticker_raises():
    with pytest.raises(ValueError, match="BLK"):
        group_gap({"NSC": 2.0, "IT": -2.0, "BDX": -1.0})


def test_iqr_known():
    assert iqr([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.5)
    assert iqr([3.0, 1.0, 4.0, 2.0]) == pytest.approx(1.5)  # order-invariant


def test_iqr_too_few_raises():
    with pytest.raises(ValueError):
        iqr([1.0, 2.0, 3.0])


def test_per_ticker_margins():
    out = per_ticker_margins({"A": [1.0, 3.0], "B": [2.0, 2.0]})
    assert out == {"A": 2.0, "B": 2.0}
    with pytest.raises(ValueError):
        per_ticker_margins({"A": []})


def test_decision_flips():
    clean = {"p1": "sell", "p2": "sell", "p3": "buy"}
    patched = {"p1": "buy", "p2": "sell", "p3": "buy"}
    assert decision_flips(clean, patched) == 1
    with pytest.raises(ValueError):
        decision_flips(clean, {"p1": "buy"})


# ── gate evaluation ───────────────────────────────────────────────────────────


def _stats(per_ticker: dict[str, float], n_variants: int = 4, mean_offset: float = 0.0) -> dict:
    all_margins = [value + mean_offset for value in per_ticker.values() for _ in range(n_variants)]
    return {"per_ticker": per_ticker, "all_margins": all_margins}


def test_evaluate_gates_all_pass():
    clean = _stats({"NSC": 2.0, "BLK": 1.8, "IT": -2.0, "BDX": -1.9})
    intervention = _stats({"NSC": 1.0, "BLK": 0.8, "IT": -1.2, "BDX": -0.6})
    random_arm = _stats({"NSC": 1.9, "BLK": 1.7, "IT": -1.9, "BDX": -1.8})
    gate = evaluate_gates(clean, intervention, random_arm, -3.0, -3.0)
    assert gate["pass"] is True
    assert all(gate[name]["pass"] is True for name in ("g1a", "g1b", "g2", "g3", "g4"))
    assert gate["group_gap_clean"] == pytest.approx(3.85)
    assert gate["group_gap_intervention"] == pytest.approx(1.8)


def test_gap_halving_threshold():
    # With a 2+2 group population, iqr == |group gap|, so G1a/G1b move
    # together; test the 0.5 scaling threshold as a pair.
    clean = _stats({"NSC": 2.0, "BLK": 1.8, "IT": -2.0, "BDX": -1.9})

    def scaled(f: float) -> dict:
        return _stats({"NSC": 2.0 * f, "BLK": 1.8 * f, "IT": -2.0 * f, "BDX": -1.9 * f})

    # random arm = clean → zero reduction (G2 passes whenever main reduces).
    gate_below = evaluate_gates(clean, scaled(0.49), clean, -3.0, -3.0)
    gate_above = evaluate_gates(clean, scaled(0.51), clean, -3.0, -3.0)
    assert gate_below["g1a"]["pass"] is True and gate_below["g1b"]["pass"] is True
    assert gate_below["pass"] is True
    assert gate_above["g1a"]["pass"] is False and gate_above["g1b"]["pass"] is False
    assert gate_above["pass"] is False


def test_evaluate_gates_g2_threshold_and_not_evaluable():
    clean = _stats({"NSC": 2.0, "BLK": 1.8, "IT": -2.0, "BDX": -1.9})
    # Main arm: strong reduction (gap 1.0, mean preserved).
    intervention = _stats({"NSC": 0.5, "BLK": 0.5, "IT": -0.5, "BDX": -0.5})
    # Random arm: 20% of the main reduction (pass) vs 30% (fail).
    main_reduction = 1.0 - 1.0 / 3.85
    rand_gap_pass = 3.85 - (3.85 - 1.0) * 0.20
    rand_gap_fail = 3.85 - (3.85 - 1.0) * 0.30
    random_pass = _stats({"NSC": rand_gap_pass / 2, "BLK": rand_gap_pass / 2,
                          "IT": -rand_gap_pass / 2, "BDX": -rand_gap_pass / 2})
    random_fail = _stats({"NSC": rand_gap_fail / 2, "BLK": rand_gap_fail / 2,
                          "IT": -rand_gap_fail / 2, "BDX": -rand_gap_fail / 2})
    gate_pass = evaluate_gates(clean, intervention, random_pass, -3.0, -3.0)
    gate_fail = evaluate_gates(clean, intervention, random_fail, -3.0, -3.0)
    assert gate_pass["g2"]["pass"] is True
    assert gate_fail["g2"]["pass"] is False
    assert abs(gate_pass["g2"]["reduction_main"] - main_reduction) < 1e-12

    # Not evaluable: main arm does not reduce the gap at all.
    gate_none = evaluate_gates(clean, clean, random_pass, -3.0, -3.0)
    assert gate_none["g2"]["pass"] is None
    assert gate_none["g2"]["reduction_main"] == pytest.approx(0.0)
    assert gate_none["pass"] is False  # G1a fails anyway


def test_evaluate_gates_g3_g4_thresholds():
    clean = _stats({"NSC": 2.0, "BLK": 1.8, "IT": -2.0, "BDX": -1.9})
    intervention = _stats({"NSC": 1.0, "BLK": 0.8, "IT": -1.2, "BDX": -0.6})  # mean 0.0
    random_arm = _stats({"NSC": 1.9, "BLK": 1.7, "IT": -1.9, "BDX": -1.8})
    gate_ok = evaluate_gates(clean, intervention, random_arm, -3.0, -3.05)  # G4 = 0.05
    assert gate_ok["g4"]["pass"] is True
    assert gate_ok["g3"]["pass"] is True  # shift = |0.0 + 0.025| = 0.025
    # Mean offset: base shift 0.025; +0.12 -> 0.145 (pass), +0.13 -> 0.155 (fail).
    gate_g3_pass = evaluate_gates(clean, _stats(
        {"NSC": 1.0, "BLK": 0.8, "IT": -1.2, "BDX": -0.6}, mean_offset=0.12),
        random_arm, -3.0, -3.0)
    assert gate_g3_pass["g3"]["pass"] is True
    gate_g3_fail = evaluate_gates(clean, _stats(
        {"NSC": 1.0, "BLK": 0.8, "IT": -1.2, "BDX": -0.6}, mean_offset=0.13),
        random_arm, -3.0, -3.0)
    assert gate_g3_fail["g3"]["pass"] is False
    assert gate_g3_fail["pass"] is False
    # G4 past the limit.
    gate_g4_fail = evaluate_gates(clean, intervention, random_arm, -3.0, -3.2)
    assert gate_g4_fail["g4"]["pass"] is False
    assert gate_g4_fail["pass"] is False


def test_evaluate_gates_degenerate_clean_gap_raises():
    flat = _stats({"NSC": 0.5, "BLK": 0.4, "IT": 0.45, "BDX": 0.48})
    with pytest.raises(ValueError, match="degenerate"):
        evaluate_gates(flat, flat, flat, -3.0, -3.0)
