"""flip-monotonicity-v1: curve extraction, monotonicity metrics and the cone4 criterion."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_flip_monotonicity", ROOT / "scripts/summarize_flip_monotonicity.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def _pa(rates, sign=1, n=10):
    return [{"alpha": sign * a, "on_class_n": n, "on_flip_itt": r} for a, r in rates]


def test_curve_uses_one_sign_with_denominator():
    pa = _pa([(1, 0.0), (2, 0.5)]) + _pa([(1, None)], sign=-1, n=0)
    assert fm.curve(pa, 1) == [(1, 0.0), (2, 0.5)]
    assert fm.curve(pa, -1) == []


def test_plateau_is_monotone_and_drop_is_counted():
    m = fm.monotonicity([0.0, 0.5, 1.0, 1.0])
    assert m["monotone_step_fraction"] == 1.0 and m["drops"] == 0
    m = fm.monotonicity([0.0, 1.0, 0.98, 0.0])
    assert m["drops"] == 2 and abs(m["monotone_step_fraction"] - 1 / 3) < 1e-9 and m["max_drop"] == 0.98


def test_criterion_requires_strict_gain_over_random_axes_and_no_extra_drops():
    smooth = [(1, 0.0), (2, 0.5), (4, 1.0)]
    bumpy = [(1, 0.0), (2, 0.8), (4, 0.4)]
    ok = fm.sign_verdict({"cone4": smooth, "dim_orth_rand4": bumpy, "dim": smooth})
    assert ok["status"] == "holds"
    tie = fm.sign_verdict({"cone4": smooth, "dim_orth_rand4": smooth, "dim": smooth})
    assert tie["status"] == "fails" and tie["tie_with_dim_orth_rand4"]


def test_single_sign_model_is_flagged():
    v = fm.model_verdict({"sign_+": {"status": "holds"}, "sign_-": {"status": "untestable"}})
    assert v == {"status": "holds", "signs_evaluated": ["sign_+"], "single_sign": True}
    assert fm.model_verdict({"sign_+": {"status": "untestable"}})["status"] == "untestable"
