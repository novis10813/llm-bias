"""Unit tests for the entity-concept-decision attribution pure logic (no model)."""
from __future__ import annotations

import pytest

from llm_bias.entity_concept_decision.attribution import _pearson_r, _summarize


def test_pearson_r_known_values() -> None:
    assert _pearson_r([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert _pearson_r([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert _pearson_r([1, 2, 3], [5, 5, 5]) is None
    assert _pearson_r([1, 2], [2, 4]) is None  # needs >= 3
    assert _pearson_r([1, 2, 3], [1, 9, 9]) is not None


def _fake_attribution(stance_score: float, margin: float, sal_s: list[float], sal_m: list[float]) -> dict:
    return {
        "id": "x",
        "stance_score": stance_score,
        "margin": margin,
        "stance_saliency": sal_s,
        "margin_saliency": sal_m,
        "top_stance_tokens": [{"rank": 0, "token": "a", "saliency": sal_s[0]}],
        "top_margin_tokens": [{"rank": 0, "token": "b", "saliency": sal_m[0]}],
    }


def test_summarize_r2_and_per_company_correlation() -> None:
    company_ids = ["A", "B", "C", "D"]
    # stance scores perfectly track margins (r=1).
    attributions = {
        "A": _fake_attribution(1.0, -2.0, [1.0, 0.5, 0.0], [0.1, 0.2, 0.3]),
        "B": _fake_attribution(2.0, -1.5, [0.5, 1.0, 0.5], [0.3, 0.2, 0.1]),
        "C": _fake_attribution(3.0, -1.0, [0.0, 0.5, 1.0], [0.9, 0.8, 0.7]),
        "D": _fake_attribution(4.0, -0.5, [0.5, 0.5, 0.5], [0.1, 0.1, 0.9]),
    }
    xs = [attributions[c]["stance_score"] for c in company_ids]
    ys = [attributions[c]["margin"] for c in company_ids]
    summary = _summarize(company_ids, attributions, xs, ys, layer=15, final_layer=31, stance_meta={"layer": 15, "n_pairs": 4}, top_k=3)
    # xs=1,2,3,4 vs ys=-2,-1.5,-1,-0.5 are perfectly linear -> r2 ~ 1.
    assert summary["stance_score_vs_margin_r2"] == pytest.approx(1.0)
    assert summary["n_companies"] == 4
    assert summary["layer"] == 15
    # Per-company A: sal_s=[1,.5,0], sal_m=[.1,.2,.3] are perfectly anti-linear -> r ~ -1.
    assert summary["per_company"]["A"]["stance_vs_margin_saliency_r"] == pytest.approx(-1.0)
    # Per-company C: sal_s=[0,.5,1], sal_m=[.9,.8,.7] anti-linear -> r ~ -1.
    assert summary["per_company"]["C"]["stance_vs_margin_saliency_r"] == pytest.approx(-1.0)
    # Constant saliency -> correlation undefined (None).
    attributions["B"]["stance_saliency"] = [0.5, 0.5, 0.5]
    xs2 = [attributions[c]["stance_score"] for c in company_ids]
    summary2 = _summarize(company_ids, attributions, xs2, ys, layer=15, final_layer=31, stance_meta={"layer": 15, "n_pairs": 4}, top_k=3)
    assert summary2["per_company"]["B"]["stance_vs_margin_saliency_r"] is None
    assert summary["scientific_status"] == "not_evaluated"
    assert summary["purpose"] == "development"


def test_summarize_constant_margin_gives_none_r2() -> None:
    company_ids = ["A", "B", "C"]
    attributions = {
        c: _fake_attribution(1.0, -1.0, [1.0, 0.0], [0.0, 1.0]) for c in company_ids
    }
    xs = [1.0, 1.0, 1.0]
    ys = [-1.0, -1.0, -1.0]
    summary = _summarize(company_ids, attributions, xs, ys, layer=15, final_layer=31, stance_meta={"n_pairs": 2}, top_k=2)
    assert summary["stance_score_vs_margin_r2"] is None
    assert summary["stance_score_vs_margin_r"] is None
