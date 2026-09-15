"""Focused tests for the multi-layer concept-vs-stance separability analysis."""
from __future__ import annotations

import torch

from llm_bias.entity_concept_decision.layer_scan import _analyze_layer, _fit_stance_direction, _ols_r2, _pearson_r2

DIM = 8


def _vec(*idx_val):
    v = torch.zeros(DIM, dtype=torch.float32)
    for i, val in idx_val:
        v[i] = val
    return v


def _inputs():
    rows = [
        {"id": "G1P", "concept_id": "G", "group_id": "G1", "role": "primary", "pole": "P", "text": "p"},
        {"id": "G1N", "concept_id": "G", "group_id": "G1", "role": "primary", "pole": "N", "text": "n"},
        {"id": "G2P", "concept_id": "G", "group_id": "G2", "role": "primary", "pole": "P", "text": "p2"},
        {"id": "G2N", "concept_id": "G", "group_id": "G2", "role": "primary", "pole": "N", "text": "n2"},
        {"id": "GE-PH", "concept_id": "G", "group_id": "GE", "role": "evaluation", "pole": "PH", "text": "a"},
        {"id": "GE-PL", "concept_id": "G", "group_id": "GE", "role": "evaluation", "pole": "PL", "text": "b"},
        {"id": "GE-NH", "concept_id": "G", "group_id": "GE", "role": "evaluation", "pole": "NH", "text": "c"},
        {"id": "GE-NL", "concept_id": "G", "group_id": "GE", "role": "evaluation", "pole": "NL", "text": "d"},
        {"id": "C1P", "concept_id": "C", "group_id": "C1", "role": "primary", "pole": "P", "text": "e"},
        {"id": "C1N", "concept_id": "C", "group_id": "C1", "role": "primary", "pole": "N", "text": "f"},
        {"id": "C2P", "concept_id": "C", "group_id": "C2", "role": "primary", "pole": "P", "text": "g"},
        {"id": "C2N", "concept_id": "C", "group_id": "C2", "role": "primary", "pole": "N", "text": "h"},
    ]
    vectors = {
        "G1P": _vec((1, 1.0), (0, 0.5)), "G1N": _vec((1, -1.0), (0, 0.5)),
        "G2P": _vec((1, 1.2), (0, 0.5)), "G2N": _vec((1, -1.2), (0, 0.5)),
        "GE-PH": _vec((0, 1.0), (3, 0.1)), "GE-PL": _vec((0, -1.0), (3, 0.1)),
        "GE-NH": _vec((0, 1.0), (4, 0.1)), "GE-NL": _vec((0, -1.0), (4, 0.1)),
        "C1P": _vec((0, 1.0), (2, 0.1)), "C1N": _vec((0, -1.0), (2, 0.1)),
        "C2P": _vec((0, 1.1), (2, 0.1)), "C2N": _vec((0, -1.1), (2, 0.1)),
        "c1": _vec((1, 1.0)), "c2": _vec((0, 0.1)), "c3": _vec((0, -0.1)), "c4": _vec((1, -1.0)),
    }
    comparisons = [
        {"id": "G1-primary", "concept_id": "G", "family_id": "G1", "kind": "primary", "positive_id": "G1P", "negative_id": "G1N"},
        {"id": "G2-primary", "concept_id": "G", "family_id": "G2", "kind": "primary", "positive_id": "G2P", "negative_id": "G2N"},
        {"id": "GE-eval-pos", "concept_id": "G", "family_id": "GE", "kind": "evaluation_at_positive_concept", "positive_id": "GE-PH", "negative_id": "GE-PL"},
        {"id": "GE-eval-neg", "concept_id": "G", "family_id": "GE", "kind": "evaluation_at_negative_concept", "positive_id": "GE-NH", "negative_id": "GE-NL"},
        {"id": "C1-primary", "concept_id": "C", "family_id": "C1", "kind": "primary", "positive_id": "C1P", "negative_id": "C1N"},
        {"id": "C2-primary", "concept_id": "C", "family_id": "C2", "kind": "primary", "positive_id": "C2P", "negative_id": "C2N"},
    ]
    return rows, comparisons, vectors, ["c1", "c2", "c3", "c4"]


def test_ols_r2_and_stance_direction():
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y = X @ torch.tensor([1.0, 2.0])
    assert _ols_r2(X, y) > 0.999
    assert _ols_r2(X, torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])) is None
    comps = [
        {"kind": "evaluation_at_positive_concept", "positive_id": "a", "negative_id": "b"},
        {"kind": "evaluation_at_negative_concept", "positive_id": "c", "negative_id": "d"},
        {"kind": "primary", "positive_id": "e", "negative_id": "f"},
    ]
    vecs = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 0.0]), "c": torch.tensor([2.0, 0.0]), "d": torch.tensor([0.0, 0.0]), "e": torch.tensor([0.0, 1.0]), "f": torch.tensor([0.0, 0.0])}
    sd, n = _fit_stance_direction(vecs, comps, 1e-6)
    assert n == 2 and sd is not None and abs(float(sd[0]) - 1.0) < 1e-5


def test_pearson_r2_extremes():
    assert _pearson_r2([1, 2, 3, 4], [2, 4, 6, 8]) > 0.999
    assert _pearson_r2([1, 2, 3, 4], [8, 6, 4, 2]) > 0.999  # sign irrelevant for r2
    assert _pearson_r2([1, 2, 3], [5, 5, 5]) is None  # constant y
    assert _pearson_r2([1, 2], [1, 2]) is None  # too few points


def test_g_orthogonal_to_stance_separates_companies_after_orthogonalization():
    rows, comparisons, vectors, companies = _inputs()
    out = _analyze_layer(vectors, rows, comparisons, companies, ["G"], random_seed=1729, random_count=8, min_norm=1e-6)
    g = out["concepts"]["G"]
    assert out["stance_status"] == "ok" and out["stance_n_pairs"] == 2
    # G direction = e1, orthogonal to stance (e0)
    assert abs(g["cos_with_stance"]) < 1e-5
    assert g["orthogonalized_status"] == "ok"
    # companies c1 (G high) and c4 (G low) separate after orthogonalization
    ortho = g["company_orthogonalized_scores"]
    assert ortho["c1"] > 0.5 and ortho["c4"] < -0.5
    assert g["company_orthogonalized_spread"] > 1.5
    # LO-group: each excluded group's delta is positive (direction e1)
    assert all(item["group_mean_delta"] > 1.0 for item in g["leave_one_group_out"])
    assert g["random_primary_q95"] is not None


def test_c_aligned_with_stance_collapses_after_orthogonalization():
    rows, comparisons, vectors, companies = _inputs()
    out = _analyze_layer(vectors, rows, comparisons, companies, ["C"], random_seed=1729, random_count=8, min_norm=1e-6)
    c = out["concepts"]["C"]
    # C direction = e0, exactly along stance
    assert c["cos_with_stance"] > 0.999
    assert c["orthogonalized_status"] == "degenerate_after_orthogonalization"
    assert c["company_orthogonalized_scores"] == {}
    assert c["company_orthogonalized_spread"] is None
    # raw company scores all near zero on e0 (companies have no e0 component except c2/c3)
    raw = c["company_raw_scores"]
    assert abs(raw["c1"]) < 1e-5 and abs(raw["c4"]) < 1e-5
