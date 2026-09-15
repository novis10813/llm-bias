"""Unit tests for the axis-decomposition probe (no model loads)."""
from __future__ import annotations

import torch

from llm_bias.entity_concept_decision.axis_decomposition import (
    _analyze,
    _build_axes,
)

COMPANIES = ["A:rev0:ord0", "B:rev0:ord0"]
AXES = ["pc1", "pc2", "stance", "mean_state", "random_0", "random_1"]
SLOPES = {"pc1": 2.0, "pc2": 0.0, "stance": 1.0, "mean_state": 0.0, "random_0": 1.0, "random_1": 1.0}


def _linear_records(dose: float = 0.1) -> list[dict]:
    records = []
    for cid in COMPANIES:
        records.append({"id": cid, "role": "baseline", "axis": "none", "dose": 0.0,
                        "margin": -2.0, "status": "ok"})
        for axis in AXES:
            for sign in (1.0, -1.0):
                signed = sign * dose
                delta = signed * SLOPES[axis]
                records.append({"id": cid, "role": "intervention", "axis": axis,
                                "dose": signed, "delta_margin": delta,
                                "margin": -2.0 + delta, "status": "ok"})
    return records


def test_analyze_recovers_exact_slopes_and_fractions() -> None:
    records = _linear_records()
    summary = _analyze(records, COMPANIES, [0.1], d_model=4, axis_names=AXES)
    for axis in AXES:
        assert abs(summary["per_axis"][axis]["slope"] - SLOPES[axis]) < 1e-9, axis
    # random draws: (w.r)^2 in {1, 1} -> mean_sq = 1 -> ||w||^2 = 4*1 = 4
    assert abs(summary["w_norm_estimate"]["w_norm_sq"] - 4.0) < 1e-9
    assert abs(summary["w_norm_estimate"]["w_norm"] - 2.0) < 1e-9
    assert summary["w_norm_estimate"]["n_random_draws"] == 8  # 2 axes x 2 companies x 2 signs
    fr = summary["weight_fractions"]
    assert abs(fr["pc1"] - 1.0) < 1e-9
    assert abs(fr["stance"] - 0.25) < 1e-9
    assert abs(fr["mean_state"] - 0.0) < 1e-9
    assert abs(fr["sum_pc_subspace"] - 1.0) < 1e-9
    assert abs(fr["sum_all_structured"] - 1.25) < 1e-9
    # slope t-statistic: zero spread in the linear records -> se == 0 -> None
    assert summary["per_axis"]["pc1"]["slope_t"] is None


def test_analyze_ignores_non_finite_rows() -> None:
    records = _linear_records()
    # drop one company's pc1 draws at +0.1; pc1 slope estimate must be unchanged
    records = [r for r in records if not (r["axis"] == "pc1" and r["dose"] == 0.1 and r["id"] == "B:rev0:ord0")]
    records.append({"id": "B:rev0:ord0", "role": "intervention", "axis": "pc1",
                    "dose": 0.1, "status": "non_finite"})
    summary = _analyze(records, COMPANIES, [0.1], d_model=4, axis_names=AXES)
    assert abs(summary["per_axis"]["pc1"]["slope"] - 2.0) < 1e-9
    assert summary["per_axis"]["pc1"]["n_draws"] == 3  # one draw dropped, non_finite skipped


def test_build_axes_shape_unit_and_pc_alignment() -> None:
    # company states differ along two known orthogonal directions u, v
    u = torch.zeros(6); u[0] = 1.0
    v = torch.zeros(6); v[3] = 1.0
    base = torch.full((6,), 0.5)
    states = [base + a * u + b * v for a, b in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))]
    stance = torch.zeros(6); stance[5] = 1.0
    axes = _build_axes(states, stance, n_pcs=8, n_random=2, random_seed=0)
    # min(8, min(4 companies, 6 dims)) = 4 PCs + stance + mean_state + 2 randoms
    assert len(axes) == 4 + 1 + 1 + 2
    for name, vector in axes:
        assert abs(float(vector.norm()) - 1.0) < 1e-6, name
        assert vector.shape == (6,)
    # top PCs must span {u, v}: each of u, v aligns with some PC
    pcs = [vec for name, vec in axes if name.startswith("pc")]
    for target in (u, v):
        best = max(abs(float(target @ pc)) for pc in pcs)
        assert best > 0.999, f"no PC aligns with a known variance direction (best {best})"
    # mean_state direction: all states share `base`, so the mean is base (u/v cancel)
    mean_vec = dict(axes)["mean_state"]
    assert abs(abs(float(base @ mean_vec)) - float(base.norm())) < 1e-6


def test_build_axes_deterministic_randoms() -> None:
    states = [torch.randn(6) for _ in range(4)]
    stance = torch.zeros(6); stance[0] = 1.0
    a = _build_axes(states, stance, n_pcs=2, n_random=3, random_seed=7)
    b = _build_axes(states, stance, n_pcs=2, n_random=3, random_seed=7)
    assert [n for n, _ in a] == [n for n, _ in b]
    for (_, va), (_, vb) in zip(a, b):
        assert torch.equal(va, vb)
    c = _build_axes(states, stance, n_pcs=2, n_random=3, random_seed=8)
    assert not all(torch.equal(va, vc) for (_, va), (_, vc) in zip(a, c))
