"""Unit tests for the stance causal probe (no model loads)."""
from __future__ import annotations

import math

import torch

from llm_bias.entity_concept_decision.stance_causal import (
    _additive_transform,
    _analyze,
    _fit_stance_direction,
)


def _fake_records() -> list[dict]:
    # 2 companies, doses +/-0.05, directions stance + random_0.
    # stance: +0.05 -> +0.2 delta, -0.05 -> -0.1 delta; randoms: ~0.
    records = []
    for cid, plus, minus in (("A:rev0:ord0", 0.20, -0.10), ("B:rev0:ord0", 0.30, -0.30)):
        records.append({"id": cid, "role": "baseline", "dose": 0.0, "direction": "none",
                        "margin": -2.0, "p_buy_2way": 0.12, "elapsed_seconds": 0.1})
        for dose, direction, delta in (
            (0.05, "stance", plus), (-0.05, "stance", minus),
            (0.05, "random_0", 0.01), (-0.05, "random_0", -0.01),
        ):
            records.append({"id": cid, "role": "intervention", "dose": dose, "direction": direction,
                            "margin": -2.0 + delta, "delta_margin": delta,
                            "p_buy_2way": 1.0 / (1.0 + math.exp(-(-2.0 + delta))),
                            "elapsed_seconds": 0.1})
    return records


def test_analyze_aggregates_dose_response_and_flips() -> None:
    records = _fake_records()
    summary = _analyze(records, ["A:rev0:ord0", "B:rev0:ord0"])
    by_dose = {d["dose"]: d for d in summary["dose_response"]}
    assert set(by_dose) == {0.05, -0.05}
    assert by_dose[0.05]["stance_mean_delta"] == 0.25
    assert by_dose[0.05]["random_mean_delta"] == 0.01
    assert by_dose[0.05]["stance_minus_random"] == 0.24
    assert by_dose[0.05]["n_companies"] == 2
    assert by_dose[0.05]["n_random_draws"] == 2
    # slope: mean delta / dose for the positive dose.
    assert summary["stance_slopes_positive_doses"] == [{"dose": 0.05, "mean_delta_per_unit": 5.0}]
    # per-company stance deltas
    assert summary["company_stance_deltas"]["A:rev0:ord0"] == {"0.05": 0.2, "-0.05": -0.1}
    # flips at max + dose: none of the fake companies cross p>0.5.
    assert summary["flips_at_max_plus_dose"]["n_flipped"] == 0


def test_fit_stance_direction_is_unit_mean_of_pairs() -> None:
    states = {
        "P1": torch.tensor([2.0, 0.0]), "N1": torch.tensor([1.0, 0.0]),
        "P2": torch.tensor([3.0, 0.0]), "N2": torch.tensor([1.0, 0.0]),
    }
    comparisons = [
        {"kind": "evaluation_at_positive_concept", "positive_id": "P1", "negative_id": "N1"},
        {"kind": "evaluation_at_negative_concept", "positive_id": "P2", "negative_id": "N2"},
        {"kind": "primary", "positive_id": "P1", "negative_id": "N1"},  # ignored
    ]
    direction, pair_ids = _fit_stance_direction(states, comparisons)
    assert pair_ids == ["P1>N1", "P2>N2"]
    # mean diff = [(1,0) + (2,0)] / 2 = (1.5, 0); unit = (1, 0)
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-6)
    assert float(direction.norm()) == 1.0


def test_fit_stance_direction_requires_pairs() -> None:
    try:
        _fit_stance_direction({"a": torch.tensor([1.0])}, [{"kind": "primary", "positive_id": "a", "negative_id": "a"}])
    except ValueError as exc:
        assert "no evaluation pairs" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_additive_transform_only_touches_span() -> None:
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    transform = _additive_transform(direction, 0.5, start=1, end=3, device=torch.device("cpu"))
    x = torch.zeros(1, 5, 4)
    y = transform(x)
    assert torch.equal(x, torch.zeros(1, 5, 4)), "input must not be mutated"
    assert torch.all(y[:, 0, :] == 0.0) and torch.all(y[:, 3:, :] == 0.0)
    assert torch.allclose(y[:, 1:3, 0], torch.full((2,), 0.5))
    assert y.shape == x.shape
