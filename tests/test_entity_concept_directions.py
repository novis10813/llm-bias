"""Tests for model-free equal-family concept fitting."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from llm_bias.entity_concept_decision.concepts import concept_scores, fit_concept_direction
from llm_bias.entity_concept_decision.materials import load_material_bundle


def _bundle(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "concepts": [
            {
                "concept_id": "c1",
                "definition": "A synthetic test concept.",
                "positive_pole": "mountain",
                "negative_pole": "coast",
                "excluded_interpretations": ["general sentiment"],
            }
        ],
        "pairs": [
            {"id": "p1", "concept_id": "c1", "family_id": "f1", "split": "fit", "text_positive": "Fit one positive.", "text_negative": "Fit one negative.", "review_status": "approved", "confound_tags": []},
            {"id": "p2", "concept_id": "c1", "family_id": "f1", "split": "fit", "text_positive": "Fit two positive.", "text_negative": "Fit two negative.", "review_status": "approved", "confound_tags": []},
            {"id": "p3", "concept_id": "c1", "family_id": "f2", "split": "fit", "text_positive": "Fit three positive.", "text_negative": "Fit three negative.", "review_status": "approved", "confound_tags": []},
            {"id": "p4", "concept_id": "c1", "family_id": "v1", "split": "validation", "text_positive": "Validation positive.", "text_negative": "Validation negative.", "review_status": "approved", "confound_tags": []},
            {"id": "p5", "concept_id": "c1", "family_id": "a1", "split": "audit", "text_positive": "Audit positive.", "text_negative": "Audit negative.", "review_status": "approved", "confound_tags": []},
        ],
    }
    path = tmp_path / "materials.json"
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    return load_material_bundle(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def _fit(bundle, *, positive, negative, pair_ids=("p3", "p1", "p2"), basis=None, min_norm=1e-6):
    if basis is None:
        basis = torch.eye(4, dtype=torch.float32)[:, :2]
    return fit_concept_direction(
        bundle,
        "c1",
        positive=positive,
        negative=negative,
        pair_ids=pair_ids,
        basis=basis,
        min_norm=min_norm,
    )


def test_fit_weights_families_equally_and_sorts_pair_reduction(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    # f1 has two [2,0,0,0] rows and f2 has one [0,2,0,0] row.
    positive = torch.tensor([[0.0, 2.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    negative = torch.zeros_like(positive)
    positive_before, negative_before = positive.clone(), negative.clone()

    result = _fit(bundle, positive=positive, negative=negative)

    expected = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert result.status == "ok"
    assert torch.allclose(result.direction, expected / math.sqrt(2), atol=1e-6, rtol=0.0)
    assert torch.allclose(result.coordinates, torch.tensor([1.0 / math.sqrt(2)] * 2), atol=1e-6, rtol=0.0)
    assert result.n_pairs == 3
    assert result.n_families == 2
    assert torch.equal(positive, positive_before)
    assert torch.equal(negative, negative_before)

    reversed_result = _fit(
        bundle,
        positive=positive[[2, 0, 1]],
        negative=negative[[2, 0, 1]],
        pair_ids=("p2", "p3", "p1"),
    )
    assert torch.equal(result.direction, reversed_result.direction)


def test_label_exchange_reverses_direction(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    positive = torch.tensor([[2.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 2.0, 0.0, 0.0]])
    negative = torch.zeros_like(positive)
    forward = _fit(bundle, positive=positive, negative=negative)
    reverse = _fit(bundle, positive=negative, negative=positive)
    assert torch.allclose(reverse.direction, -forward.direction, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "pair_ids, positive, negative, message",
    [
        (("p1", "p2"), torch.ones(2, 4), torch.zeros(2, 4), "exactly match"),
        (("p1", "p1", "p3"), torch.ones(3, 4), torch.zeros(3, 4), "duplicate"),
        (("p1", "p2", "p4"), torch.ones(3, 4), torch.zeros(3, 4), "exactly match"),
        (("p1", "p2", "p3", "p5"), torch.ones(4, 4), torch.zeros(4, 4), "exactly match"),
    ],
)
def test_fit_rejects_wrong_pair_sets_and_audit_mixing(
    tmp_path: Path, pair_ids, positive, negative, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _fit(_bundle(tmp_path), positive=positive, negative=negative, pair_ids=pair_ids)


def test_fit_degenerate_source_and_projected_and_exact_threshold(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    zeros = torch.zeros(3, 4)
    source_zero = _fit(bundle, positive=zeros, negative=zeros)
    assert source_zero.status == "degenerate"
    assert source_zero.reason == "source_norm_below_minimum"
    assert source_zero.retained_fraction is None
    assert source_zero.direction is None and source_zero.coordinates is None

    values = torch.tensor([[0.0, 2.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    basis_outside = torch.eye(4, dtype=torch.float32)[:, 2:]
    projected_zero = _fit(bundle, positive=values, negative=zeros, basis=basis_outside)
    assert projected_zero.status == "degenerate"
    assert projected_zero.reason == "projected_norm_below_minimum"
    assert projected_zero.retained_fraction == pytest.approx(0.0)

    exact = _fit(bundle, positive=values, negative=zeros, min_norm=math.sqrt(2.0) + 1e-6)
    assert exact.status == "degenerate"
    assert exact.reason == "source_norm_below_minimum"


def test_fit_rejects_nonfinite_inputs_and_invalid_threshold(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        _fit(bundle, positive=torch.tensor([[float("nan"), 0.0, 0.0, 0.0]] * 3), negative=torch.zeros(3, 4))
    with pytest.raises(ValueError, match="positive"):
        _fit(bundle, positive=torch.ones(3, 4), negative=torch.zeros(2, 4))
    with pytest.raises(ValueError, match="min_norm"):
        _fit(bundle, positive=torch.ones(3, 4), negative=torch.zeros(3, 4), min_norm=0.0)


def test_concept_scores_requires_unit_direction_and_returns_fp32(tmp_path: Path) -> None:
    values = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    values_before, direction_before = values.clone(), direction.clone()
    scores = concept_scores(values, direction)
    assert scores.dtype == torch.float32
    assert torch.equal(scores, torch.tensor([1.0, -1.0]))
    assert torch.equal(values, values_before)
    assert torch.equal(direction, direction_before)
    with pytest.raises(ValueError, match="unit norm"):
        concept_scores(values, torch.tensor([2.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        concept_scores(torch.ones(2, 2, 1), direction)


def test_concept_fit_is_in_memory_and_has_no_tensor_serializer(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    values = torch.tensor([[2.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 2.0, 0.0, 0.0]])
    result = _fit(bundle, positive=values, negative=torch.zeros_like(values))
    assert isinstance(result.direction, torch.Tensor)
    assert isinstance(result.coordinates, torch.Tensor)
    assert "direction" in asdict(result)
