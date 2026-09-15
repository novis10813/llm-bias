"""Model-free concept-direction fitting over reviewed material pairs."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch

from llm_bias.core.analysis.subspaces import project_onto_basis
from llm_bias.entity_concept_decision.materials import MaterialBundle


@dataclass(frozen=True)
class ConceptFit:
    concept_id: str
    status: str
    reason: str | None
    direction: torch.Tensor | None
    coordinates: torch.Tensor | None
    source_norm: float
    projected_norm: float
    retained_fraction: float | None
    n_pairs: int
    n_families: int
    materials_sha256: str


def _cpu_floating(value: object, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU floating tensor")
    if not torch.is_floating_point(value) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense CPU floating tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_pair_ids(pair_ids: Sequence[str], *, expected: set[str]) -> dict[str, int]:
    if isinstance(pair_ids, (str, bytes)) or not isinstance(pair_ids, Sequence):
        raise ValueError("pair_ids must be a sequence of unique strings")
    result: dict[str, int] = {}
    for index, pair_id in enumerate(pair_ids):
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"pair_ids[{index}] must be a non-empty string")
        if pair_id in result:
            raise ValueError(f"pair_ids contains duplicate id: {pair_id}")
        result[pair_id] = index
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(f"pair_ids must exactly match fit rows; missing={missing}, extra={extra}")
    return result


def _validate_inputs(
    bundle: MaterialBundle,
    concept_id: str,
    positive: torch.Tensor,
    negative: torch.Tensor,
    pair_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]:
    if not isinstance(bundle, MaterialBundle):
        raise ValueError("bundle must be a MaterialBundle")
    fit_pairs = [
        pair for pair in bundle.pairs if pair.concept_id == concept_id and pair.split == "fit"
    ]
    if not fit_pairs:
        raise ValueError(f"concept {concept_id!r} has no fit rows")
    expected_ids = {pair.id for pair in fit_pairs}
    positions = _validate_pair_ids(pair_ids, expected=expected_ids)
    positive = _cpu_floating(positive, name="positive")
    negative = _cpu_floating(negative, name="negative")
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative must have shape [n, d]")
    if positive.shape != negative.shape or positive.shape[0] == 0:
        raise ValueError("positive and negative must have the same non-empty shape")
    if positive.shape[0] != len(pair_ids):
        raise ValueError("pair_ids length must match positive and negative row count")
    ordered = sorted(fit_pairs, key=lambda pair: pair.id)
    reordered_positive = torch.stack(
        [positive[positions[pair.id]] for pair in ordered], dim=0
    )
    reordered_negative = torch.stack(
        [negative[positions[pair.id]] for pair in ordered], dim=0
    )
    return reordered_positive, reordered_negative, [(pair.id, pair.family_id) for pair in ordered]


def _finite_norm(value: torch.Tensor, *, name: str) -> float:
    norm = float(torch.linalg.vector_norm(value).item())
    if not math.isfinite(norm):
        raise ValueError(f"{name} is non-finite")
    return norm


def fit_concept_direction(
    bundle: MaterialBundle,
    concept_id: str,
    *,
    positive: torch.Tensor,
    negative: torch.Tensor,
    pair_ids: Sequence[str],
    basis: torch.Tensor,
    min_norm: float,
) -> ConceptFit:
    """Fit one equal-family-weighted concept direction in a supplied basis."""
    if not isinstance(concept_id, str) or not concept_id:
        raise ValueError("concept_id must be a non-empty string")
    if isinstance(min_norm, bool) or not isinstance(min_norm, Real):
        raise ValueError("min_norm must be a finite positive real number")
    min_norm = float(min_norm)
    if not math.isfinite(min_norm) or min_norm <= 0.0:
        raise ValueError("min_norm must be a finite positive real number")

    positive, negative, ordered_rows = _validate_inputs(
        bundle, concept_id, positive, negative, pair_ids
    )
    differences = positive.to(dtype=torch.float32) - negative.to(dtype=torch.float32)
    if not torch.isfinite(differences).all():
        raise ValueError("positive-negative differences are non-finite")

    by_family: dict[str, list[torch.Tensor]] = {}
    for difference, (_, family_id) in zip(differences, ordered_rows, strict=True):
        by_family.setdefault(family_id, []).append(difference)
    family_means = [
        torch.stack(by_family[family_id], dim=0).mean(dim=0)
        for family_id in sorted(by_family)
    ]
    source = torch.stack(family_means, dim=0).mean(dim=0)
    if not torch.isfinite(source).all():
        raise ValueError("source direction is non-finite")
    source_norm = _finite_norm(source, name="source norm")
    projected = project_onto_basis(source, basis)
    projected_norm = _finite_norm(projected, name="projected norm")
    retained_fraction = (
        None if source_norm == 0.0 else float(projected_norm / source_norm)
    )
    if retained_fraction is not None and not math.isfinite(retained_fraction):
        raise ValueError("retained fraction is non-finite")

    reason: str | None = None
    if source_norm <= min_norm:
        reason = "source_norm_below_minimum"
    elif projected_norm <= min_norm:
        reason = "projected_norm_below_minimum"
    if reason is not None:
        return ConceptFit(
            concept_id=concept_id,
            status="degenerate",
            reason=reason,
            direction=None,
            coordinates=None,
            source_norm=source_norm,
            projected_norm=projected_norm,
            retained_fraction=retained_fraction,
            n_pairs=len(ordered_rows),
            n_families=len(by_family),
            materials_sha256=bundle.source_sha256,
        )

    direction = (projected / projected_norm).contiguous()
    if not torch.isfinite(direction).all():
        raise ValueError("normalized direction is non-finite")
    coordinates = (basis.to(dtype=torch.float32).T @ direction).contiguous()
    if not torch.isfinite(coordinates).all():
        raise ValueError("direction coordinates are non-finite")
    return ConceptFit(
        concept_id=concept_id,
        status="ok",
        reason=None,
        direction=direction,
        coordinates=coordinates,
        source_norm=source_norm,
        projected_norm=projected_norm,
        retained_fraction=retained_fraction,
        n_pairs=len(ordered_rows),
        n_families=len(by_family),
        materials_sha256=bundle.source_sha256,
    )


def concept_scores(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Score rows by a supplied unit direction without centering or thresholds."""
    values = _cpu_floating(values, name="values")
    direction = _cpu_floating(direction, name="direction")
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must have shape [n, d] with n > 0")
    if direction.ndim != 1 or direction.shape[0] != values.shape[1]:
        raise ValueError("direction must have shape [d] matching values")
    norm = float(torch.linalg.vector_norm(direction.to(dtype=torch.float64)).item())
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("direction must have unit norm within FP64 tolerance")
    scores = values.to(dtype=torch.float32) @ direction.to(dtype=torch.float32)
    if not torch.isfinite(scores).all():
        raise ValueError("scores are non-finite")
    return scores.contiguous()


__all__ = ["ConceptFit", "concept_scores", "fit_concept_direction"]
