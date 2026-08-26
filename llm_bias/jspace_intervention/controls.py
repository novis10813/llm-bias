"""Deterministic matched controls for J-space interventions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _torch_seed(seed: int) -> int:
    return int(seed) % (2**63 - 1)


def shuffled_evidence_positions(
    evidence_span: tuple[int, int],
    selected: Sequence[int],
    *,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    """Sample matched evidence positions excluding the selected positions."""
    start, end = evidence_span
    candidates = [index for index in range(start, end) if index not in set(selected)]
    if len(candidates) < count:
        raise ValueError("evidence span has too few positions for a disjoint control")
    generator = torch.Generator(device="cpu").manual_seed(_torch_seed(seed))
    order = torch.randperm(len(candidates), generator=generator)[:count].tolist()
    return tuple(sorted(candidates[index] for index in order))


def matched_random_direction(direction: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return a seeded isotropic direction with the same Euclidean norm."""
    vector = direction.detach().float().cpu()
    generator = torch.Generator(device="cpu").manual_seed(_torch_seed(seed))
    random = torch.randn(vector.shape, generator=generator)
    random = random / random.norm().clamp_min(1e-12)
    return random * vector.norm()


def matched_random_direction_pair(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random pair whose 2×2 Gram matrix matches the input pair."""
    source = source.detach().float().cpu()
    target = target.detach().float().cpu()
    matrix = torch.stack((source, target), dim=-1)
    gram = matrix.T @ matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    if float(eigenvalues.min()) <= 1e-8:
        raise ValueError("source/target directions are too collinear for a matched control")
    gram_root = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
    generator = torch.Generator(device="cpu").manual_seed(_torch_seed(seed))
    random = torch.randn(matrix.shape, generator=generator)
    basis, _ = torch.linalg.qr(random, mode="reduced")
    matched = basis @ gram_root
    return matched[:, 0], matched[:, 1]


def norm_match_intervention(
    original: torch.Tensor,
    control_patched: torch.Tensor,
    reference_patched: torch.Tensor,
    *,
    control_positions: Sequence[int],
    reference_positions: Sequence[int],
) -> torch.Tensor:
    """Rescale a control delta to the local norm of the primary delta."""
    control_indices = list(control_positions)
    reference_indices = list(reference_positions)
    control_delta = (
        control_patched[:, control_indices, :].float()
        - original[:, control_indices, :].float()
    )
    reference_delta = (
        reference_patched[:, reference_indices, :].float()
        - original[:, reference_indices, :].float()
    )
    control_norm = control_delta.norm()
    reference_norm = reference_delta.norm()
    if float(reference_norm) == 0.0:
        return original
    if float(control_norm) <= 1e-12:
        raise ValueError("control intervention has zero norm and cannot be dose matched")
    matched = original.clone()
    matched[:, control_indices, :] = (
        original[:, control_indices, :].float()
        + control_delta * (reference_norm / control_norm)
    ).to(original.dtype)
    return matched


def matched_random_prototypes(
    source: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    random_source = {}
    random_target = {}
    for offset, layer in enumerate(sorted(source)):
        src, tgt = matched_random_direction_pair(
            source[layer], target[layer], seed=seed + 1009 * offset
        )
        random_source[layer] = src
        random_target[layer] = tgt
    return random_source, random_target


__all__ = [
    "matched_random_direction",
    "matched_random_direction_pair",
    "matched_random_prototypes",
    "norm_match_intervention",
    "shuffled_evidence_positions",
]
