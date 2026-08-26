"""Pure residual transforms used by J-space interventions."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def _validated_positions(tensor: torch.Tensor, positions: Sequence[int]) -> list[int]:
    if tensor.ndim != 3:
        raise ValueError("residual tensor must have shape [batch, sequence, d_model]")
    unique = sorted({int(position) for position in positions})
    if not unique:
        raise ValueError("at least one intervention position is required")
    if unique[0] < 0 or unique[-1] >= tensor.shape[1]:
        raise ValueError("intervention position is outside the sequence")
    return unique


def steer_positions(
    tensor: torch.Tensor,
    *,
    positions: Sequence[int],
    direction: torch.Tensor,
    coordinate_delta: float,
) -> torch.Tensor:
    """Add a direction at selected positions without changing other entries."""
    selected = _validated_positions(tensor, positions)
    vector = direction.reshape(-1)
    if vector.numel() != tensor.shape[-1]:
        raise ValueError("direction and residual widths differ")
    if not torch.isfinite(vector).all():
        raise ValueError("direction contains non-finite values")
    patched = tensor.clone()
    delta = float(coordinate_delta) * vector.to(device=tensor.device, dtype=tensor.dtype)
    patched[:, selected, :] = patched[:, selected, :] + delta
    return patched


def coordinate_intervention(
    tensor: torch.Tensor,
    *,
    positions: Sequence[int],
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    alpha: float = 1.0,
    mode: str = "swap",
) -> torch.Tensor:
    """Edit source/target least-squares coordinates at selected positions.

    Modes are ``swap`` (exchange both coordinates), ``source_ablation`` (set
    the source coordinate to the target coordinate), and ``target_addition``
    (set the target coordinate to the source coordinate). The latter two sum
    to the full swap delta.
    """
    if mode not in {"swap", "source_ablation", "target_addition"}:
        raise ValueError(f"unknown coordinate intervention mode: {mode}")
    selected = _validated_positions(tensor, positions)
    source = source_direction.reshape(-1)
    target = target_direction.reshape(-1)
    if source.numel() != tensor.shape[-1] or target.numel() != tensor.shape[-1]:
        raise ValueError("direction and residual widths differ")
    matrix = torch.stack((source, target), dim=1).to(
        device=tensor.device, dtype=torch.float32
    )
    if not torch.isfinite(matrix).all():
        raise ValueError("directions contain non-finite values")
    if torch.linalg.matrix_rank(matrix) < 2:
        raise ValueError("source and target directions are linearly dependent")
    pinv = torch.linalg.pinv(matrix)
    values = tensor[:, selected, :].float()
    coordinates = values @ pinv.T
    if mode == "swap":
        desired = coordinates[..., [1, 0]]
    elif mode == "source_ablation":
        desired = torch.stack((coordinates[..., 1], coordinates[..., 1]), dim=-1)
    else:
        desired = torch.stack((coordinates[..., 0], coordinates[..., 0]), dim=-1)
    delta = (desired - coordinates) @ matrix.T
    patched = tensor.clone()
    patched[:, selected, :] = (
        values + float(alpha) * delta
    ).to(dtype=tensor.dtype)
    return patched


def coordinate_swap(
    tensor: torch.Tensor,
    *,
    positions: Sequence[int],
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Exchange source and target coordinates, preserving the complement."""
    return coordinate_intervention(
        tensor,
        positions=positions,
        source_direction=source_direction,
        target_direction=target_direction,
        alpha=alpha,
        mode="swap",
    )


def perturbation_norm(before: torch.Tensor, after: torch.Tensor) -> float:
    """Return the L2 norm of a compactly measured intervention delta."""
    if before.shape != after.shape:
        raise ValueError("before and after shapes differ")
    return float(torch.linalg.vector_norm((after - before).float()).detach().cpu())


__all__ = [
    "coordinate_intervention",
    "coordinate_swap",
    "perturbation_norm",
    "steer_positions",
]
