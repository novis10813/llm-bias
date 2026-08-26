"""Clean-pass source-loading position selection."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch


@dataclass(frozen=True)
class LoadedPositions:
    positions: tuple[int, ...]
    scores: tuple[float, ...]
    threshold: float
    loaded: bool

    def to_dict(self) -> dict:
        return {
            "positions": list(self.positions),
            "scores": list(self.scores),
            "threshold": self.threshold,
            "loaded": self.loaded,
        }


def select_loaded_positions(
    residuals: Mapping[int, torch.Tensor],
    directions: Mapping[int, torch.Tensor],
    *,
    evidence_span: tuple[int, int],
    top_k: int = 3,
    threshold: float = 0.0,
) -> LoadedPositions:
    """Select evidence positions by median cosine loading over layers.

    Residuals must contain one prompt with shape ``[1, sequence, d_model]``.
    Selection uses clean activations only and does not inspect model outputs.
    """
    if set(residuals) != set(directions) or not residuals:
        raise ValueError("residuals and directions must contain identical layers")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    start, end = evidence_span
    first = residuals[next(iter(residuals))]
    if first.ndim != 3 or first.shape[0] != 1:
        raise ValueError("position selection requires residuals shaped [1, sequence, d_model]")
    if start < 0 or end <= start or end > first.shape[1]:
        raise ValueError("invalid evidence span")

    per_layer = []
    for layer in sorted(residuals):
        values = residuals[layer][0, start:end, :].float()
        direction = directions[layer].reshape(-1).to(values.device, torch.float32)
        if values.shape[-1] != direction.numel():
            raise ValueError("residual and direction widths differ")
        direction_norm = torch.linalg.vector_norm(direction)
        value_norms = torch.linalg.vector_norm(values, dim=-1)
        denominator = value_norms * direction_norm
        cosine = torch.where(
            denominator > 0,
            values @ direction / denominator,
            torch.zeros_like(denominator),
        )
        per_layer.append(cosine)
    median = torch.stack(per_layer).median(dim=0).values
    count = min(top_k, median.numel())
    values, local_positions = torch.topk(median, k=count)
    absolute_positions = local_positions + start
    pairs = sorted(
        (
            (int(position.detach().cpu()), float(value.detach().cpu()))
            for position, value in zip(absolute_positions, values, strict=True)
        ),
        key=lambda item: item[0],
    )
    loaded = bool(pairs and max(score for _, score in pairs) >= threshold)
    return LoadedPositions(
        positions=tuple(position for position, _ in pairs),
        scores=tuple(score for _, score in pairs),
        threshold=float(threshold),
        loaded=loaded,
    )


__all__ = ["LoadedPositions", "select_loaded_positions"]
