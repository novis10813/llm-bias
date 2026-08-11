"""Research-neutral restricted readout and final-position helpers."""
from __future__ import annotations
import math
from typing import Any
import torch


def last_unmasked_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
    return positions.masked_fill(attention_mask == 0, -1).max(dim=1).values


def restricted_softmax(logits: torch.Tensor, token_ids: list[int] | torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(token_ids, device=logits.device, dtype=torch.long)
    if logits.ndim == 1: logits = logits.unsqueeze(0)
    selected = logits.index_select(-1, ids)
    probs = torch.softmax(selected.float(), dim=-1)
    return probs.squeeze(0) if probs.shape[0] == 1 else probs


def distribution_stats(probabilities: torch.Tensor, scores: torch.Tensor | list[float]) -> dict[str, float]:
    p = probabilities.float()
    s = torch.as_tensor(scores, device=p.device, dtype=p.dtype)
    if p.ndim != 1 or s.ndim != 1 or p.numel() != s.numel(): raise ValueError("probability and score vectors must be one-dimensional and equal length")
    if not torch.isfinite(p).all() or (p < 0).any() or not math.isclose(float(p.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5): raise ValueError("probabilities must be finite, non-negative, and sum to one")
    entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    mean = (p * s).sum()
    centered = s - mean
    variance = (p * centered.square()).sum()
    return {"expected_score": float(mean), "entropy_nats": float(entropy), "effective_temperature": float(variance.sqrt().clamp_min(1e-12))}


def effective_temperature(hidden: torch.Tensor, final_norm: Any | None = None) -> torch.Tensor:
    normalized = final_norm(hidden) if final_norm is not None else hidden
    inverse = normalized.float().norm(dim=-1)
    return inverse.reciprocal().clamp_min(1e-12)
