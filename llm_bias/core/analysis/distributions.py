"""Distribution-level statistics, without experiment-specific estimands."""
from __future__ import annotations
import math
from typing import Any
import torch


def restricted_softmax(logits: torch.Tensor, token_ids: list[int] | torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(token_ids, device=logits.device, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0 or len(set(ids.tolist())) != ids.numel():
        raise ValueError("token_ids must be a non-empty unique one-dimensional sequence")
    values = logits.unsqueeze(0) if logits.ndim == 1 else logits
    if values.ndim != 2:
        raise ValueError("logits must be one- or two-dimensional")
    selected = values.index_select(-1, ids)
    probabilities = torch.softmax(selected.float(), dim=-1)
    return probabilities.squeeze(0) if logits.ndim == 1 else probabilities


def validate_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    p = probabilities.float()
    if p.ndim < 1 or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not torch.allclose(p.sum(dim=-1), torch.ones_like(p.sum(dim=-1)), rtol=1e-5, atol=1e-5):
        raise ValueError("probabilities must sum to one")
    return p


def distribution_stats(probabilities: torch.Tensor, scores: torch.Tensor | list[float] | None = None) -> dict[str, float]:
    """Compute entropy and (optionally) a restricted expected score."""
    p = validate_probabilities(probabilities)
    if p.ndim != 1:
        raise ValueError("distribution_stats expects one probability vector")
    entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    result = {"entropy_nats": float(entropy)}
    if scores is not None:
        s = torch.as_tensor(scores, device=p.device, dtype=p.dtype)
        if s.ndim != 1 or s.numel() != p.numel():
            raise ValueError("probability and score vectors must have equal length")
        result["expected_score"] = float((p * s).sum())
    return result


def full_vocabulary_stats(probabilities: torch.Tensor) -> dict[str, float]:
    """Statistics for a complete vocabulary distribution."""
    p = validate_probabilities(probabilities)
    if p.ndim != 1:
        raise ValueError("full_vocabulary_stats expects one probability vector")
    entropy = distribution_stats(p)["entropy_nats"]
    return {
        "entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(p.numel()),
        "perplexity": math.exp(entropy),
        "top1_probability": float(p.max()),
        "effective_vocabulary": math.exp(entropy),
    }


def mean_full_vocabulary(probability_vectors: torch.Tensor) -> torch.Tensor:
    """Average complete softmax vectors before any top-k selection."""
    p = validate_probabilities(probability_vectors)
    if p.ndim != 2:
        raise ValueError("probability_vectors must be [observations, vocabulary]")
    return p.double().mean(dim=0).float()


def effective_temperature(hidden: torch.Tensor, final_norm: Any | None = None) -> torch.Tensor:
    normalized = final_norm(hidden) if final_norm is not None else hidden
    inverse = normalized.float().norm(dim=-1)
    if not torch.isfinite(inverse).all() or (inverse <= 0).any():
        raise ValueError("final-normalized residual norm must be finite and positive")
    return inverse.reciprocal()
