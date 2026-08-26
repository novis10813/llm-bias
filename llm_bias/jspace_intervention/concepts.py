"""J-lens token directions and weighted sector prototypes."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch


def unembedding_weight(model: Any) -> torch.Tensor:
    """Return the frozen vocabulary-by-residual unembedding weight."""
    head = getattr(model, "_lm_head", None)
    if head is not None and hasattr(head, "weight"):
        return head.weight
    embedding = getattr(model, "_embed_tokens", None)
    if embedding is not None and hasattr(embedding, "weight"):
        return embedding.weight
    raise TypeError("model does not expose an unembedding or tied embedding weight")


def token_direction(
    unembedding: torch.Tensor,
    jacobian: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    """Return row ``token_id`` of ``W_U J_l`` in residual coordinates."""
    if unembedding.ndim != 2 or jacobian.ndim != 2:
        raise ValueError("unembedding and jacobian must be matrices")
    if jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError("jacobian must be square")
    if unembedding.shape[1] != jacobian.shape[0]:
        raise ValueError("unembedding and jacobian residual widths differ")
    if token_id < 0 or token_id >= unembedding.shape[0]:
        raise ValueError(f"token_id {token_id} is out of range")
    weight = unembedding[token_id].to(device=jacobian.device, dtype=jacobian.dtype)
    return weight @ jacobian


def concept_coordinate(residual: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Read the least-squares scalar coordinate along one direction."""
    if residual.shape[-1] != direction.numel():
        raise ValueError("residual and direction widths differ")
    vector = direction.reshape(-1).to(device=residual.device, dtype=residual.dtype)
    denominator = torch.dot(vector, vector)
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("direction must have positive finite norm")
    return residual @ vector / denominator


def sector_prototype(
    directions: Mapping[int, torch.Tensor],
    weights: Mapping[int, float],
) -> torch.Tensor:
    """Create a unit weighted mean of unit token directions."""
    if not directions:
        raise ValueError("at least one token direction is required")
    if set(directions) != set(weights):
        raise ValueError("directions and weights must have identical token IDs")
    pieces = []
    for token_id in sorted(directions):
        weight = float(weights[token_id])
        if weight < 0:
            raise ValueError("prototype weights must be nonnegative")
        vector = directions[token_id]
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("prototype direction must have positive finite norm")
        pieces.append(weight * vector / norm)
    prototype = torch.stack(pieces).sum(0)
    norm = torch.linalg.vector_norm(prototype)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("weighted prototype has zero or non-finite norm")
    return prototype / norm


def concept_spec_hash(spec: Mapping[str, Any]) -> str:
    """Stable provenance hash without persisting direction tensors."""
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "concept_coordinate",
    "concept_spec_hash",
    "sector_prototype",
    "token_direction",
    "unembedding_weight",
]
