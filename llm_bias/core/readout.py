"""Compatibility facade for shared analysis primitives.

New code should import from :mod:`llm_bias.core.analysis`.
"""
from .analysis.distributions import distribution_stats, effective_temperature, restricted_softmax, full_vocabulary_stats, mean_full_vocabulary
from .analysis.transport import transport_residual_delta
import torch

def last_unmasked_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
    return positions.masked_fill(attention_mask == 0, -1).max(dim=1).values

__all__ = ["last_unmasked_positions", "restricted_softmax", "distribution_stats", "effective_temperature", "full_vocabulary_stats", "mean_full_vocabulary", "transport_residual_delta"]
