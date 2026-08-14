"""Shared representation and Jacobian analysis contract."""
from .transport import (GRADIENT_ATTRIBUTION_METHOD, JACOBIAN_TRANSPORT_METHOD, RESIDUAL_PATCH_METHOD, transport_method_metadata, transport_residual_delta)
from .distributions import (distribution_stats, effective_temperature, full_vocabulary_stats, mean_full_vocabulary, restricted_softmax, validate_probabilities)
from .records import top_k_attribution_records, top_k_token_records
from .statistics import bootstrap_mean_ci, direction_hash, holm_bonferroni, paired_bootstrap_ci, sign_flip_pvalue
from .directions import OnlineDirection, cosine_and_statistics, quantile_bounds

__all__ = ["GRADIENT_ATTRIBUTION_METHOD", "JACOBIAN_TRANSPORT_METHOD", "RESIDUAL_PATCH_METHOD", "transport_residual_delta", "distribution_stats", "effective_temperature", "full_vocabulary_stats", "mean_full_vocabulary", "restricted_softmax", "top_k_token_records", "bootstrap_mean_ci", "paired_bootstrap_ci", "sign_flip_pvalue", "holm_bonferroni", "OnlineDirection", "cosine_and_statistics", "quantile_bounds", "direction_hash"]
