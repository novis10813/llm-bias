"""Compatibility facade for direction/statistics primitives."""
from .analysis.directions import OnlineDirection, quantile_bounds, cosine_and_statistics
from .analysis.statistics import direction_hash
__all__ = ["OnlineDirection", "quantile_bounds", "cosine_and_statistics", "direction_hash"]
