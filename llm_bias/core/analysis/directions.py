"""Generic high-minus-low direction fitting primitives."""
from __future__ import annotations
import hashlib, math
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch
from .statistics import cosine_statistics, direction_hash

@dataclass
class OnlineDirection:
    high_sum: torch.Tensor
    low_sum: torch.Tensor
    n_high: int = 0
    n_low: int = 0
    def add(self, vector: torch.Tensor, *, high: bool) -> None:
        value = vector.detach().float().flatten().cpu()
        if value.shape != self.high_sum.shape: raise ValueError("direction vector width mismatch")
        if high: self.high_sum += value; self.n_high += 1
        else: self.low_sum += value; self.n_low += 1
    def normalized(self) -> tuple[torch.Tensor, float]:
        if not self.n_high or not self.n_low: raise ValueError("both high and low groups require observations")
        direction = self.high_sum / self.n_high - self.low_sum / self.n_low
        norm = float(direction.norm())
        if not math.isfinite(norm) or norm <= 1e-8: raise ValueError("high-minus-low direction has near-zero norm")
        return direction / norm, norm

def quantile_bounds(values: Iterable[float], low: float=.25, high: float=.75) -> tuple[float,float]:
    data = np.asarray(list(values), dtype=float)
    if data.size < 2 or not np.isfinite(data).all(): raise ValueError("quantile data must contain at least two finite values")
    if not 0 <= low < high <= 1: raise ValueError("invalid quantiles")
    low_value, high_value = float(np.quantile(data, low)), float(np.quantile(data, high))
    if not low_value < high_value: raise ValueError("q25 and q75 must be strictly ordered")
    return low_value, high_value

def cosine_and_statistics(cosines: Iterable[float], targets: Iterable[float]): return cosine_statistics(cosines, targets)
