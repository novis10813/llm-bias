"""Research-neutral online high-minus-low direction and correlations."""
from __future__ import annotations
import hashlib, math
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch

@dataclass
class OnlineDirection:
    high_sum: torch.Tensor
    low_sum: torch.Tensor
    n_high: int = 0
    n_low: int = 0
    def add(self, vector: torch.Tensor, *, high: bool) -> None:
        value = vector.detach().float().flatten().cpu()
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

def cosine_and_statistics(cosines: Iterable[float], targets: Iterable[float]) -> dict[str,float|int|None]:
    x, y = np.asarray(list(cosines),float), np.asarray(list(targets),float)
    if x.size != y.size or x.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError("statistics require equal non-empty finite arrays")
    result: dict[str,float|int|None] = {"n_eval": int(x.size), "mean_cosine": float(x.mean())}
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        result.update(pearson_r=None, spearman_r=None, linear_r2=None, statistic_flag="degenerate")
        return result
    from scipy.stats import pearsonr, spearmanr
    result["pearson_r"] = float(pearsonr(x,y).statistic); result["spearman_r"] = float(spearmanr(x,y).statistic)
    pred = np.polyval(np.polyfit(x,y,1), x)
    result["linear_r2"] = float(1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)); result["statistic_flag"] = "ok"
    return result

def direction_hash(direction: torch.Tensor) -> str:
    return hashlib.sha256(direction.detach().float().cpu().numpy().tobytes()).hexdigest()
