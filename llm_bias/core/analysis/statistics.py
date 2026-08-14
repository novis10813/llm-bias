"""Deterministic, bounded-memory statistical primitives."""
from __future__ import annotations
import hashlib, math
from typing import Iterable
import numpy as np
import torch


def bootstrap_mean_ci(values: Iterable[float], *, seed: int = 0, n_resamples: int = 10_000, confidence: float = .95) -> dict[str, float | int | list[float] | None]:
    data = torch.as_tensor(list(values), dtype=torch.float64)
    if data.numel() == 0:
        return {"count": 0, "mean": None, "bootstrap_ci": [None, None]}
    if n_resamples < 1 or not 0 < confidence < 1 or not torch.isfinite(data).all():
        raise ValueError("invalid bootstrap inputs")
    g = torch.Generator().manual_seed(seed)
    means = data[torch.randint(data.numel(), (n_resamples, data.numel()), generator=g)].mean(1)
    alpha = (1 - confidence) / 2
    bounds = torch.quantile(means, torch.tensor([alpha, 1-alpha], dtype=means.dtype))
    return {"count": int(data.numel()), "mean": float(data.mean()), "bootstrap_ci": [float(bounds[0]), float(bounds[1])]}


def paired_bootstrap_ci(differences: Iterable[float], *, seed: int = 0, n_resamples: int = 10_000, confidence: float = .95) -> list[float] | list[None]:
    result = bootstrap_mean_ci(differences, seed=seed, n_resamples=n_resamples, confidence=confidence)
    return result["bootstrap_ci"]  # type: ignore[return-value]


def sign_flip_pvalue(differences: Iterable[float], *, seed: int = 0, n_resamples: int = 10_000, alternative: str = "two-sided") -> float | None:
    data = torch.as_tensor(list(differences), dtype=torch.float64)
    if data.numel() == 0: return None
    if n_resamples < 1 or not torch.isfinite(data).all() or alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("invalid sign-flip inputs")
    g = torch.Generator().manual_seed(seed)
    null = (data * (torch.randint(0, 2, (n_resamples, data.numel()), generator=g) * 2 - 1)).mean(1)
    observed = data.mean()
    if alternative == "greater": count = (null >= observed).sum()
    elif alternative == "less": count = (null <= observed).sum()
    else: count = (null.abs() >= observed.abs()).sum()
    return float((int(count) + 1) / (n_resamples + 1))


def holm_bonferroni(p_values: Iterable[float]) -> list[float]:
    values = [float(p) for p in p_values]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values): raise ValueError("p-values must be finite in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def cosine_statistics(cosines: Iterable[float], targets: Iterable[float]) -> dict[str, float | int | None]:
    x, y = np.asarray(list(cosines), float), np.asarray(list(targets), float)
    if x.size != y.size or x.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError("statistics require equal non-empty finite arrays")
    result: dict[str, float | int | None] = {"n_eval": int(x.size), "mean_cosine": float(x.mean())}
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        result.update(pearson_r=None, spearman_r=None, linear_r2=None, statistic_flag="degenerate")
        return result
    from scipy.stats import pearsonr, spearmanr
    result.update(pearson_r=float(pearsonr(x, y).statistic), spearman_r=float(spearmanr(x, y).statistic))
    pred = np.polyval(np.polyfit(x, y, 1), x)
    result.update(linear_r2=float(1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)), statistic_flag="ok")
    return result


# Descriptive aliases used by callers that spell out the correction/test.
holm_correction = holm_bonferroni
bootstrap_ci = bootstrap_mean_ci
paired_sign_flip_test = sign_flip_pvalue


def direction_hash(direction: torch.Tensor) -> str:
    return hashlib.sha256(direction.detach().float().cpu().numpy().tobytes()).hexdigest()
