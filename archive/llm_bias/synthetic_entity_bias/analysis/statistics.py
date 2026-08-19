"""Statistical helpers for the synthetic entity-bias analysis tables."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from scipy import stats

BOOTSTRAP_SEED = 20260812
BOOTSTRAP_RESAMPLES = 2_000
CONFIDENCE_LEVEL = 0.95


def finite_array(values: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(values), dtype=np.float64)
    if data.ndim != 1 or not len(data):
        raise ValueError("statistics require a non-empty one-dimensional sample")
    if not np.all(np.isfinite(data)):
        raise ValueError("statistics require finite values")
    return data


def descriptive_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    data = finite_array(values)
    n = len(data)
    sample_std = float(np.std(data, ddof=1)) if n > 1 else None
    q25, q75 = np.quantile(data, (0.25, 0.75), method="linear")
    return {
        "n": n,
        "mean": float(np.mean(data)),
        "sample_std": sample_std,
        "sem": sample_std / math.sqrt(n) if sample_std is not None else None,
        "median": float(np.median(data)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "negative_fraction": float(np.mean(data < 0)),
        "zero_fraction": float(np.mean(data == 0)),
        "positive_fraction": float(np.mean(data > 0)),
    }


def _seed_for(key: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_interval(
    values: Iterable[float],
    statistic: Callable[[np.ndarray], float],
    *,
    key: str,
) -> tuple[float, float]:
    data = finite_array(values)
    if len(data) == 1:
        value = float(statistic(data))
        return value, value
    rng = np.random.default_rng(_seed_for(key))
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        estimates[index] = statistic(data[rng.integers(0, len(data), len(data))])
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    lower, upper = np.quantile(estimates, (alpha, 1 - alpha), method="linear")
    return float(lower), float(upper)


def mean_median_intervals(values: Iterable[float], *, key: str) -> dict[str, float]:
    data = finite_array(values)
    mean_lower, mean_upper = bootstrap_interval(data, np.mean, key=f"{key}:mean")
    median_lower, median_upper = bootstrap_interval(data, np.median, key=f"{key}:median")
    return {
        "mean_ci95_lower": mean_lower,
        "mean_ci95_upper": mean_upper,
        "median_ci95_lower": median_lower,
        "median_ci95_upper": median_upper,
    }


def bootstrap_mean_difference(
    left: Iterable[float],
    right: Iterable[float],
    *,
    key: str,
) -> tuple[float, float]:
    a, b = finite_array(left), finite_array(right)
    rng = np.random.default_rng(_seed_for(key))
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        a_sample = a[rng.integers(0, len(a), len(a))]
        b_sample = b[rng.integers(0, len(b), len(b))]
        estimates[index] = np.mean(a_sample) - np.mean(b_sample)
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    lower, upper = np.quantile(estimates, (alpha, 1 - alpha), method="linear")
    return float(lower), float(upper)


def holm_adjust(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    valid = [(index, float(value)) for index, value in enumerate(values) if value is not None]
    if any(not 0 <= value <= 1 or not math.isfinite(value) for _, value in valid):
        raise ValueError("p-values must be finite and in [0, 1]")
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[index] = min(1.0, running)
    return [adjusted.get(index) for index in range(len(values))]


def _test(call: Callable[[], Any], *, minimum_n: int, n: int) -> dict[str, Any]:
    if n < minimum_n:
        return {"statistic": None, "p_value": None, "status": "insufficient_sample", "reason": f"requires n >= {minimum_n}"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = call()
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(statistic) or not math.isfinite(p_value):
            raise ValueError("non-finite result")
        return {"statistic": statistic, "p_value": p_value, "status": "ok", "reason": ""}
    except (ValueError, ZeroDivisionError) as exc:
        return {"statistic": None, "p_value": None, "status": "degenerate", "reason": str(exc)}


def one_sample_tests(values: Iterable[float]) -> dict[str, Any]:
    data = finite_array(values)
    std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    effect = float(np.mean(data) / std) if std > 0 else None
    t_test = _test(lambda: stats.ttest_1samp(data, 0.0), minimum_n=2, n=len(data))
    wilcoxon = _test(lambda: stats.wilcoxon(data, alternative="two-sided"), minimum_n=1, n=len(data))
    return {
        "cohens_dz": effect,
        "t_statistic": t_test["statistic"],
        "t_p_value": t_test["p_value"],
        "t_status": t_test["status"],
        "t_reason": t_test["reason"],
        "wilcoxon_statistic": wilcoxon["statistic"],
        "wilcoxon_p_value": wilcoxon["p_value"],
        "wilcoxon_status": wilcoxon["status"],
        "wilcoxon_reason": wilcoxon["reason"],
    }


def paired_tests(left: Iterable[float], right: Iterable[float]) -> dict[str, Any]:
    a, b = finite_array(left), finite_array(right)
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    differences = a - b
    tests = one_sample_tests(differences)
    pearson = _test(lambda: stats.pearsonr(a, b), minimum_n=2, n=len(a))
    spearman = _test(lambda: stats.spearmanr(a, b), minimum_n=2, n=len(a))
    tests.update({
        "pearson_r": pearson["statistic"], "pearson_p_value": pearson["p_value"],
        "pearson_status": pearson["status"], "pearson_reason": pearson["reason"],
        "spearman_r": spearman["statistic"], "spearman_p_value": spearman["p_value"],
        "spearman_status": spearman["status"], "spearman_reason": spearman["reason"],
    })
    return tests


def independent_tests(left: Iterable[float], right: Iterable[float]) -> dict[str, Any]:
    a, b = finite_array(left), finite_array(right)
    welch = _test(lambda: stats.ttest_ind(a, b, equal_var=False), minimum_n=2, n=min(len(a), len(b)))
    mann_whitney = _test(lambda: stats.mannwhitneyu(a, b, alternative="two-sided"), minimum_n=1, n=min(len(a), len(b)))
    pooled_denominator = len(a) + len(b) - 2
    effect = None
    if pooled_denominator > 0:
        pooled_variance = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / pooled_denominator
        if pooled_variance > 0:
            cohens_d = float((np.mean(a) - np.mean(b)) / math.sqrt(pooled_variance))
            correction = 1 - 3 / (4 * (len(a) + len(b)) - 9) if len(a) + len(b) > 2 else 1.0
            effect = cohens_d * correction
    return {
        "hedges_g": effect,
        "welch_t_statistic": welch["statistic"], "welch_t_p_value": welch["p_value"],
        "welch_t_status": welch["status"], "welch_t_reason": welch["reason"],
        "mann_whitney_u": mann_whitney["statistic"], "mann_whitney_p_value": mann_whitney["p_value"],
        "mann_whitney_status": mann_whitney["status"], "mann_whitney_reason": mann_whitney["reason"],
    }


def kruskal_test(groups: Iterable[Iterable[float]]) -> dict[str, Any]:
    arrays = [finite_array(group) for group in groups]
    return _test(lambda: stats.kruskal(*arrays), minimum_n=2, n=len(arrays))
