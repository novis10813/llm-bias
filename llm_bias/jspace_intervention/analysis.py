"""Ticker-clustered paired-effect summaries for intervention records."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def ticker_clustered_effect(
    rows: Iterable[dict],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict:
    """Estimate the equal-ticker mean and hierarchical bootstrap interval."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ticker = str(row.get("ticker") or "")
        if not ticker:
            raise ValueError("every intervention row must contain a ticker")
        value = float(row["delta_margin"])
        if not np.isfinite(value):
            raise ValueError("delta_margin must be finite")
        by_ticker[ticker].append(value)
    if not by_ticker:
        raise ValueError("no intervention rows")
    tickers = sorted(by_ticker)
    ticker_means = np.array([np.mean(by_ticker[ticker]) for ticker in tickers])
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        selected_tickers = rng.choice(tickers, size=len(tickers), replace=True)
        selected_means = []
        for ticker in selected_tickers:
            values = np.asarray(by_ticker[str(ticker)], dtype=float)
            resampled = rng.choice(values, size=len(values), replace=True)
            selected_means.append(float(np.mean(resampled)))
        samples[index] = np.mean(selected_means)
    return {
        "mean_delta_margin": float(ticker_means.mean()),
        "median_ticker_delta_margin": float(np.median(ticker_means)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "ticker_count": len(tickers),
        "record_count": sum(len(values) for values in by_ticker.values()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "unit": "ticker",
    }


def grouped_effects(
    rows: Iterable[dict], *, seed: int = 0, bootstrap_samples: int = 2000
) -> list[dict]:
    """Summarize each intervention/source/target/dose condition."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["intervention_type"],
            row.get("source_prototype"),
            row.get("target_prototype"),
            row.get("token"),
            float(row["alpha"]),
        )
        groups[key].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        intervention, source, target, token, alpha = key
        result.append(
            {
                "intervention_type": intervention,
                "source_prototype": source,
                "target_prototype": target,
                "token": token,
                "alpha": alpha,
                **ticker_clustered_effect(
                    group, seed=seed, bootstrap_samples=bootstrap_samples
                ),
            }
        )
    return result


__all__ = ["grouped_effects", "ticker_clustered_effect"]
