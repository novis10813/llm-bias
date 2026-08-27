"""Ticker-clustered paired summaries for header-span sensitivity."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from llm_bias.core.analysis.statistics import holm_bonferroni, sign_flip_pvalue
from llm_bias.span_sensitivity.conditions import CONDITIONS


def _ticker_bootstrap(
    by_ticker: dict[str, list[float]], *, seed: int, samples: int
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    tickers = sorted(by_ticker)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(tickers, size=len(tickers), replace=True)
        estimates[index] = np.mean(
            [np.mean(by_ticker[str(ticker)]) for ticker in selected]
        )
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def grouped_effects(
    rows: Iterable[dict], *, seed: int = 0, bootstrap_samples: int = 2000
) -> list[dict]:
    """Summarize condition-minus-original margins with ticker as the unit."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    by_prompt_conditions: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        condition = str(row.get("condition") or "")
        if condition not in CONDITIONS:
            raise ValueError(f"unknown span condition: {condition!r}")
        key = (str(row.get("prompt_id") or ""), condition)
        if not key[0] or key in seen:
            raise ValueError(f"missing or duplicate prompt/condition row: {key}")
        seen.add(key)
        by_prompt_conditions[key[0]].add(condition)
        grouped[condition].append(row)
    incomplete = {
        prompt_id: sorted(set(CONDITIONS) - conditions)
        for prompt_id, conditions in by_prompt_conditions.items()
        if conditions != set(CONDITIONS)
    }
    if incomplete:
        first = next(iter(sorted(incomplete.items())))
        raise ValueError(f"incomplete span conditions for {first[0]}: {first[1]}")
    missing = set(CONDITIONS) - set(grouped)
    if missing:
        raise ValueError(f"missing span conditions: {sorted(missing)}")

    summaries = []
    raw_p_values = []
    for condition in CONDITIONS[1:]:
        condition_rows = grouped[condition]
        by_ticker: dict[str, list[float]] = defaultdict(list)
        flips = 0
        nonzero = 0
        for row in condition_rows:
            ticker = str(row.get("ticker") or "")
            if not ticker:
                raise ValueError("every span-sensitivity row must contain a ticker")
            delta = float(row["delta_margin"])
            margin = float(row["margin"])
            original = float(row["original_margin"])
            if not np.isfinite([delta, margin, original]).all():
                raise ValueError("margin values must be finite")
            by_ticker[ticker].append(delta)
            nonzero += delta != 0.0
            flips += (original > 0) != (margin > 0)
        ticker_means = [float(np.mean(by_ticker[ticker])) for ticker in sorted(by_ticker)]
        ci = _ticker_bootstrap(
            by_ticker, seed=seed, samples=bootstrap_samples
        )
        p_value = sign_flip_pvalue(
            ticker_means, seed=seed, n_resamples=bootstrap_samples
        )
        assert p_value is not None
        raw_p_values.append(p_value)
        summaries.append(
            {
                "condition": condition,
                "mean_delta_margin": float(np.mean(ticker_means)),
                "median_ticker_delta_margin": float(np.median(ticker_means)),
                "ci95": [ci[0], ci[1]],
                "ticker_count": len(ticker_means),
                "prompt_count": len(condition_rows),
                "nonzero_prompt_count": nonzero,
                "margin_sign_flip_count": flips,
                "sign_flip_pvalue": p_value,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": seed,
                "unit": "ticker",
            }
        )
    adjusted = holm_bonferroni(raw_p_values)
    for summary, value in zip(summaries, adjusted, strict=True):
        summary["holm_adjusted_pvalue"] = value
    return summaries


__all__ = ["grouped_effects"]
