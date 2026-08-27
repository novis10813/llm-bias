"""Ticker-clustered paired-effect summaries for intervention records."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from llm_bias.core.analysis.statistics import holm_bonferroni, sign_flip_pvalue


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
        if "swap_fraction" in row:
            dose_name, dose_value = "swap_fraction", row["swap_fraction"]
        elif "gain" in row:
            dose_name, dose_value = "gain", row["gain"]
        else:
            dose_name, dose_value = "alpha", row["alpha"]
        loaded = bool(row.get("loaded_positions", {}).get("loaded", True))
        no_op_value = 1.0 if dose_name == "gain" else 0.0
        if not loaded and float(dose_value) != no_op_value:
            continue
        key = (
            row["intervention_type"],
            row.get("source_prototype"),
            row.get("target_prototype"),
            row.get("prototype"),
            row.get("token"),
            row.get("direction_control"),
            row.get("position_control"),
            dose_name,
            float(dose_value),
        )
        groups[key].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        (
            intervention, source, target, prototype, token,
            direction_control, position_control, dose_name, dose_value,
        ) = key
        result.append(
            {
                "intervention_type": intervention,
                "source_prototype": source,
                "target_prototype": target,
                "prototype": prototype,
                "token": token,
                "direction_control": direction_control,
                "position_control": position_control,
                "dose_name": dose_name,
                dose_name: dose_value,
                **ticker_clustered_effect(
                    group, seed=seed, bootstrap_samples=bootstrap_samples
                ),
            }
        )
    return result


def _screen_ticker_stats(
    values_by_record: dict[str, float],
    tickers: dict[str, str],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    """Equal-ticker mean plus hierarchical bootstrap CI for one slope set.

    ``values_by_record`` maps prompt record IDs to one scalar (a per-prompt
    symmetric slope or specificity).  Prompts are averaged within ticker and
    tickers are weighted equally, reusing the intervention bootstrap contract.
    """
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for record_id, value in values_by_record.items():
        by_ticker[tickers[record_id]].append(value)
    rows = [
        {"ticker": ticker, "delta_margin": value}
        for ticker in sorted(by_ticker)
        for value in by_ticker[ticker]
    ]
    summary = ticker_clustered_effect(rows, seed=seed, bootstrap_samples=bootstrap_samples)
    mean = float(summary["mean_delta_margin"])
    ticker_means = [float(np.mean(by_ticker[ticker])) for ticker in sorted(by_ticker)]
    sign_consistent = (
        int(sum(1 for value in ticker_means if (value > 0) == (mean > 0)))
        if mean != 0.0
        else 0
    )
    return {
        "mean": mean,
        "ci95": summary["ci95"],
        "sign_flip_p": sign_flip_pvalue(
            ticker_means, seed=seed, n_resamples=bootstrap_samples
        ),
        "sign_consistent_tickers": sign_consistent,
        "ticker_count": summary["ticker_count"],
        "record_count": summary["record_count"],
    }


def analyze_token_screen(
    rows: Iterable[dict],
    *,
    positive_dose: float,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict:
    """Summarize one token screen run as per-prompt symmetric slopes.

    The estimand is ``(delta_M(+a) - delta_M(-a)) / (2a)`` with
    ``M = logP(buy) - logP(sell)`` and ``delta_M`` relative to the clean
    margin.  Specificity is the per-record token-arm slope minus the
    matched-random-arm slope.  Prompts are averaged within ticker, tickers
    are weighted equally, and a hierarchical ticker bootstrap supplies the CI.
    Sign-flip p-values are Holm-adjusted over candidates on specificity.
    """
    if not math.isfinite(positive_dose) or positive_dose <= 0.0:
        raise ValueError("positive_dose must be finite and positive")
    entries: dict[int, dict] = {}
    for row in rows:
        entry = entries.setdefault(int(row["token_id"]), {
            "token": row["candidate"],
            "representation_side": row["representation_side"],
            "mean_positive": row["mean_positive"],
            "mean_negative": row["mean_negative"],
            "band_probability_diff": row["band_probability_diff"],
            "band_smoothed_log_ratio": row["band_smoothed_log_ratio"],
            "band_js_contribution": row["band_js_contribution"],
            "doses": {},
            "tickers": {},
            "loaded": {},
            "relative_norms": [],
        })
        entry["tickers"][row["record_id"]] = row["ticker"]
        entry["loaded"][row["record_id"]] = bool(
            row.get("loaded_positions", {}).get("loaded", True)
        )
        delivered_dose = row.get("delivered_dose")
        if float(row["alpha"]) != 0.0 and isinstance(delivered_dose, dict):
            relative_norm = float(delivered_dose["relative_perturbation_norm"])
            if not math.isfinite(relative_norm):
                raise ValueError("relative perturbation norm must be finite")
            entry["relative_norms"].append(relative_norm)
        entry["doses"].setdefault(row["arm"], {}).setdefault(row["record_id"], {})[
            float(row["alpha"])
        ] = float(row["delta_margin"])
    if not entries:
        raise ValueError("no token screen rows")
    candidates = []
    holm_inputs: list[tuple[int, float]] = []
    for token_id in sorted(entries):
        entry = entries[token_id]
        slopes: dict[str, dict[str, float]] = {}
        for arm, per_record in entry["doses"].items():
            arm_slopes = {}
            for record_id, doses in per_record.items():
                if positive_dose not in doses or -positive_dose not in doses:
                    raise ValueError(
                        f"missing symmetric doses for arm {arm!r} record {record_id}"
                    )
                arm_slopes[record_id] = (
                    doses[positive_dose] - doses[-positive_dose]
                ) / (2.0 * positive_dose)
            slopes[arm] = arm_slopes
        loaded_count = sum(bool(value) for value in entry["loaded"].values())
        record_count = len(entry["loaded"])
        candidate = {
            "token": entry["token"],
            "token_id": token_id,
            "representation_side": entry["representation_side"],
            "mean_positive": entry["mean_positive"],
            "mean_negative": entry["mean_negative"],
            "band_probability_diff": entry["band_probability_diff"],
            "band_smoothed_log_ratio": entry["band_smoothed_log_ratio"],
            "band_js_contribution": entry["band_js_contribution"],
            "token_arm": _screen_ticker_stats(
                slopes["token"], entry["tickers"],
                seed=seed, bootstrap_samples=bootstrap_samples,
            ) if "token" in slopes else None,
            "matched_random_arm": _screen_ticker_stats(
                slopes["matched_random"], entry["tickers"],
                seed=seed, bootstrap_samples=bootstrap_samples,
            ) if "matched_random" in slopes else None,
            "specificity": None,
            "loading_coverage": {
                "loaded_record_count": loaded_count,
                "record_count": record_count,
                "fraction": loaded_count / record_count,
            },
            "safety": {
                "max_relative_perturbation_norm": (
                    max(entry["relative_norms"]) if entry["relative_norms"] else None
                ),
                "mean_relative_perturbation_norm": (
                    float(np.mean(entry["relative_norms"]))
                    if entry["relative_norms"] else None
                ),
            },
        }
        if "token" in slopes and "matched_random" in slopes:
            specificity = {
                record_id: slopes["token"][record_id] - slopes["matched_random"][record_id]
                for record_id in slopes["token"]
            }
            candidate["specificity"] = _screen_ticker_stats(
                specificity, entry["tickers"],
                seed=seed, bootstrap_samples=bootstrap_samples,
            )
            holm_inputs.append((len(candidates), float(candidate["specificity"]["sign_flip_p"])))
        candidates.append(candidate)
    if holm_inputs:
        adjusted = holm_bonferroni(p for _, p in holm_inputs)
        for (index, _), adjusted_p in zip(holm_inputs, adjusted, strict=True):
            candidates[index]["specificity"]["sign_flip_p_holm"] = float(adjusted_p)
    buy_shifting = []
    sell_shifting = []
    minimum_loading_fraction = 0.9
    minimum_sign_consistency = 0.7
    for candidate in candidates:
        specificity = candidate["specificity"]
        if specificity is None:
            candidate["screen_direction"] = "not_evaluated"
            continue
        lower, upper = specificity["ci95"]
        coverage_ok = candidate["loading_coverage"]["fraction"] >= minimum_loading_fraction
        consistency = (
            specificity["sign_consistent_tickers"] / specificity["ticker_count"]
            if specificity["ticker_count"] else 0.0
        )
        consistency_ok = consistency >= minimum_sign_consistency
        candidate["shortlist_gate"] = {
            "specificity_ci_excludes_zero": lower > 0.0 or upper < 0.0,
            "loading_coverage_ok": coverage_ok,
            "sign_consistency_fraction": consistency,
            "sign_consistency_ok": consistency_ok,
            "passed": False,
        }
        if lower > 0.0:
            candidate["screen_direction"] = "buy_shifting"
            if coverage_ok and consistency_ok:
                candidate["shortlist_gate"]["passed"] = True
                buy_shifting.append(candidate)
        elif upper < 0.0:
            candidate["screen_direction"] = "sell_shifting"
            if coverage_ok and consistency_ok:
                candidate["shortlist_gate"]["passed"] = True
                sell_shifting.append(candidate)
        else:
            candidate["screen_direction"] = "not_specific"

    def _shortlist(items: list[dict]) -> list[dict]:
        ranked = sorted(
            items, key=lambda item: (-abs(item["specificity"]["mean"]), item["token_id"])
        )
        return ranked[:2]

    return {
        "estimand": "symmetric_slope",
        "estimand_definition": (
            "(delta_M(+a) - delta_M(-a)) / (2a); M = logP(buy) - logP(sell); "
            "delta_M is relative to the clean margin"
        ),
        "specificity_definition": "per-record token-arm slope minus matched-random-arm slope",
        "positive_dose": float(positive_dose),
        "unit": "ticker",
        "aggregation": "equal-weight mean of per-ticker prompt-mean slopes",
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "interpretation": "exploratory_discovery",
        "shortlist_gate": {
            "specificity_ci95_excludes_zero": True,
            "minimum_loading_fraction": minimum_loading_fraction,
            "minimum_sign_consistency_fraction": minimum_sign_consistency,
            "maximum_per_direction": 2,
        },
        "candidates": candidates,
        "shortlist": {
            "buy_shifting": [item["token_id"] for item in _shortlist(buy_shifting)],
            "sell_shifting": [item["token_id"] for item in _shortlist(sell_shifting)],
        },
    }


__all__ = ["analyze_token_screen", "grouped_effects", "ticker_clustered_effect"]
