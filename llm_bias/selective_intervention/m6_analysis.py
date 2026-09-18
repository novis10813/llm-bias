"""M6 external-population metrics and frozen interpretations.

Protocol: docs/selective-intervention/details/proposal-m6.md.
All primary statistics operate on one mean margin per company; prompt-level
records are retained only for compact secondary diagnostics.
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .analysis import iqr, per_ticker_margins
from .template import GATE_ANON_SHIFT, GATE_MEAN_SHIFT, GATE_SPECIFICITY_FRACTION

M6_SPREAD_FRACTION = 0.5
M6_BOOTSTRAP_SAMPLES = 10_000
M6_BOOTSTRAP_SEED = 42
M6_SECTORS = (
    "Health Care",
    "Information Technology",
    "Financials",
    "Industrials",
)
M6_COMPANIES_PER_SECTOR = 3
M6_EXTERNAL_COMPANIES = 12


def validate_external_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen 12-company selection manifest."""
    if manifest.get("schema_version") != "selective-intervention-m6-external-v1":
        raise ValueError("external manifest schema_version is not M6 v1")
    if int(manifest.get("selection_seed", -1)) != M6_BOOTSTRAP_SEED:
        raise ValueError("M6 selection seed must be the frozen value 42")
    companies = manifest.get("companies")
    if not isinstance(companies, list) or len(companies) != M6_EXTERNAL_COMPANIES:
        raise ValueError("M6 external manifest must contain exactly 12 companies")
    seen: set[str] = set()
    counts = {sector: 0 for sector in M6_SECTORS}
    normalized: list[dict[str, str]] = []
    for item in companies:
        if not isinstance(item, Mapping):
            raise ValueError("M6 company entries must be objects")
        ticker = str(item.get("ticker", "")).strip()
        name = str(item.get("name", "")).strip()
        sector = str(item.get("sector", "")).strip()
        if not ticker or not name or sector not in counts:
            raise ValueError(f"invalid M6 company entry: {item!r}")
        if ticker in seen:
            raise ValueError(f"duplicate M6 ticker: {ticker}")
        seen.add(ticker)
        counts[sector] += 1
        normalized.append({"ticker": ticker, "name": name, "sector": sector})
    if counts != {sector: M6_COMPANIES_PER_SECTOR for sector in M6_SECTORS}:
        raise ValueError(f"M6 sector counts must be 3 each, got {counts}")
    eligible = manifest.get("eligible_pool")
    if not isinstance(eligible, list):
        raise ValueError("M6 manifest must record eligible_pool")
    eligible_tickers = {
        str(item.get("ticker", "")).strip()
        for item in eligible
        if isinstance(item, Mapping)
    }
    if not seen.issubset(eligible_tickers):
        raise ValueError("final M6 companies must be members of eligible_pool")
    exclusions = manifest.get("exclusion_list")
    if not isinstance(exclusions, list):
        raise ValueError("M6 manifest must record exclusion_list")
    if seen & {str(value).strip() for value in exclusions}:
        raise ValueError("final M6 companies overlap exclusion_list")
    return {
        "schema_version": manifest["schema_version"],
        "selection_seed": M6_BOOTSTRAP_SEED,
        "companies": normalized,
        "eligible_pool": eligible,
        "exclusion_list": [str(value) for value in exclusions],
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("quantile requires non-empty values and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def spread_ratio(clean: Mapping[str, float], intervention: Mapping[str, float]) -> dict[str, float]:
    """Return clean/intervention IQRs and the external spread ratio."""
    if set(clean) != set(intervention):
        raise ValueError("clean and intervention company keys differ")
    if len(clean) != M6_EXTERNAL_COMPANIES:
        raise ValueError(f"M6 spread requires {M6_EXTERNAL_COMPANIES} companies")
    s_clean = iqr(list(clean.values()))
    s_intervention = iqr(list(intervention.values()))
    if not math.isfinite(s_clean) or s_clean <= 0:
        raise ValueError("M6 clean spread must be finite and positive")
    if not math.isfinite(s_intervention) or s_intervention < 0:
        raise ValueError("M6 intervention spread must be finite and non-negative")
    return {
        "clean": s_clean,
        "intervention": s_intervention,
        "ratio": s_intervention / s_clean,
        "reduction": 1.0 - s_intervention / s_clean,
    }


def bootstrap_spread_ratio(
    clean: Mapping[str, float],
    intervention: Mapping[str, float],
    *,
    samples: int = M6_BOOTSTRAP_SAMPLES,
    seed: int = M6_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired company bootstrap percentile CI for the IQR spread ratio."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    point = spread_ratio(clean, intervention)
    keys = tuple(sorted(clean))
    rng = random.Random(int(seed))
    ratios: list[float] = []
    for _ in range(samples):
        draw = [keys[rng.randrange(len(keys))] for _ in keys]
        clean_iqr = iqr([clean[key] for key in draw])
        intervention_iqr = iqr([intervention[key] for key in draw])
        if clean_iqr <= 0 or not math.isfinite(clean_iqr):
            continue
        ratio = intervention_iqr / clean_iqr
        if math.isfinite(ratio):
            ratios.append(ratio)
    if len(ratios) < max(100, samples // 2):
        raise ValueError("too few finite bootstrap spread ratios")
    return {
        **point,
        "bootstrap": {
            "method": "paired_company_percentile",
            "confidence": 0.95,
            "samples": int(samples),
            "finite_samples": len(ratios),
            "seed": int(seed),
            "ci": [_quantile(ratios, 0.025), _quantile(ratios, 0.975)],
        },
    }


def interpret_spread_ratio(result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply M6's pre-registered point/CI three-level interpretation."""
    ratio = float(result["ratio"])
    ci = result.get("bootstrap", {}).get("ci")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("spread result is missing a 95% bootstrap CI")
    upper = float(ci[1])
    if ratio <= M6_SPREAD_FRACTION and upper <= M6_SPREAD_FRACTION:
        status = "confirmed"
    elif ratio <= M6_SPREAD_FRACTION:
        status = "suggestive_unresolved"
    else:
        status = "fail"
    return {
        "status": status,
        "point_pass": ratio <= M6_SPREAD_FRACTION,
        "ci_confirmation": upper <= M6_SPREAD_FRACTION,
        "point_limit": M6_SPREAD_FRACTION,
        "ci_upper": upper,
    }


def _stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_company: dict[str, list[float]] = {}
    all_margins: list[float] = []
    for row in records:
        ticker = str(row["ticker"])
        margin = float(row["margin"])
        if not math.isfinite(margin):
            raise ValueError("M6 records contain a non-finite margin")
        by_company.setdefault(ticker, []).append(margin)
        all_margins.append(margin)
    return {
        "per_company": per_ticker_margins(by_company),
        "all_margins": all_margins,
    }


def _shift_gate(name: str, value: float, limit: float) -> dict[str, Any]:
    return {"gate": name, "value": abs(value), "limit": limit, "pass": abs(value) <= limit}


def evaluate_m6(
    clean_records: Sequence[Mapping[str, Any]],
    main_records: Sequence[Mapping[str, Any]],
    random_records: Sequence[Mapping[str, Any]],
    *,
    anon_clean: float,
    anon_intervention: float,
    bootstrap_samples: int = M6_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = M6_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute M6 spread, specificity, and secondary safety diagnostics."""
    clean = _stats(clean_records)
    main = _stats(main_records)
    random = _stats(random_records)
    spread = bootstrap_spread_ratio(
        clean["per_company"], main["per_company"],
        samples=bootstrap_samples, seed=bootstrap_seed,
    )
    primary = interpret_spread_ratio(spread)
    random_spread = spread_ratio(clean["per_company"], random["per_company"])
    d_main = spread["reduction"]
    d_random = random_spread["reduction"]
    if d_main > 0:
        specificity = {
            "gate": "G-M6-2",
            "main_reduction": d_main,
            "random_reduction": d_random,
            "limit": GATE_SPECIFICITY_FRACTION * d_main,
            "pass": d_random <= GATE_SPECIFICITY_FRACTION * d_main,
        }
    else:
        specificity = {
            "gate": "G-M6-2",
            "main_reduction": d_main,
            "random_reduction": d_random,
            "limit": None,
            "pass": None,
            "reason": "main-arm spread reduction is non-positive",
        }
    mean_clean = statistics.fmean(clean["all_margins"])
    mean_main = statistics.fmean(main["all_margins"])
    safety_g3 = _shift_gate("G-M6-3'", mean_main - mean_clean, GATE_MEAN_SHIFT)
    safety_g4 = _shift_gate("G-M6-4'", float(anon_intervention) - float(anon_clean), GATE_ANON_SHIFT)
    return {
        "primary": {"spread": spread, "interpretation": primary},
        "specificity": specificity,
        "safety": {
            "g3": {**safety_g3, "mean_clean": mean_clean, "mean_intervention": mean_main},
            "g4": {**safety_g4, "anon_clean": float(anon_clean), "anon_intervention": float(anon_intervention)},
        },
        "clean": {"per_company": clean["per_company"], "spread": spread["clean"]},
        "main": {"per_company": main["per_company"], "spread": spread["intervention"]},
        "random": {"per_company": random["per_company"], "spread": random_spread["intervention"]},
        "raw_runtime_payloads": False,
    }


def generation_flip_stats(
    clean_records: Sequence[Mapping[str, Any]],
    intervention_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize clean sell → intervention buy greedy flips."""
    clean = {str(row["prompt_id"]): row.get("generated_decision") for row in clean_records}
    intervention = {str(row["prompt_id"]): row.get("generated_decision") for row in intervention_records}
    if set(clean) != set(intervention):
        raise ValueError("generation records have mismatched prompt IDs")
    comparable = [key for key in clean if clean[key] in {"buy", "sell"} and intervention[key] in {"buy", "sell"}]
    flips = sum(1 for key in comparable if clean[key] == "sell" and intervention[key] == "buy")
    return {
        "n_prompts": len(clean),
        "n_comparable": len(comparable),
        "sell_to_buy_flips": flips,
        "flip_rate": flips / len(comparable) if comparable else None,
    }


__all__ = [
    "M6_BOOTSTRAP_SAMPLES",
    "M6_BOOTSTRAP_SEED",
    "M6_COMPANIES_PER_SECTOR",
    "M6_EXTERNAL_COMPANIES",
    "M6_SECTORS",
    "bootstrap_spread_ratio",
    "evaluate_m6",
    "generation_flip_stats",
    "interpret_spread_ratio",
    "spread_ratio",
    "validate_external_manifest",
]
