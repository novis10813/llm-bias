"""Deterministic M6 external-population analysis tests."""
from __future__ import annotations

import pytest

from llm_bias.selective_intervention.m6_analysis import (
    bootstrap_spread_ratio,
    evaluate_m6,
    generation_flip_stats,
    interpret_spread_ratio,
    validate_external_manifest,
)


SECTORS = ("Health Care", "Information Technology", "Financials", "Industrials")


def _manifest() -> dict:
    companies = []
    eligible = []
    for sector_index, sector in enumerate(SECTORS):
        for company_index in range(3):
            ticker = f"E{sector_index}{company_index}"
            item = {"ticker": ticker, "name": f"External {ticker}", "sector": sector}
            companies.append(item)
            eligible.append(item)
    return {
        "schema_version": "selective-intervention-m6-external-v1",
        "selection_seed": 42,
        "eligible_pool": eligible,
        "exclusion_list": ["NSC", "BLK", "IT", "BDX"],
        "companies": companies,
    }


def _records(values: dict[str, float], arm: str) -> list[dict]:
    return [
        {"prompt_id": f"{ticker}:{variant}", "ticker": ticker, "margin": value, "arm": arm}
        for ticker, value in values.items()
        for variant in range(4)
    ]


def test_validate_external_manifest_requires_three_per_sector():
    validated = validate_external_manifest(_manifest())
    assert len(validated["companies"]) == 12
    broken = _manifest()
    broken["companies"] = broken["companies"][:-1]
    with pytest.raises(ValueError, match="exactly 12"):
        validate_external_manifest(broken)


def test_validate_external_manifest_rejects_seed_search_or_bad_sector_counts():
    broken = _manifest()
    broken["selection_seed"] = 7
    with pytest.raises(ValueError, match="seed"):
        validate_external_manifest(broken)
    broken = _manifest()
    broken["companies"][0]["sector"] = "Unknown"
    with pytest.raises(ValueError, match="invalid"):
        validate_external_manifest(broken)


def test_bootstrap_spread_ratio_is_deterministic():
    clean = {f"E{i}": float(i) for i in range(12)}
    intervention = {key: value * 0.4 for key, value in clean.items()}
    first = bootstrap_spread_ratio(clean, intervention, samples=500, seed=42)
    second = bootstrap_spread_ratio(clean, intervention, samples=500, seed=42)
    assert first == second
    assert first["ratio"] == pytest.approx(0.4)
    assert first["bootstrap"]["samples"] == 500


def test_interpret_spread_ratio_has_three_levels():
    confirmed = {"ratio": 0.4, "bootstrap": {"ci": [0.2, 0.5]}}
    suggestive = {"ratio": 0.4, "bootstrap": {"ci": [0.1, 0.8]}}
    failed = {"ratio": 0.6, "bootstrap": {"ci": [0.2, 0.9]}}
    assert interpret_spread_ratio(confirmed)["status"] == "confirmed"
    assert interpret_spread_ratio(suggestive)["status"] == "suggestive_unresolved"
    assert interpret_spread_ratio(failed)["status"] == "fail"


def test_evaluate_m6_uses_company_level_spread_and_secondary_safety():
    clean = {f"E{i}": float(i) for i in range(12)}
    clean_mean = sum(clean.values()) / len(clean)
    main = {key: clean_mean + 0.4 * (value - clean_mean) for key, value in clean.items()}
    random = {key: value for key, value in clean.items()}
    result = evaluate_m6(
        _records(clean, "clean"),
        _records(main, "dose_100"),
        _records(random, "ctrl_random"),
        anon_clean=-3.0,
        anon_intervention=-3.02,
        bootstrap_samples=500,
        bootstrap_seed=42,
    )
    assert result["primary"]["spread"]["ratio"] == pytest.approx(0.4)
    assert result["primary"]["interpretation"]["status"] in {"confirmed", "suggestive_unresolved"}
    assert result["specificity"]["pass"] is True
    assert result["safety"]["g3"]["pass"] is True
    assert result["safety"]["g4"]["pass"] is True


def test_generation_flip_stats_does_not_confuse_margin_with_generation():
    clean = [
        {"prompt_id": "a", "generated_decision": "sell"},
        {"prompt_id": "b", "generated_decision": "buy"},
        {"prompt_id": "c", "generated_decision": None},
    ]
    intervention = [
        {"prompt_id": "a", "generated_decision": "buy"},
        {"prompt_id": "b", "generated_decision": "buy"},
        {"prompt_id": "c", "generated_decision": "buy"},
    ]
    result = generation_flip_stats(clean, intervention)
    assert result["n_comparable"] == 2
    assert result["sell_to_buy_flips"] == 1
    assert result["flip_rate"] == pytest.approx(0.5)
