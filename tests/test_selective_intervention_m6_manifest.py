"""Deterministic M6 external selection manifest tests."""
from __future__ import annotations

import csv

import pytest

from llm_bias.selective_intervention.m6_manifest import prepare_external_manifest
from llm_bias.selective_intervention.m6_analysis import M6_SECTORS, validate_external_manifest


def _write_csv(path, *, n_per_sector=4):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "company_name", "gics_sector"])
        writer.writeheader()
        for sector_index, sector in enumerate(M6_SECTORS):
            for company_index in range(n_per_sector):
                writer.writerow({
                    "ticker": f"S{sector_index}{company_index}",
                    "company_name": f"Company S{sector_index}{company_index}",
                    "gics_sector": sector,
                })


def test_prepare_manifest_is_deterministic_and_stratified(tmp_path):
    source = tmp_path / "constituents.csv"
    _write_csv(source, n_per_sector=5)
    first = prepare_external_manifest(source, exclusions=("S00",), seed=42)
    second = prepare_external_manifest(source, exclusions=("S00",), seed=42)
    assert first == second
    validated = validate_external_manifest(first)
    assert len(validated["companies"]) == 12
    assert all(sum(item["sector"] == sector for item in first["companies"]) == 3 for sector in M6_SECTORS)
    assert "S00" not in {item["ticker"] for item in first["companies"]}


def test_prepare_manifest_fails_on_inconsistent_source_rows(tmp_path):
    source = tmp_path / "constituents.csv"
    _write_csv(source)
    with source.open("a", encoding="utf-8") as handle:
        handle.write("S00,Changed Name,Health Care\n")
    with pytest.raises(ValueError, match="inconsistent"):
        prepare_external_manifest(source)
