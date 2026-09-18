"""Deterministic M6 external-company manifest preparation."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .m6_analysis import M6_BOOTSTRAP_SEED, M6_COMPANIES_PER_SECTOR, M6_SECTORS

M6_MANIFEST_SCHEMA = "selective-intervention-m6-external-v1"
DEFAULT_EXCLUSION = (
    "AMAT", "GLW", "HPE", "IT",
    "AXP", "BLK", "C", "GS",
    "ABT", "BDX", "DHR", "SYK",
    "CSX", "DE", "HON", "NSC",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_external_manifest(
    constituents_csv: str | Path,
    *,
    exclusions: list[str] | tuple[str, ...] = DEFAULT_EXCLUSION,
    seed: int = M6_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build the frozen 4-sector × 3-company M6 selection manifest."""
    # Fix the seed before reading or inspecting the candidate pool. The function
    # never searches over seeds or uses model-derived values.
    rng = random.Random(int(seed))
    path = Path(constituents_csv)
    excluded = {str(ticker).strip() for ticker in exclusions}
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {"ticker", "company_name", "gics_sector"}
    by_ticker: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"constituents CSV requires columns {sorted(required)}")
        for row in reader:
            ticker = str(row["ticker"]).strip()
            name = str(row["company_name"]).strip()
            sector = str(row["gics_sector"]).strip()
            if not ticker or not name or sector not in M6_SECTORS:
                continue
            previous = by_ticker.get(ticker)
            if previous is not None and previous != (name, sector):
                raise ValueError(f"ticker {ticker} has inconsistent name/sector rows")
            by_ticker[ticker] = (name, sector)
    eligible = [
        {"ticker": ticker, "name": name, "sector": sector}
        for ticker, (name, sector) in sorted(by_ticker.items())
        if ticker not in excluded
    ]
    selected: list[dict[str, str]] = []
    for sector in M6_SECTORS:
        candidates = [item for item in eligible if item["sector"] == sector]
        if len(candidates) < M6_COMPANIES_PER_SECTOR:
            raise ValueError(f"eligible pool has fewer than 3 companies in {sector}")
        selected.extend(rng.sample(candidates, M6_COMPANIES_PER_SECTOR))
    selected.sort(key=lambda item: (M6_SECTORS.index(item["sector"]), item["ticker"]))
    return {
        "schema_version": M6_MANIFEST_SCHEMA,
        "selection_seed": int(seed),
        "selection_rule": "stratified_random_3_per_sector_from_canonical_sp500_pool",
        "source": {
            "path": str(path),
            "sha256": _sha256(path),
            "sectors": list(M6_SECTORS),
        },
        "exclusion_list": sorted(excluded),
        "eligible_pool": eligible,
        "companies": selected,
    }


def write_external_manifest(
    output: str | Path,
    constituents_csv: str | Path,
    *,
    exclusions: list[str] | tuple[str, ...] = DEFAULT_EXCLUSION,
    seed: int = M6_BOOTSTRAP_SEED,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = prepare_external_manifest(constituents_csv, exclusions=exclusions, seed=seed)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["DEFAULT_EXCLUSION", "prepare_external_manifest", "write_external_manifest"]
