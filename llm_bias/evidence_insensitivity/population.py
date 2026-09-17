"""Frozen S&P 500 population and deterministic stratified splits."""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import Any

SEED = 20260916
EXPECTED_ROWS = 503
DEFAULT_DATA_PATH = Path("data/all_constituents_2020_2025.csv")


def load_population(path: str | Path = DEFAULT_DATA_PATH) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: row.get(key, "").strip() for key in ("ticker", "company_name", "gics_sector")}
            for row in csv.DictReader(handle)
            if row.get("index_name") == "S&P 500" and row.get("year") == "2024"
        ]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} S&P 500 2024 rows, got {len(rows)}")
    tickers = [row["ticker"] for row in rows]
    if len(set(tickers)) != len(tickers):
        raise ValueError("population tickers must be unique")
    if any(not row["company_name"] for row in rows):
        raise ValueError("population company names must be non-empty")
    if any(not row["ticker"] or not row["gics_sector"] for row in rows):
        raise ValueError("population ticker and sector fields must be non-empty")
    return sorted(rows, key=lambda row: row["ticker"])


def _stratified_sample(rows: list[dict[str, str]], n: int, seed: int) -> set[str]:
    if n < 0 or n > len(rows):
        raise ValueError("sample size out of range")
    by_sector: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_sector.setdefault(row["gics_sector"], []).append(row)
    sectors = sorted(by_sector)
    raw = {sector: n * len(by_sector[sector]) / len(rows) for sector in sectors}
    counts = {sector: int(raw[sector]) for sector in sectors}
    remainder = n - sum(counts.values())
    for sector in sorted(sectors, key=lambda s: (-(raw[s] - counts[s]), s))[:remainder]:
        counts[sector] += 1
    selected: set[str] = set()
    for index, sector in enumerate(sectors):
        bucket = list(by_sector[sector])
        local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{index}:{sector}".encode()).digest()[:8], "big")
        random.Random(local_seed).shuffle(bucket)
        selected.update(row["ticker"] for row in bucket[:counts[sector]])
    if len(selected) != n:
        raise AssertionError(f"stratified sample produced {len(selected)} rows, expected {n}")
    return selected


def population_digest(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256("\n".join(sorted(row["ticker"] for row in rows)).encode()).hexdigest()


def make_splits(rows: list[dict[str, str]], *, seed: int = SEED) -> tuple[dict[str, str], set[str]]:
    holdout_n = round(len(rows) * 0.20)
    holdout = _stratified_sample(rows, holdout_n, seed)
    order_swap = _stratified_sample(rows, 100, seed)
    return ({row["ticker"]: ("hold-out" if row["ticker"] in holdout else "discovery") for row in rows}, order_swap)


def split_counts(assignments: dict[str, str]) -> dict[str, int]:
    return {name: sum(value == name for value in assignments.values()) for name in ("discovery", "hold-out")}
