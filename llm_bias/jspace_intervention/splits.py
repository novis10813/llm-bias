"""Deterministic ticker-grouped experiment splits."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping

SPLITS = ("discovery", "calibration", "test")


def _key(ticker: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{ticker}".encode()).hexdigest()


def assign_ticker_splits(
    ticker_to_sector: Mapping[str, str],
    *,
    seed: int = 0,
    ratios: tuple[int, int, int] = (3, 1, 1),
) -> dict[str, str]:
    """Assign whole tickers within each sector to 60/20/20-style folds."""
    if any(value <= 0 for value in ratios):
        raise ValueError("split ratios must be positive")
    denominator = sum(ratios)
    schedule = [
        split
        for split, count in zip(SPLITS, ratios, strict=True)
        for _ in range(count)
    ]
    by_sector: dict[str, list[str]] = defaultdict(list)
    for ticker, sector in ticker_to_sector.items():
        if not ticker or not sector:
            raise ValueError("ticker and sector must be non-empty")
        by_sector[sector].append(ticker)
    assignments = {}
    for sector, tickers in sorted(by_sector.items()):
        ordered = sorted(set(tickers), key=lambda ticker: _key(ticker, seed))
        if len(ordered) < denominator:
            raise ValueError(
                f"sector {sector!r} has {len(ordered)} tickers; at least {denominator} required"
            )
        for index, ticker in enumerate(ordered):
            assignments[ticker] = schedule[index % denominator]
    return assignments


def assign_balanced_ticker_splits(
    ticker_to_sector: Mapping[str, str],
    ticker_buy_rate: Mapping[str, float],
    ticker_marketcap: Mapping[str, float],
    *,
    seed: int = 0,
) -> dict[str, str]:
    """Balance 3/1/1 splits locally along baseline buy-rate order.

    Each consecutive block of five companies with similar buy rates receives
    three discovery, one calibration, and one test assignment. Market cap and a
    fixed hash decide assignment order within each block.
    """
    missing = set(ticker_to_sector) - set(ticker_buy_rate)
    if missing:
        raise ValueError(f"missing baseline buy rates for tickers: {sorted(missing)[:5]}")
    by_sector: dict[str, list[str]] = defaultdict(list)
    for ticker, sector in ticker_to_sector.items():
        by_sector[sector].append(ticker)
    assignments = {}
    base_schedule = ["discovery", "discovery", "discovery", "calibration", "test"]
    for sector, tickers in sorted(by_sector.items()):
        if len(tickers) < 5:
            raise ValueError(f"sector {sector!r} needs at least five tickers")
        ordered = sorted(
            tickers,
            key=lambda ticker: (float(ticker_buy_rate[ticker]), _key(ticker, seed)),
        )
        counts = defaultdict(int)
        targets = {
            "discovery": len(ordered) * 3 / 5,
            "calibration": len(ordered) / 5,
            "test": len(ordered) / 5,
        }
        for block_start in range(0, len(ordered), 5):
            block = ordered[block_start : block_start + 5]
            block.sort(
                key=lambda ticker: (
                    math.log1p(max(0.0, float(ticker_marketcap.get(ticker, 0.0)))),
                    _key(ticker, seed + block_start),
                )
            )
            if len(block) == 5:
                rotation = int(_key(f"{sector}:{block_start}", seed)[:8], 16) % 5
                schedule = base_schedule[rotation:] + base_schedule[:rotation]
            else:
                schedule = []
                provisional = dict(counts)
                for _ in block:
                    split = min(
                        targets,
                        key=lambda name: provisional.get(name, 0) - targets[name],
                    )
                    schedule.append(split)
                    provisional[split] = provisional.get(split, 0) + 1
            for ticker, split in zip(block, schedule, strict=True):
                assignments[ticker] = split
                counts[split] += 1
    return assignments


def split_counts(
    assignments: Mapping[str, str], ticker_to_sector: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ticker, split in assignments.items():
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        counts[ticker_to_sector[ticker]][split] += 1
    return {sector: dict(values) for sector, values in sorted(counts.items())}


__all__ = [
    "SPLITS",
    "assign_balanced_ticker_splits",
    "assign_ticker_splits",
    "split_counts",
]
