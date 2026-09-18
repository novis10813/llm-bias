"""Create the frozen M6 external-company selection manifest.

The selection seed and four-sector stratification are protocol constants. This
script performs no model inference and must run before any M6 model forward.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_bias.selective_intervention.m6_manifest import DEFAULT_EXCLUSION, write_external_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constituents-csv",
        default="data/sp500_constituents_2020_2025.csv",
        help="Canonical constituent source",
    )
    parser.add_argument(
        "--output",
        default="data/baseline/selective-intervention-m6/external-manifest.json",
        help="M6 manifest output path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUSION),
        help="Tickers excluded from the external pool (defaults to the source 16)",
    )
    args = parser.parse_args()
    output = write_external_manifest(
        args.output,
        args.constituents_csv,
        exclusions=args.exclude,
        seed=args.seed,
    )
    print(f"wrote frozen M6 external manifest: {output}")


if __name__ == "__main__":
    main()
