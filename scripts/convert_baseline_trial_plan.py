"""
Convert a ../baseline repo trial plan into the legacy-wide prompt CSV
consumed by `prompt-analysis` in this repo.

Source: `../baseline/runs/<run>/trial_plan.jsonl` — one record per trial
with a fully rendered `prompt` field. Record identity is `trial_key`:
`(set_index, trial_index)` can repeat for the same (ticker, condition)
when the plan samples different evidence sets.

Output layout (legacy-wide schema, `prompt-analysis inspect-input` verifiable):
- one row per ticker, in first-appearance order in the source file
- static columns: Date (constant cohort label), ticker, name, sector, marketcap
- one prompt column per (condition, occurrence) where occurrence is the
  0-based index of a ticker's records for that condition in source order:
  `prompt_with_context_<condition>_<occurrence>`

The `ticker` column is carried into generated-output records by the
prompt-analysis generate stage; condition and occurrence are recoverable
from the `prompt_column`/`index` fields in all stage outputs.

A provenance JSON file is written next to the CSV (source path, SHA-256,
record counts, condition/trial matrix, output SHA-256).

Usage:
    uv run python scripts/convert_baseline_trial_plan.py \
        --source ../baseline/runs/qwen36-27b-50stocks/trial_plan.jsonl \
        --output data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

REQUIRED_FIELDS = ("ticker", "condition", "trial_index", "prompt", "name", "sector")
STATIC_COLUMNS = ("Date", "ticker", "name", "sector", "marketcap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(source: Path) -> list[dict]:
    records = []
    seen_keys = set()
    for line_number, line in enumerate(source.open(encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        missing = [field for field in REQUIRED_FIELDS if not str(record.get(field, "")).strip()]
        if missing:
            raise ValueError(f"{source} line {line_number}: missing fields {missing}")
        if "trial_key" in record:
            key = record["trial_key"]
            if key in seen_keys:
                raise ValueError(f"{source} line {line_number}: duplicate trial_key {key}")
            seen_keys.add(key)
        records.append(record)
    if not records:
        raise ValueError(f"{source} contains no records")
    return records


def build_matrix(records: list[dict]) -> tuple[list[str], dict[str, int]]:
    """Return (tickers in first-appearance order, per-condition occurrence counts)."""
    tickers: list[str] = []
    counts: dict[str, int] = {}
    for record in records:
        if record["ticker"] not in tickers:
            tickers.append(record["ticker"])
        counts[record["condition"]] = counts.get(record["condition"], 0) + 1
    return tickers, counts


def convert(source: Path, output: Path, date: str) -> dict:
    records = load_records(source)
    tickers, counts = build_matrix(records)
    conditions = list(counts)
    columns = [
        (condition, occurrence)
        for condition in conditions
        for occurrence in range(counts[condition] // len(tickers) + (1 if counts[condition] % len(tickers) else 0))
    ]

    # Occurrence index per (ticker, condition), in source order.
    occurrences: dict[tuple[str, str], int] = {}
    cells: dict[tuple[str, str, int], dict] = {}
    static: dict[str, dict] = {}
    for record in records:
        key = (record["ticker"], record["condition"])
        occurrence = occurrences.get(key, 0)
        cell_key = (*key, occurrence)
        if cell_key in cells:
            raise ValueError(f"duplicate (ticker, condition, occurrence): {cell_key}")
        cells[cell_key] = record
        occurrences[key] = occurrence + 1
        if record["ticker"] not in static:
            static[record["ticker"]] = record

    output.parent.mkdir(parents=True, exist_ok=True)
    header = list(STATIC_COLUMNS) + [
        f"prompt_with_context_{condition}_{occurrence}" for condition, occurrence in columns
    ]
    missing_cells = 0
    written_cells = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for ticker in tickers:
            row = {
                "Date": date,
                "ticker": ticker,
                "name": static[ticker]["name"],
                "sector": static[ticker]["sector"],
                "marketcap": static[ticker]["marketcap"],
            }
            for condition, occurrence in columns:
                record = cells.get((ticker, condition, occurrence))
                if record is None:
                    missing_cells += 1
                else:
                    written_cells += 1
                row[f"prompt_with_context_{condition}_{occurrence}"] = (
                    "" if record is None else record["prompt"]
                )
            writer.writerow([row[column] for column in header])
    if written_cells != len(records):
        raise ValueError(
            f"wrote {written_cells} prompt cells but source has {len(records)} records; "
            "non-uniform per-ticker trial counts are not supported"
        )

    provenance = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_record_count": len(records),
        "source_provider_models": sorted({str(r.get("model", "")) for r in records if r.get("model")}),
        "record_identity": "trial_key; (set_index, trial_index) may repeat per (ticker, condition)",
        "tickers": len(tickers),
        "conditions": [
            {"condition": condition, "trials_per_ticker": counts[condition] // len(tickers)}
            for condition in conditions
        ],
        "prompt_columns": len(columns),
        "missing_cells": missing_cells,
        "date": date,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "layout": "legacy-wide; one row per ticker; prompt_with_context_<condition>_<occurrence> columns",
    }
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance["output_path"] = str(output)
    provenance["provenance_path"] = str(provenance_path)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="path to trial_plan.jsonl")
    parser.add_argument("--output", required=True, help="path for the legacy-wide prompt CSV")
    parser.add_argument("--date", default="2026-01-01", help="constant Date label for the cohort")
    args = parser.parse_args()

    provenance = convert(Path(args.source), Path(args.output), args.date)
    print(
        f"wrote {provenance['output']} "
        f"({provenance['tickers']} tickers, {provenance['prompt_columns']} prompt columns, "
        f"{provenance['missing_cells']} missing cells)"
    )
    print(f"wrote {provenance['provenance_path']}")


if __name__ == "__main__":
    main()
