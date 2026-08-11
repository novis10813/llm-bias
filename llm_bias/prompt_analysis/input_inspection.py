"""Read-only compatibility and data-quality inspection for prompt CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from llm_bias.prompt_analysis.input_schema import (
    PROMPT_COLUMN_PATTERN,
    RETURN_PAIR_REQUIRED,
    describe_prompt_input,
)

LABELS = ("very bearish", "bearish", "neutral", "bullish", "very bullish")
THRESHOLDS = {
    "very_bearish": "return < -0.02",
    "bearish": "-0.02 <= return < -0.005",
    "neutral": "-0.005 <= return <= 0.005",
    "bullish": "0.005 < return <= 0.02",
    "very_bullish": "return > 0.02",
}


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _label_for(value: float) -> str:
    if value < -0.02:
        return "very bearish"
    if value < -0.005:
        return "bearish"
    if value <= 0.005:
        return "neutral"
    if value <= 0.02:
        return "bullish"
    return "very bullish"


def _date_ok(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)) and _date_parse(value)


def _date_parse(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def inspect_input(path: str | Path) -> dict[str, Any]:
    """Inspect a CSV without modifying it, returning a JSON-compatible report."""
    source = Path(path)
    raw = source.read_bytes()
    report: dict[str, Any] = {
        "input": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema": None,
        "rows": 0,
        "counts": {},
        "missing": {},
        "warnings": [],
        "errors": [],
    }
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        report["errors"].append(f"input is not valid UTF-8: {exc}")
        return report
    try:
        reader = csv.DictReader(text.splitlines(keepends=True), strict=True)
        fields = reader.fieldnames
        if not fields:
            report["errors"].append("CSV has no header row")
            return report
        if any(field is None or not field.strip() for field in fields):
            report["errors"].append("CSV header contains an empty column name")
            return report
        if len(fields) != len(set(fields)):
            report["errors"].append("CSV header contains duplicate column names")
            return report
        schema = describe_prompt_input(fields)
        legacy_columns = list(schema.legacy_columns)
        if schema.ten_k_change_complete:
            report["schema"] = "ten-k-change"
        elif schema.legacy_complete and schema.return_pairs_complete:
            report["errors"].append("input matches both legacy-wide and return-pairs schemas; specify a non-ambiguous CSV")
            return report
        elif not schema.legacy_complete and not schema.return_pairs_complete:
            report["errors"].append(
                "input does not match a complete formal schema (legacy-wide requires Date and prompt_with/without_context_*; "
                "return-pairs is missing: "
                + ", ".join(schema.missing_return_pair_columns)
                + ")"
            )
            return report
        if report["schema"] is None:
            report["schema"] = (
                "legacy-wide" if schema.legacy_complete else "return-pairs"
            )
        rows: list[dict[str, str | None]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                report["errors"].append(f"row {row_number} has more fields than the header")
                continue
            rows.append(row)
        report["rows"] = len(rows)
    except (csv.Error, UnicodeError) as exc:
        report["errors"].append(f"invalid CSV: {exc}")
        return report

    if report["schema"] == "legacy-wide":
        _inspect_legacy(rows, fields, legacy_columns, report)
    elif report["schema"] == "return-pairs":
        _inspect_returns(rows, report)
    else:
        _inspect_ten_k_change(rows, report)
    return report


def _inspect_legacy(rows: list[dict[str, str | None]], fields: list[str], columns: list[str], report: dict[str, Any]) -> None:
    missing = {f: sum(_empty(r.get(f)) for r in rows) for f in ["Date", *columns]}
    report["missing"] = {k: v for k, v in missing.items() if v}
    invalid_dates = sum(not _date_ok(str(r.get("Date", "")).strip()) for r in rows if not _empty(r.get("Date")))
    report["counts"] = {
        "conditions": {"with_context": sum(1 for c in columns if c.startswith("prompt_with_context_")), "without_context": sum(1 for c in columns if c.startswith("prompt_without_context_"))},
        "prompt_columns": len(columns),
        "nonempty_prompt_cells": sum(not _empty(r.get(c)) for r in rows for c in columns),
    }
    if invalid_dates or missing.get("Date"):
        report["errors"].append(f"Date contains {invalid_dates} invalid ISO dates and {missing.get('Date', 0)} missing values")
    malformed = [
        field
        for field in fields
        if field.startswith("prompt_")
        and not PROMPT_COLUMN_PATTERN.fullmatch(field)
    ]
    if malformed:
        report["errors"].append("prompt-like columns must match ^prompt_(with|without)_context_(.+)$: " + ", ".join(malformed))


def _inspect_ten_k_change(rows: list[dict[str, str | None]], report: dict[str, Any]) -> None:
    from llm_bias.prompt_analysis.input_data import TEN_K_ITEM_NAMES

    invalid_year = invalid_cik = invalid_item = 0
    fields: Counter[str] = Counter()
    for row in rows:
        year = str(row.get("year") or "").strip()
        cik = str(row.get("cik") or "").strip()
        item = str(row.get("item") or "").strip()
        if not re.fullmatch(r"\d{4}", year):
            invalid_year += 1
        if not cik:
            invalid_cik += 1
        if "=" not in item:
            invalid_item += 1
            continue
        field, value = (part.strip() for part in item.split("=", 1))
        if field not in TEN_K_ITEM_NAMES:
            invalid_item += 1
        else:
            fields[field] += 1
    report["counts"] = {
        "fields": dict(sorted(fields.items())),
        "invalid_year": invalid_year,
        "invalid_cik": invalid_cik,
        "invalid_item": invalid_item,
    }
    if invalid_year or invalid_cik or invalid_item:
        report["errors"].append(
            f"ten-k-change rows invalid: year={invalid_year}, cik={invalid_cik}, item={invalid_item}"
        )


def _inspect_returns(rows: list[dict[str, str | None]], report: dict[str, Any]) -> None:
    missing = {f: sum(_empty(r.get(f)) for r in rows) for f in RETURN_PAIR_REQUIRED}
    report["missing"] = {k: v for k, v in missing.items() if v}
    duplicate_ids: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    tickers: Counter[str] = Counter()
    invalid_dates = invalid_returns = label_mismatches = 0
    for row in rows:
        values = {k: "" if row.get(k) is None else str(row[k]).strip() for k in RETURN_PAIR_REQUIRED}
        pair_id = "|".join(values[k] for k in ("cik", "filename", "item"))
        duplicate_ids[pair_id] += 1
        if values["filing_date"] and not _date_ok(values["filing_date"]):
            invalid_dates += 1
        value: float | None = None
        if values["fwd_return_1d"]:
            try:
                value = float(values["fwd_return_1d"])
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                invalid_returns += 1
        label = values["return_label"]
        if label:
            labels[label] += 1
            if label not in LABELS:
                report["errors"].append(f"unknown return_label {label!r}")
            elif value is not None and _label_for(value) != label:
                label_mismatches += 1
        if values["ticker"]: tickers[values["ticker"]] += 1
    dups = {k: v for k, v in duplicate_ids.items() if v > 1}
    report["counts"] = {
        "pair_ids_unique": not dups,
        "duplicate_pair_ids": dups,
        "conditions": {"prompt": len(rows), "counterfactual_prompt": len(rows)},
        "tickers": dict(sorted(tickers.items())),
        "labels": dict(sorted(labels.items())),
        "label_threshold_mismatches": label_mismatches,
    }
    if report["missing"]:
        report["errors"].append("required columns contain missing values: " + ", ".join(f"{k}={v}" for k, v in report["missing"].items()))
    if invalid_dates: report["errors"].append(f"filing_date has {invalid_dates} invalid ISO dates")
    if invalid_returns: report["errors"].append(f"fwd_return_1d has {invalid_returns} missing, non-numeric, or non-finite values")
    if dups: report["errors"].append(f"duplicate pair IDs found ({len(dups)} IDs)")
    if label_mismatches: report["errors"].append(f"return_label disagrees with five-class thresholds in {label_mismatches} rows")
    report["thresholds"] = THRESHOLDS


def inspect_input_to_json(path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    report = inspect_input(path)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        Path(output).write_text(rendered, encoding="utf-8")
    return report
