"""Dataset adapters for prompt-analysis CSV inputs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm_bias.prompt_analysis.input_schema import (
    PROMPT_COLUMN_PATTERN,
    detect_dataset_format,
)

RETURN_LABELS = ("very bullish", "bullish", "neutral", "bearish", "very bearish")
TEN_K_ITEM_NAMES = {
    "company": "company name",
    "state_location": "state/location",
    "state_of_inc": "state of incorporation",
    "sic": "SIC code",
}
TEN_K_PROMPT_TEMPLATE = (
    "In year {year}, what is the {item_name} of the company with CIK code {cik}? "
    "Answer without explanation"
)


@dataclass(frozen=True)
class PromptColumn:
    """A prompt column and its condition metadata."""

    name: str
    index: str
    context: str
    condition: str | None = None


@dataclass(frozen=True)
class PromptTable:
    """Normalized prompt rows with their resolved input schema."""

    columns: list[PromptColumn]
    rows: list[dict[str, Any]]
    dataset_format: str


def discover_prompt_columns(
    fieldnames: Iterable[str],
    selected: Iterable[str] | None = None,
) -> list[PromptColumn]:
    """Find and parse ``prompt_{with,without}_context_*`` CSV columns."""
    available = list(fieldnames)
    requested = (
        list(selected)
        if selected is not None
        else [
            name for name in available if PROMPT_COLUMN_PATTERN.fullmatch(name)
        ]
    )
    if not requested:
        raise ValueError(
            "CSV has no prompt_with_context_* or prompt_without_context_* columns"
        )

    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"CSV is missing requested prompt columns: {', '.join(missing)}"
        )

    columns: list[PromptColumn] = []
    for name in requested:
        match = PROMPT_COLUMN_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(
                f"prompt column {name!r} must match "
                "'prompt_with_context_*' or 'prompt_without_context_*'"
            )
        columns.append(
            PromptColumn(
                name=name,
                index=match.group("index"),
                context=match.group("context"),
            )
        )
    return columns


def load_prompt_table(
    input_path: Path,
    selected_columns: Iterable[str] | None = None,
    max_rows: int | None = None,
    *,
    dataset_format: str = "auto",
) -> PromptTable:
    """Load and normalize one supported prompt dataset."""
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header")
        resolved = detect_dataset_format(reader.fieldnames, dataset_format)
        if resolved == "legacy-wide":
            if "Date" not in reader.fieldnames:
                raise ValueError("legacy-wide CSV requires a Date column")
            columns = discover_prompt_columns(reader.fieldnames, selected_columns)
            rows = []
            for row_index, row in enumerate(reader):
                if max_rows is not None and row_index >= max_rows:
                    break
                rows.append(row)
        elif resolved == "ten-k-change":
            if selected_columns:
                invalid = set(selected_columns) - {"prompt"}
                if invalid:
                    raise ValueError("ten-k-change prompt columns only support prompt")
            columns = [PromptColumn("prompt", "ten_k_change", "ten_k_change")]
            rows = []
            for row_index, row in enumerate(reader):
                if max_rows is not None and row_index >= max_rows:
                    break
                year = row.get("year", "").strip()
                cik = row.get("cik", "").strip()
                item = row.get("item", "").strip()
                if not year.isdigit() or len(year) != 4:
                    raise ValueError(f"ten-k-change row {row_index} has invalid year")
                if not cik:
                    raise ValueError(f"ten-k-change row {row_index} has empty cik")
                if "=" not in item:
                    raise ValueError(f"ten-k-change row {row_index} item must contain '='")
                item_field, item_value = item.split("=", 1)
                item_field, item_value = item_field.strip(), item_value.strip()
                if item_field not in TEN_K_ITEM_NAMES:
                    raise ValueError(f"ten-k-change row {row_index} has invalid item")
                item_name = TEN_K_ITEM_NAMES[item_field]
                expanded = dict(row)
                expanded.update(
                    {
                        "input_schema": "ten-k-change",
                        "row_index": row_index,
                        "year": int(year),
                        "cik": cik,
                        "item": item,
                        "item_field": item_field,
                        "item_value": item_value,
                        "item_name": item_name,
                        "prompt": TEN_K_PROMPT_TEMPLATE.format(
                            year=year, item_name=item_name, cik=cik
                        ),
                        "Date": f"{year}-12-31",
                        "index": "ten_k_change",
                        "context": "ten_k_change",
                    }
                )
                rows.append(expanded)
        else:
            if selected_columns:
                invalid = set(selected_columns) - {"original", "counterfactual"}
                if invalid:
                    raise ValueError(
                        "return-pairs prompt columns must be original or counterfactual"
                    )
            columns = [
                PromptColumn("original", "return_pair", "original", "original"),
                PromptColumn(
                    "counterfactual",
                    "return_pair",
                    "counterfactual",
                    "counterfactual",
                ),
            ]
            rows = []
            seen: set[str] = set()
            for pair_index, row in enumerate(reader):
                if max_rows is not None and pair_index >= max_rows:
                    break
                pair_id = "|".join(
                    row[name].strip() for name in ("cik", "filename", "item")
                )
                if not all(
                    row[name].strip() for name in ("cik", "filename", "item")
                ):
                    raise ValueError(
                        f"return-pairs row {pair_index} has empty pair identity field"
                    )
                if pair_id in seen:
                    raise ValueError(f"duplicate return-pairs pair_id: {pair_id}")
                seen.add(pair_id)
                try:
                    fwd_return_1d = float(row["fwd_return_1d"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"return-pairs row {pair_index} has invalid fwd_return_1d"
                    ) from exc
                if not math.isfinite(fwd_return_1d):
                    raise ValueError(
                        f"return-pairs row {pair_index} has non-finite fwd_return_1d"
                    )
                target_label = row["return_label"].strip()
                if target_label not in RETURN_LABELS:
                    raise ValueError(
                        f"return-pairs row {pair_index} has invalid return_label: "
                        f"{target_label!r}"
                    )
                for condition, prompt_key in (
                    ("original", "prompt"),
                    ("counterfactual", "counterfactual_prompt"),
                ):
                    expanded = dict(row)
                    expanded.update(
                        {
                            "input_schema": "return-pairs",
                            "pair_id": pair_id,
                            "filing_date": row["filing_date"],
                            "ticker": row["ticker"],
                            "peer_ticker": row["peer_ticker"],
                            "condition": condition,
                            "target_label": target_label,
                            "fwd_return_1d": fwd_return_1d,
                            "prompt_column": condition,
                            "prompt": row[prompt_key],
                            "Date": row["filing_date"],
                            "index": "return_pair",
                            "context": condition,
                        }
                    )
                    rows.append(expanded)
    if not rows:
        raise ValueError(f"{input_path} contains no data rows")
    return PromptTable(columns=columns, rows=rows, dataset_format=resolved)
