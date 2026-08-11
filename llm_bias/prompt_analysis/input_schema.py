"""Prompt CSV schema description shared by inspection and execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

PROMPT_COLUMN_PATTERN = re.compile(
    r"^prompt_(?P<context>with|without)_context_(?P<index>.+)$"
)
RETURN_PAIR_REQUIRED = (
    "cik",
    "filename",
    "item",
    "filing_date",
    "ticker",
    "peer_ticker",
    "system_prompt",
    "prompt",
    "counterfactual_prompt",
    "return_label",
    "fwd_return_1d",
)
TEN_K_CHANGE_REQUIRED = ("year", "cik", "item")


@dataclass(frozen=True)
class PromptInputSchema:
    """Completeness facts derived from one CSV header."""

    legacy_columns: tuple[str, ...]
    legacy_complete: bool
    return_pairs_complete: bool
    missing_return_pair_columns: tuple[str, ...]
    ten_k_change_complete: bool


def describe_prompt_input(fieldnames: Iterable[str]) -> PromptInputSchema:
    """Describe supported prompt schemas without choosing caller policy."""
    names = tuple(fieldnames)
    available = set(names)
    legacy_columns = tuple(
        name for name in names if PROMPT_COLUMN_PATTERN.fullmatch(name)
    )
    missing_return_pair_columns = tuple(
        name for name in RETURN_PAIR_REQUIRED if name not in available
    )
    return PromptInputSchema(
        legacy_columns=legacy_columns,
        legacy_complete="Date" in available and bool(legacy_columns),
        return_pairs_complete=not missing_return_pair_columns,
        missing_return_pair_columns=missing_return_pair_columns,
        ten_k_change_complete=tuple(names) == TEN_K_CHANGE_REQUIRED,
    )


def detect_dataset_format(
    fieldnames: Iterable[str], dataset_format: str = "auto"
) -> str:
    """Resolve the execution schema while rejecting incomplete explicit inputs."""
    if dataset_format not in {"auto", "legacy-wide", "return-pairs", "ten-k-change"}:
        raise ValueError(
            "dataset_format must be auto, legacy-wide, return-pairs, or ten-k-change"
        )
    schema = describe_prompt_input(fieldnames)
    if dataset_format == "ten-k-change":
        if not schema.ten_k_change_complete:
            raise ValueError(
                "ten-k-change CSV requires exact columns: "
                + ",".join(TEN_K_CHANGE_REQUIRED)
            )
        return dataset_format
    if dataset_format == "return-pairs":
        if schema.missing_return_pair_columns:
            raise ValueError(
                "return-pairs CSV is missing required columns: "
                + ", ".join(schema.missing_return_pair_columns)
            )
        return dataset_format
    if dataset_format == "auto":
        if schema.ten_k_change_complete:
            return "ten-k-change"
        if schema.return_pairs_complete:
            return "return-pairs"
    return "legacy-wide"
