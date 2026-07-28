"""Calibration prompt sources for standalone Jacobian-lens fitting."""

from __future__ import annotations

import json
from pathlib import Path


def builtin_calibration_prompts(count: int = 16) -> list[str]:
    """Return deterministic, non-task calibration passages."""
    if count < 1:
        raise ValueError("calibration prompt count must be positive")
    seeds = [
        "A small village sits beside a river and depends on the water for travel and trade.",
        "Researchers collected careful notes before comparing the results of the experiment.",
        "The old library contains maps, letters, and books from many different periods.",
        "During the afternoon, clouds moved across the valley while the temperature slowly fell.",
        "A teacher asked the class to explain the pattern using a short example and clear evidence.",
        "The mechanic inspected the engine, replaced a worn part, and tested the vehicle again.",
        "Several birds gathered on the roof before the storm arrived from the western horizon.",
        "The museum displayed tools that showed how people solved practical problems in the past.",
    ]
    return [
        seeds[index % len(seeds)] + f" This is calibration passage {index}."
        for index in range(count)
    ]


def load_calibration_prompts(
    path: str | Path,
    *,
    field: str = "text",
    count: int | None = None,
) -> list[str]:
    """Load prompts from JSONL objects or a UTF-8 text file with one prompt per line."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if count is not None and count < 1:
        raise ValueError("calibration prompt count must be positive")

    prompts: list[str] = []
    with source.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if source.suffix.lower() == ".jsonl":
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {source}:{line_number}"
                    ) from exc
                if not isinstance(record, dict) or not isinstance(record.get(field), str):
                    raise ValueError(
                        f"{source}:{line_number} must contain string field {field!r}"
                    )
                stripped = record[field].strip()
            if stripped:
                prompts.append(stripped)
            if count is not None and len(prompts) >= count:
                break
    if not prompts:
        raise ValueError(f"{source} contains no calibration prompts")
    return prompts
