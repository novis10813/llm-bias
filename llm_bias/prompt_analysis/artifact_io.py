"""Small, deterministic helpers for prompt-analysis JSONL artifacts.

Artifacts are intentionally treated as byte-addressed inputs.  The backward
stage records the exact parent JSONL hash and refuses to reuse an output whose
recorded parent no longer matches.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import sha256_file


ARTIFACT_SCHEMA_VERSION = 1


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL artifact and reject malformed/non-object rows."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"artifact row {source}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"artifact is empty: {source}")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically write compact, UTF-8 JSONL and return the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("JSONL rows must be mappings")
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(destination)
    return destination


def _metadata_hash(metadata: Mapping[str, Any]) -> str | None:
    for key in (
        "artifact_sha256",
        "output_sha256",
        "generated_outputs_sha256",
        "sha256",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value
    for key in ("artifact", "output", "generated_outputs"):
        nested = metadata.get(key)
        if isinstance(nested, Mapping):
            value = nested.get("sha256")
            if isinstance(value, str) and len(value) == 64:
                return value
    return None


def sidecar_artifact_hash(path: str | Path) -> str | None:
    """Read a sibling metadata hash if one is present, without guessing rows."""
    source = Path(path)
    candidates = [source.parent / "metadata.json", source.with_suffix(source.suffix + ".metadata.json")]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metadata JSON: {candidate}") from exc
        if isinstance(value, Mapping):
            digest = _metadata_hash(value)
            if digest is not None:
                return digest
    return None


def load_parent_jsonl(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    previous_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load a parent artifact and fail closed on a digest mismatch.

    ``previous_sha256`` is used by reruns against an existing backward output;
    ``expected_sha256`` is useful to a coordinator that has a manifest-level
    digest.  A sibling forward metadata digest is also honored.
    """
    source = Path(path)
    actual = sha256_file(source)
    for label, expected in (
        ("expected parent", expected_sha256),
        ("recorded parent", previous_sha256),
        ("forward metadata", sidecar_artifact_hash(source)),
    ):
        if expected is not None and expected != actual:
            raise ValueError(
                f"parent forward artifact hash mismatch ({label}): "
                f"expected {expected}, got {actual}"
            )
    return read_jsonl(source), actual


def write_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Atomically write an indented JSON metadata sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
