"""Neutral, atomic artifact materialization primitives.

The helpers in this module deliberately know nothing about experiment payloads. They
stream JSONL/CSV into a temporary file, fsync it, and publish it with one atomic
rename. Raw activations, gradients, tensors, and non-finite values are rejected.
"""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import (
    _atomic_replace,
    count_jsonl_records,
    file_sha256,
    sha256_file,
)

_RAW = {"activation", "activations", "gradient", "gradients", "residual", "residuals", "hidden_state", "hidden_states"}

def _check(value: Any, key: str = "") -> None:
    normalized = key.lower().replace("-", "_")
    if any(part in normalized.split("_") for part in _RAW) or normalized in _RAW:
        raise ValueError(f"raw activation/gradient payloads are not supported: {key}")
    if hasattr(value, "detach") or value.__class__.__module__.startswith("numpy"):
        raise ValueError("tensor/ndarray payloads are not supported")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite artifact values are not supported")
    if isinstance(value, Mapping):
        for k, v in value.items():
            _check(v, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            _check(v, key)

def _publish(path: Path, data: bytes, *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    return _atomic_replace(path, data)

def write_json(path: str | Path, value: Any, *, overwrite: bool = False, indent: int | None = 2) -> Path:
    _check(value)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False) + "\n"
    return _publish(Path(path), text.encode("utf-8"), overwrite=overwrite)

def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> int:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Unlike the historical helper, this never accumulates the full artifact.
    import os, tempfile
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError("JSONL records must be mappings")
                _check(row)
                stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(destination)
        return count
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str], *, overwrite: bool = False) -> int:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {destination}")
    names = list(fields)
    destination.parent.mkdir(parents=True, exist_ok=True)
    import os, tempfile
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(name); count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                if not isinstance(row, Mapping): raise TypeError("CSV rows must be mappings")
                _check(row)
                extra = set(row) - set(names)
                if extra: raise ValueError(f"unexpected output fields: {sorted(extra)}")
                writer.writerow(dict(row)); count += 1
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(destination); return count
    except BaseException:
        temporary.unlink(missing_ok=True); raise

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file(): raise FileNotFoundError(source)
    rows = []
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip(): continue
            try: value = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON in {source}:{number}") from exc
            if not isinstance(value, dict): raise ValueError(f"artifact row {source}:{number} is not an object")
            rows.append(value)
    if not rows: raise ValueError(f"artifact is empty: {source}")
    return rows

def write_metadata(path: str | Path, metadata: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    return write_json(path, dict(metadata), overwrite=overwrite)

def sidecar_metadata(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    for candidate in (source.parent / "metadata.json", source.with_suffix(source.suffix + ".metadata.json")):
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping): raise ValueError(f"metadata must be a JSON object: {candidate}")
            return dict(value)
    return None

def sidecar_artifact_hash(path: str | Path) -> str | None:
    metadata = sidecar_metadata(path)
    if metadata is None: return None
    for key in ("artifact_sha256", "output_sha256", "generated_outputs_sha256", "sha256", "parent_sha256"):
        value = metadata.get(key)
        if isinstance(value, str) and len(value) == 64: return value
    for key in ("artifact", "output", "generated_outputs"):
        nested = metadata.get(key)
        if isinstance(nested, Mapping) and isinstance(nested.get("sha256"), str): return nested["sha256"]
    return None

def load_parent_jsonl(path: str | Path, *, expected_sha256: str | None = None, previous_sha256: str | None = None) -> tuple[list[dict[str, Any]], str]:
    actual = sha256_file(path)
    for label, expected in (("expected parent", expected_sha256), ("recorded parent", previous_sha256), ("forward metadata", sidecar_artifact_hash(path))):
        if expected is not None and expected != actual:
            raise ValueError(f"parent forward artifact hash mismatch ({label}): expected {expected}, got {actual}")
    return read_jsonl(path), actual

def verified_artifact_ref(path: str | Path, *, artifact_type: str, role: str = "output", stage: str = "", record_count: int | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file(): raise FileNotFoundError(source)
    if any(part in artifact_type.lower().replace("-", "_").split("_") for part in _RAW): raise ValueError("raw activation/gradient artifacts are not supported")
    ref = {"artifact_type": artifact_type, "role": role, "stage": stage, "path": source.as_posix(), "sha256": file_sha256(source)}
    if record_count is None and source.suffix.lower() == ".jsonl": record_count = count_jsonl_records(source)
    if record_count is not None: ref["record_count"] = record_count
    if metadata: ref["metadata"] = dict(metadata)
    return ref

def verify_artifact_ref(ref: Mapping[str, Any], *, base_directory: str | Path | None = None) -> bool:
    path = Path(str(ref["path"]))
    if base_directory is not None and not path.is_absolute(): path = Path(base_directory) / path
    return path.is_file() and file_sha256(path) == ref.get("sha256")

# Compatibility spelling used by prompt-analysis.
_metadata_hash = lambda metadata: next((metadata[k] for k in ("artifact_sha256", "output_sha256", "generated_outputs_sha256", "sha256") if isinstance(metadata.get(k), str)), None)
