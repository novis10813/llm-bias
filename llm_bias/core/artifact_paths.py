"""Deterministic identities and paths for experiment artifacts.

This module is deliberately independent from experiment producers.  It defines the
on-disk contract used by later workflow units without reading or writing model
activations, gradients, or other large runtime state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from llm_bias.core.lens_artifacts import (
    canonical_lens_path,
    lens_artifact_root,
    model_slug,
)

DEFAULT_ARTIFACT_ROOT = Path("artifacts")


def _safe_slug(value: str, *, label: str) -> str:
    """Apply the repository's model-slug spelling to a generic identity."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    # model_slug is the established convention: Hub IDs use ``--`` and local
    # paths use their final component.  Reusing it also keeps model/dataset IDs
    # safe as directory names.
    try:
        result = model_slug(value.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot derive a {label} slug from {value!r}") from exc
    if result in {".", ".."} or "/" in result or "\\" in result:
        raise ValueError(f"invalid {label} slug: {result!r}")
    return result


def dataset_slug(dataset_name: str) -> str:
    """Return a stable, filesystem-safe slug for a dataset identity."""
    return _safe_slug(dataset_name, label="dataset")


def dataset_slug_from_input_path(input_path: str | Path) -> str:
    """Derive a dataset slug from an input filename stem."""
    path = Path(input_path)
    if not path.name or not path.stem:
        raise ValueError("input_path must have a non-empty filename stem")
    return dataset_slug(path.stem)


def model_artifact_root(
    model_name: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> Path:
    """Return ``artifacts/<model-slug>`` for a model identity."""
    return Path(artifact_root) / model_slug(model_name)


def jacobian_lens_root(
    model_name: str, *, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT
) -> Path:
    """Return the model's Jacobian-lens directory."""
    return lens_artifact_root(model_name, artifact_root=artifact_root)


def jacobian_lens_path(
    model_name: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    filename: str = "jacobian_lens.pt",
) -> Path:
    """Return the canonical Jacobian-lens file path for a model."""
    if Path(filename).name != filename or not filename:
        raise ValueError("filename must be a single non-empty file name")
    if filename == "jacobian_lens.pt":
        return canonical_lens_path(model_name, artifact_root=artifact_root)
    return jacobian_lens_root(model_name, artifact_root=artifact_root) / filename


def dataset_artifact_root(
    model_name: str,
    dataset_name: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return ``artifacts/<model-slug>/<dataset-slug>``."""
    return model_artifact_root(model_name, artifact_root=artifact_root) / dataset_slug(
        dataset_name
    )


def run_root(
    model_name: str,
    dataset_name: str,
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return ``artifacts/<model-slug>/<dataset-slug>/runs/<run-id>``."""
    if not isinstance(run_id, str) or not run_id.strip() or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one non-empty directory name")
    run_slug = _safe_slug(run_id, label="run")
    if run_slug in {".", ".."}:
        raise ValueError("run_id cannot be a traversal component")
    return dataset_artifact_root(
        model_name, dataset_name, artifact_root=artifact_root
    ) / "runs" / run_slug


def run_manifest_path(run_directory: str | Path) -> Path:
    """Return the manifest location inside a run directory."""
    return Path(run_directory) / "manifest.json"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and stable record IDs."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    return sha256_bytes(canonical_json_bytes(value))


def stable_record_id(*parts: Any) -> str:
    """Return a deterministic ``record_`` ID with 24 lowercase hex digits.

    Callers may pass one mapping/list or several identity components.  Components
    are kept separate in the canonical array, so ``("ab", "c")`` cannot collide
    with ``("a", "bc")``.
    """
    return "record_" + sha256_json(list(parts))[:24]


# Short aliases make the contract convenient to use without duplicating hashing
# implementations in producers.
record_id = stable_record_id
file_sha256 = sha256_file


def _atomic_replace(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(
    path: str | Path, value: Any, *, indent: int | None = 2
) -> Path:
    """Atomically write one JSON document and return its destination."""
    if indent is None:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    else:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False
        )
    return _atomic_replace(Path(path), (text + "\n").encode("utf-8"))


def atomic_write_jsonl(
    path: str | Path, records: Iterable[Any]
) -> int:
    """Atomically write JSONL records and return the number of records written."""
    lines: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("JSONL records must be mappings")
        lines.append(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    _atomic_replace(Path(path), ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return len(lines)


# Explicit verb aliases are useful at call sites and preserve one implementation.
write_json_atomic = atomic_write_json
write_jsonl_atomic = atomic_write_jsonl


def count_jsonl_records(path: str | Path) -> int:
    """Count non-empty JSONL lines, validating that each line is JSON."""
    count = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            count += 1
    return count
