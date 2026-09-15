"""Strict verification of completed run inputs against a registered manifest."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import file_sha256
from llm_bias.core.artifacts.io import read_jsonl

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def _finite(value: Any, *, field: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite float at {field}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _finite(nested, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _finite(nested, field=f"{field}[{index}]")


def _parse_json(path: Path, *, field: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant at {field}: {value}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {field}: {path}") from exc
    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
            raise
        raise ValueError(f"invalid JSON in {field}: {path}") from exc
    _finite(value, field=field)
    return value


def _relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    if "\\" in value or "//" in value or value.startswith("/"):
        raise ValueError(f"{field} is not a normalized POSIX relative path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field} is not a normalized POSIX relative path: {value!r}")
    return value


def _positive_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _resolve_inside(root: Path, relative: str, *, field: str) -> Path:
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes run root: {relative!r}") from exc
    return resolved


def _validate_ref_path(ref: Any, *, index: int, root: Path) -> str:
    if not isinstance(ref, Mapping):
        raise ValueError(f"output_refs[{index}] must be an object")
    relative = _relative_path(ref.get("path"), field=f"output_refs[{index}].path")
    _resolve_inside(root, relative, field=f"output_refs[{index}].path")
    _digest(ref.get("sha256"), field=f"output_refs[{index}].sha256")
    return relative


def _validate_required_spec(required: Mapping[str, tuple[str, int | None]]) -> list[tuple[str, str, int | None]]:
    if not isinstance(required, Mapping) or not required:
        raise ValueError("required must be a non-empty mapping")
    validated: list[tuple[str, str, int | None]] = []
    for path_value, expected in required.items():
        relative = _relative_path(path_value, field="required path")
        suffix = Path(relative).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError(f"required path has unsupported extension: {relative!r}")
        if not isinstance(expected, tuple) or len(expected) != 2:
            raise ValueError(f"required[{relative!r}] must be (sha256, record_count)")
        expected_sha, expected_count = expected
        digest = _digest(expected_sha, field=f"required[{relative}].sha256")
        if suffix == ".json":
            if expected_count is not None:
                raise ValueError(f"required[{relative}].record_count must be None for JSON")
        else:
            _positive_count(expected_count, field=f"required[{relative}].record_count")
        validated.append((relative, digest, expected_count))
    return validated


def _parse_jsonl(path: Path, *, field: str) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl(path)
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {field}: {path}") from exc
    for index, row in enumerate(rows):
        _finite(row, field=f"{field}[{index}]")
    return rows


def verify_completed_inputs(
    run_root: str | Path,
    *,
    dataset: str,
    model: str,
    run_id: str,
    expected_manifest_sha256: str,
    required: Mapping[str, tuple[str, int | None]],
) -> dict[str, dict[str, str | int]]:
    """Verify selected completed outputs registered by a run manifest.

    Only ``output_refs`` in the top-level manifest are considered.  The function
    validates and hashes the required files, while deliberately leaving unrelated
    output files unopened.
    """
    expected_manifest_sha256 = _digest(
        expected_manifest_sha256, field="expected_manifest_sha256"
    )
    required_specs = _validate_required_spec(required)

    root = Path(run_root).resolve(strict=False)
    manifest_path = root / "manifest.json"
    actual_manifest_sha256 = file_sha256(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "manifest sha256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )

    manifest = _parse_json(manifest_path, field="manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ValueError("manifest.schema_version must be integer 1")
    if manifest.get("status") != "complete":
        raise ValueError(f"manifest.status must be complete, got {manifest.get('status')!r}")
    for field, expected in (("dataset", dataset), ("model", model), ("run_id", run_id)):
        if manifest.get(field) != expected:
            raise ValueError(
                f"manifest.{field} mismatch: expected {expected!r}, got {manifest.get(field)!r}"
            )

    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("manifest.stages must be an object")
    output_refs = manifest.get("output_refs")
    if not isinstance(output_refs, list):
        raise ValueError("manifest.output_refs must be a list")

    refs_by_path: dict[str, Mapping[str, Any]] = {}
    for index, ref in enumerate(output_refs):
        relative = _validate_ref_path(ref, index=index, root=root)
        if relative in refs_by_path:
            raise ValueError(f"duplicate output_refs path: {relative}")
        refs_by_path[relative] = ref

    result: dict[str, dict[str, str | int]] = {}
    for relative, expected_sha256, expected_count in required_specs:
        ref = refs_by_path.get(relative)
        if ref is None:
            raise ValueError(f"required path is not registered in output_refs: {relative}")
        if ref.get("role") != "output":
            raise ValueError(f"output_refs path has invalid role: {relative}")
        if ref.get("status") != "complete":
            raise ValueError(f"output_refs path has invalid status: {relative}")

        stage = ref.get("stage")
        if not isinstance(stage, str) or not stage:
            raise ValueError(f"output_refs[{relative}].stage must be non-empty")
        stage_record = stages.get(stage)
        if not isinstance(stage_record, Mapping) or stage_record.get("status") != "complete":
            raise ValueError(f"stage {stage!r} for {relative} is not complete")

        ref_sha256 = _digest(ref.get("sha256"), field=f"output_refs[{relative}].sha256")
        if ref_sha256 != expected_sha256:
            raise ValueError(
                f"sha256 mismatch for {relative}: expected {expected_sha256}, got ref {ref_sha256}"
            )

        suffix = Path(relative).suffix.lower()
        ref_count = ref.get("record_count")
        if suffix == ".json":
            if expected_count is not None or ref_count is not None:
                raise ValueError(f"record_count must be absent or None for JSON: {relative}")
        else:
            _positive_count(ref_count, field=f"output_refs[{relative}].record_count")
            if ref_count != expected_count:
                raise ValueError(
                    f"record_count mismatch for {relative}: expected {expected_count}, got {ref_count}"
                )

        resolved = _resolve_inside(root, relative, field=relative)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual_sha256 = file_sha256(resolved)
        if actual_sha256 != expected_sha256 or actual_sha256 != ref_sha256:
            raise ValueError(
                f"sha256 mismatch for {relative}: expected {expected_sha256}, got {actual_sha256}"
            )

        if suffix == ".json":
            payload = _parse_json(resolved, field=relative)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSON artifact must be an object: {relative}")
        else:
            rows = _parse_jsonl(resolved, field=relative)
            if len(rows) != expected_count or len(rows) != ref_count:
                raise ValueError(
                    f"record_count mismatch for {relative}: expected {expected_count}, got {len(rows)}"
                )

        item: dict[str, str | int] = {
            "path": str(resolved),
            "sha256": actual_sha256,
            "stage": stage,
        }
        if suffix == ".jsonl":
            item["record_count"] = len(rows)
        result[relative] = item

    return result
