"""Fail-closed parent and prepared-artifact checks for the entity-cell workflow."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.artifacts.lifecycle import ArtifactRun


def load_complete_run(path: str | Path, *, label: str) -> tuple[ArtifactRun, dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{label} is missing manifest.json")
    run = ArtifactRun.open(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run.status != "complete":
        raise ValueError(f"{label} must be a complete run")
    if payload.get("status") != "complete":
        raise ValueError(f"{label} manifest status is not complete")
    return run, payload


def require_stage(payload: Mapping[str, Any], stage: str, *, label: str) -> None:
    status = payload.get("stages", {}).get(stage, {}).get("status")
    if status != "complete":
        raise ValueError(f"{label} requires completed stage {stage!r}")


def require_file(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise ValueError(f"{label} is missing: {value}")
    return value


def verify_registered_artifact(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    artifact_type: str,
    label: str,
) -> dict[str, Any]:
    target = Path(path)
    for ref in payload.get("artifacts", []):
        if ref.get("artifact_type") == artifact_type and (Path(str(ref.get("path"))) == target or Path(str(ref.get("path"))).name == target.name):
            if target.is_file() and ref.get("sha256") != sha256_file(target):
                raise ValueError(f"{label} hash does not match its parent manifest")
            if ref.get("status") != "complete":
                raise ValueError(f"{label} is not complete in its parent manifest")
            return dict(ref)
    raise ValueError(f"{label} is not registered in its parent manifest")


def validate_prepared_directory(prepared_dir: str | Path) -> dict[str, Any]:
    """Validate the immutable preparation run and all hashes used downstream."""
    root = Path(prepared_dir)
    run, manifest = load_complete_run(root, label="prepared input run")
    require_stage(manifest, "prepare", label="prepared input run")
    metadata_path = require_file(root / "prepare" / "metadata.json", label="prepared metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping) or metadata.get("artifact_type") != "entity_cell_prepare_metadata":
        raise ValueError("prepared metadata artifact type is invalid")
    for name, artifact_type in (
        ("header_variants.jsonl", "entity_cell_header_variant"),
        ("generic_baseline.jsonl", "entity_cell_generic_baseline_prompt"),
        ("financial_prompts.jsonl", "entity_cell_financial_prompt"),
        ("e2_donor_contracts.jsonl", "entity_cell_e2_donor_contract"),
    ):
        path = require_file(root / "prepare" / name, label=f"prepared {name}")
        verify_registered_artifact(manifest, path, artifact_type=artifact_type, label=f"prepared {name}")
        recorded = metadata.get("prepared_artifact_sha256", {}).get(name.removesuffix(".jsonl"))
        if recorded is not None and recorded != sha256_file(path):
            raise ValueError(f"prepared {name} hash does not match metadata")
    return dict(metadata)


def check_provenance(metadata: Mapping[str, Any], *, model: str | None = None, split_manifest_sha256: str | None = None, config_sha256: str | None = None, tokenizer_identity: str | None = None) -> None:
    if model is not None and str(metadata.get("model")) != str(model):
        raise ValueError("model provenance does not match upstream artifact")
    if split_manifest_sha256 is not None and metadata.get("split_manifest_sha256") != split_manifest_sha256:
        raise ValueError("split manifest provenance does not match upstream artifact")
    if config_sha256 is not None and metadata.get("config_sha256") != config_sha256:
        raise ValueError("config provenance does not match upstream artifact")
    if tokenizer_identity is not None:
        tokenizer = metadata.get("tokenizer", {})
        actual = tokenizer.get("identity_sha256") if isinstance(tokenizer, Mapping) else metadata.get("tokenizer_sha256")
        if actual != tokenizer_identity and metadata.get("tokenizer_sha256") != tokenizer_identity:
            raise ValueError("tokenizer provenance does not match upstream artifact")


__all__ = ["check_provenance", "load_complete_run", "require_file", "require_stage", "validate_prepared_directory", "verify_registered_artifact"]
