"""Compatibility facade for synthetic pilot artifact materialization."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
from llm_bias.core.artifact_manifest import RunManifest
from llm_bias.core.artifacts.io import write_csv as _write_csv, write_json as _write_json

def write_json(path: str | Path, value: Any) -> Path:
    _write_json(path, value)
    return Path(path)

def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> Path:
    _write_csv(path, rows, fields)
    return Path(path)

def start_manifest(model, dataset, run_id, artifact_root):
    manifest = RunManifest.create(model, dataset, run_id, artifact_root=artifact_root)
    manifest.start(); manifest.save()
    return manifest

def fail_manifest(manifest, error):
    if manifest.status not in {"complete", "failed"}:
        manifest.fail(str(error)); manifest.save()

def complete_manifest(manifest, *, required_stages: set[str], postcheck: bool):
    if manifest.status != "running": raise ValueError("manifest must be running")
    if not postcheck or any(manifest.stages.get(stage, {}).get("status") != "complete" for stage in required_stages):
        raise ValueError("cannot complete manifest before successful postchecks")
    manifest.complete(); manifest.save()
