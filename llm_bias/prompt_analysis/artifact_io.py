"""Backward-compatible facade for shared artifact I/O."""
from pathlib import Path
from llm_bias.core.artifacts.io import (
    load_parent_jsonl,
    read_jsonl,
    sidecar_artifact_hash,
    sidecar_metadata,
    sha256_file,
    write_jsonl as _write_jsonl,
    write_metadata,
)

def write_jsonl(path, rows):
    _write_jsonl(path, rows)
    return Path(path)

ARTIFACT_SCHEMA_VERSION = 1
__all__ = ["ARTIFACT_SCHEMA_VERSION", "read_jsonl", "write_jsonl", "load_parent_jsonl", "sidecar_metadata", "sidecar_artifact_hash", "sha256_file", "write_metadata"]
