"""Backward-compatible facade for shared artifact I/O."""
from pathlib import Path
from llm_bias.core.artifacts.io import (
    load_parent_jsonl,
    read_jsonl,
    sidecar_artifact_hash,
    sidecar_metadata,
    sha256_file,
    write_jsonl as _write_jsonl,
    write_metadata as _write_metadata,
)

def write_jsonl(path, rows):
    """Preserve prompt-analysis rerun semantics on the shared safe writer."""
    _write_jsonl(path, rows, overwrite=True)
    return Path(path)


def write_metadata(path, metadata):
    """Replace prompt-analysis sidecars when an existing run is recomputed."""
    return _write_metadata(path, metadata, overwrite=True)

ARTIFACT_SCHEMA_VERSION = 1
__all__ = ["ARTIFACT_SCHEMA_VERSION", "read_jsonl", "write_jsonl", "load_parent_jsonl", "sidecar_metadata", "sidecar_artifact_hash", "sha256_file", "write_metadata"]
