"""Shared artifact materialization and run lifecycle contract."""
from .io import *
from .lifecycle import ArtifactContext, ArtifactRun, ArtifactRunContext, RunContext, Stage, StageContext, run_context

__all__ = ["ArtifactContext", "ArtifactRun", "ArtifactRunContext", "RunContext", "Stage", "StageContext", "run_context", "write_json", "write_jsonl", "write_csv", "read_jsonl", "write_metadata", "sidecar_metadata", "sidecar_artifact_hash", "load_parent_jsonl", "verified_artifact_ref", "verify_artifact_ref"]
