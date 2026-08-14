"""Versioned run-manifest lifecycle for compact experiment artifacts.

A manifest records identities, hashes, references, stages, and counts.  It never
contains raw activation or gradient payloads; those data are outside this
repository's artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from llm_bias.core.artifact_paths import (
    DEFAULT_ARTIFACT_ROOT,
    atomic_write_json,
    count_jsonl_records,
    dataset_slug,
    file_sha256,
    model_slug,
    run_manifest_path,
    run_root,
)

SCHEMA_VERSION = 1
ArtifactRole = Literal["input", "lens", "output"]
_TERMINAL_STATUSES = {"complete", "failed"}
_RAW_PAYLOAD_RE = re.compile(
    r"(?:^|[_-])(?:raw[_-]?)?(?:activation|activations|gradient|gradients)(?:$|[_-])",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_status(status: str) -> str:
    if status not in {"created", "running", "complete", "failed"}:
        raise ValueError(f"unsupported run status: {status!r}")
    return status


def _relative_ref(path: Path, run_directory: Path) -> str:
    try:
        return path.relative_to(run_directory).as_posix()
    except ValueError:
        return path.as_posix()


def _reject_raw_payload_type(artifact_type: str) -> None:
    normalized = artifact_type.lower().replace(" ", "_")
    if _RAW_PAYLOAD_RE.search(normalized):
        raise ValueError("raw activation/gradient artifacts are not supported")


@dataclass
class RunManifest:
    """Mutable manifest for one model/dataset/run identity."""

    model: str
    dataset: str
    run_id: str
    run_directory: Path
    schema_version: int = SCHEMA_VERSION
    status: str = "created"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    input_refs: list[dict[str, Any]] = field(default_factory=list)
    lens_refs: list[dict[str, Any]] = field(default_factory=list)
    output_refs: list[dict[str, Any]] = field(default_factory=list)
    record_counts: dict[str, int] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        self.model = str(self.model)
        self.dataset = str(self.dataset)
        self.run_id = str(self.run_id)
        self.run_directory = Path(self.run_directory)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version: {self.schema_version}")
        self.status = _validate_status(self.status)
        if not self.model.strip() or not self.dataset.strip() or not self.run_id.strip():
            raise ValueError("model, dataset, and run_id must be non-empty")

    @classmethod
    def new(
        cls,
        model: str,
        dataset: str,
        run_id: str,
        *,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ) -> "RunManifest":
        """Build an unsaved manifest at the canonical run root."""
        return cls(
            model=model,
            dataset=dataset,
            run_id=run_id,
            run_directory=run_root(
                model, dataset, run_id, artifact_root=artifact_root
            ),
        )

    @classmethod
    def create(
        cls,
        model: str,
        dataset: str,
        run_id: str,
        *,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ) -> "RunManifest":
        """Build and atomically persist a new manifest."""
        manifest = cls.new(model, dataset, run_id, artifact_root=artifact_root)
        if manifest.run_directory.exists() or manifest.manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {manifest.run_directory}")
        manifest.run_directory.mkdir(parents=True, exist_ok=False)
        manifest.save()
        return manifest

    @property
    def manifest_path(self) -> Path:
        return run_manifest_path(self.run_directory)

    @property
    def model_slug(self) -> str:
        return model_slug(self.model)

    @property
    def dataset_slug(self) -> str:
        return dataset_slug(self.dataset)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation of this manifest."""
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "model_slug": self.model_slug,
            "dataset": self.dataset,
            "dataset_slug": self.dataset_slug,
            "run_id": self.run_id,
            "run_root": self.run_directory.as_posix(),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": list(self.artifacts),
            "input_refs": list(self.input_refs),
            "lens_refs": list(self.lens_refs),
            "output_refs": list(self.output_refs),
            "record_counts": dict(self.record_counts),
            "stages": dict(self.stages),
            "error": self.error,
        }

    def save(self) -> Path:
        """Atomically persist the manifest and return its path."""
        self.updated_at = _utc_now()
        return atomic_write_json(self.manifest_path, self.to_dict())

    def start(self) -> "RunManifest":
        """Mark the run as running."""
        if self.status in _TERMINAL_STATUSES:
            raise ValueError(f"cannot start a {self.status} run")
        self.status = "running"
        self.error = None
        return self

    def complete(self) -> "RunManifest":
        """Mark the run as completed after all registrations are present."""
        if self.status == "failed":
            raise ValueError("cannot complete a failed run")
        self.status = "complete"
        self.error = None
        return self

    def fail(self, error: str) -> "RunManifest":
        """Mark the run as failed with a concise diagnostic."""
        if not str(error).strip():
            raise ValueError("error must be non-empty")
        self.status = "failed"
        self.error = str(error)
        return self

    def start_stage(self, stage: str) -> "RunManifest":
        if not str(stage).strip():
            raise ValueError("stage must be non-empty")
        self.stages[str(stage)] = {"status": "running", "started_at": _utc_now()}
        return self

    def finish_stage(
        self,
        stage: str,
        *,
        status: str = "complete",
        record_count: int | None = None,
    ) -> "RunManifest":
        if status not in {"complete", "failed"}:
            raise ValueError("stage status must be complete or failed")
        current = dict(self.stages.get(str(stage), {}))
        current.update({"status": status, "finished_at": _utc_now()})
        if record_count is not None:
            _validate_record_count(record_count)
            current["record_count"] = record_count
        self.stages[str(stage)] = current
        return self

    def register_artifact(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        stage: str,
        role: ArtifactRole = "output",
        status: str = "complete",
        record_count: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register one hashed input, lens, or output artifact.

        ``path`` must exist unless a caller supplies a precomputed ``sha256`` via
        ``metadata``.  Registration stores references and compact metadata only.
        """
        if role not in {"input", "lens", "output"}:
            raise ValueError("role must be input, lens, or output")
        if not str(artifact_type).strip() or not str(stage).strip():
            raise ValueError("artifact_type and stage must be non-empty")
        _reject_raw_payload_type(str(artifact_type))
        if status not in {"pending", "complete", "failed"}:
            raise ValueError("artifact status must be pending, complete, or failed")
        if record_count is not None:
            _validate_record_count(record_count)
        artifact_path = Path(path)
        supplied_metadata = dict(metadata or {})
        supplied_hash = supplied_metadata.pop("sha256", None)
        if supplied_hash is not None and (
            not isinstance(supplied_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", supplied_hash)
        ):
            raise ValueError("metadata sha256 must be a lowercase 64-character digest")
        if artifact_path.is_file():
            digest = file_sha256(artifact_path)
            if supplied_hash is not None and supplied_hash != digest:
                raise ValueError(
                    f"artifact sha256 mismatch: expected {supplied_hash}, got {digest}"
                )
        elif supplied_hash is not None:
            digest = supplied_hash
        else:
            raise FileNotFoundError(f"artifact does not exist: {artifact_path}")
        if record_count is None and artifact_path.suffix.lower() == ".jsonl" and artifact_path.is_file():
            record_count = count_jsonl_records(artifact_path)
        ref: dict[str, Any] = {
            "artifact_type": str(artifact_type),
            "stage": str(stage),
            "status": status,
            "role": role,
            "path": _relative_ref(artifact_path, self.run_directory),
            "sha256": digest,
        }
        if record_count is not None:
            ref["record_count"] = record_count
        if supplied_metadata:
            ref["metadata"] = supplied_metadata
        # Re-registering a path updates it rather than creating ambiguous refs.
        self.artifacts = [item for item in self.artifacts if item.get("path") != ref["path"]]
        self.artifacts.append(ref)
        role_refs = (self.input_refs, self.lens_refs, self.output_refs)
        for refs in role_refs:
            refs[:] = [item for item in refs if item.get("path") != ref["path"]]
        refs = {"input": self.input_refs, "lens": self.lens_refs, "output": self.output_refs}[role]
        refs.append(dict(ref))
        self.record_counts = {}
        for item in self.artifacts:
            if item.get("record_count") is not None:
                key = str(item["artifact_type"])
                self.record_counts[key] = self.record_counts.get(key, 0) + int(item["record_count"])
        return ref

    def register(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias for :meth:`register_artifact`."""
        return self.register_artifact(*args, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        """Load and validate a previously persisted manifest."""
        import json

        manifest_path = Path(path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("run manifest must be a JSON object")
        required = {"schema_version", "model", "dataset", "run_id", "status"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"run manifest missing fields: {sorted(missing)}")
        return cls(
            model=data["model"],
            dataset=data["dataset"],
            run_id=data["run_id"],
            run_directory=manifest_path.parent,
            schema_version=int(data["schema_version"]),
            status=data["status"],
            created_at=data.get("created_at", _utc_now()),
            updated_at=data.get("updated_at", _utc_now()),
            artifacts=list(data.get("artifacts", [])),
            input_refs=list(data.get("input_refs", [])),
            lens_refs=list(data.get("lens_refs", [])),
            output_refs=list(data.get("output_refs", [])),
            record_counts={str(k): int(v) for k, v in data.get("record_counts", {}).items()},
            stages=dict(data.get("stages", {})),
            error=data.get("error"),
        )


def _validate_record_count(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("record_count must be a non-negative integer")


def create_run_manifest(
    model: str,
    dataset: str,
    run_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> RunManifest:
    """Create and persist a manifest at the canonical run root."""
    return RunManifest.create(model, dataset, run_id, artifact_root=artifact_root)


def register_artifact(
    manifest: RunManifest,
    path: str | Path,
    *,
    artifact_type: str,
    stage: str,
    role: ArtifactRole = "output",
    status: str = "complete",
    record_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Functional wrapper around :meth:`RunManifest.register_artifact`."""
    return manifest.register_artifact(
        path,
        artifact_type=artifact_type,
        stage=stage,
        role=role,
        status=status,
        record_count=record_count,
        metadata=metadata,
    )


ArtifactManifest = RunManifest
