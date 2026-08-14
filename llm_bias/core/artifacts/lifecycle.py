"""Run and stage context for artifact-producing workflows."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from llm_bias.core.artifact_manifest import RunManifest

@dataclass
class StageContext:
    run: "ArtifactRun"
    name: str
    record_count: int | None = None
    _entered: bool = False
    def __enter__(self) -> "StageContext":
        self.run.manifest.start_stage(self.name); self.run.manifest.save(); self._entered = True
        return self
    def count(self, value: int) -> "StageContext":
        self.record_count = value; return self
    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.run.manifest.finish_stage(self.name, record_count=self.record_count)
        else:
            self.run.manifest.finish_stage(self.name, status="failed")
            self.run.manifest.fail(str(exc))
        self.run.manifest.save()
        return False

@dataclass
class ArtifactRun:
    manifest: RunManifest
    @classmethod
    def create(cls, model: str, dataset: str, run_id: str, *, artifact_root: str | Path = "artifacts") -> "ArtifactRun":
        manifest = RunManifest.create(model, dataset, run_id, artifact_root=artifact_root)
        manifest.start(); manifest.save()
        return cls(manifest)
    @classmethod
    def open(cls, manifest: RunManifest | str | Path) -> "ArtifactRun":
        return cls(manifest if isinstance(manifest, RunManifest) else RunManifest.load(manifest))
    @property
    def run_directory(self) -> Path: return self.manifest.run_directory
    @property
    def status(self) -> str: return self.manifest.status
    def save(self) -> "ArtifactRun": self.manifest.save(); return self
    def stage(self, name: str) -> StageContext: return StageContext(self, name)
    def start(self) -> "ArtifactRun": self.manifest.start(); self.manifest.save(); return self
    def fail(self, error: BaseException | str) -> "ArtifactRun":
        self.manifest.fail(str(error)); self.manifest.save(); return self
    def finalize(self, *, required_stages: set[str] | None = None, postcheck: bool = True) -> "ArtifactRun":
        required_stages = required_stages or set(self.manifest.stages)
        if not postcheck or any(self.manifest.stages.get(s, {}).get("status") != "complete" for s in required_stages):
            raise ValueError("cannot finalize manifest before successful postchecks")
        self.manifest.complete(); self.manifest.save(); return self
    complete = finalize

# Names likely used by workflow wrappers.
RunContext = ArtifactRun
ArtifactRunContext = ArtifactRun
ArtifactContext = ArtifactRun
Stage = StageContext

@contextmanager
def run_context(model: str, dataset: str, run_id: str, *, artifact_root: str | Path = "artifacts") -> Iterator[ArtifactRun]:
    run = ArtifactRun.create(model, dataset, run_id, artifact_root=artifact_root)
    try:
        yield run
    except BaseException as exc:
        run.fail(exc)
        raise
