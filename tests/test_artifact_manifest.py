from __future__ import annotations

import json

import pytest

from llm_bias.core.artifact_manifest import (
    SCHEMA_VERSION,
    RunManifest,
    create_run_manifest,
    register_artifact,
)
from llm_bias.core.artifact_paths import atomic_write_jsonl


def test_manifest_lifecycle_registers_refs_hashes_and_counts(tmp_path):
    manifest = create_run_manifest(
        "Qwen/Qwen3.5-4B", "return-pairs", "run-001", artifact_root=tmp_path
    )
    assert manifest.manifest_path.is_file()
    assert manifest.to_dict()["schema_version"] == SCHEMA_VERSION
    assert manifest.status == "created"

    inputs = manifest.run_directory / "inputs.jsonl"
    outputs = manifest.run_directory / "readout.jsonl"
    atomic_write_jsonl(inputs, [{"record_id": "a"}, {"record_id": "b"}])
    atomic_write_jsonl(outputs, [{"record_id": "a"}, {"record_id": "b"}])
    lens = tmp_path / "Qwen--Qwen3.5-4B" / "jacobian-lens" / "jacobian_lens.pt"
    lens.parent.mkdir(parents=True)
    lens.write_bytes(b"lens")

    manifest.start().start_stage("readout")
    input_ref = register_artifact(
        manifest,
        inputs,
        artifact_type="return_pairs_input",
        stage="readout",
        role="input",
    )
    register_artifact(
        manifest,
        lens,
        artifact_type="jacobian_lens",
        stage="readout",
        role="lens",
    )
    output_ref = manifest.register_artifact(
        outputs,
        artifact_type="prompt_readout",
        stage="readout",
        role="output",
    )
    manifest.finish_stage("readout", record_count=2).complete().save()

    loaded = RunManifest.load(manifest.manifest_path)
    payload = loaded.to_dict()
    assert payload["status"] == "complete"
    assert payload["input_refs"][0]["sha256"] == input_ref["sha256"]
    assert payload["lens_refs"][0]["artifact_type"] == "jacobian_lens"
    assert payload["output_refs"][0]["sha256"] == output_ref["sha256"]
    assert payload["record_counts"] == {
        "return_pairs_input": 2,
        "prompt_readout": 2,
    }
    assert payload["stages"]["readout"]["record_count"] == 2


def test_two_datasets_have_independent_run_roots(tmp_path):
    first = create_run_manifest("model", "dataset-a", "same-run", artifact_root=tmp_path)
    second = create_run_manifest("model", "dataset-b", "same-run", artifact_root=tmp_path)
    assert first.run_directory != second.run_directory
    assert first.manifest_path.is_file()
    assert second.manifest_path.is_file()
    assert json.loads(first.manifest_path.read_text())["dataset_slug"] == "dataset-a"
    assert json.loads(second.manifest_path.read_text())["dataset_slug"] == "dataset-b"


def test_manifest_rejects_raw_activation_and_missing_artifact(tmp_path):
    manifest = RunManifest.new("model", "dataset", "run", artifact_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        manifest.register_artifact(
            tmp_path / "missing.jsonl",
            artifact_type="output",
            stage="readout",
        )
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"not allowed")
    with pytest.raises(ValueError):
        manifest.register_artifact(
            raw,
            artifact_type="raw_activations",
            stage="readout",
        )


def test_reregistering_path_removes_previous_role_reference(tmp_path):
    manifest = RunManifest.new("model", "dataset", "run", artifact_root=tmp_path)
    artifact = manifest.run_directory / "artifact.jsonl"
    atomic_write_jsonl(artifact, [{"record_id": "a"}])

    manifest.register_artifact(
        artifact,
        artifact_type="prompt_input",
        stage="readout",
        role="input",
    )
    manifest.register_artifact(
        artifact,
        artifact_type="prompt_output",
        stage="readout",
        role="output",
    )

    assert not manifest.input_refs
    assert len(manifest.output_refs) == 1
    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0]["role"] == "output"


def test_failed_run_preserves_diagnostic(tmp_path):
    manifest = RunManifest.new("model", "dataset", "run", artifact_root=tmp_path)
    manifest.start().fail("readout failed").save()
    loaded = RunManifest.load(manifest.manifest_path)
    assert loaded.status == "failed"
    assert loaded.error == "readout failed"
