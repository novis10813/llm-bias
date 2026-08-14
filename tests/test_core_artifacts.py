from __future__ import annotations

import pytest

from llm_bias.core.artifacts import (
    ArtifactRun,
    load_parent_jsonl,
    sidecar_artifact_hash,
    verified_artifact_ref,
    verify_artifact_ref,
    write_jsonl,
    write_metadata,
)


def test_streaming_jsonl_and_verified_ref(tmp_path):
    path = tmp_path / "rows.jsonl"
    assert write_jsonl(path, ({"record_id": str(i)} for i in range(3))) == 3
    ref = verified_artifact_ref(path, artifact_type="records", stage="forward")
    assert ref["record_count"] == 3
    assert verify_artifact_ref(ref)


def test_jsonl_no_overwrite_and_parent_hash(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"record_id": "a"}])
    with pytest.raises(FileExistsError):
        write_jsonl(path, [{"record_id": "b"}])
    rows, digest = load_parent_jsonl(path)
    assert rows == [{"record_id": "a"}]
    metadata = tmp_path / "metadata.json"
    write_metadata(metadata, {"artifact_sha256": digest})
    assert sidecar_artifact_hash(path) == digest
    load_parent_jsonl(path, expected_sha256=digest)


def test_run_context_finalizes_and_rejects_reuse(tmp_path):
    run = ArtifactRun.create("model", "dataset", "run", artifact_root=tmp_path)
    with run.stage("forward") as stage:
        stage.count(1)
    run.finalize(required_stages={"forward"})
    assert run.status == "complete"
    with pytest.raises(FileExistsError):
        ArtifactRun.create("model", "dataset", "run", artifact_root=tmp_path)


def test_raw_payload_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "bad.jsonl", [{"activation": [1, 2]}])
