from __future__ import annotations

import json

import pytest

from llm_bias.core.artifact_paths import (
    atomic_write_json,
    atomic_write_jsonl,
    count_jsonl_records,
    dataset_artifact_root,
    dataset_slug,
    dataset_slug_from_input_path,
    jacobian_lens_path,
    model_artifact_root,
    run_root,
    sha256_file,
    sha256_json,
    stable_record_id,
)


def test_model_and_dataset_paths_are_deterministic(tmp_path):
    assert model_artifact_root("Qwen/Qwen3.5-4B", artifact_root=tmp_path) == (
        tmp_path / "Qwen--Qwen3.5-4B"
    )
    assert dataset_slug("return-pairs/v1") == "return-pairs--v1"
    assert dataset_slug_from_input_path("/tmp/My Dataset.v1.csv") == "My-Dataset.v1"
    assert jacobian_lens_path("Qwen/Qwen3.5-4B", artifact_root=tmp_path) == (
        tmp_path / "Qwen--Qwen3.5-4B" / "jacobian-lens" / "jacobian_lens.pt"
    )
    assert run_root("Qwen/Qwen3.5-4B", "dataset-a", "run-001", artifact_root=tmp_path) == (
        tmp_path / "Qwen--Qwen3.5-4B" / "dataset-a" / "runs" / "run-001"
    )


def test_local_model_path_uses_final_component(tmp_path):
    model = tmp_path / "weights" / "model"
    model.mkdir(parents=True)
    assert model_artifact_root(str(model), artifact_root=tmp_path) == tmp_path / "model"


def test_stable_ids_and_json_hashes_are_order_independent():
    left = {"prompt": "x", "row": 1}
    right = {"row": 1, "prompt": "x"}
    assert stable_record_id(left) == stable_record_id(right)
    assert stable_record_id(left) == "record_8a703649c0b30516a16cb8bc"
    assert stable_record_id(left).startswith("record_")
    assert len(stable_record_id(left)) == len("record_") + 24
    assert stable_record_id("ab", "c") != stable_record_id("a", "bc")
    assert sha256_json(left) == sha256_json(right)


def test_atomic_json_and_jsonl_writes_are_replaced(tmp_path):
    document = tmp_path / "nested" / "manifest.json"
    assert atomic_write_json(document, {"b": 2, "a": 1}) == document
    assert json.loads(document.read_text()) == {"a": 1, "b": 2}
    records = tmp_path / "nested" / "records.jsonl"
    assert atomic_write_jsonl(records, [{"id": 1}, {"id": 2}]) == 2
    assert count_jsonl_records(records) == 2
    digest = sha256_file(records)
    atomic_write_jsonl(records, [{"id": 3}])
    assert count_jsonl_records(records) == 1
    assert sha256_file(records) != digest


def test_invalid_identity_is_rejected():
    with pytest.raises(ValueError):
        dataset_slug("")
    with pytest.raises(ValueError):
        run_root("model", "dataset", "../escape")
