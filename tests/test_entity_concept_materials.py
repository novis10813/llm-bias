"""Tests for strict, model-free concept material loading."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from llm_bias.entity_concept_decision.materials import load_material_bundle


def _payload() -> dict:
    return {
        "schema_version": 1,
        "concepts": [
            {
                "concept_id": "c1",
                "definition": "A synthetic test concept.",
                "positive_pole": "mountain",
                "negative_pole": "coast",
                "excluded_interpretations": ["general sentiment"],
            }
        ],
        "pairs": [
            {
                "id": "p1",
                "concept_id": "c1",
                "family_id": "fit-a",
                "split": "fit",
                "text_positive": "The setting is mountainous.",
                "text_negative": "The setting is coastal.",
                "review_status": "approved",
                "confound_tags": [],
            },
            {
                "id": "p2",
                "concept_id": "c1",
                "family_id": "fit-a",
                "split": "fit",
                "text_positive": "A mountain landscape appears.",
                "text_negative": "A coastal landscape appears.",
                "review_status": "approved",
                "confound_tags": ["grammar"],
            },
            {
                "id": "p3",
                "concept_id": "c1",
                "family_id": "fit-b",
                "split": "fit",
                "text_positive": "This place has steep mountains.",
                "text_negative": "This place has a broad coast.",
                "review_status": "approved",
                "confound_tags": [],
            },
            {
                "id": "p4",
                "concept_id": "c1",
                "family_id": "validation-a",
                "split": "validation",
                "text_positive": "Validation mountain sentence.",
                "text_negative": "Validation coast sentence.",
                "review_status": "approved",
                "confound_tags": [],
            },
            {
                "id": "p5",
                "concept_id": "c1",
                "family_id": "audit-a",
                "split": "audit",
                "text_positive": "Audit mountain sentence.",
                "text_negative": "Audit coast sentence.",
                "review_status": "approved",
                "confound_tags": [],
            },
        ],
    }


def _write_materials(tmp_path: Path, payload: object, *, raw: str | None = None) -> tuple[Path, str]:
    path = tmp_path / "materials.json"
    data = raw.encode("utf-8") if raw is not None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_load_material_bundle_validates_hash_and_preserves_schema(tmp_path: Path) -> None:
    path, digest = _write_materials(tmp_path, _payload())

    bundle = load_material_bundle(path, expected_sha256=digest)

    assert bundle.source_sha256 == digest
    assert bundle.concepts[0].concept_id == "c1"
    assert len(bundle.pairs) == 5
    assert bundle.pairs[1].confound_tags == ("grammar",)


def test_material_loader_rejects_wrong_hash_and_hash_format(tmp_path: Path) -> None:
    path, digest = _write_materials(tmp_path, _payload())
    with pytest.raises(ValueError, match="hash mismatch"):
        load_material_bundle(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="64 lowercase"):
        load_material_bundle(path, expected_sha256=digest.upper())


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda p: p["concepts"][0].update({"unexpected": 1}), "unknown field"),
        (lambda p: p["pairs"][0].update({"review_status": "pending"}), "approved"),
        (lambda p: p["pairs"][0].update({"split": "holdout"}), "split"),
        (lambda p: p["pairs"][0].update({"concept_id": "missing"}), "unknown concept"),
        (lambda p: p["concepts"][0].update({"excluded_interpretations": []}), "must not be empty"),
    ],
)
def test_material_loader_rejects_unknown_or_unapproved_material(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = _payload()
    mutation(payload)
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match=message):
        load_material_bundle(path, expected_sha256=digest)


def test_material_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    raw = json.dumps(_payload()).replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1)
    path, digest = _write_materials(tmp_path, _payload(), raw=raw)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_material_bundle(path, expected_sha256=digest)


def test_material_loader_rejects_family_cross_split_and_missing_split(tmp_path: Path) -> None:
    payload = _payload()
    payload["pairs"][3]["family_id"] = "fit-a"
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match="multiple splits"):
        load_material_bundle(path, expected_sha256=digest)

    payload = _payload()
    payload["pairs"] = payload["pairs"][:-1]
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match="audit"):
        load_material_bundle(path, expected_sha256=digest)


def test_material_loader_rejects_normalized_duplicate_across_splits(tmp_path: Path) -> None:
    payload = _payload()
    payload["pairs"][3]["text_positive"] = "THE SETTING IS MOUNTAINOUS."
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match="normalized material text"):
        load_material_bundle(path, expected_sha256=digest)


def test_material_loader_rejects_nonfinite_json(tmp_path: Path) -> None:
    payload = _payload()
    payload["pairs"][0]["confound_tags"] = [float("nan")]
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match="non-finite"):
        load_material_bundle(path, expected_sha256=digest)


def test_material_loader_rejects_duplicate_ids_and_invalid_text(tmp_path: Path) -> None:
    payload = _payload()
    payload["pairs"][1]["id"] = "p1"
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match="duplicate pair id"):
        load_material_bundle(path, expected_sha256=digest)

    payload = _payload()
    payload["pairs"][1]["text_negative"] = payload["pairs"][1]["text_positive"]
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match="must differ"):
        load_material_bundle(path, expected_sha256=digest)

    payload = _payload()
    payload["concepts"][0]["positive_pole"] = " mountain"
    path, digest = _write_materials(tmp_path, payload)
    with pytest.raises(ValueError, match="surrounding whitespace"):
        load_material_bundle(path, expected_sha256=digest)


def test_material_loader_does_not_mutate_source_payload(tmp_path: Path) -> None:
    payload = _payload()
    before = copy.deepcopy(payload)
    path, digest = _write_materials(tmp_path, payload)
    load_material_bundle(path, expected_sha256=digest)
    assert payload == before
