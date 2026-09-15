from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_bias.core.artifacts.verification import verify_completed_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "run"
    (root / "prepare").mkdir(parents=True)
    (root / "analyze").mkdir()
    summary = root / "analyze" / "summary.json"
    rows = root / "prepare" / "rows.jsonl"
    summary.write_text('{"ok": true, "score": 1.25}\n', encoding="utf-8")
    rows.write_text('{"id": "a"}\n{"id": "b", "value": 2}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": "entity-to-dial",
        "model": "qwen3.5-4b",
        "run_id": "run-1",
        "status": "complete",
        "stages": {
            "prepare": {"status": "complete"},
            "analyze": {"status": "complete"},
        },
        "output_refs": [
            {
                "path": "analyze/summary.json",
                "role": "output",
                "status": "complete",
                "stage": "analyze",
                "sha256": _sha256(summary),
            },
            {
                "path": "prepare/rows.jsonl",
                "role": "output",
                "status": "complete",
                "stage": "prepare",
                "sha256": _sha256(rows),
                "record_count": 2,
            },
        ],
        "artifacts": [{"path": "ignored/extra.json", "sha256": "0" * 64}],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, {"manifest": manifest_path, "summary": summary, "rows": rows}


def _verify(root: Path, paths: dict[str, Path]):
    return verify_completed_inputs(
        root,
        dataset="entity-to-dial",
        model="qwen3.5-4b",
        run_id="run-1",
        expected_manifest_sha256=_sha256(paths["manifest"]),
        required={
            "analyze/summary.json": (_sha256(paths["summary"]), None),
            "prepare/rows.jsonl": (_sha256(paths["rows"]), 2),
        },
    )


def _manifest(root: Path) -> Path:
    return root / "manifest.json"


def _mutate_manifest(root: Path, **changes) -> None:
    path = _manifest(root)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_verify_completed_inputs_returns_only_required_compact_results(tmp_path):
    root, paths = _make_run(tmp_path)

    result = _verify(root, paths)

    assert result == {
        "analyze/summary.json": {
            "path": str(paths["summary"].resolve()),
            "sha256": _sha256(paths["summary"]),
            "stage": "analyze",
        },
        "prepare/rows.jsonl": {
            "path": str(paths["rows"].resolve()),
            "sha256": _sha256(paths["rows"]),
            "stage": "prepare",
            "record_count": 2,
        },
    }


def test_extra_unopened_output_does_not_block_required_verification(tmp_path):
    root, paths = _make_run(tmp_path)
    extra = root / "extra.json"
    extra.write_text("not JSON", encoding="utf-8")
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"].append(
        {
            "path": "extra.json",
            "role": "output",
            "status": "complete",
            "stage": "analyze",
            "sha256": "0" * 64,
        }
    )
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")

    assert set(_verify(root, paths)) == {"analyze/summary.json", "prepare/rows.jsonl"}


@pytest.mark.parametrize(
    "relative",
    ["/outside.json", "../outside.json", "prepare/../outside.json", "prepare//rows.jsonl", "./rows.jsonl", "prepare\\rows.jsonl"],
)
def test_invalid_ref_paths_fail_closed(tmp_path, relative):
    root, paths = _make_run(tmp_path)
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"][1]["path"] = relative
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="path"):
        _verify(root, paths)


def test_symlink_escape_fails_closed(tmp_path):
    root, paths = _make_run(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"id": "outside"}\n', encoding="utf-8")
    link = root / "prepare" / "linked.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"][1]["path"] = "prepare/linked.jsonl"
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes run root"):
        _verify(root, paths)


def test_duplicate_ref_and_bad_stage_or_status_fail_closed(tmp_path):
    root, paths = _make_run(tmp_path)
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"].append(dict(data["output_refs"][0]))
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _verify(root, paths)

    root, paths = _make_run(tmp_path / "bad-stage")
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"][1]["stage"] = "missing"
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stage"):
        _verify(root, paths)

    root, paths = _make_run(tmp_path / "bad-status")
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"][1]["status"] = "pending"
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        _verify(root, paths)


def test_manifest_identity_and_digest_mutations_fail_closed(tmp_path):
    root, paths = _make_run(tmp_path)
    with pytest.raises(ValueError, match="manifest sha256 mismatch"):
        verify_completed_inputs(
            root,
            dataset="entity-to-dial",
            model="qwen3.5-4b",
            run_id="run-1",
            expected_manifest_sha256="0" * 64,
            required={"analyze/summary.json": (_sha256(paths["summary"]), None)},
        )

    for field in ("schema_version", "status", "dataset", "model", "run_id"):
        root, paths = _make_run(tmp_path / field)
        value = {"schema_version": 2, "status": "running", "dataset": "other", "model": "other", "run_id": "other"}[field]
        _mutate_manifest(root, **{field: value})
        with pytest.raises(ValueError, match=field):
            _verify(root, paths)

    root, paths = _make_run(tmp_path / "expected")
    with pytest.raises(ValueError, match="sha256"):
        verify_completed_inputs(
            root,
            dataset="entity-to-dial",
            model="qwen3.5-4b",
            run_id="run-1",
            expected_manifest_sha256=_sha256(paths["manifest"]),
            required={"analyze/summary.json": ("1" * 64, None)},
        )

    root, paths = _make_run(tmp_path / "ref")
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"][0]["sha256"] = "1" * 64
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        _verify(root, paths)

    root, paths = _make_run(tmp_path / "actual")
    paths["summary"].write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        _verify(root, paths)


def test_json_and_jsonl_count_contracts_fail_closed(tmp_path):
    root, paths = _make_run(tmp_path)
    with pytest.raises(ValueError, match="record_count"):
        verify_completed_inputs(
            root,
            dataset="entity-to-dial",
            model="qwen3.5-4b",
            run_id="run-1",
            expected_manifest_sha256=_sha256(paths["manifest"]),
            required={"analyze/summary.json": (_sha256(paths["summary"]), 1)},
        )

    root, paths = _make_run(tmp_path / "jsonl-count")
    with pytest.raises(ValueError, match="record_count"):
        verify_completed_inputs(
            root,
            dataset="entity-to-dial",
            model="qwen3.5-4b",
            run_id="run-1",
            expected_manifest_sha256=_sha256(paths["manifest"]),
            required={"prepare/rows.jsonl": (_sha256(paths["rows"]), True)},
        )

    root, paths = _make_run(tmp_path / "ref-count")
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    data["output_refs"][1]["record_count"] = True
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="record_count"):
        _verify(root, paths)

    root, paths = _make_run(tmp_path / "missing-count")
    data = json.loads(_manifest(root).read_text(encoding="utf-8"))
    del data["output_refs"][1]["record_count"]
    _manifest(root).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="record_count"):
        _verify(root, paths)


def _refresh_ref(root: Path, paths: dict[str, Path], *, ref_index: int, path_key: str) -> None:
    manifest_path = _manifest(root)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["output_refs"][ref_index]["sha256"] = _sha256(paths[path_key])
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


def test_non_object_and_non_finite_json_payloads_fail_closed(tmp_path):
    root, paths = _make_run(tmp_path)
    paths["summary"].write_text("[1, 2]\n", encoding="utf-8")
    _refresh_ref(root, paths, ref_index=0, path_key="summary")
    with pytest.raises(ValueError, match="object"):
        _verify(root, paths)

    root, paths = _make_run(tmp_path / "jsonl-object")
    paths["rows"].write_text('1\n{"id": "b"}\n', encoding="utf-8")
    _refresh_ref(root, paths, ref_index=1, path_key="rows")
    with pytest.raises(ValueError, match="object"):
        _verify(root, paths)

    root, paths = _make_run(tmp_path / "json-finite")
    paths["summary"].write_text('{"value": 1e400}\n', encoding="utf-8")
    _refresh_ref(root, paths, ref_index=0, path_key="summary")
    with pytest.raises(ValueError, match="non-finite"):
        _verify(root, paths)

    root, paths = _make_run(tmp_path / "jsonl-finite")
    paths["rows"].write_text('{"value": NaN}\n{"id": "b"}\n', encoding="utf-8")
    _refresh_ref(root, paths, ref_index=1, path_key="rows")
    with pytest.raises(ValueError, match="non-finite"):
        _verify(root, paths)


def test_missing_required_file_preserves_file_not_found(tmp_path):
    root, paths = _make_run(tmp_path)
    paths["rows"].unlink()

    with pytest.raises(FileNotFoundError):
        _verify(root, paths)


def test_required_paths_must_be_json_or_jsonl_and_mapping_nonempty(tmp_path):
    root, paths = _make_run(tmp_path)
    common = {
        "run_root": root,
        "dataset": "entity-to-dial",
        "model": "qwen3.5-4b",
        "run_id": "run-1",
        "expected_manifest_sha256": _sha256(paths["manifest"]),
    }
    with pytest.raises(ValueError, match="non-empty mapping"):
        verify_completed_inputs(required={}, **common)
    with pytest.raises(ValueError, match="extension"):
        verify_completed_inputs(required={"prepare/data.csv": ("0" * 64, None)}, **common)


def test_invalid_digest_is_rejected(tmp_path):
    root, paths = _make_run(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        verify_completed_inputs(
            root,
            dataset="entity-to-dial",
            model="qwen3.5-4b",
            run_id="run-1",
            expected_manifest_sha256="A" * 64,
            required={"analyze/summary.json": (_sha256(paths["summary"]), None)},
        )
