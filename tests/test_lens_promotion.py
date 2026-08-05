import importlib.util
import json
from pathlib import Path

import pytest


def _promotion_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "promote_qwen_lens_candidate.py"
    )
    spec = importlib.util.spec_from_file_location("lens_promotion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promotion_archives_previous_canonical_and_copies_winner(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    candidate = tmp_path / "candidates" / "mixed" / "jacobian_lens.pt"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"selected lens")
    candidate_metadata = _promotion_module().complete_lens_metadata(
        metadata={
            "model": ".cache/models/model",
            "source_layers": [0],
            "provenance": {"workflow": "test"},
        },
        lens_path=candidate,
    )
    candidate.with_name("jacobian_lens.pt.metadata.json").write_text(
        json.dumps(candidate_metadata), encoding="utf-8"
    )
    canonical = (
        tmp_path / "artifacts" / "model" / "jacobian-lens" / "jacobian_lens.pt"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"old lens")
    canonical.with_name("jacobian_lens.pt.metadata.json").write_text(
        json.dumps({"model": "models/model"}),
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "selected_candidate": "mixed",
                "selection_rule": "rule",
                "selection_uncertainty": {"comparisons": {"english": {}}},
                "candidates": {
                    "mixed": {
                        "lens_path": str(candidate),
                        "lens_binary_sha256": candidate_metadata["binary_sha256"],
                        "lens_metadata_sha256": candidate_metadata["metadata_sha256"],
                        "summary": {"selection_score": -0.5},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = _promotion_module().promote(
        model_name=".cache/models/model",
        evaluation_path=evaluation,
        archive_root=tmp_path / "archive",
    )

    assert canonical.read_bytes() == b"selected lens"
    assert Path(result["archive"]).joinpath("jacobian_lens.pt").read_bytes() == (
        b"old lens"
    )
    metadata = json.loads(
        canonical.with_name("jacobian_lens.pt.metadata.json").read_text()
    )
    assert metadata["selected_candidate"] == "mixed"
    assert metadata["selection_status"] == "canonical"
    assert metadata["selection_uncertainty"] == {
        "comparisons": {"english": {}}
    }
    assert result["selection_uncertainty"] == {
        "comparisons": {"english": {}}
    }


def _evaluation_for_candidate(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    module = _promotion_module()
    candidate = tmp_path / "candidates" / "mixed" / "jacobian_lens.pt"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"selected lens")
    metadata = module.complete_lens_metadata(
        metadata={
            "model": ".cache/models/model",
            "source_layers": [0],
            "provenance": {"workflow": "test"},
        },
        lens_path=candidate,
    )
    metadata_path = candidate.with_name("jacobian_lens.pt.metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "selected_candidate": "mixed",
                "selection_rule": "rule",
                "candidates": {
                    "mixed": {
                        "lens_path": str(candidate),
                        "lens_binary_sha256": metadata["binary_sha256"],
                        "lens_metadata_sha256": metadata["metadata_sha256"],
                        "summary": {"selection_score": -0.5},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return evaluation, metadata


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("lens_binary_sha256", "missing the selected candidate binary hash"),
        ("lens_metadata_sha256", "missing the selected candidate metadata hash"),
    ],
)
def test_promotion_requires_evaluation_recorded_hashes(tmp_path, field, message):
    evaluation, _metadata = _evaluation_for_candidate(tmp_path)
    value = json.loads(evaluation.read_text(encoding="utf-8"))
    del value["candidates"]["mixed"][field]
    evaluation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _promotion_module().promote(
            model_name=".cache/models/model",
            evaluation_path=evaluation,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("lens_binary_sha256", "binary hash disagrees with the evaluation record"),
        ("lens_metadata_sha256", "metadata hash disagrees with the evaluation record"),
    ],
)
def test_promotion_rejects_evaluation_hash_mismatch(tmp_path, field, message):
    evaluation, _metadata = _evaluation_for_candidate(tmp_path)
    value = json.loads(evaluation.read_text(encoding="utf-8"))
    value["candidates"]["mixed"][field] = "0" * 64
    evaluation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _promotion_module().promote(
            model_name=".cache/models/model",
            evaluation_path=evaluation,
        )
