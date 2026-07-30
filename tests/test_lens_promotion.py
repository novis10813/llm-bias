import importlib.util
import json
from pathlib import Path


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
    candidate.with_name("jacobian_lens.pt.metadata.json").write_text(
        json.dumps({"model": "models/model", "source_layers": [0]}),
        encoding="utf-8",
    )
    canonical = tmp_path / "artifacts" / "lenses" / "model" / "jacobian_lens.pt"
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
