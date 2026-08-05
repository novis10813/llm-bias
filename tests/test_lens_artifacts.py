import json
from types import SimpleNamespace

import pytest

from llm_bias.core.lens_artifacts import (
    canonical_lens_checkpoint_path,
    canonical_lens_path,
    complete_lens_metadata,
    lens_metadata_path,
    model_slug,
    validate_lens_for_model,
    validate_lens_metadata,
)


def test_model_specific_canonical_lens_paths():
    assert model_slug(".cache/models/llama-3.2-1b-instruct") == (
        "llama-3.2-1b-instruct"
    )
    assert model_slug("org/model-name") == "org--model-name"
    assert canonical_lens_path("org/model-name").as_posix() == (
        "artifacts/org--model-name/jacobian-lens/jacobian_lens.pt"
    )
    assert canonical_lens_checkpoint_path("org/model-name").as_posix() == (
        "artifacts/org--model-name/jacobian-lens/checkpoints/"
        "jacobian_lens.checkpoint.pt"
    )
    assert canonical_lens_checkpoint_path(
        "org/model-name", "abcdef0123456789"
    ).as_posix() == (
        "artifacts/org--model-name/jacobian-lens/checkpoints/"
        "jacobian_lens.abcdef012345.checkpoint.pt"
    )


def test_complete_metadata_records_and_validates_hashes(tmp_path):
    lens_path = tmp_path / "jacobian_lens.pt"
    lens_path.write_bytes(b"lens")
    metadata = complete_lens_metadata(
        metadata={"model": "models/example", "provenance": {"workflow": "test"}},
        lens_path=lens_path,
    )

    validate_lens_metadata(metadata=metadata, lens_path=lens_path)
    assert metadata["artifact_type"] == "jacobian_lens"
    assert metadata["schema_version"] == 2
    assert len(metadata["binary_sha256"]) == 64
    assert len(metadata["metadata_sha256"]) == 64

    lens_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="binary_sha256"):
        validate_lens_metadata(metadata=metadata, lens_path=lens_path)


def test_complete_lens_validation_rejects_partial_coverage(tmp_path):
    model = SimpleNamespace(d_model=16, n_layers=4)
    lens = SimpleNamespace(d_model=16, source_layers=[0, 2])

    with pytest.raises(ValueError, match="missing L1"):
        validate_lens_for_model(
            model=model,
            lens=lens,
            model_name="models/example",
            lens_path=tmp_path / "jacobian_lens.pt",
        )


def test_lens_metadata_rejects_wrong_model(tmp_path):
    lens_path = tmp_path / "jacobian_lens.pt"
    lens_path.write_bytes(b"lens")
    metadata = complete_lens_metadata(
        metadata={
            "model": ".cache/models/other-model",
            "d_model": 16,
            "n_layers": 4,
            "source_layers": [0, 1, 2],
            "provenance": {"workflow": "test"},
        },
        lens_path=lens_path,
    )
    lens_metadata_path(lens_path).write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    model = SimpleNamespace(d_model=16, n_layers=4)
    lens = SimpleNamespace(d_model=16, source_layers=[0, 1, 2])

    with pytest.raises(ValueError, match="does not match"):
        validate_lens_for_model(
            model=model,
            lens=lens,
            model_name=".cache/models/example-model",
            lens_path=lens_path,
        )
