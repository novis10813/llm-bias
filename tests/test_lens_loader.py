from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from jlens import JacobianLens

from llm_bias.core.lens_artifacts import complete_lens_metadata, lens_metadata_path
from llm_bias.core.lens_loader import load_validated_lens


def _write_lens(path, *, model_name="org/model", layers=(0, 1)):
    path.parent.mkdir(parents=True, exist_ok=True)
    JacobianLens(
        {layer: torch.eye(2) for layer in layers}, n_prompts=2, d_model=2
    ).save(str(path))
    metadata = complete_lens_metadata(
        metadata={
            "model": model_name,
            "d_model": 2,
            "n_layers": 3,
            "source_layers": list(layers),
            "provenance": {"workflow": "test"},
        },
        lens_path=path,
    )
    lens_metadata_path(path).write_text(json.dumps(metadata), encoding="utf-8")


def test_loader_uses_explicit_local_path(tmp_path):
    lens_path = tmp_path / "lens.pt"
    _write_lens(lens_path)

    loaded = load_validated_lens(
        model=SimpleNamespace(d_model=2, n_layers=3),
        model_name="org/model",
        lens_path=lens_path,
    )

    assert loaded.path == lens_path
    assert loaded.lens.source_layers == [0, 1]
    assert loaded.source == "test"


def test_loader_resolves_canonical_without_network(tmp_path):
    canonical = tmp_path / "org--model" / "jacobian-lens" / "jacobian_lens.pt"
    _write_lens(canonical)

    loaded = load_validated_lens(
        model=SimpleNamespace(d_model=2, n_layers=3),
        model_name="org/model",
        artifact_root=tmp_path,
    )

    assert loaded.path == canonical


def test_loader_rejects_missing_or_partial_artifact(tmp_path):
    model = SimpleNamespace(d_model=2, n_layers=3)
    with pytest.raises(FileNotFoundError, match="jacobian-lens install"):
        load_validated_lens(
            model=model,
            model_name="org/model",
            artifact_root=tmp_path,
        )

    lens_path = tmp_path / "partial.pt"
    _write_lens(lens_path, layers=(0,))
    with pytest.raises(ValueError, match="missing L1"):
        load_validated_lens(
            model=model,
            model_name="org/model",
            lens_path=lens_path,
        )
