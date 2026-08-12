from __future__ import annotations

import json

import pytest

from llm_bias.core.lens_registry import (
    ModelIdentity,
    find_pretrained_lens,
    load_pretrained_lens_registry,
    model_identity_from_config,
)


def _config(*, architecture="Qwen3_5ForConditionalGeneration", d_model=2560, n_layers=32):
    return {
        "_name_or_path": "Qwen/Qwen3.5-4B",
        "architectures": [architecture],
        "text_config": {
            "hidden_size": d_model,
            "num_hidden_layers": n_layers,
        },
    }


def test_tracked_registry_resolves_exact_qwen_identity():
    entries = load_pretrained_lens_registry()
    identity = model_identity_from_config(
        _config(), requested_model=".cache/models/qwen3.5-4b"
    )

    entry = find_pretrained_lens(identity, entries)

    assert entry is not None
    assert entry.model.base_model == "Qwen/Qwen3.5-4B"
    assert entry.revision == "a4114d7752d11eb546e6cf372213d7e75526d3a1"
    assert entry.source_layers == tuple(range(31))


@pytest.mark.parametrize(
    "identity",
    [
        ModelIdentity("Qwen/Qwen3.5-4B-Instruct", "Qwen3_5ForConditionalGeneration", 2560, 32),
        ModelIdentity("Qwen/Qwen3.5-4B", "OtherArchitecture", 2560, 32),
        ModelIdentity("Qwen/Qwen3.5-4B", "Qwen3_5ForConditionalGeneration", 2048, 32),
        ModelIdentity("Qwen/Qwen3.5-4B", "Qwen3_5ForConditionalGeneration", 2560, 31),
    ],
)
def test_registry_does_not_fuzzy_match(identity):
    assert find_pretrained_lens(identity, load_pretrained_lens_registry()) is None


def test_local_config_can_use_explicit_canonical_base_model():
    config = _config()
    config.pop("_name_or_path")

    identity = model_identity_from_config(
        config,
        requested_model="/models/qwen",
        base_model="Qwen/Qwen3.5-4B",
    )

    assert identity.base_model == "Qwen/Qwen3.5-4B"


def test_local_config_without_canonical_identity_fails_closed():
    config = _config()
    config.pop("_name_or_path")
    with pytest.raises(ValueError, match="does not prove"):
        model_identity_from_config(config, requested_model="/models/qwen")


def test_registry_rejects_floating_revision(tmp_path):
    registry = json.loads(
        __import__("pathlib").Path("config/pretrained_lenses.json").read_text()
    )
    registry["entries"][0]["source"]["revision"] = "main"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="full lowercase"):
        load_pretrained_lens_registry(path)
