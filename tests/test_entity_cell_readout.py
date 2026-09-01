import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.lens_loader import LoadedLens
from llm_bias.entity_cell.readout import readout_selected_component


class _Tokenizer:
    name_or_path = "org/model-tokenizer"
    _ids = {"buy": 0, "sell": 1, "technology": 2, "software": 3, "bank": 4, "cloud": 5, "internet": 6}

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        value = text.strip()
        if value not in self._ids:
            raise ValueError(value)
        return {"input_ids": [self._ids[value]]}

    def decode(self, ids, **_kwargs):
        inverse = {value: key for key, value in self._ids.items()}
        return inverse[int(ids[0])]


class _Lens:
    d_model = 3
    source_layers = list(range(31))

    def __init__(self):
        self.calls = []

    def transport(self, value, layer):
        self.calls.append((value.detach().clone(), layer))
        return value + torch.tensor([0.0, 1.0, 0.0])


class _Model:
    d_model = 3
    n_layers = 32
    model_revision = "rev-1"

    def __init__(self):
        self._final_norm = nn.RMSNorm(3, eps=1e-6)
        self._lm_head = nn.Linear(3, 7, bias=False)
        with torch.no_grad():
            self._lm_head.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.1, 0.1, 0.1],
            ]))


def _loaded_lens(tmp_path: Path, lens: _Lens) -> LoadedLens:
    path = tmp_path / "jacobian_lens.pt"
    path.write_bytes(b"canonical-lens")
    metadata = {
        "model": "org/model",
        "d_model": 3,
        "n_layers": 32,
        "source_layers": list(range(31)),
        "binary_sha256": sha256_file(path),
        "provenance": {
            "revision": "rev-1",
            "tokenizer_identity": "org/model-tokenizer",
            "source": "test-canonical",
        },
    }
    return LoadedLens(lens=lens, path=path, metadata=metadata)


def _kwargs(tmp_path, lens):
    return dict(
        model=_Model(),
        tokenizer=_Tokenizer(),
        model_name="org/model",
        component_vector=torch.tensor([1.0, 0.0, 0.0]),
        source_layer=3,
        component_identity={"layer": 1, "head": 2},
        prompt_id="p1",
        ticker="AAA",
        selection_status={"selection_eligible": True, "selected_top_five": True, "rank": 1},
        loaded_lens=_loaded_lens(tmp_path, lens),
        expected_lens_sha256=sha256_file(tmp_path / "jacobian_lens.pt"),
        expected_model_revision="rev-1",
        expected_tokenizer_identity="org/model-tokenizer",
        sector_vocabulary={"Technology": ["technology", "technology", "software"], "Finance": ["bank"]},
    )


def test_selected_aggregate_transports_once_and_emits_compact_scores(tmp_path):
    lens = _Lens()
    result = readout_selected_component(**_kwargs(tmp_path, lens))

    assert len(lens.calls) == 1
    assert lens.calls[0][1] == 3
    assert result["component"]["kind"] == "aggregate_head"
    assert result["readout"]["top_k"] == 7
    assert result["readout"]["fixed_buy_sell"]["buy"]["token_id"] == 0
    assert result["readout"]["sector_vocabulary_duplicates_removed"] == {"Technology": [2]}
    top = result["readout"]["top_tokens"]
    assert [row["score"] for row in top] == sorted((row["score"] for row in top), reverse=True)
    assert [row["token_id"] for row in top] == [0, 1, 3, 4, 6, 2, 5]
    encoded = json.dumps(result)
    assert '"component_vector": [' not in encoded
    assert '"full_vocabulary": [' not in encoded
    assert '"transported_vector": [' not in encoded


def test_validation_rejects_selection_identity_revision_and_layer(tmp_path):
    lens = _Lens()
    kwargs = _kwargs(tmp_path, lens)
    kwargs["selection_status"] = {"selection_eligible": False, "selected_top_five": True}
    with pytest.raises(ValueError, match="selected eligible"):
        readout_selected_component(**kwargs)

    kwargs = _kwargs(tmp_path, lens)
    kwargs["expected_model_revision"] = "wrong"
    with pytest.raises(ValueError, match="model revision"):
        readout_selected_component(**kwargs)

    kwargs = _kwargs(tmp_path, lens)
    kwargs["source_layer"] = 31
    with pytest.raises(ValueError, match="cover"):
        readout_selected_component(**kwargs)

    kwargs = _kwargs(tmp_path, lens)
    kwargs["model_name"] = "other/model"
    with pytest.raises(ValueError, match="model identity"):
        readout_selected_component(**kwargs)


def test_source_group_is_explicit_and_forbidden_payloads_are_rejected(tmp_path):
    lens = _Lens()
    kwargs = _kwargs(tmp_path, lens)
    kwargs.update(component_kind="source_group", source_group="identity_header")
    result = readout_selected_component(**kwargs)
    assert result["component"]["source_group"] == "identity_header"

    kwargs = _kwargs(tmp_path, lens)
    kwargs["component_provenance"] = {"component_vector": [1, 2, 3]}
    with pytest.raises(ValueError, match="compact readout"):
        readout_selected_component(**kwargs)
