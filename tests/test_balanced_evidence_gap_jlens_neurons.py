"""Unit tests for the J-lens neuron structural diagnostic (no model load)."""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from llm_bias.balanced_evidence_gap import jlens_neurons as J


def test_control_neurons_for_layer_matches_phase3_rule():
    # Frozen Phase 3 / 2C control indices in the order persisted by the
    # formal run's prepare stage (seed 42+layer, sample of 9216)
    assert J.control_neurons_for_layer(19) == [
        8101, 2977, 9120, 3551, 5252, 4805, 5258, 454, 7965, 5856,
    ]
    assert J.control_neurons_for_layer(20) == [
        2834, 1067, 3901, 7614, 5077, 5828, 2803, 8323, 3065, 2646,
    ]
    assert J.control_neurons_for_layer(26) == [
        7640, 8204, 1818, 3987, 7075, 7687, 4349, 5396, 7178, 9064,
    ]
    # deterministic and disjoint across layers by construction of the seed
    assert J.control_neurons_for_layer(19) == J.control_neurons_for_layer(19)
    assert J.control_neurons_for_layer(15) != J.control_neurons_for_layer(19)
    assert all(0 <= n < J.MOE_WIDTH for n in J.control_neurons_for_layer(15))


def test_pairwise_cosine_identity_and_orthogonality():
    a = torch.tensor([1.0, 2.0, 0.0])
    b = torch.tensor([3.0, 6.0, 0.0])  # parallel to a
    c = torch.tensor([0.0, 0.0, 5.0])  # orthogonal
    out = J.pairwise_cosine({"a": a, "b": b, "c": c})
    assert out["a"]["a"] == pytest.approx(1.0, abs=1e-6)
    assert out["a"]["b"] == pytest.approx(1.0, abs=1e-6)
    assert out["a"]["c"] == pytest.approx(0.0, abs=1e-6)


def _fake_model(d_model: int = 4, vocab: int = 12, n_neurons: int = 8,
                n_layers: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    down = nn.Linear(n_neurons, d_model, bias=False)  # weight [d_model, n_neurons]
    lm_head = nn.Linear(d_model, vocab, bias=False)

    class _Layer:
        def __init__(self):
            self.mlp = types.SimpleNamespace(down_proj=down)

    class _Model:
        def __init__(self):
            self.layers = [_Layer() for _ in range(n_layers)]
            self._lm_head = lm_head

        def unembed(self, residual):
            return self._lm_head(residual)

    class _Tokenizer:
        def decode(self, ids):
            return " ".join(f"t{i}" for i in ids)

    class _Lens:
        def __init__(self):
            self.lens = types.SimpleNamespace(
                jacobians={i: torch.eye(d_model) for i in range(n_layers)},
            )

    return _Model(), _Tokenizer(), _Lens()


def test_readout_one_matches_manual_computation():
    model, tokenizer, lens = _fake_model()
    layer, neuron = 2, 3
    w = model.layers[layer].mlp.down_proj.weight[:, neuron].detach().float()
    expected_logits = model._lm_head(w.reshape(1, -1)).float()[0]

    out = J.readout_one(model, lens, tokenizer, layer, neuron,
                        buy_id=1, sell_id=2)
    assert out["margin_buy_sell"] == pytest.approx(
        float(expected_logits[1] - expected_logits[2]), abs=1e-5)
    assert out["transported_norm"] == pytest.approx(float(w.norm()), abs=1e-5)
    assert len(out["top10"]) == J.TOP_K
    # softmax probabilities sum to ~1 over the top-k subset of a 12-vocab
    assert sum(t["prob"] for t in out["top10"]) <= 1.0 + 1e-6
    assert all(torch.isfinite(torch.tensor([t["prob"] for t in out["top10"]])))
    # only compact fields persisted (no vectors)
    assert set(out) == {
        "layer", "neuron", "transported_norm", "margin_buy_sell",
        "prob_buy", "prob_sell", "top10",
    }


def test_readout_one_transport_uses_jacobian():
    # non-identity Jacobian must change the decoded direction
    model, tokenizer, lens = _fake_model()
    jacobian = torch.eye(4)
    jacobian[0, 0] = 5.0
    lens.lens.jacobians[2] = jacobian
    neuron, layer = 3, 2
    out = J.readout_one(model, lens, tokenizer, layer, neuron,
                        buy_id=1, sell_id=2)
    w = model.layers[layer].mlp.down_proj.weight[:, neuron].detach().float()
    expected_logits = model._lm_head((w @ jacobian.T).reshape(1, -1)).float()[0]
    assert out["margin_buy_sell"] == pytest.approx(
        float(expected_logits[1] - expected_logits[2]), abs=1e-5)


def test_readout_one_rejects_out_of_range_neuron():
    model, tokenizer, lens = _fake_model()
    with pytest.raises(ValueError, match="out of range"):
        J.readout_one(model, lens, tokenizer, 2, 999, buy_id=1, sell_id=2)
