from types import SimpleNamespace

import torch
from torch import nn

from llm_bias.counterfactual_patching.interventions import (
    _patch_tensor_span,
    _replace_first,
    next_logits,
    normalized_span_mapping,
    patched_next_logits,
)


class _TupleBlock(nn.Module):
    def forward(self, hidden):
        return hidden + 1 + hidden.mean(dim=1, keepdim=True), "tail"


class _TinyModel:
    n_layers = 2

    def __init__(self):
        self.layers = nn.ModuleList([_TupleBlock(), _TupleBlock()])
        self.weight = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )

    def forward(self, input_ids):
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2).clone()
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden

    def unembed(self, residual):
        return residual @ self.weight.T


def test_normalized_span_mapping_uses_nearest_target_centers():
    assert normalized_span_mapping(1, 1) == [0]
    assert normalized_span_mapping(2, 3) == [0, 2]
    assert normalized_span_mapping(3, 2) == [0, 1, 1]


def test_patch_tensor_span_maps_variable_length_replacement():
    tensor = torch.zeros(1, 5, 2)
    replacement = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

    patched = _patch_tensor_span(
        tensor, source_span=(1, 3), replacement=replacement
    )

    expected = tensor.clone()
    expected[:, 1, :] = replacement[0]
    expected[:, 2, :] = replacement[2]
    assert torch.equal(patched, expected)


def test_patched_next_logits_handles_variable_length_spans():
    model = _TinyModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    replacement = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

    baseline = next_logits(model, input_ids)
    patched = patched_next_logits(
        model,
        input_ids,
        layer=0,
        source_span=(1, 3),
        replacement=replacement,
    )

    assert patched.shape == baseline.shape
    assert not torch.equal(patched, baseline)


def test_replace_first_preserves_tuple_tail():
    first = torch.zeros(1, 3, 4)
    second = torch.ones(1, 2)
    replacement = torch.full_like(first, 3)
    result = _replace_first((first, second), replacement)
    assert isinstance(result, tuple)
    assert torch.equal(result[0], replacement)
    assert torch.equal(result[1], second)


def test_replace_first_tensor():
    first = torch.zeros(1, 3, 4)
    replacement = torch.ones_like(first)
    assert torch.equal(_replace_first(first, replacement), replacement)


def test_replace_first_preserves_list_tail():
    first = torch.zeros(1, 3, 4)
    second = torch.ones(1, 2)
    replacement = torch.full_like(first, 3)
    result = _replace_first([first, second], replacement)
    assert isinstance(result, list)
    assert torch.equal(result[0], replacement)
    assert torch.equal(result[1], second)


def test_replace_first_updates_model_output_hidden_state():
    first = torch.zeros(1, 3, 4)
    replacement = torch.ones_like(first)
    output = SimpleNamespace(last_hidden_state=first, cache="untouched")
    result = _replace_first(output, replacement)
    assert result is output
    assert torch.equal(result.last_hidden_state, replacement)
    assert result.cache == "untouched"
