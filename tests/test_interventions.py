import torch

from llm_bias.interventions import _replace_first


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
