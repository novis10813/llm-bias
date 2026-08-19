import pytest
import torch

from llm_bias.counterfactual_patching.interventions import (
    _add_tensor_spans,
    _patch_tensor_spans,
)


def test_patch_tensor_spans_patches_each_occurrence_without_touching_context():
    tensor = torch.zeros(1, 8, 2)
    patched = _patch_tensor_spans(
        tensor,
        source_spans=[(1, 2), (5, 7)],
        replacements=[torch.tensor([[2.0, 3.0]]), torch.tensor([[4.0, 5.0], [6.0, 7.0]])],
    )
    assert torch.equal(patched[:, 0], tensor[:, 0])
    assert torch.equal(patched[:, 1], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(patched[:, 5], torch.tensor([[4.0, 5.0]]))
    assert torch.equal(patched[:, 6], torch.tensor([[6.0, 7.0]]))
    assert torch.equal(patched[:, 7], tensor[:, 7])


def test_add_tensor_spans_rejects_overlapping_spans():
    with pytest.raises(ValueError, match="overlap"):
        _add_tensor_spans(
            torch.zeros(1, 4, 2),
            source_spans=[(1, 3), (2, 4)],
            direction=torch.ones(2),
            alpha=1.0,
        )
