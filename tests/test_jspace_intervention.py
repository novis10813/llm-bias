from __future__ import annotations

import pytest
import torch

from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.jspace_intervention.concepts import (
    concept_coordinate,
    sector_prototype,
    token_direction,
)
from llm_bias.jspace_intervention.positions import select_loaded_positions
from llm_bias.jspace_intervention.runner import run_swap_record
from llm_bias.jspace_intervention.schemas import InterventionConfig
from llm_bias.jspace_intervention.transforms import (
    coordinate_intervention,
    coordinate_swap,
    steer_positions,
)


class _IdentityBlock(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_IdentityBlock(), _IdentityBlock()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


def test_residual_interventions_apply_and_remove_hooks() -> None:
    model = _FakeModel()
    value = torch.zeros(1, 3, 2)

    with residual_interventions(model, {0: lambda tensor: tensor + 2}):
        assert torch.equal(model(value), torch.full_like(value, 2))

    assert torch.equal(model(value), value)
    assert not model.layers[0]._forward_hooks


def test_residual_interventions_preserve_tuple_block_outputs() -> None:
    class TupleBlock(torch.nn.Module):
        def forward(self, value: torch.Tensor):
            return value, "cache"

    class TupleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([TupleBlock()])

    model = TupleModel()
    value = torch.zeros(1, 1, 2)
    with residual_interventions(model, {0: lambda tensor: tensor + 1}):
        output = model.layers[0](value)

    assert torch.equal(output[0], torch.ones_like(value))
    assert output[1] == "cache"
    assert not model.layers[0]._forward_hooks


def test_residual_interventions_remove_hooks_after_error() -> None:
    model = _FakeModel()

    try:
        with residual_interventions(model, {0: lambda tensor: tensor}):
            raise RuntimeError("stop")
    except RuntimeError:
        pass

    assert not model.layers[0]._forward_hooks


def test_swap_runner_changes_margin_through_tuple_decoder_hooks() -> None:
    class TupleBlock(torch.nn.Module):
        def forward(self, value: torch.Tensor):
            return (value,)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([TupleBlock()])
            self.n_layers = 1
            self.embedding = torch.nn.Embedding(5, 2)
            self.register_buffer(
                "unembedding",
                torch.tensor(
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
                     [1.0, 0.0], [0.0, 1.0]]
                ),
            )
            with torch.no_grad():
                self.embedding.weight.zero_()
                self.embedding.weight[0] = torch.tensor([2.0, 0.0])

        def forward(self, input_ids, attention_mask=None):
            value = self.embedding(input_ids)
            for layer in self.layers:
                value = layer(value)[0]
            return value

        def unembed(self, value):
            return value @ self.unembedding.T

    class Tokenizer:
        mapping = {"P": [0], "Pbuy": [0, 1], "Psell": [0, 2]}

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": self.mapping[text]}

    config = InterventionConfig.from_dict(
        {
            "source": {
                "name": "source", "sector": "Technology", "score_type": "test",
                "tokens": [{"token": "source", "token_id": 3, "weight": 1.0}],
            },
            "target": {
                "name": "target", "sector": "Financial Services", "score_type": "test",
                "tokens": [{"token": "target", "token_id": 4, "weight": 1.0}],
            },
            "layers": [0],
            "alphas": [0, 1],
        }
    )
    model = Model()
    rows = run_swap_record(
        model=model,
        tokenizer=Tokenizer(),
        lens=None,
        scoring_prompt="P",
        evidence_span=(0, 1),
        config=config,
        device="cpu",
        source_prototypes={0: torch.tensor([1.0, 0.0])},
        target_prototypes={0: torch.tensor([0.0, 1.0])},
    )

    assert rows[0]["clean_margin"] == pytest.approx(-2.0)
    assert rows[1]["intervened_margin"] == pytest.approx(2.0)
    assert rows[1]["delta_margin"] == pytest.approx(4.0)
    assert not model.layers[0]._forward_hooks


def test_token_direction_and_sector_prototype() -> None:
    unembedding = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    jacobian = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    assert torch.equal(token_direction(unembedding, jacobian, 1), torch.tensor([0.0, 6.0]))
    prototype = sector_prototype(
        {0: torch.tensor([2.0, 0.0]), 1: torch.tensor([0.0, 3.0])},
        {0: 1.0, 1: 1.0},
    )
    assert torch.allclose(prototype, torch.tensor([2**-0.5, 2**-0.5]))


def test_steering_changes_coordinate_only_at_selected_positions() -> None:
    tensor = torch.zeros(1, 3, 2)
    direction = torch.tensor([2.0, 0.0])
    patched = steer_positions(
        tensor, positions=[1], direction=direction, coordinate_delta=0.5
    )

    assert concept_coordinate(patched[0, 1], direction).item() == 0.5
    assert torch.equal(patched[0, 0], tensor[0, 0])
    assert torch.equal(patched[0, 2], tensor[0, 2])


def test_coordinate_swap_exchanges_coordinates_and_preserves_complement() -> None:
    tensor = torch.tensor([[[2.0, 5.0, 7.0], [1.0, 3.0, 9.0]]])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])

    patched = coordinate_swap(
        tensor,
        positions=[0],
        source_direction=source,
        target_direction=target,
        alpha=1.0,
    )

    assert torch.equal(patched[0, 0], torch.tensor([5.0, 2.0, 7.0]))
    assert torch.equal(patched[0, 1], tensor[0, 1])


def test_source_ablation_and_target_addition_sum_to_swap_delta() -> None:
    tensor = torch.tensor([[[2.0, 5.0, 7.0]]])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    removed = coordinate_intervention(
        tensor, positions=[0], source_direction=source,
        target_direction=target, mode="source_ablation",
    )
    added = coordinate_intervention(
        tensor, positions=[0], source_direction=source,
        target_direction=target, mode="target_addition",
    )
    swapped = coordinate_swap(
        tensor, positions=[0], source_direction=source, target_direction=target
    )

    assert torch.allclose(
        (removed - tensor) + (added - tensor), swapped - tensor
    )


def test_loaded_position_selection_uses_median_layer_cosine() -> None:
    residuals = {
        14: torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]]]),
        15: torch.tensor([[[0.0, 1.0], [0.9, 0.1], [0.7, 0.3], [-1.0, 0.0]]]),
    }
    directions = {14: torch.tensor([1.0, 0.0]), 15: torch.tensor([1.0, 0.0])}

    selected = select_loaded_positions(
        residuals, directions, evidence_span=(1, 4), top_k=2, threshold=0.5
    )

    assert selected.positions == (1, 2)
    assert selected.loaded is True
    assert min(selected.scores) > 0.5
