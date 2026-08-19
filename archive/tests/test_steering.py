from pathlib import Path

import torch
from torch import nn

from llm_bias.counterfactual_patching.steering import (
    fit_direction,
    load_direction,
    norm_matched_random_direction,
    permuted_direction,
    save_direction,
)


class _Block(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class _Model:
    n_layers = 2

    def __init__(self):
        self.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, input_ids):
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_fit_direction_uses_recorded_spans_and_round_trips(tmp_path: Path):
    model = _Model()
    records = [
        {"record_id": "a", "input_ids": [[1, 1, 1, 1]], "entity_spans": [{"token_start": 1, "token_end": 2}], "margin": 2.0},
        {"record_id": "b", "input_ids": [[2, 2, 2, 2]], "entity_spans": [{"token_start": 1, "token_end": 2}], "margin": 1.0},
        {"record_id": "c", "input_ids": [[3, 3, 3, 3]], "entity_spans": [{"token_start": 1, "token_end": 2}], "margin": -1.0},
        {"record_id": "d", "input_ids": [[4, 4, 4, 4]], "entity_spans": [{"token_start": 1, "token_end": 2}], "margin": -2.0},
    ]
    direction, metadata = fit_direction(model, records, layer=0, device="cpu")
    assert direction.shape == (2,)
    assert torch.isclose(direction.norm(), torch.tensor(1.0))
    assert metadata.source_record_ids == ["a", "b", "c", "d"]
    path = tmp_path / "direction.pt"
    save_direction(path, direction, metadata)
    loaded, loaded_metadata = load_direction(path)
    assert torch.allclose(direction, loaded)
    assert loaded_metadata == metadata


def test_direction_controls_are_norm_matched():
    direction = torch.tensor([3.0, 4.0])
    random = norm_matched_random_direction(direction, seed=4)
    permuted = permuted_direction(direction)
    assert torch.isclose(random.norm(), direction.norm())
    assert torch.isclose(permuted.norm(), direction.norm())
    assert torch.equal(permuted, torch.tensor([4.0, 3.0]))
