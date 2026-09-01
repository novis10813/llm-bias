import json
from pathlib import Path

import pytest
import torch
from torch import nn

from llm_bias.entity_cell.cli import build_parser
from llm_bias.entity_cell.suppression import (
    E3_ALPHA_GRID,
    E3_BETA_GRID,
    compute_source_attenuation_deltas,
    deterministic_source_subset,
    e3_compact_record,
    norm_matched_source_delta,
    preservation_metrics,
    validate_dose_grid,
)


def test_frozen_dose_grids_and_validation():
    assert E3_ALPHA_GRID == (1.0, 0.5, 0.0, -1.0, -2.0, -3.0)
    assert E3_BETA_GRID == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert validate_dose_grid(E3_BETA_GRID, name="beta", lower=0, upper=1) == E3_BETA_GRID
    with pytest.raises(ValueError):
        validate_dose_grid([0, 0], name="dose")
    with pytest.raises(ValueError):
        validate_dose_grid([2], name="beta", upper=1)


def test_norm_matching_excludes_low_norm_and_preserves_target_norm():
    target = torch.tensor([3.0, 4.0])
    delta, eligible, reason = norm_matched_source_delta(target, torch.zeros(2))
    assert delta is None and not eligible and reason == "source_direction_norm_below_epsilon"
    delta, eligible, reason = norm_matched_source_delta(target, torch.tensor([0.0, 2.0]))
    assert eligible and reason is None
    assert torch.linalg.vector_norm(delta) == pytest.approx(torch.linalg.vector_norm(target), rel=1e-6)


def test_source_attenuation_isolates_identity_and_supports_grouping():
    source = {
        0: {
            "identity_header": torch.tensor([2.0, 0.0]),
            "evidence": torch.tensor([0.0, 3.0]),
            "instruction_context": torch.tensor([1.0, 1.0]),
            "other_prefix": torch.tensor([0.5, 0.5]),
        },
        1: {
            "identity_header": torch.tensor([1.0, 0.0]),
            "evidence": torch.tensor([0.0, 2.0]),
            "instruction_context": torch.tensor([1.0, 0.0]),
            "other_prefix": torch.tensor([0.0, 1.0]),
        },
    }
    deltas, diagnostics = compute_source_attenuation_deltas(source, [0], beta=0.5)
    assert torch.allclose(deltas[0], torch.tensor([-1.0, 0.0]))
    assert diagnostics["eligible"]
    grouped, diagnostics = compute_source_attenuation_deltas(source, [0, 1], beta=0.5, grouping="group")
    assert diagnostics["eligible"]
    assert torch.linalg.vector_norm(torch.cat([grouped[0], grouped[1]])) == pytest.approx(torch.linalg.vector_norm(torch.tensor([1.0, 0.0, 0.5, 0.0])), rel=1e-6)
    untouched, _ = compute_source_attenuation_deltas(source, [0], beta=0.5, mode="whole_head")
    assert not torch.allclose(untouched[0], deltas[0])


def test_random_subset_and_preservation_are_deterministic():
    assert deterministic_source_subset(2, range(8), seed=4) == deterministic_source_subset(2, range(8), seed=4)
    assert len(deterministic_source_subset(2, range(8), seed=4)) == 2
    metrics = preservation_metrics({"identity_header": 2, "evidence": 3, "instruction_context": 4, "other_prefix": 5}, {"identity_header": 1, "evidence": 3, "instruction_context": 4, "other_prefix": 5})
    assert metrics["identity_header"]["delta"] == -1
    assert metrics["evidence"]["delta"] == 0
    with pytest.raises(ValueError):
        preservation_metrics({"identity_header": float("nan")}, {})


def test_serialization_rejects_nonfinite_and_keeps_compact_payload():
    row = e3_compact_record(ticker="A", prompt_id="p", phase="e3-a", scope="all_positions", dose=0, margin=1, clean_margin=2, anonymous_margin=0, flip=True, contributions={"L11H0": {"identity_header": -1}}, provenance={"cell": {"layer": 0, "neuron": 1}})
    assert row["provenance"]["raw_runtime_payloads"] is False
    assert "tensor" not in json.dumps(row)
    with pytest.raises(ValueError):
        e3_compact_record(ticker="A", prompt_id="p", phase="e3-a", scope="all_positions", dose=0, margin=1, clean_margin=2, anonymous_margin=0, flip=False, contributions={}, controls={"hidden_states": [1]})
    with pytest.raises(TypeError):
        e3_compact_record(ticker="A", prompt_id="p", phase="e3-a", scope="all_positions", dose=0, margin=1, clean_margin=2, anonymous_margin=0, flip=False, contributions={}, controls={"object": object()})
    with pytest.raises(ValueError):
        e3_compact_record(ticker="A", prompt_id="p", phase="e3-a", scope="all_positions", dose=float("nan"), margin=1, clean_margin=2, anonymous_margin=0, flip=False, contributions={})


def test_e3_cli_preserves_previous_commands_and_exposes_discovery():
    parser = build_parser()
    assert parser.parse_args(["prepare", "--input", "i", "--split-manifest", "s", "--baseline", "b", "--baseline-identity", "v", "--model", "m", "--run-id", "r"]).command == "prepare"
    args = parser.parse_args(["run", "--prepared-dir", "p", "--model", "m", "--run-id", "r", "--stage", "e3-downstream", "--e3-head", "11", "2", "--e3-grouping", "group"])
    assert args.stages == ["e3-downstream"] and args.e3_head == [[11, 2]] and args.e3_grouping == "group"
    assert parser.parse_args(["analyze", "--run-root", "r", "--experiment", "e3"]).experiment == "e3"
