import json

import pytest
import torch
from torch import nn

from llm_bias.entity_cell.analysis import (
    anonymous_progress,
    collision_selectivity_summary,
    denominator_eligibility,
    held_variant_metrics,
    summarize_amnesia,
    surface_control_summary,
    trusted_candidate_eligibility,
)
from llm_bias.entity_cell.cli import build_parser
from llm_bias.entity_cell.mlp_cells import (
    OnlineVectorStats,
    collect_generic_stats,
    mlp_hooks,
    rank_stability_scores,
    record_post_swiglu,
    surface_form_controls,
)


class _Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]]))
        self.seen = None

    def forward(self, value):
        self.seen = value.detach().clone()
        return value @ self.weight.t()


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(2, 3, bias=False)
        self.up_proj = nn.Linear(2, 3, bias=False)
        self.down_proj = _Down()
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.ones(3, 2))
            self.up_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

    def forward(self, value):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, value):
        return self.mlp(value)


class _Model(nn.Module):
    input_device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])

    def forward(self, input_ids):
        value = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return self.layers[0](value)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=True, **_kwargs):
        return {"input_ids": [ord(char) % 17 for char in text]}


def test_hook_reads_post_swiglu_and_scales_only_selected_channel():
    model = _Model()
    ids = torch.tensor([[1, 2]])
    values = record_post_swiglu(model, ids, layers=[0], position=1)[0]
    expected = torch.nn.functional.silu(torch.tensor([4.0, 4.0, 4.0])) * torch.tensor([2.0, 2.0, 4.0])
    assert torch.allclose(values, expected)

    with mlp_hooks(model, [0], channel_scales={0: {1: 0.0}}, scope="all_positions"):
        model(ids)
    observed = model.layers[0].mlp.down_proj.seen
    baseline = torch.nn.functional.silu(torch.tensor([2.0, 2.0, 2.0])) * torch.tensor([1.0, 1.0, 2.0])
    assert torch.allclose(observed[0, 0], torch.tensor([baseline[0], 0.0, baseline[2]]))
    assert torch.allclose(observed[0, 1], torch.tensor([expected[0], 0.0, expected[2]]))


def test_e1_rejects_layers_outside_frozen_localization_band():
    model = _Model()
    with pytest.raises(ValueError, match="L0-L5"):
        with mlp_hooks(model, [6]):
            pass


def test_header_only_scope_and_cleanup_after_exception():
    model = _Model()
    ids = torch.tensor([[1, 2]])
    with mlp_hooks(model, [0], channel_scales={0: {0: 0.0}}, scope="header_only", scope_positions=[1]):
        model(ids)
    seen = model.layers[0].mlp.down_proj.seen
    assert seen[0, 0, 0] != 0
    assert seen[0, 1, 0] == 0
    assert not model.layers[0].mlp.down_proj._forward_pre_hooks
    with pytest.raises(RuntimeError):
        with mlp_hooks(model, [0]):
            raise RuntimeError("boom")
    assert not model.layers[0].mlp.down_proj._forward_pre_hooks


def test_hook_registration_failure_cleans_already_registered_handles(monkeypatch):
    model = _Model()
    original = model.layers[0].mlp.down_proj.register_forward_pre_hook
    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")
    monkeypatch.setattr(model.layers[0].mlp.down_proj, "register_forward_pre_hook", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        with mlp_hooks(model, [0]):
            pass
    assert not model.layers[0].mlp.down_proj._forward_pre_hooks
    monkeypatch.setattr(model.layers[0].mlp.down_proj, "register_forward_pre_hook", original)


def test_online_stats_and_stability_tie_order_are_deterministic():
    stats = {0: OnlineVectorStats()}
    for value in (torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0])):
        stats[0].update(value)
    assert stats[0].count == 2
    assert torch.allclose(stats[0].mean, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(stats[0].std, torch.tensor([1.0, 1.0], dtype=torch.float64))
    ranked = rank_stability_scores({0: [torch.tensor([3.0, 3.0]), torch.tensor([3.0, 3.0])]}, stats, top_k=2)
    assert [(row["layer"], row["neuron"]) for row in ranked] == [(0, 0), (0, 1)]
    assert all("activation" not in json.dumps(row).lower() for row in ranked)


def test_progress_controls_and_eligibility():
    assert anonymous_progress(0.0, 1.0, -1.0) == pytest.approx(0.5)
    assert denominator_eligibility(1.0, 1.05) == (False, "anonymous_gap_below_0.1")
    assert denominator_eligibility(1.0, 0.8)[0]
    local = [{"layer": 0, "neuron": 1, "rank": 1}, {"layer": 0, "neuron": 2, "rank": 2}]
    held = [{"layer": 0, "neuron": 2, "rank": 1}, {"layer": 0, "neuron": 1, "rank": 2}]
    metrics = held_variant_metrics(local, held)
    assert metrics["top1_agreement"] is False
    assert metrics["top5_overlap"] == 2
    summary = summarize_amnesia(
        [
            {"ticker": "A", "scope": "all_positions", "candidate": candidate, "prompt_id": prompt, "alpha": -3.0, "anonymous_progress": progress, "eligible": True}
            for candidate, prompt, progress in (
                ("target", "p1", 0.8), ("wrong_entity", "p1", 0.1), ("matched_random", "p1", 0.2),
                ("target", "p2", 0.7), ("wrong_entity", "p2", 0.1), ("matched_random", "p2", 0.2),
            )
        ]
    )["A:all_positions"]
    assert summary["trusted_candidate_entity_cell"]
    assert trusted_candidate_eligibility(metrics, summary)["eligible"]
    assert not trusted_candidate_eligibility({"top5_overlap": 0}, {"trusted_candidate_entity_cell": False, "exclusion_reasons": ["missing"]})["eligible"]


def test_cli_keeps_prepare_and_adds_run_and_analyze():
    parser = build_parser()
    assert parser.parse_args(["prepare", "--input", "i", "--split-manifest", "s", "--baseline", "b", "--baseline-identity", "adapted:v1", "--model", "m", "--run-id", "r"]).command == "prepare"
    assert parser.parse_args(["run", "--prepared-dir", "p", "--model", "m", "--run-id", "r", "--stage", "e1-baseline"]).stages == ["e1-baseline"]
    assert parser.parse_args(["analyze", "--run-root", "r"]).command == "analyze"


def test_collision_surface_controls_and_rot13():
    candidates = {"A": [{"layer": 0, "neuron": 1}], "B": [{"layer": 0, "neuron": 1}], "C": [{"layer": 0, "neuron": 2}]}
    collisions = collision_selectivity_summary(candidates, scores_by_ticker={"A": {(0, 1): 3.0}, "B": {(0, 1): 1.0}, "C": {(0, 2): 0.0}})
    assert collisions["A"]["same_neuron_other_ticker_count"] == 1
    assert collisions["A"]["entity_selectivity_z"] > 0
    controls = surface_form_controls("Stock Ticker: [AB]\nStock Name: [Alpha Co]", "AB", "Alpha Co")
    assert "[NO]" in controls["name_form_control"]
    summary = surface_control_summary(
        [{"layer": 0, "neuron": 1}],
        {"anonymous_ticker": [{"layer": 0, "neuron": 1}], "anonymous_name": [{"layer": 0, "neuron": 1}], "name_form_control": [{"layer": 0, "neuron": 2}]},
    )
    assert summary["form_robust"]


def test_generic_collection_is_streaming_by_interface(monkeypatch):
    model = _Model()
    calls = []
    original = record_post_swiglu
    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr("llm_bias.entity_cell.mlp_cells.record_post_swiglu", wrapped)
    stats = collect_generic_stats(model, _Tokenizer(), (str(i) for i in range(3)), layers=[0], position=-1)
    assert stats[0].count == 3
    assert stats[0].compact()["width"] == 3
    assert "mean" not in stats[0].compact()
    assert len(calls) == 3
