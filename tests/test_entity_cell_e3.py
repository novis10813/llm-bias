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
    args = parser.parse_args(["run", "--prepared-dir", "p", "--model", "m", "--run-id", "r", "--stage", "e3-downstream", "--e3-head", "11", "2", "--e3-grouping", "group", "--e3-peer-tickers", "ADI", "MU", "FTV"])
    assert args.stages == ["e3-downstream"] and args.e3_head == [[11, 2]] and args.e3_grouping == "group"
    assert args.e3_peer_tickers == ["ADI", "MU", "FTV"]
    assert parser.parse_args(["analyze", "--run-root", "r", "--experiment", "e3"]).experiment == "e3"


def test_e3_upstream_record_with_cross_ticker_label_and_source(monkeypatch):
    import contextlib
    from llm_bias.entity_cell import e3

    monkeypatch.setattr(e3, "mlp_hooks", lambda *args, **kwargs: contextlib.nullcontext())

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "prompt"

    prompt_row = {
        "ticker": "ADI",
        "name": "Analog Devices",
        "prompt": "Stock Ticker: [ADI]\nStock Name: [Analog Devices]\n--- Evidence ---\nTest evidence",
        "prompt_id": "prompt_001",
        "source_groups": {"identity_header": {"ranges": [[0, 5]]}},
        "final_query_position": 10,
    }
    candidate = {"layer": 0, "neuron": 104}

    # Mock score_fn so model is not called
    records = e3.run_upstream_suppression_record(
        model=None,
        tokenizer=FakeTokenizer(),
        prompt_row=prompt_row,
        candidate=candidate,
        wrong_candidate=None,
        random_neuron=500,
        alpha_grid=[1.0, -3.0],
        scopes=["all_positions"],
        score_fn=lambda prompt: 0.5 if "ANON" not in prompt else 0.1,
        target_label="target_cell_cross_ticker",
        source_ticker="FTNT",
    )
    assert len(records) == 4  # 2 doses * 2 controls (target_cell_cross_ticker, matched_random)
    cross_ticker_recs = [r for r in records if r["controls"]["candidate"] == "target_cell_cross_ticker"]
    assert len(cross_ticker_recs) == 2
    assert cross_ticker_recs[0]["controls"]["source_ticker"] == "FTNT"
    assert cross_ticker_recs[0]["controls"]["cell"] == {"layer": 0, "neuron": 104}
    assert cross_ticker_recs[0]["ticker"] == "ADI"


def test_analyze_e3_records_groups_by_candidate_and_computes_contrasts():
    from llm_bias.entity_cell.e3 import analyze_e3_records

    rows = [
        # FTNT target
        e3_compact_record(
            ticker="FTNT", prompt_id="p1", phase="e3-a", scope="all_positions",
            dose=-3.0, margin=0.2, clean_margin=0.5, anonymous_margin=0.1,
            flip=False, contributions={}, controls={"candidate": "target", "eligible": True}
        ),
        # FTNT matched_random
        e3_compact_record(
            ticker="FTNT", prompt_id="p1", phase="e3-a", scope="all_positions",
            dose=-3.0, margin=0.45, clean_margin=0.5, anonymous_margin=0.1,
            flip=False, contributions={}, controls={"candidate": "matched_random", "eligible": True}
        ),
        # ADI target_cell_cross_ticker
        e3_compact_record(
            ticker="ADI", prompt_id="p2", phase="e3-a", scope="all_positions",
            dose=-3.0, margin=0.4, clean_margin=0.6, anonymous_margin=0.1,
            flip=False, contributions={}, controls={"candidate": "target_cell_cross_ticker", "source_ticker": "FTNT", "eligible": True}
        ),
    ]
    analysis = analyze_e3_records(rows)
    assert analysis["schema_version"] == 1
    assert "groups" in analysis
    group_keys = [g["group"] for g in analysis["groups"]]
    assert "FTNT:e3-a:all_positions:target" in group_keys
    assert "FTNT:e3-a:all_positions:matched_random" in group_keys
    assert "ADI:e3-a:all_positions:target_cell_cross_ticker" in group_keys

    # Check that cross_ticker_contrasts contains FTNT_vs_ADI
    contrasts = analysis["cross_ticker_contrasts"]
    assert "FTNT_vs_ADI_all_positions" in contrasts
    contrast = contrasts["FTNT_vs_ADI_all_positions"]
    assert contrast["target_ticker"] == "FTNT"
    assert contrast["peer_ticker"] == "ADI"
    assert "specificity_delta" in contrast


def test_wrong_candidate_fallback_when_single_trusted_ticker():
    trusted = {"FTNT": {"layer": 0, "neuron": 104}}
    cells = [
        {"ticker": "FTNT", "localization_candidates": [{"layer": 0, "neuron": 104}]},
        {"ticker": "FTV", "localization_candidates": [{"layer": 0, "neuron": 5101}]},
    ]
    candidate = trusted["FTNT"]
    ticker = "FTNT"
    wrong = next((cell for name, cell in trusted.items() if name != ticker and int(cell["layer"]) == int(candidate["layer"])), None)
    assert wrong is None
    if wrong is None and cells:
        all_tickers = sorted(str(row["ticker"]) for row in cells)
        if ticker in all_tickers:
            idx = all_tickers.index(ticker)
            cells_by_ticker = {str(row["ticker"]): row.get("localization_candidates", ()) for row in cells}
            for step in range(1, len(all_tickers)):
                next_ticker = all_tickers[(idx + step) % len(all_tickers)]
                for c in cells_by_ticker.get(next_ticker, ()):
                    if (int(c["layer"]), int(c["neuron"])) != (int(candidate["layer"]), int(candidate["neuron"])):
                        wrong = c
                        break
                if wrong is not None:
                    break
    assert wrong == {"layer": 0, "neuron": 5101}


