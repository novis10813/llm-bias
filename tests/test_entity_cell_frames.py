"""E1 V2 surface-varying frame localization: frozen specs, preparation, and gates."""
import codecs
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.entity_cell import (
    FINANCIAL_PROMPT_COLUMNS,
    prepare_artifacts,
    prepare_inputs,
    validate_prepared_inputs,
)
from llm_bias.entity_cell.analysis import (
    frame_surface_control_summary,
    select_wrong_entity_cell,
    v2_candidate_eligibility,
)
from llm_bias.entity_cell.mlp_cells import (
    CANDIDATE_LAYERS,
    OnlineVectorStats,
    rank_absolute_activations,
    run_amnesia_curve,
)
from llm_bias.entity_cell.preparation import (
    FRAME_HELD_VARIANT_IDS,
    FRAME_LOCALIZATION_VARIANT_IDS,
    FRAME_VARIANT_COUNT,
    FRAME_VARIANT_SPECS,
    TEMPLATE_CONTROL_NAME,
    TEMPLATE_CONTROL_PROMPT,
    render_frame_variants,
)
from llm_bias.entity_cell import pipeline as e1_pipeline
from llm_bias.entity_cell.cli import build_parser

PROMPT = """Use the report to answer.
Stock Ticker: [{ticker}]
Stock Name: [{name}]
--- Evidence ---
{ticker} reported growth for {name}.
---
Respond with JSON."""


NAME = "Alpha Systems, Inc."


class _Tokenizer:
    """Character-level fake tokenizer (one token per character)."""

    chat_template = "fake"
    name_or_path = "fake-tokenizer"
    vocab_size = 251

    def apply_chat_template(self, messages, **_kwargs):
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, return_special_tokens_mask=False):
        result = {"input_ids": [ord(char) % 251 for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        if return_special_tokens_mask:
            result["special_tokens_mask"] = [False] * len(text)
        return result


def _write_inputs(tmp_path: Path, *, tickers=("ALFA", "BETA")) -> tuple[Path, Path, Path]:
    source = tmp_path / "prompts.csv"
    fields = ["Date", "ticker", "name", "sector", "marketcap", *FINANCIAL_PROMPT_COLUMNS]
    names = {"ALFA": "Alpha Systems, Inc.", "BETA": "Beta Logic, Inc."}
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ticker in tickers:
            writer.writerow(
                {
                    "Date": "2026-01-01",
                    "ticker": ticker,
                    "name": names[ticker],
                    "sector": "Technology",
                    "marketcap": "100",
                    **{column: PROMPT.format(ticker=ticker, name=names[ticker]) for column in FINANCIAL_PROMPT_COLUMNS},
                }
            )
    split = tmp_path / "splits.json"
    assignments = {ticker: "discovery" for ticker in tickers}
    split.write_text(
        json.dumps({"schema_version": 1, "input_sha256": sha256_file(source), "assignments": assignments}),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        "\n".join(
            json.dumps({"prompt_id": f"p{index}", "prompt": f"generic prompt {index}", "prompt_sha256": __import__("hashlib").sha256(f"generic prompt {index}".encode()).hexdigest()})
            for index in range(2)
        ) + "\n",
        encoding="utf-8",
    )
    return source, split, baseline


def _prepare_kwargs(tmp_path: Path):
    source, split, baseline = _write_inputs(tmp_path)
    return dict(
        tokenizer=_Tokenizer(),
        input_path=source,
        split_manifest=split,
        baseline_source=baseline,
        baseline_identity="adapted:test-v1",
        baseline_expected_count=2,
    )


def test_frame_specs_are_frozen_twelve_with_split_families():
    assert len(FRAME_VARIANT_SPECS) == FRAME_VARIANT_COUNT == 12
    assert [row[0] for row in FRAME_VARIANT_SPECS] == list(range(12))
    assert all(row[2].count("{name}") == 1 for row in FRAME_VARIANT_SPECS)
    assert FRAME_LOCALIZATION_VARIANT_IDS == tuple(range(8))
    assert FRAME_HELD_VARIANT_IDS == tuple(range(8, 12))
    assert [row[1] for row in FRAME_VARIANT_SPECS] == ["frame"] * 8 + ["frame_held"] * 4


def test_render_frame_variants_maps_name_span_exactly_once():
    variants = render_frame_variants(NAME)
    assert len(variants) == 12
    for variant in variants:
        prompt = variant["prompt"]
        assert prompt.count(NAME) == 1
        start, end = variant["name_char_span"]
        assert prompt[start:end] == NAME
        # Natural prose: no frozen header markers or brackets around the name.
        assert "Stock Ticker" not in prompt and "Stock Name" not in prompt
        assert f"[{NAME}]" not in prompt
    assert render_frame_variants(NAME) == variants


def test_template_control_contains_neutral_name_once_and_no_company_identity():
    assert TEMPLATE_CONTROL_PROMPT.count(TEMPLATE_CONTROL_NAME) == 1
    assert NAME not in TEMPLATE_CONTROL_PROMPT
    assert "[NEUT]" in TEMPLATE_CONTROL_PROMPT


def test_prepare_inputs_v2_frames_maps_spans_and_validates(tmp_path):
    prepared = prepare_inputs(localization_family="v2-frames", **_prepare_kwargs(tmp_path))
    config = prepared["config"]
    assert config["version"] == "v2"
    assert config["localization_family"] == "v2-frames"
    frames = prepared["frame_variants"]
    assert len(frames) == 2 * FRAME_VARIANT_COUNT
    template = prepared["template_control"]
    assert template["artifact_type"] == "entity_cell_template_control"
    # Character-level tokenizer: formatted prefix "<user>" is 6 chars, so the
    # name token span equals its character span shifted by 6.
    for row in frames:
        prompt = row["prompt"]
        raw_start = prompt.find(row["name"])
        span = row["company_name_content_token_span"]
        assert span["char_start"] == 6 + raw_start
        assert span["char_end"] == 6 + raw_start + len(row["name"])
        assert span["token_start"] == span["char_start"]
        assert span["final_content_token"] == span["char_end"] - 1
        assert row["final_company_name_content_token"] == span["final_content_token"]
        assert len(row["input_ids"]) == row["token_count"] == len(e1_pipeline.format_prompt(_Tokenizer(), prompt, use_chat_template=True, enable_thinking=False))
    template_span = template["company_name_content_token_span"]
    assert template_span["char_start"] == 6 + TEMPLATE_CONTROL_PROMPT.find(TEMPLATE_CONTROL_NAME)
    validate_prepared_inputs(prepared)


def test_prepare_inputs_default_family_stays_v1_without_frame_outputs(tmp_path):
    prepared = prepare_inputs(**_prepare_kwargs(tmp_path))
    assert prepared["config"]["version"] == "v1"
    assert prepared["config"]["localization_family"] == "v1-header"
    assert "frame_variants" not in prepared
    assert "template_control" not in prepared
    validate_prepared_inputs(prepared)
    with pytest.raises(ValueError, match="unknown localization family"):
        prepare_inputs(localization_family="v3", **_prepare_kwargs(tmp_path))


def test_prepare_artifacts_v2_writes_registers_and_cli(tmp_path):
    root = prepare_artifacts(
        localization_family="v2-frames",
        **_prepare_kwargs(tmp_path),
        model="fake-model",
        run_id="prep-v2",
        artifact_root=tmp_path / "artifacts",
    )
    metadata = json.loads((root / "prepare" / "metadata.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert metadata["version"] == "v2"
    assert metadata["localization_family"] == "v2-frames"
    assert metadata["prepared_artifact_sha256"]["frame_variants"] == sha256_file(root / "prepare" / "frame_variants.jsonl")
    assert (root / "prepare" / "template_control.json").is_file()
    assert manifest["status"] == "complete"
    types = [ref["artifact_type"] for ref in manifest["artifacts"]]
    assert "entity_cell_frame_variant" in types
    assert "entity_cell_template_control" in types
    args = build_parser().parse_args(
        ["prepare", "--input", "i", "--split-manifest", "s", "--baseline", "b", "--baseline-identity", "a", "--model", "m", "--run-id", "r", "--localization-family", "v2-frames"]
    )
    assert args.localization_family == "v2-frames"


def test_frame_control_vectors_retokense_from_scratch(tmp_path, monkeypatch):
    prepared = prepare_inputs(localization_family="v2-frames", **_prepare_kwargs(tmp_path))
    rows = [row for row in prepared["frame_variants"] if row["ticker"] == "ALFA" and row["variant_family"] == "frame"]
    original_ids = {tuple(row["input_ids"]) for row in rows}
    captured = []

    def fake_record(model, ids, *, layers, position):
        captured.append((ids[0].tolist(), position))
        return {layer: torch.zeros(4) for layer in layers}

    monkeypatch.setattr(e1_pipeline, "record_post_swiglu", fake_record)
    e1_pipeline._frame_control_vectors(_Tokenizer(), object(), rows, "ZZ", target_device="cpu")
    assert len(captured) == len(rows)
    for row, (ids, position) in zip(rows, captured):
        # The control prompt must be re-tokenized, never the original input_ids.
        assert tuple(ids) not in original_ids
        control_prompt = row["prompt"].replace(NAME, "ZZ")
        assert ids == [ord(char) % 251 for char in e1_pipeline.format_prompt(_Tokenizer(), control_prompt, use_chat_template=True, enable_thinking=False)]
        expected = 6 + control_prompt.find("ZZ") + 1  # last token of the two-char replacement
        assert position == expected
    # A name occurring twice is rejected before any model call.
    duplicated = {**rows[0], "prompt": rows[0]["prompt"].replace(NAME, NAME + " " + NAME)}
    with pytest.raises(ValueError, match="exactly once"):
        e1_pipeline._frame_control_vectors(_Tokenizer(), object(), [duplicated], "ZZ", target_device="cpu")


def _cells(pairs):
    return [{"layer": layer, "neuron": neuron} for layer, neuron in pairs]


def test_select_wrong_entity_cell_rule():
    cells_by_ticker = {
        "AA": _cells([(1, 2), (1, 2), (9, 9), (9, 9), (9, 9)]),
        "BB": _cells([(1, 2), (2, 3), (4, 4), (4, 4), (4, 4)]),
        "CC": _cells([(7, 7), (8, 8), (9, 9), (9, 9), (9, 9)]),
    }
    target = {"layer": 1, "neuron": 2}
    wrong, degenerate = select_wrong_entity_cell("AA", ["AA", "BB", "CC"], cells_by_ticker, target)
    assert (wrong["layer"], wrong["neuron"]) == (2, 3)
    assert degenerate is False
    # All of BB's top-5 equal the target -> degenerate control.
    cells_by_ticker["BB"] = _cells([(1, 2)] * 5)
    wrong, degenerate = select_wrong_entity_cell("AA", ["AA", "BB", "CC"], cells_by_ticker, target)
    assert degenerate is True
    assert wrong["layer"] == 1 and wrong["neuron"] == 2
    # Wrap-around: the alphabetically next ticker of the last one is the first.
    wrong, _ = select_wrong_entity_cell("CC", ["AA", "BB", "CC"], cells_by_ticker, {"layer": 7, "neuron": 7})
    assert wrong["layer"] == 1
    with pytest.raises(ValueError, match="target ticker"):
        select_wrong_entity_cell("ZZ", ["AA", "BB", "CC"], cells_by_ticker, target)


def test_frame_surface_control_summary_requires_absence_from_both_controls():
    candidates = _cells([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    controls = {
        "anonymous_name_frames": _cells([(1, 2), (9, 1), (9, 2), (9, 3), (9, 4)]),
        "name_form_control_frames": _cells([(9, 1), (9, 2), (9, 3), (9, 4), (9, 5)]),
    }
    summary = frame_surface_control_summary(candidates, controls)
    assert summary["anonymous_name_frames"]["candidate_in_top5"] is True
    assert summary["name_form_control_frames"]["candidate_in_top5"] is False
    assert summary["form_robust"] is False
    clean = dict(controls)
    clean["anonymous_name_frames"] = _cells([(9, 1), (9, 2), (9, 3), (9, 4), (9, 5)])
    assert frame_surface_control_summary(candidates, clean)["form_robust"] is True


def test_v2_candidate_eligibility_binds_all_four_gates():
    held = {"top5_overlap": 2, "top1_agreement": True}
    surface = {"form_robust": True}
    candidates = _cells([(3, 11), (4, 12), (5, 13), (6, 14), (7, 15)])
    signature = [{"layer": 9, "neuron": 9}, {"layer": 9, "neuron": 10}]
    amnesia = {"trusted_candidate_entity_cell": True, "exclusion_reasons": []}
    result = v2_candidate_eligibility(held_metrics=held, surface_control_summary=surface, candidate_cells=candidates, template_signature=signature, amnesia_summary=amnesia)
    assert result["eligible"] is True
    assert result["label"] == "trusted candidate entity cell"
    assert v2_candidate_eligibility(held_metrics={"top5_overlap": 0}, surface_control_summary=surface, candidate_cells=candidates, template_signature=signature, amnesia_summary=amnesia)["exclusion_reasons"] == ["held_variant_top5_overlap_zero"]
    assert "not_form_robust" in v2_candidate_eligibility(held_metrics=held, surface_control_summary={"form_robust": False}, candidate_cells=candidates, template_signature=signature, amnesia_summary=amnesia)["exclusion_reasons"]
    # Set semantics: any candidate top-5 cell in the template signature blocks.
    assert "in_template_signature" in v2_candidate_eligibility(held_metrics=held, surface_control_summary=surface, candidate_cells=candidates, template_signature=[{"layer": 7, "neuron": 15}], amnesia_summary=amnesia)["exclusion_reasons"]
    blocked = v2_candidate_eligibility(held_metrics=held, surface_control_summary=surface, candidate_cells=candidates, template_signature=signature, amnesia_summary={"trusted_candidate_entity_cell": False, "exclusion_reasons": ["eligible_prompt_count_below_2"]})
    assert blocked["eligible"] is False
    assert "eligible_prompt_count_below_2" in blocked["exclusion_reasons"]


def test_rank_absolute_activations_orders_by_abs_z():
    stats = {}
    for layer in CANDIDATE_LAYERS[:2]:
        accumulator = OnlineVectorStats()
        accumulator.update(torch.zeros(4))
        accumulator.update(torch.zeros(4))
        stats[layer] = accumulator
    vectors = {
        next(iter(stats)): torch.tensor([0.0, 0.0, 0.0, 0.0]),
        max(stats): torch.tensor([-5.0, 0.0, 1.0, 0.0]),  # |z| = 5 after epsilon-stabilized std
    }
    ranked = rank_absolute_activations(vectors, stats, top_k=5)
    assert ranked[0]["neuron"] == 0
    assert ranked[0]["layer"] == max(stats)
    assert ranked[0]["abs_z"] >= 1.0
    assert [row["rank"] for row in ranked] == [1, 2, 3, 4, 5]


def test_amnesia_curve_skips_wrong_entity_when_degenerate(monkeypatch):
    import llm_bias.entity_cell.mlp_cells as mlp_cells

    monkeypatch.setattr(mlp_cells, "mlp_hooks", lambda *args, **kwargs: nullcontext())
    prompt = f"Use the report. Stock Ticker: [ALFA] Stock Name: [{NAME}]"
    rows = [{"ticker": "ALFA", "name": NAME, "prompt": prompt, "prompt_id": "p1"}]
    candidate = {"layer": 2, "neuron": 3}
    output = run_amnesia_curve(
        object(), _Tokenizer(),
        prompt_rows=rows, candidate=candidate, wrong_candidate=candidate,
        random_neuron=7, wrong_degenerate=True,
        score_fn=lambda _prompt: 1.0,
    )
    assert {row["candidate"] for row in output} == {"target", "matched_random"}
    assert all(row["wrong_entity_degenerate"] is True for row in output)
    output = run_amnesia_curve(
        object(), _Tokenizer(),
        prompt_rows=rows, candidate=candidate,
        wrong_candidate={"layer": 2, "neuron": 9},
        random_neuron=7,
        score_fn=lambda _prompt: 1.0,
    )
    assert {row["candidate"] for row in output} == {"target", "wrong_entity", "matched_random"}
    assert all(row["wrong_entity_degenerate"] is False for row in output)
    # ROT13 surface form is still applied to names with mixed content.
    assert codecs.decode(NAME, "rot_13") == "Nycun Flfgrzf, Vap."
