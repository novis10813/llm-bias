import ast
import csv
import json
from pathlib import Path

import pytest

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.continuation_scoring import CandidateMargin, CandidateScore
from llm_bias.span_sensitivity.analysis import grouped_effects
from llm_bias.span_sensitivity.conditions import (
    CONDITIONS,
    Identity,
    build_conditions,
    name_form_identity,
)
from llm_bias.span_sensitivity import pipeline
from llm_bias.span_sensitivity.cli import build_parser


PROMPT = """Refer to the evidence below to make a final investment decision.
Stock Ticker: [{ticker}]
Stock Name: [{name}]
--- Evidence ---
1. {ticker} announced a product update for {name}.
---
Respond with one valid JSON object containing only the keys \"decision\" (buy | sell) and \"reason\". Do not choose hold."""


class _Tokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, **_kwargs):
        return f"USER:{messages[0]['content']}\nASSISTANT:"

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) % 251 for char in text]}


def _margin(value: float) -> CandidateMargin:
    return CandidateMargin(
        CandidateScore("buy", [1], value / 2, 1),
        CandidateScore("sell", [2], -value / 2, 1),
    )


def test_header_conditions_change_only_requested_fields():
    source = Identity("ALFA", "Alpha Systems, Inc.")
    peer = Identity("BETA", "Beta Logic, Inc.")
    prompt = PROMPT.format(ticker=source.ticker, name=source.name)

    rows = build_conditions(prompt, source=source, peer=peer, key="p1", seed=3)

    assert [row.condition for row in rows] == list(CONDITIONS)
    assert rows[0].prompt == prompt
    anonymous_ticker = rows[1]
    assert "Stock Ticker: [ANON]" in anonymous_ticker.prompt
    assert "Stock Name: [Alpha Systems, Inc.]" in anonymous_ticker.prompt
    assert "ALFA announced" in anonymous_ticker.prompt
    swapped = rows[4]
    assert "Stock Ticker: [BETA]" in swapped.prompt
    assert "Stock Name: [Beta Logic, Inc.]" in swapped.prompt
    assert "ALFA announced" in swapped.prompt
    assert swapped.ticker_mentions_outside_header == 1
    assert swapped.name_mentions_outside_header == 1


def test_name_form_control_preserves_surface_shape():
    source = Identity("AbC1", "Alpha-Tech, Inc.")
    transformed = name_form_identity(source)
    assert transformed != source
    assert len(transformed.ticker) == len(source.ticker)
    assert len(transformed.name) == len(source.name)
    assert [char.isupper() for char in transformed.name] == [char.isupper() for char in source.name]
    assert [char.isalpha() for char in transformed.name] == [char.isalpha() for char in source.name]


def test_conditions_reject_mismatched_header_metadata():
    prompt = PROMPT.format(ticker="ALFA", name="Alpha Systems, Inc.")
    with pytest.raises(ValueError, match="does not match CSV metadata"):
        build_conditions(
            prompt,
            source=Identity("WRONG", "Alpha Systems, Inc."),
            peer=Identity("BETA", "Beta Logic, Inc."),
            key="p1",
        )


def test_grouped_effects_use_equal_ticker_means_and_holm_correction():
    rows = []
    for ticker, deltas in {"A": [1.0, 3.0], "B": [-1.0]}.items():
        for index, delta in enumerate(deltas):
            prompt_id = f"{ticker}-{index}"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "ticker": ticker,
                    "condition": "original",
                    "margin": 1.0,
                    "original_margin": 1.0,
                    "delta_margin": 0.0,
                }
            )
            for condition in CONDITIONS[1:]:
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "ticker": ticker,
                        "condition": condition,
                        "margin": 1.0 + delta,
                        "original_margin": 1.0,
                        "delta_margin": delta,
                    }
                )

    summaries = grouped_effects(rows, seed=7, bootstrap_samples=100)

    assert len(summaries) == 6
    assert summaries[0]["mean_delta_margin"] == pytest.approx(0.5)
    assert summaries[0]["ticker_count"] == 2
    assert summaries[0]["prompt_count"] == 3
    assert all("holm_adjusted_pvalue" in row for row in summaries)


def _write_input(path: Path) -> None:
    fields = [
        "Date", "ticker", "name", "sector", "marketcap",
        "prompt_with_context_attribute_0",
    ]
    records = [
        ("ALFA", "Alpha Systems, Inc."),
        ("BETA", "Beta Logic, Inc."),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ticker, name in records:
            writer.writerow(
                {
                    "Date": "2026-01-01",
                    "ticker": ticker,
                    "name": name,
                    "sector": "Technology",
                    "marketcap": "100",
                    "prompt_with_context_attribute_0": PROMPT.format(ticker=ticker, name=name),
                }
            )


def test_pipeline_writes_prepare_forward_analysis_and_finalizes(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    split = tmp_path / "splits.json"
    _write_input(source)
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "test_splits",
                "input_sha256": sha256_file(source),
                "seed": 4,
                "assignments": {"ALFA": "discovery", "BETA": "discovery"},
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _Tokenizer()
    monkeypatch.setattr(pipeline, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(
        pipeline,
        "load_model",
        lambda _name, device_map=None: (object(), tokenizer, "cpu"),
    )

    def fake_score(_model, _tokenizer, prompt, _positive, _negative, **_kwargs):
        if "Stock Ticker: [ANON]" in prompt and "Stock Name: [Anonymous Company]" in prompt:
            return _margin(0.5)
        if "Stock Ticker: [ANON]" in prompt:
            return _margin(0.75)
        if "Stock Name: [Anonymous Company]" in prompt:
            return _margin(1.25)
        return _margin(1.0)

    monkeypatch.setattr(pipeline, "score_margin", fake_score)

    root = pipeline.run_pipeline(
        input_path=source,
        split_manifest=split,
        model_name="fake-model",
        run_id="pilot",
        artifact_root=tmp_path / "artifacts",
        bootstrap_samples=50,
    )

    prepared = [json.loads(line) for line in (root / "prepare" / "prepared_prompts.jsonl").read_text().splitlines()]
    results = [json.loads(line) for line in (root / "forward" / "margin_results.jsonl").read_text().splitlines()]
    summary = json.loads((root / "analysis" / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(prepared) == 14
    assert len(results) == 14
    assert all(row["mutation_scope"] == "header_identity_fields_only" for row in results)
    assert len(summary["groups"]) == 6
    assert manifest["status"] == "complete"
    assert set(manifest["stages"]) == {"prepare", "forward", "analyze"}


def test_cli_defaults_to_technology_discovery():
    args = build_parser().parse_args(["run", "--run-id", "pilot"])
    assert args.sector == "Technology"
    assert args.split == "discovery"
    assert args.max_seq_len == 1024


def test_package_does_not_import_other_experiments():
    package = Path("llm_bias/span_sensitivity")
    forbidden = {
        "llm_bias.baseline_trial",
        "llm_bias.jspace_intervention",
        "llm_bias.prompt_analysis",
    }
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name == blocked or name.startswith(blocked + ".")
            for name in imported
            for blocked in forbidden
        ), source


def test_split_manifest_must_match_input(tmp_path):
    source = tmp_path / "input.csv"
    split = tmp_path / "splits.json"
    _write_input(source)
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_sha256": "0" * 64,
                "assignments": {"ALFA": "discovery", "BETA": "discovery"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not bound"):
        pipeline.run_pipeline(
            input_path=source,
            split_manifest=split,
            model_name="fake",
            run_id="bad",
            artifact_root=tmp_path / "artifacts",
        )
