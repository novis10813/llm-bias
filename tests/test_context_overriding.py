"""Deterministic regression tests for B V1 context overriding."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.artifacts.io import write_jsonl
from llm_bias.jspace_intervention import cli
from llm_bias.jspace_intervention import context_overriding as context
from llm_bias.jspace_intervention import cross_sector_patching as cross

BUY_ID = 240
SELL_ID = 241


class _CharTokenizer:
    chat_template = "fake"

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, **_kwargs):
        del add_special_tokens
        for suffix, token in ((" Buy", BUY_ID), (" Sell", SELL_ID), ("buy", BUY_ID), ("sell", SELL_ID)):
            if text.endswith(suffix):
                prefix = text[:-len(suffix)]
                values = [ord(char) for char in prefix] + [token]
                offsets = [(i, i + 1) for i in range(len(prefix))] + [(len(prefix), len(text))]
                break
        else:
            values = [ord(char) for char in text]
            offsets = [(i, i + 1) for i in range(len(text))]
        if return_offsets_mapping:
            return SimpleNamespace(input_ids=values, offset_mapping=offsets, special_tokens_mask=[0] * len(values))
        return SimpleNamespace(input_ids=values)

    def apply_chat_template(self, messages, **_kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"


class _SumLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden + hidden.sum(dim=1, keepdim=True)


class _RMS(torch.nn.Module):
    variance_epsilon = 1e-6

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(1))


class _PatchingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 1)
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), _SumLayer()])
        self.n_layers = 2
        self._final_norm = _RMS()
        self._lm_head = torch.nn.Linear(1, 256, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            for token in (ord("a"), ord("b")):
                self.embedding.weight[token, 0] = 1
            for token in (ord("y"), ord("z")):
                self.embedding.weight[token, 0] = -1
            self._lm_head.weight.zero_()
            self._lm_head.weight[BUY_ID, 0] = 1
            self._lm_head.weight[SELL_ID, 0] = -1

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _prepared_records(tmp_path: Path) -> Path:
    rows, assignments = _cross_rows()
    prepared = cross.prepare_cross_sector_pairs(
        rows, assignments, split_name="discovery", seed=7, max_pairs=1
    )
    path = tmp_path / "prepared.jsonl"
    write_jsonl(path, prepared, overwrite=False)
    return path


def _cross_rows() -> tuple[list[dict], dict[str, str]]:
    def row(ticker, name, sector, origin, trial, valence="both"):
        return {
            "condition": "attribute",
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "trial_key": trial,
            "trial_index": 0,
            "set_index": 0,
            "evidence_origin_sector": origin,
            "prompt": "ignored",
            "evidence": [
                {"side": "buy", "kind": "qual", "text": "growth outlook"},
                {"side": "buy", "kind": "quant", "text": "revenue rises 5 percent"},
                {"side": "sell", "kind": "qual", "text": "risk outlook"},
                {"side": "sell", "kind": "quant", "text": "margin falls 5 percent"},
            ],
        }

    rows = [
        row("T1", "Tech One", "Technology", "Technology", "t1-tech"),
        row("T1", "Tech One", "Technology", "Financial Services", "t1-fin"),
        row("T2", "Tech Two", "Technology", "Technology", "t2-tech"),
        row("F1", "Finance One", "Financial Services", "Technology", "f1-tech"),
        row("F1", "Finance One", "Financial Services", "Financial Services", "f1-fin"),
        row("F2", "Finance Two", "Financial Services", "Technology", "f2-tech"),
    ]
    return rows, {ticker: "discovery" for ticker in ("T1", "T2", "F1", "F2")}


def test_load_filters_negative_evidence_only(tmp_path: Path) -> None:
    path = _prepared_records(tmp_path)
    loaded = context.load_prepared_context_records(path)
    assert loaded
    assert {row["evidence_valence"] for row in loaded} == {"negative"}


def test_instruction_context_excludes_final_position() -> None:
    tokenizer = _CharTokenizer()
    prompt = "HDR|aaaa|bbbb|INSTR"
    spans = {
        "qual": [prompt.index("aaaa"), prompt.index("aaaa") + 4],
        "quant": [prompt.index("bbbb"), prompt.index("bbbb") + 4],
    }
    from llm_bias.jspace_intervention.activation_patching import (
        _build_position_mapping,
        resolve_prompt_spans,
    )

    source = resolve_prompt_spans(tokenizer, prompt, spans)
    mapping = _build_position_mapping(source, source, "instruction_context", len(prompt), len(prompt))
    assert mapping
    assert max(mapping) == len(prompt) - 2
    assert len(prompt) - 1 not in mapping


def test_ids_are_unique_across_controls_directions_and_spans(tmp_path: Path, monkeypatch) -> None:
    path = _prepared_records(tmp_path)
    rows = context.load_prepared_context_records(path)
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    model.input_device = "cpu"
    monkeypatch.setattr(context, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(context, "load_model", lambda _name: (model, tokenizer, "cpu"))
    run_dir = context.run_cross_sector_context_overriding_pipeline(
        prepared_pairs=path,
        model_name="fake-model",
        run_id="ids",
        layers=[0],
        artifact_root=tmp_path / "artifacts",
        decision_prefix="",
    )
    records = [json.loads(line) for line in (run_dir / "forward/context_overriding_records.jsonl").read_text().splitlines()]
    assert records
    assert len({record["record_id"] for record in records}) == len(records)
    assert {record["span_condition"] for record in records} == set(context.DEFAULT_SPANS)
    assert {record["patching_direction"] for record in records} >= {
        "Technology_to_Financial Services",
        "Financial Services_to_Technology",
    }
    assert len(records) == len(rows) * 3 * 2 - sum(row["control_type"] == "self_source" for row in rows) * 3


def test_self_source_context_patch_is_no_op(tmp_path: Path, monkeypatch) -> None:
    path = _prepared_records(tmp_path)
    prepared = next(row for row in context.load_prepared_context_records(path) if row["control_type"] == "self_source")
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    model.input_device = "cpu"
    monkeypatch.setattr(context, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(context, "load_model", lambda _name: (model, tokenizer, "cpu"))
    run_dir = context.run_cross_sector_context_overriding_pipeline(
        prepared_pairs=path,
        model_name="fake-model",
        run_id="self",
        layers=[0],
        spans=["instruction_context"],
        artifact_root=tmp_path / "artifacts",
        decision_prefix="",
        max_records=next(i for i, row in enumerate(context.load_prepared_context_records(path), 1) if row["record_id"] == prepared["record_id"]),
    )
    records = [json.loads(line) for line in (run_dir / "forward/context_overriding_records.jsonl").read_text().splitlines()]
    self_records = [row for row in records if row["control_type"] == "self_source"]
    assert self_records
    assert all(row["delta_margin"] == pytest.approx(0.0) for row in self_records)


def test_analysis_reports_equal_pair_means_and_paired_contrasts() -> None:
    records = []
    for pair_id, primary, header, final in (("p1", 3.0, 1.0, 0.5), ("p2", 5.0, 1.0, 1.5)):
        for span, delta in (("instruction_context", primary), ("header", header), ("final_position", final)):
            records.append({
                "pair_id": pair_id,
                "layer": 14,
                "span_condition": span,
                "patching_direction": "Technology_to_Financial Services",
                "evidence_origin_sector": "Technology",
                "control_type": "cross_sector",
                "delta_margin": delta,
                "flip": False,
                "normalized_transfer": None,
            })
    result = context.analyze_context_overriding_records(records)
    means = {(row["span_condition"], row["control_type"]): row for row in result["equal_pair_means"]}
    assert means[("instruction_context", "cross_sector")]["equal_pair_mean_delta_margin"] == pytest.approx(4.0)
    contrasts = {row["contrast"]: row for row in result["paired_contrasts"]}
    assert contrasts["primary_minus_header"]["equal_pair_mean_contrast"] == pytest.approx(3.0)
    assert contrasts["primary_minus_final"]["equal_pair_mean_contrast"] == pytest.approx(3.0)


def test_pipeline_and_cli_fake_model(tmp_path: Path, monkeypatch) -> None:
    path = _prepared_records(tmp_path)
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    model.input_device = "cpu"
    monkeypatch.setattr(context, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(context, "load_model", lambda _name: (model, tokenizer, "cpu"))
    run_dir = context.run_cross_sector_context_overriding_pipeline(
        prepared_pairs=path,
        model_name="fake-model",
        run_id="pipeline",
        layers=[0],
        spans=["instruction_context", "header", "final_position"],
        artifact_root=tmp_path / "artifacts",
        decision_prefix="",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "analyze/summary.json").read_text())
    assert manifest["status"] == "complete"
    assert summary["formal_success_gate"] is False
    assert summary["paired_contrasts"]
    serialized = (run_dir / "forward/context_overriding_records.jsonl").read_text().lower()
    assert "residual" not in serialized
    assert "activation" not in serialized

    captured = {}
    monkeypatch.setattr(context, "run_cross_sector_context_overriding_pipeline", lambda **kwargs: captured.update(kwargs) or tmp_path / "run")
    monkeypatch.setattr(sys, "argv", [
        "jspace-intervention", "run-cross-sector-context-overriding",
        "--prepared-pairs", str(path), "--model", "fake-model", "--run-id", "cli",
        "--layers", "14-16,20", "--span", "instruction_context", "--span", "header",
        "--max-records", "2",
    ])
    cli.main()
    assert captured["layers"] == [14, 15, 16, 20]
    assert captured["spans"] == ("instruction_context", "header")
    assert captured["dataset"] == "cross-sector-context-overriding"
