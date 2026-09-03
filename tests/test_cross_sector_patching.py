"""Deterministic regression tests for A V1 cross-sector header patching."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from llm_bias.core.artifacts.io import write_jsonl
from llm_bias.jspace_intervention import cli
from llm_bias.jspace_intervention import cross_sector_patching as cross
from llm_bias.jspace_intervention.activation_patching import run_activation_patching_record

BUY_ID = 240
SELL_ID = 241


class _CharTokenizer:
    chat_template = "fake"

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, **_kwargs):
        del add_special_tokens
        suffix = next(((value, token) for value, token in ((" Buy", BUY_ID), (" Sell", SELL_ID), ("buy", BUY_ID), ("sell", SELL_ID)) if text.endswith(value)), None)
        if suffix is None:
            values = [ord(char) for char in text]
            offsets = [(i, i + 1) for i in range(len(text))]
        else:
            value, token = suffix
            prefix = text[:-len(value)]
            values = [ord(char) for char in prefix] + [token]
            offsets = [(i, i + 1) for i in range(len(prefix))] + [(len(prefix), len(text))]
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

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight


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


def _row(ticker: str, name: str, sector: str, origin: str, trial: str, *, leak: bool = False) -> dict:
    suffix = f" for {ticker}" if leak else ""
    return {
        "condition": "attribute",
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "trial_key": trial,
        "trial_index": 0,
        "set_index": 0,
        "evidence_origin_sector": origin,
        "prompt": "legacy prompt is ignored",
        "evidence": [
            {"side": "buy", "kind": "qual", "text": f"growth outlook{suffix}"},
            {"side": "buy", "kind": "quant", "text": "revenue rises 5 percent"},
            {"side": "sell", "kind": "qual", "text": "risk outlook"},
            {"side": "sell", "kind": "quant", "text": "margin falls 5 percent"},
        ],
    }


def _rows() -> tuple[list[dict], dict[str, str]]:
    unbalanced = _row("T1", "Tech One", "Technology", "Technology", "t1-unbalanced")
    unbalanced["evidence"][0]["side"] = "sell"
    rows = [
        unbalanced,
        _row("T1", "Tech One", "Technology", "Technology", "t1-tech"),
        _row("T1", "Tech One", "Technology", "Financial Services", "t1-fin"),
        _row("T2", "Tech Two", "Technology", "Technology", "t2-tech"),
        _row("F1", "Finance One", "Financial Services", "Technology", "f1-tech"),
        _row("F1", "Finance One", "Financial Services", "Financial Services", "f1-fin"),
        _row("F2", "Finance Two", "Financial Services", "Technology", "f2-tech"),
        _row("T1", "Tech One", "Technology", "Technology", "t1-leak", leak=True),
    ]
    assignments = {ticker: "discovery" for ticker in ("T1", "T2", "F1", "F2")}
    return rows, assignments


def test_preparation_is_deterministic_balanced_and_byte_matched() -> None:
    rows, assignments = _rows()
    first = cross.prepare_cross_sector_pairs(rows, assignments, split_name="discovery", seed=7, max_pairs=1)
    second = cross.prepare_cross_sector_pairs(rows, assignments, split_name="discovery", seed=7, max_pairs=1)

    assert first == second
    assert len({row["record_id"] for row in first}) == len(first)
    cross_rows = [row for row in first if row["control_type"] == "cross_sector"]
    assert {row["evidence_origin_sector"] for row in cross_rows} == set(cross.SECTORS)
    assert {row["evidence_valence"] for row in cross_rows} == {"positive", "negative"}
    for row in cross_rows:
        marker = "--- Evidence ---"
        source_tail = row["source_prompt"][row["source_prompt"].index(marker):]
        target_tail = row["target_prompt"][row["target_prompt"].index(marker):]
        assert source_tail == target_tail
        assert row["trial_row_sha256"]
        assert len(row["evidence_item_hashes"]) == 4
    assert all(row["source_trial_key"] != "t1-leak" for row in first)
    assert all(row["trial_row_sha256"] for row in first)
    name_form = next(row for row in first if row["control_type"] == "name_form")
    assert len(name_form["source_identity"]["ticker"]) == len(name_form["target_identity"]["ticker"])
    assert len(name_form["source_identity"]["name"]) == len(name_form["target_identity"]["name"])
    assert name_form["source_identity"]["ticker"] != name_form["target_identity"]["ticker"]


def test_preparation_supports_explicit_smoke_pair() -> None:
    rows, assignments = _rows()
    assignments["T1"] = "calibration"
    with pytest.raises(ValueError, match="not an eligible Technology ticker"):
        cross.prepare_cross_sector_pairs(
            rows, assignments, split_name="discovery", seed=7,
            source_ticker="T1", target_ticker="F1"
        )
    prepared = cross.prepare_cross_sector_pairs(
        rows, assignments, split_name="discovery", seed=7,
        source_ticker="T1", target_ticker="F1", allow_cross_split_smoke=True
    )
    cross_rows = [row for row in prepared if row["control_type"] == "cross_sector"]
    assert cross_rows
    assert {
        (row["source_identity"]["ticker"], row["target_identity"]["ticker"])
        for row in cross_rows
    } == {("T1", "F1")}
    assert all(row["cross_split_smoke"] for row in cross_rows)
    assert {row["source_identity_split"] for row in cross_rows} == {"calibration"}
    assert {row["target_identity_split"] for row in cross_rows} == {"discovery"}


def test_preparation_truncates_unequal_sector_ticker_counts() -> None:
    rows, assignments = _rows()
    rows.append(
        _row(
            "F3", "Finance Three", "Financial Services",
            "Financial Services", "f3-fin"
        )
    )
    assignments["F3"] = "discovery"
    prepared = cross.prepare_cross_sector_pairs(
        rows, assignments, split_name="discovery", seed=7
    )
    assert prepared
    assert len({row["pair_id"] for row in prepared}) <= 2


def test_target_complete_header_mapping_covers_unequal_target() -> None:
    mapping = cross.target_complete_header_mapping((0, 4), (2, 9))
    assert list(mapping) == [2, 3, 4, 5, 6, 7, 8]
    assert set(mapping.values()) == {0, 1, 2, 3}


def test_self_source_header_patch_is_no_op() -> None:
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    prompt = "HDR[qual=abcd][quant=12]INSTR"
    chars = {
        "qual": [prompt.index("abcd"), prompt.index("abcd") + 4],
        "quant": [prompt.index("12"), prompt.index("12") + 2],
    }
    record = run_activation_patching_record(
        model=model, tokenizer=tokenizer, source_prompt=prompt, target_prompt=prompt,
        source_evidence_char_spans=chars, target_evidence_char_spans=chars,
        layers=[0], span_condition="header", device="cpu",
    )
    assert record["patched_margin"] == pytest.approx(record["target_clean_margin"])
    assert record["delta_margin"] == pytest.approx(0.0)
    assert record["flip"] is False


def test_pipeline_writes_compact_complete_run(monkeypatch, tmp_path: Path) -> None:
    rows, assignments = _rows()
    prepared = cross.prepare_cross_sector_pairs(rows, assignments, split_name="discovery", seed=7, max_pairs=1)
    prepared_path = tmp_path / "prepared.jsonl"
    write_jsonl(prepared_path, prepared, overwrite=False)
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    model.input_device = "cpu"
    monkeypatch.setattr(cross, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(cross, "load_model", lambda _name: (model, tokenizer, "cpu"))

    run_dir = cross.run_cross_sector_header_patching_pipeline(
        prepared_pairs=prepared_path, model_name="fake-model", run_id="cross-smoke",
        layers=[0], decision_prefix="", artifact_root=tmp_path / "artifacts",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in (run_dir / "forward" / "header_patch_records.jsonl").read_text().splitlines()]
    summary = json.loads((run_dir / "analyze" / "summary.json").read_text())
    assert manifest["status"] == "complete"
    assert records and records[0]["artifact_type"] == cross.RECORD_ARTIFACT_TYPE
    assert len({record["record_id"] for record in records}) == len(records)
    assert {record["control_type"] for record in records} >= {
        "cross_sector", "same_sector_peer", "name_form", "self_source"
    }
    assert set(records[0]) >= {"source_identity", "target_identity", "delta_margin", "positions_patched", "trial_row_sha256"}
    serialized = json.dumps(records).lower()
    assert "source_residual" not in serialized
    assert "activation" not in serialized
    assert summary["formal_success_gate"] is False


def test_cli_dispatches_prepare_and_run(monkeypatch, tmp_path: Path) -> None:
    rows, assignments = _rows()
    raw_path = tmp_path / "raw.jsonl"
    split_path = tmp_path / "splits.json"
    output_path = tmp_path / "pairs.jsonl"
    write_jsonl(raw_path, rows, overwrite=False)
    split_path.write_text(json.dumps({"artifact_type": "jspace_intervention_splits", "assignments": assignments}))
    monkeypatch.setattr(sys, "argv", [
        "jspace-intervention", "prepare-cross-sector-patching", "--raw-trials", str(raw_path),
        "--split-manifest", str(split_path), "--split", "discovery", "--output", str(output_path),
        "--max-pairs", "1", "--max-records", "1",
    ])
    cli.main()
    assert output_path.is_file()

    captured = {}
    monkeypatch.setattr(cross, "run_cross_sector_header_patching_pipeline", lambda **kwargs: captured.update(kwargs) or tmp_path / "run")
    monkeypatch.setattr(sys, "argv", [
        "jspace-intervention", "run-cross-sector-header-patching", "--prepared-pairs", str(output_path),
        "--model", "fake-model", "--run-id", "cross", "--layers", "0-2,4", "--max-records", "1",
    ])
    cli.main()
    assert captured["layers"] == [0, 1, 2, 4]
    assert captured["dataset"] == "cross-sector-header-patching"
