"""Fake-model regression tests for evidence-insensitivity Phase 1 (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.artifacts.io import _RAW, write_json
from llm_bias.evidence_insensitivity import pipeline
from llm_bias.evidence_insensitivity.analysis import (
    classify_group,
    descriptive_stats,
    evaluate_gates,
    group_rows,
)
from llm_bias.evidence_insensitivity.population import load_population, make_splits
from llm_bias.evidence_insensitivity.screening import parse_decision
from llm_bias.evidence_insensitivity.template import CONDITIONS, build_prompt, prompt_char_spans

GENERATED_JSON = '{"decision": "buy", "reason": "fake"}'


class CharTokenizer:
    """Character-level stand-in tokenizer (prefix property, offset mapping)."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        result = {"input_ids": [ord(char) % 251 for char in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            result["special_tokens_mask"] = [0] * len(text)
        return result

    def decode(self, values, **_kwargs):
        return "".join(chr(value) for value in values)


class FakeScreenModel:
    """Minimal deterministic model for the screening contract.

    ``forward`` returns [1, seq, vocab] logits (the ``score_margin`` fallback
    path of ``score_single_token_margin_fp32``); ``hf_model.generate`` appends
    a fixed JSON decision suffix so ``parse_decision`` has valid input.
    """

    def __init__(self, vocab: int = 256):
        self.vocab = vocab
        self.hf_model = self
        self.config = SimpleNamespace(eos_token_id=None)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        sequence = int(input_ids.shape[1])
        logits = torch.zeros(1, sequence, self.vocab, dtype=torch.float32)
        for index in range(sequence):
            token = int(input_ids[0, index]) % self.vocab
            logits[0, index, token] = 1.0
        return SimpleNamespace(logits=logits, last_hidden_state=None)

    def __call__(self, input_ids, attention_mask=None, use_cache=False):
        return self.forward(input_ids, attention_mask=attention_mask, use_cache=use_cache)

    def generate(self, prompt_ids, **_kwargs):
        extra = torch.tensor([[ord(char) % 251 for char in GENERATED_JSON]], dtype=torch.long)
        return torch.cat([prompt_ids, extra], dim=1)


def _record(ticker, condition, arm, decision, margin, sector="Technology", split="discovery"):
    return {
        "ticker": ticker,
        "gics_sector": sector,
        "split": split,
        "arm": arm,
        "condition": condition,
        "decision": decision,
        "margin": margin,
    }


# ---------------------------------------------------------------------------
# Template and population
# ---------------------------------------------------------------------------


def test_frozen_prompt_and_spans_cover_prompt():
    prompt = build_prompt("AAA", "Alpha", "N15")
    assert "5% increase" in prompt and "15%" in prompt
    spans = prompt_char_spans(prompt)
    assert spans["header"][0] == 0
    assert spans["header"][1] == spans["evidence"][0]
    assert spans["evidence"][1] == spans["instruction"][0]
    assert spans["instruction"][1] == len(prompt)


def test_condition_matrix():
    assert CONDITIONS == ("zero", "N6", "N8", "N10", "N15", "P6", "P8", "P10", "P15")


def test_prepare_builds_exact_formal_and_pilot_counts():
    population = load_population()
    rows, provenance = pipeline._prepare_rows(population, CharTokenizer(), pilot=False, use_chat_template=False)
    assert len(rows) == 5336
    assert provenance["status"] == "formal"
    pilot_rows, pilot_provenance = pipeline._prepare_rows(population, CharTokenizer(), pilot=True, use_chat_template=False)
    assert len(pilot_rows) == 259
    assert pilot_provenance["status"] == "pilot"


def test_population_loader_and_splits_are_deterministic():
    rows = load_population()
    assert len(rows) == 503
    assert len({row["ticker"] for row in rows}) == 503
    first, second = make_splits(rows)
    assert first == make_splits(rows)[0]
    assert second == make_splits(rows)[1]
    assert len(second) == 100
    assert sum(value == "hold-out" for value in first.values()) in {100, 101}


def test_population_loader_fails_closed_on_duplicate(tmp_path):
    source = tmp_path / "population.csv"
    source.write_text(
        "index_name,year,ticker,company_name,gics_sector\n"
        "S&P 500,2024,A,Alpha,Technology\n" * 503,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_population(source)


# ---------------------------------------------------------------------------
# Screening parse
# ---------------------------------------------------------------------------


def test_parse_decision_valid_and_invalid():
    assert parse_decision('{"decision": "buy", "reason": "r"}') == "buy"
    assert parse_decision('{"decision": "sell", "reason": "r"}') == "sell"
    assert parse_decision("not json") is None
    assert parse_decision('{"decision": "hold", "reason": "r"}') is None
    assert parse_decision('["buy"]') is None
    assert parse_decision('{"reason": "no decision key"}') is None
    # Pilot-verified model format: pretty-printed JSON + trailing stop marker
    # (protocol §2 / Rev 1.1) -- the object must still parse.
    pretty = '{\n  "decision": "sell",\n  "reason": "r"\n}\n\n'
    assert parse_decision(pretty) == "sell"
    assert parse_decision("preamble text {\"decision\": \"buy\"} trailing garbage") == "buy"
    assert parse_decision('{"decision": "buy", "reason": "unterminated') is None


# ---------------------------------------------------------------------------
# Grouping (protocol §5)
# ---------------------------------------------------------------------------


def test_group_rule_boundaries_and_parse_failure_exclusion():
    assert classify_group("sell", "buy") == "evidence-responsive"
    assert classify_group("sell", "sell") == "fixed-sell"
    assert classify_group("buy", "buy") == "fixed-buy"
    assert classify_group("buy", "sell") == "mixed"
    assert classify_group(None, "buy") is None
    assert classify_group("sell", None) is None

    records = [
        _record("A", condition, "primary", decision, margin)
        for condition, decision, margin in [("zero", "sell", -1.0), ("N15", "sell", -1.0), ("P15", "buy", 1.0)]
    ] + [
        _record("B", condition, "primary", decision, margin)
        for condition, decision, margin in [("zero", None, -1.0), ("N15", "sell", -1.0), ("P15", "sell", -0.5)]
    ] + [
        _record("C", condition, "primary", decision, margin)
        for condition, decision, margin in [("zero", "sell", -1.0), ("N15", None, -1.0), ("P15", "buy", 1.0)]
    ]
    groups = {row["ticker"]: row for row in group_rows(records)}
    assert groups["A"]["group"] == "evidence-responsive"
    assert groups["A"]["prior_label"] is None
    assert groups["B"]["group"] == "fixed-sell"
    assert groups["B"]["prior_label"] is None  # d0 unparsed -> no prior label
    assert groups["C"]["group"] is None  # N15 parse failure -> excluded
    assert groups["C"]["prior_label"] is None
    # contrast_c is margin-based and independent of parse success
    assert groups["A"]["contrast_c"] == pytest.approx(2.0)
    assert groups["C"]["contrast_c"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Gates (protocol §6)
# ---------------------------------------------------------------------------


def test_gate_p2_zero_margin_counts_mismatch():
    records = [_record("A", "N15", "primary", "sell", 0.0)] * 10
    result = evaluate_gates(records, [], {"text_mismatch_count": 0, "max_abs_delta_margin": 0.0})
    assert result["gates"]["G-P2"]["value"] == 0.0
    assert result["gates"]["G-P2"]["pass"] is False

    records = [
        _record("A", "N15", "primary", "buy", 1.0),
        _record("A", "P15", "primary", "sell", -1.0),
        _record("B", "N15", "primary", "sell", 0.0),
    ]
    result = evaluate_gates(records, [], {"text_mismatch_count": 0, "max_abs_delta_margin": 0.0})
    assert result["gates"]["G-P2"]["value"] == pytest.approx(2 / 3)
    assert result["gates"]["G-P2"]["pass"] is False


def test_gate_thresholds_and_determinism():
    records = [_record("A", "N15", "primary", "buy", 1.0)] * 20
    groups = [{"group": "evidence-responsive"}] * 10 + [{"group": "fixed-sell"}] * 10
    result = evaluate_gates(records, groups, {"text_mismatch_count": 0, "max_abs_delta_margin": 0.0})
    assert all(result["gates"][name]["pass"] for name in ("G-P1", "G-P2", "G-P3", "G-P4"))
    assert result["fallback"] is False

    drifted = evaluate_gates(records, groups, {"text_mismatch_count": 0, "max_abs_delta_margin": 1e-6})
    assert drifted["gates"]["G-P3"]["pass"] is False


def test_gate_p4_branching_fallback():
    records = [_record("A", "N15", "primary", "sell", -1.0)] * 20
    groups = [{"group": "fixed-sell"}] * 20
    result = evaluate_gates(records, groups, {"text_mismatch_count": 0, "max_abs_delta_margin": 0.0})
    assert result["gates"]["G-P4"]["pass"] is False
    assert result["fallback"] is True


# ---------------------------------------------------------------------------
# Descriptive statistics (protocol §8)
# ---------------------------------------------------------------------------


def test_descriptive_stats_primary_only_curve_and_order_swap_block():
    records = [
        _record("A", "N15", "primary", "sell", -1.0),
        _record("B", "N15", "primary", "sell", -2.0),
        _record("B", "P15", "primary", "buy", 1.0),
        _record("A", "N15", "order_swap", "buy", 0.5),
        _record("TICKER", "zero", "anon", "sell", -3.0),
    ]
    groups = group_rows([row for row in records if row["arm"] == "primary" and row["ticker"] in {"A", "B"}])
    stats = descriptive_stats(records, groups)
    # order-swap and anonymous prompts must not pollute the population curve
    assert stats["population_polarity"]["N15"]["mean_margin"] == pytest.approx(-1.5)
    assert stats["population_polarity"]["P15"]["mean_margin"] == pytest.approx(1.0)
    assert stats["order_swap"]["n_pairs"] == 1
    assert stats["order_swap"]["decision_flip_rate"] == pytest.approx(1.0)
    assert stats["order_swap"]["decision_flip"]["sell_to_buy_count"] == 1
    assert stats["order_swap"]["mean_abs_delta_margin"] == pytest.approx(1.5)
    assert stats["anonymous_base_rate"]["zero"]["margin"] == pytest.approx(-3.0)
    # prefix-match statistic requires generated_text fields
    for row in records:
        row["generated_text"] = '{"decision": "buy", "reason": "r"}'
    assert descriptive_stats(records, groups)["generation_prefix_match"] == pytest.approx(1.0)
    # Pilot-verified pretty-printed format also counts (protocol §8 item 8)
    for row in records:
        row["generated_text"] = '{\n  "decision": "buy",\n  "reason": "r"\n}\n\n'
    assert descriptive_stats(records, groups)["generation_prefix_match"] == pytest.approx(1.0)
    records[0]["generated_text"] = '{\n  "reason": "first",\n  "decision": "buy"\n}'
    assert descriptive_stats(records, groups)["generation_prefix_match"] == pytest.approx(4 / 5)


def test_summary_keys_pass_core_serializer_guard(tmp_path):
    records = [
        _record("A", "N15", "primary", "sell", -1.0),
        _record("A", "P15", "primary", "buy", 1.0),
        _record("A", "zero", "primary", "sell", -2.0),
        _record("A", "N15", "order_swap", "sell", -1.2),
        _record("TICKER", "zero", "anon", "sell", -3.0),
    ]
    groups = group_rows([row for row in records if row["arm"] == "primary" and row["ticker"] == "A"])
    gate = evaluate_gates(records, groups, {"text_mismatch_count": 0, "max_abs_delta_margin": 0.0})
    summary = {
        "schema_version": "evidence-insensitivity-phase1-v1",
        "status": "formal",
        "n_prompts": len(records),
        "gates": gate["gates"],
        "fallback": gate["fallback"],
        "groups": groups,
        "descriptive_stats": descriptive_stats(records, groups),
        "raw_runtime_payloads": False,
    }

    def walk(obj, key=""):
        normalized = str(key).lower().replace("-", "_")
        assert not any(part in normalized.split("_") for part in _RAW), f"reserved key {key}"
        if isinstance(obj, dict):
            for child_key, child in obj.items():
                walk(child, str(child_key))
        elif isinstance(obj, list):
            for child in obj:
                walk(child, key)

    walk(summary)
    path = write_json(tmp_path / "summary.json", summary, overwrite=True)
    assert json.loads(path.read_text()) == summary


# ---------------------------------------------------------------------------
# Full lifecycle smoke (fake model, CPU)
# ---------------------------------------------------------------------------


def test_pipeline_lifecycle_smoke_fake_model(tmp_path):
    run_directory = pipeline.run_pilot(
        "fake-smoke-01",
        artifact_root=tmp_path,
        model=FakeScreenModel(),
        tokenizer=CharTokenizer(),
        device="cpu",
        use_chat_template=False,
    )
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert all(stage["status"] == "complete" for stage in manifest["stages"].values())

    prompts = [json.loads(line) for line in (run_directory / "prepare" / "prompts.jsonl").read_text().splitlines()]
    assert len(prompts) == 259
    records = [json.loads(line) for line in (run_directory / "forward" / "records.jsonl").read_text().splitlines()]
    assert len(records) == 259
    assert all(record["decision"] == "buy" for record in records)

    metadata = json.loads((run_directory / "forward" / "metadata.json").read_text(encoding="utf-8"))
    determinism = metadata["determinism_check"]
    assert determinism["n_prompts"] == 20
    assert determinism["text_mismatch_count"] == 0
    assert determinism["max_abs_delta_margin"] == 0.0

    summary = json.loads((run_directory / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "pilot"
    assert summary["n_prompts"] == 259
    # Fake generation always emits "buy": all pilot companies are fixed-buy,
    # the responsive group is empty, so the pre-registered fallback triggers.
    assert all(row["group"] == "fixed-buy" for row in summary["groups"])
    assert summary["gates"]["G-P4"]["pass"] is False
    assert summary["fallback"] is True
    assert summary["gates"]["G-P1"]["pass"] is True
    assert summary["gates"]["G-P3"]["pass"] is True
