import csv
import json
from pathlib import Path

import pytest

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.entity_cell import (
    FINANCIAL_PROMPT_COLUMNS,
    HELD_VARIANT_IDS,
    LOCALIZATION_VARIANT_IDS,
    prepare_artifacts,
    prepare_inputs,
    render_header_variants,
    validate_baseline_contract,
    validate_prepared_inputs,
)
from llm_bias.entity_cell.cli import build_parser


class _Tokenizer:
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


PROMPT = """Use the report to answer.
Stock Ticker: [{ticker}]
Stock Name: [{name}]
--- Evidence ---
{ticker} reported growth for {name}.
---
Respond with JSON."""


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


def test_header_variants_preserve_identity_and_tail_and_have_frozen_families():
    prompt = PROMPT.format(ticker="AB12", name="Alpha Beta Holdings")
    variants = render_header_variants(prompt)
    assert len(variants) == 12
    assert [row["variant_id"] for row in variants] == list(range(12))
    assert [row["variant_family"] for row in variants[:8]] == ["localization"] * 8
    assert [row["variant_family"] for row in variants[8:]] == ["held"] * 4
    header_end = prompt.index("\n--- Evidence ---")
    tail = prompt[header_end:]
    for row in variants:
        assert "Stock Ticker: [AB12]" in row["prompt"]
        assert "Stock Name: [Alpha Beta Holdings]" in row["prompt"]
        assert row["prompt"].endswith(tail)
    assert LOCALIZATION_VARIANT_IDS == tuple(range(8))
    assert HELD_VARIANT_IDS == tuple(range(8, 12))


def test_prepare_inputs_isolates_split_and_resolves_multitoken_name_and_groups(tmp_path):
    source, split, baseline = _write_inputs(tmp_path)
    prepared = prepare_inputs(
        tokenizer=_Tokenizer(),
        input_path=source,
        split_manifest=split,
        baseline_source=baseline,
        baseline_identity="adapted:test-v1",
        split="discovery",
        baseline_expected_count=2,
    )
    assert {row["ticker"] for row in prepared["financial_prompts"]} == {"ALFA", "BETA"}
    assert len(prepared["header_variants"]) == 24
    assert len(prepared["financial_prompts"]) == 6
    financial = prepared["financial_prompts"][0]
    span = financial["company_name_content_token_span"]
    assert span["token_end"] - span["token_start"] > 1
    assert span["final_content_token"] == span["token_end"] - 1
    groups = financial["source_groups"]
    positions = {position for group in groups.values() for start, end in group["ranges"] for position in range(start, end)}
    assert positions == set(range(financial["final_query_position"]))
    assert sum(group["token_count"] for group in groups.values()) == financial["final_query_position"]
    validate_prepared_inputs(prepared)

    split.write_text(
        json.dumps({"schema_version": 1, "input_sha256": sha256_file(source), "assignments": {"ALFA": "test", "BETA": "discovery"}}),
        encoding="utf-8",
    )
    discovery = prepare_inputs(
        tokenizer=_Tokenizer(), input_path=source, split_manifest=split,
        baseline_source=baseline, baseline_identity="adapted:test-v1", split="discovery", baseline_expected_count=2,
    )
    assert {row["ticker"] for row in discovery["financial_prompts"]} == {"BETA"}


def test_baseline_contract_requires_explicit_adaptation_and_hashes(tmp_path):
    _source, _split, baseline = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="adapted"):
        validate_baseline_contract(baseline, identity=None, expected_count=2)
    with pytest.raises(ValueError, match="exact paper"):
        validate_baseline_contract(baseline, identity="paper-appendix-a-v1", expected_count=2)
    rows = validate_baseline_contract(baseline, identity="adapted:test-v1", expected_count=2)
    assert all(len(row["prompt_sha256"]) == 64 for row in rows)
    baseline.write_text(baseline.read_text().replace("generic prompt 0", "changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_baseline_contract(baseline, identity="adapted:test-v1", expected_count=2)


def test_invalid_inputs_and_no_raw_tensors(tmp_path):
    source, split, baseline = _write_inputs(tmp_path)
    bad = json.loads(split.read_text())
    bad["assignments"]["ALFA"] = "unknown"
    split.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown split"):
        prepare_inputs(tokenizer=_Tokenizer(), input_path=source, split_manifest=split, baseline_source=baseline, baseline_identity="adapted:test-v1", baseline_expected_count=2)
    with pytest.raises(ValueError, match="raw runtime"):
        validate_prepared_inputs({"config": {"header_variant_count": 12, "baseline_expected_count": 0}, "header_variants": [], "financial_prompts": [], "generic_baseline": [], "activation": [1]})


def test_prepare_artifacts_writes_compact_provenance_and_parser(tmp_path):
    source, split, baseline = _write_inputs(tmp_path)
    root = prepare_artifacts(
        input_path=source, split_manifest=split, baseline_source=baseline,
        baseline_identity="adapted:test-v1", tokenizer=_Tokenizer(), model="fake-model",
        run_id="prep", artifact_root=tmp_path / "artifacts", baseline_expected_count=2,
    )
    metadata = json.loads((root / "prepare" / "metadata.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert metadata["input_sha256"] == sha256_file(source)
    assert metadata["config_sha256"]
    assert metadata["tokenizer_sha256"]
    assert metadata["prepared_artifact_sha256"]["header_variants"] == sha256_file(root / "prepare" / "header_variants.jsonl")
    assert metadata["raw_runtime_payloads"] is False
    assert manifest["status"] == "complete"
    assert manifest["stages"]["prepare"]["status"] == "complete"
    assert all("activation" not in json.dumps(ref).lower() for ref in manifest["artifacts"])
    args = build_parser().parse_args(["prepare", "--input", "i.csv", "--split-manifest", "s.json", "--baseline", "b.jsonl", "--baseline-identity", "adapted:v1", "--model", "fake", "--run-id", "r"])
    assert args.command == "prepare"
    assert args.baseline_identity == "adapted:v1"
