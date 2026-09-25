"""Pure-file checks for the post-hoc complete-object cone decision parser."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "reparse_concept_cone_decisions",
    Path(__file__).resolve().parents[1] / "scripts/reparse_concept_cone_decisions.py",
)
assert spec and spec.loader
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


@pytest.mark.parametrize("wrapper,kind", [
    ('{\n"decision":"buy", "reason":"evidence"\n}', "bare_json"),
    ('```json\n{"decision":"buy", "reason":"evidence"}\n```', "fenced_json"),
    ('thought\n{"decision":"buy", "reason":"evidence"}', "thought_bare_json"),
    ('thought\n```json\n{"decision":"buy", "reason":"evidence"}\n```', "thought_fenced_json"),
])
def test_accepts_only_whole_valid_objects(wrapper, kind):
    assert parser.parse_complete_decision(wrapper) == ("buy", kind)


@pytest.mark.parametrize("text", [
    'prefix {"decision":"buy","reason":"evidence"}',
    '{"decision":"buy","reason":"evidence"} trailing text',
    '```json\n{"decision":"buy","reason":"evidence"}',
    '```json\n{"decision":"buy","reason":"evidence"}\n``` trailing',
    'thought extra\n{"decision":"buy","reason":"evidence"}',
    '{"decision":"buy","reason":"evidence","extra":1}',
    '{"decision":"BUY","reason":"evidence"}',
    '{"decision":"buy","reason":"  "}',
    '{"decision":"buy"}',
    '{"decision":"buy","reason":"incomplete',
])
def test_rejects_partial_and_invalid(text):
    assert parser.parse_complete_decision(text) == (None, "invalid")


def test_reparse_keeps_strict_counts_and_pairs_by_company(tmp_path):
    valid = '{"decision":"sell","reason":"evidence"}'
    fenced = '```json\n{"decision":"buy","reason":"evidence"}\n```'
    bad = '{"decision":"buy","reason":"incomplete'
    original = {
        "complete": True,
        "metadata": {"mode": "evaluation", "model_slug": "gemma4-12b-it", "layer": 27,
                     "alphas": [0.0, 2.0], "target_tickers": ["AAA", "BBB"]},
        "targets": {
            "AAA": {"cone_centroid": [
                {"alpha": 0.0, "margin": -1.0, "generated_text": valid, "decision": "sell", "parse_ok": True},
                {"alpha": 2.0, "margin": 1.0, "generated_text": fenced, "decision": "unparsed", "parse_ok": False}]},
            "BBB": {"cone_centroid": [
                {"alpha": 0.0, "margin": 0.0, "generated_text": bad, "decision": "unparsed", "parse_ok": False},
                {"alpha": 2.0, "margin": 0.5, "generated_text": fenced, "decision": "unparsed", "parse_ok": False}]},
        },
    }
    path = tmp_path / "result.json"
    raw = json.dumps(original)
    path.write_text(raw, encoding="utf-8")
    result, markdown = parser.reparse_run(path)
    assert path.read_text() == raw  # original untouched
    assert result["summary"][0]["strict_parsed"] == 1
    assert result["summary"][1]["strict_parsed"] == 0
    assert result["summary"][1]["reparsed"] == 2
    assert result["summary"][1]["valid_flip_pairs"] == 1
    assert result["summary"][1]["flipped"] == 1
    assert result["summary"][1]["flip_rate"] == 1.0
    assert result["format_counts"] == {"bare_json": 1, "fenced_json": 2, "invalid": 1}
    assert "| AAA | -1.000 (sell) | +1.000 (buy) |" in markdown
    original["targets"]["AAA"]["cone_centroid"][1]["alpha"] = 3.0
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="incomplete alpha grid"):
        parser.reparse_run(path)


def test_dim_reparse_keeps_layer_specific_baseline_pairs_and_original_bytes(tmp_path):
    import hashlib

    alphas = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    tickers = [f"E{i:03}" for i in range(101)]

    def row(alpha, decision, wrapper):
        obj = '{"decision":"' + decision + '","reason":"evidence"}'
        text = f"```json\n{obj}\n```" if wrapper else obj
        return {"alpha": alpha, "margin": alpha, "generated_text": text,
                "decision": "unparsed" if wrapper else decision, "parse_ok": not wrapper}

    targets = {t: {
        "L14": {"rows": [row(a, "sell" if a == 0 else "buy", True) for a in alphas]},
        "L15": {"rows": [row(a, "buy", False) for a in alphas]},
    } for t in tickers}
    original = {"complete": True, "metadata": {
        "schema": "dim-tokenwise-crossmodel-v1", "dim_arm": "tokenwise", "mode": "evaluation",
        "model_slug": "gemma4-12b-it", "layers": [14, 15], "alphas": alphas,
        "target_tickers": tickers}, "targets": targets}
    path = tmp_path / "result.json"
    raw = json.dumps(original).encode()
    path.write_bytes(raw)
    parsed, markdown = parser.reparse_run(path)
    assert path.read_bytes() == raw
    assert parsed["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert parsed["summary"]["14"][1]["strict_parsed"] == 0
    assert parsed["summary"]["14"][1]["sell_to_buy"] == 101
    assert parsed["summary"]["14"][1]["sell_to_buy_rate"] == 1
    assert parsed["summary"]["14"][1]["buy_to_sell_rate"] is None
    assert parsed["summary"]["15"][1]["buy_to_sell"] == 0
    assert parsed["summary"]["15"][1]["sell_to_buy_rate"] is None
    assert "L14 sell→buy | 0/101 | 101/101" in markdown
    assert "L15 sell→buy | —" in markdown
    original["targets"][tickers[0]]["L14"]["rows"][0]["generated_text"] = 'prefix {"decision":"sell"}'
    path.write_text(json.dumps(original))
    parsed, _ = parser.reparse_run(path)
    assert parsed["summary"]["14"][1]["sell_valid_pairs"] == 100
    original["targets"][tickers[0]]["L14"]["rows"][0]["parse_ok"] = True
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="strict decision"):
        parser.reparse_run(path)
