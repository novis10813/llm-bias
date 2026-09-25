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
