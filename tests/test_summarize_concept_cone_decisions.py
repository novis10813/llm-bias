"""Generated-decision transition statistics must use alpha-0 decisions, not margins."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "summarize_concept_cone_decisions",
    Path(__file__).resolve().parents[1] / "scripts/summarize_concept_cone_decisions.py",
)
assert spec and spec.loader
transitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transitions)


def run_file(tmp_path):
    def row(alpha, margin, decision):
        return {"alpha": alpha, "margin": margin, "decision": decision,
                "parse_ok": decision in {"buy", "sell"}}

    raw = {
        "complete": True,
        "metadata": {"mode": "evaluation", "model_slug": "fake", "layer": 16,
                     "target_tickers": ["AAA", "BBB", "CCC"], "alphas": [0.0, 2.0]},
        "targets": {
            # Margin sign deliberately disagrees with actual generated alpha-0 decision.
            "AAA": {"cone_centroid": [row(0.0, -2.0, "buy"), row(2.0, 3.0, "sell")]},
            "BBB": {"cone_centroid": [row(0.0, 5.0, "sell"), row(2.0, -1.0, "buy")]},
            "CCC": {"cone_centroid": [row(0.0, 2.0, "unparsed"), row(2.0, 2.0, "buy")]},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw


def test_strict_pairs_and_directional_denominators_ignore_margin(tmp_path):
    path, _ = run_file(tmp_path)
    out, md = transitions.summarize(path)
    baseline, alpha2 = out["summary"]
    assert (baseline["baseline_buy"], baseline["baseline_sell"], baseline["paired"]) == (1, 1, 2)
    assert alpha2["alpha_parsed"] == 3 and alpha2["paired"] == 2
    assert alpha2["transitions"] == {"buy_to_buy": 0, "buy_to_sell": 1,
                                      "sell_to_buy": 1, "sell_to_sell": 0}
    assert (alpha2["paired_baseline_buy"], alpha2["paired_baseline_sell"]) == (1, 1)
    assert (alpha2["flipped"], alpha2["flip_rate"], alpha2["buy_to_sell_rate"], alpha2["sell_to_buy_rate"]) == (2, 1.0, 1.0, 1.0)
    assert "2/2" in md and "Classes are generated buy/sell at alpha 0, not margin sign or C2 T." in md


def test_empty_baseline_class_is_undefined_not_zero(tmp_path):
    path, raw = run_file(tmp_path)
    raw["targets"]["BBB"]["cone_centroid"][0]["decision"] = "buy"
    path.write_text(json.dumps(raw), encoding="utf-8")
    out, md = transitions.summarize(path)
    alpha2 = out["summary"][1]
    assert alpha2["paired_baseline_sell"] == 0
    assert alpha2["sell_to_buy_rate"] is None
    assert "— (0 sell)" in md


def test_supplementary_requires_matching_input_hash_and_grid(tmp_path):
    path, raw = run_file(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    extra = {"schema": "concept-cone-decision-reparse-v1", "source_sha256": digest,
             "tickers": raw["metadata"]["target_tickers"], "alphas": [0.0, 2.0],
             "model_slug": "fake", "layer": 16,
             "targets": {t: [{"alpha": r["alpha"], "decision": r["decision"] if r["decision"] != "unparsed" else "sell"}
                             for r in raw["targets"][t]["cone_centroid"]] for t in ("AAA", "BBB", "CCC")}}
    extra["targets"]["CCC"][1]["decision"] = "sell"  # stable supplemental sell
    other = tmp_path / "decision_reparse.json"
    other.write_text(json.dumps(extra), encoding="utf-8")
    out, md = transitions.summarize(path, other)
    alpha2 = out["summary"][1]
    assert out["decision_source"] == "supplementary"
    assert alpha2["paired"] == 3 and alpha2["paired_baseline_sell"] == 2
    assert alpha2["sell_to_buy_rate"] == 0.5
    assert "Supplementary parser:" in md
    extra["source_sha256"] = "outdated"
    other.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match original"):
        transitions.summarize(path, other)
