"""Confirmation-v1 dose supplement on a tiny model: reads a completed source run, never writes to it."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from llm_bias.core.steering import protocol as R

from steering_fakes import FakeJlens
from test_probe_steering_confirmation import SLUG, _argv, _setup, runner

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "probe_steering_confirmation_supplement", ROOT / "scripts/probe_steering_confirmation_supplement.py")
supp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supp)

SOURCE = "confirmation-v1-src-smoke-01"
SOURCE_ARMS = ["gates", "ranking", "alpha0", "cal", "dim", "random", "ops", "evidence", "anon"]


def _source(tmp_path, monkeypatch):
    population, model_dir, tok = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    assert runner.main(_argv(population, model_dir, SOURCE, SOURCE_ARMS, evaluation), model=model, tokenizer=tok) == 0
    monkeypatch.setitem(supp.SUPPLEMENT, SLUG, {"low": (-0.5, 0.5), "random": (3.0,)})
    runs = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs"
    return population, model_dir, tok, evaluation, model, runs


def _supp_argv(population, model_dir, run_id, evaluation, extra=()):
    return ["--model", str(model_dir), "--phase", "smoke", "--run-id", run_id, "--source-run-id", SOURCE,
            "--smoke-tickers", *evaluation[:2], "--population-csv", str(population), "--allow-dirty", *extra]


def _digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_supplement_adds_only_new_alphas_and_replicates_source_rows(tmp_path, monkeypatch):
    population, model_dir, tok, evaluation, model, runs = _source(tmp_path, monkeypatch)
    before = _digest(runs / SOURCE)
    run = "confirmation-v1-supp-smoke-01"
    assert supp.main(_supp_argv(population, model_dir, run, evaluation), model=model, tokenizer=tok) == 0
    assert _digest(runs / SOURCE) == before
    expected = {"dim": {"dim"}, "ops": {"neuron", "cone2", "cone4", "cone4_projection", "dim_orth_rand4"},
                "evidence": {"dim_pos", "dim_neg"}, "anon": {"dim_balanced"},
                "random": {f"random_s{s}" for s in runner.RANDOM_SEEDS}}
    for arm, names in expected.items():
        result = json.loads((runs / run / arm / "result.json").read_text())
        source = json.loads((runs / SOURCE / arm / "result.json").read_text())
        grid = [3.0] if arm == "random" else [-0.5, 0.5]
        assert result["complete"] and set(result["operators"]) == names, arm
        assert result["metadata"]["run_id"] == run and result["metadata"]["supplement_of"]["run_id"] == SOURCE
        assert "operator_drift" not in result
        for name in names:
            assert result["operators"][name]["grid"] == grid
            assert result["operators"][name]["dose"] == source["operators"][name]["dose"]
            assert [p["alpha"] for p in result["summary"][name]["per_alpha"]] == grid
            assert result["replication"][name]["text_identical"] is True
            assert result["replication"][name]["margin_abs_diff"] <= 1e-6
    # resuming a complete supplement generates nothing
    assert supp.main(_supp_argv(population, model_dir, run, evaluation, ("--max-rows", "0")),
                     model=model, tokenizer=tok) == 0


def test_supplement_refuses_alphas_already_in_the_source_grid(tmp_path, monkeypatch):
    population, model_dir, tok, evaluation, model, _ = _source(tmp_path, monkeypatch)
    monkeypatch.setitem(supp.SUPPLEMENT, SLUG, {"low": (runner.SMOKE_GRID[0],), "random": ()})
    with pytest.raises(ValueError, match="already in the source grid"):
        supp.main(_supp_argv(population, model_dir, "confirmation-v1-overlap-smoke-01", evaluation, ("--arms", "dim")),
                  model=model, tokenizer=tok)


def test_operator_drift_is_refused_unless_recorded(tmp_path, monkeypatch):
    population, model_dir, tok, evaluation, model, runs = _source(tmp_path, monkeypatch)
    path = runs / SOURCE / "random/result.json"
    stored = json.loads(path.read_text())
    stored["operators"]["random_s0"]["direction_sha256"] = "0" * 64
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="random/random_s0/extra"):
        supp.main(_supp_argv(population, model_dir, "confirmation-v1-drift-smoke-01", evaluation, ("--arms", "random")),
                  model=model, tokenizer=tok)
    run = "confirmation-v1-drift-smoke-02"
    assert supp.main(_supp_argv(population, model_dir, run, evaluation, ("--arms", "random", "--allow-operator-drift")),
                     model=model, tokenizer=tok) == 0
    result = json.loads((runs / run / "random/result.json").read_text())
    assert [d["label"] for d in result["operator_drift"]] == ["random/random_s0/extra"]


def test_supplement_refuses_reusing_the_source_run_id(tmp_path, monkeypatch):
    population, model_dir, tok, evaluation, model, _ = _source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="must differ"):
        supp.main(_supp_argv(population, model_dir, SOURCE, evaluation), model=model, tokenizer=tok)
