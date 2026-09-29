"""rdo-cone-v2: train/validation split, lr schedule, convergence gate, run log and an end-to-end smoke."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from llm_bias.core.steering import protocol as R

from steering_fakes import FakeJlens
from test_probe_rdo_cone import SLUG, _setup

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_rdo_cone_v2", ROOT / "scripts/probe_rdo_cone_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def test_validation_split_is_disjoint_deterministic_and_inside_construction():
    construction = [f"T{i:03}" for i in range(402)]
    train, val = v2.train_val_split(construction, 80, v2.VAL_SEED)
    assert len(train) == 322 and len(val) == 80 and not set(train) & set(val)
    assert set(train) | set(val) == set(construction)
    assert val == v2.train_val_split(list(reversed(construction)), 80, v2.VAL_SEED)[1]


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    factors = [v2.lr_factor(s, 100, 10, 0.1) for s in range(100)]
    assert factors[0] == pytest.approx(0.1) and factors[9] == pytest.approx(1.0)
    assert factors[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(factors[10:], factors[11:]))
    assert v2.lr_factor(100, 100, 10, 0.1) == pytest.approx(0.1)


def test_target_loss_keeps_pushing_past_zero_and_reduces_to_v1_at_zero_target():
    logits = torch.zeros(6)
    logits[2], logits[3] = 0.5, 0.0
    at_zero, margin = v2.target_loss(logits, 2, 3, 0.0)
    v1, _ = v2.base.addition_loss(logits, 2, 3)
    assert float(margin) == 0.5 and float(at_zero) == pytest.approx(float(v1))
    assert float(v2.target_loss(logits, 2, 3, 2.0)[0]) > float(at_zero)


def test_convergence_gate_needs_every_unit():
    val = {"rdo1": {"frac_positive": 0.9}, "rco_b1": {"frac_positive": 0.8}, "rco_b2": {"frac_positive": 0.79}}
    assert v2.convergence(val, ["rdo1", "rco_b1"])["converged"] is True
    assert v2.convergence(val, ["rdo1", "rco_b1", "rco_b2"])["converged"] is False


def _argv(population, model_dir, ranking, run_id, evaluation, phase="smoke"):
    return ["--model", str(model_dir), "--phase", phase, "--run-id", run_id, "--ranking-json", str(ranking),
            "--population-csv", str(population), "--smoke-tickers", *evaluation[:2], "--allow-dirty",
            "--steps", "4", "--batch", "2", "--warmup", "1", "--val-every", "2", "--cone-dim", "2",
            "--cone-samples", "2", "--smoke-train", "4", "--smoke-val", "3", "--target-margin", "1",
            "--with-controls"]


def test_smoke_validates_logs_evaluates_and_resumes(tmp_path, monkeypatch):
    population, model_dir, tok, ranking = _setup(tmp_path, monkeypatch)
    _, construction, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    argv = _argv(population, model_dir, ranking, "rdo-cone-v2-test-smoke-01", evaluation)
    assert v2.main(argv, model=model, tokenizer=tok) == 0
    root = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs/rdo-cone-v2-test-smoke-01"

    training = json.loads((root / "training.json").read_text())
    assert len(training["curves"]["rdo1"]) == 4 and len(training["curves"]["rco"]) == 4
    assert [v["step"] for v in training["validation"]["rco"]] == [0, 2, 4]
    assert set(training["validation"]["rco"][-1]["units"]) == {"rco_b1", "rco_b2", "rco_centroid", "rco_v1", "rco_v2"}
    assert set(training["convergence"]["frac_positive"]) == {"rdo1", "rco_b1", "rco_b2"}
    assert isinstance(training["convergence"]["converged"], bool)
    val = set(training["metadata"]["val_tickers"])
    assert len(val) == 3 and val <= set(construction) and not val & set(evaluation)

    log = (root / "train.log").read_text()
    assert "[rdo1] val step 0" in log and "[rco] step 3" in log and "convergence:" in log and "alpha0" in log

    result = json.loads((root / "result.json").read_text())
    assert result["complete"] is True and result["convergence"] == training["convergence"]
    assert set(result["rows"]["rdo1"]) == {"+1", "+2", "+4", "+8", "+16", "+32"}
    assert set(result["rows"]["rco_b1"]) == {"+1", "+2", "+4"}
    assert set(result["rows"]["rco_s1"]) == {"+1", "+2"}
    assert set(result["rows"]["rand1"]) == {"+1", "+2", "+4", "+8"}
    norms = {name: table["inject_norm_median"] for name, table in result["dose_tables"].items()}
    assert all(v == pytest.approx(norms["dim"], rel=1e-4) for v in norms.values())
    assert "hidden" not in json.dumps(result)

    before = (root / "directions.json").read_text(), (root / "result.json").read_text()
    assert v2.main(argv, model=model, tokenizer=tok) == 0
    assert ((root / "directions.json").read_text(), (root / "result.json").read_text()) == before


def test_tune_phase_trains_without_evaluating_and_rejects_bad_run_ids(tmp_path, monkeypatch):
    population, model_dir, tok, ranking = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    monkeypatch.setattr(R, "git_provenance", lambda cwd=None: {"git_commit": "x", "code_dirty": False,
                                                                  "code_dirty_paths": [], "other_status_lines": []})
    argv = _argv(population, model_dir, ranking, "rdo-cone-v2-test-tune-01", evaluation, phase="tune")
    assert v2.main(argv, model=model, tokenizer=tok) == 0
    root = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs/rdo-cone-v2-test-tune-01"
    assert (root / "training.json").exists() and not (root / "result.json").exists()
    # tune uses the full 322-company training split
    assert json.loads((root / "training.json").read_text())["metadata"]["train_tickers_n"] == 322
    with pytest.raises(ValueError, match="run id"):
        v2.main(_argv(population, model_dir, ranking, "rdo-cone-v1-test-smoke-01", evaluation),
                model=model, tokenizer=tok)
