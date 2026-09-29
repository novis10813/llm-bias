"""rdo-cone-evidence-v1: stored directions are verified and re-evaluated on the evidence conditions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from llm_bias.core.steering import protocol as R

from steering_fakes import FakeJlens
from test_probe_rdo_cone import SLUG, _argv, _setup, rdo

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_rdo_evidence", ROOT / "scripts/probe_rdo_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def _source_run(tmp_path, monkeypatch):
    population, model_dir, tok, ranking = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    assert rdo.main(_argv(population, model_dir, ranking, "rdo-cone-v1-test-smoke-01", evaluation),
                    model=model, tokenizer=tok) == 0
    source = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs/rdo-cone-v1-test-smoke-01"
    return population, model_dir, tok, ranking, evaluation, model, source


def _evidence_argv(population, model_dir, ranking, evaluation, source, run_id="rdo-cone-v1-evidence-test-smoke-01"):
    return ["--model", str(model_dir), "--phase", "smoke", "--run-id", run_id, "--ranking-json", str(ranking),
            "--population-csv", str(population), "--directions-run", str(source), "--smoke-tickers", *evaluation[:2],
            "--allow-dirty", "--names", "dim", "rdo1", "rco_b2", "--doses", "2", "4"]


def test_evidence_run_covers_every_condition_direction_and_dose_and_resumes(tmp_path, monkeypatch):
    population, model_dir, tok, ranking, evaluation, model, source = _source_run(tmp_path, monkeypatch)
    argv = _evidence_argv(population, model_dir, ranking, evaluation, source)
    assert evidence.main(argv, model=model, tokenizer=tok) == 0
    out = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs/rdo-cone-v1-evidence-test-smoke-01/result.json"
    result = json.loads(out.read_text())
    assert result["complete"] is True
    assert set(result["baseline"]) == {"neg", "mixed2", "zero"}
    for condition in ("neg", "mixed2", "zero"):
        assert set(result["baseline"][condition]) == set(evaluation[:2])
        assert set(result["rows"][condition]) == {"dim", "rdo1", "rco_b2"}
        assert set(result["rows"][condition]["rdo1"]) == {"+2", "+4"}
        assert set(result["summary"][condition]["per_direction"]) == {"dim", "rdo1", "rco_b2"}
    norms = {n: t["inject_norm_median"] for n, t in result["dose_tables"].items()}
    assert norms["rdo1"] == pytest.approx(norms["dim"], rel=1e-4)
    assert result["metadata"]["source_directions_sha256"]
    before = out.read_text()
    assert evidence.main(argv, model=model, tokenizer=tok) == 0
    assert out.read_text() == before


def test_evidence_run_refuses_tampered_directions_and_unknown_names(tmp_path, monkeypatch):
    population, model_dir, tok, ranking, evaluation, model, source = _source_run(tmp_path, monkeypatch)
    stored = json.loads((source / "directions.json").read_text())
    stored["directions"]["rdo1"]["unit"][0] += 0.5
    (source / "directions.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="SHA-256"):
        evidence.main(_evidence_argv(population, model_dir, ranking, evaluation, source, "rdo-cone-v1-evidence-test-smoke-02"),
                      model=model, tokenizer=tok)
    stored["directions"]["rdo1"]["unit"][0] -= 0.5
    (source / "directions.json").write_text(json.dumps(stored))
    argv = _evidence_argv(population, model_dir, ranking, evaluation, source, "rdo-cone-v1-evidence-test-smoke-03")
    argv[argv.index("rco_b2")] = "rco_b9"
    with pytest.raises(ValueError, match="rco_b9"):
        evidence.main(argv, model=model, tokenizer=tok)
