"""End-to-end smoke of the confirmation-v1 runner on a tiny model, plus interrupted-resume equivalence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from llm_bias.core.steering import prompts as P
from llm_bias.core.steering import protocol as R

from steering_fakes import CharTokenizer, FakeJlens

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_steering_confirmation", ROOT / "scripts/probe_steering_confirmation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

SLUG = "toy-model"
SECTORS = ("Energy", "Utilities", "Unspecified", "Financials", "Industrials")


def _setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path, CharTokenizer]:
    from safetensors.torch import save_file

    monkeypatch.chdir(tmp_path)
    population = tmp_path / "pop.csv"
    lines = ["index_name,year,ticker,company_name,gics_sector"]
    lines += [f"S&P 500,2024,T{i:03},Name {i:03} Inc.,{SECTORS[i % len(SECTORS)]}" for i in range(503)]
    population.write_text("\n".join(lines) + "\n")
    tok = CharTokenizer()
    companies = R.load_population(population)
    k, _ = P.common_instruction_suffix(tok, [P.render_decision_prompt(t, c["name"], cond)
                                             for cond in P.CONDITIONS for t, c in list(companies.items())[:5]])
    model_dir = tmp_path / SLUG
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")
    save_file({"layers.1.down.weight": torch.randn(32, 64, generator=torch.Generator().manual_seed(1))},
              str(model_dir / "model.safetensors"))
    c2 = tmp_path / "artifacts" / SLUG / "balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json"
    c2.parent.mkdir(parents=True)
    c2.write_text(json.dumps({"curves": {"instruction": {
        str(i): {"mean_normalized_transfer": [0.1, 0.5, 0.2][i], "n_directions": 854} for i in range(3)}}}))
    monkeypatch.setitem(R.MODEL_REGISTRY, SLUG, R.ModelSpec(
        SLUG, 3, 1, (1,), k, "bf16", R.sha256_bytes(c2.read_bytes()), "dense", "layers.{layer}.down.weight", None))
    arms = tuple(a for a in runner.ARM_ORDER if a != "dim_layers")
    monkeypatch.setitem(runner.TIERS, SLUG, {"tier1": arms, "tier2": ()})
    monkeypatch.setattr(R, "MAX_NEW_TOKENS", 5)
    return population, model_dir, tok


def _argv(population: Path, model_dir: Path, run_id: str, arms: list[str], evaluation: list[str], extra=()):
    return ["--model", str(model_dir), "--phase", "smoke", "--run-id", run_id, "--arms", *arms,
            "--smoke-tickers", *evaluation[:2], "--population-csv", str(population), "--allow-dirty", *extra]


def test_smoke_runs_every_arm_and_writes_compact_results(tmp_path, monkeypatch):
    population, model_dir, tok = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    code = runner.main(_argv(population, model_dir, "confirmation-v1-test-smoke-01", ["all"], evaluation),
                       model=model, tokenizer=tok)
    assert code == 0
    root = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs/confirmation-v1-test-smoke-01"
    gates = json.loads((root / "gates/result.json").read_text())
    assert gates["checks"]["failed"] == [] and gates["checks"]["zero_hook_identical"]
    for arm in ("dim", "random", "jitter", "ops", "shuffle", "evidence", "anon", "loso", "loso_construction",
                "split_seed", "c2v3", "c2v3_gen"):
        result = json.loads((root / arm / "result.json").read_text())
        assert result["complete"] is True, arm
        assert "summary" in result
    dim = json.loads((root / "dim/result.json").read_text())
    assert dim["summary"]["structural_zero_ok"] is True
    assert dim["operators"]["dim"]["grid"] == list(runner.SMOKE_GRID)
    ops = json.loads((root / "ops/result.json").read_text())
    assert set(ops["operators"]) == {"neuron", "cone2", "cone4", "cone4_projection", "dim_orth_rand4"}
    assert ops["operators"]["cone4"]["dose"]["cos_to_dim_median"] == pytest.approx(0.5, abs=1e-4)
    assert ops["operators"]["cone4_projection"]["dose"]["projection_on_dim_median"] == pytest.approx(
        dim["operators"]["dim"]["dose"]["inject_norm_median"], rel=1e-4)
    random_arm = json.loads((root / "random/result.json").read_text())
    for name, spec_ in random_arm["operators"].items():
        assert spec_["dose"]["inject_norm_median"] == pytest.approx(dim["operators"]["dim"]["dose"]["inject_norm_median"], rel=1e-4)
    c2 = json.loads((root / "c2v3/result.json").read_text())
    assert c2["summary"]["self_patch_max_abs"] <= 1e-6 and len(c2["rows"]["patch"]) == 4
    gen = json.loads((root / "c2v3_gen/result.json").read_text())
    assert gen["summary"]["self_patch_identical"] is True
    cal = json.loads((root / "cal/calibration.json").read_text())
    assert cal["complete"] and len(cal["companies"]) == runner.SMOKE_CAL_COMPANIES
    anon = json.loads((root / "anon/result.json").read_text())
    assert len(anon["rows"]["dim_zero"]) == 10
    text = json.dumps(dim)
    assert "hidden" not in text and "states" not in text
    # a second invocation resumes everything without generating (budget 0 would raise otherwise)
    assert runner.main(_argv(population, model_dir, "confirmation-v1-test-smoke-01", ["dim", "random"], evaluation,
                             ("--max-rows", "0")), model=model, tokenizer=tok) == 0


def test_interrupted_resume_equals_uninterrupted_run(tmp_path, monkeypatch):
    population, model_dir, tok = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    arms = ["gates", "ranking", "alpha0", "cal", "dim", "random", "ops", "c2v3"]
    assert runner.main(_argv(population, model_dir, "confirmation-v1-a-smoke-01", arms, evaluation),
                       model=model, tokenizer=tok) == 0
    codes = []
    for _ in range(200):
        code = runner.main(_argv(population, model_dir, "confirmation-v1-b-smoke-01", arms, evaluation,
                                 ("--max-rows", "7")), model=model, tokenizer=tok)
        codes.append(code)
        if code == 0:
            break
    assert codes[0] == 3 and codes[-1] == 0 and len(codes) > 3
    base = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs"
    for arm in ("alpha0", "dim", "random", "ops", "c2v3"):
        a = json.loads((base / f"confirmation-v1-a-smoke-01/{arm}/result.json").read_text())
        b = json.loads((base / f"confirmation-v1-b-smoke-01/{arm}/result.json").read_text())
        for payload in (a, b):
            payload["metadata"].pop("run_id")
        assert a == b, arm
    a = json.loads((base / "confirmation-v1-a-smoke-01/cal/calibration.json").read_text())
    b = json.loads((base / "confirmation-v1-b-smoke-01/cal/calibration.json").read_text())
    a["metadata"].pop("run_id"), b["metadata"].pop("run_id")
    assert a == b


def test_resume_refuses_changed_metadata_and_corrupted_rows(tmp_path, monkeypatch):
    population, model_dir, tok = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    run = "confirmation-v1-c-smoke-01"
    arms = ["gates", "ranking", "alpha0", "cal", "dim"]
    assert runner.main(_argv(population, model_dir, run, arms, evaluation), model=model, tokenizer=tok) == 0
    path = tmp_path / "artifacts" / SLUG / f"concept-cone-steering/runs/{run}/dim/result.json"
    stored = json.loads(path.read_text())
    key = next(iter(stored["rows"]["dim"]))
    stored["rows"]["dim"][key][0]["decision"] = "buy" if stored["rows"]["dim"][key][0]["decision"] != "buy" else "sell"
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="re-derive"):
        runner.main(_argv(population, model_dir, run, ["dim"], evaluation), model=model, tokenizer=tok)
    stored["metadata"]["code_sha256"]["x"] = "changed"
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="metadata differs"):
        runner.main(_argv(population, model_dir, run, ["dim"], evaluation), model=model, tokenizer=tok)


def test_full_phase_refuses_dirty_code_and_bad_run_ids(tmp_path, monkeypatch):
    population, model_dir, tok = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "git_provenance", lambda cwd=None: {"git_commit": "x", "code_dirty": True,
                                                               "code_dirty_paths": ["scripts/a.py"],
                                                               "other_status_lines": 0})
    model = FakeJlens(3, full_attention_only=True)
    argv = ["--model", str(model_dir), "--phase", "full", "--run-id", "confirmation-v1-x-full-01",
            "--population-csv", str(population)]
    with pytest.raises(ValueError, match="committed code"):
        runner.main(argv, model=model, tokenizer=tok)
    argv[5] = "confirmation-v1-x-smoke-01"
    with pytest.raises(ValueError, match="run id"):
        runner.main(argv, model=model, tokenizer=tok)
    assert runner.resolve_arms(SLUG, ["tier1"])[0] == "gates"
    with pytest.raises(ValueError, match="unknown arm"):
        runner.resolve_arms(SLUG, ["depth"])
