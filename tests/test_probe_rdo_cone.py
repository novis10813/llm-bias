"""rdo-cone-v1: cone math, differentiable training objective and an end-to-end smoke on a tiny model."""
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
spec = importlib.util.spec_from_file_location("probe_rdo_cone", ROOT / "scripts/probe_rdo_cone.py")
rdo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rdo)

SLUG = "toy-model"
SECTORS = ("Energy", "Utilities", "Unspecified", "Financials", "Industrials")


def test_gram_schmidt_gives_orthonormal_rows_and_keeps_the_first_direction():
    rows = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
    out = rdo.gram_schmidt(rows)
    assert torch.allclose(out @ out.T, torch.eye(4), atol=1e-5)
    assert torch.allclose(out[0], rows[0] / rows[0].norm(), atol=1e-6)
    with pytest.raises(ValueError):
        rdo.gram_schmidt(torch.stack([rows[0], rows[0]]))


def test_cone_samples_are_nonnegative_unit_coefficients_and_unit_directions():
    generator = torch.Generator().manual_seed(1)
    basis = rdo.gram_schmidt(torch.randn(3, 12, generator=torch.Generator().manual_seed(2)))
    for _ in range(20):
        s = rdo.sample_cone_coefficients(3, generator)
        assert bool((s >= 0).all()) and float(s.norm()) == pytest.approx(1.0, abs=1e-6)
        assert float(rdo.cone_unit(basis, s).norm()) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        rdo.cone_unit(basis, torch.tensor([1.0, -1.0, 0.0]))
    # a sample with all mass on one axis is that basis vector
    assert torch.allclose(rdo.cone_unit(basis, torch.tensor([0.0, 1.0, 0.0])), basis[1], atol=1e-6)


def test_addition_loss_prefers_buy_and_side_effect_kl_ignores_the_decision_tokens():
    logits = torch.zeros(10)
    logits[2], logits[3] = 3.0, -3.0     # buy id 2, sell id 3
    good, margin = rdo.addition_loss(logits, 2, 3)
    bad, _ = rdo.addition_loss(logits.flip(0), 2, 3)
    assert float(margin) == 6.0 and float(good) < float(bad)
    shifted = logits.clone()
    shifted[2] += 5.0                     # only a decision token changes
    assert float(rdo.side_effect_kl(logits, shifted, (2, 3))) == pytest.approx(0.0, abs=1e-6)
    shifted[5] += 2.0                     # an ordinary token changes
    assert float(rdo.side_effect_kl(logits, shifted, (2, 3))) > 1e-3


def _setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path, CharTokenizer, Path]:
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
    save_file({"w": torch.zeros(2, 2)}, str(model_dir / "model.safetensors"))
    monkeypatch.setitem(R.MODEL_REGISTRY, SLUG, R.ModelSpec(
        SLUG, 3, 1, (1,), k, "bf16", "0" * 64, "dense", "layers.{layer}.down.weight", None))
    monkeypatch.setattr(R, "MAX_NEW_TOKENS", 5)
    _, construction, evaluation = R.split_population(population, R.SPLIT_SEED)
    ranking = tmp_path / "ranking.json"
    rows = [{"ticker": t, "margin": -5.0 + 0.01 * i} for i, t in enumerate(sorted(companies))]
    ranking.write_text(json.dumps({"metadata": {"split_sha256": R.split_sha256(construction, evaluation)},
                                   "rows": rows, "complete": True}))
    return population, model_dir, tok, ranking


def _argv(population: Path, model_dir: Path, ranking: Path, run_id: str, evaluation: list[str]) -> list[str]:
    return ["--model", str(model_dir), "--phase", "smoke", "--run-id", run_id, "--ranking-json", str(ranking),
            "--population-csv", str(population), "--smoke-tickers", *evaluation[:2], "--allow-dirty",
            "--steps", "3", "--batch", "2", "--cone-dim", "2", "--cone-samples", "2", "--smoke-train", "4"]


def test_smoke_trains_directions_evaluates_every_operator_and_resumes(tmp_path, monkeypatch):
    population, model_dir, tok, ranking = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    argv = _argv(population, model_dir, ranking, "rdo-cone-v1-test-smoke-01", evaluation)
    assert rdo.main(argv, model=model, tokenizer=tok) == 0
    root = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs/rdo-cone-v1-test-smoke-01"

    directions = json.loads((root / "directions.json").read_text())["directions"]
    assert set(directions) == {"rdo1", "rco_b1", "rco_b2", "rco_centroid", "rco_s1", "rco_s2"}
    for entry in directions.values():
        assert torch.tensor(entry["unit"]).norm().item() == pytest.approx(1.0, abs=1e-4)
    basis = torch.tensor([directions["rco_b1"]["unit"], directions["rco_b2"]["unit"]])
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-4)

    training = json.loads((root / "training.json").read_text())
    assert len(training["curves"]["rdo1"]) == 3 and len(training["curves"]["rco"]) == 3
    assert "basis_margin_mean" in training["curves"]["rco"][0]

    result = json.loads((root / "result.json").read_text())
    assert result["complete"] is True and set(result["baseline"]) == set(evaluation[:2])
    assert set(result["rows"]) == {"dim", "rand1", *directions}
    for name in ("dim", "rand1", "rdo1"):
        assert set(result["rows"][name]) == {"+1", "+2", "+4", "+8"}
    for name in ("rco_b1", "rco_centroid", "rco_s1"):
        assert set(result["rows"][name]) == {"+1", "+2", "+4"}
    # every direction is dosed at the DIM per-token norm
    norms = {name: table["inject_norm_median"] for name, table in result["dose_tables"].items()}
    assert all(v == pytest.approx(norms["dim"], rel=1e-4) for v in norms.values())
    assert set(result["summary"]["per_direction"]) == set(result["rows"])
    text = json.dumps(result)
    assert "hidden" not in text and "states" not in text

    # a second invocation resumes: training is reused and no row is regenerated
    before = (root / "directions.json").read_text()
    assert rdo.main(argv, model=model, tokenizer=tok) == 0
    assert (root / "directions.json").read_text() == before


def test_run_refuses_a_ranking_from_another_split_and_bad_run_ids(tmp_path, monkeypatch):
    population, model_dir, tok, ranking = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    model = FakeJlens(3, full_attention_only=True)
    stored = json.loads(ranking.read_text())
    stored["metadata"]["split_sha256"] = "0" * 64
    ranking.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="ranking"):
        rdo.main(_argv(population, model_dir, ranking, "rdo-cone-v1-test-smoke-01", evaluation), model=model, tokenizer=tok)
    with pytest.raises(ValueError, match="run id"):
        rdo.main(_argv(population, model_dir, ranking, "wrong-id", evaluation), model=model, tokenizer=tok)
