"""evidence-scan-v1: evidence pool derivation, prompt rendering, summaries and an end-to-end smoke."""
from __future__ import annotations

import importlib.util
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pytest

from llm_bias.balanced_evidence_gap.template import build_prompt
from llm_bias.core.steering import prompts as P
from llm_bias.core.steering import protocol as R

from steering_fakes import FakeJlens
from test_probe_rdo_cone import SLUG, _setup

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = _load("probe_evidence_scan")
pool_builder = _load("build_evidence_pool")


def _company_row(ticker: str, seed: int) -> dict[str, str]:
    """A synthetic baseline row: four price-increase and four price-decrease items, 30 random 2+2 combinations."""
    up = [f"{ticker} signed a contract worth ${i}0 million, supporting a 5% increase in the stock price." for i in range(4)]
    down = [f"{ticker} faces a cost overrun of ${i}0 million, leading to a 5% decrease in the stock price." for i in range(4)]
    rng = random.Random(seed)
    row = {"ticker": ticker, "name": f"{ticker} Corp", "sector": "Utilities"}
    for k in range(30):
        items = rng.sample(up, 2) + rng.sample(down, 2)
        rng.shuffle(items)
        body = "\n".join(f"{n + 1}. {item}" for n, item in enumerate(items))
        row[f"prompt_with_context_attribute_{k}"] = f"header\n--- Evidence ---\n{body}\n---\nRespond"
    return row


def _write_pool(path: Path, tickers: list[str]) -> None:
    companies = {t: pool_builder.build_company(_company_row(t, i)) for i, t in enumerate(tickers)}
    path.write_text(json.dumps({"schema": scan.POOL_SCHEMA, "companies": companies}))


def test_pool_recovers_the_two_plus_two_polarity_and_rejects_ambiguous_sources():
    company = pool_builder.build_company(_company_row("AAA", 0))
    assert sum(i["polarity"] == "+" for i in company["items"]) == 4
    for combo in company["attribute_combos"]:
        assert sum(company["items"][k]["polarity"] == "+" for k in combo) == 2
    assert all("increase" in i["text"] for i in company["items"] if i["polarity"] == "+")
    row = _company_row("AAA", 0)
    # dropping every combination that separates the sides leaves the split ambiguous
    for k in range(30):
        row[f"prompt_with_context_attribute_{k}"] = row["prompt_with_context_attribute_0"]
    with pytest.raises(ValueError, match="not unique|8-item"):
        pool_builder.build_company(row)


def test_render_reproduces_the_frozen_balanced_prompt():
    items, polarity = scan.unit_evidence({}, scan.REF)
    assert polarity == "++--"
    for reverse in (False, True):
        assert scan.render_scan_prompt("ABC", "Abc Inc.", items, reverse) == build_prompt("ABC", "Abc Inc.", 0, reverse)
    assert scan.render_scan_prompt("ABC", "Abc Inc.", items, False) == P.render_decision_prompt("ABC", "Abc Inc.", "balanced")


def test_load_pool_rejects_a_combination_that_is_not_two_plus_two(tmp_path):
    path = tmp_path / "pool.json"
    _write_pool(path, ["AAA"])
    payload = json.loads(path.read_text())
    payload["companies"]["AAA"]["attribute_combos"][0] = [0, 1, 2, 3][:4]
    company = payload["companies"]["AAA"]
    plus = [k for k, i in enumerate(company["items"]) if i["polarity"] == "+"]
    company["attribute_combos"][0] = plus[:3] + [next(k for k in range(8) if k not in plus)]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="two positive"):
        scan.load_pool(path)


def test_helpers_shard_rate_and_regression_effects():
    assert scan.shard_items(list("abcdef"), (1, 3)) == ["b", "e"]
    with pytest.raises(ValueError):
        scan.parse_shard("3/3")
    assert scan.buy_rate([1.0, -1.0, 0.0, 2.0]) == 0.5
    # margin = company offset + 2 when the last item is positive: the fixed-effect regression recovers it
    rng = random.Random(0)
    tickers, pols, y = [], [], []
    for t, offset in (("A", -3.0), ("B", 1.0), ("C", 0.5)):
        for _ in range(40):
            p = rng.choice(scan.PATTERNS)
            tickers.append(t)
            pols.append(p)
            y.append(offset + (2.0 if p[-1] == "+" else 0.0))
    effects = scan.position_effects(np.array(y), tickers, pols)
    assert effects["position_4"] == pytest.approx(2.0, abs=1e-6)
    assert effects["position_1"] < 0     # 2+2 prompts: a positive first item makes a positive last item less likely
    assert scan.r_squared(np.array(y), np.hstack([scan._dummies(tickers), scan._dummies([p[-1] for p in pols])])) \
        == pytest.approx(1.0)


def _synthetic_rows(pool: dict, tickers: list[str], generate: bool) -> dict:
    rng = random.Random(1)
    rows = {}
    for t in tickers:
        offset = rng.uniform(-3, 3)
        for unit in scan.UNITS:
            _, pol = scan.unit_evidence(pool[t], unit)
            for reverse in scan.RENDERINGS:
                margin = offset + (1.5 if pol[-1] == "+" else -1.5) + rng.gauss(0, 0.3) + (0.5 if reverse else 0.0)
                row = {"margin": margin}
                if generate and reverse == scan.CANON:
                    row.update({"generated_text": "{}", "decision": "buy" if margin > 0 else "sell",
                                "format": "complete_object", "finish": "eos", "n_new_tokens": 5})
                rows[scan.row_key(unit, t, reverse)] = row
    return rows


def test_summaries_report_rates_patterns_and_generation_agreement(tmp_path):
    tickers = ["T001", "T002", "T003", "T004"]
    path = tmp_path / "pool.json"
    _write_pool(path, tickers)
    pool = scan.load_pool(path)[0]["companies"]
    for t in tickers:
        pool[t]["name_hint"] = t
    rows = _synthetic_rows(pool, tickers, generate=True)
    screen = scan.screen_summary(rows, pool, tickers)
    assert screen["n_cells"] == 4 * 30
    assert set(screen["by_pattern_canonical"]) <= set(scan.PATTERNS)
    assert screen["by_last_polarity_canonical"]["+"]["buy_rate"] > screen["by_last_polarity_canonical"]["-"]["buy_rate"]
    assert screen["position_effects_canonical"]["position_4"] > 2.0
    assert screen["variance_explained_canonical"]["company_plus_pattern"] > 0.95
    assert screen["options_order_shift_mean_margin"] == pytest.approx(-0.5, abs=0.2)
    fractions = screen["company_buy_fraction"]
    assert fractions["all_sell"] + fractions["mixed"] + fractions["all_buy"] == 4
    confirm = scan.confirm_summary(rows, pool, tickers, screen)
    assert confirm["attribute"]["parse_rate"] == 1.0 and confirm["attribute"]["margin_sign_agrees_with_generation"] == 1.0
    assert set(confirm["by_pattern"]) == set(scan.PATTERNS) and confirm["reference_frozen"]["n"] == 4


def test_smoke_scans_generates_and_resumes(tmp_path, monkeypatch):
    population, model_dir, tok, _ = _setup(tmp_path, monkeypatch)
    _, _, evaluation = R.split_population(population, R.SPLIT_SEED)
    tickers = evaluation[:2]
    pool_path = tmp_path / "pool.json"
    _write_pool(pool_path, tickers)
    model = FakeJlens(3, full_attention_only=True)
    argv = ["--model", str(model_dir), "--phase", "smoke", "--run-id", "evidence-scan-v1-test-smoke-01",
            "--pool", str(pool_path), "--population-csv", str(population), "--smoke-tickers", *tickers,
            "--smoke-units", "2", "--allow-dirty"]
    assert scan.main(argv, model=model, tokenizer=tok) == 0
    root = tmp_path / "artifacts" / SLUG / "concept-cone-steering/runs/evidence-scan-v1-test-smoke-01"
    result = json.loads((root / "result.json").read_text())
    assert result["complete"] is True and len(result["rows"]) == 2 * 3 * 2
    canonical = result["rows"][scan.row_key("a00", tickers[0], False)]
    assert {"margin", "decision", "generated_text"} <= set(canonical)
    assert set(result["rows"][scan.row_key("a00", tickers[0], True)]) == {"margin"}
    assert "pool_sha256" in result["metadata"] and "hidden" not in json.dumps(result)
    assert "done 12 rows" in (root / "scan.log").read_text()

    before = (root / "result.json").read_text()
    assert scan.main(argv, model=model, tokenizer=tok) == 0
    assert (root / "result.json").read_text() == before
    with pytest.raises(ValueError, match="run id"):
        scan.main([*argv[:5], "rdo-cone-v2-test-smoke-01", *argv[6:]], model=model, tokenizer=tok)
    with pytest.raises(ValueError, match="held-out"):
        scan.main([*argv[:11], "T000", *argv[13:]], model=model, tokenizer=tok)
