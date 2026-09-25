"""Contract tests for llm_bias.core.steering and decision_readout (fake tokenizer / tiny model, no checkpoints)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from llm_bias.balanced_evidence_gap.template import build_prompt_v2
from llm_bias.core import decision_readout as readout
from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.steering import directions as D
from llm_bias.core.steering import evaluate as E
from llm_bias.core.steering import prompts as P
from llm_bias.core.steering import protocol as R
from llm_bias.core.steering import summary as S
from llm_bias.entity_to_dial.heldout_transfer import _render_frozen_prompt

from steering_fakes import CharTokenizer, FakeJlens, TOKEN_ID

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_concept_cone", ROOT / "scripts/probe_concept_cone.py")
cone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cone)

COMPANIES = {"ABNB": "Airbnb", "AEP": "American Electric Power", "MO": "Altria Group", "X": "X Corp."}


# ── prompts ────────────────────────────────────────────────────────────────────────────


def test_conditions_are_byte_identical_to_frozen_renderers():
    for ticker, name in COMPANIES.items():
        assert P.render_decision_prompt(ticker, name, "balanced") == _render_frozen_prompt(
            ticker, name, order=0, reverse=False)
        for condition in ("pos", "neg"):
            assert P.render_decision_prompt(ticker, name, condition) == build_prompt_v2(ticker, name, condition, False)
    zero = P.render_decision_prompt("ABNB", "Airbnb", "zero")
    assert "— Evidence —\n\n\n\n—\n\n" in zero
    mixed = P.render_decision_prompt("ABNB", "Airbnb", "mixed2")
    assert mixed.index("Q3 revenue") < mixed.index("Gross margin") and "Free cash" not in mixed
    with pytest.raises(ValueError):
        P.render_decision_prompt("ABNB", "Airbnb", "hawkish")


def test_prompt_family_sha_reproduces_v2_serialization():
    companies = {t: {"name": n} for t, n in COMPANIES.items()}
    payload = json.dumps([(t, _render_frozen_prompt(t, companies[t]["name"], order=0, reverse=False))
                          for t in sorted(companies)], ensure_ascii=False, separators=(",", ":")).encode()
    assert P.prompt_family_sha256(companies, "balanced") == R.sha256_bytes(payload)


def test_spans_are_consistent_and_zero_evidence_span_is_explicitly_empty():
    tok = CharTokenizer()
    fps = {c: P.format_decision_prompt(tok, P.render_decision_prompt("ABNB", "Airbnb", c), suffix_tokens=40,
                                       key=c) for c in P.CONDITIONS}
    zero = fps["zero"].spans
    assert zero["evidence"][0] == zero["evidence"][1] == zero["instruction"][0]
    for condition, fp in fps.items():
        s = fp.spans
        assert s["entity"][1] <= s["evidence"][0] <= s["instruction"][0] < s["instruction"][1] <= s["final"][0]
        assert s["steer_suffix"] == (s["instruction"][1] - 40, s["instruction"][1])
        assert fp.score_ids[:len(fp.ids)] == fp.ids and s["answer_prefix"] == (len(fp.ids), len(fp.score_ids))
        assert (fp.buy_id, fp.sell_id) == (TOKEN_ID["buy"], TOKEN_ID["sell"])
        suffix = fp.ids[s["steer_suffix"][0]:s["steer_suffix"][1]]
        assert suffix == fps["balanced"].ids[fps["balanced"].spans["steer_suffix"][0]:fps["balanced"].spans["steer_suffix"][1]]
    with pytest.raises(ValueError, match="shorter than K"):
        P.format_decision_prompt(tok, P.render_decision_prompt("ABNB", "Airbnb", "zero"), suffix_tokens=10_000)


def test_common_suffix_is_equal_across_conditions_and_identities():
    tok = CharTokenizer()
    prompts = [P.render_decision_prompt(t, n, c) for c in P.CONDITIONS for t, n in
               list(COMPANIES.items()) + list(P.ANON_IDENTITIES)]
    k, ids = P.common_instruction_suffix(tok, prompts)
    single, _ = P.common_instruction_suffix(tok, prompts[:3])
    assert k == single and len(ids) == k


def test_anon_collision_check_flags_real_names_and_tickers():
    population = {"QZX": {"name": "Something"}, "MO": {"name": "Company Z"}}
    assert set(P.anon_collisions(population)) == {"ticker:QZX", "name:Company Z"}
    assert P.anon_collisions({"MO": {"name": "Altria"}}) == []
    assert len(P.ANON_IDENTITIES) == 10 and P.ANON_IDENTITIES[0] == ("TICKER", "Company X")


# ── protocol ───────────────────────────────────────────────────────────────────────────


def _population(tmp_path: Path) -> Path:
    path = tmp_path / "pop.csv"
    sectors = ["Energy", "Utilities", "Unspecified", "Financials"]
    lines = ["index_name,year,ticker,company_name,gics_sector"]
    lines += [f"S&P 500,2024,T{i:03},Name {i},{sectors[i % 4]}" for i in range(503)]
    lines += ["S&P 500,2023,OLD,Old,Energy"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_split_population_matches_v2_implementation(tmp_path):
    path = _population(tmp_path)
    ours = R.split_population(path, 20260923)
    theirs = cone.split_population(path, 20260923)
    assert ours == theirs and len(ours[1]) == 402 and len(ours[2]) == 101
    assert R.split_sha256(ours[1], ours[2]) == R.sha256_bytes(
        json.dumps({"construction": ours[1], "evaluation": ours[2]}, sort_keys=True).encode())
    assert R.split_population(path, 20260924)[2] != ours[2]


def test_template_date_is_pinned_only_for_templates_that_read_it(monkeypatch):
    import datetime as dt

    import transformers.utils.chat_template_utils as ctu

    template = "{{ reasoning_effort | default('medium') }} {{ strftime_now('%Y-%m-%d') }}"

    class Tomorrow(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 9, 26, 0, 30)

    monkeypatch.setattr(ctu, "datetime", Tomorrow)
    harmony = CharTokenizer(template)
    render = lambda kw: ctu.render_jinja_template([[{"role": "user", "content": "x"}]], chat_template=template, **kw)[0][0]
    assert render(R.low_reasoning_kwargs(harmony)) == "low 2026-09-26"
    assert render(R.template_render_kwargs(harmony)) == "low 2026-09-25"
    assert R.template_provenance(harmony)["chat_template_date"] == "2026-09-25"
    plain = CharTokenizer("{{ messages }}")
    assert R.template_render_kwargs(plain) == {} and "chat_template_date" not in R.template_provenance(plain)
    assert R.template_render_kwargs(harmony).keys() == {"reasoning_effort", "strftime_now"}
    assert (R.template_render_kwargs(harmony)["strftime_now"]("%Y-%m-%d")
            == cone.template_render_kwargs(harmony)["strftime_now"]("%Y-%m-%d"))


def test_c2_source_is_bound_to_registry_sha(tmp_path, monkeypatch):
    slug = "glm4-9b-0414"
    path = tmp_path / slug / "balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json"
    path.parent.mkdir(parents=True)
    curve = {str(i): {"mean_normalized_transfer": 0.1 + (i == 19), "n_directions": 854} for i in range(40)}
    path.write_text(json.dumps({"curves": {"instruction": curve}}))
    with pytest.raises(ValueError, match="SHA"):
        R.c2_427_source(slug, tmp_path)
    frozen = R.MODEL_REGISTRY[slug]
    monkeypatch.setitem(R.MODEL_REGISTRY, slug, R.ModelSpec(**{**frozen.__dict__,
                                                                 "c2_sha256": R.sha256_bytes(path.read_bytes())}))
    source = R.c2_427_source(slug, tmp_path)
    assert source["instruction_peak"] == 19 and len(source["instruction_T"]) == 40


def test_registry_matches_v2_frozen_constants():
    for slug, cfg in cone.DIM_CROSSMODEL_CONFIG.items():
        spec = R.MODEL_REGISTRY[slug]
        assert (spec.n_layers, spec.peak, spec.suffix_tokens, spec.dtype, spec.c2_sha256) == (
            cfg["n_layers"], cfg["peak"], cfg["suffix_tokens"], cfg["dtype"], cfg["c2_sha256"])
    assert R.MODEL_REGISTRY["qwen3.5-4b"].peak == cone.PAPER_C2_LAYERS["qwen3.5-4b"]


def test_git_provenance_scopes_the_dirty_flag(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/a.py").write_text("x")
    (tmp_path / "notes.md").write_text("y")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    assert R.git_provenance(tmp_path)["code_dirty"] is False
    (tmp_path / "notes.md").write_text("changed")
    clean_code = R.git_provenance(tmp_path)
    assert clean_code["code_dirty"] is False and clean_code["other_status_lines"] == 1
    (tmp_path / "scripts/a.py").write_text("changed")
    assert R.git_provenance(tmp_path)["code_dirty_paths"] == ["scripts/a.py"]


# ── decision readout ───────────────────────────────────────────────────────────────────


def test_path_class_and_unparsed_kind():
    assert readout.path_class('{"decision": "buy", "reason": "x"}') == "direct_json"
    assert readout.path_class('{\n  "decision": "buy"}') == "brace_newline"
    assert readout.path_class("thought\n{}") == "thought"
    assert readout.path_class("```json\n{}\n```") == "fenced"
    assert readout.path_class("analysis blah assistantfinal{}") == "harmony_final"
    assert readout.path_class("analysisWe need to") == "harmony_analysis"
    assert readout.path_class("I think") == "other"
    assert readout.unparsed_kind("buy", "max_new_tokens") is None
    assert readout.unparsed_kind("unparsed", "max_new_tokens") == "truncated"
    assert readout.unparsed_kind("unparsed", "eos") == "collapsed"
    assert readout.generation_finish([5, 1], {1}, 10) == "eos"
    assert readout.generation_finish([5] * 10, {1}, 10) == "max_new_tokens"


def test_realized_margin_reads_the_decision_value_step():
    tok = CharTokenizer()
    new_ids = tok('{"decision": "sell", "reason": "x"}', add_special_tokens=False).input_ids + [1]
    logits = [torch.zeros(len(TOKEN_ID)) for _ in new_ids]
    step = new_ids.index(TOKEN_ID["sell"])
    logits[step][TOKEN_ID["buy"]] = 1.0
    logits[step][TOKEN_ID["sell"]] = 3.5
    out = readout.realized_margin(tok, new_ids, logits, "sell")
    assert out["realized_status"] == "ok" and out["realized_step"] == step
    assert out["realized_token_ids"] == [TOKEN_ID["buy"], TOKEN_ID["sell"]]
    assert out["realized_margin"] == pytest.approx(-2.5)
    assert readout.realized_margin(tok, new_ids, logits, "unparsed")["realized_status"] == "unparsed"
    assert readout.realized_margin(tok, new_ids, logits, "buy")["realized_status"] == "decision_value_not_found"
    with pytest.raises(ValueError):
        readout.realized_margin(tok, new_ids, logits[:-1], "sell")


# ── evaluate ───────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fake():
    tok = CharTokenizer()
    model = FakeJlens(num_layers=3)
    fp = P.format_decision_prompt(tok, P.render_decision_prompt("ABNB", "Airbnb", "balanced"), suffix_tokens=40,
                                  key="ABNB")
    return tok, model, fp


def test_chunked_fp32_margin_equals_v2_log_prob_tail(fake):
    tok, model, fp = fake
    ids = torch.tensor([list(fp.score_ids)])
    res = record_residuals(model, ids, [2])[2][:, -1, :]
    log_probs = fp32_next_token_log_probs(model, res)
    expected = float(log_probs[0, fp.buy_id] - log_probs[0, fp.sell_id])
    assert E.fp32_margin(model, res, fp.buy_id, fp.sell_id, chunk=7) == pytest.approx(expected, abs=1e-5)


def test_rows_are_complete_and_rederivable(fake):
    tok, model, fp = fake
    ev = E.Evaluator(model, tok, max_new_tokens=8, printer=None)
    base = ev.row(fp, 0.0, baseline=True)
    E.validate_row(base, 0.0, "base", baseline=True)
    assert base["n_new_tokens"] == len(base["new_ids"]) <= 8
    tampered = dict(base, decision="buy")
    with pytest.raises(ValueError, match="re-derive"):
        E.validate_row(tampered, 0.0, "t", baseline=True)
    steered = ev.steered_row(fp, 0, torch.randn(40, 32), 2.0)   # L1 is full attention: the edit propagates
    E.validate_row(steered, 2.0, "steered")
    assert steered["margin"] != base["margin"]


def test_structural_zero_hooks_reproduce_alpha_zero(fake):
    tok, model, fp = fake
    ev = E.Evaluator(model, tok, max_new_tokens=6, printer=None)
    base = ev.row(fp, 0.0)
    zero_shift = ev.steered_row(fp, 0, torch.zeros(40, 32), 4.0)
    last_layer = ev.steered_row(fp, 2, torch.randn(40, 32) * 50, 4.0)   # after the final block, suffix only
    for row in (zero_shift, last_layer):
        assert row["generated_text"] == base["generated_text"] and abs(row["margin"] - base["margin"]) <= 1e-6


def test_teacher_forced_margin_matches_generation_logits(fake):
    tok, model, fp = fake
    ev = E.Evaluator(model, tok, max_new_tokens=6, printer=None)
    new_ids, logits = ev.generate(fp)
    step = 3
    pair = [TOKEN_ID["buy"], TOKEN_ID["sell"]]
    lp = torch.log_softmax(logits[step].float(), dim=-1)
    base_row = {"realized_status": "ok", "realized_step": step, "realized_token_ids": pair, "new_ids": new_ids}
    assert ev.teacher_forced_margin(fp, base_row) == pytest.approx(float(lp[pair[0]] - lp[pair[1]]), abs=1e-4)
    assert ev.teacher_forced_margin(fp, dict(base_row, realized_status="unparsed")) is None


# ── directions ─────────────────────────────────────────────────────────────────────────


def _groups(k=12, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    top = torch.randn(10, k * 2, d, generator=g) + 1.0
    bottom = torch.randn(10, k * 2, d, generator=g)
    return top, bottom


def test_dim_difference_and_dose_modes():
    top, bottom = _groups()
    d, stats = D.fit_dim_difference(top, bottom)
    assert torch.allclose(d, top.mean(0) - bottom.mean(0)) and len(stats["difference_sha256"]) == 64
    old, _ = cone.fit_dim_difference(top, bottom)
    assert torch.equal(old, d)
    u, sha = D.shared_random_direction(16, 3)
    assert D.shared_random_direction(16, 3)[1] == sha and D.shared_random_direction(16, 4)[1] != sha
    base = D.equal_norm(d, u)
    assert torch.allclose(base.norm(dim=-1), d.norm(dim=-1), rtol=1e-5)
    assert torch.allclose(D.unit_rows(base), u.expand_as(base), atol=1e-6)     # shared across tokens
    axes, info = D.cone_axes(top, bottom, d, list(range(10)), list(range(-10, 0)))
    d_hat = D.unit_rows(d)
    for axis in axes:
        assert torch.allclose(axis.norm(dim=-1), torch.ones(axis.shape[0]), atol=1e-5)
        assert (axis * d_hat).sum(-1).abs().max() <= 1e-5
    c4 = D.cone_unit(d, axes, 4)
    assert torch.allclose((c4 * d_hat).sum(-1), torch.full((d.shape[0],), 0.5), atol=1e-5)
    assert torch.allclose(D.cone_unit(d, axes, 1), d_hat)
    proj = D.equal_projection(d, c4)
    assert torch.allclose((proj * d_hat).sum(-1), d.norm(dim=-1), rtol=1e-4)
    assert torch.allclose(proj.norm(dim=-1), 2 * d.norm(dim=-1), rtol=1e-4)
    table = D.dose_table(d, D.equal_norm(d, c4), median_h=10.0)
    assert table["cos_to_dim_median"] == pytest.approx(0.5, abs=1e-5)
    rand_axes, digests = D.orth_random_axes(d, [7, 8, 9])
    assert len(digests) == 3 and (rand_axes[0] * d_hat).sum(-1).abs().max() <= 1e-5
    assert info["n_rows"] == 100 * d.shape[0]


def test_neuron_rule_on_dense_and_sandwich_norm_checkpoint(tmp_path):
    from safetensors.torch import save_file

    spec = R.ModelSpec("toy", 2, 1, (1,), 4, "bf16", "0" * 64, "dense", "m.{layer}.down.weight", "m.{layer}.post.weight")
    down = torch.zeros(4, 6)
    down[:, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    down[:, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    gain = torch.tensor([0.1, 5.0, 1.0, 1.0])
    save_file({"m.1.down.weight": down, "m.1.post.weight": gain}, str(tmp_path / "model.safetensors"))
    writes, labels = D.neuron_write_vectors(tmp_path, spec, 1)
    assert writes.shape == (6, 4) and torch.allclose(writes[4], torch.tensor([0.0, 5.0, 0.0, 0.0]))
    pick = D.select_neuron(writes, labels, torch.tensor([1.0, 1.2, 0.0, 0.0]))
    assert pick["neuron"] == "4" and pick["cos"] > pick["second_cos"]


def test_loso_placebo_and_shuffle_groups_exclude_correctly():
    sectors = ["Energy", "Unspecified", "Utilities", "Financials"]
    companies = {f"T{i:02}": {"sector": sectors[i % 4]} for i in range(80)}
    ranking = [{"ticker": t, "margin": float(i)} for i, t in enumerate(sorted(companies))]
    folds = D.loso_folds(ranking, companies, "Energy", seed=5)
    energy = {t for t, c in companies.items() if c["sector"] == "Energy"}
    unspecified = {t for t, c in companies.items() if c["sector"] == "Unspecified"}
    loso = set(folds["loso"]["top_10"] + folds["loso"]["bottom_10"])
    assert not loso & (energy | unspecified)
    assert not set(folds["comparator"]["top_10"] + folds["comparator"]["bottom_10"]) & unspecified
    assert folds["placebo"]["excluded_n"] == folds["loso"]["excluded_n"]
    top, bottom = D.shuffled_groups(ranking, seed=100)
    real_top, real_bottom = D.top_bottom(ranking)
    assert not (set(top) | set(bottom)) & (set(real_top) | set(real_bottom)) and len(set(top) | set(bottom)) == 20


# ── summary ────────────────────────────────────────────────────────────────────────────


def _row(decision, margin=0.0, kind=None, realized=None):
    return {"decision": decision, "margin": margin, "unparsed_kind": kind, "realized_margin": realized}


def test_flip_stats_itt_conditional_and_null_denominators():
    baseline = {"A": _row("sell"), "B": _row("sell"), "C": _row("buy"), "D": _row("sell")}
    steered = {"A": _row("buy", 2), "B": _row("unparsed", 1, "collapsed"), "C": _row("sell", 1), "D": _row("sell")}
    up = S.flip_stats(baseline, steered, 1.0)
    assert (up["on_class_n"], up["on_flip"], up["on_flip_itt"]) == (3, 1, 1 / 3)
    assert up["on_flip_conditional"] == 0.5 and up["off_flip_itt"] == 1.0 and up["collapsed"] == 1
    only_buy = S.flip_stats({"C": _row("buy")}, {"C": _row("sell")}, 1.0)
    assert only_buy["on_flip_itt"] is None and only_buy["on_class_n"] == 0


def _ladder(threshold, parse_fail_above=None, companies=24, buy=12, fail_n=None):
    baseline = {f"C{i}": _row("buy" if i < buy else "sell") for i in range(companies)}
    rows = {}
    for i, (t, b) in enumerate(baseline.items()):
        rows[t] = {}
        for m in S.CAL_LADDER:
            for sign in (1.0, -1.0):
                failing = fail_n is None or i % (companies // fail_n) == 0
                if parse_fail_above is not None and m > parse_fail_above and failing:
                    rows[t][sign * m] = _row("unparsed", kind="collapsed")
                    continue
                goal = "buy" if sign > 0 else "sell"
                flipped = m >= threshold and b["decision"] != goal
                rows[t][sign * m] = _row(goal if flipped else b["decision"])
    return baseline, rows


def test_cal_rule_derives_grid_and_bounds():
    baseline, rows = _ladder(threshold=1.0, parse_fail_above=8.0)
    cal = S.cal_rule(baseline, rows)
    assert cal["alpha_50"] == 1.0 and cal["alpha_hi"] == 8.0 and cal["alpha_lo"] == 0.5 and not cal["abort"]
    grid = cal["full_grid"]
    assert 0.0 in grid and 2.0 in grid and -2.0 in grid and 16.0 in grid and len(grid) - 1 <= S.GRID_CAP
    assert cal["reduced_grid"] == [-8.0, -1.0, 0.0, 1.0, 8.0]
    assert cal["collapse_dose"] == 16.0


def test_cal_rule_censoring_and_abort():
    baseline, rows = _ladder(threshold=0.01)
    assert S.cal_rule(baseline, rows)["left_censored"] is True
    baseline, rows = _ladder(threshold=1000)
    cal = S.cal_rule(baseline, rows)
    assert cal["alpha_50"] is None and cal["abort"] and cal["full_grid"] is None
    baseline, rows = _ladder(threshold=4.0, parse_fail_above=1.0, fail_n=4)   # 4/24 unparsed above 1
    cal = S.cal_rule(baseline, rows)
    assert cal["alpha_hi"] == 1.0 and cal["alpha_50"] == 4.0 and "alpha_hi < alpha_50" in cal["abort"]


def test_cal_rule_uses_cross_condition_fallback_for_empty_class():
    baseline, rows = _ladder(threshold=2.0, buy=24)      # every company buys at alpha 0
    fb_base, fb_rows = _ladder(threshold=0.5, buy=0)      # e.g. the neg condition: every company sells
    cal = S.cal_rule(baseline, rows, {"sell->buy": {"baseline": fb_base, "rows": fb_rows}})
    assert cal["directions"]["sell->buy"]["cross_condition"] is True
    assert cal["alpha_flip50"] == {"sell->buy": 0.5, "buy->sell": 2.0} and cal["alpha_50"] == 1.0


def test_full_grid_respects_cap_and_keeps_anchors():
    grid = S.full_grid(0.0625, 32.0, 4.0)
    nonzero = [a for a in grid if a]
    assert len(nonzero) <= S.GRID_CAP and 2.0 in grid and 64.0 in grid and 4.0 in grid and 32.0 in grid
    assert grid == sorted(grid) and all(-a in grid for a in grid)


def test_bootstrap_c5_smoothness_and_flip_dose():
    ci = S.bootstrap_ci(list(range(20)), lambda xs: sum(xs) / len(xs), samples=200, seed=1)
    assert ci == S.bootstrap_ci(list(range(20)), lambda xs: sum(xs) / len(xs), samples=200, seed=1)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    baseline = {f"C{i}": _row("sell") for i in range(30)}
    dim = {t: _row("buy") for t in baseline}
    rand = {"s0": {t: _row("sell") for t in baseline}, "s1": {t: _row("sell") for t in baseline}}
    assert S.c5_contrast(baseline, dim, rand, 1.0, samples=200)["supports"] is True
    smooth = S.smoothness([(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (4.0, 3.5)])
    assert smooth["monotone_step_fraction"] == 1.0 and smooth["reversals"] == 0 and smooth["spearman"] == 1.0
    rows = {0.5: _row("sell"), 1.0: _row("buy"), -1.0: _row("sell")}
    assert S.flip_dose(_row("sell"), rows, 1.0) == {"eligible": True, "dose": 1.0, "censor": None}
    assert S.flip_dose(_row("sell"), {0.5: _row("unparsed"), 1.0: _row("buy")}, 1.0)["censor"] == "collapsed"
    assert S.flip_dose(_row("sell"), {0.5: _row("sell")}, 1.0)["censor"] == "blocked"
    assert S.flip_dose(_row("buy"), rows, 1.0) == {"eligible": False}
    assert math.isclose(S.spearman([1, 2, 3], [3, 2, 1]), -1.0)
