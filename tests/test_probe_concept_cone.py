"""Deterministic contract tests for the exploratory cross-model cone CLI (no checkpoints)."""
from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location("probe_concept_cone", Path(__file__).resolve().parents[1] / "scripts/probe_concept_cone.py")
assert spec and spec.loader
cone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cone)


def test_paper_layer_source_is_bound_to_427_company_peak(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"curves": {"instruction": {
        "15": {"mean_normalized_transfer": 0.3, "n_directions": 854},
        "16": {"mean_normalized_transfer": 0.4, "n_directions": 854},
    }}}), encoding="utf-8")
    row = cone.c2_427_layer_source("qwen3.5-4b")
    assert row["instruction_peak"] == 16 and row["n_directions"] == 854
    assert len(row["sha256"]) == 64
    source.write_text(json.dumps({"curves": {"instruction": {
        "15": {"mean_normalized_transfer": 0.8, "n_directions": 854},
        "16": {"mean_normalized_transfer": 0.4, "n_directions": 854},
    }}}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not select"):
        cone.c2_427_layer_source("qwen3.5-4b")
    assert cone.C2_LAYERS["qwen3.5-4b"] == 15  # old run stays reproducible


def test_full_population_split_is_deterministic_and_disjoint():
    path = Path("data/sp500_constituents_2020_2025.csv")
    companies, construction, evaluation = cone.split_population(path, 20260923)
    assert (len(companies), len(construction), len(evaluation)) == (503, 402, 101)
    assert set(construction).isdisjoint(evaluation)
    assert (construction, evaluation) == cone.split_population(path, 20260923)[1:]
    assert (construction, evaluation) != cone.split_population(path, 42)[1:]


def test_population_fails_closed_when_missing_row(tmp_path):
    source = Path("data/sp500_constituents_2020_2025.csv")
    rows = source.read_text(encoding="utf-8").splitlines()
    dest = tmp_path / "population.csv"
    dest.write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="503"):
        cone.split_population(dest, 20260923)


def test_common_suffix_rejects_short_or_mismatched_tokens(monkeypatch):
    monkeypatch.setattr(cone, "_render_frozen_prompt", lambda ticker, name, **kw: ticker)
    monkeypatch.setattr(cone, "format_prompt", lambda tokenizer, prompt, **kw: prompt)
    monkeypatch.setattr(cone, "instruction_char_span", lambda prompt: (0, len(prompt)))
    monkeypatch.setattr(cone, "token_span", lambda tokenizer, text, *args, **kw: (0, 21))
    endings = {"AAA": list(range(21)), "BBB": [1234] + list(range(1, 21))}
    monkeypatch.setattr(cone, "input_ids", lambda tokenizer, text, **kw: endings[text])
    assert cone.prepare_instruction_suffix(None, {t: {"name": t} for t in endings}) == 20
    endings["BBB"][-1] = 77
    with pytest.raises(ValueError, match="common instruction suffix"):
        cone.prepare_instruction_suffix(None, {t: {"name": t} for t in endings})


def test_model_local_ranking_uses_prefix_and_tie_breaks(monkeypatch):
    companies = {t: {"name": t} for t in ["AAA", "BBB", "CCC"]}
    monkeypatch.setattr(cone, "_render_frozen_prompt", lambda ticker, name, **kw: ticker)
    monkeypatch.setattr(cone, "format_prompt", lambda tokenizer, prompt, **kw: prompt)
    monkeypatch.setattr(cone, "answer_token_ids", lambda tokenizer, text: (1, 2))
    seen = []

    def ids(tokenizer, text, **kw):
        assert text.endswith(cone.DECISION_PREFIX)
        seen.append(text)
        return [ord(text[0])]

    monkeypatch.setattr(cone, "input_ids", ids)
    monkeypatch.setattr(cone, "record_residuals", lambda model, ids, layers: {0: ids.float().unsqueeze(-1)})
    monkeypatch.setattr(cone, "fp32_next_token_log_probs", lambda model, res: res)
    monkeypatch.setattr(cone, "margin_from_log_probs", lambda res, buy_id, sell_id: 1.0 if res.item() == ord("C") else 0.0)
    model = SimpleNamespace(n_layers=1, input_device=torch.device("cpu"))
    result = cone.rank_construction(model, None, companies, ["CCC", "BBB", "AAA"])
    assert [r["ticker"] for r in result] == ["AAA", "BBB", "CCC"]
    assert len(seen) == 3


def test_alpha_table_requires_complete_grid():
    rows = [{"alpha": a, "margin": a - 2, "decision": "sell" if a == 0 else "buy"} for a in (0.0, 2.0)]
    result = {"targets": {"MO": {"cone_centroid": rows}}}
    table = cone.alpha_table(result, ["MO"], [0.0, 2.0])
    assert "| ticker | alpha=0 | alpha=2 |" in table
    assert "| MO | -2.000 (sell) | +0.000 (buy) |" in table
    with pytest.raises(ValueError, match="incomplete alpha"):
        cone.alpha_table(result, ["MO"], [0.0, 3.0])
    counts = cone.alpha_summary(result, ["MO"], [0.0, 2.0])
    assert counts[1]["flipped"] == 1 and counts[1]["valid_flip_pairs"] == 1


def test_evaluation_scores_prefix_but_generates_from_plain_prompt(monkeypatch):
    monkeypatch.setattr(cone, "_render_frozen_prompt", lambda ticker, name, **kw: "PROMPT")
    monkeypatch.setattr(cone, "format_prompt", lambda tokenizer, prompt, **kw: prompt)
    monkeypatch.setattr(cone, "token_span", lambda tokenizer, text, *args, **kw: (0, 20))
    monkeypatch.setattr(cone, "answer_token_ids", lambda tokenizer, text: (1, 2))
    monkeypatch.setattr(cone, "input_ids", lambda tokenizer, text, **kw: [1] * 25 + ([2, 3] if text.endswith(cone.DECISION_PREFIX) else []))
    monkeypatch.setattr(cone, "InjectedModelAdapter", lambda model, hf_model: model)
    monkeypatch.setattr(cone, "fp32_next_token_log_probs", lambda model, residual: residual)
    monkeypatch.setattr(cone, "margin_from_log_probs", lambda log_probs, buy, sell: 0.5)
    seen = []

    def record(model, ids, layers):
        seen.append(("score", ids.shape[1]))
        return {1: torch.zeros(1, ids.shape[1], 3)}

    def generate(model, ids, config):
        seen.append(("generate", ids.shape[1]))
        return torch.cat([ids, torch.tensor([[9]])], dim=1)

    monkeypatch.setattr(cone, "record_residuals", record)
    monkeypatch.setattr(cone, "generate_tokens", generate)

    @contextmanager
    def intervention(model, transforms):
        # Assert injection follows common instruction suffix rather than legacy 100.
        transformed = transforms[0](torch.zeros(1, 25, 3))
        assert torch.count_nonzero(transformed[:, :2, :]) == 0
        assert torch.count_nonzero(transformed[:, 2:20, :]) == 54
        yield

    monkeypatch.setattr(cone, "residual_interventions", intervention)

    class Tokenizer:
        def __call__(self, text, return_tensors):
            return SimpleNamespace(input_ids=torch.tensor([[1] * 25]))

        def decode(self, ids, **kwargs):
            return '{"decision":"buy", "reason":"' + 'x' * 1100 + '"}'

    model = SimpleNamespace(input_device=torch.device("cpu"), n_layers=2)
    result = cone.run_cone_evaluation(model, Tokenizer(), torch.ones(18, 3, 4),
                                       {"MO": {"name": "MO"}}, ["MO"], [0.0, 2.0],
                                       inject_layer=0, fixed_prefix=True, save_full_text=True,
                                       operator_label="Token-wise DIM")
    assert seen == [("score", 27), ("generate", 25)] * 2
    assert result["targets"]["MO"]["cone_centroid"][0]["parse_ok"] is True
    assert len(result["targets"]["MO"]["cone_centroid"][0]["generated_text"]) > 1024


def test_dim_direction_unit_vectors_and_fail_closed():
    top = torch.zeros(10, 100, 3)
    bottom = torch.zeros_like(top)
    top[:, :, 0] = 3
    bottom[:, :, 0] = 1
    direction, norms = cone.fit_dim_direction(top, bottom)
    assert direction.shape == (100, 3)
    assert torch.allclose(direction[:, 0], torch.ones(100))
    assert torch.allclose(direction[:, 1:], torch.zeros(100, 2))
    assert norms["min"] == pytest.approx(2)
    one, scalar_norms = cone.fit_dim_direction(top[:, -1, :], bottom[:, -1, :])
    assert one.shape == (3,) and one.tolist() == pytest.approx([1, 0, 0])
    assert scalar_norms["min"] == pytest.approx(2)
    top[0, 3] = bottom[0, 3]
    top[1:, 3] = bottom[1:, 3]
    with pytest.raises(ValueError, match="degenerate"):
        cone.fit_dim_direction(top, bottom)
    top[0, 3, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        cone.fit_dim_direction(top, bottom)
    with pytest.raises(ValueError, match="shape"):
        cone.fit_dim_direction(top[:9], bottom)


def test_dim_c2_preselection_checks_every_layer_and_fixed_threshold():
    curve = {str(i): {"mean_normalized_transfer": -0.001, "n_directions": 854} for i in range(32)}
    for layer, t in {14: 0.2542, 15: 0.4004, 16: 0.4076, 17: 0.3193, 18: 0.2866}.items():
        curve[str(layer)]["mean_normalized_transfer"] = t
    chosen = cone.validate_dim_c2_curve(curve)
    assert list(chosen) == [0, 14, 15, 16, 17, 18, 31]
    assert chosen[16] == pytest.approx(0.4076)
    curve["2"]["mean_normalized_transfer"] = 0.35
    with pytest.raises(ValueError, match="candidate"):
        cone.validate_dim_c2_curve(curve)
    curve["2"]["mean_normalized_transfer"] = -0.001
    curve["31"]["n_directions"] = 853
    with pytest.raises(ValueError, match="854"):
        cone.validate_dim_c2_curve(curve)


def test_dim_ranking_source_and_eval_disjointness_fail_closed():
    build = [f"B{i:03}" for i in range(402)]
    heldout = [f"E{i:03}" for i in range(101)]
    ranked = [{"ticker": t, "margin": float(i)} for i, t in enumerate(build)]
    expected = {k: f"value_{k}" for k in cone.DIM_PAPER_SOURCE_FIELDS}
    expected.update({"schema": "concept-cone-sp500-paper-v1", "mode": "evaluation", "model_slug": "qwen3.5-4b", "layer": 16,
                     "dtype": "bf16", "split_seed": 20260923, "alphas": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                     "max_new_tokens": 192, "decision_prefix": cone.DECISION_PREFIX,
                     "construction_tickers": build, "evaluation_tickers": heldout, "k_pairs": 20, "k_cone": 4})
    source = {"metadata": expected.copy(), "complete": True, "effective_tokens": 100,
              "ranking": ranked, "top_20": build[-20:], "bottom_20": build[:20]}
    assert cone.validate_dim_paper_ranking(source, expected, build, heldout) == (build[-10:], build[:10])
    source["ranking"][0] = {"ticker": heldout[0], "margin": 0.0}
    with pytest.raises(ValueError, match="construction"):
        cone.validate_dim_paper_ranking(source, expected, build, heldout)
    source["ranking"][0] = {"ticker": build[0], "margin": 0.0}
    source["ranking"][3] = {"ticker": build[3], "margin": float("nan")}
    with pytest.raises(ValueError, match="nonfinite"):
        cone.validate_dim_paper_ranking(source, expected, build, heldout)
    source["ranking"][3] = {"ticker": build[3], "margin": 3.0}
    source["metadata"]["split_sha256"] = "tampered"
    with pytest.raises(ValueError, match="metadata"):
        cone.validate_dim_paper_ranking(source, expected, build, heldout)


def test_dim_summary_uses_alpha_zero_generated_decisions_and_handles_empty_class():
    def row(alpha, margin, decision):
        return {"alpha": alpha, "margin": margin, "decision": decision, "parse_ok": decision != "unparsed",
                "generated_text": '{"decision":"' + decision + '"}' if decision != "unparsed" else "truncated"}

    targets = {
        "A": {"L16": {"rows": [row(0.0, -1, "sell"), row(2.0, 2, "buy")]}},
        "B": {"L16": {"rows": [row(0.0, 1, "buy"), row(2.0, 2, "unparsed")]}},
    }
    summary = cone.dim_layer_summary(targets, ["A", "B"], 16, [0.0, 2.0])
    change = summary[1]
    assert change["n"] == 2 and change["strict_parsed"] == 1
    assert change["mean_delta_margin"] == pytest.approx(2)
    assert change["baseline_buy_n"] == 1 and change["baseline_sell_n"] == 1
    assert change["buy_valid_pairs"] == 0 and change["buy_to_sell_rate"] is None
    assert change["sell_valid_pairs"] == 1 and change["sell_to_buy"] == 1
    assert change["sell_to_buy_rate"] == pytest.approx(1)
    assert change["flip_rate"] == pytest.approx(1)
    assert "—" in cone.dim_alpha_table({16: summary}, [0.0, 2.0])
    targets["A"]["L16"]["rows"][1]["margin"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        cone.dim_layer_summary(targets, ["A", "B"], 16, [0.0, 2.0])


def test_dim_resume_rejects_cross_layer_or_alpha_corruption():
    row = {"alpha": 0.0, "margin": 1.0, "decision": "sell", "parse_ok": True,
           "generated_text": '{"decision":"sell"}'}
    meta = {"schema": "dim-tokenwise-paper-v1", "dim_arm": "tokenwise", "layers": [0, 16], "alphas": [0.0]}
    result = {"metadata": meta, "targets": {"A": {"L0": {"rows": [row]}}}, "complete": False}
    cone.validate_dim_resume(result, meta, ["A"], [0, 16], [0.0])
    result["targets"]["A"]["L99"] = {"rows": [row]}
    with pytest.raises(ValueError, match="layer"):
        cone.validate_dim_resume(result, meta, ["A"], [0, 16], [0.0])
    del result["targets"]["A"]["L99"]
    result["targets"]["A"]["L0"]["rows"] = [row, row]
    with pytest.raises(ValueError, match="alpha"):
        cone.validate_dim_resume(result, meta, ["A"], [0, 16], [0.0])
    result["targets"]["A"]["L0"]["rows"] = [row]
    result["complete"] = True
    with pytest.raises(ValueError, match="incomplete"):
        cone.validate_dim_resume(result, meta, ["A"], [0, 16], [0.0])


def test_dim_extraction_uses_each_layers_own_post_block_states(monkeypatch):
    top = [f"T{i}" for i in range(10)]
    bottom = [f"B{i}" for i in range(10)]
    companies = {ticker: {"name": ticker} for ticker in top + bottom}
    monkeypatch.setattr(cone, "_render_frozen_prompt", lambda ticker, name, **kw: ticker)
    monkeypatch.setattr(cone, "format_prompt", lambda tokenizer, text, **kw: text)
    monkeypatch.setattr(cone, "instruction_char_span", lambda text: (0, len(text)))
    monkeypatch.setattr(cone, "token_span", lambda *args, **kw: (0, 100))
    monkeypatch.setattr(cone, "input_ids", lambda tokenizer, text: [1 if text.startswith("T") else 0] + [0] * 99)
    calls = []

    def record(model, ids, layers):
        assert list(layers) == [0, 16]
        calls.append(ids[0, 0].item())
        data = {}
        for layer in layers:
            state = torch.zeros(1, 100, 3)
            state[:, :, 0] = (1 if ids[0, 0] else -1) * (1 if layer == 0 else -1)
            data[layer] = state
        return data

    monkeypatch.setattr(cone, "record_residuals", record)
    model = SimpleNamespace(input_device=torch.device("cpu"))
    directions = cone.extract_dim_layer_directions(model, None, companies, top, bottom, [0, 16], 100)
    assert len(calls) == 20
    assert directions[0][0][:, 0].tolist() == [1] * 100
    assert directions[16][0][:, 0].tolist() == [-1] * 100

    def record_pre(model, ids, layers):
        return {layer: {"pre": torch.tensor([[[float(1 if ids[0, 0] else -1) *
                                                (1 if layer == 0 else -1), 0.0, 0.0]] * 100])}
                for layer in layers}

    monkeypatch.setattr(cone, "record_block_states", record_pre, raising=False)
    single = cone.extract_dim_layer_directions(model, None, companies, top, bottom, [0, 16], 100,
                                               arm="single_all")
    assert single[0][0].tolist() == pytest.approx([1, 0, 0])
    assert single[16][0].tolist() == pytest.approx([-1, 0, 0])


def test_single_all_evaluator_applies_pre_block_to_every_token_and_decoding(monkeypatch):
    monkeypatch.setattr(cone, "_render_frozen_prompt", lambda ticker, name, **kw: "PROMPT")
    monkeypatch.setattr(cone, "format_prompt", lambda tokenizer, prompt, **kw: prompt)
    monkeypatch.setattr(cone, "token_span", lambda tokenizer, text, *args, **kw: (0, 20))
    monkeypatch.setattr(cone, "answer_token_ids", lambda tokenizer, text: (1, 2))
    monkeypatch.setattr(cone, "input_ids", lambda tokenizer, text, **kw: [1] * 25 + ([2, 3] if text.endswith(cone.DECISION_PREFIX) else []))
    monkeypatch.setattr(cone, "InjectedModelAdapter", lambda model, hf_model: model)
    monkeypatch.setattr(cone, "fp32_next_token_log_probs", lambda model, residual: residual)
    monkeypatch.setattr(cone, "margin_from_log_probs", lambda *args: 1.0)
    monkeypatch.setattr(cone, "record_residuals", lambda model, ids, layers: {1: torch.zeros(1, ids.shape[1], 256)})
    monkeypatch.setattr(cone, "generate_tokens", lambda model, ids, cfg: torch.cat([ids, torch.tensor([[9]])], 1))

    class Tokenizer:
        def __call__(self, text, return_tensors):
            return SimpleNamespace(input_ids=torch.ones(1, 25, dtype=torch.long))
        def decode(self, ids, **kwargs):
            return '{"decision":"buy"}'

    seen = []
    @contextmanager
    def pre_hook(model, transforms):
        seen.append("pre")
        fn = transforms[0]
        expected = torch.zeros(1, 25, 256)
        expected[:, :, 0] = 2
        assert torch.equal(fn(torch.zeros(1, 25, 256)), expected)
        assert torch.equal(fn(torch.zeros(1, 1, 256)), expected[:, :1, :])
        yield

    monkeypatch.setattr(cone, "pre_residual_interventions", pre_hook, raising=False)
    monkeypatch.setattr(cone, "residual_interventions", lambda *args: (_ for _ in ()).throw(AssertionError("wrong hook")))
    model = SimpleNamespace(input_device=torch.device("cpu"), n_layers=2)
    result = cone.run_cone_evaluation(model, Tokenizer(), torch.nn.functional.one_hot(
        torch.tensor(0), 256).float().unsqueeze(-1),
                                       {"MO": {"name": "MO"}}, ["MO"], [0.0, 2.0], inject_layer=0,
                                       fixed_prefix=True, pre_block_all=True, save_full_text=True)
    assert seen == ["pre"]
    assert result["targets"]["MO"]["cone_centroid"][1]["decision"] == "buy"


def test_dim_paper_fake_pipeline_keeps_layers_separate_and_validates_resume(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "qwen3.5-4b"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"fake")
    population = tmp_path / "population.csv"
    population.write_text("fake population")
    build = [f"B{i:03}" for i in range(402)]
    heldout = [f"E{i:03}" for i in range(101)]
    companies = {t: {"name": t} for t in build + heldout}
    monkeypatch.setattr(cone, "split_population", lambda *args: (companies, build, heldout))
    monkeypatch.setattr(cone, "_render_frozen_prompt", lambda ticker, name, **kw: ticker)
    monkeypatch.setattr(cone, "prepare_instruction_suffix", lambda *args: 100)
    tokenizer = SimpleNamespace(name_or_path="toy tokenizer")
    monkeypatch.setattr(cone, "load_model", lambda *args, **kw: (SimpleNamespace(n_layers=32), tokenizer, None))
    c2 = {"curves": {"instruction": {str(i): {"mean_normalized_transfer": -0.001, "n_directions": 854}
                                         for i in range(32)}}}
    for layer, t in {14: .2542, 15: .4004, 16: .4076, 17: .3193, 18: .2866}.items():
        c2["curves"]["instruction"][str(layer)]["mean_normalized_transfer"] = t
    c2_path = tmp_path / "c2.json"
    c2_path.write_text(json.dumps(c2))
    digest = lambda data: hashlib.sha256(data).hexdigest()
    source_info = {"path": str(c2_path), "sha256": digest(c2_path.read_bytes()),
                   "instruction_peak": 16, "n_directions": 854}
    monkeypatch.setattr(cone, "c2_427_layer_source", lambda slug: source_info)
    split_bytes = json.dumps({"construction": build, "evaluation": heldout}, sort_keys=True).encode()
    prompts = json.dumps([(ticker, ticker) for ticker in sorted(companies)],
                         ensure_ascii=False, separators=(",", ":")).encode()
    metadata = {"schema": "concept-cone-sp500-paper-v1", "mode": "evaluation", "model": str(model_dir),
                "model_slug": "qwen3.5-4b", "model_config_sha256": digest(b"{}"),
                "tokenizer_config_sha256": digest(b"{}"), "tokenizer": tokenizer.name_or_path,
                "checkpoint_files": [{"name": "model.safetensors", "bytes": 4}],
                "population_sha256": digest(population.read_bytes()), "split_seed": 20260923,
                "split_sha256": digest(split_bytes), "construction_tickers": build,
                "evaluation_tickers": heldout, "prompt_family_sha256": digest(prompts),
                "prompt_renderer": "entity_to_dial.heldout_transfer._render_frozen_prompt(order=0,reverse=False)",
                "decision_prefix": cone.DECISION_PREFIX, "layer": 16, "k_pairs": 20, "k_cone": 4,
                "alphas": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0], "dtype": "bf16",
                "max_new_tokens": 192, "c2_layer_source": source_info}
    ranked = [{"ticker": t, "margin": float(i)} for i, t in enumerate(build)]
    paper = {"metadata": metadata, "complete": True, "effective_tokens": 100,
             "ranking": ranked, "top_20": build[-20:], "bottom_20": build[:20]}
    base = tmp_path / "artifacts/qwen3.5-4b/concept-cone-steering/runs"
    paper_path = base / "c2-guided-paper-20260924/result.json"
    paper_path.parent.mkdir(parents=True)
    paper_path.write_text(json.dumps(paper))
    out = base / "dim-layer-sweep-v1-smoke-test/tokenwise/result.json"
    args = SimpleNamespace(model=str(model_dir), model_dtype="bf16", dim_arm="tokenwise",
                           dim_layers=[0, 16], alphas=metadata["alphas"], split_seed=20260923,
                           evidence_mode="balanced", inject_layer=None, eval_individual_rays=False,
                           centroid_dims=None, skip_eval=False, leave_out_sector=None,
                           persist_directions=None, output_json=str(out), population_csv=str(population),
                           smoke_tickers=[heldout[0]])
    calls = []

    def extract(_model, _tok, _companies, top, bottom, layers, length, *, arm):
        assert top == build[-10:] and bottom == build[:10] and length == 100
        ray = (torch.nn.functional.one_hot(torch.zeros(100, dtype=torch.long), 3).float()
               if arm == "tokenwise" else torch.tensor([1.0, 0.0, 0.0]))
        return {layer: (ray, {"min": 1.0, "median": 1.0, "max": 1.0}) for layer in layers}

    def evaluate(model, tokenizer, ray, _companies, targets, alphas, **kw):
        layer = kw["inject_layer"]
        calls.append(layer)
        assert ray.shape == ((3, 1) if kw["pre_block_all"] else (100, 3, 1))
        assert kw["fixed_prefix"] and kw["save_full_text"]
        rows = [{"alpha": alpha, "margin": float(layer + alpha), "decision": "sell", "parse_ok": True,
                 "generated_text": '{"decision":"sell"}'} for alpha in alphas]
        return {"targets": {targets[0]: {"cone_centroid": rows}}}

    monkeypatch.setattr(cone, "extract_dim_layer_directions", extract)
    monkeypatch.setattr(cone, "run_cone_evaluation", evaluate)
    cone.run_dim_paper(args)
    result = json.loads(out.read_text())
    assert calls == [0, 16]
    assert result["complete"] and set(result["targets"][heldout[0]]) == {"L0", "L16"}
    assert result["summary"]["16"][1]["mean_delta_margin"] == pytest.approx(2.0)
    assert "alpha=6" in out.with_suffix(".md").read_text()
    cone.run_dim_paper(args)
    assert calls == [0, 16]
    scalar_args = SimpleNamespace(**vars(args))
    scalar_args.dim_arm = "single_all"
    scalar_args.output_json = str(base / "dim-layer-sweep-v1-smoke-test/single_all/result.json")
    cone.run_dim_paper(scalar_args)
    scalar = json.loads(Path(scalar_args.output_json).read_text())
    assert scalar["metadata"]["schema"] == "dim-single-all-paper-v1"
    assert scalar["complete"] and calls == [0, 16, 0, 16]
    result["metadata"]["source_paper_sha256"] = "tampered"
    out.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="metadata"):
        cone.run_dim_paper(args)


def test_crossmodel_preselection_uses_frozen_threshold_and_peak_neighbors():
    for slug, high in (
        ("gemma4-12b-it", {26: .1544, 27: .1903, 28: .1430, 29: .1429,
                           30: .1549, 31: .1555, 32: .1391}),
        ("glm4-9b-0414", {17: .2394, 18: .2281, 19: .4592, 20: .2579, 21: .2199}),
        ("gpt-oss-20b", {12: .3162, 13: .4978, 14: .5323, 15: .5190, 16: .3276}),
    ):
        spec = cone.DIM_CROSSMODEL_CONFIG[slug]
        curve = {str(i): {"mean_normalized_transfer": -.005, "n_directions": 854}
                 for i in range(spec["n_layers"])}
        for layer, value in high.items():
            curve[str(layer)]["mean_normalized_transfer"] = value
        transfer = cone.validate_dim_crossmodel_curve(curve, spec)
        assert list(transfer) == list(spec["layers"])
        if slug == "gemma4-12b-it":
            assert 25 not in transfer  # approved seven-layer plateau, not unconditional peak±2
        curve[str(spec["n_layers"] - 1)]["n_directions"] = 853
        with pytest.raises(ValueError, match="854"):
            cone.validate_dim_crossmodel_curve(curve, spec)
        curve[str(spec["n_layers"] - 1)]["n_directions"] = 854
        curve["1"]["mean_normalized_transfer"] = .9
        with pytest.raises(ValueError, match="peak"):
            cone.validate_dim_crossmodel_curve(curve, spec)


def test_template_render_kwargs_pins_strftime_date_only_when_template_uses_it(monkeypatch):
    import datetime as dt
    import transformers.utils.chat_template_utils as ctu

    template = "{{ reasoning_effort | default('medium') }} {{ strftime_now('%Y-%m-%d') }}"
    harmony = SimpleNamespace(chat_template=template)

    class Tomorrow(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 9, 26, 0, 30)

    monkeypatch.setattr(ctu, "datetime", Tomorrow)
    render = lambda kwargs: ctu.render_jinja_template([[{"role": "user", "content": "x"}]], chat_template=template, **kwargs)[0][0]
    assert render(cone.low_reasoning_kwargs(harmony)) == "low 2026-09-26"
    assert render(cone.template_render_kwargs(harmony)) == "low 2026-09-25"
    plain = SimpleNamespace(chat_template="{{ messages[0].content }}")
    assert cone.template_render_kwargs(plain) == {} and not cone.template_uses_date(plain)


def test_dim_difference_keeps_raw_scale_and_allows_structural_zero():
    top = torch.zeros(10, 20, 3)
    bottom = torch.zeros_like(top)
    top[:, :, 0] = 3
    bottom[:, :, 0] = 1
    difference, norms = cone.fit_dim_difference(top, bottom)
    assert torch.allclose(difference[:, 0], torch.full((20,), 2.0))
    assert norms == {"min": 2.0, "median": 2.0, "max": 2.0}
    zero, zero_norms = cone.fit_dim_difference(bottom[:, -1, :], bottom[:, -1, :])
    assert torch.count_nonzero(zero) == 0 and zero_norms["max"] == 0.0
    with pytest.raises(ValueError, match="degenerate"):
        cone.fit_dim_direction(bottom[:, -1, :], bottom[:, -1, :])


def v2_row(alpha, margin, text):
    return {"alpha": alpha, "margin": margin, "generated_text": text, **cone.parse_generation(text)}


BUY = '{"decision":"buy","reason":"evidence"}'
SELL = '{"decision":"sell","reason":"evidence"}'


def test_dim_v2_summary_pairs_both_directions_against_shared_baseline():
    nonzero = [a for a in cone.DIM_V2_ALPHAS if a != 0]

    def layer(texts):
        return {"rows": [v2_row(a, a, texts.get(a, "analysis only")) for a in nonzero]}

    targets = {
        "A": {"baseline": v2_row(0.0, 0.0, "analysis x.assistantfinal" + SELL),
              "L5": layer({1.0: BUY, 2.0: BUY, -2.0: SELL})},
        "B": {"baseline": v2_row(0.0, 0.0, "thought\n" + BUY),
              "L5": layer({-2.0: SELL, 2.0: BUY})},
    }
    assert targets["A"]["baseline"]["format"] == "harmony_final_bare_json"
    assert targets["A"]["baseline"]["strict_decision"] == "unparsed"
    summary = {row["alpha"]: row for row in cone.dim_v2_layer_summary(targets, ["A", "B"], 5)}
    assert [row for row in summary] == list(cone.DIM_V2_ALPHAS)
    assert summary[0.0]["parsed"] == 2 and summary[0.0]["strict_parsed"] == 0
    assert summary[2.0]["sell_to_buy"] == 1 and summary[2.0]["buy_to_sell"] == 0
    assert summary[-2.0]["buy_to_sell"] == 1 and summary[-2.0]["buy_valid_pairs"] == 1
    assert summary[-0.5]["valid_flip_pairs"] == 0 and summary[-0.5]["flip_rate"] is None
    assert summary[2.0]["mean_delta_margin"] == pytest.approx(2.0)
    table = cone.dim_v2_alpha_table({5: list(summary.values())})
    assert "| layer | alpha=-2 | alpha=-1 | alpha=-0.5 | alpha=0 | alpha=0.5 | alpha=1 | alpha=2 |" in table
    assert "| L5 | 0/1 | 0/0" not in table and "—" in table


def test_dim_v2_resume_requires_baseline_and_rederived_decisions():
    nonzero = [a for a in cone.DIM_V2_ALPHAS if a != 0]
    meta = {"schema": "dim-tokenwise-crossmodel-v2"}
    result = {"metadata": meta, "complete": False, "targets": {"A": {
        "baseline": v2_row(0.0, 1.0, BUY), "L5": {"rows": [v2_row(a, a, SELL) for a in nonzero]}}}}
    cone.validate_dim_v2_resume(result, meta, ["A"], [5, 7])
    result["targets"]["A"]["L5"]["rows"][0]["decision"] = "buy"
    with pytest.raises(ValueError, match="do not match"):
        cone.validate_dim_v2_resume(result, meta, ["A"], [5, 7])
    result["targets"]["A"]["L5"]["rows"][0]["decision"] = "sell"
    result["targets"]["A"]["L5"]["rows"][0]["parse_ok"] = True
    with pytest.raises(ValueError, match="row fields"):
        cone.validate_dim_v2_resume(result, meta, ["A"], [5, 7])
    del result["targets"]["A"]["L5"]["rows"][0]["parse_ok"]
    del result["targets"]["A"]["baseline"]
    with pytest.raises(ValueError, match="layer mismatch"):
        cone.validate_dim_v2_resume(result, meta, ["A"], [5, 7])
    result["targets"]["A"]["baseline"] = v2_row(0.0, 1.0, BUY)
    result["complete"] = True
    with pytest.raises(ValueError, match="incomplete"):
        cone.validate_dim_v2_resume(result, meta, ["A"], [5, 7])


@pytest.mark.parametrize("slug", ["gemma4-12b-it", "glm4-9b-0414", "gpt-oss-20b"])
@pytest.mark.parametrize("arm", ["tokenwise", "single_all"])
def test_dim_v2_fake_run_ranks_locally_and_shares_alpha_zero(tmp_path, monkeypatch, slug, arm):
    import hashlib

    monkeypatch.chdir(tmp_path)
    spec = dict(cone.DIM_CROSSMODEL_CONFIG[slug])
    model_dir = tmp_path / slug
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"fake")
    population = tmp_path / "population.csv"
    population.write_text("fake population")
    construction = [f"B{i:03}" for i in range(402)]
    evaluation = [f"E{i:03}" for i in range(101)]
    companies = {t: {"name": t} for t in construction + evaluation}
    monkeypatch.setattr(cone, "split_population", lambda *args: (companies, construction, evaluation))
    monkeypatch.setattr(cone, "_render_frozen_prompt", lambda ticker, name, **kw: ticker)
    monkeypatch.setattr(cone, "prepare_instruction_suffix", lambda *args: spec["suffix_tokens"])
    template = "{{ reasoning_effort }}" if slug == "gpt-oss-20b" else "{{ enable_thinking }}"
    tokenizer = SimpleNamespace(name_or_path="toy tokenizer", chat_template=template)

    def fake_load_model(path, *, dtype):
        assert dtype == ("native" if slug == "gpt-oss-20b" else None)
        return SimpleNamespace(n_layers=spec["n_layers"]), tokenizer, None

    monkeypatch.setattr(cone, "load_model", fake_load_model)
    ranking = [{"ticker": t, "margin": float(-i)} for i, t in enumerate(reversed(construction))]
    ranking.sort(key=lambda r: (r["margin"], r["ticker"]))
    monkeypatch.setattr(cone, "rank_construction", lambda model, tok, comp, tickers: (
        tickers == construction or pytest.fail("ranks only construction")) and ranking)
    c2 = {"curves": {"instruction": {str(i): {"mean_normalized_transfer": -.005, "n_directions": 854}
                                     for i in range(spec["n_layers"])}}}
    high = {"gemma4-12b-it": {26: .1544, 27: .1903, 28: .1430, 29: .1429, 30: .1549, 31: .1555, 32: .1391},
            "glm4-9b-0414": {17: .2394, 18: .2281, 19: .4592, 20: .2579, 21: .2199},
            "gpt-oss-20b": {12: .3162, 13: .4978, 14: .5323, 15: .5190, 16: .3276}}[slug]
    for layer, t in high.items():
        c2["curves"]["instruction"][str(layer)]["mean_normalized_transfer"] = t
    c2_path = tmp_path / "c2.json"
    c2_path.write_text(json.dumps(c2))
    spec["c2_sha256"] = hashlib.sha256(c2_path.read_bytes()).hexdigest()
    monkeypatch.setattr(cone, "c2_427_layer_source", lambda s: {
        "path": str(c2_path), "sha256": spec["c2_sha256"], "instruction_peak": spec["peak"], "n_directions": 854})
    monkeypatch.setattr(cone, "DIM_CROSSMODEL_CONFIG", {slug: spec})
    base = tmp_path / "artifacts" / slug / "concept-cone-steering" / "runs"
    out = base / f"dim-crossmodel-layer-sweep-v2-test/{arm}/result.json"
    layers = [spec["peak"], spec["n_layers"] - 1]
    args = SimpleNamespace(model=str(model_dir), model_dtype=spec["dtype"], dim_arm=arm, dim_layers=layers,
                           alphas=list(cone.DIM_V2_ALPHAS), split_seed=20260923, evidence_mode="balanced",
                           inject_layer=None, eval_individual_rays=False, centroid_dims=None, skip_eval=False,
                           leave_out_sector=None, persist_directions=None, output_json=str(out),
                           population_csv=str(population), smoke_tickers=evaluation[:2])

    def extract(_model, _tokenizer, _companies, top, bottom, layer_list, length, *, arm, normalize):
        assert not normalize and top == [r["ticker"] for r in ranking[-10:]]
        assert bottom == [r["ticker"] for r in ranking[:10]] and length == spec["suffix_tokens"]
        ray = torch.full((3,), 5.0) if arm == "single_all" else torch.full((length, 3), 5.0)
        return {layer: (ray, {"min": 5.0, "median": 5.0, "max": 5.0}) for layer in layer_list}

    calls = []

    def evaluate(_model, _tokenizer, ray, _companies, tickers, alphas, **kw):
        calls.append((kw["inject_layer"], tuple(alphas)))
        assert torch.all(ray == 5.0)  # raw difference, never unit-normalized
        assert kw["pre_block_all"] == (arm == "single_all") and kw["max_new_tokens"] == 192
        text = lambda a: ("analysis.assistantfinal" if slug == "gpt-oss-20b" else "") + (BUY if a > 0 else SELL)
        rows = [{"alpha": a, "margin": a, "decision": "unparsed", "parse_ok": False, "generated_text": text(a)}
                for a in alphas]
        return {"targets": {tickers[0]: {"cone_centroid": rows}}}

    monkeypatch.setattr(cone, "extract_dim_layer_directions", extract)
    monkeypatch.setattr(cone, "run_cone_evaluation", evaluate)
    cone.run_dim_crossmodel_v2(args)
    result = json.loads(out.read_text())
    nonzero = tuple(a for a in cone.DIM_V2_ALPHAS if a != 0)
    assert calls == [(layers[0], (0.0,))] * 2 + [(layers[0], nonzero)] * 2 + [(layers[1], nonzero)] * 2
    meta = result["metadata"]
    assert meta["schema"] == f"dim-{'tokenwise' if arm == 'tokenwise' else 'single-all'}-crossmodel-v2"
    assert meta["chat_template_kwargs"] == ({"reasoning_effort": "low"} if slug == "gpt-oss-20b" else {})
    assert meta["top_10"] == [r["ticker"] for r in ranking[-10:]] and result["ranking"] == ranking
    peak = {row["alpha"]: row for row in result["summary"][str(spec["peak"])]}
    assert peak[0.0]["baseline_sell_n"] == 2 and peak[0.5]["sell_to_buy"] == 2
    assert result["format_counts"] == {("harmony_final_bare_json" if slug == "gpt-oss-20b" else "bare_json"): 26}
    cone.run_dim_crossmodel_v2(args)
    assert len(calls) == 6  # complete result is read-only
    result["metadata"]["top_10"] = list(reversed(result["metadata"]["top_10"]))
    out.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="metadata"):
        cone.run_dim_crossmodel_v2(args)
    for field, value, match in (("alphas", [0.0, 2.0], "alpha"),
                                ("model_dtype", "native" if spec["dtype"] == "bf16" else "bf16", "dtype"),
                                ("output_json", str(base / f"dim-crossmodel-layer-sweep-v1-x/{arm}/result.json"), "output")):
        bad = SimpleNamespace(**{**vars(args), field: value})
        with pytest.raises(ValueError, match=match):
            cone.run_dim_crossmodel_v2(bad)
