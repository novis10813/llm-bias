"""Deterministic CPU contracts; no checkpoint downloads or GPU inference."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from llm_bias.core.artifacts.io import write_json, read_jsonl
from llm_bias.core.inference.mlp import mlp_coordinates
from llm_bias.core.inference.continuations import score_encoded_suffix
from llm_bias.financial_soundness.prompts import build_prompts, validate_prompts, prepare_encoded
from llm_bias.financial_soundness.analysis import rank_candidates, summarize_effects
from llm_bias.financial_soundness.pipeline import run_localization, run_causal_validation, _load_source
from llm_bias.financial_soundness.cli import build_parser, main


class Tokenizer:
    special_tokens_map = {}
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}
    def get_vocab(self):
        return {chr(i): i for i in range(128)}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(8, 8, bias=False)
    def forward(self, x):
        return x + self.mlp.down_proj(torch.tanh(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            self.embed = nn.Embedding(128, 8)
            self.layers = nn.ModuleList([Block()])
            self._lm_head = nn.Linear(8, 128, bias=False)
        self._final_norm = nn.Identity()
        self.n_layers = 1
    def forward(self, ids, attention_mask=None):
        x = self.embed(ids)
        x = x.cumsum(1) / torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self._lm_head(x))


def small_data():
    data = build_prompts(["Company A", "Company B"])
    seen, pairs = set(), []
    for pair in data["pairs"]:
        key = pair["family"], pair["split"], pair["answer_mode"]
        if key not in seen:
            seen.add(key)
            pairs.append(pair)
    data["pairs"] = pairs
    return data


@pytest.fixture
def loaded(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{}')
    (model_dir / "pytorch_model.bin").write_bytes(b"fake-weights")
    return str(model_dir), (Model().eval(), Tokenizer(), "cpu")


def test_prompts_have_three_exploration_families_and_controls():
    data = build_prompts(["Company A", "Company B"])
    validate_prompts(data)
    assert build_prompts(["Company A", "Company B"]) == data
    company = [p for p in data["pairs"] if p["family"] == "company"]
    assert all(p["a"]["expected"] is None for p in company)
    evidence = [p for p in data["pairs"] if p["family"] == "evidence"]
    assert all(p["a"]["expected"] == 1 and p["b"]["expected"] == -1 for p in evidence)
    assert all("Company A" not in p["a"]["text"] for p in evidence)
    reversed_pair = next(p for p in evidence if p["answer_mode"] == "labels-reversed")
    assert reversed_pair["a"]["positive"] == " B"
    for mutation in ("duplicate", "control", "split", "truth"):
        broken = copy.deepcopy(data)
        if mutation == "duplicate":
            broken["pairs"].append(broken["pairs"][0])
        elif mutation == "control":
            broken["pairs"] = [p for p in broken["pairs"] if p["family"] != "comparison"]
        elif mutation == "split":
            broken["pairs"][-1]["template_id"] = 0
        else:
            next(p for p in broken["pairs"] if p["family"] == "evidence")["a"]["expected"] = None
        with pytest.raises(ValueError):
            validate_prompts(broken)


def test_encoding_exact_suffix_and_fail_closed():
    data = small_data()
    encoded = prepare_encoded(data, Tokenizer())
    row = encoded["pairs"][0]["a"]
    assert row["position"] == len(row["prompt_ids"]) - 1
    assert len(row["positive_ids"]) > 1
    with pytest.raises(ValueError, match="overlong"):
        prepare_encoded(data, Tokenizer(), max_tokens=5)
    class Bad(Tokenizer):
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [len(text)]}
    with pytest.raises(ValueError, match="prefix"):
        prepare_encoded(data, Bad())


def test_hooks_scale_replace_and_cleanup():
    model = Model()
    module = model.layers[0].mlp.down_proj
    values = torch.ones(1, 4, 8)
    with mlp_coordinates(model, [0], 1, edits={0: {2: (0.0, 3.0)}}) as records:
        actual = module(values)
        expected = values.clone()
        expected[0, 1, 2] = 3.0
        assert torch.allclose(actual, nn.functional.linear(expected, module.weight))
        assert records[0][2] == 1
    assert not module._forward_pre_hooks
    with pytest.raises(RuntimeError):
        with mlp_coordinates(model, [0], 1):
            raise RuntimeError("forward failure")
    assert not module._forward_pre_hooks
    with pytest.raises(ValueError, match="neuron"):
        with mlp_coordinates(model, [0], 1, edits={0: {999: (0., None)}}):
            module(values)
    assert not module._forward_pre_hooks
    with pytest.raises(ValueError):
        with mlp_coordinates(model, [0], -1):
            pass


def test_fp32_multitoken_scoring_matches_direct_logits():
    model = Model()
    prompt, suffix = [1, 2, 3], [4, 5]
    expected = model(torch.tensor([prompt + suffix[:-1]])).logits.float().log_softmax(-1)
    expected = float((expected[0, 2, 4] + expected[0, 3, 5]).detach())
    assert score_encoded_suffix(model, prompt, suffix, "cpu") == pytest.approx(expected, abs=1e-6)


def test_heldout_cannot_change_ranking_and_zero_scales_excluded():
    responses = {}
    for split in ("discovery", "calibration", "held-out"):
        responses[("topic", "stability", "direct", split, 0)] = [(torch.tensor([2., 1., 0.]), torch.tensor([0., 1., 0.]))]
    first = rank_candidates(responses, 1)
    assert first[0]["neuron"] == 0
    responses[("topic", "stability", "direct", "held-out", 0)] = [(torch.tensor([1., 99., 0.]), torch.tensor([1., -99., 0.]))]
    assert rank_candidates(responses, 1)[0]["neuron"] == 0
    assert len(rank_candidates(responses, 3)) == 2


def test_summary_distinguishes_uniform_bias_from_discrimination():
    rows = []
    for name, sign in (("a", 1), ("b", -1)):
        rows.append(dict(layer=0, neuron=0, family="evidence", answer_mode="direct", split="held-out", condition="scale", scale=0.,
                         pair_id="p", group_id="n", member=name, margin=sign + 2., clean_margin=float(sign), correct_margin_delta=2. * sign))
    stat = summarize_effects(rows)[0]["statistics"]
    assert stat["discrimination_delta"]["mean"] == 0
    assert stat["answer_shift"]["mean"] == 2


def test_complete_pipeline_and_tamper_rejection(tmp_path, loaded):
    model_path, bundle = loaded
    source = tmp_path / "prompts.json"
    write_json(source, small_data())
    run = run_localization(source, model_path, "local", artifact_root=tmp_path / "runs", layers=[0], top_k=1, loaded=bundle)
    data, protocol, candidates, _ = _load_source(run)
    assert candidates
    assert protocol["formal_authorized"] is False
    causal = run_causal_validation(run, model_path, "causal", artifact_root=tmp_path / "runs", loaded=bundle)
    assert json.loads((causal / "manifest.json").read_text())["status"] == "complete"
    rows = read_jsonl(causal / "forward/effects.jsonl")
    assert {r["condition"] for r in rows} == {"clean", "scale", "random", "restore", "donor"}
    assert all(r["split"] != "discovery" for r in rows)
    assert all(r["margin"] == pytest.approx(r["clean_margin"], abs=1e-5) for r in rows if r["condition"] == "restore")
    assert {r["member"] for r in rows if r["condition"] == "donor"} == {"a", "b"}
    assert not bundle[0].layers[0].mlp.down_proj._forward_pre_hooks
    for artifact in causal.rglob("*.json*"):
        assert '"activations"' not in artifact.read_text()
    with pytest.raises(FileExistsError):
        run_localization(source, model_path, "local", artifact_root=tmp_path / "runs", layers=[0], loaded=bundle)
    weights = Path(model_path) / 'pytorch_model.bin'
    weights.write_bytes(b'changed-model')
    with pytest.raises(ValueError, match="identity"):
        run_causal_validation(run, model_path, "wrong-model", artifact_root=tmp_path / "runs", loaded=bundle)
    weights.write_bytes(b'fake-weights')
    (run / "prepare/prompts.json").write_text('{}')
    with pytest.raises(ValueError, match="hash"):
        _load_source(run)


def test_teacher_forcing_edits_prompt_position_not_answer_end():
    model = Model()
    prompt, suffix = [1, 2, 3], [4, 5]
    with mlp_coordinates(model, [0], 2, edits={0: {0: (0., None)}}):
        score = score_encoded_suffix(model, prompt, suffix, "cpu")
        logits = model(torch.tensor([prompt + suffix[:-1]])).logits.float().log_softmax(-1)
    expected = float((logits[0, 2, 4] + logits[0, 3, 5]).detach())
    assert score == pytest.approx(expected, abs=1e-6)
    with mlp_coordinates(model, [0], 3, edits={0: {0: (0., None)}}):
        wrong_position_score = score_encoded_suffix(model, prompt, suffix, "cpu")
    assert abs(wrong_position_score - score) > 1e-6


def test_failed_forward_marks_manifest_failed_and_removes_hooks(tmp_path, loaded, monkeypatch):
    model_path, bundle = loaded
    from llm_bias.financial_soundness import pipeline
    path = tmp_path / "prompts.json"
    write_json(path, small_data())
    def broken_score(*args, **kwargs):
        raise RuntimeError("injected scoring failure")
    monkeypatch.setattr(pipeline, "_score", broken_score)
    with pytest.raises(RuntimeError, match="injected"):
        run_localization(path, model_path, "failed", artifact_root=tmp_path / "runs", layers=[0], loaded=bundle)
    manifests = list((tmp_path / "runs").rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed"
    assert manifest["stages"]["forward"]["status"] == "failed"
    assert not bundle[0].layers[0].mlp.down_proj._forward_pre_hooks


def test_cli_is_stage_specific(tmp_path):
    out = tmp_path / "p.json"
    assert main(["prepare-prompts", "--output", str(out), "--companies", "Company A", "Company B"]) == 0
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-causal-validation", "--prompts", str(out)])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-localization", "--formal"])
