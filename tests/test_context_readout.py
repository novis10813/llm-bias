"""Regression tests for Direction C V1 context readout."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.artifacts.io import write_jsonl
from llm_bias.core.artifact_paths import sha256_file
from llm_bias.jspace_intervention import context_readout as context


class _Encoded:
    def __init__(self, ids, offsets=None):
        self.input_ids = ids
        if offsets is not None:
            self.offset_mapping = offsets
            self.special_tokens_mask = [0] * len(ids)


class _Tokenizer:
    chat_template = "fake"
    all_special_tokens: list[str] = []
    pad_token_id = 0

    def __init__(self):
        names = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu")
        self.words = {f" {name}": 10 + i for i, name in enumerate(names)}
        self.words.update({" omega": 30, " sigma": 31})

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **_kwargs):
        if not return_offsets_mapping and text in self.words:
            return _Encoded([self.words[text]])
        ids = [ord(char) % 200 for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        return _Encoded(ids, offsets if return_offsets_mapping else None)

    def apply_chat_template(self, messages, **_kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def decode(self, ids, **_kwargs):
        reverse = {value: key for key, value in self.words.items()}
        return reverse.get(int(ids[0]), f" token{int(ids[0])}")


class _Norm(torch.nn.Module):
    variance_epsilon = 1e-6

    def __init__(self, width=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))


class _Model:
    n_layers = 31

    def __init__(self):
        self._final_norm = _Norm()
        self._lm_head = torch.nn.Linear(2, 20, bias=False)
        with torch.no_grad():
            self._lm_head.weight.zero_()
            self._lm_head.weight[:, 0] = torch.arange(20, dtype=torch.float32)


class _Lens:
    source_layers = (6, 16)

    def transport(self, residual, layer):
        assert layer in self.source_layers
        return residual


class _PipelineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 31
        self.input_device = torch.device("cpu")
        self.embedding = torch.nn.Embedding(256, 2)
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(31)])
        self._final_norm = torch.nn.LayerNorm(2)
        self._lm_head = torch.nn.Linear(2, 40, bias=False)
        torch.manual_seed(4)
        torch.nn.init.normal_(self.embedding.weight)
        torch.nn.init.normal_(self._lm_head.weight)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidates = {
        "artifact_type": "frozen_candidate_suggestions", "schema_version": 2,
        "candidates": [
            {"token_id": 10 + i, "token": f" {name}", "side": "positive" if i < 6 else "negative"}
            for i, name in enumerate(("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"))
        ],
    }
    tfidf = {
        "artifact_type": "jspace_intervention_config", "schema_version": 1,
        "model": "fake",
        "source": {"score_type": "contrastive_tfidf", "tokens": [{"token_id": 30, "token": " omega"}]},
        "target": {"score_type": "contrastive_tfidf", "tokens": [{"token_id": 31, "token": " sigma"}]},
    }
    candidate_path, tfidf_path = tmp_path / "candidates.json", tmp_path / "tfidf.json"
    candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
    tfidf_path.write_text(json.dumps(tfidf), encoding="utf-8")
    return candidate_path, tfidf_path


def _pair_file(tmp_path: Path) -> Path:
    from llm_bias.jspace_intervention.valence import build_valence_pair
    row = {
        "artifact_type": "valence_pairs", "schema_version": 2,
        **build_valence_pair({
            "condition": "attribute", "ticker": "T1", "name": "Name T1", "sector": "Technology",
            "prompt": "unused", "trial_key": "t1", "trial_index": 0, "set_index": 0,
            "evidence": [
                {"side": "buy", "kind": "qual", "text": "good outlook"},
                {"side": "buy", "kind": "quant", "text": "growth five percent"},
                {"side": "sell", "kind": "qual", "text": "bad outlook"},
                {"side": "sell", "kind": "quant", "text": "loss five percent"},
            ],
        }),
    }
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [row], overwrite=False)
    return path


def test_config_freeze_duplicate_handling_and_atomic_output(tmp_path, monkeypatch):
    tokenizer = _Tokenizer()
    monkeypatch.setattr(context, "load_tokenizer", lambda _model: tokenizer)
    candidates, tfidf = _sources(tmp_path)
    output = tmp_path / "config.json"
    context.prepare_context_readout_config(candidates, tfidf, "fake", output)
    payload = json.loads(output.read_text())
    assert payload["artifact_type"] == context.CONFIG_ARTIFACT_TYPE
    assert payload["primary"] == {"layer": 16, "span": "instruction_context"}
    assert payload["top_k"] == 30
    assert payload["sources"]["frozen_candidates"]["sha256"] == sha256_file(candidates)
    assert len(payload["families"]["financial_evidence"]["tokens"]) == 12
    tfidf_payload = json.loads(tfidf.read_text())
    tfidf_payload["model"] = "other-model"
    tfidf.write_text(json.dumps(tfidf_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model does not match"):
        context.prepare_context_readout_config(
            candidates, tfidf, "fake", tmp_path / "model-mismatch.json"
        )
    tfidf_payload["model"] = "fake"
    tfidf_payload["source"]["tokens"] = [{"token_id": 10, "token": " alpha"}]
    tfidf.write_text(json.dumps(tfidf_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate token id"):
        context.prepare_context_readout_config(candidates, tfidf, "fake", tmp_path / "duplicate.json")
    assert not (tmp_path / "duplicate.json").exists()


def test_mean_softmax_before_top_k_and_exact_span_exclusion():
    model, tokenizer, lens = _Model(), _Tokenizer(), _Lens()
    residual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    vector = context.context_readout_vector(model, lens, residual, 16)
    expected = torch.softmax(context.fp32_next_token_logits(model, residual), dim=-1).mean(dim=0)
    assert torch.allclose(vector, expected)
    spans = context.resolve_prompt_spans(tokenizer, "head|aaaa|bbbb|instruction", {"qual": [5, 9], "quant": [10, 14]})
    assert spans["instruction_context"][1] == len("head|aaaa|bbbb|instruction") - 1


def test_family_mass_rank_and_compact_serialization():
    tokenizer = _Tokenizer()
    families = {name: {"tokens": [{"token_id": 10, "token": " alpha"}]} for name in context.FAMILY_NAMES}
    compact = context._compact_vector(torch.softmax(torch.arange(20, dtype=torch.float32), dim=0), tokenizer=tokenizer, top_k=3, families=families)
    assert compact["families"]["financial_evidence"]["mass"] > 0
    assert compact["families"]["financial_evidence"]["tokens"][0]["rank"] >= 1
    assert "full_vector" not in compact and "residual" not in compact


def test_config_tamper_rejected(tmp_path, monkeypatch):
    tokenizer = _Tokenizer()
    monkeypatch.setattr(context, "load_tokenizer", lambda _model: tokenizer)
    candidates, tfidf = _sources(tmp_path)
    output = tmp_path / "config.json"
    context.prepare_context_readout_config(candidates, tfidf, "fake", output)
    tfidf.write_text(tfidf.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        context.validate_context_config(json.loads(output.read_text()), tokenizer=tokenizer, model="fake")


def test_pipeline_lifecycle_and_no_raw_or_full_vectors(tmp_path, monkeypatch):
    tokenizer, model, lens = _Tokenizer(), _PipelineModel(), _Lens()
    candidates, tfidf = _sources(tmp_path)
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(context, "load_tokenizer", lambda _model: tokenizer)
    context.prepare_context_readout_config(candidates, tfidf, "fake", config_path)
    lens_path = tmp_path / "lens.pt"
    lens_path.write_bytes(b"lens")
    monkeypatch.setattr(context, "load_model", lambda _name: (model, tokenizer, "cpu"))
    monkeypatch.setattr(context, "load_validated_lens", lambda **_kwargs: SimpleNamespace(lens=lens, path=lens_path, source="fake", metadata={"revision": "test"}))
    run_dir = context.run_context_readout_pipeline(_pair_file(tmp_path), config_path, "fake", "context", artifact_root=tmp_path / "artifacts")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert all(value["status"] == "complete" for value in manifest["stages"].values())
    records = [json.loads(line) for line in (run_dir / "forward" / "l16_context_readout.jsonl").read_text().splitlines()]
    assert len(records) == 8
    assert all("full_vector" not in row and "residual" not in row for row in records)
    summary = json.loads((run_dir / "analyze" / "summary.json").read_text())
    assert summary["formal_success_gate"] is False
    assert summary["config_sha256"] == sha256_file(config_path)


def test_cli_dispatch_for_new_commands(monkeypatch, tmp_path):
    from llm_bias.jspace_intervention import cli
    calls = []
    monkeypatch.setattr("llm_bias.jspace_intervention.context_readout.prepare_context_readout_config", lambda **kwargs: calls.append(("prepare", kwargs)) or tmp_path / "config.json")
    monkeypatch.setattr("llm_bias.jspace_intervention.context_readout.run_context_readout_pipeline", lambda **kwargs: calls.append(("run", kwargs)) or tmp_path / "run")
    monkeypatch.setattr("sys.argv", ["jspace-intervention", "prepare-l16-context-readout-config", "--frozen-candidates", "c", "--tfidf-config", "t", "--model", "m", "--output", "o"])
    cli.main()
    monkeypatch.setattr("sys.argv", ["jspace-intervention", "run-l16-context-readout", "--pairs", "p", "--config", "c", "--model", "m", "--run-id", "r"])
    cli.main()
    assert [kind for kind, _ in calls] == ["prepare", "run"]
    assert calls[1][1]["dataset"] == "l16-context-readout"
