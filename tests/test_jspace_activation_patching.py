"""Regression tests for deterministic activation patching primitives."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.continuation_scoring import score_single_token_margin_fp32
from llm_bias.jspace_intervention import activation_patching
from llm_bias.jspace_intervention.activation_patching import (
    _build_position_mapping,
    _nearest_position_mapping,
    cache_source_residuals,
    evaluate_activation_patching_confirmation,
    evaluate_activation_patching_confirmation_artifacts,
    resolve_prompt_spans,
    run_activation_patching_pipeline,
    run_activation_patching_record,
    run_patched_margin,
)

BUY_ID = 240
SELL_ID = 241


class _CharTokenizer:
    """Character tokenizer with one-token Buy/Sell continuations."""

    chat_template = "fake"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ) -> SimpleNamespace:
        del add_special_tokens
        text = str(text)
        suffix = None
        for candidate, token_id in (
            (" Buy", BUY_ID),
            (" Sell", SELL_ID),
            ("buy", BUY_ID),
            ("sell", SELL_ID),
        ):
            if text.endswith(candidate):
                suffix = (candidate, token_id)
                break
        if suffix is None:
            values = [ord(char) for char in text]
            offsets = [(index, index + 1) for index in range(len(text))]
        else:
            candidate, token_id = suffix
            prefix = text[: -len(candidate)]
            values = [ord(char) for char in prefix] + [token_id]
            offsets = [(index, index + 1) for index in range(len(prefix))]
            offsets.append((len(prefix), len(text)))
        if return_offsets_mapping:
            return SimpleNamespace(
                input_ids=values,
                offset_mapping=offsets,
                special_tokens_mask=[0] * len(values),
            )
        return SimpleNamespace(input_ids=values)

    def apply_chat_template(self, messages, **_kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"


class _SumLayer(torch.nn.Module):
    """Expose post-hook states and make every position read the sequence sum."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.seen = hidden.detach().clone()
        return hidden + hidden.sum(dim=1, keepdim=True)


class _FakeRMSNorm(torch.nn.Module):
    variance_epsilon = 1e-6

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(d_model))


class _PatchingModel(torch.nn.Module):
    """Small CPU-only decoder whose final margin follows the residual sum."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 1)
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), _SumLayer()])
        self.n_layers = 2
        self._final_norm = _FakeRMSNorm(1)
        self._lm_head = torch.nn.Linear(1, 256, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            for token in (ord("a"), ord("b")):
                self.embedding.weight[token, 0] = 1.0
            for token in (ord("y"), ord("z")):
                self.embedding.weight[token, 0] = -1.0
            self._lm_head.weight.zero_()
            self._lm_head.weight[BUY_ID, 0] = 1.0
            self._lm_head.weight[SELL_ID, 0] = -1.0

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _evidence_spans(prompt: str, qualitative: str, quantitative: str) -> dict[str, list[int]]:
    qual_start = prompt.index(qualitative)
    quant_start = prompt.index(quantitative)
    return {
        "qual": [qual_start, qual_start + len(qualitative)],
        "quant": [quant_start, quant_start + len(quantitative)],
    }


def _paired_prompts() -> tuple[str, str, dict[str, list[int]], dict[str, list[int]]]:
    source = "H|aaaa|bbbb|I"
    target = "H|zzzz|yyyy|I"
    return (
        source,
        target,
        _evidence_spans(source, "aaaa", "bbbb"),
        _evidence_spans(target, "zzzz", "yyyy"),
    )


def test_resolve_prompt_spans_returns_semantic_token_ranges() -> None:
    tokenizer = _CharTokenizer()
    prompt = "HDR[qual=abcd][quant=12]INSTR"
    spans = _evidence_spans(prompt, "abcd", "12")

    resolved = resolve_prompt_spans(tokenizer, prompt, spans)
    qual_start, qual_end = spans["qual"]
    quant_start, quant_end = spans["quant"]

    assert resolved == {
        "header": (0, qual_start),
        "evidence_qual": (qual_start, qual_end),
        "evidence_quant": (quant_start, quant_end),
        "all_evidence": (qual_start, quant_end),
        "instruction_context": (quant_end, len(prompt) - 1),
        "instruction": (quant_end, len(prompt)),
        "all_positions": (0, len(prompt)),
        "final_position": (len(prompt) - 1, len(prompt)),
    }


def test_unequal_span_mapping_patches_every_requested_target_position() -> None:
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    source_residuals = {
        0: torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
    }
    mapping = _nearest_position_mapping((0, 4), (1, 3))

    assert set(mapping) == {1, 2}
    assert set(mapping.values()) == {0, 3}
    run_patched_margin(
        model=model,
        tokenizer=tokenizer,
        target_prompt="xxxx",
        source_residuals=source_residuals,
        position_mapping=mapping,
        layers=[0],
        device="cpu",
    )

    assert model.layers[1].seen is not None
    assert torch.equal(model.layers[1].seen[0, :, 0], torch.tensor([0.0, 1.0, 4.0, 0.0]))
    assert not model.layers[0]._forward_hooks


def test_instruction_context_excludes_final_position() -> None:
    tokenizer = _CharTokenizer()
    source, target, source_chars, target_chars = _paired_prompts()
    source_spans = resolve_prompt_spans(tokenizer, source, source_chars)
    target_spans = resolve_prompt_spans(tokenizer, target, target_chars)

    mapping = _build_position_mapping(
        source_spans,
        target_spans,
        "instruction_context",
        len(tokenizer(source).input_ids),
        len(tokenizer(target).input_ids),
    )

    assert mapping
    assert max(mapping) == len(target) - 2
    assert len(target) - 1 not in mapping


def test_run_patched_margin_changes_margin_and_cleans_hooks() -> None:
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    source, target, source_chars, target_chars = _paired_prompts()
    source_spans = resolve_prompt_spans(tokenizer, source, source_chars)
    target_spans = resolve_prompt_spans(tokenizer, target, target_chars)
    source_residuals = cache_source_residuals(
        model, tokenizer, source, layers=[0], device="cpu"
    )
    mapping = _build_position_mapping(
        source_spans,
        target_spans,
        "all_evidence",
        len(tokenizer(source).input_ids),
        len(tokenizer(target).input_ids),
    )

    clean = score_single_token_margin_fp32(
        model, tokenizer, target, " Buy", " Sell", device="cpu"
    )
    patched = run_patched_margin(
        model=model,
        tokenizer=tokenizer,
        target_prompt=target,
        source_residuals=source_residuals,
        position_mapping=mapping,
        layers=[0],
        device="cpu",
    )

    assert clean.value < 0
    assert patched.value > 0
    assert patched.value != pytest.approx(clean.value)
    assert not model.layers[0]._forward_hooks
    assert not model.layers[1]._forward_hooks


def test_run_activation_patching_record_is_compact_and_reports_flip() -> None:
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    source, target, source_chars, target_chars = _paired_prompts()

    record = run_activation_patching_record(
        model=model,
        tokenizer=tokenizer,
        source_prompt=source,
        target_prompt=target,
        source_evidence_char_spans=source_chars,
        target_evidence_char_spans=target_chars,
        layers=[0],
        span_condition="all_evidence",
        device="cpu",
    )

    assert set(record) == {
        "layers",
        "span_condition",
        "positions_patched",
        "position_count",
        "source_seq_len",
        "target_seq_len",
        "source_clean_margin",
        "target_clean_margin",
        "patched_margin",
        "delta_margin",
        "flip",
        "score",
    }
    assert record["layers"] == [0]
    assert record["span_condition"] == "all_evidence"
    assert record["position_count"] == len(record["positions_patched"]) == 8
    assert record["source_clean_margin"] > 0
    assert record["target_clean_margin"] < 0
    assert record["patched_margin"] > 0
    assert record["delta_margin"] > 0
    assert record["flip"] is True
    assert set(record["score"]) == {"positive", "negative"}
    assert set(record["score"]["positive"]) == {
        "candidate",
        "token_ids",
        "log_probability",
        "token_count",
    }
    assert not model.layers[0]._forward_hooks
    assert not model.layers[1]._forward_hooks


def _pair_record() -> dict:
    source, target, source_chars, target_chars = _paired_prompts()
    return {
        "schema_version": 2,
        "artifact_type": "valence_pairs",
        "record_id": "record_pair",
        "source_trial_key": "trial-1",
        "ticker": "T1",
        "name": "Test Company",
        "sector": "Technology",
        "evidence_item_hashes": {"buy_qual": "a" * 64},
        "prompts": {"positive": source, "negative": target},
        "evidence_char_spans": {
            "positive": source_chars,
            "negative": target_chars,
        },
    }


def test_run_activation_patching_pipeline_writes_compact_artifacts(
    monkeypatch, tmp_path
) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(json.dumps(_pair_record()) + "\n", encoding="utf-8")
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    model.input_device = torch.device("cpu")
    monkeypatch.setattr(activation_patching, "load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(
        activation_patching,
        "load_model",
        lambda _name: (model, tokenizer, torch.device("cpu")),
    )

    run_dir = run_activation_patching_pipeline(
        pairs_path=pairs_path,
        model_name="fake-model",
        run_id="patching-smoke",
        phase="phase2",
        layer_conditions={"single_L0": [0]},
        span_conditions=["all_evidence", "header"],
        directions=["positive_to_negative"],
        decision_prefix="",
        positive_candidate=" Buy",
        negative_candidate=" Sell",
        artifact_root=tmp_path / "artifacts",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline = [
        json.loads(line)
        for line in (run_dir / "forward" / "phase0_baseline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records = [
        json.loads(line)
        for line in (run_dir / "forward" / "phase2_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    summary = json.loads((run_dir / "analyze" / "summary.json").read_text())

    assert manifest["status"] == "complete"
    assert baseline[0]["clean_decision_match"] is True
    assert len(records) == 2
    assert {row["span_condition"] for row in records} == {
        "all_evidence",
        "header",
    }
    assert next(row for row in records if row["span_condition"] == "all_evidence")[
        "flip"
    ] is True
    assert summary["eligible_pair_count"] == 1
    assert summary["patch_record_count"] == 2
    serialized = json.dumps(records)
    assert "source_residual" not in serialized
    assert "activation" not in serialized


def test_run_activation_patching_cli_dispatch(monkeypatch, tmp_path) -> None:
    from llm_bias.jspace_intervention import cli

    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(
        activation_patching, "run_activation_patching_pipeline", fake_pipeline
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jspace-intervention",
            "run-activation-patching",
            "--pairs",
            "pairs.jsonl",
            "--model",
            "fake-model",
            "--run-id",
            "patching",
            "--phase",
            "phase3",
            "--layer-condition",
            "local=14-16,20",
            "--span-condition",
            "all_evidence",
            "--direction",
            "positive_to_negative",
        ],
    )

    cli.main()

    assert captured["phase"] == "phase3"
    assert captured["layer_conditions"] == {"local": (14, 15, 16, 20)}
    assert captured["span_conditions"] == ["all_evidence"]
    assert captured["directions"] == ["positive_to_negative"]
    assert captured["dataset"] == "jspace-causal-tracing"


def _confirmation_config() -> dict:
    return {
        "directions": ["positive_to_negative", "negative_to_positive"],
        "span_conditions": ["all_evidence", "instruction_context", "final_position"],
        "primary_diagonal": [
            {"layer_condition": "early", "span_condition": "all_evidence", "minimum_mean_transfer": 0.7},
            {"layer_condition": "middle", "span_condition": "instruction_context", "minimum_mean_transfer": 0.4},
            {"layer_condition": "late", "span_condition": "final_position", "minimum_mean_transfer": 0.7},
        ],
        "minimum_eligible_pairs": 6,
        "minimum_eligible_tickers": 3,
        "row_dominance_minimum": 0.2,
        "bootstrap_seed": 17,
        "bootstrap_samples": 200,
        "ci_level": 0.95,
    }


def _confirmation_records() -> tuple[list[dict], list[dict]]:
    baseline = []
    records = []
    ticker_values = {
        "A": {"early": (0.9, 0.8, 0.1), "middle": (0.1, 0.7, 0.1), "late": (0.1, 0.1, 0.9)},
        "B": {"early": (0.8, 0.7, 0.1), "middle": (0.1, 0.6, 0.1), "late": (0.1, 0.1, 0.8)},
        "C": {"early": (0.7, 0.6, 0.1), "middle": (0.1, 0.5, 0.1), "late": (0.1, 0.1, 0.7)},
        "D": {"early": (0.9, 0.8, 0.1), "middle": (0.1, 0.7, 0.1), "late": (0.1, 0.1, 0.9)},
        "E": {"early": (0.8, 0.7, 0.1), "middle": (0.1, 0.6, 0.1), "late": (0.1, 0.1, 0.8)},
        "F": {"early": (0.7, 0.6, 0.1), "middle": (0.1, 0.5, 0.1), "late": (0.1, 0.1, 0.7)},
    }
    spans = ["all_evidence", "instruction_context", "final_position"]
    for ticker, rows in ticker_values.items():
        for trial in range(2):
            pair_id = f"{ticker}-{trial}"
            baseline.append({"pair_record_id": pair_id, "ticker": ticker, "clean_decision_match": True})
            for layer, values in rows.items():
                for span, value in zip(spans, values, strict=True):
                    for direction in ("positive_to_negative", "negative_to_positive"):
                        records.append({
                            "pair_record_id": pair_id,
                            "ticker": ticker,
                            "layer_condition": layer,
                            "span_condition": span,
                            "patching_direction": direction,
                            "source_clean_margin": 2.0 if direction == "positive_to_negative" else -2.0,
                            "target_clean_margin": -2.0 if direction == "positive_to_negative" else 2.0,
                            "delta_margin": value * (4.0 if direction == "positive_to_negative" else -4.0),
                        })
    return records, baseline


def test_evaluate_activation_patching_confirmation_computes_matrix_and_gates() -> None:
    records, baseline = _confirmation_records()
    config = _confirmation_config()
    result = evaluate_activation_patching_confirmation(records, baseline, config, split="test")

    assert result["formal"] is True
    assert result["success"] is True
    assert result["eligible_pairs"] == 12
    assert result["eligible_tickers"] == 6
    matrix = {(row["layer_condition"], row["span_condition"]): row for row in result["matrix"]}
    assert matrix[("early", "all_evidence")]["equal_ticker_mean_transfer"] == pytest.approx(0.8)
    assert matrix[("middle", "instruction_context")]["equal_ticker_mean_transfer"] == pytest.approx(0.6)
    assert matrix[("late", "final_position")]["equal_ticker_mean_transfer"] == pytest.approx(0.8)
    assert [row["layer_condition"] for row in result["contrasts"]] == ["early", "middle", "late"]
    assert all(row["equal_ticker_mean_row_dominance"] > 0.2 for row in result["contrasts"])
    assert all(row["one_sided_exact_sign_flip_p"] == pytest.approx(1 / 64) for row in result["contrasts"])
    assert all(row["holm_adjusted_p"] == pytest.approx(3 / 64) for row in result["contrasts"])
    assert all(check["pass"] for check in result["gate_checks"])


def test_evaluate_activation_patching_confirmation_artifacts_is_atomic_and_cli_dispatch(
    monkeypatch, tmp_path
) -> None:
    records, baseline = _confirmation_records()
    records_path = tmp_path / "records.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "evaluation.json"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    baseline_path.write_text("".join(json.dumps(row) + "\n" for row in baseline), encoding="utf-8")
    config_path.write_text(json.dumps(_confirmation_config()), encoding="utf-8")

    result = evaluate_activation_patching_confirmation_artifacts(
        records_path, baseline_path, config_path, output_path, "calibration"
    )
    assert "formal" not in result
    assert result["test_authorized"] is True
    assert result["config_sha256"]
    assert result["parent_sha256"]
    assert result["baseline_sha256"]
    assert json.loads(output_path.read_text())["split"] == "calibration"
    with pytest.raises(FileExistsError):
        evaluate_activation_patching_confirmation_artifacts(
            records_path, baseline_path, config_path, output_path, "calibration"
        )

    from llm_bias.jspace_intervention import cli
    captured = {}
    def fake_evaluator(**kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(activation_patching, "evaluate_activation_patching_confirmation_artifacts", fake_evaluator)
    monkeypatch.setattr(sys, "argv", [
        "jspace-intervention", "analyze-activation-patching-confirmation",
        "--records", str(records_path), "--baseline", str(baseline_path),
        "--config", str(config_path), "--output", str(tmp_path / "cli.json"),
        "--split", "test",
    ])
    cli.main()
    assert captured["split"] == "test"
    assert captured["records_path"] == records_path


def test_evaluate_activation_patching_confirmation_rejects_incomplete_matrix() -> None:
    records, baseline = _confirmation_records()
    with pytest.raises(ValueError, match="confirmation matrix is incomplete"):
        evaluate_activation_patching_confirmation(
            records[:-1], baseline, _confirmation_config(), split="test"
        )


def test_run_activation_patching_record_rejects_invalid_span_condition() -> None:
    with pytest.raises(ValueError, match="unknown span_condition"):
        run_activation_patching_record(
            model=_PatchingModel(),
            tokenizer=_CharTokenizer(),
            source_prompt="source",
            target_prompt="target",
            source_evidence_char_spans={},
            target_evidence_char_spans={},
            layers=[0],
            span_condition="missing",
            device="cpu",
        )
