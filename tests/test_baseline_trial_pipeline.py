import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llm_bias.baseline_trial import pipeline


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def decode(self, token_ids, **_kwargs):
        if isinstance(token_ids, int):
            return f"t{token_ids}"
        return " ".join(f"t{token_id}" for token_id in token_ids)


class _FakeModel:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.n_layers = 2
        self.layers = [0, 1]
        self.device = torch.device("cpu")
        self._lm_head = SimpleNamespace(weight=torch.zeros((8, 4)))

        def _final_norm(value):
            return value

        self._final_norm = _final_norm
        self.decoder_calls = 0

    def unembed(self, residual):
        return torch.zeros(*residual.shape[:-1], 8)

    def _decoder(self, input_ids, attention_mask, use_cache):
        self.decoder_calls += 1
        batch, length = input_ids.shape
        return SimpleNamespace(
            last_hidden_state=torch.zeros(batch, length, 4, device=self.device)
        )


def _fake_record(model, layers, at):
    class _Recorder:
        def __init__(self):
            self.activations = {
                layer: torch.zeros(1, 16, 4) for layer in at
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    return _Recorder()


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    model = _FakeModel()

    def fake_load_lens_model(model_name, **kwargs):
        return SimpleNamespace(_hf_model=object()), model.tokenizer, "cpu"

    monkeypatch.setattr(pipeline, "load_lens_model", fake_load_lens_model)
    monkeypatch.setattr(pipeline, "WrappedModel", lambda *_args: model)
    monkeypatch.setattr(
        pipeline,
        "ActivationRecorder",
        lambda _layers, at: _fake_record(model, _layers, at),
    )

    lens = SimpleNamespace(
        d_model=4,
        source_layers=[0],
        jacobians={0: torch.eye(4)},
        transport_shapes=[],
        transport=lambda residual, layer: lens.transport_shapes.append(tuple(residual.shape))
        or residual,
    )
    loaded_lens = SimpleNamespace(
        lens=lens,
        path=tmp_path / "lens.pt",
        metadata={"provenance": {"workflow": "test"}},
    )

    def fake_load_validated_lens(**kwargs):
        return loaded_lens

    monkeypatch.setattr(pipeline, "load_validated_lens", fake_load_validated_lens)
    (tmp_path / "lens.pt").write_bytes(b"fake lens bytes")
    return model, loaded_lens


def _write_trial_csv(tmp_path):
    input_path = tmp_path / "trial_plan_prompts.csv"
    input_path.write_text(
        "Date,ticker,name,sector,marketcap,"
        "prompt_with_context_attribute_0,prompt_with_context_strategy_0\n"
        "2026-01-01,AAA,Alpha Corp,Energy,1e10,ask A,ask S\n"
        "2026-01-01,BBB,Beta Corp,Finance,2e10,ask B,ask T\n",
        encoding="utf-8",
    )
    return input_path


def _write_forward_artifact(run_root: Path):
    forward_dir = run_root / "forward"
    forward_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "schema_version": 1,
            "artifact_type": "generated_outputs",
            "record_id": "record_" + "a" * 24,
            "date": "2026-01-01",
            "ticker": "AAA",
            "prompt_column": "prompt_with_context_attribute_0",
            "index": "attribute_0",
            "context": "with",
            "condition": None,
            "prompt": "ask A",
            "prompt_token_ids": [1, 2, 3],
            "input_span": [0, 3],
            "generated_token_ids": [4, 5],
            "generated_text": "buy now",
            "generation_config": {"strategy": "greedy"},
            "finish_reason": "eos_token",
        },
        {
            "schema_version": 1,
            "artifact_type": "generated_outputs",
            "record_id": "record_" + "b" * 24,
            "date": "2026-01-01",
            "ticker": "BBB",
            "prompt_column": "prompt_with_context_strategy_0",
            "index": "strategy_0",
            "context": "with",
            "condition": None,
            "prompt": "ask S",
            "prompt_token_ids": [1, 2],
            "input_span": [0, 2],
            "generated_token_ids": [6],
            "generated_text": "sell",
            "generation_config": {"strategy": "greedy"},
            "finish_reason": "eos_token",
        },
    ]
    forward_path = forward_dir / "generated_outputs.jsonl"
    with forward_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "artifact_type": "generated_outputs",
        "model": "fake-model",
        "records_written": len(rows),
    }
    (forward_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return forward_path


def test_lens_forward_writes_compact_per_layer_readout(fake_env, tmp_path):
    model, loaded_lens = fake_env
    run_root = tmp_path / "runs" / "baseline-trial"
    run_root.mkdir(parents=True)
    forward_path = _write_forward_artifact(run_root)

    config = pipeline.BaselineRunConfig(
        input=forward_path.parent.parent / "input.csv",
        model="fake-model",
        lens=loaded_lens.path,
        dataset="baseline-trial",
        run_id="baseline-trial",
        artifact_root=tmp_path / "artifacts",
        run_root=run_root,
        stages=("lens-forward",),
        max_seq_len=1024,
        batch_size=8,
        top_k=3,
        max_rows=None,
        prompt_columns=None,
        generate_full=True,
        sample_per_condition=32,
        max_new_tokens=256,
        backward_input_top_k=None,
        backward_output_token_top_k=None,
        forward_artifact=forward_path,
        validate_seed=0,
    )

    output = pipeline._lens_forward(config, forward_path, model, loaded_lens.lens, loaded_lens.path)
    assert output == run_root / "lens-forward" / "lens_readout.jsonl"

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert model.decoder_calls == 2
    # Transport must be called once per (record, source layer) with a batched
    # [n_generated_positions, d_model] tensor, never one vector at a time.
    assert loaded_lens.lens.transport_shapes == [(2, 4), (1, 4)]
    first, second = records
    assert first["artifact_type"] == pipeline.LENS_FORWARD_ARTIFACT_TYPE
    assert first["record_id"] == "record_" + "a" * 24
    assert first["ticker"] == "AAA"
    assert first["parent_forward_sha256"] == pipeline.sha256_file(forward_path)
    # One readout per generated position: prompt(3)+gen(2)=5, gen(1): positions at 3,4 and 2.
    assert [position["position"] for position in first["positions"]] == [0, 1]
    assert [position["position"] for position in second["positions"]] == [0]
    for record in records:
        for position in record["positions"]:
            assert [layer["layer"] for layer in position["layers"]] == [0, 1]
            for layer in position["layers"]:
                assert len(layer["top_tokens"]) == 3
                assert layer["is_output"] is (layer["layer"] == 1)
                assert 0.0 <= layer["entropy_nats"] <= 3.0
                assert layer["effective_temperature"] > 0
    # All-zero fake logits: softmax is uniform over 8 tokens, so probabilities,
    # entropy and effective temperature are exact.
    layer0 = first["positions"][0]["layers"][0]
    assert layer0["top_tokens"][0]["probability"] == pytest.approx(0.125)
    assert layer0["entropy_nats"] == pytest.approx(math.log(8.0))
    assert layer0["normalized_entropy"] == pytest.approx(1.0)
    assert layer0["effective_inverse_temperature"] == pytest.approx(0.0)
    assert layer0["effective_temperature"] == pytest.approx(1e12)

    metadata = json.loads(
        (run_root / "lens-forward" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["lens_positions_read"] == 3
    assert metadata["records_written"] == 2
    assert metadata["backpropagation"] is False
    assert metadata["layers"] == [0, 1]
    assert metadata["parent_forward_sha256"] == first["parent_forward_sha256"]


def test_lens_forward_rejects_model_mismatch(fake_env, tmp_path):
    _model, loaded_lens = fake_env
    run_root = tmp_path / "runs" / "baseline-trial"
    run_root.mkdir(parents=True)
    forward_path = _write_forward_artifact(run_root)
    metadata_path = run_root / "forward" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"] = "other-model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    config = pipeline.BaselineRunConfig(
        input=forward_path,
        model="fake-model",
        lens=loaded_lens.path,
        dataset="baseline-trial",
        run_id="baseline-trial",
        artifact_root=tmp_path / "artifacts",
        run_root=run_root,
        stages=("lens-forward",),
        max_seq_len=1024,
        batch_size=8,
        top_k=3,
        max_rows=None,
        prompt_columns=None,
        generate_full=True,
        sample_per_condition=32,
        max_new_tokens=256,
        backward_input_top_k=None,
        backward_output_token_top_k=None,
        forward_artifact=forward_path,
        validate_seed=0,
    )
    with pytest.raises(ValueError, match="model does not match"):
        pipeline._lens_forward(
            config, forward_path, _model, loaded_lens.lens, loaded_lens.path
        )


def test_stage_gating_requires_full_forward(fake_env, tmp_path):
    _model, loaded_lens = fake_env
    input_path = _write_trial_csv(tmp_path)
    with pytest.raises(ValueError, match="full forward artifact"):
        pipeline.BaselineRunConfig.resolve(
            input_path=input_path,
            model="fake-model",
            lens=loaded_lens.path,
            dataset=None,
            run_id="run",
            artifact_root=tmp_path / "artifacts",
            stages=("forward", "lens-forward"),
            max_seq_len=1024,
            batch_size=8,
            top_k=3,
            max_rows=None,
            prompt_columns=None,
            generate_full=False,
            sample_per_condition=32,
            max_new_tokens=256,
            backward_input_top_k=None,
            backward_output_token_top_k=None,
            forward_artifact=None,
            validate_seed=0,
        )
    with pytest.raises(ValueError, match="backward requires lens-forward"):
        pipeline.BaselineRunConfig.resolve(
            input_path=input_path,
            model="fake-model",
            lens=loaded_lens.path,
            dataset=None,
            run_id="run",
            artifact_root=tmp_path / "artifacts",
            stages=("forward", "backward"),
            max_seq_len=1024,
            batch_size=8,
            top_k=3,
            max_rows=None,
            prompt_columns=None,
            generate_full=True,
            sample_per_condition=32,
            max_new_tokens=256,
            backward_input_top_k=None,
            backward_output_token_top_k=None,
            forward_artifact=None,
            validate_seed=0,
        )
    with pytest.raises(ValueError, match="validate requires backward"):
        pipeline.BaselineRunConfig.resolve(
            input_path=input_path,
            model="fake-model",
            lens=loaded_lens.path,
            dataset=None,
            run_id="run",
            artifact_root=tmp_path / "artifacts",
            stages=("forward", "lens-forward", "validate"),
            max_seq_len=1024,
            batch_size=8,
            top_k=3,
            max_rows=None,
            prompt_columns=None,
            generate_full=True,
            sample_per_condition=32,
            max_new_tokens=256,
            backward_input_top_k=None,
            backward_output_token_top_k=None,
            forward_artifact=None,
            validate_seed=0,
        )


def test_lens_forward_rejects_out_of_range_layers(fake_env, tmp_path):
    model, loaded_lens = fake_env
    run_root = tmp_path / "runs" / "baseline-trial"
    run_root.mkdir(parents=True)
    forward_path = _write_forward_artifact(run_root)
    config = pipeline.BaselineRunConfig(
        input=forward_path,
        model="fake-model",
        lens=loaded_lens.path,
        dataset="baseline-trial",
        run_id="baseline-trial",
        artifact_root=tmp_path / "artifacts",
        run_root=run_root,
        stages=("lens-forward",),
        max_seq_len=1024,
        batch_size=8,
        top_k=3,
        max_rows=None,
        prompt_columns=None,
        generate_full=True,
        sample_per_condition=32,
        max_new_tokens=256,
        backward_input_top_k=None,
        backward_output_token_top_k=None,
        forward_artifact=forward_path,
        validate_seed=0,
        lens_forward_layers=[0, 32],
    )
    with pytest.raises(ValueError, match="layers out of range 0..1"):
        pipeline._lens_forward(config, forward_path, model, loaded_lens.lens, loaded_lens.path)


def test_forward_record_ids_disambiguates_duplicate_record_ids():
    rows = [
        {"record_id": "record_x" * 8, "prompt_token_ids": [1], "generated_token_ids": [2]},
        {"record_id": "record_x" * 8, "prompt_token_ids": [3], "generated_token_ids": [4]},
    ]
    ids = pipeline._forward_record_ids(rows)
    assert len(ids) == 2 and ids[0] != ids[1]


def test_sharded_backward_stage_is_rejected(fake_env, tmp_path):
    _model, loaded_lens = fake_env
    input_path = _write_trial_csv(tmp_path)
    run_root = tmp_path / "artifacts" / "fake-model" / "trial-plan" / "runs" / "run"
    run_root.mkdir(parents=True)
    forward_path = _write_forward_artifact(run_root)
    with pytest.raises(ValueError, match="backward stage does not support sharded loading"):
        pipeline.run_baseline_trial(
            input_path=input_path,
            model="fake-model",
            lens=loaded_lens.path,
            run_id="run",
            artifact_root=tmp_path / "artifacts",
            dataset="trial-plan",
            stages=("lens-forward", "backward"),
            forward_artifact=forward_path,
            device_map="qwen27b_two_gpu",
        )


def test_run_marks_failed_manifest_on_stage_error(fake_env, tmp_path, monkeypatch):
    model, loaded_lens = fake_env
    input_path = _write_trial_csv(tmp_path)
    run_root = tmp_path / "artifacts" / "fake-model" / "trial-plan" / "runs" / "run"
    run_root.mkdir(parents=True)
    forward_path = _write_forward_artifact(run_root)

    def explode(*_args, **_kwargs):
        raise RuntimeError("lens boom")

    monkeypatch.setattr(pipeline, "_lens_forward", explode)

    with pytest.raises(RuntimeError, match="lens boom"):
        pipeline.run_baseline_trial(
            input_path=input_path,
            model="fake-model",
            lens=loaded_lens.path,
            run_id="run",
            artifact_root=tmp_path / "artifacts",
            dataset="trial-plan",
            stages=("lens-forward",),
            forward_artifact=forward_path,
        )

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["stages"]["lens_forward"]["status"] == "failed"
    assert manifest["error"] == "baseline-trial run run failed"


def test_run_registers_outputs_and_completes(fake_env, tmp_path):
    model, loaded_lens = fake_env
    input_path = _write_trial_csv(tmp_path)
    run_root = tmp_path / "artifacts" / "fake-model" / "trial-plan" / "runs" / "run"
    run_root.mkdir(parents=True)
    forward_path = _write_forward_artifact(run_root)

    pipeline.run_baseline_trial(
        input_path=input_path,
        model="fake-model",
        lens=loaded_lens.path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        dataset="trial-plan",
        stages=("lens-forward",),
        forward_artifact=forward_path,
    )

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["stages"]["lens_forward"]["status"] == "complete"
    assert manifest["stages"]["lens_forward"]["record_count"] == 2
    lens_refs = [
        ref for ref in manifest["output_refs"] if ref["path"] == "lens-forward/lens_readout.jsonl"
    ]
    assert len(lens_refs) == 1
    assert lens_refs[0]["sha256"] == pipeline.sha256_file(
        run_root / "lens-forward" / "lens_readout.jsonl"
    )
    assert lens_refs[0]["record_count"] == 2
    # Compactness contract: no raw payload fields anywhere in the run tree.
    for path in run_root.rglob("*"):
        if path.is_file() and path.suffix in {".jsonl", ".json"}:
            text = path.read_text(encoding="utf-8")
            assert "activations" not in text
            assert "hidden_states" not in text
    # Forward identity is bound to the lens-forward stage for provenance.
    stage = manifest["stages"]["lens_forward"]
    assert stage["forward_record_ids"] == [
        pipeline.stable_record_id(0, "record_" + "a" * 24),
        pipeline.stable_record_id(1, "record_" + "b" * 24),
    ]
    assert manifest["stages"]["prepare"]["lens_sha256"] == pipeline.sha256_file(
        loaded_lens.path
    )
