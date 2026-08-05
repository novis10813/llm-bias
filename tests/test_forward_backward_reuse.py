import json
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.prompt_analysis import generated_attribution


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def decode(self, token_ids, **_kwargs):
        return "tok-" + "-".join(str(int(token_id)) for token_id in token_ids)


def _forward_fixture(path):
    rows = [
        {
            "record_id": "pair-0",
            "prompt": "first",
            "prompt_column": "without",
            "prompt_token_ids": [10, 11, 12],
            "input_span": [1, 3],
            "generated_token_ids": [21, 22],
            "generated_text": "21 22",
        },
        {
            "record_id": "pair-1",
            "prompt": "second",
            "prompt_column": "with",
            "prompt_token_ids": [30, 31, 32],
            "input_span": [0, 2],
            "generated_token_ids": [41, 42],
            "generated_text": "41 42",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (path.parent / "metadata.json").write_text(
        json.dumps({"model": "fake", "artifact_sha256": sha256_file(path)}) + "\n",
        encoding="utf-8",
    )


def _patch_model(monkeypatch):
    tokenizer = _Tokenizer()
    wrapped = SimpleNamespace(device=torch.device("cpu"), tokenizer=tokenizer)
    monkeypatch.setattr(
        generated_attribution,
        "load_lens_model",
        lambda _name: (SimpleNamespace(_hf_model=object()), tokenizer, "cpu"),
    )
    monkeypatch.setattr(generated_attribution, "WrappedModel", lambda *_args: wrapped)
    monkeypatch.setattr(
        generated_attribution,
        "_attribute_generated_token",
        lambda **kwargs: {
            "token_id": kwargs["target_id"],
            "token": f"token-{kwargs['target_id']}",
            "logit": float(kwargs["target_id"]),
            "log_probability": -float(kwargs["target_id"]),
            "top_input_tokens": [
                {
                    "rank": 1,
                    "position": kwargs["input_span"][0],
                    "prompt_position": 0,
                    "token_id": 1,
                    "token": "input",
                    "attribution": 1.0,
                }
            ][: kwargs["input_top_k"] or 1],
        },
    )


def test_backward_reuses_forward_tokens_without_generation(tmp_path, monkeypatch):
    forward = tmp_path / "run" / "forward" / "generated_outputs.jsonl"
    _forward_fixture(forward)
    _patch_model(monkeypatch)
    output = generated_attribution.run_backward_attribution(
        forward_path=forward,
        model_name="fake",
        output_dir=tmp_path / "run",
        input_top_k=1,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    forward_rows = [json.loads(line) for line in forward.read_text().splitlines()]
    assert [row["record_id"] for row in rows] == [
        row["record_id"] for row in forward_rows
    ]
    assert all(row["record_id"] for row in rows)
    assert all(row["schema_version"] == 1 for row in rows)
    assert all(row["artifact_type"] == "generated_token_attribution" for row in rows)
    assert all(row["parent_forward_sha256"] for row in rows)
    assert rows[0]["generated_token_ids"] == [21, 22]
    assert [token["token_id"] for token in rows[0]["generated_tokens"]] == [21, 22]
    assert rows[0]["coverage"]["complete"] is True
    metadata = json.loads((tmp_path / "run" / "backward" / "metadata.json").read_text())
    assert metadata["parent_forward_sha256"] == rows[0]["parent_forward_sha256"]


def test_modified_forward_parent_hash_fails_closed(tmp_path, monkeypatch):
    forward = tmp_path / "run" / "forward" / "generated_outputs.jsonl"
    _forward_fixture(forward)
    _patch_model(monkeypatch)
    generated_attribution.run_backward_attribution(
        forward_artifact=forward,
        model_name="fake",
        output_dir=tmp_path / "run",
    )
    forward.write_text(forward.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="parent forward artifact hash mismatch"):
        generated_attribution.run_backward_attribution(
            forward_path=forward,
            model_name="fake",
            output_dir=tmp_path / "run",
        )


def test_input_top_k_rerun_preserves_forward_sequence(tmp_path, monkeypatch):
    forward = tmp_path / "run" / "forward" / "generated_outputs.jsonl"
    _forward_fixture(forward)
    _patch_model(monkeypatch)

    generated_attribution.run_backward_attribution(
        forward_path=forward,
        model_name="fake",
        output_dir=tmp_path / "run",
        input_top_k=1,
    )
    first = [
        json.loads(line)
        for line in (tmp_path / "run" / "backward" / "generated_token_attribution.jsonl")
        .read_text()
        .splitlines()
    ]
    generated_attribution.run_backward_attribution(
        forward_path=forward,
        model_name="fake",
        output_dir=tmp_path / "run",
        input_top_k=2,
    )
    second = [
        json.loads(line)
        for line in (tmp_path / "run" / "backward" / "generated_token_attribution.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["generated_token_ids"] for row in first] == [
        row["generated_token_ids"] for row in second
    ]
    assert [row["generated_text"] for row in first] == [row["generated_text"] for row in second]
    assert all(row["coverage"]["input_top_k"] == 2 for row in second)


def test_model_mismatch_fails_closed(tmp_path, monkeypatch):
    forward = tmp_path / "run" / "forward" / "generated_outputs.jsonl"
    _forward_fixture(forward)
    _patch_model(monkeypatch)

    with pytest.raises(ValueError, match="model does not match"):
        generated_attribution.run_backward_attribution(
            forward_path=forward,
            model_name="different-model",
            output_dir=tmp_path / "run",
        )


def test_return_pair_backward_adds_prediction_fields(tmp_path, monkeypatch):
    forward = tmp_path / "run" / "forward" / "generated_outputs.jsonl"
    forward.parent.mkdir(parents=True, exist_ok=True)
    forward.write_text(
        json.dumps(
            {
                "record_id": "record_0123456789abcdef01234567",
                "input_schema": "return-pairs",
                "pair_id": "1|filing.txt|item_1",
                "condition": "original",
                "prompt_token_ids": [10, 11],
                "input_span": [0, 2],
                "generated_token_ids": [21, 22],
                "generated_text": '{"label":"bullish","confidence":80}',
                "target_label": "bullish",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (forward.parent / "metadata.json").write_text(
        json.dumps({"model": "fake", "artifact_sha256": sha256_file(forward)}) + "\n",
        encoding="utf-8",
    )
    _patch_model(monkeypatch)

    output = generated_attribution.run_backward_attribution(
        forward_path=forward,
        model_name="fake",
        output_dir=tmp_path / "run",
    )
    row = json.loads(output.read_text().splitlines()[0])
    assert row["predicted_label"] == "bullish"
    assert row["predicted_confidence"] == 80
    assert row["parse_status"] == "valid"


def test_backward_output_path_writes_exact_arbitrary_filename(tmp_path, monkeypatch):
    forward = tmp_path / "run" / "forward" / "generated_outputs.jsonl"
    _forward_fixture(forward)
    _patch_model(monkeypatch)
    requested = tmp_path / "custom" / "my-attribution.jsonl"

    output = generated_attribution.run_backward_attribution(
        forward_artifact=forward,
        model_name="fake",
        output_path=requested,
    )

    assert output == requested
    assert requested.is_file()
    assert not (requested.parent / "generated_token_attribution.jsonl").exists()
    assert requested.with_suffix(".metadata.json").is_file()
