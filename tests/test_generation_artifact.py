import hashlib
import json
import re
from types import SimpleNamespace

import pytest
import torch

from llm_bias.prompt_analysis import generation


class _Tokenizer:
    chat_template = "available"
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, _text, **_kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def apply_chat_template(self, messages, **_kwargs):
        return " ".join(message["content"] for message in messages)

    def decode(self, token_ids, **_kwargs):
        return "generated-" + "-".join(str(token_id) for token_id in token_ids)


class _FakeHFModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompt_ids, **kwargs):
        assert torch.is_grad_enabled() is False
        self.calls.append(kwargs)
        generated = torch.tensor([[10 + len(self.calls), 0]], device=prompt_ids.device)
        return torch.cat([prompt_ids, generated], dim=1)


def _patch_fake_model(monkeypatch):
    tokenizer = _Tokenizer()
    hf_model = _FakeHFModel()
    wrapped = SimpleNamespace(
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        generation,
        "load_lens_model",
        lambda _model_name: (SimpleNamespace(_hf_model=object()), tokenizer, "cpu"),
    )
    monkeypatch.setattr(generation, "WrappedModel", lambda *_args: wrapped)
    monkeypatch.setattr(generation, "format_prompt", lambda _tokenizer, prompt, **_kwargs: prompt)
    return hf_model


def _write_return_pairs(path):
    path.write_text(
        "cik,filename,item,filing_date,ticker,peer_ticker,system_prompt,prompt,counterfactual_prompt,return_label,fwd_return_1d\n"
        "1,a.txt,item_1,2026-01-01,AAA,BBB,sys,orig,counter,neutral,0.0\n"
        "2,b.txt,item_2,2026-01-02,CCC,DDD,sys,orig2,counter2,bullish,0.1\n",
        encoding="utf-8",
    )


def test_generation_only_writes_complete_return_pair_artifact(tmp_path, monkeypatch):
    input_path = tmp_path / "pairs.csv"
    _write_return_pairs(input_path)
    hf_model = _patch_fake_model(monkeypatch)

    output_path = generation.generate_prompt_outputs(
        input_path=str(input_path),
        model_name="fake",
        output_dir=str(tmp_path / "run-root"),
        full_generation=True,
        max_new_tokens=2,
        dataset_format="return-pairs",
    )

    assert output_path == tmp_path / "run-root" / "forward" / "generated_outputs.jsonl"
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 4
    assert len(hf_model.calls) == 4
    assert {record["condition"] for record in records} == {"original", "counterfactual"}
    assert {record["pair_id"] for record in records} == {"1|a.txt|item_1", "2|b.txt|item_2"}
    assert len({record["record_id"] for record in records}) == 4
    assert all(re.fullmatch(r"record_[0-9a-f]{24}", record["record_id"]) for record in records)

    for record in records:
        assert record["schema_version"] == generation.SCHEMA_VERSION
        assert record["artifact_type"] == generation.ARTIFACT_TYPE
        assert record["generated_token_ids"] in ([11, 0], [12, 0], [13, 0], [14, 0])
        assert record["generated_text"] == "generated-" + "-".join(
            str(token_id) for token_id in record["generated_token_ids"]
        )
        assert record["prompt_token_ids"] == [1, 2]
        assert record["input_span"] == [0, 2]
        assert record["generation_config"]["do_sample"] is False
        assert record["finish_reason"] == "eos_token"

    metadata = json.loads((tmp_path / "run-root" / "forward" / "metadata.json").read_text())
    assert metadata["backpropagation"] is False
    assert metadata["records_written"] == 4


def test_ten_k_generation_writes_one_record_per_source_row(tmp_path, monkeypatch):
    input_path = tmp_path / "ten_k.csv"
    input_path.write_text(
        "year,cik,item\n2020,1,company=ACME\n2020,1,sic=1234\n2020,1,company=ACME\n",
        encoding="utf-8",
    )
    _patch_fake_model(monkeypatch)

    output_path = generation.generate_prompt_outputs(
        input_path=str(input_path),
        model_name="fake",
        output_dir=tmp_path / "run-root",
        max_new_tokens=1,
        dataset_format="ten-k-change",
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 3
    assert len({record["record_id"] for record in records}) == 3
    assert records[0]["prompt"] == (
        "In year 2020, what is the company name of the company with CIK code 1? "
        "Answer without explanation"
    )
    assert records[1]["item_name"] == "SIC code"
    metadata = json.loads((tmp_path / "run-root" / "forward" / "metadata.json").read_text())
    assert metadata["dataset_format"] == "ten-k-change"
    assert metadata["records_written"] == 3


def test_multi_run_writes_sampling_manifest_and_forward_paths(tmp_path, monkeypatch):
    input_path = tmp_path / "legacy.csv"
    input_path.write_text(
        "Date,prompt_without_context_aapl,prompt_with_context_aapl\n"
        "2026-01-01,plain,with\n"
        "2026-01-02,plain2,with2\n",
        encoding="utf-8",
    )
    _patch_fake_model(monkeypatch)

    root = tmp_path / "sampling"
    generation.generate_prompt_outputs(
        input_path=str(input_path),
        model_name="fake",
        output_dir=root,
        runs=2,
        sample_per_condition=1,
        temperature=0.7,
        seed=100,
        max_new_tokens=1,
    )

    manifest = json.loads((root / "sampling_manifest.json").read_text())
    assert manifest["artifact_type"] == "generated_output_sampling"
    assert manifest["run_indices"] == [0, 1]
    assert [entry["run_seed"] for entry in manifest["run_directories"]] == [100, 101]
    for index in (0, 1):
        path = root / f"run_{index:03d}" / "forward" / "generated_outputs.jsonl"
        assert path.is_file()
        assert manifest["run_directories"][index]["forward_artifact"] == (
            f"run_{index:03d}/forward/generated_outputs.jsonl"
        )
        assert manifest["run_directories"][index]["forward_sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_output_path_writes_requested_jsonl_file(tmp_path, monkeypatch):
    input_path = tmp_path / "pairs.csv"
    _write_return_pairs(input_path)
    _patch_fake_model(monkeypatch)
    requested = tmp_path / "results.jsonl"

    output_path = generation.generate_prompt_outputs(
        input_path=str(input_path),
        model_name="fake",
        output_path=requested,
        full_generation=True,
        max_new_tokens=1,
        dataset_format="return-pairs",
    )

    assert output_path == requested
    assert requested.is_file()
    assert not (tmp_path / "results.jsonl" / "forward" / "generated_outputs.jsonl").exists()
    assert (tmp_path / "metadata.json").is_file()


def test_legacy_sampling_keeps_deterministic_32_date_selection(tmp_path, monkeypatch):
    input_path = tmp_path / "legacy.csv"
    rows = [
        f"2026-01-{index:02d},plain-{index},with-{index}"
        for index in range(1, 41)
    ]
    input_path.write_text(
        "Date,prompt_without_context_aapl,prompt_with_context_aapl\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    _patch_fake_model(monkeypatch)

    output_path = generation.generate_prompt_outputs(
        input_path=str(input_path),
        model_name="fake",
        output_dir=str(tmp_path / "legacy-run"),
        max_new_tokens=1,
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 64
    assert len({record["date"] for record in records}) == 32


def test_generation_controls_reject_invalid_input_top_k(tmp_path):
    input_path = tmp_path / "prompts.csv"
    input_path.write_text(
        "Date,prompt_without_context_aapl\n2026-01-01,plain\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input_top_k"):
        generation.generate_prompt_outputs(
            input_path=str(input_path),
            input_top_k=0,
        )
