import json
from types import SimpleNamespace

import pytest
import torch

from llm_bias.prompt_analysis import attribution
from llm_bias.prompt_analysis import generated_attribution


def test_parse_generated_return_answer_is_strict_and_nonfatal():
    assert attribution.parse_generated_return_answer('{"label":"neutral","confidence":80}') == {
        "predicted_label": "neutral",
        "predicted_confidence": 80,
        "parse_status": "valid",
        "parse_reason": None,
    }
    invalid = attribution.parse_generated_return_answer('{"label":"neutral","confidence":80.0}')
    assert invalid["parse_status"] == "invalid"
    assert invalid["parse_reason"] == "invalid_confidence"


def test_backward_api_aliases_are_explicit():
    assert generated_attribution.analyze_generated_attribution_from_forward is generated_attribution.run_backward_attribution
    assert generated_attribution.backward_generated_attribution is generated_attribution.run_backward_attribution


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, _text, **_kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def decode(self, token_ids, **_kwargs):
        return "generated-" + "-".join(str(token_id) for token_id in token_ids)


class _FakeHFModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompt_ids, **kwargs):
        self.calls.append(kwargs)
        token_id = int(torch.initial_seed() % 10_000) + 3
        generated = torch.tensor([[token_id]], device=prompt_ids.device)
        return torch.cat([prompt_ids, generated], dim=1)


def test_generate_tokens_uses_greedy_without_sampling_temperature():
    hf_model = _FakeHFModel()
    model = SimpleNamespace(
        hf_model=hf_model,
        tokenizer=SimpleNamespace(eos_token_id=0),
    )

    attribution._generate_tokens(
        model,
        torch.tensor([[1, 2]]),
        4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )

    assert hf_model.calls == [
        {
            "max_new_tokens": 4,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": 0,
        }
    ]


def test_generate_tokens_passes_sampling_controls():
    hf_model = _FakeHFModel()
    model = SimpleNamespace(
        hf_model=hf_model,
        tokenizer=SimpleNamespace(eos_token_id=0),
    )

    attribution._generate_tokens(
        model,
        torch.tensor([[1, 2]]),
        4,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
    )

    assert hf_model.calls[0]["do_sample"] is True
    assert hf_model.calls[0]["temperature"] == pytest.approx(0.7)
    assert hf_model.calls[0]["top_p"] == pytest.approx(1.0)
    assert hf_model.calls[0]["top_k"] == 0


def test_multiple_runs_write_run_records_and_manifest(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.csv"
    input_path.write_text(
        "Date,prompt_without_context_aapl,prompt_with_context_aapl\n"
        "2026-01-01,plain,with context\n",
        encoding="utf-8",
    )
    tokenizer = _Tokenizer()
    hf_model = _FakeHFModel()
    wrapped = SimpleNamespace(
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        attribution,
        "load_lens_model",
        lambda _model_name: (SimpleNamespace(_hf_model=object()), tokenizer, "cpu"),
    )
    monkeypatch.setattr(attribution, "WrappedModel", lambda *_args: wrapped)
    monkeypatch.setattr(
        attribution,
        "format_prompt",
        lambda _tokenizer, prompt, **_kwargs: prompt,
    )
    attribution_calls = []

    def fake_attribute_generated_token(**kwargs):
        attribution_calls.append(kwargs)
        return {
            "token_id": 3,
            "token": "generated",
            "logit": 0.0,
            "log_probability": 0.0,
            "top_input_tokens": [],
        }

    monkeypatch.setattr(
        attribution,
        "_attribute_generated_token",
        fake_attribute_generated_token,
    )

    output_dir = tmp_path / "runs"
    attribution.analyze_generated_attribution(
        input_path=str(input_path),
        model_name="fake-qwen",
        output_dir=str(output_dir),
        sample_per_condition=1,
        max_new_tokens=4,
        runs=2,
        temperature=0.7,
        seed=10,
        backprop=True,
    )

    records = []
    for run_index in range(2):
        path = output_dir / f"run_{run_index:03d}" / "generated_token_attribution.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2
        assert {row["run_index"] for row in rows} == {run_index}
        assert [row["sample_index"] for row in rows] == [0, 0]
        assert all(row["generation"]["do_sample"] for row in rows)
        metadata = json.loads(
            (output_dir / f"run_{run_index:03d}" / "metadata.json").read_text()
        )
        assert metadata["backpropagation"] is True
        records.extend(rows)

    assert len({row["generated_text"] for row in records}) == 2
    assert len(attribution_calls) == len(records)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["runs"] == 2
    assert manifest["backpropagation"] is True
    assert manifest["records_per_run"] == 2
    assert [item["run_index"] for item in manifest["run_directories"]] == [0, 1]


def test_sampling_uses_shared_nonempty_dates_for_every_condition(tmp_path, monkeypatch):
    input_path = tmp_path / "legacy.csv"
    input_path.write_text(
        "Date,prompt_without_context_aapl,prompt_with_context_aapl\n"
        "2026-01-01,first,first with\n"
        "2026-01-02,second,second with\n"
        "2026-01-03,third,\n"
        "2026-01-04,,fourth with\n",
        encoding="utf-8",
    )
    tokenizer = _Tokenizer()
    hf_model = _FakeHFModel()
    wrapped = SimpleNamespace(
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        attribution,
        "load_lens_model",
        lambda _model_name: (SimpleNamespace(_hf_model=object()), tokenizer, "cpu"),
    )
    monkeypatch.setattr(attribution, "WrappedModel", lambda *_args: wrapped)
    monkeypatch.setattr(attribution, "format_prompt", lambda _tokenizer, prompt, **_kwargs: prompt)
    monkeypatch.setattr(attribution, "_generate_tokens", lambda *_args, **_kwargs: torch.tensor([[1, 2, 3]]))
    monkeypatch.setattr(
        attribution,
        "_attribute_generated_token",
        lambda **_kwargs: {
            "token_id": 3,
            "token": "generated",
            "logit": 0.0,
            "log_probability": 0.0,
            "top_input_tokens": [],
        },
    )

    output_dir = tmp_path / "shared-date-run"
    attribution.analyze_generated_attribution(
        input_path=str(input_path),
        model_name="fake",
        output_dir=str(output_dir),
        sample_per_condition=1,
        max_new_tokens=1,
        backprop=True,
    )

    rows = [json.loads(line) for line in (output_dir / "generated_token_attribution.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {row["date"] for row in rows} == {"2026-01-01"}
    assert {row["prompt_column"] for row in rows} == {
        "prompt_without_context_aapl",
        "prompt_with_context_aapl",
    }


def test_disabled_generated_attribution_is_rejected_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(
        attribution,
        "load_lens_model",
        lambda _model_name: pytest.fail("disabled attribution must not load a model"),
    )

    with pytest.raises(ValueError, match="requires backprop=True"):
        attribution.analyze_generated_attribution(
            input_path=str(tmp_path / "missing.csv"),
            model_name="fake",
            backprop=False,
        )


def test_greedy_repeated_runs_are_rejected_before_model_load(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.csv"
    input_path.write_text(
        "Date,prompt_without_context_aapl\n2026-01-01,plain\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        attribution,
        "load_lens_model",
        lambda _model_name: pytest.fail("model must not load after invalid controls"),
    )

    with pytest.raises(ValueError, match="require temperature greater than zero"):
        attribution.analyze_generated_attribution(
            input_path=str(input_path),
            runs=2,
            temperature=0.0,
            backprop=True,
        )


def test_sampling_controls_validate_finite_ranges(tmp_path):
    input_path = tmp_path / "prompts.csv"
    input_path.write_text(
        "Date,prompt_without_context_aapl\n2026-01-01,plain\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite non-negative"):
        attribution.analyze_generated_attribution(
            input_path=str(input_path),
            temperature=float("nan"),
            backprop=True,
        )
    with pytest.raises(ValueError, match="top_p"):
        attribution.analyze_generated_attribution(
            input_path=str(input_path),
            top_p=0.0,
            backprop=True,
        )
    with pytest.raises(ValueError, match="top_k"):
        attribution.analyze_generated_attribution(
            input_path=str(input_path),
            top_k=-1,
            backprop=True,
        )
