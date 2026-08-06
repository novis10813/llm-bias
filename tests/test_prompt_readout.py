from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_bias.prompt_analysis import interactive


class _Tokenizer:
    chat_template = "{% if enable_thinking %}thinking{% endif %}"

    def __call__(self, text, **_kwargs):
        return SimpleNamespace(input_ids=list(range(len(text.split()))))

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        return (
            f"formatted:{messages[0]['content']}:"
            f"thinking={enable_thinking}"
        )


def _state():
    model = SimpleNamespace(
        tokenizer=_Tokenizer(),
        device="cpu",
        n_layers=4,
        d_model=16,
    )
    lens = SimpleNamespace(source_layers=[0, 2], n_prompts=20)
    return interactive.PromptReadoutState(model, lens, "model", "lens.pt")


def test_prompt_readout_request_rejects_invalid_controls():
    with pytest.raises(ValidationError):
        interactive.PromptReadoutRequest(prompt="", top_k=0)
    with pytest.raises(ValidationError):
        interactive.PromptReadoutRequest(prompt="valid", mode="unknown")


def test_read_prompt_returns_compact_grid_metadata(monkeypatch):
    captured = {}

    def fake_read_grid(model, lens, prompt, **kwargs):
        captured.update(model=model, lens=lens, prompt=prompt, **kwargs)
        return {
            "prompt": prompt,
            "prompt_len": 2,
            "seq_len": 2,
            "grid": [],
            "layers": [0, 3],
        }

    monkeypatch.setattr(interactive, "read_grid", fake_read_grid)
    request = interactive.PromptReadoutRequest(
        prompt="  hello model  ",
        mode="jlens",
        top_k=6,
        max_seq_len=32,
        chat=True,
    )

    result = interactive.read_prompt(_state(), request)

    assert result["input_prompt"] == "hello model"
    assert result["requested_max_seq_len"] == 32
    assert result["truncated"] is False
    assert result["thinking_enabled"] is False
    assert captured["prompt"] == "formatted:hello model:thinking=False"
    assert captured["top_k"] == 6
    assert captured["chat"] is False
    assert captured["generate_continuation"] is False


def test_read_prompt_preserves_complete_grid_top_k_and_vocab(monkeypatch):
    expected_grid = [
        {
            "layer": 0,
            "is_output": False,
            "top_ids": [[21, 22, 23], [24, 21, 25]],
            "top_probs": [[0.5, 0.3, 0.2], [0.4, 0.35, 0.1]],
            "entropy": [1.1, 1.2],
            "kurtosis": [2.1, 2.2],
        },
        {
            "layer": 3,
            "is_output": True,
            "top_ids": [[23, 21, 26], [22, 23, 21]],
            "top_probs": [[0.45, 0.4, 0.1], [0.6, 0.2, 0.15]],
            "entropy": [1.3, 1.4],
            "kurtosis": [2.3, 2.4],
        },
    ]
    expected_vocab = {
        11: "context one",
        12: "context two",
        21: "shared token",
        22: "second token",
        23: "shared token",
        24: "fourth token",
        25: "fifth token",
        26: "sixth token",
    }

    def fake_read_grid(_model, _lens, prompt, **_kwargs):
        return {
            "prompt": prompt,
            "prompt_len": 2,
            "seq_len": 2,
            "layers": [0, 3],
            "context_ids": [11, 12],
            "vocab": expected_vocab,
            "grid": expected_grid,
        }

    monkeypatch.setattr(interactive, "read_grid", fake_read_grid)

    result = interactive.read_prompt(
        _state(),
        interactive.PromptReadoutRequest(
            prompt="hello model",
            chat=False,
            top_k=3,
            max_seq_len=8,
        ),
    )

    assert result["context_ids"] == [11, 12]
    assert result["vocab"] == expected_vocab
    assert result["grid"] == expected_grid
    assert result["grid"][0]["is_output"] is False
    assert result["grid"][1]["is_output"] is True
    assert result["grid"][0]["top_ids"] == [[21, 22, 23], [24, 21, 25]]
    assert result["grid"][1]["top_probs"] == [[0.45, 0.4, 0.1], [0.6, 0.2, 0.15]]
    assert result["vocab"][21] == result["vocab"][23]
    assert 21 in result["grid"][0]["top_ids"][0]
    assert 21 in result["grid"][0]["top_ids"][1]


def test_read_prompt_can_include_actual_model_response(monkeypatch):
    def fake_read_grid(_model, _lens, prompt, **_kwargs):
        return {
            "prompt": f"formatted:{prompt}",
            "prompt_len": 1,
            "seq_len": 1,
            "grid": [],
            "layers": [0, 3],
        }

    generated = {}

    def fake_generate(_model, formatted_prompt, **kwargs):
        generated.update(prompt=formatted_prompt, **kwargs)
        return "Yes, I care about our conversation.", 8

    monkeypatch.setattr(interactive, "read_grid", fake_read_grid)
    monkeypatch.setattr(interactive, "_generate_response", fake_generate)

    result = interactive.read_prompt(
        _state(),
        interactive.PromptReadoutRequest(
            prompt="Do you love me?",
            enable_thinking=True,
            generate_continuation=True,
            max_new_tokens=80,
        ),
    )

    assert result["continuation"] == "Yes, I care about our conversation."
    assert result["response_token_count"] == 8
    assert generated == {
        "prompt": (
            "formatted:formatted:Do you love me?:thinking=True"
        ),
        "max_seq_len": 256,
        "max_new_tokens": 80,
    }
    assert result["thinking_enabled"] is True


def test_logit_readout_selects_every_model_layer(monkeypatch):
    captured = {}

    def fake_read_grid(_model, lens, prompt, **_kwargs):
        captured["source_layers"] = lens.source_layers
        return {
            "prompt": prompt,
            "prompt_len": 1,
            "seq_len": 1,
            "grid": [],
            "layers": [0, 1, 2, 3],
        }

    monkeypatch.setattr(interactive, "read_grid", fake_read_grid)

    interactive.read_prompt(
        _state(),
        interactive.PromptReadoutRequest(prompt="hello", mode="logit"),
    )

    assert captured["source_layers"] == [0, 1, 2]


def test_read_prompt_requires_chat_template(monkeypatch):
    state = _state()
    state.model.tokenizer.chat_template = None
    request = interactive.PromptReadoutRequest(prompt="hello", chat=True)

    with pytest.raises(ValueError, match="chat formatting is unavailable"):
        interactive.read_prompt(state, request)
