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
