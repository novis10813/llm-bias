"""Regression tests for the dial probe (single-forward margin + dial readout).

A small real Qwen3.5 fake model (fp32, CPU) is used throughout; no
checkpoint is loaded.
"""
from __future__ import annotations

import pytest
import torch

from llm_bias.core.inference.mlp_addition import mlp_addition
from llm_bias.entity_to_dial.dial_probe import (
    answer_token_ids,
    margin_from_log_probs,
    probe_forward,
    scoring_ids,
)


class _CharTokenizer:
    """Character tokenizer with single-token buy/sell continuations."""

    BUY_ID = 240
    SELL_ID = 241
    VOCAB = 900

    chat_template = "fake"

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = True,
        **_kwargs,
    ):
        from types import SimpleNamespace

        del add_special_tokens
        text = str(text)
        for candidate, token_id in (("buy", self.BUY_ID), ("sell", self.SELL_ID)):
            if text.endswith(candidate):
                prefix = text[: -len(candidate)]
                return SimpleNamespace(
                    input_ids=[ord(char) for char in prefix] + [token_id]
                )
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def _model(num_layers: int = 2) -> object:
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=900, hidden_size=32, intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=[
            "full_attention" if (i % 4 == 3 or i == num_layers - 1) else "linear_attention"
            for i in range(num_layers)
        ],
        pad_token_id=0, eos_token_id=1,
    )
    raw = Qwen3_5ForCausalLM(config)
    raw.eval()

    class _Model:
        def __init__(self) -> None:
            self.raw = raw
            self.layers = raw.model.layers
            self.n_layers = num_layers
            self.input_device = "cpu"
            self._final_norm = raw.model.norm
            self._lm_head = raw.lm_head

        def forward(self, input_ids, attention_mask=None):
            return raw(input_ids, attention_mask=attention_mask)

    return _Model()


def _scoring_row(tokenizer: _CharTokenizer, length: int = 20) -> str:
    # Deterministic pseudo-text ending in the JSON decision prefix.
    body = "".join(chr(ord("a") + (i % 26)) for i in range(length))
    return "«" + body + '»{"decision": "'


def _ids(tokenizer: _CharTokenizer, formatted: str) -> torch.Tensor:
    return torch.tensor([scoring_ids(tokenizer, formatted)], dtype=torch.long)


def test_probe_margin_matches_fp32_tail_reference():
    tokenizer = _CharTokenizer()
    model = _model()
    formatted = _scoring_row(tokenizer)
    tensor = _ids(tokenizer, formatted)
    buy_id, sell_id = answer_token_ids(tokenizer, formatted + '{"decision": "')

    margin, _ = probe_forward(model, tensor, buy_id=buy_id, sell_id=sell_id)

    # Independent reference: plain forward + FP32 tail.
    from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
    from llm_bias.core.inference.forward import record_residuals

    final = int(model.n_layers) - 1
    residual = record_residuals(model, tensor, [final])[final]
    reference = margin_from_log_probs(
        fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
    )
    assert margin == reference


def test_probe_reads_down_projection_input_channel():
    tokenizer = _CharTokenizer()
    model = _model()
    formatted = _scoring_row(tokenizer)
    tensor = _ids(tokenizer, formatted)
    buy_id, sell_id = answer_token_ids(tokenizer, formatted + '{"decision": "')
    positions = (3, 7)
    neuron = 5
    _, dials = probe_forward(
        model, tensor, buy_id=buy_id, sell_id=sell_id,
        dial_positions=positions, dial_layer=1, dial_neuron=neuron,
    )

    # Independent reference: hook the same module directly.
    from llm_bias.core.inference.mlp import dense_down_projection

    box: dict = {}
    module = dense_down_projection(model.layers[1])
    handle = module.register_forward_pre_hook(
        lambda _m, args: box.update(values=args[0])
    )
    try:
        with torch.no_grad():
            model.forward(tensor)
    finally:
        handle.remove()
    for pos in positions:
        assert dials[pos] == box["values"][0, pos, neuron].float().item()


def test_probe_single_forward_yields_margin_and_dual_position_dials():
    tokenizer = _CharTokenizer()
    model = _model()
    formatted = _scoring_row(tokenizer)
    tensor = _ids(tokenizer, formatted)
    buy_id, sell_id = answer_token_ids(tokenizer, formatted + '{"decision": "')
    margin, dials = probe_forward(
        model, tensor, buy_id=buy_id, sell_id=sell_id,
        dial_positions=(2, int(tensor.shape[1]) - 1), dial_layer=1, dial_neuron=1,
    )
    assert isinstance(margin, float) and torch.isfinite(torch.tensor(margin))
    assert set(dials) == {2, tensor.shape[1] - 1}
    for value in dials.values():
        assert torch.isfinite(torch.tensor(value))
    # No hooks left behind.
    assert not model.layers[1]._forward_hooks
    assert not model.layers[1].mlp.down_proj._forward_pre_hooks
    assert not model.layers[int(model.n_layers) - 1]._forward_hooks


def test_mlp_addition_moves_margin_but_not_dial_input():
    tokenizer = _CharTokenizer()
    model = _model()
    formatted = _scoring_row(tokenizer)
    tensor = _ids(tokenizer, formatted)
    buy_id, sell_id = answer_token_ids(tokenizer, formatted + '{"decision": "')
    delta = 0.25
    clean_margin, clean_dials = probe_forward(
        model, tensor, buy_id=buy_id, sell_id=sell_id,
        dial_positions=(3,), dial_layer=1, dial_neuron=2,
    )
    with mlp_addition(model, 1, 2, delta):
        pushed_margin, pushed_dials = probe_forward(
            model, tensor, buy_id=buy_id, sell_id=sell_id,
            dial_positions=(3,), dial_layer=1, dial_neuron=2,
        )
    # The dial readout sits at the down-proj input — exactly where the
    # addition lands (the mlp_addition hook is registered before the dial
    # hook), so the readout moves by the fp32-rounded delta.
    expected_dial = (
        torch.tensor(clean_dials[3]) + torch.tensor(delta)
    ).item()
    assert pushed_dials[3] == expected_dial
    # The margin moves (the added intermediate channel feeds the residual).
    assert pushed_margin != clean_margin


def test_mlp_addition_is_exact_in_native_dtype():
    from types import SimpleNamespace

    torch.manual_seed(1)
    projection = torch.nn.Linear(4, 3, bias=False, dtype=torch.float32)
    model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))])
    x = torch.randn(1, 5, 4)
    x_before = x.clone()
    clean = projection(x)
    delta = 0.125
    # The addition lands on the down-proj input (intermediate channel) at
    # every position, in native dtype; the output is W (x + delta e1).
    x_pushed = x.clone()
    x_pushed[..., 1] += delta
    expected = projection(x_pushed)
    with mlp_addition(model, 0, 1, delta):
        pushed = projection(x)
    assert torch.equal(pushed, expected)
    assert not torch.equal(pushed, clean)
    assert torch.equal(x, x_before)  # input untouched
    assert not projection._forward_pre_hooks


def test_probe_fail_closed_on_bad_positions():
    tokenizer = _CharTokenizer()
    model = _model()
    formatted = _scoring_row(tokenizer)
    tensor = _ids(tokenizer, formatted)
    buy_id, sell_id = answer_token_ids(tokenizer, formatted + '{"decision": "')
    with pytest.raises(ValueError, match="dial position outside"):
        probe_forward(
            model, tensor, buy_id=buy_id, sell_id=sell_id,
            dial_positions=(tensor.shape[1] + 1,), dial_layer=1, dial_neuron=1,
        )
    with pytest.raises(ValueError, match="dial position outside"):
        probe_forward(
            model, tensor, buy_id=buy_id, sell_id=sell_id,
            dial_positions=(-1,), dial_layer=1, dial_neuron=1,
        )
    with pytest.raises(ValueError, match="dial neuron out of range"):
        probe_forward(
            model, tensor, buy_id=buy_id, sell_id=sell_id,
            dial_positions=(1,), dial_layer=1, dial_neuron=10_000,
        )
    with pytest.raises(ValueError, match="must be \\[1, sequence\\]"):
        probe_forward(
            model, tensor.squeeze(0), buy_id=buy_id, sell_id=sell_id,
        )
    # Hooks are removed even after a failed forward.
    assert not model.layers[1]._forward_hooks
    assert not model.layers[int(model.n_layers) - 1]._forward_hooks


def test_answer_token_ids_requires_distinct_single_tokens():
    tokenizer = _CharTokenizer()
    buy, sell = answer_token_ids(tokenizer, "prefix{\"decision\": \"")
    assert buy == _CharTokenizer.BUY_ID and sell == _CharTokenizer.SELL_ID
    assert buy != sell


def test_margin_from_log_probs_rejects_non_finite():
    log_probs = torch.tensor([[float("nan"), 0.5, 0.1]])
    with pytest.raises(ValueError, match="non-finite margin"):
        margin_from_log_probs(log_probs, 0, 2)
