from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.inference import GenerationConfig, encode_batch, extract_logits, finish_reason
from llm_bias.core.inference.interventions import mid_residual_interventions, record_block_states


def test_encode_batch_tracks_each_final_non_padding_position():
    encoded = encode_batch([[1, 2, 3], [4]], "cpu", pad_token_id=99)
    assert encoded.input_ids.tolist() == [[1, 2, 3], [4, 99, 99]]
    assert encoded.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert encoded.final_positions.tolist() == [2, 0]


def test_extract_logits_accepts_hf_output_shapes():
    logits = torch.randn(2, 3, 5)
    assert torch.equal(extract_logits(SimpleNamespace(logits=logits)), logits)
    assert torch.equal(extract_logits((logits, "cache")), logits)
    assert torch.equal(extract_logits({"logits": logits}), logits)


def test_extract_logits_falls_back_to_unembed_for_logitless_mapping():
    # Multimodal wrappers expose a bare text decoder whose dict-like output
    # carries last_hidden_state but no logits; unembed must handle it.
    residual = torch.randn(2, 5)
    unembedded = residual * 2
    model = SimpleNamespace(unembed=lambda value: unembedded)
    output = {"last_hidden_state": residual}
    assert torch.equal(extract_logits(output, model=model, residual=residual), unembedded)


def test_generation_config_only_adds_sampling_controls_when_sampling():
    assert GenerationConfig(pad_token_id=0).as_kwargs() == {
        "max_new_tokens": 64,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": 0,
    }
    assert GenerationConfig(temperature=0.7, top_p=0.9, top_k=10).as_kwargs()["top_k"] == 10


def test_finish_reason_distinguishes_eos_and_length():
    assert finish_reason([], eos_token_id=0, max_new_tokens=2) == "empty"
    assert finish_reason([3, 0], eos_token_id=0, max_new_tokens=2) == "eos_token"
    assert finish_reason([3, 4], eos_token_id=0, max_new_tokens=2) == "max_new_tokens"


# ── block states and mid-residual interventions ──────────────────────────────────────


def _qwen_fake(num_layers: int = 2) -> tuple[object, object]:
    """Small real Qwen3.5 model exposing jlens-style .layers + .forward."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=num_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"],
        pad_token_id=0, eos_token_id=1,
    )
    raw = Qwen3_5ForCausalLM(config)
    raw.eval()

    class _Model:
        def __init__(self) -> None:
            self.layers = raw.model.layers
            self.n_layers = num_layers

        def forward(self, input_ids, **kwargs):
            return raw(input_ids, **kwargs)

    return raw, _Model()


def test_record_block_states_captures_chain_and_contribution():
    raw, model = _qwen_fake()
    ids = torch.randint(2, 31, (1, 6))
    states = record_block_states(model, ids, [0, 1])
    for layer in (0, 1):
        for key in ("pre", "mid", "post"):
            assert states[layer][key].shape == (1, 6, 32)
        assert not states[layer]["post"].requires_grad
    # Residual chain: block l output is the bit-exact input of block l+1.
    assert torch.equal(states[1]["pre"], states[0]["post"])
    # MLP contribution identity: post - mid == mlp(post_attention_layernorm(mid)).
    block = raw.model.layers[0]
    with torch.no_grad():
        expected = block.mlp(block.post_attention_layernorm(states[0]["mid"]))
    assert torch.allclose(
        (states[0]["post"] - states[0]["mid"]).float(), expected.float(), atol=1e-5
    )


def test_record_block_states_fail_closed():
    block = torch.nn.Linear(4, 4)
    model = SimpleNamespace(layers=[block])
    with pytest.raises(TypeError):
        record_block_states(model, torch.zeros(1, 4, dtype=torch.long), [0])
    assert not block._forward_pre_hooks
    _, model = _qwen_fake()
    with pytest.raises(ValueError):
        record_block_states(model, torch.zeros(1, 4, dtype=torch.long), [5])
    assert record_block_states(model, torch.zeros(1, 4, dtype=torch.long), []) == {}


def test_mid_residual_interventions_applies_and_removes_hooks():
    raw, model = _qwen_fake()
    ids = torch.randint(2, 31, (1, 6))
    with torch.no_grad():
        clean = model.forward(ids)
    norm = raw.model.layers[0].post_attention_layernorm
    with mid_residual_interventions(model, {0: lambda t: t + 1.0}):
        with torch.no_grad():
            perturbed = model.forward(ids)
    assert not torch.equal(clean.logits, perturbed.logits)
    assert not norm._forward_pre_hooks
    with mid_residual_interventions(model, {}):
        with torch.no_grad():
            assert torch.equal(model.forward(ids).logits, clean.logits)


def test_mid_residual_interventions_shape_contract_and_exception_cleanup():
    raw, model = _qwen_fake()
    ids = torch.randint(2, 31, (1, 6))
    norm = raw.model.layers[1].post_attention_layernorm
    with pytest.raises(ValueError):
        with mid_residual_interventions(model, {1: lambda t: t[..., :16]}):
            with torch.no_grad():
                model.forward(ids)
    assert not norm._forward_pre_hooks
    with pytest.raises(RuntimeError):
        with mid_residual_interventions(model, {0: lambda t: t + 1.0}):
            with torch.no_grad():
                model.forward(ids)
            raise RuntimeError("failure")
    assert not raw.model.layers[0].post_attention_layernorm._forward_pre_hooks
