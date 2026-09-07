from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.inference.mlp_addition import mlp_addition, mlp_summed_derivatives


def fixture():
    projection = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1., 2., 3.], [-2., 1., 4.]]))
    projection.requires_grad_(False)
    model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))])
    return model, projection


def test_all_positions_and_cached_decode():
    model, module = fixture()
    for length in (4, 1):
        x = torch.zeros(1, length, 3, dtype=torch.float64)
        with mlp_addition(model, 0, 1, 2):
            assert torch.equal(module(x), torch.tensor([4., 2.]).expand(1, length, 2))
        assert torch.equal(x, torch.zeros_like(x))
        assert torch.equal(module(x), torch.zeros(1, length, 2))
    assert not module._forward_pre_hooks


def test_derivative_matches_finite_difference():
    model, module = fixture()
    x = torch.ones(1, 4, 3, dtype=torch.float64)
    def objective():
        output = module(x)
        return (output[..., 0] - output[..., 1]).sum()
    with mlp_summed_derivatives(model, [0]) as gradients:
        objective().backward()
    assert torch.equal(gradients[0], torch.tensor([12., 4., -4.]))
    for neuron in range(3):
        with mlp_addition(model, 0, neuron, 1e-4):
            plus = objective().item()
        with mlp_addition(model, 0, neuron, -1e-4):
            minus = objective().item()
        assert (plus - minus) / 2e-4 == pytest.approx(gradients[0][neuron].item())
    assert module.weight.grad is None
    assert not module._forward_pre_hooks


def test_zero_and_exception_cleanup():
    model, module = fixture()
    x = torch.ones(1, 2, 3, dtype=torch.float64)
    baseline = module(x)
    with mlp_addition(model, 0, 0, 0):
        assert torch.equal(module(x), baseline)
    with pytest.raises(RuntimeError):
        with mlp_addition(model, 0, 0, 2):
            raise RuntimeError("failure")
    with pytest.raises(RuntimeError):
        with mlp_summed_derivatives(model, [0]):
            raise RuntimeError("failure")
    assert not module._forward_pre_hooks


def test_qwen35_hybrid_architecture_derivatives_and_cached_generation():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from llm_bias.core.inference.coordinate_screen import coordinate_derivatives, next_token_margin, frozen_eval
    from llm_bias.core.inference.adapter import InjectedModelAdapter
    from llm_bias.core.inference.generation import generate_tokens, GenerationConfig
    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=['linear_attention', 'full_attention'],
        pad_token_id=0, eos_token_id=1)
    config._attn_implementation = 'eager'
    raw = Qwen3_5ForCausalLM(config)
    model = SimpleNamespace(_hf_model=raw, layers=raw.model.layers)
    ids = [4, 5, 6]
    _, derivatives = coordinate_derivatives(model, ids, 2, 3, 'cpu', [0, 1])
    with frozen_eval(model), torch.no_grad():
        for layer in (0, 1):
            neuron = int(derivatives[layer].abs().argmax())
            with mlp_addition(model, layer, neuron, 1e-3):
                plus = float(next_token_margin(model, ids, 2, 3, 'cpu'))
            with mlp_addition(model, layer, neuron, -1e-3):
                minus = float(next_token_margin(model, ids, 2, 3, 'cpu'))
            assert (plus - minus) / .002 == pytest.approx(float(derivatives[layer][neuron]), rel=.01, abs=1e-4)
        adapter = InjectedModelAdapter(model, hf_model=raw)
        generation = GenerationConfig(max_new_tokens=3, pad_token_id=0)
        baseline = generate_tokens(adapter, torch.tensor([ids]), generation)
        with mlp_addition(model, 0, 0, 0):
            assert torch.equal(generate_tokens(adapter, torch.tensor([ids]), generation), baseline)
        with mlp_addition(model, 0, 0, .1):
            changed = generate_tokens(adapter, torch.tensor([ids]), generation)
            assert changed.shape[1] > len(ids)
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in raw.model.layers)


@pytest.mark.parametrize("layer,neuron,delta", [(-1, 0, 0), (0, -1, 0), (0, 0, float('nan'))])
def test_invalid(layer, neuron, delta):
    model, _ = fixture()
    with pytest.raises(ValueError):
        with mlp_addition(model, layer, neuron, delta):
            pass
