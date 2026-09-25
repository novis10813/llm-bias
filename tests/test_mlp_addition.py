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


@pytest.mark.parametrize("layer,neuron,delta", [(-1, 0, 0), (0, -1, 0), (0, 0, float('nan'))])
def test_invalid(layer, neuron, delta):
    model, _ = fixture()
    with pytest.raises(ValueError):
        with mlp_addition(model, layer, neuron, delta):
            pass
