from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.inference.interventions import (
    mid_residual_interventions,
    pre_residual_interventions,
    residual_interventions,
)


class _Scale(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = _Scale()

    def forward(self, hidden_states, *, keyword_norm=False):
        mid = hidden_states + 1
        if keyword_norm:
            normalized = self.post_attention_layernorm(hidden_states=mid)
        else:
            normalized = self.post_attention_layernorm(mid)
        return normalized * 3


_INTERVENTIONS = {
    "pre": pre_residual_interventions,
    "mid": mid_residual_interventions,
    "post": residual_interventions,
}


@pytest.fixture(params=["pre", "mid", "post"])
def hook_site(request):
    return request.param


@pytest.fixture
def model():
    return SimpleNamespace(layers=[_Block(), _Block()])


def _hook_target(model, hook_site, layer):
    block = model.layers[layer]
    return block.post_attention_layernorm if hook_site == "mid" else block


def _installed_hooks(target, hook_site):
    return target._forward_hooks if hook_site == "post" else target._forward_pre_hooks


def _assert_no_hooks(model):
    for block in model.layers:
        for module in block.modules():
            assert not module._forward_hooks
            assert not module._forward_pre_hooks


def _forward(model, hidden_states, *, keyword=False):
    for block in model.layers:
        if keyword:
            hidden_states = block(hidden_states=hidden_states, keyword_norm=True)
        else:
            hidden_states = block(hidden_states)
    return hidden_states


def test_later_invalid_layer_removes_earlier_hook(model, hook_site):
    with pytest.raises(ValueError, match="intervention layer 2 is out of range"):
        with _INTERVENTIONS[hook_site](model, {0: lambda state: state, 2: lambda state: state}):
            pytest.fail("context entered despite an invalid layer")
    _assert_no_hooks(model)


def test_second_registration_failure_removes_first_hook(model, hook_site, monkeypatch):
    first = _hook_target(model, hook_site, 0)
    second = _hook_target(model, hook_site, 1)
    registration = "register_forward_hook" if hook_site == "post" else "register_forward_pre_hook"

    def fail_registration(*args, **kwargs):
        assert len(_installed_hooks(first, hook_site)) == 1
        raise RuntimeError("second registration failed")

    monkeypatch.setattr(second, registration, fail_registration)
    with pytest.raises(RuntimeError, match="second registration failed"):
        with _INTERVENTIONS[hook_site](model, {0: lambda state: state, 1: lambda state: state}):
            pytest.fail("context entered despite a registration failure")
    _assert_no_hooks(model)


@pytest.mark.parametrize("sequence_length", [1, 4])
@pytest.mark.parametrize("keyword", [False, True])
def test_normal_exit_preserves_transform_sites_and_removes_hooks(
    model, hook_site, sequence_length, keyword
):
    hidden_states = torch.ones(1, sequence_length, 3)
    baseline = _forward(model, hidden_states, keyword=keyword)
    with _INTERVENTIONS[hook_site](model, {0: lambda state: state + 2, 1: lambda state: state * 2}):
        for layer in (0, 1):
            assert len(_installed_hooks(_hook_target(model, hook_site, layer), hook_site)) == 1
        # The fake blocks compute 6 * (hidden + 1); the distinct expected
        # results distinguish pre-block, post-attention, and post-block edits.
        expected = {"pre": 294.0, "mid": 300.0, "post": 180.0}[hook_site]
        assert torch.equal(
            _forward(model, hidden_states, keyword=keyword), torch.full_like(hidden_states, expected)
        )
    _assert_no_hooks(model)
    assert torch.equal(_forward(model, hidden_states, keyword=keyword), baseline)


def test_forward_transform_exception_removes_all_hooks(model, hook_site):
    hidden_states = torch.ones(1, 4, 3)
    baseline = _forward(model, hidden_states)

    def fail_transform(state):
        raise RuntimeError("transform failed")

    with pytest.raises(RuntimeError, match="transform failed"):
        with _INTERVENTIONS[hook_site](model, {0: lambda state: state + 2, 1: fail_transform}):
            _forward(model, hidden_states)
    _assert_no_hooks(model)
    assert torch.equal(_forward(model, hidden_states), baseline)


def test_invalid_transform_shape_removes_all_hooks(model, hook_site):
    with pytest.raises(ValueError, match="residual transform must preserve tensor shape"):
        with _INTERVENTIONS[hook_site](model, {0: lambda state: state, 1: lambda state: state[..., :2]}):
            _forward(model, torch.ones(1, 4, 3))
    _assert_no_hooks(model)


def test_context_body_exception_removes_all_hooks(model, hook_site):
    with pytest.raises(RuntimeError, match="context body failed"):
        with _INTERVENTIONS[hook_site](model, {0: lambda state: state, 1: lambda state: state}):
            raise RuntimeError("context body failed")
    _assert_no_hooks(model)
