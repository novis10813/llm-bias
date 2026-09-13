"""Regression tests for block-level patch transforms (entity-to-dial).

Pure-tensor tests for the position/block transform semantics; a small real
Qwen3.5 fake model checks the transform wiring against the core hook
lifecycle. No checkpoint is loaded.
"""
from __future__ import annotations

import pytest
import torch

from llm_bias.core.inference.interventions import (
    record_block_states,
    residual_interventions,
)
from llm_bias.entity_to_dial.block_patch import (
    identity_mapping,
    make_block_transform,
    make_position_transform,
    nearest_position_mapping,
)


# ── position mapping ─────────────────────────────────────────────────────────


def test_nearest_position_mapping_identity_by_offset():
    assert nearest_position_mapping((2, 6), (2, 6)) == {2: 2, 3: 3, 4: 4, 5: 5}
    assert nearest_position_mapping((0, 4), (5, 9)) == {5: 0, 6: 1, 7: 2, 8: 3}


def test_nearest_position_mapping_nearest_for_unequal_lengths():
    # Every target position maps to the nearest normalized source position.
    mapping = nearest_position_mapping((0, 4), (0, 2))
    assert mapping == {0: 0, 1: 3}
    mapping = nearest_position_mapping((0, 2), (0, 4))
    assert mapping == {0: 0, 1: 0, 2: 1, 3: 1}
    assert nearest_position_mapping((0, 0), (0, 2)) == {}
    assert nearest_position_mapping((0, 2), (1, 1)) == {}


def test_identity_mapping_covers_the_span():
    assert identity_mapping((3, 7)) == {3: 3, 4: 4, 5: 5, 6: 6}


# ── position transform (clone/assign) ────────────────────────────────────────


def test_position_transform_self_source_is_exact_noop():
    torch.manual_seed(0)
    source = torch.randn(1, 8, 16)
    transform = make_position_transform(source, identity_mapping((1, 4)))
    patched = transform(source)
    assert patched is not source
    assert torch.equal(patched, source)


def test_position_transform_only_touches_mapped_positions():
    torch.manual_seed(1)
    source = torch.randn(1, 8, 16)
    target = torch.randn(1, 8, 16)
    mapping = {2: 0, 5: 7}
    patched = make_position_transform(target, mapping)(source)
    for pos in (0, 1, 3, 4, 6, 7):
        assert torch.equal(patched[:, pos], source[:, pos])
    assert torch.equal(patched[:, 2], target[:, 0])
    assert torch.equal(patched[:, 5], target[:, 7])


def test_position_transform_casts_to_target_dtype():
    target = torch.randn(1, 4, 8).bfloat16()
    source_fp32 = torch.randn(1, 4, 8)
    patched = make_position_transform(source_fp32, identity_mapping((0, 4)))(target)
    assert patched.dtype == torch.bfloat16
    assert torch.equal(patched, source_fp32.bfloat16())


# ── block transform (fp32 arithmetic) ────────────────────────────────────────


def _states(seed: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    pre = torch.randn(1, 8, 16)
    mid = pre + 2.0 * torch.randn(1, 8, 16)
    post = mid + 2.0 * torch.randn(1, 8, 16)
    return {"pre": pre, "mid": mid, "post": post}


def test_block_transform_fp32_exact_arithmetic():
    states = _states()
    mapping = {2: 0, 5: 7}
    for component, base_key, contribution in (
        ("mlp", "mid", "post"),
        ("attn", "pre", "mid"),
    ):
        transform = make_block_transform(
            captured_base=states[base_key],
            source_states=states,
            component=component,
            mapping=mapping,
        )
        base_point = "post" if component == "mlp" else "mid"
        patched = transform(states[base_point])
        # The transform computes the contribution once (full tensor, fp32)
        # and applies base[target_pos] + contribution[source_pos] per position.
        diff = (
            (states["post"].float() - states["mid"].float())
            if component == "mlp"
            else (states["mid"].float() - states["pre"].float())
        )
        for target_pos, source_pos in mapping.items():
            expected = states[base_key][:, target_pos].float() + diff[:, source_pos]
            assert torch.equal(patched[:, target_pos], expected)
        for pos in (0, 1, 3, 4, 6, 7):
            assert torch.equal(patched[:, pos], states[base_point][:, pos])


def test_block_transform_rejects_unknown_component():
    states = _states()
    with pytest.raises(ValueError, match="unknown block component"):
        make_block_transform(
            captured_base=states["pre"], source_states=states,
            component="residual", mapping={1: 1},
        )


def test_block_transform_bf16_self_source_within_one_ulp():
    # General regime: fl32(mid + fl32(post - mid)) may differ from post by
    # up to one bf16 ulp after the cast; it must never differ by more.
    torch.manual_seed(3)
    pre = torch.randn(1, 8, 16)
    mid = pre + 5.0 * torch.randn(1, 8, 16)
    post = mid + 5.0 * torch.randn(1, 8, 16)
    bf = {key: value.bfloat16() for key, value in (("pre", pre), ("mid", mid), ("post", post))}
    mapping = identity_mapping((1, 5))
    for component in ("mlp", "attn"):
        transform = make_block_transform(
            captured_base=bf["mid" if component == "mlp" else "pre"],
            source_states=bf,
            component=component,
            mapping=mapping,
        )
        base_point = bf["post" if component == "mlp" else "mid"]
        patched = transform(base_point)
        diff = (patched.float() - base_point.float()).abs()
        ulp = (
            torch.nextafter(base_point.float(), torch.full_like(base_point.float(), float("inf")))
            - base_point.float()
        ).abs()
        assert (diff[:, 1:5] <= ulp[:, 1:5]).all()
        for pos in (0, 5, 6, 7):
            assert torch.equal(patched[:, pos], base_point[:, pos])


def test_block_transform_bf16_sterbenz_regime_is_bit_exact():
    # When post is within a factor of 2 of mid per channel, post - mid is
    # exact (Sterbenz) and the sum returns post exactly: bit-exact after cast.
    torch.manual_seed(4)
    mid = 0.5 + 2.0 * torch.rand(1, 8, 16)
    post = mid * (0.75 + 0.5 * torch.rand(1, 8, 16))  # factor in [0.75, 1.25]
    pre = torch.randn(1, 8, 16)
    bf = {key: value.bfloat16() for key, value in (("pre", pre), ("mid", mid), ("post", post))}
    mapping = identity_mapping((1, 5))
    transform = make_block_transform(
        captured_base=bf["mid"], source_states=bf, component="mlp", mapping=mapping,
    )
    patched = transform(bf["post"])
    assert torch.equal(patched[:, 1:5], bf["post"][:, 1:5])


# ── wiring against the core hook lifecycle ───────────────────────────────────


def _qwen_fake(num_layers: int = 4) -> object:
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=num_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8,
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
            self.layers = raw.model.layers
            self.n_layers = num_layers
            self.input_device = "cpu"
            self._final_norm = raw.model.norm
            self._lm_head = raw.lm_head

        def forward(self, input_ids, attention_mask=None):
            return raw(input_ids, attention_mask=attention_mask)

    return _Model()


def test_block_transform_self_noop_through_residual_interventions():
    model = _qwen_fake()
    ids = torch.randint(2, 31, (1, 6))
    states = record_block_states(model, ids, [1])
    transform = make_position_transform(
        states[1]["post"], identity_mapping((1, 3))
    )
    with torch.no_grad():
        clean = model.forward(ids).logits
        with residual_interventions(model, {1: transform}):
            patched = model.forward(ids).logits
    assert torch.equal(patched, clean)
    assert not model.layers[1]._forward_hooks
    assert not model.layers[1].post_attention_layernorm._forward_pre_hooks


def test_block_transform_patch_changes_only_entity_positions():
    model = _qwen_fake()
    ids = torch.randint(2, 31, (1, 6))
    from llm_bias.core.inference.interventions import record_block_states

    states = record_block_states(model, ids, [1])
    other = torch.randn_like(states[1]["mid"]) + 3.0
    transform = make_block_transform(
        captured_base=states[1]["mid"],
        source_states={"pre": other, "mid": other, "post": other + 1.0},
        component="mlp",
        mapping=identity_mapping((2, 4)),
    )
    box: dict = {}

    def capture(_module, _inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        box["post"] = tensor

    with torch.no_grad(), residual_interventions(model, {1: transform}):
        # Register after the intervention so the hook sees the patched output.
        handle = model.layers[1].register_forward_hook(capture)
        try:
            model.forward(ids)
        finally:
            handle.remove()
    actual = box["post"]
    expected = states[1]["post"].clone()
    # Mirror the transform's fp32 arithmetic order exactly (identity mapping:
    # target position == source position).
    contribution = (other + 1.0).float() - other.float()
    expected[:, 2:4] = (states[1]["mid"].float() + contribution)[:, 2:4].to(expected.dtype)
    assert torch.equal(actual[:, 2:4], expected[:, 2:4])
    assert torch.equal(actual[:, :2], states[1]["post"][:, :2])
    assert torch.equal(actual[:, 4:], states[1]["post"][:, 4:])
