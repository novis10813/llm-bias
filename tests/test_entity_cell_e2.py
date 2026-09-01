import json
from pathlib import Path

import pytest
import torch
from torch import nn

from llm_bias.entity_cell.attention_attribution import (
    SOURCE_GROUPS,
    compact_attribution_record,
    direct_logit_attribution,
    rank_attention_heads,
    reconstruct_attention_components,
    routing_label,
    selected_head_output_patch,
)
from llm_bias.entity_cell.cli import build_parser


class FakeNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = 1e-6

    def forward(self, value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)


class FakeGQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.scaling = 2**-0.5
        self.q_proj = nn.Linear(5, 16, bias=False)
        self.k_proj = nn.Linear(5, 4, bias=False)
        self.v_proj = nn.Linear(5, 4, bias=False)
        self.o_proj = nn.Linear(8, 5, bias=False)
        self.q_norm = FakeNorm(2)
        self.k_norm = FakeNorm(2)

    def forward(self, value, position_embeddings=None, attention_mask=None):
        batch, sequence, _ = value.shape
        query, gate = torch.chunk(self.q_proj(value).view(batch, sequence, 4, 4), 2, dim=-1)
        key = self.k_norm(self.k_proj(value).view(batch, sequence, 2, 2)).transpose(1, 2)
        val = self.v_proj(value).view(batch, sequence, 2, 2).transpose(1, 2)
        query = self.q_norm(query).transpose(1, 2)
        key = key.repeat_interleave(2, dim=1)
        val = val.repeat_interleave(2, dim=1)
        scores = query @ key.transpose(-1, -2) * self.scaling
        causal = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], -float("inf"))
        weights = scores.softmax(-1)
        output = (weights @ val).transpose(1, 2).reshape(batch, sequence, 8)
        output = output * gate.sigmoid().reshape(batch, sequence, 8)
        return self.o_proj(output), weights


def groups():
    result = {name: {"ranges": []} for name in SOURCE_GROUPS}
    for index, name in enumerate(SOURCE_GROUPS[:-1]):
        result[name]["ranges"] = [[index, index + 1]]
    return result


def test_fake_gqa_reconstructs_with_repeated_kv_gate_and_o_proj_slices():
    torch.manual_seed(4)
    attention = FakeGQA()
    hidden = torch.randn(1, 4, 5)
    clean, _ = attention(hidden)
    reconstruction = reconstruct_attention_components(
        attention, hidden, source_groups=groups(), query_position=3, clean_output=clean[:, 3]
    )
    assert reconstruction.additive
    assert reconstruction.max_abs_error < 1e-6
    assert len(reconstruction.head_group_vectors) == 4
    assert all(set(value) == set(SOURCE_GROUPS) for value in reconstruction.head_group_vectors.values())
    original = attention.q_proj(hidden).view(1, 4, 4, 4)
    with torch.no_grad():
        attention.q_proj.weight.mul_(0)
        attention.q_proj.weight[:8].fill_(1)
    changed = reconstruct_attention_components(attention, hidden, source_groups=groups(), query_position=3)
    assert not torch.allclose(changed.head_group_vectors[0]["identity_header"], reconstruction.head_group_vectors[0]["identity_header"])
    assert original.shape[-1] == 4


def test_dla_routing_selection_and_compact_boundary():
    attention = FakeGQA()
    hidden = torch.randn(1, 4, 5)
    reconstruction = reconstruct_attention_components(attention, hidden, source_groups=groups(), query_position=3)
    values = direct_logit_attribution(reconstruction, torch.ones(5))
    assert set(values[0]) == set(SOURCE_GROUPS)
    assert routing_label(1.0, 0.0) == "identity-dominant"
    assert routing_label(0.0, 1.0) == "instruction-dominant"
    assert routing_label(1.0, 1.0) == "mixed"
    records = []
    for ticker in ("A", "B"):
        for _ in range(3):
            records.append({"ticker": ticker, "layer": 11, "head": 0, "groups": {"identity_header": {"frozen_scale_margin": 2.0}, "instruction_context": {"frozen_scale_margin": 0.1}}})
    assert rank_attention_heads(records)[0]["selection_eligible"]
    row = compact_attribution_record(ticker="A", prompt_id="p", layer=11, head=0, dla={0: values[0]}, additivity=reconstruction, routing="identity-dominant")
    text = json.dumps(row)
    assert "tensor" not in text and "vector" not in text and "attention_matrix" not in text
    assert row["raw_runtime_payloads"] is False


def test_patch_isolates_one_o_projection_head_and_cleans_hook():
    attention = FakeGQA()
    hidden = torch.randn(1, 4, 5)
    with selected_head_output_patch(attention, head=1, donor_output=torch.full((2,), 3.0), query_position=3):
        patched = attention(hidden)[0]
    clean = attention(hidden)[0]
    assert not torch.allclose(patched[:, 3], clean[:, 3])
    assert torch.allclose(patched[:, :3], clean[:, :3])
    assert not attention.o_proj._forward_pre_hooks


def test_rejects_recurrent_layers_and_invalid_groups():
    class LinearLayer:
        block_type = "linear_attention"

    from llm_bias.entity_cell.attention_attribution import _attention_module, validate_attention_layers
    with pytest.raises(ValueError, match="full-attention"):
        _attention_module(LinearLayer())
    with pytest.raises(ValueError, match="L3"):
        validate_attention_layers([4])
    with pytest.raises(ValueError, match="cover"):
        reconstruct_attention_components(FakeGQA(), torch.randn(1, 4, 5), source_groups={name: {"ranges": []} for name in SOURCE_GROUPS}, query_position=3)


def test_e2_cli_preserves_e1_parse_contract():
    parser = build_parser()
    assert parser.parse_args(["run", "--prepared-dir", "p", "--model", "m", "--run-id", "r", "--stage", "e1-baseline"]).stages == ["e1-baseline"]
    args = parser.parse_args(["run", "--prepared-dir", "p", "--model", "m", "--run-id", "r", "--stage", "e2-attribution", "--e2-layers", "11", "15"])
    assert args.stages == ["e2-attribution"] and args.e2_layers == [11, 15]
    readout_args = parser.parse_args(["run", "--prepared-dir", "p", "--model", "m", "--run-id", "r", "--stage", "e2-readout", "--lens-path", "lens.pt", "--expected-lens-sha256", "a" * 64])
    assert readout_args.stages == ["e2-readout"] and readout_args.lens_path == Path("lens.pt")
    assert parser.parse_args(["analyze", "--run-root", "r", "--experiment", "e2"]).experiment == "e2"
