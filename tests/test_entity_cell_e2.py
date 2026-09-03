import ast
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from llm_bias import entity_cell as _entity_cell_pkg
from llm_bias.core.continuation_scoring import fp32_next_token_logits
from llm_bias.entity_cell import e2 as e2_module
from llm_bias.entity_cell.attention_attribution import (
    SOURCE_GROUPS,
    capture_attention_forward,
    compact_attribution_record,
    direct_logit_attribution,
    frozen_margin_direction,
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

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        batch, sequence, _ = hidden_states.shape
        query, gate = torch.chunk(self.q_proj(hidden_states).view(batch, sequence, 4, 4), 2, dim=-1)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, sequence, 2, 2)).transpose(1, 2)
        val = self.v_proj(hidden_states).view(batch, sequence, 2, 2).transpose(1, 2)
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


class _PairContractTokenizer:
    """Deterministic tokenizer honoring the continuation_scoring tuple contract."""

    def __init__(self):
        self.table = {"P": [1], "Pbuy": [1, 10], "Psell": [1, 20], "Pab": [1, 11, 12]}

    def __call__(self, text, add_special_tokens=False):
        class _Output:
            input_ids = self.table[text]

        return _Output()


def test_resolve_single_token_pair_unpacks_suffix_from_tuple_contract():
    from llm_bias.entity_cell.attention_attribution import resolve_single_token_pair

    assert resolve_single_token_pair(_PairContractTokenizer(), "P", "buy", "sell") == (10, 20)
    with pytest.raises(ValueError, match="single-token"):
        resolve_single_token_pair(_PairContractTokenizer(), "P", "ab", "sell")


def test_additivity_tolerance_is_bf16_scaled_and_enforced():
    # The frozen tolerance is documented in the proposal at bf16 relative
    # scale; pin the constants so a regression to FP32-scale values is caught.
    from llm_bias.entity_cell.attention_attribution import RECONSTRUCTION_ABSOLUTE_FLOOR, RECONSTRUCTION_RELATIVE_TOLERANCE
    assert RECONSTRUCTION_ABSOLUTE_FLOOR == 1e-3
    assert RECONSTRUCTION_RELATIVE_TOLERANCE == 2e-2
    torch.manual_seed(4)
    attention = FakeGQA()
    hidden = torch.randn(1, 4, 5)
    clean, _ = attention(hidden)
    # Noise at half the frozen relative scale must still count as additive.
    noisy = reconstruct_attention_components(
        attention, hidden, source_groups=groups(), query_position=3, clean_output=(clean * (1.0 + 0.5 * RECONSTRUCTION_RELATIVE_TOLERANCE))[:, 3]
    )
    assert noisy.additive
    # A structural-scale error must still fail closed.
    structural = reconstruct_attention_components(
        attention, hidden, source_groups=groups(), query_position=3, clean_output=(clean + 2.0 * clean.abs().max())[:, 3]
    )
    assert not structural.additive


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


def test_capture_handles_keyword_only_attention_call():
    torch.manual_seed(4)
    attention = FakeGQA()
    hidden = torch.randn(1, 4, 5)
    capture, handles = capture_attention_forward(attention)
    try:
        # Qwen3.5 decoder layers call self_attn entirely by keyword.
        attention(hidden_states=hidden, attention_mask=None, position_embeddings=None)
        assert torch.equal(capture.hidden_states, hidden)
        assert capture.output is not None
        # Positional calls must keep working as well.
        capture2, handles2 = capture_attention_forward(attention)
        try:
            attention(hidden, None, None)
            assert torch.equal(capture2.hidden_states, hidden)
        finally:
            for handle in handles2:
                handle.remove()
    finally:
        for handle in handles:
            handle.remove()
    assert not attention._forward_pre_hooks


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


def _imports_in_body(node):
    """Yield imported names under ``node`` without entering nested function/class scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                yield alias.asname or alias.name
        else:
            yield from _imports_in_body(child)


def test_run_e2_stage_branch_imports_do_not_shadow_module_names():
    """Regression: a stage-branch local import of a module-level name inside
    ``run_e2`` made that name function-local for the whole function, so the
    analyze stage crashed with UnboundLocalError whenever it ran without the
    branch containing the import (e.g. ``--stage analyze`` without
    ``--stage e2-readout``; run ``entity-cell-e2-discovery-v6``)."""
    func_tree = ast.parse(inspect.getsource(e2_module.run_e2))
    func = next(n for n in func_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_e2")
    local_imports = set(_imports_in_body(func))
    module_tree = ast.parse(inspect.getsource(e2_module))
    module_names = set()
    for statement in module_tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            module_names.update(alias.asname or alias.name for alias in statement.names)
    shadowed = local_imports & module_names
    assert not shadowed, f"stage-branch imports shadow module-level names in run_e2: {sorted(shadowed)}"


class _StandardRMSNorm(nn.Module):
    """Standard RMSNorm style: ``norm(x) * w``."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.full((dim,), 2.0))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * output.to(x.dtype)


class _Llama3RMSNorm(nn.Module):
    """Llama-3 / Qwen3.5 RMSNorm style: ``norm(x) * (1 + w)``."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, x):
        output = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).float()
        return (output * (1.0 + self.weight.float())).type_as(x)


@pytest.mark.parametrize("norm_cls", [_StandardRMSNorm, _Llama3RMSNorm])
def test_frozen_margin_direction_matches_true_margin_for_norm_style(norm_cls) -> None:
    generator = torch.Generator().manual_seed(7)
    norm = norm_cls(4)
    lm_head = nn.Linear(4, 10, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(10, 4, generator=generator))
    residual = torch.tensor([0.7, -0.2, 1.3, 0.4])
    direction = frozen_margin_direction(residual, norm, lm_head, 3, 7)
    normalized = norm(residual).float()
    true_margin = float(lm_head.weight[3].float() @ normalized - lm_head.weight[7].float() @ normalized)
    assert float(torch.dot(direction, residual)) == pytest.approx(true_margin, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("norm_cls", [_StandardRMSNorm, _Llama3RMSNorm])
def test_frozen_margin_direction_matches_core_fp32_tail(norm_cls) -> None:
    """The DLA direction must contract with the same final-norm tail as core."""
    norm = norm_cls(4)
    lm_head = nn.Linear(4, 10, bias=False)
    residual = torch.tensor([0.7, -0.2, 1.3, 0.4])
    direction = frozen_margin_direction(residual, norm, lm_head, 3, 7)
    stub = type("_Stub", (), {"_final_norm": norm, "_lm_head": lm_head})()
    logits = fp32_next_token_logits(stub, residual.unsqueeze(0))[0]
    assert float(torch.dot(direction, residual)) == pytest.approx(
        float(logits[3] - logits[7]), rel=1e-5, abs=1e-6
    )
