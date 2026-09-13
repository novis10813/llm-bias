"""Block-level (attention vs MLP) patch transforms.

Research semantics for docs/entity-to-dial/details/proposal-phase-abc.md §4.3: block
contribution replacement restricted to entity positions, with fp32
block arithmetic. Hook lifecycle comes from
llm_bias.core.inference.interventions. Pipeline self no-ops use direct
copies (clone/assign, bit-exact by construction); the arithmetic
precision property of ``make_block_transform`` is covered by unit
tests (fp32-exact by construction; bf16 cast within one ulp).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

ResidualTransform = Callable[[torch.Tensor], torch.Tensor]


def nearest_position_mapping(
    source_span: tuple[int, int], target_span: tuple[int, int]
) -> dict[int, int]:
    """Target-complete nearest-normalized mapping (2B causal-tracing contract).

    Every target span position maps to the nearest normalized source
    position; equal-length spans reduce to identity-by-offset.
    """
    src_start, src_end = source_span
    tgt_start, tgt_end = target_span
    n_src = src_end - src_start
    n_tgt = tgt_end - tgt_start
    if n_src <= 0 or n_tgt <= 0:
        return {}
    mapping: dict[int, int] = {}
    for target_offset in range(n_tgt):
        source_offset = (
            0 if n_tgt == 1 else round(target_offset * (n_src - 1) / (n_tgt - 1))
        )
        mapping[tgt_start + target_offset] = src_start + source_offset
    return mapping


def identity_mapping(span: tuple[int, int]) -> dict[int, int]:
    return {pos: pos for pos in range(span[0], span[1])}


def make_position_transform(source_residual: torch.Tensor, mapping: Mapping[int, int]) -> ResidualTransform:
    """Clone/assign position transform (2B make_span_transform semantics).

    The transform clones and assigns the same values for a self-source
    mapping, so a self-source patch is an exact no-op.
    """
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        for target_pos, source_pos in mapping.items():
            patched[:, target_pos, :] = source_residual[:, source_pos, :].to(patched.dtype)
        return patched

    return transform


def make_block_transform(
    *,
    captured_base: torch.Tensor,
    source_states: Mapping[str, torch.Tensor],
    component: str,
    mapping: Mapping[int, int],
) -> ResidualTransform:
    """Block-contribution replacement transform with fp32 arithmetic.

    component "mlp": applied at the post-block point; entity positions of
    the block output become ``base_target + (post_source - mid_source)``.
    component "attn": applied at the mid point (post_attention_layernorm
    input); entity positions become ``base_target + (mid_source - pre_source)``.

    ``captured_base`` is the target's clean mid (mlp) or pre (attn) state
    from the clean target forward; ``source_states`` holds the source
    forward's pre/mid/post states for one layer. Differences and sums are
    computed in FP32 and cast back to the tensor dtype. A self-source
    patch reproduces the original state exactly in FP32 and, after a bf16
    cast, within one ulp (bit-exact whenever base and result are within a
    factor of 2 in magnitude per channel, by Sterbenz's lemma).
    """
    if component not in ("mlp", "attn"):
        raise ValueError(f"unknown block component: {component}")
    pre = source_states["pre"].float()
    mid = source_states["mid"].float()
    post = source_states["post"].float()
    contribution = (post - mid) if component == "mlp" else (mid - pre)

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        for target_pos, source_pos in mapping.items():
            value = captured_base[:, target_pos, :].float() + contribution[:, source_pos, :]
            patched[:, target_pos, :] = value.to(tensor.dtype)
        return patched

    return transform
