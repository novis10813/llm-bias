"""Phase E patch transforms (proposal-phase-e.md §2, §5).

Four transform families at the post-block residual point, restricted to
instruction-span positions via a target-complete mapping (equal-length
spans reduce to identity-by-offset):

- ``make_joint_transform``: add the joint block delta
  ``(post_s − pre_s) − (post_t − pre_t)`` (FP32 arithmetic, cast back).
  The joint state equals ``post_s − (pre_s − pre_t)`` — the 2B full
  residual swap minus the pre-L state difference.
- ``make_full_transform``: exact clone/assign of the source post state
  (bit-exact 2B ``make_span_transform`` semantics; self-source is an
  exact no-op).
- ``make_projected_transform``: add the projection of the state
  difference ``post_s − post_t`` onto the first k PCA right singular
  vectors (FP32 arithmetic, cast back).
- ``dial_channel_transplant``: position-restricted addition of the real
  source/target dial-channel value difference at the L15 MLP
  down-projection input (the dial's native 9216-dim space).

State values are transient: callers capture, compute FP32 deltas, and
apply; nothing raw is persisted.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.inference.interventions import ResidualTransform, residual_interventions
from llm_bias.core.inference.mlp import dense_down_projection

from .block_patch import make_position_transform


def _validate_mapping(mapping: Mapping[int, int]) -> None:
    if not mapping:
        raise ValueError("empty position mapping")
    for target_pos, source_pos in mapping.items():
        if not isinstance(target_pos, int) or not isinstance(source_pos, int):
            raise ValueError("mapping positions must be ints")


def make_delta_transform(deltas: Mapping[int, torch.Tensor]) -> ResidualTransform:
    """Clone/add transform: mapped target positions gain a precomputed FP32 row.

    ``deltas`` maps target position → ``[d_model]`` FP32 row. Unmapped
    positions are untouched. A self-source construction (zero rows) is
    bit-exact (adding 0.0 in the tensor dtype is exact).
    """
    if not deltas:
        raise ValueError("empty delta mapping")
    rows = {}
    for target_pos, row in deltas.items():
        row = row.detach().float()
        if row.ndim != 1:
            raise ValueError("delta rows must be 1-D")
        if not torch.isfinite(row).all():
            raise ValueError(f"non-finite delta row at position {target_pos}")
        rows[target_pos] = row

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        for target_pos, row in rows.items():
            if target_pos >= tensor.shape[1]:
                raise ValueError(f"delta position {target_pos} outside the tensor")
            if row.shape[0] != tensor.shape[-1]:
                raise ValueError("delta row width mismatch")
            row = row.to(patched.device)
            patched[:, target_pos, :] = (patched[:, target_pos, :].float() + row).to(patched.dtype)
        return patched

    return transform


def aligned_block_delta(
    source_pre: torch.Tensor,
    source_post: torch.Tensor,
    target_pre: torch.Tensor,
    target_post: torch.Tensor,
    *,
    mapping: Mapping[int, int],
) -> dict[int, torch.Tensor]:
    """FP32 joint block delta per mapped target position.

    ``Δs[p] = (post_s[q] − pre_s[q]) − (post_t[p] − pre_t[p])`` for
    ``mapping[p] = q``. Sum of the attention and MLP block deltas; the
    mid state is not needed.
    """
    _validate_mapping(mapping)
    out: dict[int, torch.Tensor] = {}
    for target_pos, source_pos in mapping.items():
        src = source_post[0, source_pos].float() - source_pre[0, source_pos].float()
        tgt = target_post[0, target_pos].float() - target_pre[0, target_pos].float()
        out[target_pos] = (src - tgt)
    return out


def make_joint_transform(
    source_pre: torch.Tensor,
    source_post: torch.Tensor,
    target_pre: torch.Tensor,
    target_post: torch.Tensor,
    *,
    mapping: Mapping[int, int],
) -> ResidualTransform:
    """Joint dual-block patch (post point): ``post_t += Δs`` (FP32 → cast)."""
    return make_delta_transform(
        aligned_block_delta(source_pre, source_post, target_pre, target_post, mapping=mapping)
    )


def make_full_transform(source_post: torch.Tensor, *, mapping: Mapping[int, int]) -> ResidualTransform:
    """Exact swap of source post state (bit-exact 2B semantics)."""
    _validate_mapping(mapping)
    return make_position_transform(source_post, mapping)


def state_difference_rows(
    source_post: torch.Tensor,
    target_post: torch.Tensor,
    *,
    mapping: Mapping[int, int],
) -> dict[int, torch.Tensor]:
    """FP32 state difference ``post_s[q] − post_t[p]`` per mapped target position."""
    _validate_mapping(mapping)
    return {
        target_pos: (source_post[0, source_pos].float() - target_post[0, target_pos].float())
        for target_pos, source_pos in mapping.items()
    }


def stack_state_difference(
    rows: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    """``{position: [d]}`` rows → ``[P, d]`` FP32 tensor, sorted by position."""
    if not rows:
        raise ValueError("empty state difference")
    positions = sorted(rows)
    width = rows[positions[0]].shape[0]
    for pos in positions:
        if rows[pos].shape[0] != width:
            raise ValueError("state difference rows have inconsistent widths")
    return torch.stack([rows[pos] for pos in positions], dim=0)


def pca_state_directions(
    delta_vectors: Sequence[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k RIGHT singular vectors of the stacked direction-delta matrix.

    ``delta_vectors[i]`` is ``[P_i, d]`` FP32 (one direction's
    ``Δs_L15``). Returns ``(V, s)`` with ``V: [d, k]`` orthonormal state
    directions and ``s: [k]`` singular values. Left singular vectors live
    in the direction×position space and are not state-space directions;
    only the right vectors are returned.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not delta_vectors:
        raise ValueError("delta_vectors must be nonempty")
    z = torch.cat([d.detach().float().cpu() for d in delta_vectors], dim=0)
    if z.ndim != 2:
        raise ValueError("each delta vector must be [P, d]")
    if not torch.isfinite(z).all():
        raise ValueError("non-finite state difference")
    n_rows, width = z.shape
    if k > min(n_rows, width):
        raise ValueError(f"k={k} exceeds the matrix rank bound {min(n_rows, width)}")
    _, s, vh = torch.linalg.svd(z, full_matrices=False)
    basis = vh[:k].T.contiguous()
    return basis, s[:k].contiguous()


def project_delta(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project ``delta: [P, d]`` onto ``span(basis: [d, k])`` (FP32)."""
    if delta.ndim != 2 or basis.ndim != 2:
        raise ValueError("delta must be [P, d] and basis [d, k]")
    if delta.shape[1] != basis.shape[0]:
        raise ValueError("delta/basis width mismatch")
    return (delta.float() @ basis.float()) @ basis.float().T


def make_projected_transform(
    source_post: torch.Tensor,
    target_post: torch.Tensor,
    basis: torch.Tensor,
    *,
    mapping: Mapping[int, int],
) -> ResidualTransform:
    """Add the projection of ``post_s − post_t`` onto ``basis`` (FP32 → cast)."""
    rows = state_difference_rows(source_post, target_post, mapping=mapping)
    delta = stack_state_difference(rows).float().cpu()
    projected = project_delta(delta, basis.float().cpu())
    positions = sorted(rows)
    return make_delta_transform({pos: projected[i] for i, pos in enumerate(positions)})


@contextmanager
def dial_channel_transplant(
    model: Any,
    layer: int,
    channel: int,
    deltas: Mapping[int, float],
) -> Iterator[None]:
    """Position-restricted dial channel transplant at the down-projection input.

    Adds ``deltas[position]`` (FP32, native units) to ``channel`` at the
    given target positions only; all other positions/channels are
    untouched. All-zero deltas are a bit-exact no-op (the hook returns
    ``None``). Strict single fire: the hook raises on a second firing
    (fail-closed against unexpected re-forwards).
    """
    if not 0 <= layer < len(model.layers):
        raise ValueError(f"dial layer {layer} out of range")
    if channel < 0:
        raise ValueError("dial channel must be nonnegative")
    nonzero: dict[int, float] = {}
    for pos, value in deltas.items():
        value = float(value)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"non-finite dial delta at position {pos}")
        if value != 0.0:
            nonzero[int(pos)] = value

    if not nonzero:
        yield
        return

    fires = 0

    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[tuple[Any, ...], Any] | None:
        nonlocal fires
        fires += 1
        if fires > 1:
            raise RuntimeError("dial transplant hook fired twice")
        values = args[0]
        if not torch.is_tensor(values) or values.ndim != 3 or values.shape[0] != 1:
            raise ValueError("MLP input must be [1, sequence, width]")
        if channel >= values.shape[-1]:
            raise ValueError("dial channel out of range")
        result = values.clone()
        for pos, delta in nonzero.items():
            if pos < 0 or pos >= values.shape[1]:
                raise ValueError(f"dial position {pos} outside the sequence")
            result[0, pos, channel] = (result[0, pos, channel].float() + delta).to(result.dtype)
        return (result, *args[1:])

    handle = dense_down_projection(model.layers[layer]).register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def dial_footprint_direction(model: Any, layer: int, channel: int) -> torch.Tensor:
    """Normalized residual-space footprint of one down-projection input channel.

    The channel's contribution direction is the WEIGHT COLUMN
    ``weight[:, channel]`` (shape ``[d_model]``), normalized in FP32.
    Descriptive only: other channels may write the same residual
    direction (confound; protocol §2).
    """
    if not 0 <= layer < len(model.layers):
        raise ValueError(f"dial layer {layer} out of range")
    weight = dense_down_projection(model.layers[layer]).weight
    if channel < 0 or channel >= weight.shape[1]:
        raise ValueError("dial channel out of range")
    vector = weight[:, channel].detach().float().cpu()
    norm = vector.norm()
    if not torch.isfinite(norm) or norm <= 0.0:
        raise ValueError("dial footprint vector is zero or non-finite")
    return (vector / norm).contiguous()


# ── Phase F: projected transplant, directional push, dual-hook, basis ──────


def make_projected_transplant(
    delta: torch.Tensor,
    basis: torch.Tensor,
    positions: Sequence[int],
) -> ResidualTransform:
    """Low-level projected transplant (Phase F §5).

    ``delta`` is ``[P, d]`` FP32 (row i ↔ ``positions[i]``); ``basis`` is
    ``[d, k]`` orthonormal state directions. Patches each position with
    ``V_k V_kᵀ delta[i]`` (FP32 projection → cast back). For k=1 with the
    Phase E basis, the rows are bit-exact equal to Phase E's
    ``project_delta`` + ``make_delta_transform`` path (unit-tested).
    A zero ``delta`` is a bit-exact no-op.
    """
    if delta.ndim != 2 or delta.shape[0] != len(positions):
        raise ValueError("delta must be [P, d] matching the position count")
    if basis.ndim != 2 or basis.shape[0] != delta.shape[1]:
        raise ValueError("basis must be [d, k] with d matching delta width")
    if not positions:
        raise ValueError("positions must be nonempty")
    projected = project_delta(delta.float().cpu(), basis.float().cpu())
    return make_delta_transform({pos: projected[i] for i, pos in enumerate(positions)})


def directional_push_transform(
    push_vector: torch.Tensor,
    positions: Sequence[int],
) -> ResidualTransform:
    """Constant-across-positions residual push (Phase F2 §2).

    Adds the same ``push_vector`` (``[d]`` FP32) at every given position;
    all other positions are untouched. A zero vector is a bit-exact no-op.
    Semantics differ from the per-position transplant: the push tests a
    direction's intrinsic loading on a neutral prompt (protocol §2).
    """
    vector = push_vector.detach().float().cpu()
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError("push vector must be a nonempty 1-D tensor")
    if not torch.isfinite(vector).all():
        raise ValueError("non-finite push vector")
    if not positions:
        raise ValueError("positions must be nonempty")
    return make_delta_transform({int(pos): vector for pos in positions})


@contextmanager
def dual_hook_interventions(
    model: Any,
    layer: int,
    residual_transform: ResidualTransform | None,
    dial_channel: int,
    dial_deltas: Mapping[int, float],
) -> Iterator[dict[str, int]]:
    """Dual-hook combined patch (Phase F §2, §5).

    Nested composite: core ``residual_interventions`` (post-block layer
    forward hook) over ``dial_channel_transplant`` (down-proj
    forward_pre_hook) — both hooks fire within the caller's single
    forward, with bit-exact Phase E/E2b hook semantics (no re-implementation).
    The residual transform is wrapped with a strict single-fire counter;
    the dial side inherits ``dial_channel_transplant``'s strictness.
    Exception-safe removal is guaranteed by the two nested context
    managers individually.

    Yields ``fires`` = ``{"residual": int, "dial": int}``: the residual
    count is the actual firing count; ``dial`` is 1 when a non-zero dial
    delta registered the hook (strict single fire then guarantees exactly
    one firing for a completed forward) and 0 otherwise.
    """
    fires: dict[str, int] = {"residual": 0, "dial": 0}
    dial_active = bool(dial_deltas) and any(float(v) != 0.0 for v in dial_deltas.values())
    with dial_channel_transplant(model, layer, dial_channel, dial_deltas) as _:
        if residual_transform is not None:

            def wrapped(tensor: torch.Tensor) -> torch.Tensor:
                fires["residual"] += 1
                if fires["residual"] > 1:
                    raise RuntimeError("dual-hook residual transform fired twice")
                return residual_transform(tensor)

            with residual_interventions(model, {layer: wrapped}):
                if dial_active:
                    fires["dial"] = 1
                yield fires
        else:
            if dial_active:
                fires["dial"] = 1
            yield fires


def load_pca_basis(
    summary_path: str | Path,
    *,
    expected_k: int = 16,
    orthonormal_atol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load + validate the persisted Phase E PCA basis (Phase F §4.1).

    Reads ``pca_basis_vectors`` (``[k, d]`` rows) and ``pca_singular_values``
    (``[k]``) from a Phase E ``analyze/summary.json`` and fails closed on:
    wrong shape, non-finite values, non-orthonormal basis (FP64 row-Gram
    check, atol ``orthonormal_atol``), or non-positive / non-decreasing
    singular values. Returns ``(basis: [d, k] FP32 CPU, singular: [k] FP32
    CPU)`` — the in-memory convention of this codebase (right singular
    vectors as columns, matching Phase E ``pca_state_directions``).
    """
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    vectors = data.get("pca_basis_vectors")
    values = data.get("pca_singular_values")
    if not isinstance(vectors, list) or not isinstance(values, list):
        raise ValueError("summary is missing pca_basis_vectors / pca_singular_values")
    basis = torch.tensor(vectors, dtype=torch.float64)
    singular = torch.tensor(values, dtype=torch.float64)
    if basis.ndim != 2 or basis.shape[0] != expected_k:
        raise ValueError(f"pca_basis_vectors must be [{expected_k}, d], got {tuple(basis.shape)}")
    if singular.ndim != 1 or singular.shape[0] != expected_k:
        raise ValueError(f"pca_singular_values must be [{expected_k}], got {tuple(singular.shape)}")
    if not torch.isfinite(basis).all():
        raise ValueError("pca_basis_vectors contain non-finite values")
    if not torch.isfinite(singular).all() or bool((singular <= 0).any()):
        raise ValueError("pca_singular_values must be positive and finite")
    if bool((singular[1:] > singular[:-1] + 1e-9).any()):
        raise ValueError("pca_singular_values must be decreasing")
    gram = basis @ basis.T
    if not torch.allclose(gram, torch.eye(expected_k, dtype=torch.float64), atol=orthonormal_atol):
        raise ValueError(f"pca_basis_vectors are not orthonormal (atol {orthonormal_atol})")
    return basis.T.contiguous().float(), singular.float().contiguous()
