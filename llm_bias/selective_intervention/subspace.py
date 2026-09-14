"""Subspace removal transform, calibration centers, and basis utilities.

Protocol: docs/selective-intervention/details/proposal-v1.md §4. For a selected
position set P the transform maps, at the intervened layer's post-block
residual ``h`` (``[1, sequence, d]``):

    h[p] <- h[p] - alpha * V (V^T (h[p] - mu[p]))      for p in P

with projection arithmetic in FP32 (cast back to the tensor dtype) and
every position outside P left bit-exact. ``V`` is the entity-difference
subspace (e-01 k=8 basis, orthonormal columns) or the frozen-seed random
control basis; ``mu`` is the entity-contrast center grid, the
identity-stripped center grid, or zero. Values are transient; callers
persist digests only.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.inference.interventions import record_block_states

from .scoring import scoring_ids

ResidualTransform = Any  # Callable[[torch.Tensor], torch.Tensor] (core alias)


# ── basis loading and generation ──────────────────────────────────────────────


def load_e01_basis(
    summary_path: str | Path,
    *,
    expected_k: int = 16,
    orthonormal_atol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load + validate the persisted Phase E PCA basis (fail-closed).

    Reads ``pca_basis_vectors`` (``[k, d]`` rows) and
    ``pca_singular_values`` (``[k]``) from a Phase E
    ``analyze/summary.json``. Fails closed on wrong shape, non-finite
    values, non-orthonormal basis (FP64 row-Gram), or non-positive /
    non-decreasing singular values. Returns ``([d, k] FP32 CPU,
    [k] FP32 CPU)`` (right singular vectors as columns).

    (Self-contained copy of the entity-to-dial loader; experiment
    packages must not import each other.)
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


def random_orthonormal_basis(d: int, k: int, seed: int) -> torch.Tensor:
    """Frozen-seed random orthonormal basis ``[d, k]`` (specificity control).

    Gaussian ``[d, k]`` under a manual-seed generator → QR; deterministic
    for a given (d, k, seed). Fails closed if the result is not
    orthonormal.
    """
    if not 0 < k <= d:
        raise ValueError(f"need 0 < k <= d, got k={k}, d={d}")
    generator = torch.Generator().manual_seed(int(seed))
    sample = torch.randn(d, k, generator=generator)
    q, _ = torch.linalg.qr(sample)
    basis = q[:, :k].contiguous()
    gram = basis.T @ basis
    if not torch.allclose(gram, torch.eye(k), atol=1e-5):
        raise ValueError("random basis is not orthonormal")
    return basis


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 over the FP32 bytes of a CPU tensor (provenance digest)."""
    cpu = tensor.detach().cpu().contiguous()
    if not torch.isfinite(cpu).all():
        raise ValueError("digest input contains non-finite values")
    return hashlib.sha256(cpu.float().numpy().tobytes()).hexdigest()


# ── nearest-position center alignment ───────────────────────────────────────


def nearest_position_mapping(
    source_span: tuple[int, int], target_span: tuple[int, int]
) -> dict[int, int]:
    """Target-complete nearest-normalized span mapping.

    Every target span position maps to the nearest normalized source
    position; equal-length spans reduce to identity-by-offset.
    (Self-contained copy of the 2B causal-tracing contract helper.)
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


def align_grid(
    grid: torch.Tensor, grid_span: tuple[int, int], target_span: tuple[int, int]
) -> dict[int, torch.Tensor]:
    """Map a center grid (rows ordered by ``grid_span``) onto target positions.

    Returns ``{target_position: row_vector}`` via nearest-normalized
    alignment (fail-closed on empty spans).
    """
    n_grid = grid_span[1] - grid_span[0]
    if grid.shape[0] != n_grid:
        raise ValueError(f"grid has {grid.shape[0]} rows, span length {n_grid}")
    mapping = nearest_position_mapping(grid_span, target_span)
    if not mapping:
        raise ValueError("empty span alignment")
    return {t: grid[g - grid_span[0]] for t, g in mapping.items()}


# ── the removal transform ─────────────────────────────────────────────────────


def subspace_removal_transform(
    basis: torch.Tensor,
    alpha: float,
    centers: Mapping[int, torch.Tensor] | None,
    positions: Sequence[int],
) -> ResidualTransform:
    """Build the position-restricted subspace removal transform.

    ``basis``: ``[d, k]`` orthonormal columns (any float dtype; cast to
    FP32). ``centers``: per-position center vectors (``[d]``); ``None``
    means the zero center (full projection). ``alpha = 0`` yields the
    identity transform (returns the input tensor unchanged, so the core
    hook treats it as a structural no-op). Fail-closed: non-finite
    alpha, empty/duplicate positions, non-finite centers, center keys
    that do not match ``positions`` exactly.
    """
    if basis.ndim != 2:
        raise ValueError(f"basis must be [d, k], got {tuple(basis.shape)}")
    d, k = basis.shape
    if not 0 < k <= d:
        raise ValueError(f"basis shape invalid: {tuple(basis.shape)}")
    gram = (basis.T @ basis).float()
    if not torch.allclose(gram, torch.eye(k, dtype=gram.dtype, device=gram.device), atol=1e-4):
        raise ValueError("basis columns are not orthonormal")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
    pos = tuple(sorted({int(p) for p in positions}))
    if not pos:
        raise ValueError("positions must be nonempty")
    if len(pos) != len(positions):
        raise ValueError("positions contain duplicates")

    V = basis.float().contiguous()
    if centers is None:
        mu: torch.Tensor | None = None
    else:
        if set(centers) != set(pos):
            raise ValueError("center keys must match positions exactly")
        mu = torch.stack([torch.as_tensor(centers[p], dtype=torch.float32) for p in pos])
        if mu.shape[1] != d:
            raise ValueError(f"center width {mu.shape[1]} != basis width {d}")
        if not torch.isfinite(mu).all():
            raise ValueError("centers contain non-finite values")

    def transform(h: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(h) or h.ndim != 3 or h.shape[0] != 1:
            raise ValueError("subspace removal expects [1, sequence, d]")
        if h.shape[2] != d:
            raise ValueError(f"hidden width {h.shape[2]} != basis width {d}")
        seq = h.shape[1]
        if any(p < 0 or p >= seq for p in pos):
            raise ValueError(f"removal position outside the sequence (len {seq})")
        if alpha == 0.0:
            return h  # structural no-op: the core hook passes the tensor through
        out = h.clone()
        idx = torch.tensor(pos, device=h.device)
        h_sel = h[0, idx].float()  # [n, d] FP32
        centered = h_sel - (mu.to(device=h.device) if mu is not None else 0.0)
        v_dev = V.to(device=h.device)
        removal = (centered @ v_dev) @ v_dev.T  # [n, d]
        out[0, idx] = (h_sel - alpha * removal).to(h.dtype)
        return out

    return transform


# ── calibration centers ───────────────────────────────────────────────────────


def calibration_centers(
    model: Any,
    tokenizer: Any,
    named_rows: Sequence[dict],
    ref_row: dict,
    anon_row: dict,
    *,
    layers: Sequence[int],
    anon_layer: int,
    ref_seq_len: int,
    device: Any,
) -> dict[str, Any]:
    """L15-window calibration: per-position cloud means + anonymous grid.

    Captures the post-block states of the named sequences (one variant
    each) at ``layers`` and the anonymous sequence at ``anon_layer``;
    returns ``{"mu_bar": {layer: [n_ref, d] FP32 CPU}, "mu_anon":
    [n_anon, d] FP32 CPU, "mu_full_ref": [n_ref_seq, d] FP32 CPU,
    "ref_span": (s, e), "anon_span": (s, e), "n_companies": int}``.

    ``mu_full_ref`` is the full-sequence cloud grid at ``anon_layer`` on
    the reference sequence's positions (nearest-normalized alignment per
    company; the full grid shares the same named-sequence forwards as the
    span grid, adding no forwards).

    Fail-closed: every named span must equal the reference span length
    (1:1 alignment contract); all captured states must be finite.
    Transient states are discarded; only the reduced means are kept.
    """
    ref_span = (int(ref_row["instruction_span"][0]), int(ref_row["instruction_span"][1]))
    n_ref = ref_span[1] - ref_span[0]
    if n_ref <= 0:
        raise ValueError("reference instruction span is empty")
    ref_seq_len = int(ref_seq_len)
    if ref_seq_len <= 0:
        raise ValueError("reference sequence length must be positive")
    layers = sorted({int(layer) for layer in layers})
    anon_layer = int(anon_layer)
    final_layer = int(model.n_layers) - 1
    if any(layer < 0 or layer > final_layer for layer in layers) or not 0 <= anon_layer <= final_layer:
        raise ValueError(f"calibration layer out of range (0..{final_layer})")
    if anon_layer not in layers:
        raise ValueError("anon_layer must be one of the calibration layers")

    accum: dict[int, torch.Tensor] = {}
    accum_full: torch.Tensor | None = None
    hidden: int | None = None
    for row in named_rows:
        span = (int(row["instruction_span"][0]), int(row["instruction_span"][1]))
        if span[1] - span[0] != n_ref:
            raise ValueError(
                f"{row['ticker']}: instruction span length {span[1] - span[0]} "
                f"!= reference length {n_ref} (1:1 alignment required)"
            )
        ids = torch.tensor(
            [scoring_ids(tokenizer, row["formatted"])], dtype=torch.long, device=device
        )
        seq_len = ids.shape[1]
        with torch.no_grad():
            states = record_block_states(model, ids, layers)
        for layer in layers:
            h = states[layer]["post"][0].float().cpu()  # [seq, d]
            seg = h[span[0] : span[1]]
            if hidden is None:
                hidden = h.shape[1]
                for l in layers:
                    accum[l] = torch.zeros(n_ref, hidden)
                if anon_layer in layers:
                    accum_full = torch.zeros(ref_seq_len, hidden)
            elif h.shape[1] != hidden:
                raise ValueError("hidden width changed across captures")
            if not torch.isfinite(h).all():
                raise ValueError(f"non-finite capture for {row['ticker']}")
            accum[layer] += seg
            if layer == anon_layer and accum_full is not None:
                for j in range(ref_seq_len):
                    p = round(j * (seq_len - 1) / (ref_seq_len - 1)) if ref_seq_len > 1 else 0
                    accum_full[j] += h[p]

    mu_bar = {layer: (accum[layer] / len(named_rows)) for layer in layers}
    if accum_full is None:
        raise RuntimeError("full-grid accumulator was not initialized")
    mu_full_ref = accum_full / len(named_rows)

    anon_span = (int(anon_row["instruction_span"][0]), int(anon_row["instruction_span"][1]))
    if anon_span[1] <= anon_span[0]:
        raise ValueError("anonymous instruction span is empty")
    ids = torch.tensor(
        [scoring_ids(tokenizer, anon_row["formatted"])], dtype=torch.long, device=device
    )
    with torch.no_grad():
        anon_states = record_block_states(model, ids, [anon_layer])
    mu_anon = anon_states[anon_layer]["post"][0, anon_span[0] : anon_span[1]].float().cpu()
    if not torch.isfinite(mu_anon).all():
        raise ValueError("non-finite anonymous center grid")

    return {
        "mu_bar": mu_bar,
        "mu_anon": mu_anon,
        "mu_full_ref": mu_full_ref,
        "ref_span": ref_span,
        "anon_span": anon_span,
        "n_companies": len(named_rows),
    }
