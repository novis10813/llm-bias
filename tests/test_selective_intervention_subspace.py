"""Unit tests for selective-intervention subspace mechanics (no model).

Covers the subspace removal transform (identity no-op, bit-exact
outside-position preservation, pure in-subspace removal, dtype handling,
fail-closed construction and forward guards), the frozen-seed random
orthonormal basis, digests, nearest-position grid alignment, and the
e-01 basis loader fail-closed paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_bias.selective_intervention.subspace import (
    align_grid,
    load_e01_basis,
    nearest_position_mapping,
    random_orthonormal_basis,
    subspace_removal_transform,
    tensor_sha256,
)


D = 16
K = 4


def _ortho(d: int = D, k: int = K) -> torch.Tensor:
    return torch.eye(d, dtype=torch.float32)[:, :k].clone()


# ── subspace removal transform ────────────────────────────────────────────────


def test_alpha_zero_returns_same_tensor():
    transform = subspace_removal_transform(_ortho(), 0.0, None, (2, 3))
    h = torch.randn(1, 6, D)
    out = transform(h)
    assert out is h  # structural no-op: core hook passes the tensor through


def test_positions_outside_span_bit_exact():
    torch.manual_seed(0)
    basis = _ortho()
    h = torch.randn(1, 8, D)
    transform = subspace_removal_transform(basis, 1.0, None, (2, 3, 4))
    out = transform(h)
    for p in (0, 1, 5, 6, 7):
        assert torch.equal(out[0, p], h[0, p])
    for p in (2, 3, 4):
        assert not torch.equal(out[0, p], h[0, p])


def test_pure_in_subspace_component_removed_exactly():
    torch.manual_seed(1)
    basis = _ortho()
    mu = torch.randn(D)
    v = basis[:, 0]
    h = (mu.unsqueeze(0) + 3.0 * v.unsqueeze(0)).unsqueeze(0)  # [1, 1, d]
    transform = subspace_removal_transform(basis, 1.0, {0: mu}, (0,))
    out = transform(h)
    assert torch.allclose(out[0, 0], mu, atol=1e-5)


def test_partial_strength_scales_removal():
    torch.manual_seed(2)
    basis = _ortho()
    mu = torch.randn(D)
    v = basis[:, 0]
    h = (mu.unsqueeze(0) + 4.0 * v.unsqueeze(0)).unsqueeze(0)
    transform = subspace_removal_transform(basis, 0.5, {0: mu}, (0,))
    out = transform(h)
    expected = mu + 2.0 * v
    assert torch.allclose(out[0, 0], expected, atol=1e-5)


def test_output_preserves_dtype():
    basis = _ortho()
    h = torch.randn(1, 4, D, dtype=torch.bfloat16)
    out = subspace_removal_transform(basis, 1.0, None, (1,))(h)
    assert out.dtype == torch.bfloat16
    assert out.shape == h.shape


def test_forward_guards():
    basis = _ortho()
    transform = subspace_removal_transform(basis, 1.0, None, (1,))
    with pytest.raises(ValueError, match="expects"):
        transform(torch.randn(3, D))
    with pytest.raises(ValueError, match="width"):
        transform(torch.randn(1, 4, D - 1))
    with pytest.raises(ValueError, match="outside the sequence"):
        transform(torch.randn(1, 1, D))


def test_construction_fail_closed():
    basis = _ortho()
    with pytest.raises(ValueError):
        subspace_removal_transform(basis, float("nan"), None, (1,))
    with pytest.raises(ValueError):
        subspace_removal_transform(basis, -0.5, None, (1,))
    with pytest.raises(ValueError):
        subspace_removal_transform(basis, 1.0, None, ())
    with pytest.raises(ValueError):
        subspace_removal_transform(basis, 1.0, None, (1, 1))
    with pytest.raises(ValueError):
        subspace_removal_transform(basis, 1.0, {2: torch.randn(D)}, (1,))
    with pytest.raises(ValueError):
        subspace_removal_transform(basis, 1.0, {1: torch.full((D,), float("inf"))}, (1,))
    bad = basis.clone()
    bad[:, 1] = bad[:, 1] + 0.5
    with pytest.raises(ValueError, match="orthonormal"):
        subspace_removal_transform(bad, 1.0, None, (1,))
    with pytest.raises(ValueError):
        subspace_removal_transform(torch.randn(D, K), 1.0, None, (1,))


# ── random basis + digests + alignment ────────────────────────────────────────


def test_random_orthonormal_basis_orthonormal_and_deterministic():
    a = random_orthonormal_basis(32, 8, seed=20260913)
    b = random_orthonormal_basis(32, 8, seed=20260913)
    c = random_orthonormal_basis(32, 8, seed=7)
    assert a.shape == (32, 8)
    assert torch.allclose(a.T @ a, torch.eye(8), atol=1e-5)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_random_basis_fail_closed():
    with pytest.raises(ValueError):
        random_orthonormal_basis(8, 9, seed=1)
    with pytest.raises(ValueError):
        random_orthonormal_basis(8, 0, seed=1)


def test_tensor_sha256():
    a = torch.randn(4, 8)
    b = a.clone()
    assert tensor_sha256(a) == tensor_sha256(b)
    assert tensor_sha256(a) != tensor_sha256(a + 1.0)
    with pytest.raises(ValueError):
        tensor_sha256(torch.full((2, 2), float("nan")))


def test_nearest_position_mapping_equal_length_offset_identity():
    mapping = nearest_position_mapping((10, 14), (20, 24))
    assert mapping == {20: 10, 21: 11, 22: 12, 23: 13}


def test_nearest_position_mapping_empty():
    assert nearest_position_mapping((0, 0), (0, 4)) == {}


def test_align_grid_rows():
    grid = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    aligned = align_grid(grid, (10, 13), (20, 23))
    assert set(aligned) == {20, 21, 22}
    torch.testing.assert_close(aligned[20], grid[0])
    torch.testing.assert_close(aligned[22], grid[2])


def test_align_grid_fail_closed():
    grid = torch.zeros(3, 4)
    with pytest.raises(ValueError):
        align_grid(grid, (0, 4), (0, 3))
    with pytest.raises(ValueError):
        align_grid(grid, (0, 0), (0, 3))


# ── e-01 basis loader ─────────────────────────────────────────────────────────


def _write_summary(tmp_path: Path, *, k: int = 8, d: int = 16) -> Path:
    basis = torch.eye(d, dtype=torch.float64)[:, :k].T  # [k, d] rows
    singular = torch.linspace(10.0, 1.0, k)
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "pca_basis_vectors": basis.tolist(),
                "pca_singular_values": singular.tolist(),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_e01_basis_roundtrip(tmp_path):
    path = _write_summary(tmp_path, k=8, d=16)
    basis, singular = load_e01_basis(path, expected_k=8)
    assert basis.shape == (16, 8)
    assert singular.shape == (8,)
    assert torch.allclose(basis.T @ basis, torch.eye(8), atol=1e-5)
    assert basis.dtype == torch.float32


@pytest.mark.parametrize(
    "mutate",
    [
        "missing_keys",
        "wrong_shape",
        "singular_increasing",
        "singular_zero",
        "non_orthonormal",
        "non_finite",
    ],
)
def test_load_e01_basis_fail_closed(tmp_path, mutate):
    path = _write_summary(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if mutate == "missing_keys":
        data = {"other": 1}
    elif mutate == "wrong_shape":
        data["pca_basis_vectors"] = data["pca_basis_vectors"][:7]
    elif mutate == "singular_increasing":
        data["pca_singular_values"] = list(reversed(data["pca_singular_values"]))
    elif mutate == "singular_zero":
        data["pca_singular_values"] = [0.0] * 8
    elif mutate == "non_orthonormal":
        data["pca_basis_vectors"] = [
            [x + 0.25 for x in row] if i == 0 else row
            for i, row in enumerate(data["pca_basis_vectors"])
        ]
    elif mutate == "non_finite":
        data["pca_basis_vectors"] = [[float("nan")] * 16 for _ in range(8)]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_e01_basis(path, expected_k=8)
