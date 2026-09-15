"""Tests for the model-free orthonormal-basis helpers."""
from __future__ import annotations

import math

import pytest
import torch

from llm_bias.core.analysis.subspaces import basis_from_rows, project_onto_basis


def test_basis_from_rows_returns_transposed_cpu_fp32_contiguous_basis() -> None:
    basis = basis_from_rows([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dimension=4, rank=2)

    assert basis.shape == (4, 2)
    assert basis.dtype == torch.float32
    assert basis.device.type == "cpu"
    assert basis.is_contiguous()
    assert torch.equal(basis, torch.eye(4, dtype=torch.float32)[:, :2])


def test_basis_from_rows_supports_one_by_one_shape() -> None:
    basis = basis_from_rows([[1]], dimension=1, rank=1)
    assert basis.shape == (1, 1)
    assert basis.item() == 1.0


def test_basis_from_rows_uses_fp64_gram_and_does_not_reorthogonalize() -> None:
    rows = [[1.0, 0.0], [0.0, 1.0 + 2e-7]]
    basis = basis_from_rows(rows, dimension=2, rank=2, atol=1e-6)

    assert basis[1, 1].item() == pytest.approx(1.0000002, abs=1e-7)
    assert torch.allclose(
        basis.double().T @ basis.double(), torch.eye(2, dtype=torch.float64), atol=1e-6, rtol=0.0
    )


def test_basis_from_rows_rejects_invalid_shape_types_and_values() -> None:
    invalid_calls = [
        ([], 4, 2),
        ([[1.0, 0.0]], 4, 2),
        ([[1.0, 0.0], [0.0]], 2, 2),
        ([[True, 0.0]], 2, 1),
        ([["1.0", 0.0]], 2, 1),
        ([[float("nan"), 0.0]], 2, 1),
        ([[float("inf"), 0.0]], 2, 1),
        ([[1.0, 0.0]], 2, 0),
        ([[1.0, 0.0]], 1, 1),
    ]
    for rows, dimension, rank in invalid_calls:
        with pytest.raises(ValueError):
            basis_from_rows(rows, dimension=dimension, rank=rank)

    with pytest.raises(ValueError, match="dimension"):
        basis_from_rows([[1.0]], dimension=True, rank=1)
    with pytest.raises(ValueError, match="rank"):
        basis_from_rows([[1.0]], dimension=1, rank=False)
    with pytest.raises(ValueError, match="atol"):
        basis_from_rows([[1.0]], dimension=1, rank=1, atol=0.0)
    with pytest.raises(ValueError, match="atol"):
        basis_from_rows([[1.0]], dimension=1, rank=1, atol=float("inf"))


def test_basis_from_rows_rejects_nonorthogonal_rows() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        basis_from_rows([[1.0, 0.0], [1.0, 1.0]], dimension=2, rank=2)


def test_project_e1_e2_basis() -> None:
    basis = basis_from_rows([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dimension=4, rank=2)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    values_before = values.clone()

    result = project_onto_basis(values, basis)

    assert result.shape == values.shape
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.equal(result, torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert torch.equal(values, values_before)
    assert result.data_ptr() != values.data_ptr()


def test_projector_is_invariant_to_rotation_and_sign_flip() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.0, 8.0]])
    canonical = torch.eye(4, dtype=torch.float32)[:, :2]
    rotation = torch.tensor([[math.sqrt(0.5), -math.sqrt(0.5)], [math.sqrt(0.5), math.sqrt(0.5)]])
    rotated = canonical @ rotation
    sign_flipped = rotated.clone()
    sign_flipped[:, 1] *= -1

    expected = project_onto_basis(values, canonical)
    assert torch.allclose(project_onto_basis(values, rotated), expected, atol=1e-6, rtol=0.0)
    assert torch.allclose(project_onto_basis(values, sign_flipped), expected, atol=1e-6, rtol=0.0)


def test_projector_is_idempotent_with_fp32_error_bound() -> None:
    basis = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
    )
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.0, 8.0]])
    projected = project_onto_basis(values, basis)
    twice = project_onto_basis(projected, basis)
    assert torch.allclose(twice, projected, atol=1e-6, rtol=0.0)


def test_project_onto_basis_supports_batched_values_and_preserves_inputs() -> None:
    basis = torch.eye(4, dtype=torch.float32)[:, :2]
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    basis_before = basis.clone()
    values_before = values.clone()

    result = project_onto_basis(values, basis)

    assert result.shape == values.shape
    assert torch.equal(result, torch.tensor([[1.0, 2.0, 0.0, 0.0], [5.0, 6.0, 0.0, 0.0]]))
    assert torch.equal(values, values_before)
    assert torch.equal(basis, basis_before)
    assert result.is_contiguous()


def test_project_onto_basis_rejects_invalid_tensors() -> None:
    basis = torch.eye(4, dtype=torch.float32)[:, :2]
    valid = torch.ones(4)
    invalid_values = [
        torch.ones(2, 2, 2),
        torch.empty(0, 4),
        torch.ones(3),
        torch.ones(4, dtype=torch.int64),
        torch.ones(4, dtype=torch.complex64),
        torch.tensor([1.0, float("nan"), 0.0, 0.0]),
        torch.tensor([1.0, float("inf"), 0.0, 0.0]),
    ]
    for values in invalid_values:
        with pytest.raises(ValueError):
            project_onto_basis(values, basis)

    for bad_basis in (
        torch.ones(4),
        torch.empty(4, 0),
        torch.ones(4, 3),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.1]]),
        torch.tensor([[1.0, 0.0], [0.0, float("nan")], [0.0, 0.0], [0.0, 0.0]]),
        torch.ones(4, 2, dtype=torch.int64),
        torch.ones(4, 2, dtype=torch.complex64),
    ):
        with pytest.raises(ValueError):
            project_onto_basis(valid, bad_basis)


def test_project_onto_basis_rejects_fp32_overflow() -> None:
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    values = torch.tensor([3.5e38, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="overflow"):
        project_onto_basis(values, basis)
