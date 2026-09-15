"""Pure CPU helpers for validating and projecting onto orthonormal bases."""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import torch


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an int, got {value!r}")
    return int(value)


def _validate_atol(atol: object) -> float:
    if isinstance(atol, bool) or not isinstance(atol, Real):
        raise ValueError(f"atol must be a finite real number, got {atol!r}")
    value = float(atol)
    if not math.isfinite(value) or not 0.0 < value <= 1e-3:
        raise ValueError(f"atol must satisfy 0 < atol <= 1e-3, got {atol!r}")
    return value


def _rows_tensor(
    rows: Sequence[Sequence[float]], *, dimension: int, rank: int
) -> torch.Tensor:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("rows must be a two-dimensional sequence")
    if len(rows) != rank:
        raise ValueError(f"rows must have {rank} rows, got {len(rows)}")

    parsed: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"rows[{row_index}] must be a sequence")
        if len(row) != dimension:
            raise ValueError(
                f"rows[{row_index}] must have dimension {dimension}, got {len(row)}"
            )
        parsed_row: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"rows[{row_index}][{column_index}] must be a finite real number"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"rows[{row_index}][{column_index}] must be finite"
                )
            parsed_row.append(numeric)
        parsed.append(parsed_row)
    return torch.tensor(parsed, dtype=torch.float64, device="cpu")


def _validate_orthonormal_rows(rows: torch.Tensor, *, atol: float) -> None:
    gram = rows @ rows.T
    expected = torch.eye(rows.shape[0], dtype=torch.float64, device="cpu")
    if not torch.isfinite(gram).all() or not torch.allclose(
        gram, expected, atol=atol, rtol=0.0
    ):
        raise ValueError(f"rows must have an orthonormal Gram matrix (atol={atol})")


def basis_from_rows(
    rows: Sequence[Sequence[float]],
    *,
    dimension: int,
    rank: int,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Validate row-wise basis vectors and return them as ``[dimension, rank]``.

    Validation is performed on CPU in FP64.  The returned basis is CPU FP32
    and contiguous; no orthogonalization or singular-value validation occurs.
    """
    dimension = _require_int(dimension, "dimension")
    rank = _require_int(rank, "rank")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if rank <= 0 or rank > dimension:
        raise ValueError(f"rank must satisfy 0 < rank <= dimension, got {rank}")
    atol = _validate_atol(atol)

    row_tensor = _rows_tensor(rows, dimension=dimension, rank=rank)
    _validate_orthonormal_rows(row_tensor, atol=atol)
    result = row_tensor.T.contiguous().to(dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("basis cannot be represented as finite FP32")
    return result


def _validate_cpu_floating_tensor(value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a CPU floating tensor")
    if value.device.type != "cpu" or not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a CPU floating tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense CPU floating tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def project_onto_basis(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project CPU floating vectors onto a validated orthonormal basis.

    ``values`` may be ``[d]`` or ``[n, d]``.  Arithmetic is explicitly FP32
    and the returned tensor is a new finite CPU FP32 tensor with the same
    shape as ``values``.
    """
    values = _validate_cpu_floating_tensor(values, "values")
    basis = _validate_cpu_floating_tensor(basis, "basis")

    if values.ndim not in (1, 2):
        raise ValueError(f"values must have shape [d] or [n, d], got {tuple(values.shape)}")
    if values.ndim == 1:
        if values.shape[0] == 0:
            raise ValueError("values must contain at least one dimension")
    elif values.shape[0] == 0:
        raise ValueError("values must contain at least one row")

    if basis.ndim != 2:
        raise ValueError(f"basis must have shape [d, k], got {tuple(basis.shape)}")
    dimension, rank = basis.shape
    if dimension <= 0 or rank <= 0 or rank > dimension:
        raise ValueError(f"basis shape must satisfy 1 <= k <= d, got {tuple(basis.shape)}")
    if values.shape[-1] != dimension:
        raise ValueError(
            f"values width {values.shape[-1]} does not match basis dimension {dimension}"
        )

    basis64 = basis.to(dtype=torch.float64)
    gram = basis64.T @ basis64
    expected = torch.eye(rank, dtype=torch.float64, device="cpu")
    if not torch.isfinite(gram).all() or not torch.allclose(
        gram, expected, atol=1e-6, rtol=0.0
    ):
        raise ValueError("basis columns must have an orthonormal Gram matrix (atol=1e-6)")

    values32 = values.to(dtype=torch.float32)
    basis32 = basis.to(dtype=torch.float32)
    coordinates = values32 @ basis32
    if not torch.isfinite(coordinates).all():
        raise ValueError("FP32 projection overflowed")
    projected = coordinates @ basis32.T
    if not torch.isfinite(projected).all():
        raise ValueError("FP32 projection overflowed")
    return projected.contiguous()
