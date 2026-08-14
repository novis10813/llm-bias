"""Generic representation transport primitives.

These helpers identify Jacobian transport explicitly; they do not claim causal
patching or gradient attribution.
"""
from __future__ import annotations
from typing import Any
import torch

JACOBIAN_TRANSPORT_METHOD = "jacobian_transport"
GRADIENT_ATTRIBUTION_METHOD = "gradient_attribution"
RESIDUAL_PATCH_METHOD = "residual_patch"


def transport_residual_delta(
    entity: torch.Tensor,
    baseline: torch.Tensor,
    *,
    layer: int,
    final_layer: int,
    lens: Any | None = None,
    jacobian_cache: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Transport a residual difference into final-layer working space."""
    entity = entity.float()
    baseline = baseline.float()
    if entity.ndim not in (1, 2) or baseline.shape != entity.shape:
        raise ValueError("residuals must have matching [d] or [batch,d] shapes")
    delta = entity - baseline
    if not torch.isfinite(delta).all():
        raise ValueError("residual delta must be finite")
    if layer == final_layer:
        return delta
    if lens is None and jacobian_cache is None:
        raise ValueError("non-final transport requires a lens or cached Jacobian")
    if jacobian_cache is not None:
        if layer not in jacobian_cache:
            raise ValueError(f"missing cached Jacobian for layer {layer}")
        jacobian = jacobian_cache[layer].to(device=delta.device, dtype=torch.float32)
        if jacobian.ndim != 2 or jacobian.shape[1] != delta.shape[-1]:
            raise ValueError("cached Jacobian has incompatible shape")
        delta = delta @ jacobian.T
    else:
        delta = lens.transport(delta, layer)
    if not torch.isfinite(delta).all() or delta.shape != entity.shape:
        raise ValueError("Jacobian transport returned invalid residual shape")
    return delta.float()


def transport_method_metadata(*, layer: int, final_layer: int) -> dict[str, Any]:
    return {
        "method": JACOBIAN_TRANSPORT_METHOD,
        "layer": int(layer),
        "final_layer": int(final_layer),
        "is_transport": layer != final_layer,
    }
