"""Reusable validation helpers for GPU-sharded Jacobian probes."""
from __future__ import annotations
from typing import Any, Iterable
import torch

def validate_probe_gradients(*, activations: dict[int, torch.Tensor], source_layers: Iterable[int], target_layer: int, source_devices: dict[int, str] | None = None) -> dict[str, Any]:
    """Validate finite, nonzero cross-device gradients for a target activation."""
    if target_layer not in activations:
        raise ValueError(f"target layer {target_layer} was not recorded")
    target = activations[target_layer]
    if not target.requires_grad:
        raise ValueError("target activation must retain gradients")
    probe = target.float().sum()
    grads = torch.autograd.grad(probe, [activations[layer] for layer in source_layers], allow_unused=False, retain_graph=False)
    result = {}
    for layer, grad in zip(source_layers, grads, strict=True):
        if not torch.isfinite(grad).all() or not torch.any(grad != 0):
            raise ValueError(f"layer {layer} gradient is non-finite or zero")
        if grad.shape != activations[layer].shape:
            raise ValueError(f"layer {layer} gradient shape mismatch")
        if source_devices and str(grad.device) != source_devices[layer]:
            raise ValueError(f"layer {layer} gradient device mismatch")
        result[str(layer)] = {"device": str(grad.device), "shape": list(grad.shape), "norm": float(grad.norm().item())}
    return {"target_layer": target_layer, "target_device": str(target.device), "gradients": result}
