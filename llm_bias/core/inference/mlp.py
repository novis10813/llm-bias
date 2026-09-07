"""Transient dense-MLP coordinate recording and interventions (no persistence)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
import math

import torch


def dense_down_projection(layer: Any) -> Any:
    """Resolve only a dense, hookable MLP; do not silently select a MoE expert."""
    owner = getattr(layer, "_hf_layer", layer)
    mlp = getattr(owner, "mlp", None)
    module = getattr(mlp, "down_proj", None)
    if module is None or not hasattr(module, "register_forward_pre_hook"):
        raise TypeError("layer does not expose a dense MLP down_proj")
    return module


@contextmanager
def mlp_coordinates(
    model: Any, layers: list[int], position: int, *,
    edits: dict[int, dict[int, tuple[float, float | None]]] | None = None,
) -> Iterator[dict[int, torch.Tensor]]:
    """Record one batch-one token before edits; then scale and optionally replace.

    Absolute positions remain fixed during teacher forcing. Values live only in
    memory; callers must reduce them to derived statistics before serialization.
    """
    if position < 0 or not layers or len(set(layers)) != len(layers):
        raise ValueError("require unique layers and an absolute nonnegative position")
    if any(layer < 0 or layer >= len(model.layers) for layer in layers):
        raise ValueError("layer out of range")
    if set(edits or {}) - set(layers):
        raise ValueError("edit layer is not recorded")
    for channels in (edits or {}).values():
        for neuron, (scale, replacement) in channels.items():
            if neuron < 0 or not math.isfinite(scale) or (replacement is not None and not math.isfinite(replacement)):
                raise ValueError("invalid coordinate edit")
    records: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for layer in layers:
            def hook(_module, args, layer=layer):
                if not args or not torch.is_tensor(args[0]):
                    raise ValueError("missing MLP input")
                values = args[0]
                if values.ndim != 3 or values.shape[0] != 1 or position >= values.shape[1]:
                    raise ValueError("require [1, sequence, width] and valid absolute position")
                vector = values[0, position].detach().float().cpu().clone()
                if not torch.isfinite(vector).all():
                    raise ValueError("non-finite MLP values")
                records[layer] = vector
                changes = (edits or {}).get(layer, {})
                if not changes:
                    return None
                result = values.clone()
                for neuron, (scale, replacement) in changes.items():
                    if neuron >= values.shape[-1]:
                        raise ValueError("neuron out of range")
                    result[0, position, neuron] *= scale
                    if replacement is not None:
                        result[0, position, neuron] = replacement
                return (result, *args[1:])
            handles.append(dense_down_projection(model.layers[layer]).register_forward_pre_hook(hook))
        yield records
    finally:
        for handle in handles:
            handle.remove()
