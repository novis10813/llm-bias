"""Residual recording and causal activation patching."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from jlens.hooks import ActivationRecorder


def record_residuals(model: Any, input_ids: torch.Tensor, layers: Iterable[int]) -> dict[int, torch.Tensor]:
    requested = sorted(set(layers))
    with torch.no_grad(), ActivationRecorder(model.layers, at=requested) as recorder:
        model.forward(input_ids)
        return {
            layer: recorder.activations[layer].detach().clone()
            for layer in requested
        }


def _replace_first(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = replacement
        return output
    raise TypeError(f"Unsupported decoder block output type: {type(output)!r}")


def patched_next_logits(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layer: int,
    position: int,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Run source input while replacing one residual position at one layer."""
    final_layer = model.n_layers - 1
    handles: list[Any] = []

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        patched = tensor.clone()
        patched[:, position, :] = replacement.to(device=patched.device, dtype=patched.dtype)
        return _replace_first(output, patched)

    with torch.no_grad():
        handles.append(model.layers[layer].register_forward_hook(patch_hook))
        try:
            with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
                model.forward(input_ids)
                final = recorder.activations[final_layer][:, -1, :].detach()
        finally:
            for handle in handles:
                handle.remove()
    return model.unembed(final).float().cpu()[0]


def patched_residuals(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layers: Iterable[int],
    patch_layer: int,
    position: int,
    replacement: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Record a source forward pass after one residual activation is patched.

    The patch hook is registered before :class:`ActivationRecorder`, so the
    recorded activation at ``patch_layer`` is the intervened representation.
    This is the representation-level counterpart to ``patched_next_logits``
    and is used by the interactive visualization endpoint.
    """
    requested = sorted(set(layers) | {patch_layer})

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        patched = tensor.clone()
        patched[:, position, :] = replacement.to(device=patched.device, dtype=patched.dtype)
        return _replace_first(output, patched)

    handle = model.layers[patch_layer].register_forward_hook(patch_hook)
    try:
        with torch.no_grad(), ActivationRecorder(model.layers, at=requested) as recorder:
            model.forward(input_ids)
            return {
                layer: recorder.activations[layer].detach().clone()
                for layer in requested
            }
    finally:
        handle.remove()


def next_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    final_layer = model.n_layers - 1
    with torch.no_grad(), ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
        final = recorder.activations[final_layer][:, -1, :].detach()
    return model.unembed(final).float().cpu()[0]
