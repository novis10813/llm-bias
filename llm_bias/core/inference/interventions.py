"""Generic, temporary residual-stream interventions.

This module contains hook lifecycle mechanics only. Experiment-specific
steering semantics live in their owning package.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

ResidualTransform = Callable[[torch.Tensor], torch.Tensor]


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
    raise TypeError(f"unsupported decoder block output type: {type(output)!r}")


@contextmanager
def residual_interventions(
    model: Any,
    transforms: Mapping[int, ResidualTransform],
) -> Iterator[None]:
    """Apply one transform after each selected block for the context lifetime.

    Every transform receives and returns a ``[batch, sequence, d_model]``
    tensor. Hooks are removed even if forward execution raises.
    """
    if not transforms:
        yield
        return
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("model does not expose decoder layers")
    handles = []

    for raw_layer, transform in sorted(transforms.items()):
        layer = int(raw_layer)
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"intervention layer {layer} is out of range")

        def hook(_module: Any, _inputs: Any, output: Any, *, fn=transform) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            replacement = fn(tensor)
            if not torch.is_tensor(replacement) or replacement.shape != tensor.shape:
                raise ValueError("residual transform must preserve tensor shape")
            return _replace_first(output, replacement)

        handles.append(layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


__all__ = ["ResidualTransform", "residual_interventions"]
