"""Generic, temporary residual-stream interventions.

This module contains hook lifecycle mechanics only. Experiment-specific
steering semantics live in their owning package.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
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


def _block_hidden(args: tuple[Any, ...], kwargs: Any) -> torch.Tensor:
    """First hidden-states tensor of a decoder-block / norm-module forward."""
    hidden = args[0] if args else (kwargs or {}).get("hidden_states")
    if not torch.is_tensor(hidden):
        raise ValueError("forward received no hidden states tensor")
    return hidden


def _post_attention_norm(model: Any, layer: int) -> Any:
    """Resolve the hookable post_attention_layernorm of one decoder block."""
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("model does not expose decoder layers")
    index = int(layer)
    if index < 0 or index >= len(layers):
        raise ValueError(f"intervention layer {layer} is out of range")
    owner = getattr(layers[index], "_hf_layer", layers[index])
    module = getattr(owner, "post_attention_layernorm", None)
    if module is None or not hasattr(module, "register_forward_pre_hook"):
        raise TypeError(f"layer {layer} does not expose a hookable post_attention_layernorm")
    return module


@contextmanager
def mid_residual_interventions(
    model: Any,
    transforms: Mapping[int, ResidualTransform],
) -> Iterator[None]:
    """Apply one transform at each selected block's post-attention residual.

    The mid residual is the input of the block's ``post_attention_layernorm``
    (the residual after the attention sub-layer and before the MLP sub-layer).
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

    for raw_layer in sorted(transforms):
        layer = int(raw_layer)
        transform = transforms[raw_layer]
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"intervention layer {layer} is out of range")
        norm = _post_attention_norm(model, layer)

        def hook(_module: Any, args: tuple[Any, ...], kwargs: Any | None = None, *, fn=transform) -> Any:
            kwargs = kwargs or {}
            hidden = args[0] if args else kwargs.get("hidden_states")
            if not torch.is_tensor(hidden):
                raise ValueError("post_attention_layernorm input is not a tensor")
            replacement = fn(hidden)
            if not torch.is_tensor(replacement) or replacement.shape != hidden.shape:
                raise ValueError("residual transform must preserve tensor shape")
            if replacement is hidden:
                return None
            if args:
                return ((replacement, *args[1:]), kwargs)
            return (args, {**kwargs, "hidden_states": replacement})

        handles.append(norm.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def record_block_states(
    model: Any,
    input_ids: torch.Tensor,
    layers: Iterable[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Record pre/post-attention/post-block residuals for the requested layers.

    Returns ``{layer: {"pre": T, "mid": T, "post": T}}`` with full-sequence
    ``[batch, sequence, d_model]`` detached tensors in the model dtype.
    ``pre`` is the block input, ``mid`` the post-attention residual (the
    ``post_attention_layernorm`` input), and ``post`` the block output.
    Values are transient; callers must reduce them to derived outputs before
    serialization.
    """
    requested = sorted(set(int(layer) for layer in layers))
    if not requested:
        return {}
    if not hasattr(model, "layers"):
        raise TypeError("model does not expose decoder layers")
    for layer in requested:
        if layer < 0 or layer >= len(model.layers):
            raise ValueError(f"intervention layer {layer} is out of range")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, sequence]")

    from .forward import EncodedBatch, _forward

    states: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in requested}
    handles: list[Any] = []
    try:
        for layer in requested:
            block = model.layers[layer]
            norm = _post_attention_norm(model, layer)

            def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: Any | None = None, *, at=layer) -> None:
                states[at]["pre"] = _block_hidden(args, kwargs).detach().clone()

            def mid_hook(_module: Any, args: tuple[Any, ...], kwargs: Any | None = None, *, at=layer) -> None:
                states[at]["mid"] = _block_hidden(args, kwargs).detach().clone()

            def post_hook(_module: Any, _inputs: Any, output: Any, *, at=layer) -> None:
                tensor = output if torch.is_tensor(output) else output[0]
                states[at]["post"] = tensor.detach().clone()

            handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(norm.register_forward_pre_hook(mid_hook, with_kwargs=True))
            handles.append(block.register_forward_hook(post_hook))
        with torch.no_grad():
            _forward(model, EncodedBatch(
                input_ids,
                torch.ones_like(input_ids),
                torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device),
            ))
    finally:
        for handle in handles:
            handle.remove()

    for layer in requested:
        missing = [key for key in ("pre", "mid", "post") if key not in states[layer]]
        if missing:
            raise RuntimeError(f"block state capture incomplete at layer {layer}: {missing}")
    return states


__all__ = [
    "ResidualTransform",
    "mid_residual_interventions",
    "record_block_states",
    "residual_interventions",
]
