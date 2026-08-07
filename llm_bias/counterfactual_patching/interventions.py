"""Residual recording and causal patching for counterfactual pairs."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch

from jlens.hooks import ActivationRecorder


def normalized_span_mapping(source_length: int, target_length: int) -> list[int]:
    """Map source span positions to nearest target span positions.

    Positions are matched by normalized span-internal token centers. The
    mapping is deterministic and never synthesizes an activation, which makes
    it suitable for causal patching between variable-length entity spans.
    """
    if source_length <= 0 or target_length <= 0:
        raise ValueError("source_length and target_length must both be positive")
    mapping: list[int] = []
    for source_index in range(source_length):
        normalized_target = (
            (source_index + 0.5) * target_length / source_length - 0.5
        )
        target_index = int(math.floor(normalized_target + 0.5))
        mapping.append(min(target_length - 1, max(0, target_index)))
    return mapping


def _span_bounds(
    *, position: int | None, source_span: tuple[int, int] | None
) -> tuple[int, int]:
    if position is not None and source_span is not None:
        raise ValueError("pass either position or source_span, not both")
    if source_span is None:
        if position is None:
            raise ValueError("one of position or source_span is required")
        source_span = (position, position + 1)
    start, end = source_span
    if start < 0 or end <= start:
        raise ValueError(f"invalid source span {source_span}")
    return start, end


def _patch_tensor_span(
    tensor: torch.Tensor,
    *,
    source_span: tuple[int, int],
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Replace a source span with nearest-mapped target span activations."""
    start, end = source_span
    if end > tensor.shape[1]:
        raise ValueError(f"source span {source_span} exceeds sequence length {tensor.shape[1]}")
    if replacement.ndim == 1:
        replacement = replacement.unsqueeze(0)
    if replacement.ndim == 2:
        replacement = replacement.unsqueeze(0)
    if replacement.ndim != 3 or replacement.shape[0] not in {1, tensor.shape[0]}:
        raise ValueError(
            "replacement must have shape [span, d_model] or [batch, span, d_model]"
        )
    if replacement.shape[0] == 1 and tensor.shape[0] != 1:
        replacement = replacement.expand(tensor.shape[0], -1, -1)
    mapping = normalized_span_mapping(end - start, replacement.shape[1])
    patched = tensor.clone()
    mapped = replacement.to(device=tensor.device, dtype=tensor.dtype)[:, mapping, :]
    patched[:, start:end, :] = mapped
    return patched


def _patch_tensor_spans(
    tensor: torch.Tensor,
    *,
    source_spans: Iterable[tuple[int, int]],
    replacements: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Patch several disjoint spans without changing sequence length."""
    spans = list(source_spans)
    values = list(replacements)
    if len(spans) != len(values) or not spans:
        raise ValueError("source_spans and replacements must be equally non-empty")
    ordered = sorted(spans)
    if any(start < 0 or end <= start for start, end in ordered):
        raise ValueError("source spans must be non-empty and ordered")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("source spans must not overlap")
    patched = tensor
    for span, replacement in zip(spans, values, strict=True):
        patched = _patch_tensor_span(
            patched,
            source_span=span,
            replacement=replacement,
        )
    return patched


def _add_tensor_spans(
    tensor: torch.Tensor,
    *,
    source_spans: Iterable[tuple[int, int]],
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Add one direction at every token in several disjoint spans."""
    spans = list(source_spans)
    if not spans:
        raise ValueError("source_spans must be non-empty")
    ordered = sorted(spans)
    if any(start < 0 or end <= start for start, end in ordered):
        raise ValueError("source spans must be non-empty and ordered")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("source spans must not overlap")
    if direction.ndim == 1:
        direction = direction.view(1, 1, -1)
    elif direction.ndim == 2:
        direction = direction.unsqueeze(0)
    if direction.ndim != 3 or direction.shape[0] not in {1, tensor.shape[0]}:
        raise ValueError("direction must have shape [d_model] or [batch, 1, d_model]")
    if direction.shape[0] == 1 and tensor.shape[0] != 1:
        direction = direction.expand(tensor.shape[0], -1, -1)
    direction = direction.to(device=tensor.device, dtype=tensor.dtype)
    patched = tensor.clone()
    for start, end in spans:
        if start < 0 or end <= start or end > tensor.shape[1]:
            raise ValueError(f"invalid source span {(start, end)}")
        patched[:, start:end, :] = patched[:, start:end, :] + alpha * direction
    return patched


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
    position: int | None = None,
    replacement: torch.Tensor,
    source_span: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Run source input while replacing one residual span at one layer.

    ``position`` remains supported for the original single-token API. For a
    variable-length entity use ``source_span`` and provide the target span
    activations in ``replacement``.
    """
    final_layer = model.n_layers - 1
    resolved_span = _span_bounds(position=position, source_span=source_span)
    handles: list[Any] = []

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        patched = _patch_tensor_span(
            tensor, source_span=resolved_span, replacement=replacement
        )
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


def patched_next_logits_multi(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layer: int,
    source_spans: Iterable[tuple[int, int]],
    replacements: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Run a forward pass while replacing several residual spans."""
    final_layer = model.n_layers - 1
    spans = list(source_spans)
    values = list(replacements)
    handles: list[Any] = []

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        patched = _patch_tensor_spans(
            tensor,
            source_spans=spans,
            replacements=values,
        )
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


def steered_next_logits(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layer: int,
    source_spans: Iterable[tuple[int, int]],
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Inject a direction at all selected span tokens and return final logits."""
    final_layer = model.n_layers - 1
    spans = list(source_spans)

    def steer_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        steered = _add_tensor_spans(
            tensor,
            source_spans=spans,
            direction=direction,
            alpha=alpha,
        )
        return _replace_first(output, steered)

    handle = model.layers[layer].register_forward_hook(steer_hook)
    try:
        with torch.no_grad(), ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
            final = recorder.activations[final_layer][:, -1, :].detach()
    finally:
        handle.remove()
    return model.unembed(final).float().cpu()[0]


def patched_residuals(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layers: Iterable[int],
    patch_layer: int,
    position: int | None = None,
    replacement: torch.Tensor,
    source_span: tuple[int, int] | None = None,
) -> dict[int, torch.Tensor]:
    """Record a source forward pass after one residual span is patched.

    The patch hook is registered before :class:`ActivationRecorder`, so the
    recorded activation at ``patch_layer`` is the intervened representation.
    This is the representation-level counterpart to ``patched_next_logits``
    and is used by the interactive visualization endpoint.
    """
    requested = sorted(set(layers) | {patch_layer})
    resolved_span = _span_bounds(position=position, source_span=source_span)

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        patched = _patch_tensor_span(
            tensor, source_span=resolved_span, replacement=replacement
        )
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
