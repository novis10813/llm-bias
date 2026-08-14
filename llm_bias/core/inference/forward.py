"""Encoded batching and final-position residual capture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .logits import extract_logits, final_position_logits


@dataclass(frozen=True)
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    final_positions: torch.Tensor


def encode_batch(ids: list[list[int]], device: Any, pad_token_id: int = 0) -> EncodedBatch:
    if not ids:
        raise ValueError("empty batch")
    if any(not row for row in ids):
        raise ValueError("all-padding row")
    width = max(map(len, ids))
    target = torch.device(device)
    input_ids = torch.full((len(ids), width), pad_token_id, dtype=torch.long, device=target)
    mask = torch.zeros_like(input_ids)
    for index, row in enumerate(ids):
        input_ids[index, : len(row)] = torch.as_tensor(row, dtype=torch.long, device=target)
        mask[index, : len(row)] = 1
    final_positions = mask.sum(-1) - 1
    return EncodedBatch(input_ids, mask, final_positions)


def _forward(model: Any, encoded: EncodedBatch) -> Any:
    runner = getattr(model, "forward", None)
    if runner is None:
        runner = model
    try:
        return runner(encoded.input_ids, attention_mask=encoded.attention_mask)
    except TypeError:
        return runner(encoded.input_ids)


def capture_final_residuals(
    model: Any,
    encoded: EncodedBatch,
    layers: Iterable[int],
    *,
    keep_device: bool = False,
) -> dict[int, torch.Tensor]:
    """Capture requested jlens layer residuals at each row's final position."""
    requested = sorted(set(int(layer) for layer in layers))
    if not requested:
        return {}
    if not hasattr(model, "layers"):
        return {}
    from jlens.hooks import ActivationRecorder

    with ActivationRecorder(model.layers, at=requested) as recorder:
        _forward(model, encoded)
        values = {}
        for layer in requested:
            activation = recorder.activations[layer]
            selected = activation[torch.arange(activation.shape[0], device=activation.device), encoded.final_positions.to(activation.device)]
            values[layer] = selected.detach() if keep_device else selected.detach().cpu()
        return values


def forward_batch(
    model: Any,
    ids: list[list[int]],
    layers: Iterable[int] = (),
    device: Any = "cpu",
    *,
    pad_token_id: int = 0,
    final_norm: Any | None = None,
    keep_activations_device: bool = False,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
    """Run a padded batch and return final logits, residuals, temperatures.

    Returned logits, residuals (unless ``keep_activations_device``), and
    temperatures are detached.  Temperatures are computed before CPU offload.
    """
    encoded = encode_batch(ids, device, pad_token_id)
    requested = sorted(set(int(layer) for layer in layers))
    if hasattr(model, "layers") and requested:
        from jlens.hooks import ActivationRecorder
        with torch.no_grad(), ActivationRecorder(model.layers, at=requested) as recorder:
            output = _forward(model, encoded)
            residuals: dict[int, torch.Tensor] = {}
            for layer in requested:
                activation = recorder.activations[layer]
                residuals[layer] = activation[
                    torch.arange(len(ids), device=activation.device),
                    encoded.final_positions.to(activation.device),
                ].float()
            residual = residuals[requested[-1]]
            logits = extract_logits(output, model=model, residual=residual)
    else:
        with torch.no_grad():
            output = _forward(model, encoded)
        logits = extract_logits(output, model=model)
        residuals = {}

    if logits.ndim == 2:
        selected_logits = logits
    else:
        selected_logits = final_position_logits(logits, encoded.final_positions).float()
    selected_logits = selected_logits.float()
    if residuals:
        normalized = residual
        if final_norm is not None:
            params = list(final_norm.parameters())
            norm_device = params[0].device if params else normalized.device
            normalized = final_norm(normalized.to(norm_device))
        temperatures = normalized.float().norm(dim=-1).reciprocal()
        if not torch.isfinite(temperatures).all() or (temperatures <= 0).any():
            raise ValueError("non-finite effective temperature")
        temperatures = temperatures.detach().cpu()
        residuals = {
            layer: value.detach() if keep_activations_device else value.detach().cpu()
            for layer, value in residuals.items()
        }
    else:
        temperatures = torch.ones(len(ids), dtype=torch.float32)
    return selected_logits.detach().cpu(), residuals, temperatures


def record_residuals(model: Any, input_ids: torch.Tensor, layers: Iterable[int]) -> dict[int, torch.Tensor]:
    """Record full-sequence residuals for intervention consumers."""
    requested = sorted(set(int(layer) for layer in layers))
    if not requested or not hasattr(model, "layers"):
        return {}
    from jlens.hooks import ActivationRecorder
    with torch.no_grad(), ActivationRecorder(model.layers, at=requested) as recorder:
        _forward(model, EncodedBatch(input_ids, torch.ones_like(input_ids), torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device)))
        return {layer: recorder.activations[layer].detach().clone() for layer in requested}


__all__ = ["EncodedBatch", "capture_final_residuals", "encode_batch", "forward_batch", "record_residuals"]
