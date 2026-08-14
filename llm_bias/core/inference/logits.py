"""Logit extraction for Hugging Face and jlens model adapters."""
from __future__ import annotations

from typing import Any

import torch


def extract_logits(output: Any, *, model: Any | None = None, residual: torch.Tensor | None = None) -> torch.Tensor:
    """Extract a logits tensor from common HF/jlens output contracts."""
    value = output
    if hasattr(value, "logits"):
        value = value.logits
    elif isinstance(value, dict):
        if "logits" not in value:
            raise ValueError("model output mapping has no logits field")
        value = value["logits"]
    elif isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("model output tuple is empty")
        value = value[0]
    if torch.is_tensor(value):
        return value
    if residual is not None and model is not None and hasattr(model, "unembed"):
        return model.unembed(residual)
    raise TypeError(f"cannot extract logits from model output {type(output)!r}")


def final_position_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Select one final non-padding position per batch row."""
    if logits.ndim != 3 or positions.ndim != 1 or logits.shape[0] != positions.shape[0]:
        raise ValueError("logits must be [batch, sequence, vocab] and positions [batch]")
    return logits[torch.arange(logits.shape[0], device=logits.device), positions.to(logits.device)]


def next_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract final-position logits from a jlens decoder adapter."""
    if not hasattr(model, "layers") or not hasattr(model, "unembed"):
        output = model(input_ids)
        positions = torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device)
        return final_position_logits(extract_logits(output, model=model), positions)[0].float().cpu()
    from jlens.hooks import ActivationRecorder
    final_layer = int(model.n_layers) - 1
    with torch.no_grad(), ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
        residual = recorder.activations[final_layer][:, -1, :].detach()
    return model.unembed(residual).float().cpu()[0]


__all__ = ["extract_logits", "final_position_logits", "next_logits"]
