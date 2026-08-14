"""Injection boundary for inference without loading a checkpoint."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal model contract consumed by the inference core."""

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any: ...


class InjectedModelAdapter:
    """Delegate an injected HF/jlens-compatible model without loading it."""

    def __init__(self, model: Any, *, hf_model: Any | None = None, device: Any | None = None) -> None:
        self.model = model
        self.hf_model = hf_model if hf_model is not None else getattr(model, "hf_model", model)
        self.device = torch.device(device) if device is not None else getattr(model, "device", torch.device("cpu"))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        runner = getattr(self.model, "forward", self.model)
        return runner(input_ids, **kwargs)


__all__ = ["InjectedModelAdapter", "ModelAdapter"]
