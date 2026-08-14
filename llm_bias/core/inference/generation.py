"""Generation controls and output normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    use_cache: bool = True
    pad_token_id: int | None = None

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
            "pad_token_id": self.pad_token_id,
        }
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        return kwargs


def generate_tokens(model: Any, prompt_ids: torch.Tensor, config: GenerationConfig | None = None, **kwargs: Any) -> torch.Tensor:
    """Generate a single complete sequence, accepting HF output variants."""
    if config is None:
        config = GenerationConfig(**kwargs)
    elif kwargs:
        raise TypeError("pass either config or generation keyword arguments")
    with torch.no_grad():
        result = model.hf_model.generate(prompt_ids, **config.as_kwargs())
    sequences = getattr(result, "sequences", result)
    if isinstance(sequences, (tuple, list)):
        if not sequences:
            raise ValueError("model.generate returned no sequences")
        sequences = sequences[0]
    if not isinstance(sequences, torch.Tensor):
        raise TypeError("model.generate must return a tensor or output with sequences")
    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        raise ValueError("generation requires exactly one sequence")
    return sequences


def finish_reason(generated_ids: list[int], *, eos_token_id: int | list[int] | tuple[int, ...] | None, max_new_tokens: int) -> str:
    if not generated_ids:
        return "empty"
    eos_ids = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else ({eos_token_id} if eos_token_id is not None else set())
    if eos_ids.intersection(generated_ids):
        return "eos_token"
    if len(generated_ids) >= max_new_tokens:
        return "max_new_tokens"
    return "model_stop"


__all__ = ["GenerationConfig", "finish_reason", "generate_tokens"]
