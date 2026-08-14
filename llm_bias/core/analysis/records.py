"""Compact token records for persisted analysis artifacts."""
from __future__ import annotations
from typing import Any
import torch


def top_k_token_records(probabilities: torch.Tensor, *, top_k: int, tokenizer: Any) -> list[dict[str, Any]]:
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional vocabulary vector")
    if not 1 <= top_k <= probabilities.numel():
        raise ValueError(f"top_k must be between 1 and {probabilities.numel()}")
    top = probabilities.topk(top_k)
    records = []
    for rank, (token_id, probability) in enumerate(zip(top.indices.tolist(), top.values.tolist(), strict=True), 1):
        records.append({"rank": rank, "token_id": int(token_id), "token": tokenizer.decode([int(token_id)], skip_special_tokens=False), "probability": float(probability)})
    return records


def top_k_attribution_records(values: torch.Tensor, *, top_k: int, tokenizer: Any, value_name: str = "mean_attribution") -> list[dict[str, Any]]:
    if values.ndim != 1:
        raise ValueError("attribution values must be one-dimensional")
    top = values.topk(min(top_k, values.numel()))
    return [{"rank": rank, "token_id": int(token_id), "token": tokenizer.decode([int(token_id)], skip_special_tokens=False), value_name: float(value)} for rank, (token_id, value) in enumerate(zip(top.indices.tolist(), top.values.tolist(), strict=True), 1) if float(value) > 0]
