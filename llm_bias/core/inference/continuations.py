"""Exact encoded continuation scoring through the shared FP32 tail."""
from __future__ import annotations

from typing import Any
import torch
from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals


def score_encoded_candidates(model: Any, prompt_ids: list[int], suffixes: list[list[int]], device: Any) -> list[float]:
    """Share the prompt forward when all exact continuations contain one token."""
    if not prompt_ids or not suffixes or any(not suffix for suffix in suffixes):
        raise ValueError("prompt and suffixes must be nonempty")
    if all(len(suffix) == 1 for suffix in suffixes):
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        final = len(model.layers) - 1
        with torch.no_grad():
            residual = record_residuals(model, ids, [final])[final][:, -1]
            logp = fp32_next_token_log_probs(model, residual)[0]
        tokens = [suffix[0] for suffix in suffixes]
        if any(token < 0 or token >= logp.numel() for token in tokens):
            raise ValueError("suffix token out of vocabulary")
        values = logp[tokens]
        if not torch.isfinite(values).all():
            raise ValueError("non-finite continuation score")
        return values.cpu().tolist()
    return [score_encoded_suffix(model, prompt_ids, suffix, device) for suffix in suffixes]


def score_encoded_suffix(model: Any, prompt_ids: list[int], suffix_ids: list[int], device: Any) -> float:
    """Sum conditional log probabilities, excluding the final unneeded input token.

    The caller owns tokenization/prefix verification. Hooks remain active during
    this forward and must address prompt positions, not the extended last token.
    """
    if not prompt_ids or not suffix_ids:
        raise ValueError("prompt and suffix must be nonempty")
    ids = torch.tensor([prompt_ids + suffix_ids[:-1]], dtype=torch.long, device=device)
    final = len(model.layers) - 1
    with torch.no_grad():
        residual = record_residuals(model, ids, [final])[final]
        total = 0.0
        for offset, token in enumerate(suffix_ids):
            logp = fp32_next_token_log_probs(model, residual[:, len(prompt_ids) - 1 + offset])
            if not 0 <= token < logp.shape[-1]:
                raise ValueError("suffix token out of vocabulary")
            value = logp[0, token]
            if not torch.isfinite(value):
                raise ValueError("non-finite continuation score")
            total += float(value.cpu())
    return total
