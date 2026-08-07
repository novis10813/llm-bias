"""Tokenizer-aware continuation likelihood scoring.

This module deliberately contains no task-specific semantics.  It scores a
candidate string as a continuation of an already formatted prompt and exposes
an oriented margin between two candidate scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class CandidateScore:
    """Compact score for one candidate continuation."""

    candidate: str
    token_ids: list[int]
    log_probability: float
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateMargin:
    """Difference between two candidate continuation log probabilities."""

    positive: CandidateScore
    negative: CandidateScore

    @property
    def value(self) -> float:
        return self.positive.log_probability - self.negative.log_probability

    @property
    def definition(self) -> str:
        return f"logP({self.positive.candidate})-logP({self.negative.candidate})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
            "margin": self.value,
            "margin_definition": self.definition,
        }


def _input_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if torch.is_tensor(values):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def continuation_token_ids(tokenizer: Any, prompt: str, candidate: str) -> tuple[list[int], list[int]]:
    """Return prompt IDs and exact candidate suffix IDs.

    The suffix is obtained from tokenizing the complete ``prompt + candidate``
    string.  A non-prefix tokenization is rejected because silently replacing it
    with standalone candidate tokenization would score a different string.
    """
    prompt_ids = _input_ids(tokenizer, prompt, add_special_tokens=True)
    combined_ids = _input_ids(tokenizer, prompt + candidate, add_special_tokens=True)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "prompt tokenization is not a prefix of prompt+candidate; "
            "cannot score an exact continuation without an explicit boundary"
        )
    suffix = combined_ids[len(prompt_ids) :]
    if not suffix:
        raise ValueError(f"candidate {candidate!r} produced no continuation tokens")
    return prompt_ids, suffix


def _extract_logits(output: Any, model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract ``[batch, sequence, vocab]`` logits from common model APIs."""
    if hasattr(model, "unembed") and hasattr(model, "layers"):
        # jlens wrappers can return a residual tensor with the same rank as
        # logits, so inspect the wrapper contract before treating tensors as
        # vocabulary logits.
        from jlens.hooks import ActivationRecorder

        final_layer = model.n_layers - 1
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
            residual = recorder.activations[final_layer]
        logits = model.unembed(residual)
    elif hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, dict) and "logits" in output:
        logits = output["logits"]
    elif torch.is_tensor(output) and output.ndim == 3:
        logits = output
    else:
        raise TypeError("model output does not expose logits or an unembedding interface")
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("model logits must have shape [batch, sequence, vocab]")
    return logits


def score_token_ids(
    model: Any,
    tokenizer: Any,
    prompt: str,
    token_ids: list[int],
    *,
    device: torch.device | str | None = None,
    forward: Callable[[torch.Tensor], Any] | None = None,
) -> CandidateScore:
    """Score an exact token suffix with teacher forcing."""
    prompt_ids = _input_ids(tokenizer, prompt, add_special_tokens=True)
    if not token_ids:
        raise ValueError("token_ids must be non-empty")
    combined_ids = prompt_ids + [int(token_id) for token_id in token_ids]
    target_device = torch.device(device) if device is not None else None
    input_tensor = torch.tensor([combined_ids], dtype=torch.long)
    if target_device is not None:
        input_tensor = input_tensor.to(target_device)
    with torch.no_grad():
        output = forward(input_tensor) if forward is not None else model.forward(input_tensor)
        logits = _extract_logits(output, model, input_tensor).float()
    log_probs = torch.log_softmax(logits[0], dim=-1)
    start = len(prompt_ids) - 1
    selected = torch.stack(
        [log_probs[start + index, token_id] for index, token_id in enumerate(token_ids)]
    )
    return CandidateScore(
        candidate="",
        token_ids=[int(token_id) for token_id in token_ids],
        log_probability=float(selected.sum().detach().cpu()),
        token_count=len(token_ids),
    )


def score_candidate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate: str,
    *,
    device: torch.device | str | None = None,
    forward: Callable[[torch.Tensor], Any] | None = None,
) -> CandidateScore:
    """Score a candidate using the exact prompt-plus-candidate token suffix."""
    _prompt_ids, suffix = continuation_token_ids(tokenizer, prompt, candidate)
    score = score_token_ids(
        model,
        tokenizer,
        prompt,
        suffix,
        device=device,
        forward=forward,
    )
    return CandidateScore(
        candidate=candidate,
        token_ids=score.token_ids,
        log_probability=score.log_probability,
        token_count=score.token_count,
    )


def score_margin(
    model: Any,
    tokenizer: Any,
    prompt: str,
    positive: str,
    negative: str,
    *,
    device: torch.device | str | None = None,
    forward: Callable[[torch.Tensor], Any] | None = None,
) -> CandidateMargin:
    """Score two candidates and return ``positive minus negative``."""
    return CandidateMargin(
        positive=score_candidate(
            model, tokenizer, prompt, positive, device=device, forward=forward
        ),
        negative=score_candidate(
            model, tokenizer, prompt, negative, device=device, forward=forward
        ),
    )


def score_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Iterable[str],
    *,
    device: torch.device | str | None = None,
    forward: Callable[[torch.Tensor], Any] | None = None,
) -> list[CandidateScore]:
    """Score candidates independently, preserving caller order."""
    return [
        score_candidate(model, tokenizer, prompt, candidate, device=device, forward=forward)
        for candidate in candidates
    ]
