"""Compact generation and decision-position screening records."""
from __future__ import annotations

import json
import math
from typing import Any

import torch

from llm_bias.core.continuation_scoring import score_single_token_margin_fp32
from llm_bias.core.inference.adapter import InjectedModelAdapter
from llm_bias.core.inference.generation import GenerationConfig, generate_tokens
from llm_bias.core.prompt_input.encoding import input_ids
from .template import DECISION_PREFIX


def _generation_target(model: Any) -> Any:
    """Expose the raw HF model for ``model.hf_model.generate``.

    jlens wrappers keep the raw model under ``_hf_model``; core generation
    expects the ``hf_model`` attribute (see ``InjectedModelAdapter``).
    """
    hf_model = getattr(model, "_hf_model", None)
    if hf_model is None:
        return model
    return InjectedModelAdapter(model, hf_model=hf_model)


def parse_decision(generated_text: str) -> str | None:
    """Parse the first JSON object from a greedy completion.

    The model pretty-prints the object and appends stop-marker text after
    the closing brace, so the object is extracted with ``raw_decode`` from
    the first ``{`` and trailing content is ignored (protocol §2).
    """
    if not isinstance(generated_text, str):
        return None
    start = generated_text.find("{")
    if start == -1:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(generated_text[start:])
    except json.JSONDecodeError:
        return None
    decision = value.get("decision") if isinstance(value, dict) else None
    return decision if decision in {"buy", "sell"} else None


def screen_prompt(model: Any, tokenizer: Any, prompt_text: str, *, device: torch.device | str = "cpu") -> dict[str, Any]:
    prompt_ids = torch.tensor([input_ids(tokenizer, prompt_text, add_special_tokens=True)], dtype=torch.long, device=device)
    generated = generate_tokens(
        _generation_target(model),
        prompt_ids,
        GenerationConfig(max_new_tokens=128, temperature=0.0, pad_token_id=getattr(tokenizer, "eos_token_id", None)),
    )
    ids = generated[0].tolist()
    input_length = prompt_ids.shape[1]
    new_ids = ids[input_length:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    score = score_single_token_margin_fp32(model, tokenizer, prompt_text + DECISION_PREFIX, "buy", "sell", device=device)
    margin = float(score.value)
    if not math.isfinite(margin):
        raise ValueError("non-finite decision margin")
    return {
        "margin": margin,
        "decision": parse_decision(generated_text),
        "generated_text": generated_text,
        "n_new_tokens": len(new_ids),
        "scoring_mode": "fp32_single_token" if len(score.positive.token_ids) == 1 and len(score.negative.token_ids) == 1 else "continuation_fallback",
    }
