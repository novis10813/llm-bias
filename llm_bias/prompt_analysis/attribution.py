"""Generated-token Semantic Scope attribution primitives."""

from __future__ import annotations

from typing import Any

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.model import WrappedModel

from llm_bias.core.prompt_input import decode_token

RETURN_LABELS = {"very bullish", "bullish", "neutral", "bearish", "very bearish"}


def parse_generated_return_answer(text: Any) -> dict[str, Any]:
    """Parse a strict return-classification JSON answer without raising."""
    result = {
        "predicted_label": None,
        "predicted_confidence": None,
        "parse_status": "invalid",
        "parse_reason": None,
    }
    if not isinstance(text, str):
        result["parse_reason"] = "generated_text_not_string"
        return result
    start = text.find("{")
    if start < 0:
        result["parse_reason"] = "json_object_not_found"
        return result
    try:
        import json

        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        result["parse_reason"] = "malformed_json"
        return result
    if not isinstance(payload, dict):
        result["parse_reason"] = "json_payload_not_object"
        return result
    label = payload.get("label")
    confidence = payload.get("confidence")
    if label not in RETURN_LABELS:
        result["parse_reason"] = "invalid_label"
        return result
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 0 <= confidence <= 100
    ):
        result["parse_reason"] = "invalid_confidence"
        return result
    return {
        "predicted_label": label,
        "predicted_confidence": confidence,
        "parse_status": "valid",
        "parse_reason": None,
    }


def _semantic_scope_scores(gradient: torch.Tensor) -> torch.Tensor:
    """Return one Semantic Scope influence score per input position."""
    if gradient.ndim != 3:
        raise ValueError(
            f"expected [batch, sequence, hidden] gradient, got {gradient.shape}"
        )
    return torch.linalg.vector_norm(gradient.float(), ord=2, dim=-1)


@torch.enable_grad()
def _attribute_generated_token(
    *,
    model: WrappedModel,
    prompt_ids: torch.Tensor,
    generated_prefix: torch.Tensor,
    target_id: int,
    input_top_k: int | None,
    input_span: tuple[int, int],
) -> dict[str, Any]:
    full_ids = torch.cat([prompt_ids, generated_prefix], dim=1)
    attention_mask = torch.ones_like(full_ids)
    embedding_box: dict[str, torch.Tensor] = {}

    def root_embedding(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        rooted = output.detach().requires_grad_(True)
        embedding_box["value"] = rooted
        return rooted

    final_layer = model.n_layers - 1
    handle = model._embed.register_forward_hook(root_embedding)
    try:
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model._decoder(
                input_ids=full_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            residual = recorder.activations[final_layer][0, -1].float()
        logits = model.unembed(residual).float()
        target_logit = logits[target_id]
        log_probability = logits.log_softmax(dim=-1)[target_id]
        embedding = embedding_box["value"]
        gradient = torch.autograd.grad(target_logit, embedding)[0]
    finally:
        handle.remove()

    input_start, input_end = input_span
    contribution = _semantic_scope_scores(gradient)[0, input_start:input_end]
    top_k = input_end - input_start if input_top_k is None else min(
        input_top_k, input_end - input_start
    )
    top = contribution.topk(top_k)
    return {
        "token_id": target_id,
        "token": decode_token(model.tokenizer, target_id),
        "logit": float(target_logit.detach()),
        "log_probability": float(log_probability.detach()),
        "top_input_tokens": [
            {
                "rank": rank,
                "position": input_start + int(position),
                "prompt_position": int(position),
                "token_id": int(prompt_ids[0, input_start + position]),
                "token": decode_token(
                    model.tokenizer, int(prompt_ids[0, input_start + position])
                ),
                "attribution": float(value),
            }
            for rank, (position, value) in enumerate(
                zip(top.indices.tolist(), top.values.tolist(), strict=True),
                start=1,
            )
        ],
    }
