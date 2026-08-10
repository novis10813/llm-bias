"""Execution stages for the easy-bias binary-association experiment."""

from __future__ import annotations

import hashlib
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch

from llm_bias.core.continuation_scoring import continuation_token_ids, score_margin
from llm_bias.core.lens_artifacts import validate_lens_for_model
from jlens.hooks import ActivationRecorder
from llm_bias.counterfactual_patching.binary_association import (
    CANDIDATES,
    BinaryAssociationPair,
    RenderedPrompt,
    iter_rendered_prompts,
    validate_pair,
)
from llm_bias.counterfactual_patching.interventions import (
    _add_tensor_spans,
    _patch_tensor_span,
    _patch_tensor_spans,
    _replace_first,
    normalized_span_mapping,
    record_residuals,
)


def _input_ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    try:
        encoded = tokenizer(text, return_tensors="pt")
    except TypeError:
        encoded = tokenizer(text)
    values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    values = torch.as_tensor(values, dtype=torch.long)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    return values.to(device)


def _last_non_entity_position(spans: list[tuple[int, int]], sequence_length: int) -> int:
    for position in range(sequence_length - 1, -1, -1):
        if all(not start <= position < end for start, end in spans):
            return position
    raise ValueError("entity spans cover the entire prompt")


def _non_entity_span_candidates(
    spans: list[tuple[int, int]],
    sequence_length: int,
    span_length: int,
    occupied: list[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    if span_length <= 0:
        raise ValueError("span_length must be positive")
    candidates: list[tuple[int, int]] = []
    for start in range(0, sequence_length - span_length + 1):
        end = start + span_length
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            continue
        if any(start < span_end and end > span_start for span_start, span_end in occupied):
            continue
        candidates.append((start, end))
    return candidates


def _matched_non_entity_spans(
    source_spans: list[tuple[int, int]],
    target_spans: list[tuple[int, int]],
    source_length: int,
    target_length: int,
    *,
    key: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Choose deterministic, disjoint non-entity spans with matched widths."""
    source_selected: list[tuple[int, int]] = []
    target_selected: list[tuple[int, int]] = []
    for index, ((source_start, source_end), (target_start, target_end)) in enumerate(
        zip(source_spans, target_spans, strict=True)
    ):
        source_width = source_end - source_start
        target_width = target_end - target_start
        source_candidates = _non_entity_span_candidates(
            source_spans, source_length, source_width, source_selected
        )
        target_candidates = _non_entity_span_candidates(
            target_spans, target_length, target_width, target_selected
        )
        if not source_candidates or not target_candidates:
            raise ValueError(
                "could not find disjoint matched non-entity control spans "
                f"for occurrence {index}"
            )
        source_digest = hashlib.sha256(f"{key}:source:{index}".encode()).digest()
        target_digest = hashlib.sha256(f"{key}:target:{index}".encode()).digest()
        source_selected.append(source_candidates[int.from_bytes(source_digest[:8], "big") % len(source_candidates)])
        target_selected.append(target_candidates[int.from_bytes(target_digest[:8], "big") % len(target_candidates)])
    return source_selected, target_selected


def _hooked_logits(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layer: int,
    transform: Any,
) -> torch.Tensor:
    """Return full-sequence logits after a temporary residual hook."""
    final_layer = model.n_layers - 1

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        return _replace_first(output, transform(tensor))

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad(), ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
            residual = recorder.activations[final_layer].detach()
    finally:
        handle.remove()
    return model.unembed(residual).float().cpu()[0]


def _score_with_transform(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate: str,
    *,
    layer: int,
    transform: Any,
    device: torch.device,
) -> float:
    prompt_ids, suffix = continuation_token_ids(tokenizer, prompt, candidate)
    input_ids = torch.tensor([prompt_ids + suffix], dtype=torch.long, device=device)
    logits = _hooked_logits(model, input_ids, layer=layer, transform=transform)
    log_probs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    return float(sum(log_probs[start + index, token_id] for index, token_id in enumerate(suffix)))


def score_rendered_prompt(
    model: Any,
    tokenizer: Any,
    record: RenderedPrompt,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """Compute the fixed mother-minus-father baseline margin."""
    margin = score_margin(
        model,
        tokenizer,
        record.formatted_prompt,
        CANDIDATES[0],
        CANDIDATES[1],
        device=device,
    )
    return {
        "record_id": f"{record.career_id}-{record.prompt_order}",
        "career_id": record.career_id,
        "career": record.career,
        "split": record.split,
        "prompt_order": record.prompt_order,
        "formatted_prompt": record.formatted_prompt,
        "input_ids": _input_ids(tokenizer, record.formatted_prompt, torch.device(device))[0].detach().cpu().tolist(),
        "entity_spans": [span.to_dict() for span in record.entity_spans],
        "margin": margin.value,
        "margin_definition": margin.definition,
        "mother_score": margin.positive.to_dict(),
        "father_score": margin.negative.to_dict(),
        "task_type": record.task_type,
    }


def run_baseline(
    model: Any,
    tokenizer: Any,
    rendered_path: str | Path,
    output_path: str | Path,
    *,
    device: torch.device | str,
    max_rows: int | None = None,
) -> Path:
    records = iter_rendered_prompts(rendered_path)
    if max_rows is not None:
        records = islice(records, max_rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            row = score_rendered_prompt(model, tokenizer, record, device=device)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def _span_tuples(record: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(span["token_start"]), int(span["token_end"]))
        for span in record["entity_spans"]
    ]


def patch_pair_single_layer(
    model: Any,
    tokenizer: Any,
    pair: BinaryAssociationPair,
    *,
    layer: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Run entity, non-entity, matched-random, and permuted-target patches."""
    validate_pair(pair)
    device = torch.device(device)
    source_ids = _input_ids(tokenizer, pair.source_prompt, device)
    target_ids = _input_ids(tokenizer, pair.target_prompt, device)
    target_residuals = record_residuals(model, target_ids, [layer])
    source_spans = [
        (int(span["token_start"]), int(span["token_end"]))
        for span in pair.source_entity_spans
    ]
    target_spans = [
        (int(span["token_start"]), int(span["token_end"]))
        for span in pair.target_entity_spans
    ]
    target_layer_residuals = target_residuals[layer]
    entity_replacements = [
        target_layer_residuals[:, start:end, :] for start, end in target_spans
    ]

    def score_transform(transform: Any) -> float:
        scores = [
            _score_with_transform(
                model,
                tokenizer,
                pair.source_prompt,
                candidate,
                layer=layer,
                transform=transform,
                device=device,
            )
            for candidate in CANDIDATES
        ]
        return scores[0] - scores[1]

    def multi_span_transform(
        spans: list[tuple[int, int]], replacements: list[torch.Tensor]
    ) -> Any:
        return lambda tensor: _patch_tensor_spans(
            tensor, source_spans=spans, replacements=replacements
        )

    source_margin = score_margin(
        model,
        tokenizer,
        pair.source_prompt,
        CANDIDATES[0],
        CANDIDATES[1],
        device=device,
    ).value
    target_margin = score_margin(
        model,
        tokenizer,
        pair.target_prompt,
        CANDIDATES[0],
        CANDIDATES[1],
        device=device,
    ).value
    patched_margin = score_transform(
        multi_span_transform(source_spans, entity_replacements)
    )

    source_control = _last_non_entity_position(source_spans, source_ids.shape[1])
    target_control = _last_non_entity_position(target_spans, target_ids.shape[1])
    control_margin = score_transform(
        lambda tensor: _patch_tensor_span(
            tensor,
            source_span=(source_control, source_control + 1),
            replacement=target_layer_residuals[:, target_control, :],
        )
    )

    permuted_margin = score_transform(
        multi_span_transform(source_spans, list(reversed(entity_replacements)))
    )
    matched_source_spans, matched_target_spans = _matched_non_entity_spans(
        source_spans,
        target_spans,
        source_ids.shape[1],
        target_ids.shape[1],
        key=pair.pair_id,
    )
    matched_replacements = [
        target_layer_residuals[:, start:end, :]
        for start, end in matched_target_spans
    ]
    matched_margin = score_transform(
        multi_span_transform(matched_source_spans, matched_replacements)
    )
    span_mappings = [
        normalized_span_mapping(end - start, target_end - target_start)
        for (start, end), (target_start, target_end) in zip(
            source_spans, target_spans, strict=True
        )
    ]
    return {
        "pair_id": pair.pair_id,
        "contrast_id": pair.contrast_id,
        "direction": pair.direction,
        "prompt_order": pair.prompt_order,
        "layer": layer,
        "source_span_count": len(source_spans),
        "target_span_count": len(target_spans),
        "source_entity_spans": [list(span) for span in source_spans],
        "target_entity_spans": [list(span) for span in target_spans],
        "span_mappings": span_mappings,
        "mapping_strategy": "normalized_span_internal_token_centers_nearest",
        "source_margin": source_margin,
        "target_margin": target_margin,
        "patched_margin": patched_margin,
        "control_margin": control_margin,
        "permuted_target_margin": permuted_margin,
        "matched_random_margin": matched_margin,
        "direct_effect": target_margin - source_margin,
        "causal_patch_effect": patched_margin - source_margin,
        "control_patch_effect": control_margin - source_margin,
        "permuted_target_effect": permuted_margin - source_margin,
        "matched_random_effect": matched_margin - source_margin,
        "corrected_effect": patched_margin - control_margin,
        "source_control_position": source_control,
        "target_control_position": target_control,
        "matched_random_source_spans": [list(span) for span in matched_source_spans],
        "matched_random_target_spans": [list(span) for span in matched_target_spans],
        "task_type": pair.task_type,
        "margin_definition": pair.margin_definition,
    }


def steer_pair(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    device: torch.device | str,
    source_spans: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Inject an aggregate direction and score exact candidate continuations."""
    device = torch.device(device)
    spans = source_spans if source_spans is not None else _span_tuples(record)

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return _add_tensor_spans(
            tensor,
            source_spans=spans,
            direction=direction,
            alpha=alpha,
        )

    scores = [
        _score_with_transform(
            model,
            tokenizer,
            record["formatted_prompt"],
            candidate,
            layer=layer,
            transform=transform,
            device=device,
        )
        for candidate in CANDIDATES
    ]
    return {
        "record_id": record["record_id"],
        "career_id": record["career_id"],
        "split": record["split"],
        "prompt_order": record["prompt_order"],
        "layer": layer,
        "alpha": alpha,
        "mother_log_probability": scores[0],
        "father_log_probability": scores[1],
        "mother_minus_father_logprob_margin": scores[0] - scores[1],
        "baseline_margin": float(record["margin"]),
        "steering_effect": scores[0] - scores[1] - float(record["margin"]),
        "margin_definition": record["margin_definition"],
        "task_type": record["task_type"],
    }


def validate_binary_lens(
    model: Any,
    *,
    model_name: str,
    lens_path: str | Path,
) -> dict[str, Any]:
    """Validate a complete model-specific lens without fitting or reading it."""
    import jlens

    path = Path(lens_path)
    if not path.is_file():
        raise FileNotFoundError(f"Jacobian lens does not exist: {path}")
    lens = jlens.JacobianLens.load(str(path))
    metadata = validate_lens_for_model(
        model=model,
        lens=lens,
        model_name=model_name,
        lens_path=path,
        require_complete=True,
    )
    return {
        "compatible": True,
        "model": model_name,
        "lens_path": str(path),
        "d_model": int(model.d_model),
        "n_layers": int(model.n_layers),
        "source_layers": [int(layer) for layer in lens.source_layers],
        "metadata": metadata,
    }
