"""Online execution of compact J-space intervention trials."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from llm_bias.core.continuation_scoring import score_margin
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.jspace_intervention.concepts import (
    sector_prototype,
    token_direction,
    unembedding_weight,
)
from llm_bias.jspace_intervention.positions import select_loaded_positions
from llm_bias.jspace_intervention.schemas import InterventionConfig, PrototypeSpec
from llm_bias.jspace_intervention.transforms import coordinate_intervention, steer_positions


def layer_prototypes(
    model: Any,
    lens: Any,
    spec: PrototypeSpec,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Derive layer-specific prototypes without persisting direction vectors."""
    unembedding = unembedding_weight(model)
    weights = {token.token_id: token.weight for token in spec.tokens}
    result = {}
    for layer in layers:
        if layer not in lens.jacobians:
            raise ValueError(f"canonical lens does not contain layer {layer}")
        jacobian = lens.jacobians[layer].float()
        directions = {
            token.token_id: token_direction(unembedding, jacobian, token.token_id)
            for token in spec.tokens
        }
        result[layer] = sector_prototype(directions, weights).detach()
    return result


def _input_tensor(tokenizer: Any, text: str, device: Any) -> torch.Tensor:
    ids = input_ids(tokenizer, text, add_special_tokens=True)
    return torch.tensor([ids], dtype=torch.long, device=device)


def _margin_dict(margin: Any) -> dict:
    result = margin.to_dict()
    positive = result["positive"]["log_probability"]
    negative = result["negative"]["log_probability"]
    result["two_candidate_positive_probability"] = float(
        torch.sigmoid(torch.tensor(positive - negative)).item()
    )
    return result


def run_swap_record(
    *,
    model: Any,
    tokenizer: Any,
    lens: Any,
    scoring_prompt: str,
    evidence_span: tuple[int, int],
    config: InterventionConfig,
    device: Any,
    source_prototypes: Mapping[int, torch.Tensor] | None = None,
    target_prototypes: Mapping[int, torch.Tensor] | None = None,
) -> list[dict]:
    """Run clean and source→target prototype swaps for one formatted prompt."""
    source = dict(source_prototypes or layer_prototypes(model, lens, config.source, config.layers))
    target = dict(target_prototypes or layer_prototypes(model, lens, config.target, config.layers))
    if set(source) != set(config.layers) or set(target) != set(config.layers):
        raise ValueError("precomputed prototypes must cover all configured layers")
    clean_tensor = _input_tensor(tokenizer, scoring_prompt, device)
    residuals = record_residuals(model, clean_tensor, config.layers)
    selected = select_loaded_positions(
        residuals,
        source,
        evidence_span=evidence_span,
        top_k=config.top_positions,
        threshold=config.loading_threshold,
    )
    clean_margin = score_margin(
        model,
        tokenizer,
        scoring_prompt,
        config.positive_candidate,
        config.negative_candidate,
        device=device,
    )
    clean_value = clean_margin.value
    rows = []
    for mode in config.coordinate_modes:
        for alpha in config.alphas:
            transforms = {}
            if alpha != 0 and selected.loaded:
                for layer in config.layers:
                    source_direction = source[layer]
                    target_direction = target[layer]

                    def transform(
                        tensor: torch.Tensor,
                        *,
                        src=source_direction,
                        tgt=target_direction,
                        dose=alpha,
                        edit_mode=mode,
                    ) -> torch.Tensor:
                        return coordinate_intervention(
                            tensor,
                            positions=selected.positions,
                            source_direction=src,
                            target_direction=tgt,
                            alpha=dose,
                            mode=edit_mode,
                        )

                    transforms[layer] = transform
            with residual_interventions(model, transforms):
                margin = clean_margin if alpha == 0 or not selected.loaded else score_margin(
                    model,
                    tokenizer,
                    scoring_prompt,
                    config.positive_candidate,
                    config.negative_candidate,
                    device=device,
                )
            rows.append(
                {
                    "intervention_type": f"sector_coordinate_{mode}",
                    "source_prototype": config.source.name,
                    "target_prototype": config.target.name,
                    "layers": list(config.layers),
                    "alpha": alpha,
                    "loaded_positions": selected.to_dict(),
                    "clean_margin": clean_value,
                    "intervened_margin": margin.value,
                    "delta_margin": margin.value - clean_value,
                    "score": _margin_dict(margin),
                }
            )
    return rows


def run_token_steering_record(
    *,
    model: Any,
    tokenizer: Any,
    lens: Any,
    scoring_prompt: str,
    evidence_span: tuple[int, int],
    token_id: int,
    token: str,
    layers: Sequence[int],
    alphas: Sequence[float],
    coordinate_scales: Mapping[int, float],
    top_positions: int,
    loading_threshold: float,
    positive_candidate: str,
    negative_candidate: str,
    device: Any,
) -> list[dict]:
    """Run one token-coordinate dose sweep for one formatted prompt."""
    if set(layers) != set(coordinate_scales):
        raise ValueError("coordinate scales must cover every intervention layer")
    unembedding = unembedding_weight(model)
    directions = {
        layer: token_direction(unembedding, lens.jacobians[layer].float(), token_id).detach()
        for layer in layers
    }
    clean_tensor = _input_tensor(tokenizer, scoring_prompt, device)
    residuals = record_residuals(model, clean_tensor, layers)
    selected = select_loaded_positions(
        residuals,
        directions,
        evidence_span=evidence_span,
        top_k=top_positions,
        threshold=loading_threshold,
    )
    clean_margin = score_margin(
        model, tokenizer, scoring_prompt, positive_candidate, negative_candidate,
        device=device,
    )
    rows = []
    divisor = len(layers) ** 0.5
    for alpha in alphas:
        transforms = {}
        if alpha != 0 and selected.loaded:
            for layer in layers:
                direction = directions[layer]
                coordinate_delta = alpha * float(coordinate_scales[layer]) / divisor

                def transform(
                    tensor: torch.Tensor,
                    *,
                    vector=direction,
                    delta=coordinate_delta,
                ) -> torch.Tensor:
                    return steer_positions(
                        tensor,
                        positions=selected.positions,
                        direction=vector,
                        coordinate_delta=delta,
                    )

                transforms[layer] = transform
        with residual_interventions(model, transforms):
            margin = clean_margin if alpha == 0 or not selected.loaded else score_margin(
                model, tokenizer, scoring_prompt, positive_candidate, negative_candidate,
                device=device,
            )
        rows.append(
            {
                "intervention_type": "token_coordinate_steering",
                "token": token,
                "token_id": token_id,
                "layers": list(layers),
                "alpha": float(alpha),
                "loaded_positions": selected.to_dict(),
                "clean_margin": clean_margin.value,
                "intervened_margin": margin.value,
                "delta_margin": margin.value - clean_margin.value,
                "score": _margin_dict(margin),
            }
        )
    return rows


__all__ = ["layer_prototypes", "run_swap_record", "run_token_steering_record"]
