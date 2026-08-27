"""Online execution of compact J-space intervention trials."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from llm_bias.core.continuation_scoring import (
    categorical_kl_divergence,
    next_token_log_probabilities,
    score_margin,
    score_single_token_margin_fp32,
)
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.jspace_intervention.concepts import (
    concept_coordinate,
    sector_prototype,
    token_direction,
    unembedding_weight,
)
from llm_bias.jspace_intervention.controls import (
    matched_random_direction,
    matched_random_prototypes,
    norm_match_intervention_to_target,
    shuffled_evidence_positions,
)
from llm_bias.jspace_intervention.positions import select_loaded_positions
from llm_bias.jspace_intervention.schemas import (
    GainConfig,
    InterventionConfig,
    PrototypeSpec,
    TokenScreenCandidate,
    TokenScreenConfig,
)
from llm_bias.jspace_intervention.transforms import (
    coordinate_gain,
    coordinate_intervention,
    steer_positions,
)


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


def token_screen_directions(
    model: Any,
    lens: Any,
    candidates: Sequence[TokenScreenCandidate],
    layers: Sequence[int],
) -> dict[int, dict[int, torch.Tensor]]:
    """Precompute per-layer token directions for one screen prompt run.

    The direction for a candidate at layer ``l`` is the residual-space row
    ``W_U J_l[token_id]``; the candidate ``representation_side`` is provenance
    only and never enters the direction or the dose sign.
    """
    unembedding = unembedding_weight(model)
    result: dict[int, dict[int, torch.Tensor]] = {}
    for candidate in candidates:
        layer_directions = {}
        for layer in layers:
            if layer not in lens.jacobians:
                raise ValueError(f"canonical lens does not contain layer {layer}")
            jacobian = lens.jacobians[layer].float()
            layer_directions[layer] = token_direction(
                unembedding, jacobian, candidate.token_id
            ).detach()
        result[candidate.token_id] = layer_directions
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


def _position_control_sets(
    controls: Sequence[str],
    *,
    evidence_span: tuple[int, int],
    selected: Sequence[int],
    sequence_length: int,
    seed: int,
) -> dict[str, tuple[int, ...]]:
    result = {}
    for control in controls:
        if control == "evidence":
            result[control] = tuple(selected)
        elif control == "final_position":
            result[control] = (sequence_length - 1,)
        elif control == "shuffled_evidence":
            result[control] = shuffled_evidence_positions(
                evidence_span, selected, count=len(selected), seed=seed
            )
        else:
            raise ValueError(f"unsupported position control: {control}")
    return result


def _swap_diagnostics(
    residuals: Mapping[int, torch.Tensor],
    source: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int],
    swap_fraction: float,
    mode: str,
    patched_tensors: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, float]:
    source_before = []
    target_before = []
    source_after = []
    target_after = []
    perturbation_squared = 0.0
    state_squared = 0.0
    condition_numbers = []
    for layer, tensor in residuals.items():
        src = source[layer].to(device=tensor.device, dtype=torch.float32)
        tgt = target[layer].to(device=tensor.device, dtype=torch.float32)
        matrix = torch.stack((src, tgt), dim=-1)
        condition = float(torch.linalg.cond(matrix).detach().cpu())
        if not torch.isfinite(torch.tensor(condition)):
            raise ValueError(f"non-finite source/target condition number at layer {layer}")
        condition_numbers.append(condition)
        before_values = tensor[:, list(positions), :].float()
        if patched_tensors is not None:
            patched = patched_tensors[layer]
        elif swap_fraction == 0:
            patched = tensor
        else:
            patched = coordinate_intervention(
                tensor,
                positions=positions,
                source_direction=src,
                target_direction=tgt,
                alpha=swap_fraction,
                mode=mode,
            )
        after_values = patched[:, list(positions), :].float()
        pinv = torch.linalg.pinv(matrix)
        before_coordinates = before_values @ pinv.T
        after_coordinates = after_values @ pinv.T
        source_before.extend(before_coordinates[..., 0].flatten().tolist())
        target_before.extend(before_coordinates[..., 1].flatten().tolist())
        source_after.extend(after_coordinates[..., 0].flatten().tolist())
        target_after.extend(after_coordinates[..., 1].flatten().tolist())
        perturbation_squared += float((after_values - before_values).square().sum().cpu())
        state_squared += float(before_values.square().sum().cpu())
    perturbation_norm = perturbation_squared ** 0.5
    state_norm = state_squared ** 0.5
    return {
        "source_coordinate_before_mean": sum(source_before) / len(source_before),
        "target_coordinate_before_mean": sum(target_before) / len(target_before),
        "source_coordinate_after_mean": sum(source_after) / len(source_after),
        "target_coordinate_after_mean": sum(target_after) / len(target_after),
        "perturbation_norm": perturbation_norm,
        "state_norm": state_norm,
        "relative_perturbation_norm": perturbation_norm / max(state_norm, 1e-12),
        "direction_condition_number_max": max(condition_numbers),
        "direction_condition_number_mean": sum(condition_numbers) / len(condition_numbers),
    }


def _gain_diagnostics(
    residuals: Mapping[int, torch.Tensor],
    directions: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int],
    gain: float,
    patched_tensors: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, float]:
    before_coordinates = []
    after_coordinates = []
    perturbation_squared = 0.0
    state_squared = 0.0
    direction_norms = []
    for layer, tensor in residuals.items():
        direction = directions[layer].to(device=tensor.device, dtype=torch.float32)
        norm = float(direction.norm().detach().cpu())
        if not torch.isfinite(torch.tensor(norm)) or norm <= 0:
            raise ValueError(f"invalid gain direction norm at layer {layer}")
        direction_norms.append(norm)
        values = tensor[:, list(positions), :].float()
        denominator = direction.square().sum()
        before = (values @ direction) / denominator
        patched = (
            patched_tensors[layer]
            if patched_tensors is not None
            else coordinate_gain(
                tensor, positions=positions, direction=direction, gain=gain
            )
        )
        after_values = patched[:, list(positions), :].float()
        after = (after_values @ direction) / denominator
        before_coordinates.extend(before.flatten().tolist())
        after_coordinates.extend(after.flatten().tolist())
        perturbation_squared += float((after_values - values).square().sum().cpu())
        state_squared += float(values.square().sum().cpu())
    perturbation_norm = perturbation_squared ** 0.5
    state_norm = state_squared ** 0.5
    return {
        "coordinate_before_mean": sum(before_coordinates) / len(before_coordinates),
        "coordinate_after_mean": sum(after_coordinates) / len(after_coordinates),
        "perturbation_norm": perturbation_norm,
        "state_norm": state_norm,
        "relative_perturbation_norm": perturbation_norm / max(state_norm, 1e-12),
        "direction_norm_min": min(direction_norms),
        "direction_norm_max": max(direction_norms),
    }


def _screen_diagnostics(
    residuals: Mapping[int, torch.Tensor],
    directions: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int],
    coordinate_deltas: Mapping[int, float],
    patched_tensors: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, float]:
    before_coordinates = []
    after_coordinates = []
    perturbation_squared = 0.0
    state_squared = 0.0
    direction_norms = []
    for layer, tensor in residuals.items():
        direction = directions[layer].to(device=tensor.device, dtype=torch.float32)
        norm = float(direction.norm().detach().cpu())
        if not torch.isfinite(torch.tensor(norm)) or norm <= 0:
            raise ValueError(f"invalid screen direction norm at layer {layer}")
        direction_norms.append(norm)
        values = tensor[:, list(positions), :].float()
        vector = direction.reshape(-1)
        denominator = vector.square().sum()
        before = (values @ vector) / denominator
        patched = (
            patched_tensors[layer]
            if patched_tensors is not None
            else steer_positions(
                tensor,
                positions=positions,
                direction=vector,
                coordinate_delta=coordinate_deltas[layer],
            )
        )
        after_values = patched[:, list(positions), :].float()
        after = (after_values @ vector) / denominator
        before_coordinates.extend(before.flatten().tolist())
        after_coordinates.extend(after.flatten().tolist())
        perturbation_squared += float((after_values - values).square().sum().cpu())
        state_squared += float(values.square().sum().cpu())
    perturbation_norm = perturbation_squared ** 0.5
    state_norm = state_squared ** 0.5
    return {
        "coordinate_before_mean": sum(before_coordinates) / len(before_coordinates),
        "coordinate_after_mean": sum(after_coordinates) / len(after_coordinates),
        "perturbation_norm": perturbation_norm,
        "state_norm": state_norm,
        "relative_perturbation_norm": perturbation_norm / max(state_norm, 1e-12),
        "direction_norm_min": min(direction_norms),
        "direction_norm_max": max(direction_norms),
    }


def _aggregate_live_diagnostics(
    layer_diagnostics: Mapping[int, Mapping[str, float]],
) -> dict[str, float]:
    if not layer_diagnostics:
        raise ValueError("live intervention produced no layer diagnostics")
    rows = list(layer_diagnostics.values())
    result = {}
    for key in rows[0]:
        values = [float(row[key]) for row in rows]
        if key in {"perturbation_norm", "state_norm"}:
            result[key] = sum(value * value for value in values) ** 0.5
        elif key.endswith("_max"):
            result[key] = max(values)
        elif key.endswith("_min"):
            result[key] = min(values)
        elif key != "relative_perturbation_norm":
            result[key] = sum(values) / len(values)
    result["relative_perturbation_norm"] = (
        result["perturbation_norm"] / max(result["state_norm"], 1e-12)
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
    control_seed: int = 0,
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
    clean_log_probs = next_token_log_probabilities(
        model, tokenizer, scoring_prompt, device=device
    )
    positions_by_control = _position_control_sets(
        config.position_controls,
        evidence_span=evidence_span,
        selected=selected.positions,
        sequence_length=clean_tensor.shape[1],
        seed=control_seed,
    )
    directions_by_control = {"prototype": (source, target)}
    if "matched_random" in config.direction_controls:
        directions_by_control["matched_random"] = matched_random_prototypes(
            source, target, seed=control_seed
        )
    paired_primary_norms: dict[tuple[str, float], dict[int, float]] = {}
    has_controls = any(
        direction != "prototype" or position != "evidence"
        for direction in config.direction_controls
        for position in config.position_controls
    )
    if has_controls and selected.loaded:
        for mode in config.coordinate_modes:
            for swap_fraction in config.swap_fractions:
                if swap_fraction == 0:
                    continue
                layer_norms: dict[int, float] = {}
                primary_transforms = {}
                for layer in config.layers:
                    source_direction = source[layer]
                    target_direction = target[layer]

                    def primary_transform(
                        tensor: torch.Tensor,
                        *,
                        src=source_direction,
                        tgt=target_direction,
                        fraction=swap_fraction,
                        edit_mode=mode,
                        positions=selected.positions,
                        layer_id=layer,
                    ) -> torch.Tensor:
                        patched = coordinate_intervention(
                            tensor,
                            positions=positions,
                            source_direction=src,
                            target_direction=tgt,
                            alpha=fraction,
                            mode=edit_mode,
                        )
                        layer_norms[layer_id] = float(
                            (
                                patched[:, list(positions), :].float()
                                - tensor[:, list(positions), :].float()
                            ).norm().detach().cpu()
                        )
                        return patched

                    primary_transforms[layer] = primary_transform
                with residual_interventions(model, primary_transforms), torch.inference_mode():
                    model.forward(clean_tensor)
                if set(layer_norms) != set(config.layers):
                    raise RuntimeError("paired primary swap forward missed intervention layers")
                paired_primary_norms[(mode, swap_fraction)] = layer_norms
    rows = []
    for direction_control in config.direction_controls:
        active_source, active_target = directions_by_control[direction_control]
        for position_control in config.position_controls:
            intervention_positions = positions_by_control[position_control]
            dose_matched_control = (
                direction_control != "prototype" or position_control != "evidence"
            )
            for mode in config.coordinate_modes:
                for swap_fraction in config.swap_fractions:
                    transforms = {}
                    live_diagnostics: dict[int, dict[str, float]] = {}
                    if swap_fraction != 0 and selected.loaded:
                        for layer in config.layers:
                            source_direction = active_source[layer]
                            target_direction = active_target[layer]
                            paired_target_norm = (
                                paired_primary_norms[(mode, swap_fraction)][layer]
                                if dose_matched_control
                                else None
                            )

                            def transform(
                                tensor: torch.Tensor,
                                *,
                                src=source_direction,
                                tgt=target_direction,
                                fraction=swap_fraction,
                                edit_mode=mode,
                                positions=intervention_positions,
                                layer_id=layer,
                                target_norm=paired_target_norm,
                                match_control=dose_matched_control,
                            ) -> torch.Tensor:
                                patched = coordinate_intervention(
                                    tensor,
                                    positions=positions,
                                    source_direction=src,
                                    target_direction=tgt,
                                    alpha=fraction,
                                    mode=edit_mode,
                                )
                                if match_control:
                                    patched = norm_match_intervention_to_target(
                                        tensor,
                                        patched,
                                        control_positions=positions,
                                        target_norm=float(target_norm),
                                    )
                                live_diagnostics[layer_id] = _swap_diagnostics(
                                    {layer_id: tensor.detach()},
                                    {layer_id: src},
                                    {layer_id: tgt},
                                    positions=positions,
                                    swap_fraction=fraction,
                                    mode=edit_mode,
                                    patched_tensors={layer_id: patched.detach()},
                                )
                                return patched

                            transforms[layer] = transform
                    delivered_fraction = swap_fraction if selected.loaded else 0.0
                    with residual_interventions(model, transforms):
                        if swap_fraction == 0 or not selected.loaded:
                            margin = clean_margin
                            intervened_log_probs = clean_log_probs
                        else:
                            margin = score_margin(
                                model,
                                tokenizer,
                                scoring_prompt,
                                config.positive_candidate,
                                config.negative_candidate,
                                device=device,
                            )
                            # Keep diagnostics from the prompt-only forward so every
                            # condition uses the same causal prefix as the paired primary.
                            live_diagnostics.clear()
                            intervened_log_probs = next_token_log_probabilities(
                                model, tokenizer, scoring_prompt, device=device
                            )
                    diagnostics = (
                        _aggregate_live_diagnostics(live_diagnostics)
                        if delivered_fraction != 0
                        else _swap_diagnostics(
                            residuals,
                            active_source,
                            active_target,
                            positions=intervention_positions,
                            swap_fraction=0.0,
                            mode=mode,
                        )
                    )
                    paired_norm = None
                    dose_match_error = None
                    if delivered_fraction != 0:
                        paired_norm = (
                            sum(
                                value * value
                                for value in paired_primary_norms.get(
                                    (mode, swap_fraction), {}
                                ).values()
                            ) ** 0.5
                            if has_controls else diagnostics["perturbation_norm"]
                        )
                        dose_match_error = abs(
                            diagnostics["perturbation_norm"] - paired_norm
                        ) / max(paired_norm, 1e-12)
                    rows.append(
                        {
                            "intervention_type": f"sector_coordinate_{mode}",
                            "source_prototype": config.source.name,
                            "target_prototype": config.target.name,
                            "direction_control": direction_control,
                            "position_control": position_control,
                            "dose_matched_control": dose_matched_control,
                            "dose_match_basis": (
                                "not_applicable"
                                if delivered_fraction == 0
                                else "paired_primary_layer_delta_norm"
                                if dose_matched_control else "primary"
                            ),
                            "paired_primary_perturbation_norm": paired_norm,
                            "dose_match_relative_error": dose_match_error,
                            "intervention_positions": list(intervention_positions),
                            "layers": list(config.layers),
                            "swap_fraction": swap_fraction,
                            "delivered_swap_fraction": delivered_fraction,
                            "loaded_positions": selected.to_dict(),
                            "clean_margin": clean_value,
                            "intervened_margin": margin.value,
                            "delta_margin": margin.value - clean_value,
                            "next_token_kl": categorical_kl_divergence(
                                clean_log_probs, intervened_log_probs
                            ),
                            "delivered_dose": diagnostics,
                            "score": _margin_dict(margin),
                        }
                    )
    return rows


def run_concept_gain_record(
    *,
    model: Any,
    tokenizer: Any,
    lens: Any,
    scoring_prompt: str,
    evidence_span: tuple[int, int],
    config: GainConfig,
    device: Any,
    prototypes: Mapping[int, torch.Tensor] | None = None,
    control_seed: int = 0,
) -> list[dict]:
    """Run an exact coordinate-gain sweep for one prototype and prompt."""
    directions = dict(
        prototypes
        or layer_prototypes(model, lens, config.prototype, config.layers)
    )
    if set(directions) != set(config.layers):
        raise ValueError("precomputed gain directions must cover all configured layers")
    clean_tensor = _input_tensor(tokenizer, scoring_prompt, device)
    residuals = record_residuals(model, clean_tensor, config.layers)
    selected = select_loaded_positions(
        residuals,
        directions,
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
    clean_log_probs = next_token_log_probabilities(
        model, tokenizer, scoring_prompt, device=device
    )
    positions_by_control = _position_control_sets(
        config.position_controls,
        evidence_span=evidence_span,
        selected=selected.positions,
        sequence_length=clean_tensor.shape[1],
        seed=control_seed,
    )
    directions_by_control = {"prototype": directions}
    if "matched_random" in config.direction_controls:
        directions_by_control["matched_random"] = {
            layer: matched_random_direction(
                direction, seed=control_seed + 1009 * offset
            )
            for offset, (layer, direction) in enumerate(sorted(directions.items()))
        }
    paired_primary_norms: dict[float, dict[int, float]] = {}
    has_controls = any(
        direction != "prototype" or position != "evidence"
        for direction in config.direction_controls
        for position in config.position_controls
    )
    if has_controls and selected.loaded:
        for gain in config.gains:
            if gain == 1:
                continue
            layer_norms: dict[int, float] = {}
            primary_transforms = {}
            for layer in config.layers:
                direction = directions[layer]

                def primary_transform(
                    tensor: torch.Tensor,
                    *,
                    vector=direction,
                    multiplier=gain,
                    positions=selected.positions,
                    layer_id=layer,
                ) -> torch.Tensor:
                    patched = coordinate_gain(
                        tensor,
                        positions=positions,
                        direction=vector,
                        gain=multiplier,
                    )
                    layer_norms[layer_id] = float(
                        (
                            patched[:, list(positions), :].float()
                            - tensor[:, list(positions), :].float()
                        ).norm().detach().cpu()
                    )
                    return patched

                primary_transforms[layer] = primary_transform
            with residual_interventions(model, primary_transforms), torch.inference_mode():
                model.forward(clean_tensor)
            if set(layer_norms) != set(config.layers):
                raise RuntimeError("paired primary gain forward missed intervention layers")
            paired_primary_norms[gain] = layer_norms
    rows = []
    for direction_control in config.direction_controls:
        active_directions = directions_by_control[direction_control]
        for position_control in config.position_controls:
            intervention_positions = positions_by_control[position_control]
            dose_matched_control = (
                direction_control != "prototype" or position_control != "evidence"
            )
            for gain in config.gains:
                transforms = {}
                live_diagnostics: dict[int, dict[str, float]] = {}
                if gain != 1 and selected.loaded:
                    for layer in config.layers:
                        direction = active_directions[layer]
                        paired_target_norm = (
                            paired_primary_norms[gain][layer]
                            if dose_matched_control else None
                        )

                        def transform(
                            tensor: torch.Tensor,
                            *,
                            vector=direction,
                            multiplier=gain,
                            positions=intervention_positions,
                            layer_id=layer,
                            target_norm=paired_target_norm,
                            match_control=dose_matched_control,
                        ) -> torch.Tensor:
                            patched = coordinate_gain(
                                tensor,
                                positions=positions,
                                direction=vector,
                                gain=multiplier,
                            )
                            if match_control:
                                patched = norm_match_intervention_to_target(
                                    tensor,
                                    patched,
                                    control_positions=positions,
                                    target_norm=float(target_norm),
                                )
                            live_diagnostics[layer_id] = _gain_diagnostics(
                                {layer_id: tensor.detach()},
                                {layer_id: vector},
                                positions=positions,
                                gain=multiplier,
                                patched_tensors={layer_id: patched.detach()},
                            )
                            return patched

                        transforms[layer] = transform
                delivered_gain = gain if selected.loaded else 1.0
                with residual_interventions(model, transforms):
                    if gain == 1 or not selected.loaded:
                        margin = clean_margin
                        intervened_log_probs = clean_log_probs
                    else:
                        margin = score_margin(
                            model,
                            tokenizer,
                            scoring_prompt,
                            config.positive_candidate,
                            config.negative_candidate,
                            device=device,
                        )
                        # Keep diagnostics from the prompt-only forward so every
                        # condition uses the same causal prefix as the paired primary.
                        live_diagnostics.clear()
                        intervened_log_probs = next_token_log_probabilities(
                            model, tokenizer, scoring_prompt, device=device
                        )
                diagnostics = (
                    _aggregate_live_diagnostics(live_diagnostics)
                    if delivered_gain != 1
                    else _gain_diagnostics(
                        residuals,
                        active_directions,
                        positions=intervention_positions,
                        gain=1.0,
                    )
                )
                paired_norm = None
                dose_match_error = None
                if delivered_gain != 1:
                    paired_norm = (
                        sum(
                            value * value
                            for value in paired_primary_norms.get(gain, {}).values()
                        ) ** 0.5
                        if has_controls else diagnostics["perturbation_norm"]
                    )
                    dose_match_error = abs(
                        diagnostics["perturbation_norm"] - paired_norm
                    ) / max(paired_norm, 1e-12)
                rows.append(
                    {
                        "intervention_type": (
                            "concept_token_gain"
                            if len(config.prototype.tokens) == 1
                            else "sector_prototype_gain"
                        ),
                        "prototype": config.prototype.name,
                        "token": (
                            config.prototype.tokens[0].token
                            if len(config.prototype.tokens) == 1
                            else None
                        ),
                        "token_id": (
                            config.prototype.tokens[0].token_id
                            if len(config.prototype.tokens) == 1
                            else None
                        ),
                        "direction_control": direction_control,
                        "position_control": position_control,
                        "dose_matched_control": dose_matched_control,
                        "dose_match_basis": (
                            "not_applicable"
                            if delivered_gain == 1
                            else "paired_primary_layer_delta_norm"
                            if dose_matched_control else "primary"
                        ),
                        "paired_primary_perturbation_norm": paired_norm,
                        "dose_match_relative_error": dose_match_error,
                        "intervention_positions": list(intervention_positions),
                        "layers": list(config.layers),
                        "gain": gain,
                        "delivered_gain": delivered_gain,
                        "loaded_positions": selected.to_dict(),
                        "clean_margin": clean_margin.value,
                        "intervened_margin": margin.value,
                        "delta_margin": margin.value - clean_margin.value,
                        "next_token_kl": categorical_kl_divergence(
                            clean_log_probs, intervened_log_probs
                        ),
                        "delivered_dose": diagnostics,
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


def run_token_screen_record(
    *,
    model: Any,
    tokenizer: Any,
    scoring_prompt: str,
    evidence_span: tuple[int, int],
    config: TokenScreenConfig,
    device: Any,
    directions: Mapping[int, Mapping[int, torch.Tensor]],
    control_seed: int = 0,
) -> list[dict]:
    """Run one formatted prompt across every frozen candidate and arm.

    The clean residuals and clean margin are computed once per prompt.  For
    each candidate the PRIMARY evidence positions are selected from the clean
    pass, a per-layer local scale is the median absolute concept coordinate at
    those positions (exact zero floored to 1e-6), and the delivered coordinate
    delta is ``alpha * local_scale / sqrt(n_layers)``.  A positive alpha adds
    the candidate direction; a negative alpha subtracts it.  The matched-random
    arm uses the same PRIMARY positions and scales with a seeded same-norm
    random direction, so no norm rescaling or extra position control is needed.
    """
    layers = list(config.layers)
    missing = [
        candidate.token_id
        for candidate in config.candidates
        if candidate.token_id not in directions
    ]
    if missing:
        raise ValueError(f"precomputed directions miss candidates: {missing}")
    clean_tensor = _input_tensor(tokenizer, scoring_prompt, device)
    residuals = record_residuals(model, clean_tensor, layers)
    clean_margin = score_single_token_margin_fp32(
        model,
        tokenizer,
        scoring_prompt,
        config.positive_candidate,
        config.negative_candidate,
        device=device,
    )
    rows = []
    divisor = len(layers) ** 0.5
    for candidate_index, candidate in enumerate(config.candidates):
        layer_directions = {layer: directions[candidate.token_id][layer] for layer in layers}
        selected = select_loaded_positions(
            residuals,
            layer_directions,
            evidence_span=evidence_span,
            top_k=config.top_positions,
            threshold=config.loading_threshold,
        )
        scales = {}
        for layer in layers:
            values = residuals[layer][0, list(selected.positions), :].float()
            coordinate = concept_coordinate(values, layer_directions[layer])
            scale = float(coordinate.abs().median())
            scales[layer] = scale if scale > 0.0 else 1e-6
        arm_directions: dict[str, Mapping[int, torch.Tensor]] = {"token": layer_directions}
        if "matched_random" in config.controls:
            arm_directions["matched_random"] = {
                layer: matched_random_direction(
                    layer_directions[layer],
                    seed=control_seed + 1009 * candidate_index + 7 * offset,
                )
                for offset, layer in enumerate(sorted(layers))
            }
        for arm in config.controls:
            active_directions = arm_directions[arm]
            for alpha in config.alphas:
                transforms = {}
                live_diagnostics: dict[int, dict[str, float]] = {}
                delivered_alpha = alpha if alpha != 0.0 and selected.loaded else 0.0
                if delivered_alpha != 0.0:
                    for layer in layers:
                        vector = active_directions[layer]
                        delta = delivered_alpha * scales[layer] / divisor

                        def transform(
                            tensor: torch.Tensor,
                            *,
                            vec=vector,
                            dose=delta,
                            layer_id=layer,
                            positions=selected.positions,
                        ) -> torch.Tensor:
                            patched = steer_positions(
                                tensor,
                                positions=positions,
                                direction=vec,
                                coordinate_delta=dose,
                            )
                            live_diagnostics[layer_id] = _screen_diagnostics(
                                {layer_id: tensor.detach()},
                                {layer_id: vec},
                                positions=positions,
                                coordinate_deltas={layer_id: dose},
                                patched_tensors={layer_id: patched.detach()},
                            )
                            return patched

                        transforms[layer] = transform
                with residual_interventions(model, transforms):
                    margin = (
                        clean_margin
                        if delivered_alpha == 0.0
                        else score_single_token_margin_fp32(
                            model,
                            tokenizer,
                            scoring_prompt,
                            config.positive_candidate,
                            config.negative_candidate,
                            device=device,
                        )
                    )
                diagnostics = (
                    _aggregate_live_diagnostics(live_diagnostics)
                    if delivered_alpha != 0.0
                    else _screen_diagnostics(
                        residuals,
                        active_directions,
                        positions=selected.positions,
                        coordinate_deltas={layer: 0.0 for layer in layers},
                    )
                )
                rows.append(
                    {
                        "intervention_type": "token_coordinate_screen",
                        "outcome_scoring": config.outcome_scoring,
                        "candidate": candidate.token,
                        "token_id": candidate.token_id,
                        "representation_side": candidate.representation_side,
                        "mean_positive": candidate.mean_positive,
                        "mean_negative": candidate.mean_negative,
                        "band_probability_diff": candidate.band_probability_diff,
                        "band_smoothed_log_ratio": candidate.band_smoothed_log_ratio,
                        "band_js_contribution": candidate.band_js_contribution,
                        "arm": arm,
                        "alpha": float(alpha),
                        "delivered_alpha": delivered_alpha,
                        "layers": layers,
                        "intervention_positions": list(selected.positions),
                        "loaded_positions": selected.to_dict(),
                        "coordinate_scale_min": min(scales.values()),
                        "coordinate_scale_mean": sum(scales.values()) / len(scales),
                        "coordinate_scale_max": max(scales.values()),
                        "clean_margin": clean_margin.value,
                        "intervened_margin": margin.value,
                        "delta_margin": margin.value - clean_margin.value,
                        "delivered_dose": diagnostics,
                        "score": _margin_dict(margin),
                    }
                )
    return rows


__all__ = [
    "layer_prototypes",
    "run_concept_gain_record",
    "run_swap_record",
    "run_token_screen_record",
    "run_token_steering_record",
    "token_screen_directions",
]
