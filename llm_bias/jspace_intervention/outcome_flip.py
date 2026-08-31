"""V2 outcome-conditioned decision-flip workflow (protocol Draft 1).

Implements the frozen protocol in ``docs/jspace-token-experiments/proposal-v2.md``:
an antisymmetric outcome-gradient axis fitted on discovery prompts, paired
dose-matched controls, held-out Buy/Sell decision flips as the primary
outcome, and deterministic full JSON generation as the required behavioral
validation.

Raw activations, direction vectors, and gradients are never persisted: every
calibration/test run recomputes the direction in memory and verifies the
frozen direction identity by hash, failing closed on any mismatch.  Artifacts
carry hashes, norms, compact dose diagnostics, flip outcomes, generated
decision text, and provenance only.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from llm_bias.core.analysis.statistics import (
    direction_hash,
    holm_bonferroni,
    paired_bootstrap_ci,
    sign_flip_pvalue,
)
from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import (
    continuation_token_ids,
    fp32_next_token_log_probs,
    score_single_token_margin_fp32,
)
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.generation import (
    GenerationConfig,
    finish_reason,
    generate_tokens,
)
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import input_ids, token_span
from llm_bias.jspace_intervention import transforms
from llm_bias.jspace_intervention.controls import (
    _torch_seed,
    matched_random_direction,
)
from llm_bias.jspace_intervention.pipeline import _iter_prompt_records
from llm_bias.jspace_intervention.prompting import (
    EVIDENCE_END,
    EVIDENCE_START,
    prepare_scoring_prompt,
)
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig

OUTCOME_FLIP_DEFAULT_PROMPT_COLUMN = "prompt_with_context_attribute_0"
OUTCOME_FLIP_ARTIFACT_TYPE = "outcome_flip_config"
ARM_ORDER = (
    "outcome",
    "matched_random",
    "label_permutation",
    "shuffled_evidence",
    "final_position",
)
_STEERING_ARMS = ARM_ORDER
_TIE_RULE = "exclude_exact_zero"
_FITTING_TARGET = "decision_position_next_token_logit_difference_fp32"
_AGGREGATION = "position_mean->unit_prompt->ticker_mean->equal_weight_ticker_mean->unit"
_SELECTION_RULE = "largest_dose_with_parse_success_then_paired_flip_rate"

_DECISION_RE = re.compile(r'"decision"\s*:\s*"(buy|sell)"', re.IGNORECASE)
_ITEM_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)


# ---------------------------------------------------------------------------
# Decision parsing and evidence positions
# ---------------------------------------------------------------------------


def parse_decision(text: str) -> str | None:
    """Parse a buy/sell decision from a generated JSON continuation."""
    match = _DECISION_RE.search(str(text))
    return match.group(1).lower() if match is not None else None


def evidence_item_char_spans(raw_prompt: str) -> list[tuple[int, int]]:
    """Character spans of the numbered items inside the evidence block."""
    start_marker = raw_prompt.find(EVIDENCE_START)
    if start_marker < 0:
        return []
    end_marker = raw_prompt.find(EVIDENCE_END, start_marker + len(EVIDENCE_START))
    if end_marker < 0:
        return []
    region_start = start_marker + len(EVIDENCE_START)
    region = raw_prompt[region_start:end_marker]
    matches = list(_ITEM_RE.finditer(region))
    spans: list[tuple[int, int]] = []
    for index, match in enumerate(matches):
        item_start = region_start + match.start()
        item_end = region_start + (
            matches[index + 1].start() if index + 1 < len(matches) else len(region)
        )
        while item_end > item_start and raw_prompt[item_end - 1] in " \t\r\n":
            item_end -= 1
        spans.append((item_start, item_end))
    return spans


def evidence_item_end_positions(
    tokenizer: Any,
    scoring_prompt: str,
    raw_prompt: str,
) -> list[int]:
    """Absolute token index of each evidence item's final token."""
    raw_offset = scoring_prompt.find(raw_prompt)
    if raw_offset < 0:
        raise ValueError(
            "formatted scoring prompt does not contain the raw user prompt"
        )
    spans = evidence_item_char_spans(raw_prompt)
    if not spans:
        raise ValueError("no numbered evidence items found in the evidence block")
    positions: set[int] = set()
    for start, end in spans:
        span = token_span(
            tokenizer,
            scoring_prompt,
            raw_offset + start,
            raw_offset + end,
            add_special_tokens=True,
        )
        if span is None:
            raise ValueError("evidence item does not map to token positions")
        positions.add(int(span[1]) - 1)
    return sorted(positions)


# ---------------------------------------------------------------------------
# Direction fitting (in-memory only; fail closed on identity mismatch)
# ---------------------------------------------------------------------------


def _candidate_token_id(tokenizer: Any, prompt: str, candidate: str) -> int:
    suffix = continuation_token_ids(tokenizer, prompt, candidate)[1]
    if len(suffix) != 1:
        raise ValueError(f"answer candidate {candidate!r} must be a single token")
    return int(suffix[0])


@torch.enable_grad()
def fit_prompt_layer_gradients(
    model: Any,
    input_tensor: torch.Tensor,
    *,
    fitted_layers: Sequence[int],
    positive_id: int,
    negative_id: int,
) -> dict[int, torch.Tensor]:
    """One grad-enabled forward; outcome gradient per fitted layer.

    Returns the full-sequence gradient of the decision-position logit
    difference ``log P(positive) - log P(negative)`` (FP32 final norm and
    unembedding) with respect to each fitted layer's residual output.  The
    returned CPU float32 tensors must stay in memory and never be persisted.

    The HFLensModel wrapper freezes every parameter, so a forward with
    frozen leaves builds no autograd graph at all.  The first fitted
    layer's output is therefore re-rooted as a leaf before downstream
    layers see it (the same convention as jlens
    ``ActivationRecorder.start_graph_at``), so the retained graph spans
    exactly the fitted layers.
    """
    layers = sorted(set(int(layer) for layer in fitted_layers))
    if not layers:
        raise ValueError("fitted_layers must be non-empty")
    decoder_layers = getattr(model, "layers", None)
    if decoder_layers is None:
        raise TypeError("model does not expose decoder layers")
    for layer in layers:
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(f"fitted layer {layer} is out of range")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    root_layer = layers[0]

    def make_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            if layer_id == root_layer:
                tensor.requires_grad_(True)
            captured[layer_id] = tensor
            return output

        return hook

    for layer in layers:
        handles.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))
    try:
        output = model.forward(input_tensor)
        final_residual = getattr(output, "last_hidden_state", None)
        if not torch.is_tensor(final_residual):
            raise TypeError("model forward must return last_hidden_state")
        log_probs = fp32_next_token_log_probs(model, final_residual[:, -1, :])
        target = log_probs[0, int(positive_id)] - log_probs[0, int(negative_id)]
        grads = torch.autograd.grad(
            target, [captured[layer] for layer in layers], retain_graph=False
        )
    finally:
        for handle in handles:
            handle.remove()
    result: dict[int, torch.Tensor] = {}
    for layer, grad in zip(layers, grads):
        if grad is None:
            raise ValueError(f"no gradient contributed to fitted layer {layer}")
        result[layer] = grad.detach().float().cpu()[0]
    return result


def _unit_position_mean(
    full: torch.Tensor, positions: Sequence[int]
) -> torch.Tensor | None:
    """Mean of a full-sequence vector over positions, unit-normalized."""
    if not positions:
        return None
    unique = sorted(set(int(position) for position in positions))
    selected = full[torch.as_tensor(unique, device=full.device)]
    mean = selected.mean(0)
    norm = float(mean.norm())
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return mean / norm


def label_permutation_signs(tickers: Sequence[str], seed: int) -> dict[str, int]:
    """Deterministic per-ticker ±1 signs for the label-permutation axis."""
    generator = torch.Generator(device="cpu").manual_seed(_torch_seed(int(seed)))
    signs: dict[str, int] = {}
    for ticker in sorted(set(str(ticker) for ticker in tickers)):
        signs[ticker] = (
            1 if torch.randint(0, 2, (1,), generator=generator).item() == 0 else -1
        )
    return signs


@dataclass(frozen=True)
class DirectionFit:
    """In-memory outcome-axis fit for every frozen position rule."""

    directions: dict[str, dict[int, torch.Tensor]]
    permutation_directions: dict[str, dict[int, torch.Tensor]]
    signs: dict[str, int]
    pre_unit_norm: dict[str, dict[int, float]]
    permutation_pre_unit_norm: dict[str, dict[int, float]]
    zero_prompt_counts: dict[str, dict[int, int]]
    valid_ticker_counts: dict[str, dict[int, int]]
    ticker_count: int
    record_count: int


def _aggregate_rule(
    prompt_vectors: list[tuple[str, dict[int, torch.Tensor | None]]],
    layers: Sequence[int],
    *,
    signs: Mapping[str, int] | None,
) -> tuple[dict[int, torch.Tensor], dict[int, float], dict[int, int], dict[int, int]]:
    """Aggregate per-prompt unit vectors with equal ticker weight."""
    layers = sorted(set(int(layer) for layer in layers))
    by_ticker: dict[str, dict[int, list[torch.Tensor]]] = {}
    zero_counts = {layer: 0 for layer in layers}
    for ticker, vectors in prompt_vectors:
        for layer in layers:
            vector = vectors.get(layer)
            if vector is None:
                zero_counts[layer] += 1
                continue
            if signs is not None:
                vector = vector * int(signs[ticker])
            by_ticker.setdefault(ticker, {}).setdefault(layer, []).append(vector)
    directions: dict[int, torch.Tensor] = {}
    pre_unit: dict[int, float] = {}
    valid_tickers: dict[int, int] = {}
    for layer in layers:
        ticker_means: list[torch.Tensor] = []
        for ticker in sorted(by_ticker):
            values = by_ticker[ticker].get(layer)
            if values:
                ticker_means.append(torch.stack(values).mean(0))
        if not ticker_means:
            raise ValueError(
                f"no valid discovery prompts contributed to layer {layer}; "
                "direction identity is fail-closed"
            )
        combined = torch.stack(ticker_means).mean(0)
        norm = float(combined.norm())
        if norm <= 0.0 or not math.isfinite(norm):
            raise ValueError(f"layer {layer} aggregated to a zero-norm axis")
        pre_unit[layer] = norm
        valid_tickers[layer] = len(ticker_means)
        directions[layer] = combined / norm
    return directions, pre_unit, zero_counts, valid_tickers


def fit_outcome_directions(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: OutcomeFlipConfig,
    device: Any,
) -> DirectionFit:
    """Fit the antisymmetric outcome axis for every frozen position rule.

    Fitting order follows the frozen protocol: position mean within a prompt,
    unit normalization per prompt gradient, ticker-mean of prompts, then an
    equal-weight mean over discovery tickers and a final normalization.  The
    label-permutation axis reruns the identical pipeline with deterministic
    per-ticker gradient signs.
    """
    if not records:
        raise ValueError("direction fitting requires discovery records")
    layers = list(config.fitted_layers)
    positive_id: int | None = None
    negative_id: int | None = None
    per_rule: dict[str, list[tuple[str, dict[int, torch.Tensor | None]]]] = {
        rule: [] for rule in config.position_rules
    }
    for record in records:
        scoring_prompt = record["scoring_prompt"]
        current_positive = _candidate_token_id(
            tokenizer, scoring_prompt, config.positive_candidate
        )
        current_negative = _candidate_token_id(
            tokenizer, scoring_prompt, config.negative_candidate
        )
        if positive_id is not None and positive_id != current_positive:
            raise ValueError("positive candidate token id is not stable across prompts")
        if negative_id is not None and negative_id != current_negative:
            raise ValueError("negative candidate token id is not stable across prompts")
        positive_id = current_positive
        negative_id = current_negative
        tensor = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
        grads = fit_prompt_layer_gradients(
            model,
            tensor,
            fitted_layers=layers,
            positive_id=current_positive,
            negative_id=current_negative,
        )
        for rule in config.position_rules:
            vectors: dict[int, torch.Tensor | None] = {}
            for layer in layers:
                vectors[layer] = _unit_position_mean(
                    grads[layer], record["positions"][rule]
                )
            per_rule[rule].append((record["ticker"], vectors))
    tickers = sorted(
        {ticker for ticker, _ in per_rule[next(iter(config.position_rules))]}
    )
    signs = label_permutation_signs(tickers, config.fitting_seed)
    directions: dict[str, dict[int, torch.Tensor]] = {}
    permutation_directions: dict[str, dict[int, torch.Tensor]] = {}
    pre_unit: dict[str, dict[int, float]] = {}
    permutation_pre_unit: dict[str, dict[int, float]] = {}
    zero_counts: dict[str, dict[int, int]] = {}
    valid_tickers: dict[str, dict[int, int]] = {}
    for rule in config.position_rules:
        directions[rule], pre_unit[rule], zero_counts[rule], valid_tickers[rule] = (
            _aggregate_rule(per_rule[rule], layers, signs=None)
        )
        permutation_directions[rule], permutation_pre_unit[rule], _, _ = (
            _aggregate_rule(per_rule[rule], layers, signs=signs)
        )
    return DirectionFit(
        directions=directions,
        permutation_directions=permutation_directions,
        signs=dict(sorted(signs.items())),
        pre_unit_norm=pre_unit,
        permutation_pre_unit_norm=permutation_pre_unit,
        zero_prompt_counts=zero_counts,
        valid_ticker_counts=valid_tickers,
        ticker_count=len(tickers),
        record_count=len(records),
    )


def _directions_identity(
    directions: Mapping[int, torch.Tensor],
    pre_unit: Mapping[int, float],
) -> dict[str, dict[str, float | str]]:
    return {
        str(layer): {
            "sha256": direction_hash(directions[layer]),
            "pre_unit_norm": float(pre_unit[layer]),
        }
        for layer in sorted(directions)
    }


def build_direction_identity(
    *,
    model_name: str,
    input_path: Path,
    split_manifest: Path,
    config_path: Path,
    config: OutcomeFlipConfig,
    prompt_columns: Sequence[str],
    records: list[dict[str, Any]],
    fit: DirectionFit,
) -> dict[str, Any]:
    """Compact direction identity: hashes, norms, and provenance only."""
    return {
        "artifact_type": "outcome_flip_direction_identity",
        "schema_version": 1,
        "split": "discovery",
        "model": model_name,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prompt_columns": sorted(prompt_columns),
        "record_ids": [record["record_id"] for record in records],
        "record_count": len(records),
        "ticker_count": fit.ticker_count,
        "fitted_layers": list(config.fitted_layers),
        "fitting_seed": config.fitting_seed,
        "fitting_target": _FITTING_TARGET,
        "aggregation": _AGGREGATION,
        "position_rules": list(config.position_rules),
        "directions": {
            rule: _directions_identity(fit.directions[rule], fit.pre_unit_norm[rule])
            for rule in config.position_rules
        },
        "label_permutation_directions": {
            rule: _directions_identity(
                fit.permutation_directions[rule],
                fit.permutation_pre_unit_norm[rule],
            )
            for rule in config.position_rules
        },
        "label_permutation_signs": dict(fit.signs),
        "zero_prompt_vector_counts": {
            rule: {
                str(layer): count
                for layer, count in fit.zero_prompt_counts[rule].items()
            }
            for rule in config.position_rules
        },
        "valid_ticker_counts": {
            rule: {
                str(layer): count
                for layer, count in fit.valid_ticker_counts[rule].items()
            }
            for rule in config.position_rules
        },
    }


def verify_direction_identity(
    *,
    model: Any,
    tokenizer: Any,
    discovery_records: list[dict[str, Any]],
    config: OutcomeFlipConfig,
    identity: Mapping[str, Any],
    device: Any,
) -> tuple[
    dict[str, dict[int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]],
]:
    """Recompute the discovery axis and require hash identity (fail closed)."""
    fit = fit_outcome_directions(
        model=model,
        tokenizer=tokenizer,
        records=discovery_records,
        config=config,
        device=device,
    )
    for rule in identity["position_rules"]:
        for key, section in (
            ("directions", fit.directions),
            ("label_permutation_directions", fit.permutation_directions),
        ):
            expected = identity[key][rule]
            for layer in identity["fitted_layers"]:
                actual = direction_hash(section[rule][int(layer)])
                if actual != expected[str(layer)]["sha256"]:
                    raise ValueError(
                        f"recomputed {key} identity mismatch for position rule "
                        f"{rule!r} layer {layer}; run is fail-closed"
                    )
    return fit.directions, fit.permutation_directions


# ---------------------------------------------------------------------------
# Combinations and paired record runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Combination:
    """One frozen candidate (band, position rule, relative dose)."""

    band_start: int
    band_end: int
    position_rule: str
    relative_dose: float

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(range(self.band_start, self.band_end + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "band": [self.band_start, self.band_end],
            "position_rule": self.position_rule,
            "relative_dose": self.relative_dose,
        }


def iter_combinations(config: OutcomeFlipConfig) -> list[Combination]:
    combinations: list[Combination] = []
    for start, end in config.candidate_bands:
        for rule in config.position_rules:
            for dose in config.dose_grid:
                combinations.append(Combination(start, end, rule, float(dose)))
    return combinations


def _shuffled_control_positions(
    evidence_span: tuple[int, int],
    primary: Sequence[int],
    *,
    seed: int,
) -> tuple[int, ...]:
    """Seeded shuffled evidence positions, disjoint from primary when possible.

    When the primary rule already covers the whole evidence span the
    exclusion leaves no candidates; the control then resamples the full span
    (documented fallback, still within the frozen evidence span).
    """
    start, end = evidence_span
    count = len(primary)
    if count == 0:
        raise ValueError("primary evidence positions are empty")
    candidates = [index for index in range(start, end) if index not in set(primary)]
    if len(candidates) < count:
        candidates = list(range(start, end))
    if len(candidates) < count:
        raise ValueError("evidence span has too few positions for the control")
    generator = torch.Generator(device="cpu").manual_seed(_torch_seed(int(seed)))
    order = torch.randperm(len(candidates), generator=generator)[:count].tolist()
    return tuple(sorted(candidates[index] for index in order))


def _positional_transform(
    vector: torch.Tensor,
    position_scales: Mapping[int, float],
    full_seq: int,
    delivered: dict[int, float] | None = None,
):
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] != full_seq:
            # Decode step of a cached generation: the intervention is defined
            # on the prompt (prefill) positions only.
            return tensor
        patched = transforms.steer_positions_scaled(
            tensor,
            positions=list(position_scales),
            direction=vector,
            position_scales=position_scales,
        )
        if delivered is not None:
            for position in position_scales:
                delivered[int(position)] = float(
                    (patched[0, int(position), :] - tensor[0, int(position), :])
                    .norm()
                )
        return patched

    return transform


def prepare_clean_record(
    *,
    model: Any,
    tokenizer: Any,
    record: Mapping[str, Any],
    config: OutcomeFlipConfig,
    device: Any,
) -> dict[str, Any]:
    """One clean pass per prompt: residuals at fitted layers plus the margin."""
    tensor = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
    captured = record_residuals(model, tensor, config.fitted_layers)
    clean = score_single_token_margin_fp32(
        model,
        tokenizer,
        record["scoring_prompt"],
        config.positive_candidate,
        config.negative_candidate,
        device=device,
    )
    value = float(clean.value)
    decision = "buy" if value > 0.0 else ("sell" if value < 0.0 else "tie")
    return {
        "residuals": captured,
        "clean": clean,
        "clean_value": value,
        "clean_decision": decision,
        "seq_len": int(len(record["prompt_ids"])),
    }


def _generate_decision(
    *,
    model: Any,
    tokenizer: Any,
    record: Mapping[str, Any],
    config: OutcomeFlipConfig,
    device: Any,
) -> dict[str, Any]:
    """Deterministic greedy continuation from the same scoring prompt."""
    prompt_ids = list(record["prompt_ids"])
    tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generation_model = SimpleNamespace(hf_model=getattr(model, "_hf_model", model))
    generated = generate_tokens(
        generation_model,
        tensor,
        GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        ),
    )
    new_ids = generated[0, len(prompt_ids):].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    hf_config = getattr(generation_model.hf_model, "config", None)
    eos = getattr(hf_config, "eos_token_id", None)
    decision = parse_decision(config.decision_prefix + text)
    return {
        "max_new_tokens": config.max_new_tokens,
        "finish_reason": finish_reason(
            new_ids, eos_token_id=eos, max_new_tokens=config.max_new_tokens
        ),
        "generated_text": text,
        "parsed_decision": decision,
        "parse_success": decision is not None,
    }


def run_outcome_flip_combination(
    *,
    model: Any,
    tokenizer: Any,
    record: Mapping[str, Any],
    config: OutcomeFlipConfig,
    combination: Combination,
    clean: Mapping[str, Any],
    device: Any,
    directions: Mapping[int, torch.Tensor],
    permutation_directions: Mapping[int, torch.Tensor],
    control_seed: int,
    generate_clean: bool = False,
    generate_outcome_all: bool = False,
    generate_outcome_flipped: bool = False,
) -> list[dict[str, Any]]:
    """Run every paired arm of one combination for one formatted prompt.

    The clean residuals and clean margin are computed once per prompt and
    shared across arms.  Every arm uses the same positions, local scales,
    and nominal dose; the only differences are the direction source, the
    position set, or a zero dose, so delivered relative perturbation is
    dose-matched by construction and reported per position.
    """
    layers = list(combination.layers)
    scoring_prompt = record["scoring_prompt"]
    evidence_span = tuple(record["evidence_span"])
    primary = list(record["positions"][combination.position_rule])
    seq_len = int(clean["seq_len"])
    residuals = clean["residuals"]
    rows: list[dict[str, Any]] = []

    if generate_clean:
        rows.append(
            {
                "arm": "clean",
                "sign": 0,
                "combination": combination.to_dict(),
                "layers": layers,
                "intervention_positions": [],
                "clean_margin": clean["clean_value"],
                "clean_decision": clean["clean_decision"],
                "intervened_margin": clean["clean_value"],
                "delta_margin": 0.0,
                "intervened_decision": clean["clean_decision"],
                "flipped": False,
                "flip_role": None,
                "position_scale_min": None,
                "position_scale_mean": None,
                "position_scale_max": None,
                "delivered_dose": {
                    "relative_perturbation_min": 0.0,
                    "relative_perturbation_mean": 0.0,
                    "relative_perturbation_max": 0.0,
                    "safety_bound": config.safety_bound,
                    "safety_ok": True,
                },
                "generation": _generate_decision(
                    model=model, tokenizer=tokenizer, record=record,
                    config=config, device=device,
                ),
            }
        )

    def position_scales(positions: Sequence[int]) -> dict[int, dict[int, float]]:
        scales: dict[int, dict[int, float]] = {}
        for layer in layers:
            values = residuals[layer][0, [int(p) for p in positions], :].float()
            norms = values.norm(dim=-1)
            scales[layer] = {
                int(position): max(float(norm), config.scale_floor)
                for position, norm in zip(positions, norms)
            }
        return scales

    for arm_index, arm in enumerate(_STEERING_ARMS):
        for sign in (1, -1):
            if arm == "outcome":
                arm_directions = {layer: directions[layer] for layer in layers}
                arm_positions = primary
            elif arm == "label_permutation":
                arm_directions = {
                    layer: permutation_directions[layer] for layer in layers
                }
                arm_positions = primary
            elif arm == "matched_random":
                arm_directions = {
                    layer: matched_random_direction(
                        directions[layer],
                        seed=control_seed + 1009 * arm_index + 7 * offset,
                    )
                    for offset, layer in enumerate(layers)
                }
                arm_positions = primary
            elif arm == "shuffled_evidence":
                arm_directions = {layer: directions[layer] for layer in layers}
                arm_positions = list(
                    _shuffled_control_positions(
                        evidence_span,
                        primary,
                        seed=control_seed + 1009 * arm_index + 7 * (sign + 2),
                    )
                )
            else:  # final_position
                arm_directions = {layer: directions[layer] for layer in layers}
                arm_positions = [seq_len - 1]
            scales = position_scales(arm_positions)
            signed_scales = {
                layer: {
                    position: float(sign) * combination.relative_dose * scale
                    for position, scale in per_position.items()
                }
                for layer, per_position in scales.items()
            }
            delivered_by_layer: dict[int, dict[int, float]] = {
                layer: {} for layer in layers
            }
            transformed = {
                layer: _positional_transform(
                    arm_directions[layer],
                    signed_scales[layer],
                    seq_len,
                    delivered=delivered_by_layer[layer],
                )
                for layer in layers
            }
            with residual_interventions(model, transformed):
                margin = score_single_token_margin_fp32(
                    model,
                    tokenizer,
                    scoring_prompt,
                    config.positive_candidate,
                    config.negative_candidate,
                    device=device,
                )
                intervened_value = float(margin.value)
                intervened_decision = (
                    "buy" if intervened_value > 0.0
                    else ("sell" if intervened_value < 0.0 else "tie")
                )
                flipped = (
                    clean["clean_decision"] != "tie"
                    and intervened_decision != "tie"
                    and clean["clean_decision"] != intervened_decision
                )
                flip_role = None
                if flipped:
                    toward = "buy" if intervened_decision == "buy" else "sell"
                    flip_role = (
                        "target" if (sign > 0) == (toward == "buy") else "reverse"
                    )
                needs_generation = arm == "outcome" and (
                    generate_outcome_all
                    or (generate_outcome_flipped and flip_role == "target")
                )
                generation = (
                    _generate_decision(
                        model=model, tokenizer=tokenizer, record=record,
                        config=config, device=device,
                    )
                    if needs_generation
                    else None
                )
            relatives: list[float] = []
            for layer in layers:
                for position in arm_positions:
                    applied = delivered_by_layer[layer].get(int(position))
                    if applied is None:
                        raise ValueError(
                            f"delivered dose was not recorded for layer {layer} "
                            f"position {position}; intervention did not apply"
                        )
                    relatives.append(abs(applied) / scales[layer][int(position)])
            flat_scales = [
                scale
                for per_position in scales.values()
                for scale in per_position.values()
            ]
            rows.append(
                {
                    "arm": arm,
                    "sign": sign,
                    "combination": combination.to_dict(),
                    "layers": layers,
                    "intervention_positions": sorted(set(arm_positions)),
                    "clean_margin": clean["clean_value"],
                    "clean_decision": clean["clean_decision"],
                    "intervened_margin": intervened_value,
                    "delta_margin": intervened_value - clean["clean_value"],
                    "intervened_decision": intervened_decision,
                    "flipped": flipped,
                    "flip_role": flip_role,
                    "position_scale_min": min(flat_scales) if flat_scales else None,
                    "position_scale_mean": (
                        sum(flat_scales) / len(flat_scales) if flat_scales else None
                    ),
                    "position_scale_max": max(flat_scales) if flat_scales else None,
                    "delivered_dose": {
                        "relative_perturbation_min": (
                            min(relatives) if relatives else 0.0
                        ),
                        "relative_perturbation_mean": (
                            sum(relatives) / len(relatives) if relatives else 0.0
                        ),
                        "relative_perturbation_max": (
                            max(relatives) if relatives else 0.0
                        ),
                        "safety_bound": config.safety_bound,
                        "safety_ok": (
                            (max(relatives) if relatives else 0.0)
                            <= config.safety_bound
                        ),
                    },
                    "score": margin.to_dict(),
                    "generation": generation,
                }
            )
    rows.append(
        {
            "arm": "noop",
            "sign": 0,
            "combination": combination.to_dict(),
            "layers": layers,
            "intervention_positions": [],
            "clean_margin": clean["clean_value"],
            "clean_decision": clean["clean_decision"],
            "intervened_margin": clean["clean_value"],
            "delta_margin": 0.0,
            "intervened_decision": clean["clean_decision"],
            "flipped": False,
            "flip_role": None,
            "position_scale_min": None,
            "position_scale_mean": None,
            "position_scale_max": None,
            "delivered_dose": {
                "relative_perturbation_min": 0.0,
                "relative_perturbation_mean": 0.0,
                "relative_perturbation_max": 0.0,
                "safety_bound": config.safety_bound,
                "safety_ok": True,
            },
            "score": clean["clean"].to_dict(),
            "generation": None,
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Analysis: flip estimands, controls, gates, calibration selection
# ---------------------------------------------------------------------------


def _stratum_labels(edges: Sequence[float]) -> list[str]:
    labels = [f"|M| <= {edges[0]}"]
    for low, high in zip(edges, edges[1:]):
        labels.append(f"{low} < |M| <= {high}")
    labels.append(f"|M| > {edges[-1]}")
    return labels


def _stratum_index(value: float, edges: Sequence[float]) -> int:
    magnitude = abs(value)
    for index, edge in enumerate(edges):
        if magnitude <= edge:
            return index
    return len(edges)


def _ticker_flip_rates(
    rows: list[dict[str, Any]], role: str | None
) -> dict[str, float]:
    by_ticker: dict[str, list[float]] = {}
    for row in rows:
        indicator = (
            1.0
            if (row["flipped"] and (role is None or row["flip_role"] == role))
            else 0.0
        )
        by_ticker.setdefault(row["ticker"], []).append(indicator)
    return {
        ticker: sum(values) / len(values) for ticker, values in by_ticker.items()
    }


def _equal_ticker_mean(rates: Mapping[str, float]) -> float | None:
    if not rates:
        return None
    return sum(rates.values()) / len(rates)


def _mcnemar_exact(pairs: list[tuple[bool, bool]]) -> float | None:
    """Two-sided exact paired test on per-record binary outcomes."""
    discordant_positive = sum(1 for a, b in pairs if a and not b)
    discordant_negative = sum(1 for a, b in pairs if b and not a)
    total = discordant_positive + discordant_negative
    if total == 0:
        return None

    def choose(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        k = min(k, n - k)
        result = 1
        for index in range(k):
            result = result * (n - index) // (index + 1)
        return result

    lower = min(discordant_positive, discordant_negative)
    mass = sum(choose(total, k) for k in range(lower + 1))
    return min(1.0, 2.0 * mass / (2 ** total))


def _direction_summary(
    combo_rows: list[dict[str, Any]],
    *,
    direction: str,
    config: OutcomeFlipConfig,
) -> dict[str, Any]:
    sign = 1 if direction == "buy" else -1
    target_decision = "buy" if direction == "buy" else "sell"
    opposite_decision = "sell" if direction == "buy" else "buy"
    eligible = [
        row
        for row in combo_rows
        if row["sign"] == sign and row["clean_decision"] == opposite_decision
    ]
    reverse_pool = [
        row
        for row in combo_rows
        if row["sign"] == sign and row["clean_decision"] == target_decision
    ]
    tie_count = sum(
        1
        for row in combo_rows
        if row["arm"] == "outcome"
        and row["sign"] == sign
        and row["clean_decision"] == "tie"
    )
    arms: dict[str, Any] = {}
    for arm in _STEERING_ARMS:
        target_rates = _ticker_flip_rates(
            [row for row in eligible if row["arm"] == arm], role="target"
        )
        reverse_rates = _ticker_flip_rates(
            [row for row in reverse_pool if row["arm"] == arm], role="reverse"
        )
        arms[arm] = {
            "eligible_ticker_count": len(target_rates),
            "eligible_record_count": sum(1 for row in eligible if row["arm"] == arm),
            "target_flip_rate": _equal_ticker_mean(target_rates),
            # An empty reverse pool means no reverse-eligible records exist,
            # so the observed reverse rate is exactly 0.0 (the gate is then
            # vacuously satisfiable rather than undefined).
            "reverse_flip_rate": _equal_ticker_mean(reverse_rates) or 0.0,
            "ticker_target_rates": {
                ticker: rate for ticker, rate in sorted(target_rates.items())
            },
        }
    specificity_point = None
    specificity_ci: list[float] | None = None
    specificity_p: float | None = None
    exact_paired_p: float | None = None
    outcome_rates = arms["outcome"]["ticker_target_rates"]
    random_rates = arms["matched_random"]["ticker_target_rates"]
    common_tickers = sorted(set(outcome_rates) & set(random_rates))
    if common_tickers:
        differences = [
            outcome_rates[ticker] - random_rates[ticker] for ticker in common_tickers
        ]
        specificity_point = sum(differences) / len(differences)
        specificity_ci = paired_bootstrap_ci(
            differences,
            seed=config.bootstrap_seed,
            n_resamples=config.bootstrap_samples,
        )
        specificity_p = sign_flip_pvalue(
            differences,
            seed=config.bootstrap_seed,
            n_resamples=config.bootstrap_samples,
            alternative="greater",
        )
        pair_counts = Counter(
            (row["ticker"], row["arm"])
            for row in eligible
            if row["arm"] in {"outcome", "matched_random"}
        )
        if pair_counts and all(count == 1 for count in pair_counts.values()):
            pairs = [
                (
                    bool(
                        next(
                            row["flipped"] and row["flip_role"] == "target"
                            for row in eligible
                            if row["ticker"] == ticker and row["arm"] == "outcome"
                        )
                    ),
                    bool(
                        next(
                            row["flipped"] and row["flip_role"] == "target"
                            for row in eligible
                            if row["ticker"] == ticker
                            and row["arm"] == "matched_random"
                        )
                    ),
                )
                for ticker in common_tickers
            ]
            exact_paired_p = _mcnemar_exact(pairs)
    labels = _stratum_labels(config.clean_margin_edges)
    strata = []
    for index, label in enumerate(labels):
        rows = [
            row
            for row in eligible
            if row["arm"] == "outcome"
            and _stratum_index(row["clean_margin"], config.clean_margin_edges)
            == index
        ]
        rates = _ticker_flip_rates(rows, role="target")
        strata.append(
            {
                "stratum": label,
                "record_count": len(rows),
                "target_flip_rate": _equal_ticker_mean(rates),
            }
        )
    generation_rows = [
        row
        for row in combo_rows
        if row["arm"] == "outcome" and row["sign"] == sign and row["generation"]
    ]
    parse_success = (
        sum(1 for row in generation_rows if row["generation"]["parse_success"])
        / len(generation_rows)
        if generation_rows
        else None
    )
    flipped_target_rows = [
        row for row in generation_rows if row["flipped"] and row["flip_role"] == "target"
    ]
    agreement = (
        sum(
            1
            for row in flipped_target_rows
            if row["generation"]["parsed_decision"] == target_decision
        )
        / len(flipped_target_rows)
        if flipped_target_rows
        else None
    )
    safety_max = max(
        (row["delivered_dose"]["relative_perturbation_max"] for row in combo_rows),
        default=0.0,
    )
    outcome_target = arms["outcome"]["target_flip_rate"]
    outcome_reverse = arms["outcome"]["reverse_flip_rate"]
    random_reverse = arms["matched_random"]["reverse_flip_rate"]
    return {
        "direction": direction,
        "target_decision": target_decision,
        "eligible_record_count": sum(1 for row in eligible if row["arm"] == "outcome"),
        "tie_excluded_count": tie_count,
        "arms": arms,
        "specificity": {
            "point": specificity_point,
            "ci95": specificity_ci,
            "p_one_sided": specificity_p,
            "exact_paired_p": exact_paired_p,
        },
        "reverse_not_excess": (
            outcome_reverse is not None
            and random_reverse is not None
            and outcome_reverse <= random_reverse + 1e-15
        ),
        "net_specificity": (
            outcome_target - outcome_reverse
            if outcome_target is not None and outcome_reverse is not None
            else None
        ),
        "margin_strata": strata,
        "generation": {
            "record_count": len(generation_rows),
            "parse_success_rate": parse_success,
            "flipped_target_count": len(flipped_target_rows),
            "decision_agreement": agreement,
        },
        "safety_max_relative_perturbation": safety_max,
        "safety_ok": safety_max <= config.safety_bound,
    }


def analyze_outcome_flip(
    rows: list[dict[str, Any]],
    *,
    config: OutcomeFlipConfig,
    split: str,
) -> dict[str, Any]:
    """Ticker-level flip estimands, controls, and frozen gates."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["arm"] in {"clean", "noop"}:
            continue
        grouped.setdefault(json.dumps(row["combination"], sort_keys=True), []).append(
            row
        )
    entries = [
        {
            "combination": json.loads(key),
            "directions": {
                direction: _direction_summary(
                    grouped[key], direction=direction, config=config
                )
                for direction in ("buy", "sell")
            },
        }
        for key in sorted(grouped)
    ]
    result: dict[str, Any] = {
        "split": split,
        "combinations": entries,
    }
    if split == "calibration":
        result["selection"] = _select_calibration(
            entries, config=config, raw_rows=rows
        )
    if split == "test":
        result["gates"] = _test_gates(entries, config=config)
    return result


def _test_gates(
    entries: list[dict[str, Any]], *, config: OutcomeFlipConfig
) -> dict[str, Any]:
    if len(entries) != 1:
        raise ValueError("test analysis requires exactly one frozen combination")
    directions = entries[0]["directions"]
    p_values = []
    direction_detail: dict[str, Any] = {}
    for direction, detail in directions.items():
        arms = detail["arms"]
        specific = detail["specificity"]
        p = specific["p_one_sided"]
        p_values.append(p if p is not None else 1.0)
        direction_detail[direction] = {
            "target_flip_rate": arms["outcome"]["target_flip_rate"],
            "specificity_ci95": specific["ci95"],
            "specificity_p_one_sided": p,
            "controls_below_min_rate": (
                (arms["matched_random"]["target_flip_rate"] or 0.0)
                < config.min_flip_rate
                and (arms["label_permutation"]["target_flip_rate"] or 0.0)
                < config.min_flip_rate
            ),
        }
    adjusted = holm_bonferroni(p_values)
    for direction, p in zip(directions, adjusted):
        direction_detail[direction]["holm_adjusted_p"] = p
    both_nonzero = all(
        (directions[d]["arms"]["outcome"]["target_flip_rate"] or 0.0) > 0.0
        for d in directions
    )
    ci_positive = all(
        directions[d]["specificity"]["ci95"] is not None
        and directions[d]["specificity"]["ci95"][0] > 0.0
        for d in directions
    )
    min_rate = all(
        (directions[d]["arms"]["outcome"]["target_flip_rate"] or 0.0)
        >= config.min_flip_rate
        for d in directions
    )
    reverse_ok = all(directions[d]["reverse_not_excess"] is True for d in directions)
    net_ok = all(
        directions[d]["net_specificity"] is not None
        and directions[d]["net_specificity"] > 0.0
        for d in directions
    )
    controls_ok = all(
        direction_detail[d]["controls_below_min_rate"] for d in directions
    )
    holm_ok = all(p < 0.05 for p in adjusted)
    parse_rates = [
        detail["generation"]["parse_success_rate"] for detail in directions.values()
    ]
    flipped_targets = sum(
        detail["generation"]["flipped_target_count"] for detail in directions.values()
    )
    parse_ok = (
        any(rate is not None for rate in parse_rates)
        and all(
            rate is not None and rate >= config.parse_success_gate
            for rate in parse_rates
        )
        and flipped_targets > 0
    )
    agreements = [
        detail["generation"]["decision_agreement"] for detail in directions.values()
    ]
    agreement_ok = (
        flipped_targets > 0
        and all(
            rate is not None and rate >= config.agreement_gate
            for rate in agreements
        )
    )
    safety_ok = all(detail["safety_ok"] for detail in directions.values())
    gates = {
        "both_directions_nonzero_target_flips": both_nonzero,
        "specificity_ci_lower_positive": ci_positive,
        "min_flip_rate": min_rate,
        "reverse_not_excess": reverse_ok,
        "net_specificity_positive": net_ok,
        "controls_below_min_rate": controls_ok,
        "generation_parse_success": parse_ok,
        "generation_agreement": agreement_ok,
        "safety_bound_respected": safety_ok,
        "holm_adjusted_specificity": holm_ok,
    }
    gates["success"] = all(gates.values())
    gates["directions"] = direction_detail
    return gates


def _select_calibration(
    entries: list[dict[str, Any]],
    *,
    config: OutcomeFlipConfig,
    raw_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Frozen Draft-1 selection: largest safe dose, then paired flip rate."""
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        combo = entry["combination"]
        combo_key = json.dumps(combo, sort_keys=True)
        combo_rows = [
            row
            for row in raw_rows
            if row["arm"] != "clean"
            and json.dumps(row["combination"], sort_keys=True) == combo_key
        ]
        generations = [
            row["generation"]
            for row in combo_rows
            if row["arm"] == "outcome" and row["generation"]
        ]
        parse_rate = (
            sum(1 for row in generations if row["parse_success"]) / len(generations)
            if generations
            else None
        )
        dose_ok = parse_rate is not None and parse_rate >= config.parse_success_gate
        directions = entry["directions"]
        paired_sum = (
            sum(directions[d]["specificity"]["point"] for d in directions)
            if all(
                directions[d]["specificity"]["point"] is not None
                for d in directions
            )
            else None
        )
        flips = {
            direction: sum(
                1
                for row in combo_rows
                if row["arm"] == "outcome"
                and row["sign"] == (1 if direction == "buy" else -1)
                and row["flipped"]
                and row["flip_role"] == "target"
            )
            for direction in ("buy", "sell")
        }
        candidates.append(
            {
                "combination": combo,
                "parse_success_rate": parse_rate,
                "dose_eligible": dose_ok,
                "paired_specificity_sum": paired_sum,
                "target_flips": flips,
                "both_directions_flipped": all(flips.values()),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["dose_eligible"] and item["paired_specificity_sum"] is not None
    ]
    if not eligible:
        raise ValueError(
            "no calibration combination passed the dose parse-success gate; "
            "V2 is fail-closed and does not enter test"
        )
    selected = max(
        eligible,
        key=lambda item: (
            item["paired_specificity_sum"],
            item["combination"]["relative_dose"],
            -item["combination"]["band"][0],
            -item["combination"]["band"][1],
            item["combination"]["position_rule"],
        ),
    )
    if not selected["both_directions_flipped"]:
        raise ValueError(
            "selected calibration combination lacks target flips in both "
            "directions; V2 is fail-closed and does not enter test"
        )
    return {
        "artifact_type": "outcome_flip_selection",
        "schema_version": 1,
        "split": "calibration",
        "selection_rule": _SELECTION_RULE,
        "selected": selected["combination"],
        "selected_parse_success_rate": selected["parse_success_rate"],
        "selected_paired_specificity_sum": selected["paired_specificity_sum"],
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Pipeline: prepare -> forward -> analyze -> finalize for each split
# ---------------------------------------------------------------------------


def _prompt_records(
    input_path: Path,
    *,
    assignments: Mapping[str, str],
    split_name: str,
    source_sector: str,
    prompt_columns: set[str],
    max_records: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _iter_prompt_records(
        input_path,
        assignments=assignments,
        split_name=split_name,
        source_sector=source_sector,
        prompt_columns=prompt_columns,
    ):
        if max_records is not None and len(records) >= max_records:
            break
        record["record_id"] = stable_record_id(
            record["ticker"], record["prompt_column"], split_name
        )
        records.append(record)
    return records


def _preflight_records(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    config: OutcomeFlipConfig,
    max_seq_len: int,
) -> None:
    """Validate every selected prompt before creating a persistent run."""
    if not records:
        raise ValueError("no prompt records match the requested sector and split")
    for record in records:
        scoring_prompt, _ = prepare_scoring_prompt(
            tokenizer, record["prompt"], decision_prefix=config.decision_prefix
        )
        length = len(input_ids(tokenizer, scoring_prompt, add_special_tokens=True))
        if length > max_seq_len:
            raise ValueError(
                f"formatted scoring prompt has {length} tokens, limit {max_seq_len}"
            )


def _materialize_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    config: OutcomeFlipConfig,
) -> list[dict[str, Any]]:
    """Attach scoring prompts, token ids, and frozen-rule positions."""
    materialized: list[dict[str, Any]] = []
    for record in records:
        scoring_prompt, evidence_span = prepare_scoring_prompt(
            tokenizer, record["prompt"], decision_prefix=config.decision_prefix
        )
        positions: dict[str, list[int]] = {}
        if "evidence_span_all" in config.position_rules:
            positions["evidence_span_all"] = list(
                range(int(evidence_span[0]), int(evidence_span[1]))
            )
        if "evidence_item_end" in config.position_rules:
            positions["evidence_item_end"] = evidence_item_end_positions(
                tokenizer, scoring_prompt, record["prompt"]
            )
        materialized.append(
            {
                **record,
                "scoring_prompt": scoring_prompt,
                "prompt_ids": input_ids(
                    tokenizer, scoring_prompt, add_special_tokens=True
                ),
                "evidence_span": [int(evidence_span[0]), int(evidence_span[1])],
                "positions": positions,
            }
        )
    return materialized


def _load_bound_payload(
    path: Path, *, label: str, expected_type: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != expected_type:
        raise ValueError(
            f"{label} has artifact_type {payload.get('artifact_type')!r}"
        )
    return payload


def _verify_bound_inputs(
    *,
    identity: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    input_path: Path,
    split_manifest: Path,
    config_path: Path,
    config: OutcomeFlipConfig,
    model_name: str,
) -> None:
    if identity is None:
        return
    if identity["model"] != model_name:
        raise ValueError("direction identity model does not match the run model")
    if identity["input_sha256"] != sha256_file(input_path):
        raise ValueError(
            "run input does not match the discovery input frozen in the identity"
        )
    if identity["split_manifest_sha256"] != sha256_file(split_manifest):
        raise ValueError("run split manifest does not match the discovery manifest")
    if identity["config_sha256"] != sha256_file(config_path):
        raise ValueError(
            "run config does not match the discovery config frozen in the identity"
        )
    if identity["fitted_layers"] != list(config.fitted_layers):
        raise ValueError("run config fitted_layers differ from the direction identity")
    if list(identity["position_rules"]) != list(config.position_rules):
        raise ValueError(
            "run config position_rules differ from the direction identity"
        )
    if selection is not None:
        if (
            selection.get("config_sha256") is not None
            and selection["config_sha256"] != sha256_file(config_path)
        ):
            raise ValueError("selection config does not match the run config")


def _discovery_verification_records(
    input_path: Path,
    *,
    assignments: Mapping[str, str],
    config: OutcomeFlipConfig,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Re-derive the exact discovery record set frozen in the identity."""
    wanted = set(identity["record_ids"])
    records = _prompt_records(
        input_path,
        assignments=assignments,
        split_name="discovery",
        source_sector=config.source_sector,
        prompt_columns=set(identity["prompt_columns"]),
        max_records=None,
    )
    derived = [record for record in records if record["record_id"] in wanted]
    if {record["record_id"] for record in derived} != wanted:
        raise ValueError(
            "current input no longer contains the exact discovery record set "
            "frozen in the direction identity; run is fail-closed"
        )
    order = {record_id: index for index, record_id in enumerate(identity["record_ids"])}
    derived.sort(key=lambda record: order[record["record_id"]])
    return derived


def _prepare_stage(
    run: ArtifactRun,
    *,
    records: list[dict[str, Any]],
    split_name: str,
    config: OutcomeFlipConfig,
    prompt_columns: Sequence[str],
    identity: Mapping[str, Any] | None,
) -> int:
    prepare_dir = run.run_directory / "prepare"
    prompt_records_path = prepare_dir / "prompt_records.jsonl"
    prepare_metadata_path = prepare_dir / "metadata.json"
    with run.stage("prepare") as stage:
        count = write_jsonl(
            prompt_records_path,
            (
                {
                    "schema_version": 1,
                    "artifact_type": "outcome_flip_prompt_record",
                    "record_id": record["record_id"],
                    "ticker": record["ticker"],
                    "name": record["name"],
                    "sector": record["sector"],
                    "marketcap": record["marketcap"],
                    "prompt_column": record["prompt_column"],
                    "split": split_name,
                    "prompt": record["prompt"],
                }
                for record in records
            ),
            overwrite=False,
        )
        metadata: dict[str, Any] = {
            "artifact_type": "outcome_flip_prepare_metadata",
            "schema_version": 1,
            "split": split_name,
            "source_sector": config.source_sector,
            "prompt_columns": sorted(prompt_columns),
            "fitted_layers": list(config.fitted_layers),
            "candidate_bands": [list(band) for band in config.candidate_bands],
            "position_rules": list(config.position_rules),
            "dose_grid": list(config.dose_grid),
            "safety_bound": config.safety_bound,
            "scale_floor": config.scale_floor,
            "tie_rule": config.tie_rule,
            "clean_margin_edges": list(config.clean_margin_edges),
            "min_flip_rate": config.min_flip_rate,
            "parse_success_gate": config.parse_success_gate,
            "agreement_gate": config.agreement_gate,
            "max_new_tokens": config.max_new_tokens,
            "fitting_seed": config.fitting_seed,
            "record_count": count,
        }
        if identity is not None:
            metadata["direction_identity"] = identity["input"]
        write_metadata(prepare_metadata_path, metadata, overwrite=False)
        stage.count(count)
    run.manifest.register_artifact(
        prompt_records_path,
        artifact_type="outcome_flip_prompt_record",
        stage="prepare",
        role="output",
        record_count=count,
    )
    run.manifest.register_artifact(
        prepare_metadata_path,
        artifact_type="outcome_flip_prepare_metadata",
        stage="prepare",
        role="output",
    )
    run.manifest.save()
    return count


def _forward_stage(
    run: ArtifactRun,
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    config: OutcomeFlipConfig,
    split_name: str,
    records: list[dict[str, Any]],
    combinations: list[Combination],
    directions: Mapping[str, Mapping[int, torch.Tensor]] | None,
    permutation_directions: Mapping[str, Mapping[int, torch.Tensor]] | None,
    identity: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
) -> tuple[Path, int]:
    forward_dir = run.run_directory / "forward"
    results_path = forward_dir / "outcome_flip_results.jsonl"
    forward_metadata_path = forward_dir / "metadata.json"

    def rows() -> Any:
        for record in records:
            control_seed = int(record["record_id"].rsplit("_", 1)[-1], 16)
            clean = prepare_clean_record(
                model=model,
                tokenizer=tokenizer,
                record=record,
                config=config,
                device=device,
            )
            for combination in combinations:
                rule = combination.position_rule
                if directions is None or rule not in directions:
                    raise ValueError(
                        f"missing verified direction for position rule {rule!r}"
                    )
                record_rows = run_outcome_flip_combination(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    config=config,
                    combination=combination,
                    clean=clean,
                    device=device,
                    directions=directions[rule],
                    permutation_directions=permutation_directions[rule],
                    control_seed=control_seed,
                    generate_clean=split_name == "calibration",
                    generate_outcome_all=split_name == "calibration",
                    generate_outcome_flipped=split_name == "test",
                )
                for result in record_rows:
                    yield {
                        "schema_version": 1,
                        "artifact_type": "outcome_flip_result",
                        "record_id": record["record_id"],
                        "ticker": record["ticker"],
                        "sector": record["sector"],
                        "prompt_column": record["prompt_column"],
                        "split": split_name,
                        **result,
                    }

    with run.stage("forward") as stage:
        count = write_jsonl(results_path, rows(), overwrite=False)
        stage.count(count)
    metadata: dict[str, Any] = {
        "artifact_type": "outcome_flip_metadata",
        "schema_version": 1,
        "split": split_name,
        "combination_count": len(combinations),
        "arms": list(_STEERING_ARMS) + ["noop"],
        "arms_per_combination": 2 * len(_STEERING_ARMS) + 1
        + (1 if split_name == "calibration" else 0),
        "tie_rule": config.tie_rule,
        "outcome_scoring": "single_token_fp32_final_norm_unembedding_next_token",
        "fitting_target": _FITTING_TARGET,
        "direction_recomputed": directions is not None,
        "generation": {
            "deterministic_greedy": True,
            "max_new_tokens": config.max_new_tokens,
            "clean_records": split_name == "calibration",
            "outcome_records": (
                "all" if split_name == "calibration" else "target_flipped"
            ),
        },
        "record_count": count,
    }
    if identity is not None:
        metadata["direction_identity"] = identity["input"]
    if selection is not None:
        metadata["calibration_selection"] = selection.get("selected")
    write_metadata(forward_metadata_path, metadata, overwrite=False)
    run.manifest.register_artifact(
        results_path,
        artifact_type="outcome_flip_result",
        stage="forward",
        role="output",
        record_count=count,
    )
    run.manifest.register_artifact(
        forward_metadata_path,
        artifact_type="outcome_flip_metadata",
        stage="forward",
        role="output",
    )
    run.manifest.save()
    return results_path, count


def _run_outcome_flip_pipeline(
    *,
    input_path: Path,
    split_manifest: Path,
    config_path: Path,
    model_name: str,
    run_id: str,
    dataset: str,
    artifact_root: Path,
    split_name: str,
    identity: Mapping[str, Any] | None,
    identity_path: Path | None,
    selection: Mapping[str, Any] | None,
    selection_path: Path | None,
    prompt_columns: set[str],
    max_records: int | None,
    max_seq_len: int,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("artifact_type") != OUTCOME_FLIP_ARTIFACT_TYPE:
        raise ValueError(
            f"config artifact_type must be {OUTCOME_FLIP_ARTIFACT_TYPE!r}"
        )
    outcome_config = OutcomeFlipConfig.from_dict(config)
    if outcome_config.model != model_name:
        raise ValueError(
            f"run model {model_name!r} does not match model "
            f"{outcome_config.model!r} frozen in config"
        )
    if outcome_config.split_manifest_sha256 != sha256_file(split_manifest):
        raise ValueError(
            "run split manifest does not match the manifest frozen in config"
        )
    if split_name == "discovery":
        if identity_path is not None or selection_path is not None:
            raise ValueError("discovery runs take no identity or selection inputs")
    else:
        if identity is None:
            raise ValueError(f"{split_name} runs require --direction-identity")
        if split_name == "test" and selection is None:
            raise ValueError("test runs require --calibration-selection")
        _verify_bound_inputs(
            identity=identity,
            selection=selection,
            input_path=input_path,
            split_manifest=split_manifest,
            config_path=config_path,
            config=outcome_config,
            model_name=model_name,
        )

    split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    assignments = {str(k): str(v) for k, v in split_payload["assignments"].items()}
    records = _prompt_records(
        input_path,
        assignments=assignments,
        split_name=split_name,
        source_sector=outcome_config.source_sector,
        prompt_columns=prompt_columns,
        max_records=max_records,
    )
    preflight_tokenizer = load_tokenizer(model_name)
    _preflight_records(
        preflight_tokenizer,
        records,
        config=outcome_config,
        max_seq_len=max_seq_len,
    )
    verification_records: list[dict[str, Any]] | None = None
    if identity is not None:
        verification_records = _discovery_verification_records(
            input_path,
            assignments=assignments,
            config=outcome_config,
            identity=identity,
        )
        _preflight_records(
            preflight_tokenizer,
            verification_records,
            config=outcome_config,
            max_seq_len=max_seq_len,
        )
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(
        input_path, artifact_type="trial_plan_prompts", stage="prepare", role="input"
    )
    run.manifest.register_artifact(
        split_manifest,
        artifact_type="jspace_intervention_splits",
        stage="prepare",
        role="input",
    )
    run.manifest.register_artifact(
        config_path,
        artifact_type=OUTCOME_FLIP_ARTIFACT_TYPE,
        stage="prepare",
        role="input",
    )
    if identity_path is not None:
        run.manifest.register_artifact(
            identity_path,
            artifact_type="outcome_flip_direction_identity",
            stage="prepare",
            role="input",
        )
    if selection_path is not None:
        run.manifest.register_artifact(
            selection_path,
            artifact_type="outcome_flip_selection",
            stage="prepare",
            role="input",
        )
    run.manifest.save()
    try:
        _prepare_stage(
            run,
            records=records,
            split_name=split_name,
            config=outcome_config,
            prompt_columns=prompt_columns,
            identity=identity,
        )
        model, tokenizer, fallback_device = load_model(model_name)
        device = getattr(model, "input_device", fallback_device)
        if split_name == "discovery":
            materialized = _materialize_records(records, tokenizer, outcome_config)
            fit = fit_outcome_directions(
                model=model,
                tokenizer=tokenizer,
                records=materialized,
                config=outcome_config,
                device=device,
            )
            identity_payload = build_direction_identity(
                model_name=model_name,
                input_path=input_path,
                split_manifest=split_manifest,
                config_path=config_path,
                config=outcome_config,
                prompt_columns=prompt_columns,
                records=records,
                fit=fit,
            )
            forward_dir = run.run_directory / "forward"
            identity_output = forward_dir / "direction_identity.json"
            forward_metadata_path = forward_dir / "metadata.json"
            with run.stage("forward") as stage:
                write_json(identity_output, identity_payload, overwrite=False)
                write_metadata(
                    forward_metadata_path,
                    {
                        "artifact_type": "outcome_flip_metadata",
                        "schema_version": 1,
                        "split": "discovery",
                        "operation": "direction_fitting",
                        "fitting_target": _FITTING_TARGET,
                        "aggregation": _AGGREGATION,
                        "autograd": True,
                        "backpropagation": True,
                        "fitted_layers": list(outcome_config.fitted_layers),
                        "position_rules": list(outcome_config.position_rules),
                        "record_count": fit.record_count,
                        "ticker_count": fit.ticker_count,
                    },
                    overwrite=False,
                )
                stage.count(fit.record_count)
            run.manifest.register_artifact(
                identity_output,
                artifact_type="outcome_flip_direction_identity",
                stage="forward",
                role="output",
            )
            run.manifest.register_artifact(
                forward_metadata_path,
                artifact_type="outcome_flip_metadata",
                stage="forward",
                role="output",
            )
            run.manifest.save()
            analyze_dir = run.run_directory / "analyze"
            fit_summary_path = analyze_dir / "fit_summary.json"
            analyze_metadata_path = analyze_dir / "metadata.json"
            with run.stage("analyze") as stage:
                write_json(
                    fit_summary_path,
                    {
                        "artifact_type": "outcome_flip_fit_summary",
                        "schema_version": 1,
                        "forward_identity": "forward/direction_identity.json",
                        "forward_identity_sha256": sha256_file(identity_output),
                        "record_count": fit.record_count,
                        "ticker_count": fit.ticker_count,
                        "zero_prompt_vector_counts": identity_payload[
                            "zero_prompt_vector_counts"
                        ],
                        "valid_ticker_counts": identity_payload[
                            "valid_ticker_counts"
                        ],
                        "pre_unit_norm": {
                            rule: {
                                str(layer): value
                                for layer, value in fit.pre_unit_norm[rule].items()
                            }
                            for rule in outcome_config.position_rules
                        },
                        "label_permutation_signs": fit.signs,
                    },
                    overwrite=False,
                )
                write_metadata(
                    analyze_metadata_path,
                    {
                        "artifact_type": "outcome_flip_analysis_metadata",
                        "schema_version": 1,
                        "interpretation": "discovery_fitting",
                        "record_count": fit.record_count,
                    },
                    overwrite=False,
                )
                stage.count(fit.record_count)
            run.manifest.register_artifact(
                fit_summary_path,
                artifact_type="outcome_flip_fit_summary",
                stage="analyze",
                role="output",
            )
            run.manifest.register_artifact(
                analyze_metadata_path,
                artifact_type="outcome_flip_analysis_metadata",
                stage="analyze",
                role="output",
            )
            run.manifest.save()
            run.finalize(required_stages={"prepare", "forward", "analyze"})
            return run.run_directory
        materialized = _materialize_records(records, tokenizer, outcome_config)
        verification_materialized = _materialize_records(
            verification_records, tokenizer, outcome_config
        )
        directions, permutation_directions = verify_direction_identity(
            model=model,
            tokenizer=tokenizer,
            discovery_records=verification_materialized,
            config=outcome_config,
            identity=identity,
            device=device,
        )
        if split_name == "calibration":
            combinations = iter_combinations(outcome_config)
        else:
            selected = selection["selected"]
            frozen = iter_combinations(outcome_config)
            combination = Combination(
                band_start=int(selected["band"][0]),
                band_end=int(selected["band"][1]),
                position_rule=str(selected["position_rule"]),
                relative_dose=float(selected["relative_dose"]),
            )
            if combination not in frozen:
                raise ValueError(
                    "calibration selection combination is not in the frozen "
                    "candidate set"
                )
            combinations = [combination]
        results_path, _ = _forward_stage(
            run,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=outcome_config,
            split_name=split_name,
            records=materialized,
            combinations=combinations,
            directions=directions,
            permutation_directions=permutation_directions,
            identity=identity,
            selection=selection,
        )
        analyze_dir = run.run_directory / "analyze"
        rows = [
            json.loads(line)
            for line in results_path.open(encoding="utf-8")
            if line.strip()
        ]
        summary = analyze_outcome_flip(
            rows, config=outcome_config, split=split_name
        )
        selection_payload: dict[str, Any] | None = None
        selection_output: Path | None = None
        if split_name == "calibration":
            summary_path = analyze_dir / "calibration_summary.json"
            selection_output = analyze_dir / "outcome_flip_selection.json"
            selection_payload = dict(summary["selection"])
            selection_payload["config_sha256"] = sha256_file(config_path)
            selection_payload["direction_identity_sha256"] = sha256_file(identity_path)
        else:
            summary_path = analyze_dir / "outcome_flip_analysis.json"
        with run.stage("analyze") as stage:
            write_json(
                summary_path,
                {
                    "artifact_type": (
                        "outcome_flip_calibration_summary"
                        if split_name == "calibration"
                        else "outcome_flip_analysis"
                    ),
                    "schema_version": 1,
                    "forward": "forward/outcome_flip_results.jsonl",
                    "forward_sha256": sha256_file(results_path),
                    **summary,
                },
                overwrite=False,
            )
            if selection_output is not None and selection_payload is not None:
                write_json(selection_output, selection_payload, overwrite=False)
            write_metadata(
                analyze_dir / "metadata.json",
                {
                    "artifact_type": "outcome_flip_analysis_metadata",
                    "schema_version": 1,
                    "interpretation": (
                        "calibration_selection"
                        if split_name == "calibration"
                        else "held_out_test"
                    ),
                    "record_count": len(rows),
                    **(
                        {"gates": summary["gates"]}
                        if "gates" in summary
                        else {}
                    ),
                },
                overwrite=False,
            )
            stage.count(len(rows))
        run.manifest.register_artifact(
            summary_path,
            artifact_type=(
                "outcome_flip_calibration_summary"
                if split_name == "calibration"
                else "outcome_flip_analysis"
            ),
            stage="analyze",
            role="output",
        )
        if selection_output is not None:
            run.manifest.register_artifact(
                selection_output,
                artifact_type="outcome_flip_selection",
                stage="analyze",
                role="output",
            )
        run.manifest.register_artifact(
            analyze_dir / "metadata.json",
            artifact_type="outcome_flip_analysis_metadata",
            stage="analyze",
            role="output",
        )
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def _enable_deterministic_gpu() -> None:
    """Enable deterministic GPU algorithms before the first CUDA use.

    The direction-identity protocol hashes raw float32 bytes, so direction
    fitting must be bit-exact reproducible across processes.  bf16 GEMM
    backward is non-deterministic by default, so the whole V2 run (fitting,
    scoring, generation) runs under ``torch.use_deterministic_algorithms``.
    The cuBLAS workspace env var must be set before cuBLAS is initialized.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)


def run_outcome_flip_pipeline(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    config_path: str | Path,
    model_name: str,
    run_id: str,
    dataset: str = "jspace-outcome-direction-flip",
    artifact_root: str | Path = "artifacts",
    split_name: str = "discovery",
    direction_identity_path: str | Path | None = None,
    calibration_selection_path: str | Path | None = None,
    max_records: int | None = None,
    max_seq_len: int = 1024,
    prompt_columns: set[str] | None = None,
) -> Path:
    """Run one V2 split into a canonical run tree.

    Discovery fits the outcome axis and emits the direction identity;
    calibration verifies it, sweeps the frozen candidate combinations, and
    emits the frozen selection; test verifies both and runs the single frozen
    combination with the full gate set.
    """
    _enable_deterministic_gpu()
    input_path = Path(input_path)
    split_manifest = Path(split_manifest)
    config_path = Path(config_path)
    for path, label in (
        (input_path, "baseline prompt CSV"),
        (split_manifest, "split manifest"),
        (config_path, "outcome flip config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    identity: Mapping[str, Any] | None = None
    identity_path: Path | None = None
    if direction_identity_path is not None:
        identity_path = Path(direction_identity_path)
        identity = _load_bound_payload(
            identity_path,
            label="direction identity",
            expected_type="outcome_flip_direction_identity",
        )
        if identity.get("split") != "discovery":
            raise ValueError("direction identity must come from a discovery run")
    selection: Mapping[str, Any] | None = None
    selection_path: Path | None = None
    if calibration_selection_path is not None:
        selection_path = Path(calibration_selection_path)
        selection = _load_bound_payload(
            selection_path,
            label="calibration selection",
            expected_type="outcome_flip_selection",
        )
        if selection.get("split") != "calibration":
            raise ValueError(
                "calibration selection must come from a calibration run"
            )
        bound = selection.get("direction_identity_sha256")
        if identity_path is None or bound is None or bound != sha256_file(identity_path):
            raise ValueError(
                "selection does not bind the provided direction identity"
            )
    resolved = (
        set(prompt_columns)
        if prompt_columns
        else {OUTCOME_FLIP_DEFAULT_PROMPT_COLUMN}
    )
    return _run_outcome_flip_pipeline(
        input_path=input_path,
        split_manifest=split_manifest,
        config_path=config_path,
        model_name=model_name,
        run_id=run_id,
        dataset=dataset,
        artifact_root=Path(artifact_root),
        split_name=split_name,
        identity=identity,
        identity_path=identity_path,
        selection=selection,
        selection_path=selection_path,
        prompt_columns=resolved,
        max_records=max_records,
        max_seq_len=max_seq_len,
    )


__all__ = [
    "ARM_ORDER",
    "OUTCOME_FLIP_ARTIFACT_TYPE",
    "OUTCOME_FLIP_DEFAULT_PROMPT_COLUMN",
    "Combination",
    "DirectionFit",
    "analyze_outcome_flip",
    "build_direction_identity",
    "evidence_item_char_spans",
    "evidence_item_end_positions",
    "fit_outcome_directions",
    "fit_prompt_layer_gradients",
    "iter_combinations",
    "label_permutation_signs",
    "parse_decision",
    "prepare_clean_record",
    "run_outcome_flip_combination",
    "run_outcome_flip_pipeline",
    "verify_direction_identity",
]
