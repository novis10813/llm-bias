"""B V1 cross-sector context overriding under negative evidence.

This workflow consumes the prepared cross-sector records produced by
``cross_sector_patching``. It filters those records to negative evidence and
uses the shared activation-patching primitives to patch one span at one layer.
Residual caches stay in memory and only compact margin/provenance records reach
artifacts.
"""
from __future__ import annotations

import bisect
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import sha256_file, sha256_json, stable_record_id
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import (
    CandidateMargin,
    continuation_token_ids,
    score_single_token_margin_fp32,
)
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.jspace_intervention.activation_patching import (
    cache_source_residuals,
    run_activation_patching_record,
)
from llm_bias.jspace_intervention.cross_sector_patching import (
    PAIRS_ARTIFACT_TYPE,
    _prepare_scoring_prompt,
)
from llm_bias.jspace_intervention.valence import PROMPT_TEMPLATE_VERSION

ARTIFACT_SCHEMA_VERSION = 1
RECORD_ARTIFACT_TYPE = "cross_sector_context_overriding_record"
ANALYSIS_ARTIFACT_TYPE = "cross_sector_context_overriding_analysis"
CONFIRMATION_ARTIFACT_TYPE = "cross_sector_context_overriding_confirmation_evaluation"
PREPARE_ARTIFACT_TYPE = "cross_sector_context_overriding_prepare_metadata"
DEFAULT_DATASET = "cross-sector-context-overriding"
DEFAULT_DECISION_PREFIX = '{\n  "decision": "'
DEFAULT_POSITIVE_CANDIDATE = "buy"
DEFAULT_NEGATIVE_CANDIDATE = "sell"
DEFAULT_LAYERS = tuple(range(14, 22))
DEFAULT_SPANS = ("instruction_context", "header", "final_position")
SPAN_CONDITIONS = frozenset(DEFAULT_SPANS)


def load_prepared_context_records(path: str | Path) -> list[dict[str, Any]]:
    """Load Direction A's prepared artifact and retain negative records only."""
    rows = read_jsonl(path)
    for row in rows:
        if row.get("artifact_type") != PAIRS_ARTIFACT_TYPE:
            raise ValueError(
                f"prepared record is not {PAIRS_ARTIFACT_TYPE}: {path}"
            )
    return [row for row in rows if row.get("evidence_valence") == "negative"]


def _validate_spans(spans: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(span) for span in spans))
    if not result:
        raise ValueError("at least one span is required")
    unknown = set(result) - SPAN_CONDITIONS
    if unknown:
        raise ValueError(f"unknown context overriding spans: {sorted(unknown)}")
    return result


def _normalized_transfer(source: float, target: float, delta: float) -> float | None:
    denominator = source - target
    if denominator == 0.0:
        return None
    value = delta / denominator
    return value if math.isfinite(value) else None


def _direction_name(
    source_identity: Mapping[str, Any], target_identity: Mapping[str, Any]
) -> str:
    return f"{source_identity['sector']}_to_{target_identity['sector']}"


def run_context_overriding_record(
    *,
    model: Any,
    tokenizer: Any,
    prepared_record: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    target_identity: Mapping[str, Any],
    source_prompt: str,
    target_prompt: str,
    source_spans: Mapping[str, Sequence[int]],
    target_spans: Mapping[str, Sequence[int]],
    layer: int,
    span: str,
    source_residuals: Mapping[int, Any],
    source_clean_margin: CandidateMargin,
    target_clean_margin: CandidateMargin,
    positive_candidate: str,
    negative_candidate: str,
    device: Any,
    direction: str,
) -> dict[str, Any]:
    """Patch one prepared arm/span/layer and return its compact record."""
    span = str(span)
    if span not in SPAN_CONDITIONS:
        raise ValueError(f"unknown context overriding span: {span!r}")
    result = run_activation_patching_record(
        model=model,
        tokenizer=tokenizer,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        source_evidence_char_spans={key: list(value) for key, value in source_spans.items()},
        target_evidence_char_spans={key: list(value) for key, value in target_spans.items()},
        layers=[int(layer)],
        span_condition=span,
        positive_candidate=positive_candidate,
        negative_candidate=negative_candidate,
        device=device,
        source_residuals={int(layer): source_residuals[int(layer)]},
        source_clean_margin=source_clean_margin,
        target_clean_margin=target_clean_margin,
    )
    source_margin = float(source_clean_margin.value)
    target_margin = float(target_clean_margin.value)
    delta = float(result["delta_margin"])
    source_ticker = str(source_identity["ticker"])
    target_ticker = str(target_identity["ticker"])
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": RECORD_ARTIFACT_TYPE,
        "record_id": stable_record_id(
            "cross-sector-context-overriding",
            prepared_record["record_id"],
            source_ticker,
            target_ticker,
            direction,
            int(layer),
            span,
        ),
        "prepared_record_id": str(prepared_record["record_id"]),
        "pair_id": str(prepared_record["pair_id"]),
        "control_type": str(prepared_record["control_type"]),
        "patching_direction": direction,
        "source_identity": dict(source_identity),
        "target_identity": dict(target_identity),
        "evidence_valence": str(prepared_record["evidence_valence"]),
        "evidence_origin_sector": str(prepared_record["evidence_origin_sector"]),
        "span_condition": span,
        "layer": int(layer),
        "layers": [int(layer)],
        "source_clean_margin": source_margin,
        "target_clean_margin": target_margin,
        "patched_margin": float(result["patched_margin"]),
        "delta_margin": delta,
        "normalized_transfer": _normalized_transfer(source_margin, target_margin, delta),
        "flip": bool(result["flip"]),
        "positions_patched": list(result["positions_patched"]),
        "position_count": int(result["position_count"]),
        "source_seq_len": int(result["source_seq_len"]),
        "target_seq_len": int(result["target_seq_len"]),
        "source_prompt_sha256": sha256_json(source_prompt),
        "target_prompt_sha256": sha256_json(target_prompt),
        "source_trial_record_id": prepared_record.get("source_trial_record_id"),
        "source_trial_key": prepared_record.get("source_trial_key"),
        "source_trial_index": prepared_record.get("source_trial_index"),
        "source_set_index": prepared_record.get("source_set_index"),
        "trial_row_sha256": prepared_record.get("trial_row_sha256"),
        "evidence_item_hashes": dict(prepared_record.get("evidence_item_hashes", {})),
        "input_sha256": prepared_record.get("input_sha256"),
        "split_manifest_sha256": prepared_record.get("split_manifest_sha256"),
        "prompt_template": PROMPT_TEMPLATE_VERSION,
    }


def _pair_means(
    grouped: Mapping[tuple[int, str, str, str, str], Mapping[str, Sequence[Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (layer, span, direction, origin, control), by_pair in sorted(grouped.items()):
        pair_delta = [
            sum(float(record["delta_margin"]) for record in records) / len(records)
            for records in by_pair.values()
        ]
        pair_flip = [
            sum(bool(record["flip"]) for record in records) / len(records)
            for records in by_pair.values()
        ]
        transfers = [
            float(record["normalized_transfer"])
            for records in by_pair.values()
            for record in records
            if record.get("normalized_transfer") is not None
        ]
        rows.append(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "cross_sector_context_overriding_summary",
                "layer": layer,
                "span_condition": span,
                "patching_direction": direction,
                "evidence_origin_sector": origin,
                "control_type": control,
                "pair_count": len(by_pair),
                "record_count": sum(len(records) for records in by_pair.values()),
                "equal_pair_mean_delta_margin": sum(pair_delta) / len(pair_delta),
                "equal_pair_flip_rate": sum(pair_flip) / len(pair_flip),
                "equal_pair_mean_normalized_transfer": (
                    sum(transfers) / len(transfers) if transfers else None
                ),
            }
        )
    return rows


def analyze_context_overriding_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate equal-pair means and paired span contrasts.

    The primary span is compared with header and final_position within the same
    pair, layer, direction, evidence-origin stratum, and control arm. Missing
    pair/span combinations do not produce a contrast row.
    """
    grouped: dict[tuple[int, str, str, str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = (
            int(record["layer"]),
            str(record["span_condition"]),
            str(record["patching_direction"]),
            str(record["evidence_origin_sector"]),
            str(record["control_type"]),
        )
        grouped[key][str(record["pair_id"])].append(record)

    contrasts: list[dict[str, Any]] = []
    contrast_groups: dict[tuple[int, str, str, str, str], dict[str, dict[str, list[Mapping[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        key = (
            int(record["layer"]),
            str(record["patching_direction"]),
            str(record["evidence_origin_sector"]),
            str(record["control_type"]),
        )
        contrast_groups[key][str(record["span_condition"])][str(record["pair_id"])].append(record)

    for (layer, direction, origin, control), by_span in sorted(contrast_groups.items()):
        primary = by_span.get("instruction_context", {})
        for control_span, label in (
            ("header", "primary_minus_header"),
            ("final_position", "primary_minus_final"),
        ):
            comparison = by_span.get(control_span, {})
            common_pairs = sorted(set(primary) & set(comparison))
            if not common_pairs:
                continue
            values = []
            for pair_id in common_pairs:
                primary_mean = sum(float(row["delta_margin"]) for row in primary[pair_id]) / len(primary[pair_id])
                comparison_mean = sum(float(row["delta_margin"]) for row in comparison[pair_id]) / len(comparison[pair_id])
                values.append(primary_mean - comparison_mean)
            contrasts.append(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": "cross_sector_context_overriding_contrast",
                    "contrast": label,
                    "primary_span": "instruction_context",
                    "control_span": control_span,
                    "layer": layer,
                    "patching_direction": direction,
                    "evidence_origin_sector": origin,
                    "control_type": control,
                    "pair_count": len(values),
                    "equal_pair_mean_contrast": sum(values) / len(values),
                    "pair_values": {
                        pair_id: value for pair_id, value in zip(common_pairs, values, strict=True)
                    },
                }
            )
    return {"equal_pair_means": _pair_means(grouped), "paired_contrasts": contrasts}


def _confirmation_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _confirmation_bootstrap_ci(
    values: Sequence[float], *, seed: int, samples: int, confidence: float
) -> list[float]:
    if not values or samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid confirmation bootstrap inputs")
    rng = random.Random(seed)
    count = len(values)
    means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return [
        _confirmation_quantile(means, alpha),
        _confirmation_quantile(means, 1.0 - alpha),
    ]


def _confirmation_exact_sign_flip_p(values: Sequence[float]) -> float:
    """Return the exact one-sided sign-flip p-value using meet-in-the-middle."""
    if not values:
        raise ValueError("cannot sign-flip an empty sequence")
    signed_values = [float(value) for value in values]
    observed = sum(signed_values)
    midpoint = len(signed_values) // 2

    def signed_sums(items: Sequence[float]) -> list[float]:
        result = [0.0]
        for item in items:
            result = [value + sign * item for value in result for sign in (-1.0, 1.0)]
        return result

    left = signed_sums(signed_values[:midpoint])
    right = sorted(signed_sums(signed_values[midpoint:]))
    count = sum(
        len(right) - bisect.bisect_left(right, observed - value - 1e-15)
        for value in left
    )
    return count / (2 ** len(values))


def _confirmation_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_confirmation_config(config: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    if config.get("artifact_type") != "cross_sector_context_overriding_confirmation_config":
        raise ValueError("confirmation config artifact_type does not match B V1")
    if config.get("schema_version") != 1 or config.get("protocol") != "sector-context-followup-b-v1":
        raise ValueError("confirmation config schema or protocol mismatch")
    layers_value = config.get("layer_range")
    if layers_value != [14, 21]:
        raise ValueError("confirmation config layer_range must be [14, 21]")
    layers = tuple(range(int(layers_value[0]), int(layers_value[1]) + 1))
    if config.get("primary_layer") != 16:
        raise ValueError("confirmation config primary_layer must be 16")
    spans = tuple(config.get("span_conditions", ()))
    if spans != ("instruction_context", "header", "final_position"):
        raise ValueError("confirmation config span_conditions mismatch")
    if config.get("primary_span") != "instruction_context" or config.get("control_span") != "header":
        raise ValueError("confirmation config primary/control span mismatch")
    directions = tuple(config.get("directions", ()))
    if directions != ("Technology_to_Financial Services", "Financial Services_to_Technology"):
        raise ValueError("confirmation config directions mismatch")
    if config.get("aggregation") != "within_pair_average_over_directions_and_evidence_origin_strata->equal_pair_mean":
        raise ValueError("confirmation config aggregation mismatch")
    if config.get("primary_metric") != "toward_source_delta_margin=sign(source_clean_margin-target_clean_margin)*delta_margin":
        raise ValueError("confirmation config primary_metric mismatch")
    if config.get("primary_contrast") != "C_B=L16_context_toward_source_minus_L16_header_toward_source":
        raise ValueError("confirmation config primary_contrast mismatch")
    if config.get("bootstrap_samples") != 10000 or config.get("bootstrap_seed") != 20260901:
        raise ValueError("confirmation config bootstrap mismatch")
    if config.get("ci_level") != 0.95:
        raise ValueError("confirmation config ci_level mismatch")
    if config.get("data_quality_gates") != {
        "minimum_eligible_pairs": 8,
        "self_source_exact_noop": True,
    }:
        raise ValueError("confirmation config data-quality gates mismatch")
    if config.get("effect_gates") != {
        "minimum_L16_context_toward_source_mean": 0.10,
        "minimum_C_B_mean": 0.10,
        "L16_context_toward_source_bootstrap_ci_lower_gt_0": True,
        "C_B_bootstrap_ci_lower_gt_0": True,
    }:
        raise ValueError("confirmation config effect gates mismatch")
    if config.get("specificity_gates") != {
        "L16_cross_sector_context_abs_gt_same_sector_peer_context_abs": True,
        "evidence_origin_strata_both_positive": True,
    }:
        raise ValueError("confirmation config specificity gates mismatch")
    statistical = config.get("statistical_gates")
    if not isinstance(statistical, Mapping) or statistical.get("sign_flip_test") != "one_sided_exact_on_pair_toward_source":
        raise ValueError("confirmation config statistical gates mismatch")
    if statistical.get("sign_flip_alpha") != 0.05 or statistical.get("multiple_comparison") != "holm_two_contrasts_alpha_0.05":
        raise ValueError("confirmation config multiple-comparison settings mismatch")
    if statistical.get("contrasts_under_correction") != ["L16_context_toward_source", "C_B"]:
        raise ValueError("confirmation config contrast list mismatch")
    return layers, spans, directions


def _confirmation_record_key(record: Mapping[str, Any]) -> tuple[str, str, int, str, str, str]:
    return (
        str(record.get("pair_id", "")),
        str(record.get("control_type", "")),
        int(record.get("layer")),
        str(record.get("span_condition", "")),
        str(record.get("patching_direction", "")),
        str(record.get("evidence_origin_sector", "")),
    )


def evaluate_context_overriding_confirmation(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Evaluate the frozen B V1 calibration/test gates from compact records."""
    if split not in {"calibration", "test"}:
        raise ValueError("split must be calibration or test")
    layers, spans, directions = _validate_confirmation_config(config)
    if not records:
        raise ValueError("confirmation records are empty")

    seen: set[tuple[str, str, int, str, str, str]] = set()
    by_key: dict[tuple[str, str, int, str, str, str], Mapping[str, Any]] = {}
    cross_records: list[Mapping[str, Any]] = []
    peer_records: list[Mapping[str, Any]] = []
    self_source_exact_noop = True
    for record in records:
        if record.get("artifact_type") != RECORD_ARTIFACT_TYPE:
            raise ValueError("confirmation records contain an unexpected artifact_type")
        pair_id = str(record.get("pair_id", ""))
        if not pair_id:
            raise ValueError("confirmation record is missing pair_id")
        control = str(record.get("control_type", ""))
        if control == "name_form":
            continue
        key = _confirmation_record_key(record)
        if control != "same_sector_peer":
            if key in seen:
                raise ValueError(f"duplicate confirmation record: {key}")
            seen.add(key)
            by_key[key] = record
        layer = int(record.get("layer"))
        span = str(record.get("span_condition", ""))
        direction = str(record.get("patching_direction", ""))
        origin = str(record.get("evidence_origin_sector", ""))
        if layer not in layers or span not in spans:
            raise ValueError("confirmation records contain a layer/span outside the frozen config")
        if origin not in {"Technology", "Financial Services"}:
            raise ValueError("confirmation record has an unknown evidence-origin sector")
        if record.get("evidence_valence") != "negative":
            raise ValueError("confirmation records must contain negative evidence only")
        source = record.get("source_identity")
        target = record.get("target_identity")
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise ValueError("confirmation record is missing source/target identity")
        source_sector = str(source.get("sector", ""))
        target_sector = str(target.get("sector", ""))
        if control == "cross_sector":
            if direction not in directions or {source_sector, target_sector} != {"Technology", "Financial Services"}:
                raise ValueError("cross-sector confirmation record direction or sectors mismatch")
            cross_records.append(record)
        elif control == "self_source":
            if source_sector != target_sector or direction != f"{source_sector}_to_{target_sector}":
                raise ValueError("self-source confirmation record direction or sectors mismatch")
        elif control == "same_sector_peer":
            if source_sector != target_sector or direction != f"{source_sector}_to_{target_sector}":
                raise ValueError("same-sector peer confirmation record direction or sectors mismatch")
            peer_records.append(record)
        elif control == "name_form":
            pass
        else:
            raise ValueError(f"unsupported confirmation control_type: {control!r}")
        for field in ("source_clean_margin", "target_clean_margin", "delta_margin"):
            _confirmation_float(record.get(field), field=field)
        if control == "self_source" and record.get("delta_margin") != 0.0:
            self_source_exact_noop = False

    if not cross_records:
        raise ValueError("confirmation records contain no cross-sector observations")
    all_pair_ids = {str(record["pair_id"]) for record in cross_records}
    origins = ("Technology", "Financial Services")
    expected_cross = {
        (pair_id, "cross_sector", layer, span, direction, origin)
        for pair_id in all_pair_ids for layer in layers for span in spans
        for direction in directions for origin in origins
    }
    missing_cross = expected_cross - set(by_key)
    if missing_cross:
        raise ValueError(f"confirmation records are incomplete: {len(missing_cross)} cross-sector records missing")
    expected_self = {
        (pair_id, "self_source", layer, span, direction, origin)
        for pair_id in all_pair_ids for layer in layers for span in spans
        for direction in ("Technology_to_Technology", "Financial Services_to_Financial Services")
        for origin in origins
    }
    missing_self = expected_self - set(by_key)
    if missing_self:
        raise ValueError(f"confirmation records are incomplete: {len(missing_self)} self-source records missing")

    def toward(record: Mapping[str, Any]) -> float:
        gap = _confirmation_float(record["source_clean_margin"], field="source_clean_margin") - _confirmation_float(record["target_clean_margin"], field="target_clean_margin")
        delta = _confirmation_float(record["delta_margin"], field="delta_margin")
        return (1.0 if gap > 0 else -1.0 if gap < 0 else 0.0) * delta

    def clean_eligible(pair_id: str) -> bool:
        pair_records = [record for record in cross_records if str(record["pair_id"]) == pair_id]
        return all(
            _confirmation_float(record["source_clean_margin"], field="source_clean_margin") < 0.0
            and _confirmation_float(record["target_clean_margin"], field="target_clean_margin") < 0.0
            for record in pair_records
        )

    eligible_pair_ids = sorted(pair_id for pair_id in all_pair_ids if clean_eligible(pair_id))
    if not eligible_pair_ids:
        raise ValueError("no identity pairs pass the clean-outcome gate")

    def pair_observations(control: str, layer: int, span: str, pair_id: str) -> dict[tuple[str, str], Mapping[str, Any]]:
        return {
            (str(record["patching_direction"]), str(record["evidence_origin_sector"])): record
            for record in records
            if str(record["pair_id"]) == pair_id and str(record["control_type"]) == control
            and int(record["layer"]) == layer and str(record["span_condition"]) == span
        }

    pair_values: list[dict[str, Any]] = []
    context_values: list[float] = []
    contrast_values: list[float] = []
    peer_abs_values: list[float] = []
    stratum_values: dict[str, list[float]] = defaultdict(list)
    for pair_id in eligible_pair_ids:
        context = pair_observations("cross_sector", 16, "instruction_context", pair_id)
        header = pair_observations("cross_sector", 16, "header", pair_id)
        if set(context) != set(header):
            raise ValueError(f"confirmation records are incomplete for matched context/header pair {pair_id}")
        context_pair = sum(toward(record) for record in context.values()) / len(context)
        contrast_pair = sum(toward(context[key]) - toward(header[key]) for key in context) / len(context)
        peer = [
            record for record in peer_records
            if str(record["pair_id"]) == pair_id
            and int(record["layer"]) == 16
            and str(record["span_condition"]) == "instruction_context"
        ]
        if not peer:
            raise ValueError(f"confirmation records have no same-sector peer context for pair {pair_id}")
        peer_pair = sum(abs(_confirmation_float(record["delta_margin"], field="delta_margin")) for record in peer) / len(peer)
        context_abs_pair = sum(abs(_confirmation_float(record["delta_margin"], field="delta_margin")) for record in context.values()) / len(context)
        context_values.append(context_pair)
        contrast_values.append(contrast_pair)
        peer_abs_values.append(peer_pair)
        for key, record in context.items():
            stratum_values[key[1]].append(toward(record))
        header_pair = sum(toward(record) for record in header.values()) / len(header)
        pair_values.append({
            "pair_id": pair_id,
            "context_toward_source": context_pair,
            "context_abs_delta": context_abs_pair,
            "header_toward_source": header_pair,
            "C_B": contrast_pair,
            "same_sector_peer_context_abs_delta": peer_pair,
            "context_by_direction_and_origin": {
                f"{direction}|{origin}": toward(record)
                for (direction, origin), record in sorted(context.items())
            },
            "header_by_direction_and_origin": {
                f"{direction}|{origin}": toward(record)
                for (direction, origin), record in sorted(header.items())
            },
        })

    from llm_bias.core.analysis.statistics import holm_bonferroni

    seed = int(config["bootstrap_seed"])
    samples = int(config["bootstrap_samples"])
    confidence = float(config["ci_level"])
    raw_p_values = [_confirmation_exact_sign_flip_p(context_values), _confirmation_exact_sign_flip_p(contrast_values)]
    adjusted_p_values = holm_bonferroni(raw_p_values)
    estimates = {
        "L16_context_toward_source": {
            "pair_count": len(context_values), "mean": sum(context_values) / len(context_values),
            "bootstrap_95_ci": _confirmation_bootstrap_ci(context_values, seed=seed, samples=samples, confidence=confidence),
            "one_sided_exact_sign_flip_p": raw_p_values[0], "holm_adjusted_p": adjusted_p_values[0],
        },
        "C_B": {
            "pair_count": len(contrast_values), "mean": sum(contrast_values) / len(contrast_values),
            "bootstrap_95_ci": _confirmation_bootstrap_ci(contrast_values, seed=seed, samples=samples, confidence=confidence),
            "one_sided_exact_sign_flip_p": raw_p_values[1], "holm_adjusted_p": adjusted_p_values[1],
        },
        "L16_cross_sector_context_abs_delta": {"pair_count": len(context_values), "mean": sum(row["context_abs_delta"] for row in pair_values) / len(pair_values)},
        "L16_same_sector_peer_context_abs_delta": {"pair_count": len(peer_abs_values), "mean": sum(peer_abs_values) / len(peer_abs_values)},
        "evidence_origin_strata": {
            origin: {"pair_observation_count": len(values), "mean_toward_source": sum(values) / len(values)}
            for origin, values in sorted(stratum_values.items())
        },
    }
    data_gates = config["data_quality_gates"]
    effect_gates = config["effect_gates"]
    specificity_gates = config["specificity_gates"]
    statistical_gates = config["statistical_gates"]
    gates = [
        {"gate": "minimum_eligible_pairs", "observed": len(eligible_pair_ids), "threshold": int(data_gates["minimum_eligible_pairs"]), "pass": len(eligible_pair_ids) >= int(data_gates["minimum_eligible_pairs"])},
        {"gate": "self_source_exact_noop", "observed": self_source_exact_noop, "pass": self_source_exact_noop},
        {"gate": "L16_context_toward_source_mean", "observed": estimates["L16_context_toward_source"]["mean"], "threshold": float(effect_gates["minimum_L16_context_toward_source_mean"]), "pass": estimates["L16_context_toward_source"]["mean"] > float(effect_gates["minimum_L16_context_toward_source_mean"])},
        {"gate": "C_B_mean", "observed": estimates["C_B"]["mean"], "threshold": float(effect_gates["minimum_C_B_mean"]), "pass": estimates["C_B"]["mean"] > float(effect_gates["minimum_C_B_mean"])},
        {"gate": "L16_context_toward_source_bootstrap_ci_lower_gt_0", "observed": estimates["L16_context_toward_source"]["bootstrap_95_ci"][0], "pass": estimates["L16_context_toward_source"]["bootstrap_95_ci"][0] > 0.0},
        {"gate": "C_B_bootstrap_ci_lower_gt_0", "observed": estimates["C_B"]["bootstrap_95_ci"][0], "pass": estimates["C_B"]["bootstrap_95_ci"][0] > 0.0},
        {"gate": "cross_sector_context_abs_gt_same_sector_peer_context_abs", "observed": [estimates["L16_cross_sector_context_abs_delta"]["mean"], estimates["L16_same_sector_peer_context_abs_delta"]["mean"]], "pass": estimates["L16_cross_sector_context_abs_delta"]["mean"] > estimates["L16_same_sector_peer_context_abs_delta"]["mean"]},
        {"gate": "evidence_origin_strata_both_positive", "observed": estimates["evidence_origin_strata"], "pass": all(value["mean_toward_source"] > 0.0 for value in estimates["evidence_origin_strata"].values()) and set(estimates["evidence_origin_strata"]) == set(origins)},
        {"gate": "L16_context_toward_source_sign_flip", "observed": estimates["L16_context_toward_source"]["one_sided_exact_sign_flip_p"], "threshold": float(statistical_gates["sign_flip_alpha"]), "pass": estimates["L16_context_toward_source"]["one_sided_exact_sign_flip_p"] < float(statistical_gates["sign_flip_alpha"])},
        {"gate": "C_B_sign_flip", "observed": estimates["C_B"]["one_sided_exact_sign_flip_p"], "threshold": float(statistical_gates["sign_flip_alpha"]), "pass": estimates["C_B"]["one_sided_exact_sign_flip_p"] < float(statistical_gates["sign_flip_alpha"])},
        {"gate": "holm_two_contrasts", "observed": {"L16_context_toward_source": adjusted_p_values[0], "C_B": adjusted_p_values[1]}, "threshold": float(statistical_gates["sign_flip_alpha"]), "pass": all(value < float(statistical_gates["sign_flip_alpha"]) for value in adjusted_p_values)},
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": CONFIRMATION_ARTIFACT_TYPE,
        "protocol": "sector-context-followup-b-v1",
        "split": split,
        "eligible_pair_count": len(eligible_pair_ids),
        "pair_values": pair_values,
        "estimates": estimates,
        "gate_checks": gates,
        "success": all(bool(gate["pass"]) for gate in gates),
        "interpretation_limit": "B V1 fixed-negative-evidence context resample-patching sufficiency; not necessity, a unique decision route, or an attention claim",
    }
    if split == "calibration":
        result["test_authorized"] = result["success"]
    else:
        result["formal"] = True
    return result


def evaluate_context_overriding_confirmation_artifacts(
    records_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate compact B V1 records and write a provenance-bound JSON artifact."""
    records_path = Path(records_path)
    config_path = Path(config_path)
    output_path = Path(output_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("confirmation config must be a JSON object")
    result = evaluate_context_overriding_confirmation(read_jsonl(records_path), config, split=split)
    result.update({
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "parent_records": str(records_path), "parent_sha256": sha256_file(records_path),
    })
    write_json(output_path, result, overwrite=False)
    return result


def _formatted_records(
    tokenizer: Any,
    prepared: Sequence[Mapping[str, Any]],
    decision_prefix: str,
    positive_candidate: str,
    negative_candidate: str,
    max_seq_len: int,
) -> list[tuple[Mapping[str, Any], str, str, dict[str, list[int]], dict[str, list[int]]]]:
    formatted = []
    for row in prepared:
        source_prompt, source_spans = _prepare_scoring_prompt(
            tokenizer, row["source_prompt"], row["source_evidence_char_spans"], decision_prefix
        )
        target_prompt, target_spans = _prepare_scoring_prompt(
            tokenizer, row["target_prompt"], row["target_evidence_char_spans"], decision_prefix
        )
        for prompt in (source_prompt, target_prompt):
            if len(input_ids(tokenizer, prompt + positive_candidate)) > max_seq_len:
                raise ValueError(f"prepared prompt exceeds max_seq_len={max_seq_len}")
            if len(input_ids(tokenizer, prompt + negative_candidate)) > max_seq_len:
                raise ValueError(f"prepared prompt exceeds max_seq_len={max_seq_len}")
            continuation_token_ids(tokenizer, prompt, positive_candidate)
            continuation_token_ids(tokenizer, prompt, negative_candidate)
        formatted.append((row, source_prompt, target_prompt, source_spans, target_spans))
    return formatted


def run_cross_sector_context_overriding_pipeline(
    *,
    prepared_pairs: str | Path,
    model_name: str,
    run_id: str,
    layers: Sequence[int] = DEFAULT_LAYERS,
    spans: Sequence[str] = DEFAULT_SPANS,
    dataset: str = DEFAULT_DATASET,
    artifact_root: str | Path = "artifacts",
    decision_prefix: str = DEFAULT_DECISION_PREFIX,
    positive_candidate: str = DEFAULT_POSITIVE_CANDIDATE,
    negative_candidate: str = DEFAULT_NEGATIVE_CANDIDATE,
    max_records: int | None = None,
    max_seq_len: int = 1024,
) -> Path:
    """Run B V1 through prepare, forward, analyze, and finalize."""
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    resolved_layers = sorted({int(layer) for layer in layers})
    if not resolved_layers:
        raise ValueError("at least one layer is required")
    resolved_spans = _validate_spans(spans)
    prepared_path = Path(prepared_pairs)
    prepared = load_prepared_context_records(prepared_path)
    if max_records is not None:
        prepared = prepared[:max_records]
    if not prepared:
        raise ValueError("prepared input contains no negative-evidence records")

    preflight_tokenizer = load_tokenizer(model_name)
    formatted = _formatted_records(
        preflight_tokenizer,
        prepared,
        decision_prefix,
        positive_candidate,
        negative_candidate,
        max_seq_len,
    )
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(
        prepared_path,
        artifact_type=PAIRS_ARTIFACT_TYPE,
        stage="prepare",
        role="input",
    )
    run.manifest.save()
    try:
        prepare_metadata_path = run.run_directory / "prepare" / "metadata.json"
        with run.stage("prepare") as stage:
            write_metadata(
                prepare_metadata_path,
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": PREPARE_ARTIFACT_TYPE,
                    "model": model_name,
                    "prepared_pairs": str(prepared_path),
                    "prepared_pairs_sha256": sha256_file(prepared_path),
                    "negative_record_count": len(prepared),
                    "layers": resolved_layers,
                    "spans": list(resolved_spans),
                    "prompt_template": PROMPT_TEMPLATE_VERSION,
                    "decision_prefix": decision_prefix,
                    "positive_candidate": positive_candidate,
                    "negative_candidate": negative_candidate,
                    "state_cache_persisted": False,
                    "formal_success_gate": False,
                },
                overwrite=False,
            )
            stage.count(len(prepared))
        run.manifest.register_artifact(
            prepare_metadata_path,
            artifact_type=PREPARE_ARTIFACT_TYPE,
            stage="prepare",
            role="output",
        )
        run.manifest.save()

        model, tokenizer, fallback_device = load_model(model_name)
        device = getattr(model, "input_device", fallback_device)
        n_layers = int(model.n_layers)
        if resolved_layers[0] < 0 or resolved_layers[-1] >= n_layers:
            raise ValueError(f"layers must be in model range 0..{n_layers - 1}")

        records: list[dict[str, Any]] = []
        clean: dict[str, CandidateMargin] = {}
        forward_path = run.run_directory / "forward" / "context_overriding_records.jsonl"
        with run.stage("forward") as stage:
            for row, source_prompt, target_prompt, source_spans, target_spans in formatted:
                source_key = sha256_json(source_prompt)
                target_key = sha256_json(target_prompt)
                for key, prompt in ((source_key, source_prompt), (target_key, target_prompt)):
                    if key not in clean:
                        clean[key] = score_single_token_margin_fp32(
                            model, tokenizer, prompt, positive_candidate, negative_candidate, device=device
                        )
                row_cache = {
                    source_key: cache_source_residuals(
                        model, tokenizer, source_prompt, resolved_layers, device
                    )
                }
                if target_key != source_key:
                    row_cache[target_key] = cache_source_residuals(
                        model, tokenizer, target_prompt, resolved_layers, device
                    )
                directions = [
                    (
                        row["source_identity"], row["target_identity"],
                        source_prompt, target_prompt, source_spans, target_spans,
                        source_key, clean[source_key], clean[target_key],
                    )
                ]
                if row.get("control_type") != "self_source":
                    directions.append(
                        (
                            row["target_identity"], row["source_identity"],
                            target_prompt, source_prompt, target_spans, source_spans,
                            target_key, clean[target_key], clean[source_key],
                        )
                    )
                for (
                    source_identity, target_identity, src_prompt, tgt_prompt,
                    src_spans, tgt_spans, source_cache_key, source_margin, target_margin,
                ) in directions:
                    direction = _direction_name(source_identity, target_identity)
                    for layer in resolved_layers:
                        for span in resolved_spans:
                            records.append(
                                run_context_overriding_record(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prepared_record=row,
                                    source_identity=source_identity,
                                    target_identity=target_identity,
                                    source_prompt=src_prompt,
                                    target_prompt=tgt_prompt,
                                    source_spans=src_spans,
                                    target_spans=tgt_spans,
                                    layer=layer,
                                    span=span,
                                    source_residuals=row_cache[source_cache_key],
                                    source_clean_margin=source_margin,
                                    target_clean_margin=target_margin,
                                    positive_candidate=positive_candidate,
                                    negative_candidate=negative_candidate,
                                    device=device,
                                    direction=direction,
                                )
                            )
                del row_cache
            record_count = write_jsonl(forward_path, records, overwrite=False)
            stage.count(record_count)
        run.manifest.register_artifact(
            forward_path,
            artifact_type=RECORD_ARTIFACT_TYPE,
            stage="forward",
            role="output",
            record_count=record_count,
        )
        run.manifest.save()

        summary_path = run.run_directory / "analyze" / "summary.json"
        with run.stage("analyze") as stage:
            analysis = analyze_context_overriding_records(records)
            summary = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": ANALYSIS_ARTIFACT_TYPE,
                "model": model_name,
                "prepared_pairs_sha256": sha256_file(prepared_path),
                "negative_record_count": len(prepared),
                "patch_record_count": len(records),
                **analysis,
                "formal_success_gate": False,
                "interpretation_limit": (
                    "negative-evidence context resample-patching measures state sufficiency "
                    "for the fixed Buy/Sell margin; it does not establish necessity or a unique route"
                ),
            }
            write_json(summary_path, summary, overwrite=False)
            stage.count(len(summary["equal_pair_means"]) + len(summary["paired_contrasts"]))
        run.manifest.register_artifact(
            summary_path,
            artifact_type=ANALYSIS_ARTIFACT_TYPE,
            stage="analyze",
            role="output",
        )
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = [
    "ANALYSIS_ARTIFACT_TYPE",
    "CONFIRMATION_ARTIFACT_TYPE",
    "ARTIFACT_SCHEMA_VERSION",
    "evaluate_context_overriding_confirmation",
    "evaluate_context_overriding_confirmation_artifacts",
    "DEFAULT_DATASET",
    "DEFAULT_LAYERS",
    "DEFAULT_SPANS",
    "RECORD_ARTIFACT_TYPE",
    "analyze_context_overriding_records",
    "load_prepared_context_records",
    "run_context_overriding_record",
    "run_cross_sector_context_overriding_pipeline",
]
