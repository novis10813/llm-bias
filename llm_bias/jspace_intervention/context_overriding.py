"""B V1 cross-sector context overriding under negative evidence.

This workflow consumes the prepared cross-sector records produced by
``cross_sector_patching``. It filters those records to negative evidence and
uses the shared activation-patching primitives to patch one span at one layer.
Residual caches stay in memory and only compact margin/provenance records reach
artifacts.
"""
from __future__ import annotations

import math
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
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_DATASET",
    "DEFAULT_LAYERS",
    "DEFAULT_SPANS",
    "RECORD_ARTIFACT_TYPE",
    "analyze_context_overriding_records",
    "load_prepared_context_records",
    "run_context_overriding_record",
    "run_cross_sector_context_overriding_pipeline",
]
