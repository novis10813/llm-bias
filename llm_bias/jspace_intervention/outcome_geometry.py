"""V2 outcome direction geometric projection workflow (auxiliary diagnostic).

Implements the frozen protocol in
``docs/jspace-token-experiments/details/report-v2-geometry.md``: per-layer sector state
differences between the V2 source sector and a contrast sector on the
discovery split, dot-projected onto the frozen V2 outcome directions, with a
parallel/perpendicular decomposition for every position set.  Optionally the
same decomposition is reported for the frozen contrastive TF-IDF sector
prototype directions.

This is a descriptive, non-causal geometry diagnostic bound to the V2
direction identity: no interventions, no doses, no gates.  Raw activations
are never persisted -- per-prompt residuals are streamed into in-memory
sector means (float32 on CPU) and artifacts carry only norms, SHA-256 hashes
of derived vectors, projection metrics, and provenance.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.analysis.statistics import direction_hash
from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.jspace_intervention.outcome_flip import (
    OUTCOME_FLIP_ARTIFACT_TYPE,
    _enable_deterministic_gpu,
    evidence_item_end_positions,
    verify_direction_identity,
)
from llm_bias.jspace_intervention.pipeline import _iter_prompt_records
from llm_bias.jspace_intervention.prompting import prepare_scoring_prompt
from llm_bias.jspace_intervention.runner import layer_prototypes
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig, PrototypeSpec

GEOMETRY_DATASET = "jspace-outcome-direction-geometry"
TFIDF_SCORE_TYPE = "contrastive_tfidf"
TFIDF_CONFIG_ARTIFACT_TYPE = "jspace_intervention_config"
_TFIDF_INTERPRETATION = (
    "geometry/correlation between frozen contrastive TF-IDF sector prototype "
    "directions and the V2 outcome directions; descriptive only, not causal"
)

_INTERPRETATION_LIMITS = [
    "descriptive representation geometry only; not causal evidence",
    "sector state difference uses the discovery split, the same population "
    "as the V2 direction fitting; any alignment is in-sample",
    "no interventions, doses, or success gates are part of this workflow",
]


# ---------------------------------------------------------------------------
# Dot projection math (general formula; does not assume a unit direction)
# ---------------------------------------------------------------------------


def project_onto_direction(vector, direction) -> dict[str, float | None]:
    """Orthogonal dot projection of ``vector`` onto ``direction``.

    Uses the general projection coefficient
    ``c = <vector, direction> / ||direction||**2`` (valid for non-unit
    directions), then decomposes
    ``vector = c * direction + (vector - c * direction)``.  All metrics are
    computed in float64.  Returns compact scalar metrics only.
    """
    values = torch.as_tensor(vector).detach().reshape(-1).double()
    axis = torch.as_tensor(direction).detach().reshape(-1).double()
    if values.numel() == 0 or values.numel() != axis.numel():
        raise ValueError("vector and direction widths differ")
    if not torch.isfinite(values).all() or not torch.isfinite(axis).all():
        raise ValueError("non-finite projection inputs")
    axis_norm = float(axis.norm())
    values_norm = float(values.norm())
    if axis_norm <= 0.0 or not math.isfinite(axis_norm):
        raise ValueError("direction norm must be positive and finite")
    if not math.isfinite(values_norm):
        raise ValueError("vector norm is not finite")
    dot = float(values @ axis)
    coefficient = dot / (axis_norm * axis_norm)
    parallel_norm = abs(coefficient) * axis_norm
    perpendicular = values - coefficient * axis
    perpendicular_norm = float(perpendicular.norm())
    if values_norm > 0.0:
        cosine = max(-1.0, min(1.0, dot / (values_norm * axis_norm)))
        angle_deg = float(math.degrees(math.acos(cosine)))
        parallel_fraction = (parallel_norm * parallel_norm) / (
            values_norm * values_norm
        )
        pythagoras_error = abs(
            values_norm * values_norm
            - parallel_norm * parallel_norm
            - perpendicular_norm * perpendicular_norm
        ) / (values_norm * values_norm)
    else:
        cosine = None
        angle_deg = None
        parallel_fraction = None
        pythagoras_error = None
    return {
        "vector_norm": values_norm,
        "direction_norm": axis_norm,
        "dot": dot,
        "coefficient": coefficient,
        "parallel_norm": parallel_norm,
        "perpendicular_norm": perpendicular_norm,
        "cosine": cosine,
        "angle_deg": angle_deg,
        "parallel_fraction": parallel_fraction,
        "pythagoras_error": pythagoras_error,
    }


def _cosine(vector, direction) -> float | None:
    values = torch.as_tensor(vector).detach().reshape(-1).double()
    axis = torch.as_tensor(direction).detach().reshape(-1).double()
    if values.numel() == 0 or values.numel() != axis.numel():
        raise ValueError("vector and direction widths differ")
    values_norm = float(values.norm())
    axis_norm = float(axis.norm())
    if values_norm <= 0.0 or axis_norm <= 0.0:
        return None
    return max(-1.0, min(1.0, float(values @ axis) / (values_norm * axis_norm)))


# ---------------------------------------------------------------------------
# Prompt records for the two sectors (discovery split, frozen in the identity)
# ---------------------------------------------------------------------------


def _prompt_records(
    input_path: Path,
    *,
    assignments: Mapping[str, str],
    sector: str,
    prompt_columns: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _iter_prompt_records(
        input_path,
        assignments=assignments,
        split_name="discovery",
        source_sector=sector,
        prompt_columns=prompt_columns,
    ):
        record["record_id"] = stable_record_id(
            record["ticker"], record["prompt_column"], "discovery"
        )
        records.append(record)
    return records


def _preflight_records(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    config: OutcomeFlipConfig,
    max_seq_len: int,
    sector: str,
) -> None:
    if not records:
        raise ValueError(f"no discovery prompt records for sector {sector!r}")
    for record in records:
        scoring_prompt, _ = prepare_scoring_prompt(
            tokenizer, record["prompt"], decision_prefix=config.decision_prefix
        )
        length = len(input_ids(tokenizer, scoring_prompt, add_special_tokens=True))
        if length > max_seq_len:
            raise ValueError(
                f"formatted scoring prompt for {record['ticker']!r} has "
                f"{length} tokens, limit {max_seq_len}"
            )


def _materialize_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    config: OutcomeFlipConfig,
) -> list[dict[str, Any]]:
    """Attach scoring prompts, token ids, and geometry position sets.

    ``positions`` carries exactly the frozen V2 position rules (so the
    records stay compatible with ``verify_direction_identity``);
    ``position_sets`` adds the decision final position on top.
    """
    materialized: list[dict[str, Any]] = []
    for record in records:
        scoring_prompt, evidence_span = prepare_scoring_prompt(
            tokenizer, record["prompt"], decision_prefix=config.decision_prefix
        )
        prompt_ids = input_ids(tokenizer, scoring_prompt, add_special_tokens=True)
        positions: dict[str, list[int]] = {}
        if "evidence_span_all" in config.position_rules:
            positions["evidence_span_all"] = list(
                range(int(evidence_span[0]), int(evidence_span[1]))
            )
        if "evidence_item_end" in config.position_rules:
            positions["evidence_item_end"] = evidence_item_end_positions(
                tokenizer, scoring_prompt, record["prompt"]
            )
        position_sets = dict(positions)
        position_sets["final_position"] = [len(prompt_ids) - 1]
        materialized.append(
            {
                **record,
                "scoring_prompt": scoring_prompt,
                "prompt_ids": prompt_ids,
                "evidence_span": [int(evidence_span[0]), int(evidence_span[1])],
                "positions": positions,
                "position_sets": position_sets,
            }
        )
    return materialized


# ---------------------------------------------------------------------------
# Sector state accumulation (streaming; raw activations never persisted)
# ---------------------------------------------------------------------------


def accumulate_sector_states(
    model: Any,
    records: list[dict[str, Any]],
    *,
    layers: Sequence[int],
    device: Any,
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """Per-sector, per-position-set, per-layer mean state over discovery prompts.

    Streams one clean forward per prompt: the full-sequence state capture is
    moved to CPU float32 layer by layer and folded into running means, so no
    raw activation is retained or persisted.  Returns
    ``{sector: {position_set: {layer: {"sum", "count", "record_norm_sum"}}}}``.
    """
    layers = sorted(set(int(layer) for layer in layers))
    position_sets: set[str] = set()
    for record in records:
        position_sets.update(record["position_sets"])
    state: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for record in records:
        tensor = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
        captured = record_residuals(model, tensor, layers)
        sector = str(record["sector"])
        index_by_set = {
            position_set: torch.as_tensor(
                sorted(set(int(p) for p in record["position_sets"][position_set]))
            )
            for position_set in position_sets
        }
        per_sector = state.setdefault(sector, {})
        for position_set in sorted(position_sets):
            index = index_by_set[position_set]
            per_position = per_sector.setdefault(position_set, {})
            for layer in layers:
                values = captured[layer][
                    0, index.to(captured[layer].device)
                ].float().cpu()
                position_mean = values.mean(0)
                entry = per_position.setdefault(
                    layer,
                    {"sum": torch.zeros_like(position_mean), "count": 0},
                )
                entry["sum"] += position_mean
                entry["count"] += 1
                entry["record_norm_sum"] = (
                    entry.get("record_norm_sum", 0.0)
                    + float(position_mean.norm())
                )
        del captured
    for sector in state:
        for position_set in state[sector]:
            for entry in state[sector][position_set].values():
                if entry["count"] == 0:
                    raise ValueError("sector state accumulated no prompts")
    return state


def _sector_state_rows(
    state: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sector in sorted(state):
        for position_set in sorted(state[sector]):
            for layer in sorted(state[sector][position_set]):
                entry = state[sector][position_set][layer]
                mean = entry["sum"] / entry["count"]
                rows.append(
                    {
                        "artifact_type": "outcome_geometry_sector_state",
                        "schema_version": 1,
                        "sector": sector,
                        "position_set": position_set,
                        "layer": layer,
                        "state_mean_norm": float(mean.norm()),
                        "state_mean_sha256": direction_hash(mean),
                        "mean_record_state_norm": (
                            entry["record_norm_sum"] / entry["count"]
                        ),
                        "record_count": entry["count"],
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Optional TF-IDF sector prototype directions (frozen, lens-derived)
# ---------------------------------------------------------------------------


def load_tfidf_prototype_specs(
    path: Path,
    *,
    model_name: str,
    source_sector: str,
    contrast_sector: str,
) -> tuple[PrototypeSpec, PrototypeSpec, list[int]]:
    """Load frozen contrastive TF-IDF prototype specs from a sector config."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != TFIDF_CONFIG_ARTIFACT_TYPE:
        raise ValueError(
            f"tfidf config artifact_type must be {TFIDF_CONFIG_ARTIFACT_TYPE!r}"
        )
    if payload.get("model") != model_name:
        raise ValueError("tfidf config model does not match the run model")
    source = PrototypeSpec.from_dict(payload["source"])
    target = PrototypeSpec.from_dict(payload["target"])
    if source.sector != source_sector or target.sector != contrast_sector:
        raise ValueError(
            "tfidf config sectors must match source/contrast sectors "
            f"({source.sector!r}/{target.sector!r} vs "
            f"{source_sector!r}/{contrast_sector!r})"
        )
    for spec in (source, target):
        if spec.score_type != TFIDF_SCORE_TYPE:
            raise ValueError(
                f"tfidf geometry requires {TFIDF_SCORE_TYPE!r} prototypes, "
                f"got {spec.score_type!r}"
            )
    layers = sorted(set(int(layer) for layer in payload["layers"]))
    if not layers:
        raise ValueError("tfidf config has no layers")
    return source, target, layers


# ---------------------------------------------------------------------------
# Analysis: sector state difference, projection, parallel/perpendicular split
# ---------------------------------------------------------------------------


def _direction_section(
    directions: Mapping[str, Mapping[int, torch.Tensor]],
) -> dict[str, dict[str, str | float]]:
    return {
        rule: {
            str(layer): {
                "norm": float(directions[rule][layer].norm()),
                "sha256": direction_hash(directions[rule][layer]),
            }
            for layer in sorted(directions[rule])
        }
        for rule in sorted(directions)
    }


def _sector_mean(
    state: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    sector: str,
    position_set: str,
    layer: int,
) -> torch.Tensor:
    entry = state[sector][position_set][layer]
    return entry["sum"] / entry["count"]


def analyze_outcome_geometry(
    *,
    directions: Mapping[str, Mapping[int, torch.Tensor]],
    state: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    source_sector: str,
    contrast_sector: str,
    tfidf_prototypes: Mapping[str, Mapping[int, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Compact per-layer geometry metrics; no vectors are returned or stored."""
    position_sets = sorted(
        {position_set for sector in state for position_set in state[sector]}
    )
    rules = sorted(directions)
    layers = sorted(
        {layer for per_layer in state[source_sector].values() for layer in per_layer}
    )
    layer_entries: dict[str, dict[str, Any]] = {}
    for layer in layers:
        sector_state: dict[str, Any] = {}
        for position_set in position_sets:
            delta = _sector_mean(state, source_sector, position_set, layer) - _sector_mean(
                state, contrast_sector, position_set, layer
            )
            sector_state[position_set] = {
                "delta_state_norm": float(delta.norm()),
                "delta_state_sha256": direction_hash(delta),
                "projection": {
                    rule: project_onto_direction(delta, directions[rule][layer])
                    for rule in rules
                },
            }
        entry: dict[str, Any] = {
            "direction": _direction_section(directions),
            "sector_state": sector_state,
        }
        if tfidf_prototypes is not None:
            section: dict[str, Any] = {}
            for label in ("source", "contrast"):
                spec_directions = tfidf_prototypes.get(label, {})
                if layer not in spec_directions:
                    continue
                vector = spec_directions[layer]
                section[label] = {
                    "norm": float(vector.norm()),
                    "sha256": direction_hash(vector),
                    "projection": {
                        rule: project_onto_direction(vector, directions[rule][layer])
                        for rule in rules
                    },
                }
            if "source" in section and "contrast" in section:
                section["cosine_source_contrast"] = _cosine(
                    tfidf_prototypes["source"][layer],
                    tfidf_prototypes["contrast"][layer],
                )
            cosine_to_delta: dict[str, Any] = {}
            for position_set in position_sets:
                delta = _sector_mean(state, source_sector, position_set, layer) - _sector_mean(
                    state, contrast_sector, position_set, layer
                )
                per_sector: dict[str, Any] = {}
                for label in ("source", "contrast"):
                    if layer in tfidf_prototypes.get(label, {}):
                        per_sector[label] = _cosine(
                            tfidf_prototypes[label][layer], delta
                        )
                if per_sector:
                    cosine_to_delta[position_set] = per_sector
            if cosine_to_delta:
                section["cosine_to_delta_state"] = cosine_to_delta
            if section:
                entry["tfidf_prototype"] = section
        layer_entries[str(layer)] = entry
    return {
        "layers": layer_entries,
        "rules": rules,
        "position_sets": position_sets,
    }


# ---------------------------------------------------------------------------
# Pipeline: prepare -> forward -> analyze -> finalize
# ---------------------------------------------------------------------------


def _load_bound_payload(
    path: Path, *, label: str, expected_type: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != expected_type:
        raise ValueError(f"{label} has artifact_type {payload.get('artifact_type')!r}")
    return payload


def _verify_bound_inputs(
    *,
    identity: Mapping[str, Any],
    input_path: Path,
    split_manifest: Path,
    config_path: Path,
    config: OutcomeFlipConfig,
    model_name: str,
) -> None:
    if identity["model"] != model_name:
        raise ValueError("direction identity model does not match the run model")
    if identity["input_sha256"] != sha256_file(input_path):
        raise ValueError(
            "run input does not match the discovery input frozen in the identity"
        )
    if identity["split_manifest_sha256"] != sha256_file(split_manifest):
        raise ValueError(
            "run split manifest does not match the discovery manifest frozen in the identity"
        )
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


def _verification_records(
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
        sector=config.source_sector,
        prompt_columns=set(identity["prompt_columns"]),
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


def _prepare_geometry_stage(
    run: ArtifactRun,
    *,
    source_records: list[dict[str, Any]],
    contrast_records: list[dict[str, Any]],
    config: OutcomeFlipConfig,
    contrast_sector: str,
    identity: Mapping[str, Any],
    resolved_columns: set[str],
) -> int:
    prepare_dir = run.run_directory / "prepare"
    prompt_records_path = prepare_dir / "prompt_records.jsonl"
    prepare_metadata_path = prepare_dir / "metadata.json"
    rows = (
        {
            "schema_version": 1,
            "artifact_type": "outcome_geometry_prompt_record",
            "record_id": record["record_id"],
            "ticker": record["ticker"],
            "name": record["name"],
            "sector": record["sector"],
            "marketcap": record["marketcap"],
            "prompt_column": record["prompt_column"],
            "split": "discovery",
            "prompt": record["prompt"],
        }
        for record in source_records + contrast_records
    )
    with run.stage("prepare") as stage:
        count = write_jsonl(prompt_records_path, rows, overwrite=False)
        write_metadata(
            prepare_metadata_path,
            {
                "artifact_type": "outcome_geometry_prepare_metadata",
                "schema_version": 1,
                "split": "discovery",
                "source_sector": config.source_sector,
                "contrast_sector": contrast_sector,
                "prompt_columns": sorted(resolved_columns),
                "fitted_layers": list(config.fitted_layers),
                "position_rules": list(config.position_rules),
                "position_sets": sorted(
                    list(config.position_rules) + ["final_position"]
                ),
                "direction_identity_record_count": identity["record_count"],
                "source_record_count": len(source_records),
                "contrast_record_count": len(contrast_records),
            },
            overwrite=False,
        )
        stage.count(count)
    run.manifest.register_artifact(
        prompt_records_path,
        artifact_type="outcome_geometry_prompt_record",
        stage="prepare",
        role="output",
        record_count=count,
    )
    run.manifest.register_artifact(
        prepare_metadata_path,
        artifact_type="outcome_geometry_prepare_metadata",
        stage="prepare",
        role="output",
    )
    run.manifest.save()
    return count


def run_outcome_geometry_pipeline(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    config_path: str | Path,
    direction_identity_path: str | Path,
    model_name: str,
    run_id: str,
    dataset: str = GEOMETRY_DATASET,
    artifact_root: str | Path = "artifacts",
    contrast_sector: str = "Financial Services",
    lens_path: str | Path | None = None,
    tfidf_config_path: str | Path | None = None,
    max_seq_len: int = 1024,
    prompt_columns: set[str] | None = None,
) -> Path:
    """Run the outcome direction geometric projection diagnostic once.

    Consumes a frozen V2 direction identity (SHA-verified by recomputation in
    deterministic mode), streams clean discovery forwards for the source and
    contrast sectors, and emits compact per-layer projection metrics.  See
    ``docs/jspace-token-experiments/details/report-v2-geometry.md`` for the protocol and
    interpretation limits.
    """
    _enable_deterministic_gpu()
    input_path = Path(input_path)
    split_manifest = Path(split_manifest)
    config_path = Path(config_path)
    identity_path = Path(direction_identity_path)
    for path, label in (
        (input_path, "baseline prompt CSV"),
        (split_manifest, "split manifest"),
        (config_path, "outcome flip config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not str(contrast_sector).strip():
        raise ValueError("contrast_sector must be a non-empty sector name")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if config_payload.get("artifact_type") != OUTCOME_FLIP_ARTIFACT_TYPE:
        raise ValueError(f"config artifact_type must be {OUTCOME_FLIP_ARTIFACT_TYPE!r}")
    config = OutcomeFlipConfig.from_dict(config_payload)
    if config.model != model_name:
        raise ValueError(
            f"run model {model_name!r} does not match model {config.model!r} "
            "frozen in config"
        )
    if config.split_manifest_sha256 != sha256_file(split_manifest):
        raise ValueError(
            "run split manifest does not match the manifest frozen in config"
        )
    if contrast_sector == config.source_sector:
        raise ValueError("contrast sector must differ from the source sector")

    identity = _load_bound_payload(
        identity_path,
        label="direction identity",
        expected_type="outcome_flip_direction_identity",
    )
    if identity.get("split") != "discovery":
        raise ValueError("direction identity must come from a discovery run")
    _verify_bound_inputs(
        identity=identity,
        input_path=input_path,
        split_manifest=split_manifest,
        config_path=config_path,
        config=config,
        model_name=model_name,
    )

    tfidf_specs: tuple[PrototypeSpec, PrototypeSpec, list[int]] | None = None
    tfidf_path: Path | None = None
    tfidf_layers: list[int] = []
    if tfidf_config_path is not None:
        tfidf_path = Path(tfidf_config_path)
        if lens_path is None:
            raise ValueError("--tfidf-config requires --lens (canonical lens)")
        tfidf_specs = load_tfidf_prototype_specs(
            tfidf_path,
            model_name=model_name,
            source_sector=config.source_sector,
            contrast_sector=contrast_sector,
        )
        tfidf_layers = sorted(
            set(tfidf_specs[2]) & set(int(layer) for layer in config.fitted_layers)
        )
        if not tfidf_layers:
            raise ValueError(
                "tfidf config layers do not intersect the fitted layers"
            )

    split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    assignments = {str(k): str(v) for k, v in split_payload["assignments"].items()}
    resolved_columns = (
        set(prompt_columns) if prompt_columns else set(identity["prompt_columns"])
    )
    if prompt_columns is not None and resolved_columns != set(
        identity["prompt_columns"]
    ):
        raise ValueError(
            "prompt columns differ from the discovery prompt columns frozen "
            "in the direction identity"
        )
    source_records = _prompt_records(
        input_path,
        assignments=assignments,
        sector=config.source_sector,
        prompt_columns=resolved_columns,
    )
    contrast_records = _prompt_records(
        input_path,
        assignments=assignments,
        sector=contrast_sector,
        prompt_columns=resolved_columns,
    )
    preflight_tokenizer = load_tokenizer(model_name)
    _preflight_records(
        preflight_tokenizer,
        source_records,
        config=config,
        max_seq_len=max_seq_len,
        sector=config.source_sector,
    )
    _preflight_records(
        preflight_tokenizer,
        contrast_records,
        config=config,
        max_seq_len=max_seq_len,
        sector=contrast_sector,
    )
    verification_records = _verification_records(
        input_path,
        assignments=assignments,
        config=config,
        identity=identity,
    )
    _preflight_records(
        preflight_tokenizer,
        verification_records,
        config=config,
        max_seq_len=max_seq_len,
        sector=config.source_sector,
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
    run.manifest.register_artifact(
        identity_path,
        artifact_type="outcome_flip_direction_identity",
        stage="prepare",
        role="input",
    )
    if tfidf_path is not None:
        run.manifest.register_artifact(
            tfidf_path,
            artifact_type=TFIDF_CONFIG_ARTIFACT_TYPE,
            stage="prepare",
            role="input",
        )
    run.manifest.save()
    try:
        prepare_count = _prepare_geometry_stage(
            run,
            source_records=source_records,
            contrast_records=contrast_records,
            config=config,
            contrast_sector=contrast_sector,
            identity=identity,
            resolved_columns=resolved_columns,
        )
        model, tokenizer, fallback_device = load_model(model_name)
        device = getattr(model, "input_device", fallback_device)
        materialized_source = _materialize_records(source_records, tokenizer, config)
        materialized_contrast = _materialize_records(
            contrast_records, tokenizer, config
        )
        materialized_verification = _materialize_records(
            verification_records, tokenizer, config
        )
        directions, _permutation_directions = verify_direction_identity(
            model=model,
            tokenizer=tokenizer,
            discovery_records=materialized_verification,
            config=config,
            identity=identity,
            device=device,
        )
        layers = list(config.fitted_layers)
        state = accumulate_sector_states(
            model,
            materialized_source + materialized_contrast,
            layers=layers,
            device=device,
        )
        tfidf_prototypes: dict[str, dict[int, torch.Tensor]] | None = None
        loaded_lens = None
        if tfidf_specs is not None:
            loaded_lens = load_validated_lens(
                model=model,
                model_name=model_name,
                lens_path=lens_path,
                artifact_root=artifact_root,
            )
            run.manifest.register_artifact(
                loaded_lens.path,
                artifact_type="jacobian_lens",
                stage="prepare",
                role="lens",
                metadata={"source": loaded_lens.source},
            )
            run.manifest.save()
            tfidf_prototypes = {
                "source": {
                    layer: vector.detach().float().cpu()
                    for layer, vector in layer_prototypes(
                        model, loaded_lens.lens, tfidf_specs[0], tfidf_layers
                    ).items()
                },
                "contrast": {
                    layer: vector.detach().float().cpu()
                    for layer, vector in layer_prototypes(
                        model, loaded_lens.lens, tfidf_specs[1], tfidf_layers
                    ).items()
                },
            }
        with run.stage("forward") as stage:
            forward_dir = run.run_directory / "forward"
            stats_path = forward_dir / "sector_state_statistics.jsonl"
            written = write_jsonl(stats_path, _sector_state_rows(state), overwrite=False)
            forward_metadata: dict[str, Any] = {
                "artifact_type": "outcome_geometry_metadata",
                "schema_version": 1,
                "split": "discovery",
                "model": model_name,
                "direction_identity": str(identity_path),
                "direction_identity_sha256": sha256_file(identity_path),
                "direction_verified": True,
                "direction": _direction_section(directions),
                "deterministic_mode": True,
                "autograd": True,
                "backpropagation": True,
                "fitted_layers": layers,
                "position_sets": sorted(
                    {
                        position_set
                        for record in materialized_source + materialized_contrast
                        for position_set in record["position_sets"]
                    }
                ),
                "source_sector": config.source_sector,
                "contrast_sector": contrast_sector,
                "record_count": prepare_count,
                "sector_record_counts": {
                    config.source_sector: len(materialized_source),
                    contrast_sector: len(materialized_contrast),
                },
            }
            if loaded_lens is not None:
                forward_metadata["canonical_lens"] = str(loaded_lens.path)
                forward_metadata["lens_sha256"] = sha256_file(loaded_lens.path)
                forward_metadata["lens_source"] = loaded_lens.source
            if tfidf_prototypes is not None:
                forward_metadata["tfidf_config"] = str(tfidf_path)
                forward_metadata["tfidf_config_sha256"] = sha256_file(tfidf_path)
                forward_metadata["tfidf_layers"] = tfidf_layers
                forward_metadata["tfidf_prototypes"] = {
                    label: {
                        str(layer): {
                            "norm": float(vector.norm()),
                            "sha256": direction_hash(vector),
                        }
                        for layer, vector in sorted(spec.items())
                    }
                    for label, spec in tfidf_prototypes.items()
                }
                forward_metadata["tfidf_interpretation"] = _TFIDF_INTERPRETATION
            write_metadata(forward_dir / "metadata.json", forward_metadata, overwrite=False)
            stage.count(written)
        run.manifest.register_artifact(
            stats_path,
            artifact_type="outcome_geometry_sector_state",
            stage="forward",
            role="output",
            record_count=written,
        )
        run.manifest.register_artifact(
            run.run_directory / "forward" / "metadata.json",
            artifact_type="outcome_geometry_metadata",
            stage="forward",
            role="output",
        )
        run.manifest.save()

        analysis = analyze_outcome_geometry(
            directions=directions,
            state=state,
            source_sector=config.source_sector,
            contrast_sector=contrast_sector,
            tfidf_prototypes=tfidf_prototypes,
        )
        analyze_dir = run.run_directory / "analyze"
        summary_path = analyze_dir / "outcome_geometry_analysis.json"
        with run.stage("analyze") as stage:
            write_json(
                summary_path,
                {
                    "artifact_type": "outcome_geometry_analysis",
                    "schema_version": 1,
                    "split": "discovery",
                    "model": model_name,
                    "source_sector": config.source_sector,
                    "contrast_sector": contrast_sector,
                    "prompt_columns": sorted(resolved_columns),
                    "input": str(input_path),
                    "input_sha256": sha256_file(input_path),
                    "split_manifest": str(split_manifest),
                    "split_manifest_sha256": sha256_file(split_manifest),
                    "config": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "direction_identity": str(identity_path),
                    "direction_identity_sha256": sha256_file(identity_path),
                    "tfidf_config": (
                        str(tfidf_path) if tfidf_path is not None else None
                    ),
                    "tfidf_config_sha256": (
                        sha256_file(tfidf_path) if tfidf_path is not None else None
                    ),
                    "canonical_lens": (
                        str(loaded_lens.path) if loaded_lens is not None else None
                    ),
                    "lens_sha256": (
                        sha256_file(loaded_lens.path) if loaded_lens is not None else None
                    ),
                    "fitted_layers": layers,
                    "record_counts": {
                        "source": len(materialized_source),
                        "contrast": len(materialized_contrast),
                        "source_tickers": len(
                            {record["ticker"] for record in source_records}
                        ),
                        "contrast_tickers": len(
                            {record["ticker"] for record in contrast_records}
                        ),
                    },
                    "definition": {
                        "delta_state": (
                            "source sector mean state minus contrast sector mean "
                            "state per layer and position set, discovery split"
                        ),
                        "projection": (
                            "coefficient = <delta_state, d> / ||d||^2; "
                            "parallel = coefficient * d; "
                            "perpendicular = delta_state - parallel"
                        ),
                    },
                    "interpretation_limits": list(_INTERPRETATION_LIMITS),
                    **analysis,
                },
                overwrite=False,
            )
            write_metadata(
                analyze_dir / "metadata.json",
                {
                    "artifact_type": "outcome_geometry_analysis_metadata",
                    "schema_version": 1,
                    "interpretation": "descriptive_geometry",
                    "layer_count": len(layers),
                    "record_count": prepare_count,
                },
                overwrite=False,
            )
            stage.count(len(layers))
        run.manifest.register_artifact(
            summary_path,
            artifact_type="outcome_geometry_analysis",
            stage="analyze",
            role="output",
        )
        run.manifest.register_artifact(
            analyze_dir / "metadata.json",
            artifact_type="outcome_geometry_analysis_metadata",
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
    "GEOMETRY_DATASET",
    "accumulate_sector_states",
    "analyze_outcome_geometry",
    "load_tfidf_prototype_specs",
    "project_onto_direction",
    "run_outcome_geometry_pipeline",
]
