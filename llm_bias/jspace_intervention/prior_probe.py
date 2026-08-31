"""V2 zero-evidence header-only prior probe (readout-only diagnostic).

Protocol: ``docs/jspace-token-experiments/proposal-v2.md`` (section
"Zero-Evidence Header-Only Prior Probe").  Every condition uses the same
neutral evidence body; only the header ticker and sector label vary.  At the
scoring prompt's final token position the probe reports, per fitted layer,
the projection of the residual onto the frozen V2 outcome direction ``d_l``
(unit-norm, so the projection is the FP32 dot product) plus the V2
Buy-vs-Sell margin from the same position.

The probe is association/prior evidence only: no intervention, no control
arm, no success gate.  Raw residuals and direction vectors are kept in
memory and verified against the frozen direction identity; artifacts carry
compact projections, margins, and provenance only.
"""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.analysis.statistics import paired_bootstrap_ci
from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import score_single_token_margin_fp32
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids
from llm_bias.jspace_intervention import outcome_flip
from llm_bias.jspace_intervention.schemas import (
    OutcomeFlipConfig,
    PriorProbeConfig,
)

PROBE_CONFIG_ARTIFACT_TYPE = "outcome_prior_probe_config"
PROBE_TEMPLATE_VERSION = 1
NEUTRAL_EVIDENCE_ITEM = "No evidence provided."
PROBE_TEMPLATE = (
    "Refer to the evidence below to make a final investment decision.\n"
    "Stock Ticker: [{ticker}]\n"
    "Sector: [{sector}]\n"
    "--- Evidence ---\n"
    "1. {neutral_item}\n"
    "---\n"
    'Respond with one valid JSON object containing only the keys "decision" '
    '(buy | sell) and "reason" (brief justification). Do not choose hold.'
)

_INTERPRETATION_LIMITS = (
    "association/prior evidence under the tested prompt template, model, and "
    "frozen V2 direction only; no intervention or control arm was run, so "
    "results cannot be read as causal effects of the ticker or sector label "
    "on the decision. The Technology sector label is partially in-sample for "
    "the direction source (fitted on Technology discovery prompts); the "
    "Financial Services label and all probed tickers are out-of-sample. "
    "Contrast CIs describe sampling variability of the 12-condition "
    "descriptive sample, not significance gates."
)


def render_probe_prompt(
    *,
    ticker: str,
    sector: str,
    neutral_item: str = NEUTRAL_EVIDENCE_ITEM,
) -> str:
    """Render one frozen version-1 probe prompt for a (ticker, sector) cell."""
    return PROBE_TEMPLATE.format(
        ticker=ticker, sector=sector, neutral_item=neutral_item
    )


def prepare_probe_scoring_prompt(
    tokenizer: Any,
    raw_prompt: str,
    *,
    decision_prefix: str,
) -> str:
    """Chat-format one probe prompt and append the V2 decision prefix."""
    formatted = format_prompt(
        tokenizer, raw_prompt, use_chat_template=True, enable_thinking=False
    )
    if raw_prompt not in formatted:
        raise ValueError("formatted chat prompt does not contain the raw prompt")
    return formatted + decision_prefix


def project_positions(
    residuals: Mapping[int, torch.Tensor],
    directions: Mapping[int, torch.Tensor],
    *,
    position: int,
    scale_floor: float,
) -> dict[str, dict[str, float]]:
    """Per-layer projection of one residual position onto unit directions.

    ``directions[layer]`` must be unit norm (the frozen V2 directions are),
    so the projection equals the FP32 dot product.  ``projection_relative``
    divides by the V2 local scale ``max(norm, scale_floor)``; a residual
    fully aligned with the direction gives ``+1``.
    """
    if not 0 <= position:
        raise ValueError("position must be non-negative")
    result: dict[str, dict[str, float]] = {}
    for layer in sorted(residuals):
        if layer not in directions:
            raise ValueError(f"missing direction for layer {layer}")
        tensor = residuals[layer]
        if tensor.shape[0] != 1 or position >= tensor.shape[1]:
            raise ValueError(f"invalid residual shape for layer {layer}")
        state = tensor[0, position, :].float().cpu()
        direction = directions[layer].detach().float().cpu()
        direction_norm = float(direction.norm())
        if not math.isclose(direction_norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"direction for layer {layer} is not unit norm "
                f"({direction_norm}); projection normalization would be wrong"
            )
        projection = float(torch.dot(state, direction))
        norm = float(state.norm())
        if not (math.isfinite(projection) and math.isfinite(norm) and norm > 0.0):
            raise ValueError(f"non-finite projection input for layer {layer}")
        result[str(layer)] = {
            "projection": projection,
            "projection_relative": projection / max(norm, scale_floor),
            "norm": norm,
        }
    return result


# ---------------------------------------------------------------------------
# Frozen input loading and fail-closed verification
# ---------------------------------------------------------------------------


def _load_json(path: Path, *, label: str, expected_type: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != expected_type:
        raise ValueError(
            f"{label} has artifact_type {payload.get('artifact_type')!r}, "
            f"expected {expected_type!r}"
        )
    return payload


def load_probe_bindings(
    config: PriorProbeConfig,
) -> tuple[OutcomeFlipConfig, dict[str, Any]]:
    """Verify every frozen input the probe config binds, fail closed."""
    for path, label, expected in (
        (Path(config.input), "probe input", None),
        (Path(config.split_manifest), "split manifest", None),
        (Path(config.outcome_flip_config), "outcome flip config", None),
        (Path(config.direction_identity), "direction identity", None),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if sha256_file(Path(config.input)) != config.input_sha256:
        raise ValueError("probe input no longer matches the frozen input SHA")
    if sha256_file(Path(config.split_manifest)) != config.split_manifest_sha256:
        raise ValueError("split manifest no longer matches the frozen SHA")
    if (
        sha256_file(Path(config.outcome_flip_config))
        != config.outcome_flip_config_sha256
    ):
        raise ValueError("outcome flip config no longer matches the frozen SHA")
    if (
        sha256_file(Path(config.direction_identity))
        != config.direction_identity_sha256
    ):
        raise ValueError("direction identity no longer matches the frozen SHA")

    v2_payload = _load_json(
        Path(config.outcome_flip_config),
        label="outcome flip config",
        expected_type="outcome_flip_config",
    )
    v2_config = OutcomeFlipConfig.from_dict(v2_payload)
    if v2_config.model != config.model:
        raise ValueError(
            f"probe model {config.model!r} does not match outcome flip "
            f"config model {v2_config.model!r}"
        )
    if v2_config.split_manifest_sha256 != config.split_manifest_sha256:
        raise ValueError(
            "outcome flip config binds a different split manifest than the probe"
        )

    identity = _load_json(
        Path(config.direction_identity),
        label="direction identity",
        expected_type="outcome_flip_direction_identity",
    )
    if identity.get("split") != "discovery":
        raise ValueError("direction identity must come from a discovery run")
    if identity.get("model") != config.model:
        raise ValueError("direction identity model does not match the probe model")
    if identity.get("input_sha256") != config.input_sha256:
        raise ValueError("direction identity was fitted on a different input")
    if identity.get("split_manifest_sha256") != config.split_manifest_sha256:
        raise ValueError(
            "direction identity binds a different split manifest than the probe"
        )
    if identity.get("config_sha256") != config.outcome_flip_config_sha256:
        raise ValueError(
            "direction identity binds a different outcome flip config than the probe"
        )
    if config.position_rule not in identity.get("position_rules", []):
        raise ValueError(
            f"position rule {config.position_rule!r} is not in the direction identity"
        )
    return v2_config, identity


def _read_ticker_index(input_path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ticker = row.get("ticker") or ""
            if not ticker:
                continue
            sector = row.get("sector") or ""
            previous = index.get(ticker)
            if previous is not None and previous["sector"] != sector:
                raise ValueError(f"ticker {ticker!r} has conflicting sectors")
            index.setdefault(
                ticker,
                {
                    "name": row.get("name") or "",
                    "sector": sector,
                    "marketcap": row.get("marketcap") or "",
                },
            )
    return index


def build_probe_records(
    config: PriorProbeConfig,
    *,
    ticker_index: Mapping[str, Mapping[str, str]],
    assignments: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Resolve frozen conditions against the canonical input, fail closed."""
    known_sectors = {row["sector"] for row in ticker_index.values()}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for condition in config.conditions:
        key = (condition.ticker, condition.sector)
        if key in seen:
            raise ValueError(f"duplicate prior probe condition {key}")
        seen.add(key)
        info = ticker_index.get(condition.ticker)
        if info is None:
            raise ValueError(
                f"probe ticker {condition.ticker!r} is not in the input CSV"
            )
        if condition.sector not in known_sectors:
            raise ValueError(
                f"probe sector {condition.sector!r} is not a canonical "
                "sector label in the input CSV"
            )
        raw_prompt = render_probe_prompt(
            ticker=condition.ticker,
            sector=condition.sector,
            neutral_item=config.neutral_evidence_item,
        )
        records.append(
            {
                "condition_id": stable_record_id(
                    condition.ticker, condition.sector, "prior-probe"
                ),
                "ticker": condition.ticker,
                "sector": condition.sector,
                "canonical_sector": info["sector"],
                "name": info["name"],
                "marketcap": info["marketcap"],
                "own_sector": condition.sector == info["sector"],
                "split": assignments.get(condition.ticker, "unassigned"),
                "prompt": raw_prompt,
            }
        )
    return records


def _preflight_probe(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    config: PriorProbeConfig,
) -> None:
    for record in records:
        scoring_prompt = prepare_probe_scoring_prompt(
            tokenizer, record["prompt"], decision_prefix=config.decision_prefix
        )
        length = len(input_ids(tokenizer, scoring_prompt, add_special_tokens=True))
        if length > config.max_seq_len:
            raise ValueError(
                f"probe scoring prompt for {record['ticker']}/{record['sector']} "
                f"has {length} tokens, limit {config.max_seq_len}"
            )


# ---------------------------------------------------------------------------
# Analysis: descriptive group means and contrasts (no gates)
# ---------------------------------------------------------------------------


def _paired_sector_differences(
    rows: list[dict[str, Any]], *, plus: str, minus: str, value
) -> list[float]:
    """Per-ticker difference between the two probed sector labels."""
    by_ticker: dict[str, dict[str, float]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], {})[row["sector"]] = value(row)
    for ticker, levels in by_ticker.items():
        if set(levels) != {plus, minus}:
            raise ValueError(
                f"ticker {ticker} must be probed under exactly both sector labels"
            )
    return [levels[plus] - levels[minus] for levels in by_ticker.values()]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _group_metric(
    rows: list[dict[str, Any]], *, key: str, layers: Sequence[int]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    result: dict[str, Any] = {}
    for name in sorted(groups):
        members = groups[name]
        per_layer: dict[str, dict[str, float]] = {}
        for layer in layers:
            layer_key = str(layer)
            per_layer[layer_key] = {
                "projection": _mean(
                    float(row["layers"][layer_key]["projection"]) for row in members
                ),
                "projection_relative": _mean(
                    float(row["layers"][layer_key]["projection_relative"])
                    for row in members
                ),
            }
        result[name] = {
            "condition_count": len(members),
            "mean_margin": _mean(float(row["margin"]) for row in members),
            "layers": per_layer,
        }
    return result


_PROBE_METRICS = ("projection", "projection_relative")


def analyze_prior_probe(
    rows: list[dict[str, Any]],
    *,
    config: PriorProbeConfig,
    layers: Sequence[int],
) -> dict[str, Any]:
    """Descriptive group means and contrasts over the probe conditions."""
    if not rows:
        raise ValueError("prior probe analysis requires result rows")
    plus_sector, minus_sector = config.contrast_sectors
    sectors = {row["sector"] for row in rows}
    if sectors != {plus_sector, minus_sector}:
        raise ValueError(
            "probed sector labels do not match the frozen contrast_sectors"
        )
    groups = sorted({row["canonical_sector"] for row in rows})
    if len(groups) != 2:
        raise ValueError(
            "prior probe analysis expects exactly two canonical ticker groups"
        )
    # Deterministic orientation: the alphabetically later group is the plus side
    # (Technology - Financial Services for the frozen design).
    minus_group, plus_group = groups
    plus_ticker, minus_ticker = config.contrast_tickers
    if not {plus_ticker, minus_ticker} <= {row["ticker"] for row in rows}:
        raise ValueError(
            "contrast tickers must both be present in the probed conditions"
        )
    seed = config.bootstrap_seed
    n_resamples = config.bootstrap_samples

    def margin_of(row: dict[str, Any]) -> float:
        return float(row["margin"])

    def layer_of(layer: int, metric: str):
        def pick(row: dict[str, Any]) -> float:
            return float(row["layers"][str(layer)][metric])

        return pick

    sector_diffs = _paired_sector_differences(
        rows, plus=plus_sector, minus=minus_sector, value=margin_of
    )
    sector_contrast: dict[str, Any] = {
        "definition": (
            f"per-ticker difference ({plus_sector} label - {minus_sector} "
            "label), averaged over tickers; paired ticker bootstrap CI"
        ),
        "margin": {
            "point": _mean(sector_diffs),
            "ci95": paired_bootstrap_ci(
                sector_diffs, seed=seed, n_resamples=n_resamples
            ),
        },
        "layers": {},
    }
    for layer in layers:
        sector_contrast["layers"][str(layer)] = {
            metric: {
                "point": _mean(
                    _paired_sector_differences(
                        rows,
                        plus=plus_sector,
                        minus=minus_sector,
                        value=layer_of(layer, metric),
                    )
                ),
                "ci95": paired_bootstrap_ci(
                    _paired_sector_differences(
                        rows,
                        plus=plus_sector,
                        minus=minus_sector,
                        value=layer_of(layer, metric),
                    ),
                    seed=seed,
                    n_resamples=n_resamples,
                ),
            }
            for metric in _PROBE_METRICS
        }

    pair_contrast: dict[str, Any] = {
        "definition": (
            f"{plus_ticker} - {minus_ticker}, averaged over the two sector "
            "labels (point estimate only)"
        ),
        "margin": {
            "point": _mean(
                [
                    margin_of(
                        next(
                            r
                            for r in rows
                            if r["ticker"] == plus_ticker
                            and r["sector"] == sector
                        )
                    )
                    - margin_of(
                        next(
                            r
                            for r in rows
                            if r["ticker"] == minus_ticker
                            and r["sector"] == sector
                        )
                    )
                    for sector in sectors
                ]
            ),
        },
        "layers": {},
    }
    group_contrast: dict[str, Any] = {
        "definition": (
            f"mean of {plus_group} tickers minus mean of {minus_group} "
            "tickers (point estimate only)"
        ),
        "margin": {
            "point": _mean(
                margin_of(r) for r in rows if r["canonical_sector"] == plus_group
            )
            - _mean(
                margin_of(r) for r in rows if r["canonical_sector"] == minus_group
            ),
        },
        "layers": {},
    }
    for layer in layers:
        pair_contrast["layers"][str(layer)] = {}
        group_contrast["layers"][str(layer)] = {}
        for metric in _PROBE_METRICS:
            pick = layer_of(layer, metric)
            pair_contrast["layers"][str(layer)][metric] = {
                "point": _mean(
                    [
                        pick(
                            next(
                                r
                                for r in rows
                                if r["ticker"] == plus_ticker
                                and r["sector"] == sector
                            )
                        )
                        - pick(
                            next(
                                r
                                for r in rows
                                if r["ticker"] == minus_ticker
                                and r["sector"] == sector
                            )
                        )
                        for sector in sectors
                    ]
                ),
            }
            group_contrast["layers"][str(layer)][metric] = {
                "point": _mean(
                    pick(r) for r in rows if r["canonical_sector"] == plus_group
                )
                - _mean(
                    pick(r) for r in rows if r["canonical_sector"] == minus_group
                ),
            }

    return {
        "condition_count": len(rows),
        "layers": [int(layer) for layer in layers],
        "groups": {
            "by_sector_label": _group_metric(rows, key="sector", layers=layers),
            "by_ticker": _group_metric(rows, key="ticker", layers=layers),
            "by_ticker_group": _group_metric(
                rows, key="canonical_sector", layers=layers
            ),
        },
        "contrasts": {
            "sector_label": sector_contrast,
            f"{plus_ticker.lower()}_minus_{minus_ticker.lower()}": pair_contrast,
            "ticker_group": group_contrast,
        },
        "interpretation": _INTERPRETATION_LIMITS,
    }


# ---------------------------------------------------------------------------
# Pipeline: prepare -> forward -> analyze -> finalize
# ---------------------------------------------------------------------------


def _measure_condition(
    *,
    model: Any,
    tokenizer: Any,
    record: Mapping[str, Any],
    config: PriorProbeConfig,
    v2_config: OutcomeFlipConfig,
    directions: Mapping[int, torch.Tensor],
    device: Any,
) -> dict[str, Any]:
    scoring_prompt = prepare_probe_scoring_prompt(
        tokenizer, record["prompt"], decision_prefix=config.decision_prefix
    )
    prompt_ids = input_ids(tokenizer, scoring_prompt, add_special_tokens=True)
    tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    residuals = record_residuals(model, tensor, v2_config.fitted_layers)
    seq_len = int(len(prompt_ids))
    layers = project_positions(
        residuals,
        directions,
        position=seq_len - 1,
        scale_floor=config.scale_floor,
    )
    margin = score_single_token_margin_fp32(
        model,
        tokenizer,
        scoring_prompt,
        config.positive_candidate,
        config.negative_candidate,
        device=device,
    )
    value = float(margin.value)
    decision = "buy" if value > 0.0 else ("sell" if value < 0.0 else "tie")
    return {
        "condition_id": record["condition_id"],
        "ticker": record["ticker"],
        "sector": record["sector"],
        "canonical_sector": record["canonical_sector"],
        "own_sector": bool(record["own_sector"]),
        "split": record["split"],
        "scoring_prompt_token_count": seq_len,
        "measurement_position": seq_len - 1,
        "margin": value,
        "margin_decision": decision,
        "score": margin.to_dict(),
        "layers": layers,
    }


def _prepare_stage(
    run: ArtifactRun,
    *,
    records: list[dict[str, Any]],
    config: PriorProbeConfig,
    v2_config: OutcomeFlipConfig,
    identity: Mapping[str, Any],
) -> int:
    prepare_dir = run.run_directory / "prepare"
    records_path = prepare_dir / "prior_probe_records.jsonl"
    metadata_path = prepare_dir / "metadata.json"
    with run.stage("prepare") as stage:
        count = write_jsonl(
            records_path,
            (
                {
                    "schema_version": 1,
                    "artifact_type": "outcome_prior_probe_record",
                    "condition_id": record["condition_id"],
                    "ticker": record["ticker"],
                    "name": record["name"],
                    "sector": record["sector"],
                    "canonical_sector": record["canonical_sector"],
                    "marketcap": record["marketcap"],
                    "own_sector": record["own_sector"],
                    "split": record["split"],
                    "prompt": record["prompt"],
                }
                for record in records
            ),
            overwrite=False,
        )
        write_metadata(
            metadata_path,
            {
                "artifact_type": "outcome_prior_probe_metadata",
                "schema_version": 1,
                "stage": "prepare",
                "template_version": PROBE_TEMPLATE_VERSION,
                "template": PROBE_TEMPLATE,
                "neutral_evidence_item": config.neutral_evidence_item,
                "conditions": [
                    {"ticker": record["ticker"], "sector": record["sector"]}
                    for record in records
                ],
                "position_rule": config.position_rule,
                "fitted_layers": list(v2_config.fitted_layers),
                "scale_floor": config.scale_floor,
                "decision_prefix": config.decision_prefix,
                "positive_candidate": config.positive_candidate,
                "negative_candidate": config.negative_candidate,
                "max_seq_len": config.max_seq_len,
                "bootstrap_seed": config.bootstrap_seed,
                "bootstrap_samples": config.bootstrap_samples,
                "direction_identity": config.direction_identity,
                "condition_count": count,
            },
            overwrite=False,
        )
        stage.count(count)
    run.manifest.register_artifact(
        records_path,
        artifact_type="outcome_prior_probe_record",
        stage="prepare",
        role="output",
        record_count=count,
    )
    run.manifest.register_artifact(
        metadata_path,
        artifact_type="outcome_prior_probe_metadata",
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
    records: list[dict[str, Any]],
    config: PriorProbeConfig,
    v2_config: OutcomeFlipConfig,
    directions: Mapping[int, torch.Tensor],
) -> tuple[Path, int]:
    forward_dir = run.run_directory / "forward"
    results_path = forward_dir / "prior_probe_results.jsonl"
    metadata_path = forward_dir / "metadata.json"

    def rows() -> Any:
        for record in records:
            yield {
                "schema_version": 1,
                "artifact_type": "outcome_prior_probe_result",
                "position_rule": config.position_rule,
                **_measure_condition(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    config=config,
                    v2_config=v2_config,
                    directions=directions,
                    device=device,
                ),
            }

    with run.stage("forward") as stage:
        count = write_jsonl(results_path, rows(), overwrite=False)
        stage.count(count)
    write_metadata(
        metadata_path,
        {
            "artifact_type": "outcome_prior_probe_metadata",
            "schema_version": 1,
            "stage": "forward",
            "operation": "zero_evidence_header_only_prior_probe",
            "template_version": PROBE_TEMPLATE_VERSION,
            "position_rule": config.position_rule,
            "fitted_layers": list(v2_config.fitted_layers),
            "projection_definition": (
                "inner product of the final-position residual with the unit "
                "V2 direction (FP32)"
            ),
            "projection_relative_definition": (
                "projection / max(residual L2 norm, scale_floor)"
            ),
            "margin_scoring": "single_token_fp32_final_norm_unembedding_next_token",
            "measurement_position": "scoring prompt final token position",
            "direction_identity": config.direction_identity,
            "direction_identity_sha256": config.direction_identity_sha256,
            "direction_verified": True,
            "condition_count": count,
        },
        overwrite=False,
    )
    run.manifest.register_artifact(
        results_path,
        artifact_type="outcome_prior_probe_result",
        stage="forward",
        role="output",
        record_count=count,
    )
    run.manifest.register_artifact(
        metadata_path,
        artifact_type="outcome_prior_probe_metadata",
        stage="forward",
        role="output",
    )
    run.manifest.save()
    return results_path, count


def run_prior_probe_pipeline(
    *,
    config_path: str | Path,
    model_name: str,
    run_id: str,
    dataset: str = "jspace-outcome-direction-flip",
    artifact_root: str | Path = "artifacts",
) -> Path:
    """Run one zero-evidence header-only prior probe into a canonical run tree.

    The frozen probe config binds the input CSV, split manifest, V2 config,
    and direction identity by path + SHA-256.  The V2 direction is recomputed
    in memory and verified against the frozen identity (fail closed) before
    any condition is measured.
    """
    outcome_flip._enable_deterministic_gpu()
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"probe config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != PROBE_CONFIG_ARTIFACT_TYPE:
        raise ValueError(
            f"config artifact_type must be {PROBE_CONFIG_ARTIFACT_TYPE!r}"
        )
    config = PriorProbeConfig.from_dict(payload)
    if config.model != model_name:
        raise ValueError(
            f"run model {model_name!r} does not match model "
            f"{config.model!r} frozen in the probe config"
        )
    v2_config, identity = load_probe_bindings(config)

    split_payload = json.loads(Path(config.split_manifest).read_text(encoding="utf-8"))
    assignments = {str(k): str(v) for k, v in split_payload["assignments"].items()}
    ticker_index = _read_ticker_index(Path(config.input))
    records = build_probe_records(
        config, ticker_index=ticker_index, assignments=assignments
    )
    preflight_tokenizer = load_tokenizer(model_name)
    _preflight_probe(preflight_tokenizer, records, config=config)
    verification_records = outcome_flip._discovery_verification_records(
        Path(config.input),
        assignments=assignments,
        config=v2_config,
        identity=identity,
    )
    outcome_flip._preflight_records(
        preflight_tokenizer,
        verification_records,
        config=v2_config,
        max_seq_len=config.max_seq_len,
    )
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(
        Path(config.input), artifact_type="trial_plan_prompts", stage="prepare", role="input"
    )
    run.manifest.register_artifact(
        Path(config.split_manifest),
        artifact_type="jspace_intervention_splits",
        stage="prepare",
        role="input",
    )
    run.manifest.register_artifact(
        Path(config.outcome_flip_config),
        artifact_type="outcome_flip_config",
        stage="prepare",
        role="input",
    )
    run.manifest.register_artifact(
        Path(config.direction_identity),
        artifact_type="outcome_flip_direction_identity",
        stage="prepare",
        role="input",
    )
    run.manifest.register_artifact(
        config_path,
        artifact_type=PROBE_CONFIG_ARTIFACT_TYPE,
        stage="prepare",
        role="input",
    )
    run.manifest.save()
    try:
        _prepare_stage(
            run,
            records=records,
            config=config,
            v2_config=v2_config,
            identity=identity,
        )
        model, tokenizer, fallback_device = load_model(model_name)
        device = getattr(model, "input_device", fallback_device)
        verification_materialized = outcome_flip._materialize_records(
            verification_records, tokenizer, v2_config
        )
        directions, _ = outcome_flip.verify_direction_identity(
            model=model,
            tokenizer=tokenizer,
            discovery_records=verification_materialized,
            config=v2_config,
            identity=identity,
            device=device,
        )
        probe_directions = directions[config.position_rule]
        results_path, count = _forward_stage(
            run,
            model=model,
            tokenizer=tokenizer,
            device=device,
            records=records,
            config=config,
            v2_config=v2_config,
            directions=probe_directions,
        )
        analyze_dir = run.run_directory / "analyze"
        rows = [
            json.loads(line)
            for line in results_path.open(encoding="utf-8")
            if line.strip()
        ]
        layers = list(v2_config.fitted_layers)
        summary = analyze_prior_probe(rows, config=config, layers=layers)
        summary_path = analyze_dir / "prior_probe_analysis.json"
        with run.stage("analyze") as stage:
            write_json(
                summary_path,
                {
                    "artifact_type": "outcome_prior_probe_analysis",
                    "schema_version": 1,
                    "forward": "forward/prior_probe_results.jsonl",
                    "forward_sha256": sha256_file(results_path),
                    **summary,
                },
                overwrite=False,
            )
            write_metadata(
                analyze_dir / "metadata.json",
                {
                    "artifact_type": "outcome_prior_probe_metadata",
                    "schema_version": 1,
                    "stage": "analyze",
                    "interpretation": "association_prior_readout",
                    "condition_count": len(rows),
                },
                overwrite=False,
            )
            stage.count(len(rows))
        run.manifest.register_artifact(
            summary_path,
            artifact_type="outcome_prior_probe_analysis",
            stage="analyze",
            role="output",
        )
        run.manifest.register_artifact(
            analyze_dir / "metadata.json",
            artifact_type="outcome_prior_probe_metadata",
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
    "NEUTRAL_EVIDENCE_ITEM",
    "PROBE_CONFIG_ARTIFACT_TYPE",
    "PROBE_TEMPLATE",
    "PROBE_TEMPLATE_VERSION",
    "analyze_prior_probe",
    "build_probe_records",
    "load_probe_bindings",
    "prepare_probe_scoring_prompt",
    "project_positions",
    "render_probe_prompt",
    "run_prior_probe_pipeline",
]
