"""V2 auxiliary diagnostic: Jacobian-lens decode of the outcome direction d_l.

Direction decode transports the calibration-frozen outcome direction ``d_l``
into the final-layer basis with the validated canonical Jacobian lens and
unembeds it (FP32 final norm + LM head, the same tail as
``fp32_next_token_log_probs`` without the log-softmax) to report which
vocabulary tokens each transported direction points at.

The direction is never accepted or persisted: it is recomputed in memory
under the V2 no-persistence protocol (deterministic algorithms, frozen
discovery inputs) and verified layer-wise against the frozen
``outcome_flip_direction_identity``, failing closed on any mismatch.  Only
compact top-k tokens, ranks, scalar scores, and provenance hashes are
written.  This is a transported representation readout of a fitted aggregate
axis: not chain-of-thought, not a discrete reasoning path, not standalone
causal evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from llm_bias.core.analysis.statistics import direction_hash
from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import (
    continuation_token_ids,
    fp32_next_token_logits,
)
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import decode_token
from llm_bias.jspace_intervention.outcome_flip import (
    OUTCOME_FLIP_ARTIFACT_TYPE,
    Combination,
    _discovery_verification_records,
    _enable_deterministic_gpu,
    _load_bound_payload,
    _materialize_records,
    _preflight_records,
    _verify_bound_inputs,
    iter_combinations,
    verify_direction_identity,
)
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig

DECODE_SCHEMA_VERSION = 1
DECODE_ARTIFACT_TYPE = "outcome_direction_decode"
SOURCE_ARTIFACT_TYPE = "outcome_direction_decode_source"
PREPARE_METADATA_TYPE = "outcome_direction_decode_prepare_metadata"
FORWARD_METADATA_TYPE = "outcome_direction_decode_metadata"
SUMMARY_ARTIFACT_TYPE = "outcome_direction_decode_summary"
ANALYSIS_METADATA_TYPE = "outcome_direction_decode_analysis_metadata"

DEFAULT_TOP_K = 30
ANTISYMMETRY_REL_TOL = 1e-4

TRANSPORT_CONTRACT = (
    "JacobianLens.transport: row-vector d_l J_l^T (column J_l d_l), float32; "
    "no dose or alpha is applied to the decoded direction"
)
UNEMBED_CONTRACT = (
    "fp32_next_token_logits: float32 final norm + LM head, the "
    "fp32_next_token_log_probs tail without log-softmax"
)
AGGREGATION_CONTRACT = (
    "band scope = equal-weight mean of the complete per-layer direction "
    "probability vectors (float64) before any top-k selection; per-layer "
    "top-k is in per-layer transported-logit order"
)
INTERPRETATION_LABEL = (
    "transported representation readout of a fitted aggregate axis; not "
    "chain-of-thought, not a discrete reasoning path, not standalone causal "
    "evidence; top tokens describe where the layer axis points in vocabulary "
    "and do not establish that any token causally drives buy/sell preference"
)


# ---------------------------------------------------------------------------
# Decode math
# ---------------------------------------------------------------------------


def resolve_answer_token_ids(
    tokenizer: Any,
    materialized_records: list[dict[str, Any]],
    config: OutcomeFlipConfig,
) -> dict[str, int]:
    """Buy/Sell answer token IDs on the discovery scoring prompts.

    Each candidate must tokenize to exactly one continuation token on every
    discovery scoring prompt, and the ID must be unique across all records;
    either violation fails closed.
    """
    if not materialized_records:
        raise ValueError("answer token resolution requires discovery records")
    resolved: dict[str, int] = {}
    for candidate in (config.positive_candidate, config.negative_candidate):
        current: int | None = None
        for record in materialized_records:
            suffix = continuation_token_ids(
                tokenizer, record["scoring_prompt"], candidate
            )[1]
            if len(suffix) != 1:
                raise ValueError(
                    f"answer candidate {candidate!r} must be a single "
                    "continuation token on every discovery scoring prompt"
                )
            token_id = int(suffix[0])
            if current is None:
                current = token_id
            elif current != token_id:
                raise ValueError(
                    f"answer candidate {candidate!r} token id is not unique "
                    "across discovery records; run is fail-closed"
                )
        resolved[candidate] = int(current)
    return resolved


@torch.no_grad()
def transported_direction_logits(
    model: Any,
    lens: Any,
    direction: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Transported direction logit ``z_l = W_U N(J_l d_l)`` in float32.

    ``direction`` is the unit-norm layer-direction vector (CPU float32 from
    the in-memory recompute).  The transport, final norm, and unembedding are
    all computed in float32; the result is ``[vocab]`` on the LM-head device.
    """
    if layer not in lens.source_layers:
        raise ValueError(
            f"layer {layer} is not a lens source layer "
            f"({lens.source_layers[0]}..{lens.source_layers[-1]})"
        )
    head_device = model._lm_head.weight.device
    transported = lens.transport(
        direction.detach().float().cpu().to(head_device).unsqueeze(0), layer
    )
    return fp32_next_token_logits(model, transported)[0]


def _top_entries(
    z: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    order: torch.Tensor,
    top_k: int,
    tokenizer: Any,
    token_cache: dict[int, str],
) -> list[dict[str, Any]]:
    k = min(int(top_k), z.numel())
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": token_cache.setdefault(
                int(token_id), decode_token(tokenizer, int(token_id))
            ),
            "logit": float(z[token_id]),
            "probability": float(probabilities[token_id]),
        }
        for rank, token_id in enumerate(order[:k].tolist(), start=1)
    ]


def _rank_of(order: torch.Tensor, token_id: int) -> int:
    return int((order == int(token_id)).nonzero().item()) + 1


def _answer_entry(
    z: torch.Tensor,
    probabilities: torch.Tensor,
    order: torch.Tensor,
    token_id: int,
) -> dict[str, Any]:
    return {
        "token_id": int(token_id),
        "rank": _rank_of(order, token_id),
        "logit": float(z[token_id]),
        "probability": float(probabilities[token_id]),
    }


@torch.no_grad()
def decode_direction_layers(
    *,
    model: Any,
    lens: Any,
    directions: Mapping[int, torch.Tensor],
    layers: list[int],
    top_k: int,
    answer_ids: Mapping[str, int],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[int, list[torch.Tensor]], float]:
    """Decode one band of directions for both signs, in memory only.

    Returns ``(records, probability_vectors, max_antisymmetry_deviation)``
    where ``records`` is one entry per (layer, sign) and
    ``probability_vectors[sign]`` is the list of complete per-layer direction
    probability vectors (float64, CPU) ordered by ``layers``.  The
    ``z(-d) == -z(d)`` antisymmetry is self-checked per layer and any
    deviation beyond ``ANTISYMMETRY_REL_TOL`` fails closed.
    """
    if not layers:
        raise ValueError("decode requires at least one layer")
    vocab_size = int(model._lm_head.weight.shape[0])
    k = min(int(top_k), vocab_size)
    positive_id = int(answer_ids["positive"])
    negative_id = int(answer_ids["negative"])
    if positive_id == negative_id:
        raise ValueError("buy and sell answer tokens must be distinct")
    token_cache: dict[int, str] = {
        positive_id: tokenizer.decode([positive_id]),
        negative_id: tokenizer.decode([negative_id]),
    }
    records: list[dict[str, Any]] = []
    probability_vectors: dict[int, list[torch.Tensor]] = {1: [], -1: []}
    max_deviation = 0.0
    for layer in layers:
        direction = directions[int(layer)].detach().float().cpu()
        z_plus = transported_direction_logits(model, lens, direction, layer).cpu()
        z_minus = transported_direction_logits(model, lens, -direction, layer).cpu()
        deviation = float((z_minus + z_plus).abs().max())
        scale = max(1.0, float(z_plus.abs().max()))
        if not torch.isfinite(torch.as_tensor(deviation)) or deviation > ANTISYMMETRY_REL_TOL * scale:
            raise ValueError(
                f"antisymmetry self-check failed at layer {layer}: "
                f"max|z(-d)+z(d)|={deviation:.3e} exceeds "
                f"{ANTISYMMETRY_REL_TOL * scale:.3e}"
            )
        max_deviation = max(max_deviation, deviation)
        for sign, z in ((1, z_plus), (-1, z_minus)):
            probabilities = z.softmax(dim=-1)
            order = torch.argsort(z, descending=True, stable=True)
            probability_vectors[sign].append(probabilities.double().cpu())
            records.append(
                {
                    "layer": int(layer),
                    "sign": sign,
                    "direction_hash": direction_hash(direction),
                    "direction_norm": float(direction.norm()),
                    "transported_norm": float(
                        (lens.transport(direction.unsqueeze(0), layer)
                        .norm()
                        .item())
                    ),
                    "top_tokens": _top_entries(
                        z,
                        probabilities,
                        order=order,
                        top_k=k,
                        tokenizer=tokenizer,
                        token_cache=token_cache,
                    ),
                    "positive": _answer_entry(z, probabilities, order, positive_id),
                    "negative": _answer_entry(z, probabilities, order, negative_id),
                    "answer_logit_margin": float(z[positive_id] - z[negative_id]),
                    "answer_probability_margin": float(
                        probabilities[positive_id] - probabilities[negative_id]
                    ),
                }
            )
    return records, probability_vectors, max_deviation


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def build_decode_summary(
    *,
    band: list[int],
    position_rule: str,
    layers: list[int],
    top_k: int,
    records: list[dict[str, Any]],
    probability_vectors: Mapping[int, list[torch.Tensor]],
    answer_ids: Mapping[str, int],
    tokenizer: Any,
) -> dict[str, Any]:
    """Band scope (average-before-top-k) plus the per-layer answer-margin table."""
    positive_id = int(answer_ids["positive"])
    negative_id = int(answer_ids["negative"])

    band_scope: dict[str, dict[str, Any]] = {}
    for sign, label in ((1, "plus_d"), (-1, "minus_d")):
        stacked = torch.stack(probability_vectors[sign], dim=0)
        if stacked.shape[0] != len(layers):
            raise ValueError("band probability vectors do not match the layer list")
        mean_p = stacked.mean(dim=0)
        order = torch.argsort(mean_p, descending=True, stable=True)
        top_k_entries = [
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": decode_token(tokenizer, int(token_id)),
                "probability": float(mean_p[token_id]),
            }
            for rank, token_id in enumerate(order[: int(top_k)].tolist(), start=1)
        ]
        positive_mean = float(mean_p[positive_id])
        negative_mean = float(mean_p[negative_id])
        band_scope[label] = {
            "top_tokens": top_k_entries,
            "positive": {
                "token_id": positive_id,
                "rank": _rank_of(order, positive_id),
                "mean_probability": positive_mean,
            },
            "negative": {
                "token_id": negative_id,
                "rank": _rank_of(order, negative_id),
                "mean_probability": negative_mean,
            },
            "probability_margin": positive_mean - negative_mean,
        }

    by_layer: dict[int, dict[int, dict[str, Any]]] = {}
    for record in records:
        by_layer.setdefault(int(record["layer"]), {})[int(record["sign"])] = record
    if sorted(by_layer) != sorted(int(layer) for layer in layers):
        raise ValueError("decode records do not cover the requested layers")
    answer_margins: list[dict[str, Any]] = []
    for layer in sorted(by_layer):
        row: dict[str, Any] = {"layer": layer}
        for sign, label in ((1, "plus_d"), (-1, "minus_d")):
            record = by_layer[layer][sign]
            row[label] = {
                "logit_margin": record["answer_logit_margin"],
                "probability_margin": record["answer_probability_margin"],
                "positive_rank": record["positive"]["rank"],
                "negative_rank": record["negative"]["rank"],
            }
        answer_margins.append(row)

    return {
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "schema_version": DECODE_SCHEMA_VERSION,
        "band": [int(band[0]), int(band[1])],
        "position_rule": position_rule,
        "layers": [int(layer) for layer in layers],
        "top_k": int(top_k),
        "band_scope": band_scope,
        "answer_margins": answer_margins,
        "aggregation_contract": AGGREGATION_CONTRACT,
        "interpretation": INTERPRETATION_LABEL,
    }


# ---------------------------------------------------------------------------
# Pipeline: prepare -> forward -> analyze -> finalize
# ---------------------------------------------------------------------------


def _decode_layers(
    identity: Mapping[str, Any],
    combination: Combination,
) -> list[int]:
    layers = list(combination.layers)
    fitted = list(identity["fitted_layers"])
    missing = [layer for layer in layers if layer not in fitted]
    if missing:
        raise ValueError(
            f"selected band layers {missing} are not in the direction "
            "identity fitted layers; run is fail-closed"
        )
    return layers


def run_outcome_decode_pipeline(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    config_path: str | Path,
    model_name: str,
    run_id: str,
    direction_identity_path: str | Path,
    calibration_selection_path: str | Path,
    dataset: str = "jspace-outcome-direction-decode",
    artifact_root: str | Path = "artifacts",
    lens_path: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_seq_len: int = 1024,
) -> Path:
    """Run the V2 direction-decode prepare/forward/analyze/finalize workflow.

    The direction is recomputed in memory (deterministic, fail-closed against
    the frozen direction identity) and decoded layer-by-layer for both signs
    over the calibration-selected band.  No direction vector, raw gradient,
    Jacobian, activation, or residual is ever persisted.
    """
    _enable_deterministic_gpu()
    input_path = Path(input_path)
    split_manifest = Path(split_manifest)
    config_path = Path(config_path)
    identity_path = Path(direction_identity_path)
    selection_path = Path(calibration_selection_path)
    for path, label in (
        (input_path, "baseline prompt CSV"),
        (split_manifest, "split manifest"),
        (config_path, "outcome flip config"),
        (identity_path, "direction identity"),
        (selection_path, "calibration selection"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if config_payload.get("artifact_type") != OUTCOME_FLIP_ARTIFACT_TYPE:
        raise ValueError(
            f"config artifact_type must be {OUTCOME_FLIP_ARTIFACT_TYPE!r}"
        )
    outcome_config = OutcomeFlipConfig.from_dict(config_payload)
    if outcome_config.model != model_name:
        raise ValueError(
            f"run model {model_name!r} does not match model "
            f"{outcome_config.model!r} frozen in config"
        )
    if outcome_config.split_manifest_sha256 != sha256_file(split_manifest):
        raise ValueError("run split manifest does not match the manifest frozen in config")

    identity = _load_bound_payload(
        identity_path,
        label="direction identity",
        expected_type="outcome_flip_direction_identity",
    )
    if identity.get("split") != "discovery":
        raise ValueError("direction identity must come from a discovery run")
    selection = _load_bound_payload(
        selection_path,
        label="calibration selection",
        expected_type="outcome_flip_selection",
    )
    if selection.get("split") != "calibration":
        raise ValueError("calibration selection must come from a calibration run")
    bound = selection.get("direction_identity_sha256")
    if bound is None or bound != sha256_file(identity_path):
        raise ValueError("selection does not bind the provided direction identity")
    _verify_bound_inputs(
        identity=identity,
        selection=selection,
        input_path=input_path,
        split_manifest=split_manifest,
        config_path=config_path,
        config=outcome_config,
        model_name=model_name,
    )

    selected = selection["selected"]
    combination = Combination(
        band_start=int(selected["band"][0]),
        band_end=int(selected["band"][1]),
        position_rule=str(selected["position_rule"]),
        relative_dose=float(selected["relative_dose"]),
    )
    if combination not in iter_combinations(outcome_config):
        raise ValueError(
            "calibration selection combination is not in the frozen "
            "candidate set"
        )
    if combination.position_rule not in identity["position_rules"]:
        raise ValueError(
            "selected position rule is not in the direction identity; "
            "run is fail-closed"
        )

    split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    assignments = {str(key): str(value) for key, value in split_payload["assignments"].items()}
    discovery_records = _discovery_verification_records(
        input_path,
        assignments=assignments,
        config=outcome_config,
        identity=identity,
    )
    preflight_tokenizer = load_tokenizer(model_name)
    _preflight_records(
        preflight_tokenizer,
        discovery_records,
        config=outcome_config,
        max_seq_len=max_seq_len,
    )
    materialized = _materialize_records(
        discovery_records, preflight_tokenizer, outcome_config
    )
    answer_ids = resolve_answer_token_ids(preflight_tokenizer, materialized, outcome_config)
    del preflight_tokenizer

    model, tokenizer, fallback_device = load_model(model_name)
    device = getattr(model, "input_device", fallback_device)
    loaded_lens = load_validated_lens(
        model=model,
        model_name=model_name,
        lens_path=lens_path,
        artifact_root=artifact_root,
    )

    decode_layers = _decode_layers(identity, combination)
    for layer in decode_layers:
        if layer not in loaded_lens.lens.source_layers:
            raise ValueError(
                f"layer {layer} is not in the canonical lens source layers; "
                "run is fail-closed"
            )
    # Keep the Jacobians resident on the unembed (lm_head) device.
    lens = loaded_lens.lens
    head_device = model._lm_head.weight.device
    lens.jacobians = {
        layer: jacobian.to(head_device) for layer, jacobian in lens.jacobians.items()
    }

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
    run.manifest.register_artifact(
        selection_path,
        artifact_type="outcome_flip_selection",
        stage="prepare",
        role="input",
    )
    run.manifest.register_artifact(
        loaded_lens.path,
        artifact_type="jacobian_lens",
        stage="prepare",
        role="lens",
        metadata={"source": loaded_lens.source},
    )
    run.manifest.save()
    try:
        prepare_dir = run.run_directory / "prepare"
        source_path = prepare_dir / "direction_source.json"
        prepare_metadata_path = prepare_dir / "metadata.json"
        with run.stage("prepare") as stage:
            source_document = {
                "artifact_type": SOURCE_ARTIFACT_TYPE,
                "schema_version": DECODE_SCHEMA_VERSION,
                "model": model_name,
                "input": str(input_path),
                "input_sha256": sha256_file(input_path),
                "split_manifest": str(split_manifest),
                "split_manifest_sha256": sha256_file(split_manifest),
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "direction_identity": str(identity_path),
                "direction_identity_sha256": sha256_file(identity_path),
                "calibration_selection": str(selection_path),
                "calibration_selection_sha256": sha256_file(selection_path),
                "lens": str(loaded_lens.path),
                "lens_sha256": sha256_file(loaded_lens.path),
                "lens_source": loaded_lens.source,
                "selected": combination.to_dict(),
                "decode_layers": decode_layers,
                "answer_tokens": {
                    candidate: {
                        "token_id": int(token_id),
                        "token": decode_token(tokenizer, int(token_id)),
                    }
                    for candidate, token_id in sorted(answer_ids.items())
                },
                "discovery_record_count": len(discovery_records),
                "discovery_ticker_count": len({record["ticker"] for record in discovery_records}),
                "direction_source": (
                    "in-memory deterministic recompute per the V2 "
                    "no-persistence protocol; verified layer-wise against "
                    "outcome_flip_direction_identity; never persisted"
                ),
                "sign_convention": "+d is Buy steering, -d is Sell steering",
                "decode_contract": {
                    "transport": TRANSPORT_CONTRACT,
                    "unembed": UNEMBED_CONTRACT,
                    "aggregation": AGGREGATION_CONTRACT,
                    "top_k": int(top_k),
                },
            }
            write_json(source_path, source_document, overwrite=False)
            write_metadata(
                prepare_metadata_path,
                {
                    "artifact_type": PREPARE_METADATA_TYPE,
                    "schema_version": DECODE_SCHEMA_VERSION,
                    "model": model_name,
                    "split": "discovery",
                    "prompt_columns": sorted(identity["prompt_columns"]),
                    "fitted_layers": list(identity["fitted_layers"]),
                    "position_rules": list(identity["position_rules"]),
                    "selected_band": [int(combination.band_start), int(combination.band_end)],
                    "selected_position_rule": combination.position_rule,
                    "selected_relative_dose": float(combination.relative_dose),
                    "top_k": int(top_k),
                    "max_seq_len": max_seq_len,
                    "record_count": len(discovery_records),
                },
                overwrite=False,
            )
            stage.count(len(discovery_records))
        run.manifest.register_artifact(
            source_path, artifact_type=SOURCE_ARTIFACT_TYPE,
            stage="prepare", role="output",
        )
        run.manifest.register_artifact(
            prepare_metadata_path, artifact_type=PREPARE_METADATA_TYPE,
            stage="prepare", role="output",
        )
        run.manifest.save()

        forward_dir = run.run_directory / "forward"
        decode_path = forward_dir / "direction_decode.jsonl"
        forward_metadata_path = forward_dir / "metadata.json"
        with run.stage("forward") as stage:
            directions, _permutation = verify_direction_identity(
                model=model,
                tokenizer=tokenizer,
                discovery_records=materialized,
                config=outcome_config,
                identity=identity,
                device=device,
            )
            rule_directions = directions[combination.position_rule]
            records, probability_vectors, max_deviation = decode_direction_layers(
                model=model,
                lens=lens,
                directions=rule_directions,
                layers=decode_layers,
                top_k=top_k,
                answer_ids={"positive": answer_ids[outcome_config.positive_candidate],
                            "negative": answer_ids[outcome_config.negative_candidate]},
                tokenizer=tokenizer,
            )
            count = write_jsonl(
                decode_path,
                (
                    {
                        "schema_version": DECODE_SCHEMA_VERSION,
                        "artifact_type": DECODE_ARTIFACT_TYPE,
                        **record,
                    }
                    for record in records
                ),
                overwrite=False,
            )
            stage.count(count)
        forward_metadata = {
            "artifact_type": FORWARD_METADATA_TYPE,
            "schema_version": DECODE_SCHEMA_VERSION,
            "model": model_name,
            "canonical_lens": str(loaded_lens.path),
            "lens_source": loaded_lens.source,
            "layers": decode_layers,
            "signs": [1, -1],
            "top_k": int(top_k),
            "deterministic_mode": {
                "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cublas_workspace_config": ":4096:8",
            },
            "direction_recompute": {
                "identity": str(identity_path),
                "identity_sha256": sha256_file(identity_path),
                "position_rule": combination.position_rule,
                "record_count": len(discovery_records),
                "verified_layers": len(decode_layers),
                "verified_position_rules": list(identity["position_rules"]),
            },
            "transport_contract": TRANSPORT_CONTRACT,
            "unembed_contract": UNEMBED_CONTRACT,
            "self_check": {
                "check": "z(-d) == -z(d) per layer",
                "max_abs_deviation": max_deviation,
                "relative_tolerance": ANTISYMMETRY_REL_TOL,
                "passed": True,
            },
            "backpropagation": True,
            "records_written": count,
        }
        write_metadata(forward_metadata_path, forward_metadata, overwrite=False)
        run.manifest.register_artifact(
            decode_path, artifact_type=DECODE_ARTIFACT_TYPE,
            stage="forward", role="output", record_count=count,
        )
        run.manifest.register_artifact(
            forward_metadata_path, artifact_type=FORWARD_METADATA_TYPE,
            stage="forward", role="output",
        )
        run.manifest.save()

        analyze_dir = run.run_directory / "analyze"
        summary_path = analyze_dir / "direction_decode_summary.json"
        analyze_metadata_path = analyze_dir / "metadata.json"
        with run.stage("analyze") as stage:
            summary = build_decode_summary(
                band=[int(combination.band_start), int(combination.band_end)],
                position_rule=combination.position_rule,
                layers=decode_layers,
                top_k=top_k,
                records=records,
                probability_vectors=probability_vectors,
                answer_ids={"positive": answer_ids[outcome_config.positive_candidate],
                            "negative": answer_ids[outcome_config.negative_candidate]},
                tokenizer=tokenizer,
            )
            write_json(summary_path, summary, overwrite=False)
            write_metadata(
                analyze_metadata_path,
                {
                    "artifact_type": ANALYSIS_METADATA_TYPE,
                    "schema_version": DECODE_SCHEMA_VERSION,
                    "model": model_name,
                    "aggregation_contract": AGGREGATION_CONTRACT,
                    "band": [int(combination.band_start), int(combination.band_end)],
                    "position_rule": combination.position_rule,
                    "interpretation": INTERPRETATION_LABEL,
                    "record_count": count,
                },
                overwrite=False,
            )
            stage.count(count)
        run.manifest.register_artifact(
            summary_path, artifact_type=SUMMARY_ARTIFACT_TYPE,
            stage="analyze", role="output",
        )
        run.manifest.register_artifact(
            analyze_metadata_path, artifact_type=ANALYSIS_METADATA_TYPE,
            stage="analyze", role="output",
        )
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = [
    "AGGREGATION_CONTRACT",
    "ANTISYMMETRY_REL_TOL",
    "DECODE_SCHEMA_VERSION",
    "DEFAULT_TOP_K",
    "INTERPRETATION_LABEL",
    "build_decode_summary",
    "decode_direction_layers",
    "resolve_answer_token_ids",
    "run_outcome_decode_pipeline",
    "transported_direction_logits",
]
