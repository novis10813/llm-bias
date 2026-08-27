"""Artifact-producing execution pipeline for J-space interventions."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.jspace_intervention.analysis import analyze_token_screen
from llm_bias.jspace_intervention.prompting import prepare_scoring_prompt
from llm_bias.jspace_intervention.runner import (
    layer_prototypes,
    run_concept_gain_record,
    run_swap_record,
    run_token_screen_record,
    token_screen_directions,
)
from llm_bias.jspace_intervention.schemas import (
    GainConfig,
    InterventionConfig,
    TokenScreenConfig,
)

TOKEN_SCREEN_DEFAULT_PROMPT_COLUMN = "prompt_with_context_attribute_0"


def _iter_prompt_records(
    input_path: Path,
    *,
    assignments: dict[str, str],
    split_name: str,
    source_sector: str,
    prompt_columns: set[str] | None,
) -> Iterator[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ticker = row.get("ticker") or ""
            if assignments.get(ticker) != split_name or row.get("sector") != source_sector:
                continue
            for column, prompt in row.items():
                if not column.startswith("prompt_with_context") or not prompt:
                    continue
                if prompt_columns is not None and column not in prompt_columns:
                    continue
                yield {
                    "ticker": ticker,
                    "name": row.get("name") or "",
                    "sector": row.get("sector") or "",
                    "marketcap": row.get("marketcap") or "",
                    "prompt_column": column,
                    "prompt": prompt,
                }


def _preflight_records(
    tokenizer: Any,
    records: Iterator[dict[str, str]],
    *,
    config: InterventionConfig | GainConfig,
    max_records: int | None,
    max_seq_len: int,
) -> int:
    """Validate every selected prompt before creating a persistent run."""
    count = 0
    for record in records:
        if max_records is not None and count >= max_records:
            break
        scoring_prompt, _ = prepare_scoring_prompt(
            tokenizer, record["prompt"], decision_prefix=config.decision_prefix
        )
        lengths = [
            len(input_ids(tokenizer, scoring_prompt + candidate, add_special_tokens=True))
            for candidate in (config.positive_candidate, config.negative_candidate)
        ]
        if max(lengths) > max_seq_len:
            raise ValueError(
                f"formatted scoring prompt plus candidate has {max(lengths)} tokens, "
                f"limit {max_seq_len}"
            )
        count += 1
    if count == 0:
        raise ValueError("no prompt records match the requested sector and split")
    return count


def run_swap_pipeline(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    config_path: str | Path,
    model_name: str,
    run_id: str,
    dataset: str = "jspace-sector-intervention",
    artifact_root: str | Path = "artifacts",
    lens_path: str | Path | None = None,
    split_name: str = "test",
    max_records: int | None = None,
    max_seq_len: int = 1024,
    prompt_columns: set[str] | None = None,
    _operation: str = "swap",
) -> Path:
    """Run one directional J-space intervention into a canonical run tree."""
    if _operation not in {"swap", "gain"}:
        raise ValueError(f"unsupported intervention operation: {_operation}")
    input_path = Path(input_path)
    split_manifest = Path(split_manifest)
    config_path = Path(config_path)
    split_payload = json.loads(split_manifest.read_text())
    assignments = {str(k): str(v) for k, v in split_payload["assignments"].items()}
    config_payload = json.loads(config_path.read_text())
    expected_split_sha256 = config_payload.get("split_manifest_sha256")
    actual_split_sha256 = sha256_file(split_manifest)
    if expected_split_sha256 != actual_split_sha256:
        raise ValueError(
            "run split manifest does not match the discovery manifest frozen in config"
        )
    config = (
        InterventionConfig.from_dict(config_payload)
        if _operation == "swap"
        else GainConfig.from_dict(config_payload)
    )
    source_sector = (
        config.source.sector
        if isinstance(config, InterventionConfig)
        else config.prototype.sector
    )
    preflight_tokenizer = load_tokenizer(model_name)
    _preflight_records(
        preflight_tokenizer,
        _iter_prompt_records(
            input_path,
            assignments=assignments,
            split_name=split_name,
            source_sector=source_sector,
            prompt_columns=prompt_columns,
        ),
        config=config,
        max_records=max_records,
        max_seq_len=max_seq_len,
    )
    del preflight_tokenizer

    run = ArtifactRun.create(
        model_name, dataset, run_id, artifact_root=artifact_root
    )
    run.manifest.register_artifact(
        input_path, artifact_type="trial_plan_prompts", stage="prepare", role="input"
    )
    run.manifest.register_artifact(
        split_manifest, artifact_type="jspace_intervention_splits",
        stage="prepare", role="input",
    )
    run.manifest.register_artifact(
        config_path,
        artifact_type=(
            "jspace_intervention_config" if _operation == "swap" else "jspace_gain_config"
        ),
        stage="prepare",
        role="input",
    )
    run.manifest.save()

    output_path = run.run_directory / "forward" / "intervention_results.jsonl"
    metadata_path = output_path.with_suffix(".jsonl.metadata.json")
    try:
        model, tokenizer, fallback_device = load_model(model_name)
        loaded_lens = load_validated_lens(
            model=model,
            model_name=model_name,
            lens_path=lens_path,
            artifact_root=artifact_root,
        )
        device = getattr(model, "input_device", fallback_device)
        if isinstance(config, InterventionConfig):
            source_prototypes = layer_prototypes(
                model, loaded_lens.lens, config.source, config.layers
            )
            target_prototypes = layer_prototypes(
                model, loaded_lens.lens, config.target, config.layers
            )
            gain_prototypes = None
        else:
            source_prototypes = None
            target_prototypes = None
            gain_prototypes = layer_prototypes(
                model, loaded_lens.lens, config.prototype, config.layers
            )
        run.manifest.register_artifact(
            loaded_lens.path,
            artifact_type="jacobian_lens",
            stage="prepare",
            role="lens",
            metadata={"source": loaded_lens.source},
        )
        run.manifest.save()

        def rows() -> Iterator[dict[str, Any]]:
            seen = 0
            records = _iter_prompt_records(
                input_path,
                assignments=assignments,
                split_name=split_name,
                source_sector=source_sector,
                prompt_columns=prompt_columns,
            )
            for record in records:
                if max_records is not None and seen >= max_records:
                    break
                scoring_prompt, evidence_span = prepare_scoring_prompt(
                    tokenizer,
                    record["prompt"],
                    decision_prefix=config.decision_prefix,
                )
                record_id = stable_record_id(
                    record["ticker"], record["prompt_column"], split_name
                )
                if isinstance(config, InterventionConfig):
                    results = run_swap_record(
                        model=model,
                        tokenizer=tokenizer,
                        lens=loaded_lens.lens,
                        scoring_prompt=scoring_prompt,
                        evidence_span=evidence_span,
                        config=config,
                        device=device,
                        source_prototypes=source_prototypes,
                        target_prototypes=target_prototypes,
                        control_seed=int(record_id.rsplit("_", 1)[-1], 16),
                    )
                else:
                    results = run_concept_gain_record(
                        model=model,
                        tokenizer=tokenizer,
                        lens=loaded_lens.lens,
                        scoring_prompt=scoring_prompt,
                        evidence_span=evidence_span,
                        config=config,
                        device=device,
                        prototypes=gain_prototypes,
                        control_seed=int(record_id.rsplit("_", 1)[-1], 16),
                    )
                for result in results:
                    yield {
                        "schema_version": 3,
                        "artifact_type": "jspace_intervention_result",
                        "record_id": record_id,
                        "ticker": record["ticker"],
                        "sector": record["sector"],
                        "prompt_column": record["prompt_column"],
                        "split": split_name,
                        **result,
                    }
                seen += 1

        with run.stage("forward") as stage:
            count = write_jsonl(output_path, rows(), overwrite=False)
            stage.count(count)
        write_metadata(
            metadata_path,
            {
                "artifact_type": "jspace_intervention_metadata",
                "schema_version": 3,
                "model": model_name,
                "input": str(input_path),
                "split_manifest": str(split_manifest),
                "config": str(config_path),
                "split": split_name,
                "canonical_lens": str(loaded_lens.path),
                "record_count": count,
            },
            overwrite=False,
        )
        run.manifest.register_artifact(
            output_path,
            artifact_type="jspace_intervention_result",
            stage="forward",
            role="output",
            record_count=count,
        )
        run.manifest.register_artifact(
            metadata_path,
            artifact_type="jspace_intervention_metadata",
            stage="forward",
            role="output",
        )
        run.manifest.save()
        run.finalize(required_stages={"forward"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def _token_screen_prompt_records(
    input_path: Path,
    *,
    assignments: dict[str, str],
    split_name: str,
    source_sector: str,
    prompt_columns: set[str],
    max_records: int | None,
) -> list[dict[str, str]]:
    records = []
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


def _token_screen_preflight(
    tokenizer: Any,
    records: list[dict[str, str]],
    *,
    config: TokenScreenConfig,
    max_seq_len: int,
) -> None:
    """Validate every selected prompt before creating a persistent run."""
    if not records:
        raise ValueError("no prompt records match the requested sector and split")
    for record in records:
        scoring_prompt, _ = prepare_scoring_prompt(
            tokenizer, record["prompt"], decision_prefix=config.decision_prefix
        )
        lengths = [
            len(input_ids(tokenizer, scoring_prompt + candidate, add_special_tokens=True))
            for candidate in (config.positive_candidate, config.negative_candidate)
        ]
        if max(lengths) > max_seq_len:
            raise ValueError(
                f"formatted scoring prompt plus candidate has {max(lengths)} tokens, "
                f"limit {max_seq_len}"
            )


def run_token_screen_pipeline(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    config_path: str | Path,
    model_name: str,
    run_id: str,
    dataset: str = "jspace-token-screen",
    artifact_root: str | Path = "artifacts",
    lens_path: str | Path | None = None,
    split_name: str = "discovery",
    max_records: int | None = None,
    max_seq_len: int = 1024,
    prompt_columns: set[str] | None = None,
) -> Path:
    """Run the frozen token causal screen across the selected baseline prompts.

    Consumes a completed valence ``frozen_candidate_suggestions`` artifact via
    the SHA-bound ``TokenScreenConfig`` and applies the minimal first screen:
    clean-pass evidence positions, per-layer median coordinate scale, symmetric
    ``(-a, 0, +a)`` additive token steering, and a same-norm matched-random arm.
    """
    input_path = Path(input_path)
    split_manifest = Path(split_manifest)
    config_path = Path(config_path)
    for path, label in (
        (input_path, "baseline prompt CSV"),
        (split_manifest, "split manifest"),
        (config_path, "token screen config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    split_payload = json.loads(split_manifest.read_text())
    assignments = {str(k): str(v) for k, v in split_payload["assignments"].items()}
    config_payload = json.loads(config_path.read_text())
    if config_payload.get("split_manifest_sha256") != sha256_file(split_manifest):
        raise ValueError(
            "run split manifest does not match the discovery manifest frozen in config"
        )
    config = TokenScreenConfig.from_dict(config_payload)
    if config.model != model_name:
        raise ValueError(
            f"run model {model_name!r} does not match model {config.model!r} frozen in config"
        )
    candidate_artifact = Path(config.candidate_artifact_path)
    if not candidate_artifact.is_file():
        raise FileNotFoundError(f"frozen candidate artifact not found: {candidate_artifact}")
    if sha256_file(candidate_artifact) != config.candidate_artifact_sha256:
        raise ValueError("frozen candidate artifact does not match the SHA bound in config")

    resolved_prompt_columns = prompt_columns or {TOKEN_SCREEN_DEFAULT_PROMPT_COLUMN}
    preflight_tokenizer = load_tokenizer(model_name)
    records = _token_screen_prompt_records(
        input_path,
        assignments=assignments,
        split_name=split_name,
        source_sector=config.source_sector,
        prompt_columns=resolved_prompt_columns,
        max_records=max_records,
    )
    _token_screen_preflight(
        preflight_tokenizer, records, config=config, max_seq_len=max_seq_len
    )
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(
        input_path, artifact_type="trial_plan_prompts", stage="prepare", role="input"
    )
    run.manifest.register_artifact(
        split_manifest,
        artifact_type="jspace_intervention_splits", stage="prepare", role="input",
    )
    run.manifest.register_artifact(
        config_path, artifact_type="jspace_token_screen_config",
        stage="prepare", role="input",
    )
    run.manifest.register_artifact(
        candidate_artifact, artifact_type="frozen_candidate_suggestions",
        stage="prepare", role="input",
    )
    run.manifest.save()
    try:
        prepare_dir = run.run_directory / "prepare"
        prompt_records_path = prepare_dir / "prompt_records.jsonl"
        prepare_metadata_path = prepare_dir / "metadata.json"
        with run.stage("prepare") as stage:
            count = write_jsonl(
                prompt_records_path,
                (
                    {
                        "schema_version": 1,
                        "artifact_type": "token_screen_prompt_record",
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
            write_metadata(
                prepare_metadata_path,
                {
                    "artifact_type": "token_screen_prepare_metadata",
                    "schema_version": 1,
                    "model": model_name,
                    "input": str(input_path),
                    "input_sha256": sha256_file(input_path),
                    "split_manifest": str(split_manifest),
                    "split_manifest_sha256": sha256_file(split_manifest),
                    "candidate_artifact": str(candidate_artifact),
                    "candidate_artifact_sha256": config.candidate_artifact_sha256,
                    "config": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "source_sector": config.source_sector,
                    "split": split_name,
                    "prompt_columns": sorted(resolved_prompt_columns),
                    "candidate_count": len(config.candidates),
                    "layers": list(config.layers),
                    "alphas": list(config.alphas),
                    "controls": list(config.controls),
                    "top_positions": config.top_positions,
                    "loading_threshold": config.loading_threshold,
                    "positive_dose": config.positive_dose,
                    "outcome_scoring": config.outcome_scoring,
                    "record_count": count,
                },
                overwrite=False,
            )
            stage.count(count)
        run.manifest.register_artifact(
            prompt_records_path, artifact_type="token_screen_prompt_record",
            stage="prepare", role="output", record_count=count,
        )
        run.manifest.register_artifact(
            prepare_metadata_path, artifact_type="token_screen_prepare_metadata",
            stage="prepare", role="output",
        )
        run.manifest.save()

        model, tokenizer, fallback_device = load_model(model_name)
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
        directions = token_screen_directions(
            model, loaded_lens.lens, config.candidates, config.layers
        )
        device = getattr(model, "input_device", fallback_device)

        forward_dir = run.run_directory / "forward"
        results_path = forward_dir / "token_screen_results.jsonl"
        forward_metadata_path = forward_dir / "metadata.json"
        with run.stage("forward") as stage:
            rows: list[dict[str, Any]] = []
            for record in records:
                scoring_prompt, evidence_span = prepare_scoring_prompt(
                    tokenizer,
                    record["prompt"],
                    decision_prefix=config.decision_prefix,
                )
                record_rows = run_token_screen_record(
                    model=model,
                    tokenizer=tokenizer,
                    scoring_prompt=scoring_prompt,
                    evidence_span=evidence_span,
                    config=config,
                    device=device,
                    directions=directions,
                    control_seed=int(record["record_id"].rsplit("_", 1)[-1], 16),
                )
                rows.extend(
                    {
                        "schema_version": 1,
                        "artifact_type": "token_screen_result",
                        "record_id": record["record_id"],
                        "ticker": record["ticker"],
                        "sector": record["sector"],
                        "prompt_column": record["prompt_column"],
                        "split": split_name,
                        **result,
                    }
                    for result in record_rows
                )
            result_count = write_jsonl(results_path, rows, overwrite=False)
            stage.count(result_count)
        write_metadata(
            forward_metadata_path,
            {
                "artifact_type": "token_screen_metadata",
                "schema_version": 1,
                "model": model_name,
                "canonical_lens": str(loaded_lens.path),
                "lens_source": loaded_lens.source,
                "layers": list(config.layers),
                "alphas": list(config.alphas),
                "controls": list(config.controls),
                "top_positions": config.top_positions,
                "loading_threshold": config.loading_threshold,
                "positive_dose": config.positive_dose,
                "estimand": "symmetric_slope",
                "outcome_scoring": config.outcome_scoring,
                "backpropagation": False,
                "record_count": result_count,
            },
            overwrite=False,
        )
        run.manifest.register_artifact(
            results_path, artifact_type="token_screen_result",
            stage="forward", role="output", record_count=result_count,
        )
        run.manifest.register_artifact(
            forward_metadata_path, artifact_type="token_screen_metadata",
            stage="forward", role="output",
        )
        run.manifest.save()

        analyze_dir = run.run_directory / "analyze"
        summary_path = analyze_dir / "token_screen_summary.json"
        analyze_metadata_path = analyze_dir / "metadata.json"
        with run.stage("analyze") as stage:
            summary = analyze_token_screen(rows, positive_dose=config.positive_dose)
            write_json(
                summary_path,
                {
                    "artifact_type": "token_screen_analysis",
                    "schema_version": 1,
                    "forward": "forward/token_screen_results.jsonl",
                    "forward_sha256": sha256_file(results_path),
                    **summary,
                },
                overwrite=False,
            )
            write_metadata(
                analyze_metadata_path,
                {
                    "artifact_type": "token_screen_analysis_metadata",
                    "schema_version": 1,
                    "model": model_name,
                    "interpretation": "exploratory_discovery",
                    "candidate_count": len(config.candidates),
                    "shortlist_counts": {
                        key: len(value) for key, value in summary["shortlist"].items()
                    },
                },
                overwrite=False,
            )
            stage.count(len(config.candidates))
        run.manifest.register_artifact(
            summary_path, artifact_type="token_screen_analysis",
            stage="analyze", role="output",
        )
        run.manifest.register_artifact(
            analyze_metadata_path, artifact_type="token_screen_analysis_metadata",
            stage="analyze", role="output",
        )
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def run_gain_pipeline(**kwargs: Any) -> Path:
    """Run a coordinate-gain experiment using the shared artifact lifecycle."""
    return run_swap_pipeline(**kwargs, _operation="gain")


__all__ = [
    "TOKEN_SCREEN_DEFAULT_PROMPT_COLUMN",
    "run_gain_pipeline",
    "run_swap_pipeline",
    "run_token_screen_pipeline",
]
