"""Artifact-producing execution pipeline for J-space interventions."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.jspace_intervention.prompting import prepare_scoring_prompt
from llm_bias.jspace_intervention.runner import layer_prototypes, run_swap_record
from llm_bias.jspace_intervention.schemas import InterventionConfig


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
    config: InterventionConfig,
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
) -> Path:
    """Run one directional sector-prototype swap into a canonical run tree."""
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
    config = InterventionConfig.from_dict(config_payload)
    preflight_tokenizer = load_tokenizer(model_name)
    _preflight_records(
        preflight_tokenizer,
        _iter_prompt_records(
            input_path,
            assignments=assignments,
            split_name=split_name,
            source_sector=config.source.sector,
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
        config_path, artifact_type="jspace_intervention_config",
        stage="prepare", role="input",
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
        source_prototypes = layer_prototypes(
            model, loaded_lens.lens, config.source, config.layers
        )
        target_prototypes = layer_prototypes(
            model, loaded_lens.lens, config.target, config.layers
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
                source_sector=config.source.sector,
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
                )
                for result in results:
                    yield {
                        "schema_version": 1,
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
                "schema_version": 1,
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


__all__ = ["run_swap_pipeline"]
