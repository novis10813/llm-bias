"""Artifact workflow for single-sector header-span sensitivity."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import continuation_token_ids, score_margin
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids
from llm_bias.span_sensitivity.analysis import grouped_effects
from llm_bias.span_sensitivity.conditions import Identity, build_conditions

DECISION_PREFIX = '{\n  "decision": "'
POSITIVE_CANDIDATE = "buy"
NEGATIVE_CANDIDATE = "sell"
_VALID_SPLITS = {"discovery", "calibration", "test"}


def _load_split_manifest(path: Path, input_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("split manifest must be a schema-version-1 object")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("split manifest must contain non-empty assignments")
    expected_input_hash = payload.get("input_sha256")
    actual_input_hash = sha256_file(input_path)
    if expected_input_hash != actual_input_hash:
        raise ValueError("split manifest is not bound to the requested input CSV")
    normalized = {str(key): str(value) for key, value in assignments.items()}
    if any(value not in _VALID_SPLITS for value in normalized.values()):
        raise ValueError("split manifest contains an unknown split")
    return normalized, payload


def _selected_sources(
    input_path: Path,
    *,
    assignments: dict[str, str],
    sector: str,
    split: str,
    prompt_columns: set[str] | None,
    max_records: int | None,
) -> tuple[list[dict[str, str]], dict[str, Identity]]:
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    identities: dict[str, Identity] = {}
    selected_rows: list[dict[str, str]] = []
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "name", "sector"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("input CSV must contain ticker, name, and sector")
        available_prompts = {name for name in reader.fieldnames if name.startswith("prompt_with_context")}
        if prompt_columns is not None:
            missing = prompt_columns - available_prompts
            if missing:
                raise ValueError(f"unknown prompt columns: {sorted(missing)}")
        for row in reader:
            ticker = (row.get("ticker") or "").strip()
            if row.get("sector") != sector or assignments.get(ticker) != split:
                continue
            identity = Identity(ticker, (row.get("name") or "").strip())
            if not identity.ticker or not identity.name:
                raise ValueError("selected rows must contain ticker and name")
            previous = identities.setdefault(ticker, identity)
            if previous != identity:
                raise ValueError(f"ticker {ticker!r} has conflicting identities")
            for column in reader.fieldnames:
                prompt = row.get(column) or ""
                if not column.startswith("prompt_with_context") or not prompt:
                    continue
                if prompt_columns is not None and column not in prompt_columns:
                    continue
                if max_records is not None and len(selected_rows) >= max_records:
                    continue
                selected_rows.append(
                    {
                        "ticker": ticker,
                        "name": identity.name,
                        "sector": sector,
                        "marketcap": row.get("marketcap") or "",
                        "prompt_column": column,
                        "prompt": prompt,
                    }
                )
    if len(identities) < 2:
        raise ValueError("selected sector/split needs at least two distinct tickers")
    if not selected_rows:
        raise ValueError("no prompt records match the requested sector and split")
    return selected_rows, identities


def _peer_map(identities: dict[str, Identity], *, seed: int) -> dict[str, Identity]:
    ordered = sorted(
        identities,
        key=lambda ticker: stable_record_id(str(seed), ticker),
    )
    return {
        ticker: identities[ordered[(index + 1) % len(ordered)]]
        for index, ticker in enumerate(ordered)
    }


def _identity_token_count(tokenizer: Any, identity: Identity) -> int:
    return len(
        input_ids(
            tokenizer,
            f"{identity.ticker}\n{identity.name}",
            add_special_tokens=False,
        )
    )


def _prepare_rows(
    tokenizer: Any,
    sources: Iterable[dict[str, str]],
    peers: dict[str, Identity],
    *,
    split: str,
    seed: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    rows: list[dict[str, Any]] = []
    for source_row in sources:
        source = Identity(source_row["ticker"], source_row["name"])
        prompt_id = stable_record_id(source.ticker, source_row["prompt_column"], split)
        conditions = build_conditions(
            source_row["prompt"],
            source=source,
            peer=peers[source.ticker],
            key=prompt_id,
            seed=seed,
        )
        source_token_count = _identity_token_count(tokenizer, source)
        for condition in conditions:
            formatted = format_prompt(
                tokenizer,
                condition.prompt,
                use_chat_template=True,
                enable_thinking=False,
            )
            if condition.prompt not in formatted:
                raise ValueError("formatted chat prompt does not contain the raw user prompt")
            scoring_prompt = formatted + DECISION_PREFIX
            candidate_lengths = []
            for candidate in (POSITIVE_CANDIDATE, NEGATIVE_CANDIDATE):
                continuation_token_ids(tokenizer, scoring_prompt, candidate)
                candidate_lengths.append(
                    len(input_ids(tokenizer, scoring_prompt + candidate, add_special_tokens=True))
                )
            if max(candidate_lengths) > max_seq_len:
                raise ValueError(
                    f"{prompt_id}/{condition.condition} has {max(candidate_lengths)} tokens; "
                    f"limit {max_seq_len}"
                )
            replacement_token_count = _identity_token_count(tokenizer, condition.replacement)
            rows.append(
                {
                    "schema_version": 1,
                    "artifact_type": "span_sensitivity_prepared_prompt",
                    "prompt_id": prompt_id,
                    "condition_id": stable_record_id(prompt_id, condition.condition),
                    "ticker": source.ticker,
                    "name": source.name,
                    "sector": source_row["sector"],
                    "marketcap": source_row["marketcap"],
                    "prompt_column": source_row["prompt_column"],
                    "split": split,
                    "condition": condition.condition,
                    "mutation_scope": "header_identity_fields_only",
                    "mutation_fields": list(condition.mutation_fields),
                    "replacement_ticker": condition.replacement.ticker,
                    "replacement_name": condition.replacement.name,
                    "source_ticker_mentions_outside_header": condition.ticker_mentions_outside_header,
                    "source_name_mentions_outside_header": condition.name_mentions_outside_header,
                    "source_identity_token_count": source_token_count,
                    "replacement_identity_token_count": replacement_token_count,
                    "identity_token_count_delta": replacement_token_count - source_token_count,
                    "formatted_candidate_max_token_count": max(candidate_lengths),
                    "prompt": condition.prompt,
                }
            )
    return rows


def _score_rows(
    model: Any,
    tokenizer: Any,
    prepared: list[dict[str, Any]],
    *,
    device: Any,
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in prepared:
        by_prompt.setdefault(row["prompt_id"], []).append(row)
    output = []
    for prompt_id, variants in by_prompt.items():
        scores: dict[str, Any] = {}
        for row in variants:
            formatted = format_prompt(
                tokenizer, row["prompt"], use_chat_template=True, enable_thinking=False
            )
            scores[row["condition"]] = score_margin(
                model,
                tokenizer,
                formatted + DECISION_PREFIX,
                POSITIVE_CANDIDATE,
                NEGATIVE_CANDIDATE,
                device=device,
            )
        if "original" not in scores:
            raise ValueError(f"prompt {prompt_id} has no original condition")
        original_value = scores["original"].value
        for row in variants:
            margin = scores[row["condition"]]
            payload = margin.to_dict()
            output.append(
                {
                    "schema_version": 1,
                    "artifact_type": "span_sensitivity_result",
                    "prompt_id": prompt_id,
                    "condition_id": row["condition_id"],
                    "ticker": row["ticker"],
                    "sector": row["sector"],
                    "prompt_column": row["prompt_column"],
                    "split": row["split"],
                    "condition": row["condition"],
                    "mutation_scope": row["mutation_scope"],
                    "mutation_fields": row["mutation_fields"],
                    "replacement_ticker": row["replacement_ticker"],
                    "replacement_name": row["replacement_name"],
                    "source_ticker_mentions_outside_header": row["source_ticker_mentions_outside_header"],
                    "source_name_mentions_outside_header": row["source_name_mentions_outside_header"],
                    "source_identity_token_count": row["source_identity_token_count"],
                    "replacement_identity_token_count": row["replacement_identity_token_count"],
                    "identity_token_count_delta": row["identity_token_count_delta"],
                    **payload,
                    "original_margin": original_value,
                    "delta_margin": margin.value - original_value,
                    "two_candidate_buy_probability": float(torch.sigmoid(torch.tensor(margin.value)).item()),
                }
            )
    return output


def run_pipeline(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    model_name: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    dataset: str = "technology-header-span-sensitivity",
    sector: str = "Technology",
    split: str = "discovery",
    prompt_columns: set[str] | None = None,
    max_records: int | None = None,
    max_seq_len: int = 1024,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    device_map: str | None = None,
) -> Path:
    """Run preflight, scoring, ticker-clustered analysis, and finalization."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    input_path = Path(input_path)
    split_manifest = Path(split_manifest)
    assignments, split_payload = _load_split_manifest(split_manifest, input_path)
    sources, identities = _selected_sources(
        input_path,
        assignments=assignments,
        sector=sector,
        split=split,
        prompt_columns=prompt_columns,
        max_records=max_records,
    )
    peers = _peer_map(identities, seed=seed)
    preflight_tokenizer = load_tokenizer(model_name)
    prepared = _prepare_rows(
        preflight_tokenizer,
        sources,
        peers,
        split=split,
        seed=seed,
        max_seq_len=max_seq_len,
    )
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    prepared_path = run.run_directory / "prepare" / "prepared_prompts.jsonl"
    prepare_metadata_path = run.run_directory / "prepare" / "metadata.json"
    forward_path = run.run_directory / "forward" / "margin_results.jsonl"
    forward_metadata_path = run.run_directory / "forward" / "metadata.json"
    analysis_path = run.run_directory / "analysis" / "summary.json"
    try:
        run.manifest.register_artifact(
            input_path, artifact_type="trial_plan_prompts", stage="prepare", role="input"
        )
        run.manifest.register_artifact(
            split_manifest,
            artifact_type="span_sensitivity_split_manifest",
            stage="prepare",
            role="input",
        )
        with run.stage("prepare") as stage:
            prepared_count = write_jsonl(prepared_path, prepared, overwrite=False)
            write_metadata(
                prepare_metadata_path,
                {
                    "schema_version": 1,
                    "artifact_type": "span_sensitivity_prepare_metadata",
                    "input": str(input_path),
                    "input_sha256": sha256_file(input_path),
                    "split_manifest": str(split_manifest),
                    "split_manifest_sha256": sha256_file(split_manifest),
                    "split_manifest_seed": split_payload.get("seed"),
                    "sector": sector,
                    "split": split,
                    "seed": seed,
                    "mutation_scope": "header_identity_fields_only",
                    "source_prompt_count": len(sources),
                    "prepared_record_count": prepared_count,
                    "condition_count": 7,
                    "claim_boundary": "Evidence-body identity mentions are unchanged; results measure marginal sensitivity to the two header identity fields, not a full entity-only substitution.",
                },
                overwrite=False,
            )
            stage.count(prepared_count)
        for path, kind, count in (
            (prepared_path, "span_sensitivity_prepared_prompt", prepared_count),
            (prepare_metadata_path, "span_sensitivity_prepare_metadata", None),
        ):
            run.manifest.register_artifact(
                path, artifact_type=kind, stage="prepare", role="output", record_count=count
            )
        run.manifest.save()

        model, tokenizer, fallback_device = load_model(model_name, device_map=device_map)
        device = getattr(model, "input_device", fallback_device)
        with run.stage("forward") as stage:
            results = _score_rows(model, tokenizer, prepared, device=device)
            result_count = write_jsonl(forward_path, results, overwrite=False)
            write_metadata(
                forward_metadata_path,
                {
                    "schema_version": 1,
                    "artifact_type": "span_sensitivity_forward_metadata",
                    "model": model_name,
                    "positive_candidate": POSITIVE_CANDIDATE,
                    "negative_candidate": NEGATIVE_CANDIDATE,
                    "decision_prefix": DECISION_PREFIX,
                    "margin_definition": "logP(buy)-logP(sell)",
                    "record_count": result_count,
                    "prepared_artifact": str(prepared_path),
                    "prepared_sha256": sha256_file(prepared_path),
                },
                overwrite=False,
            )
            stage.count(result_count)
        for path, kind, count in (
            (forward_path, "span_sensitivity_result", result_count),
            (forward_metadata_path, "span_sensitivity_forward_metadata", None),
        ):
            run.manifest.register_artifact(
                path, artifact_type=kind, stage="forward", role="output", record_count=count
            )
        run.manifest.save()

        with run.stage("analyze") as stage:
            summaries = grouped_effects(
                read_jsonl(forward_path),
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
            write_json(
                analysis_path,
                {
                    "schema_version": 1,
                    "artifact_type": "span_sensitivity_analysis",
                    "source": str(forward_path),
                    "source_sha256": sha256_file(forward_path),
                    "sector": sector,
                    "split": split,
                    "mutation_scope": "header_identity_fields_only",
                    "groups": summaries,
                    "interpretation": "Paired fixed-continuation margin effects with ticker-level inference; not lens evidence or a standalone causal mechanism claim.",
                },
                overwrite=False,
            )
            stage.count(len(summaries))
        run.manifest.register_artifact(
            analysis_path,
            artifact_type="span_sensitivity_analysis",
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
    "DECISION_PREFIX",
    "NEGATIVE_CANDIDATE",
    "POSITIVE_CANDIDATE",
    "run_pipeline",
]
