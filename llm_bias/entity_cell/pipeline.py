"""Compact E1 workflow stages using the shared artifact lifecycle."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids
from .analysis import (
    collision_selectivity_summary,
    held_variant_metrics,
    summarize_amnesia,
    surface_control_summary,
)
from .mlp_cells import (
    CANDIDATE_LAYERS,
    collect_generic_stats,
    rank_stability_scores,
    run_amnesia_curve,
    select_matched_random_neuron,
    surface_form_controls,
)
from .preparation import HELD_VARIANT_IDS, LOCALIZATION_VARIANT_IDS, parse_header
from .lifecycle import check_provenance, validate_prepared_directory

DATASET = "entity-cell-localization"
STAGES = ("e1-baseline", "e1-localization", "e1-amnesia", "analyze")


def _group(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    result: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[key])].append(row)
    return dict(result)


def _stats_metadata(stats: Mapping[int, Any]) -> dict[str, Any]:
    # Means and standard deviations are needed to resume E1 localization, but
    # the individual baseline vectors never leave the online accumulator.
    return {str(layer): value.compact(include_moments=True) for layer, value in sorted(stats.items())}


def _prepared_ids(tokenizer: Any, row: Mapping[str, Any]) -> list[int]:
    values = row.get("input_ids")
    if isinstance(values, list) and values:
        return [int(value) for value in values]
    formatted = format_prompt(tokenizer, str(row["prompt"]), use_chat_template=True, enable_thinking=False)
    return input_ids(tokenizer, formatted, add_special_tokens=True)


def run_e1(
    *,
    prepared_dir: str | Path,
    model_name: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    stages: Iterable[str] = STAGES,
    max_tickers: int | None = None,
    device: Any | None = None,
) -> Path:
    """Run E1 from T1 JSONL inputs; max_tickers enables a one-ticker smoke run."""
    enabled = tuple(dict.fromkeys(stages))
    unknown = set(enabled) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown E1 stages: {sorted(unknown)}")
    prepared = Path(prepared_dir)
    prepared_metadata = validate_prepared_directory(prepared)
    check_provenance(prepared_metadata, model=model_name)
    headers = read_jsonl(prepared / "prepare" / "header_variants.jsonl")
    financial = read_jsonl(prepared / "prepare" / "financial_prompts.jsonl")
    baseline = read_jsonl(prepared / "prepare" / "generic_baseline.jsonl")
    if max_tickers is not None and max_tickers < 1:
        raise ValueError("max_tickers must be positive")
    tickers = sorted({str(row["ticker"]) for row in headers})
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    headers = [row for row in headers if row["ticker"] in tickers]
    financial = [row for row in financial if row["ticker"] in tickers]
    if not headers or not financial or not baseline:
        raise ValueError("prepared E1 inputs are incomplete")
    run = ArtifactRun.create(model_name, DATASET, run_id, artifact_root=artifact_root)
    output_dir = run.run_directory / "e1"
    try:
        metadata_path = prepared / "prepare" / "metadata.json"
        if metadata_path.is_file():
            run.manifest.register_artifact(metadata_path, artifact_type="entity_cell_prepare_metadata", stage="prepare", role="input")
        for path in (prepared / "prepare" / "header_variants.jsonl", prepared / "prepare" / "financial_prompts.jsonl", prepared / "prepare" / "generic_baseline.jsonl", prepared / "prepare" / "e2_donor_contracts.jsonl"):
            run.manifest.register_artifact(path, artifact_type="entity_cell_prepared_input", stage="prepare", role="input")
        from llm_bias.core.model import load_model
        model, tokenizer, fallback_device = load_model(model_name)
        run.manifest.register_artifact(
            prepared / "metadata.json",
            artifact_type="entity_cell_prepare_metadata",
            stage="prepare",
            role="input",
        )
        target_device = device or getattr(model, "input_device", fallback_device)
        stats = None
        if "e1-baseline" in enabled:
            with run.stage("e1-baseline") as stage:
                stats = collect_generic_stats(model, tokenizer, (row["prompt"] for row in baseline), device=target_device)
                path = output_dir / "baseline_stats.json"
                write_json(path, {"schema_version": 1, "artifact_type": "entity_cell_baseline_stats", "layers": _stats_metadata(stats), "baseline_count": len(baseline), "raw_runtime_payloads": False}, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_baseline_stats", stage="e1-baseline", role="output")
                stage.count(len(stats))
        if "e1-localization" in enabled:
            if stats is None:
                stats_path = output_dir / "baseline_stats.json"
                if not stats_path.is_file():
                    raise ValueError("e1-localization requires completed e1-baseline statistics")
                from .mlp_cells import OnlineVectorStats
                saved = json.loads(stats_path.read_text(encoding="utf-8"))
                stats = {int(layer): OnlineVectorStats.from_compact(value) for layer, value in saved["layers"].items()}
            with run.stage("e1-localization") as stage:
                by_ticker = _group(headers, "ticker")
                records: list[dict[str, Any]] = []
                for ticker, rows in by_ticker.items():
                    local = [row for row in rows if int(row["variant_number"]) in LOCALIZATION_VARIANT_IDS]
                    held = [row for row in rows if int(row["variant_number"]) in HELD_VARIANT_IDS]
                    local_vectors: dict[int, list[Any]] = {layer: [] for layer in CANDIDATE_LAYERS}
                    held_vectors: dict[int, list[Any]] = {layer: [] for layer in CANDIDATE_LAYERS}
                    for row in local:
                        ids = __import__("torch").tensor([_prepared_ids(tokenizer, row)], device=target_device)
                        from .mlp_cells import record_post_swiglu
                        values = record_post_swiglu(model, ids, layers=CANDIDATE_LAYERS, position=int(row["final_company_name_content_token"]))
                        for layer in CANDIDATE_LAYERS: local_vectors[layer].append(values[layer])
                    local_candidates = rank_stability_scores(local_vectors, stats)
                    for row in held:
                        ids = __import__("torch").tensor([_prepared_ids(tokenizer, row)], device=target_device)
                        from .mlp_cells import record_post_swiglu
                        values = record_post_swiglu(model, ids, layers=CANDIDATE_LAYERS, position=int(row["final_company_name_content_token"]))
                        for layer in CANDIDATE_LAYERS: held_vectors[layer].append(values[layer])
                    held_candidates = rank_stability_scores(held_vectors, stats)
                    controls: dict[str, list[dict[str, Any]]] = {}
                    for control_name in ("anonymous_ticker", "anonymous_name", "name_form_control"):
                        control_vectors = {layer: [] for layer in CANDIDATE_LAYERS}
                        for row in local:
                            ticker_value, name_value = parse_header(str(row["prompt"]))[:2]
                            control_prompt = surface_form_controls(str(row["prompt"]), ticker_value, name_value)[control_name]
                            ids = __import__("torch").tensor([_prepared_ids(tokenizer, {**row, "prompt": control_prompt})], device=target_device)
                            from .mlp_cells import record_post_swiglu
                            values = record_post_swiglu(model, ids, layers=CANDIDATE_LAYERS, position=int(row["final_company_name_content_token"]))
                            for layer in CANDIDATE_LAYERS:
                                control_vectors[layer].append(values[layer])
                        controls[control_name] = rank_stability_scores(control_vectors, stats)
                    records.append({
                        "ticker": ticker,
                        "localization_candidates": local_candidates,
                        "held_candidates": held_candidates,
                        "held_variant_metrics": held_variant_metrics(local_candidates, held_candidates),
                        "surface_control_summary": surface_control_summary(local_candidates, controls),
                    })
                path = output_dir / "cells.jsonl"
                count = write_jsonl(path, records, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_candidates", stage="e1-localization", role="output", record_count=count)
                stage.count(count)
        if "e1-amnesia" in enabled:
            if stats is None:
                stats_path = output_dir / "baseline_stats.json"
                if not stats_path.is_file():
                    raise ValueError("e1-amnesia requires completed e1-baseline statistics")
                from .mlp_cells import OnlineVectorStats
                saved = json.loads(stats_path.read_text(encoding="utf-8"))
                stats = {int(layer): OnlineVectorStats.from_compact(value) for layer, value in saved["layers"].items()}
            cells = read_jsonl(output_dir / "cells.jsonl")
            with run.stage("e1-amnesia") as stage:
                rows_out: list[dict[str, Any]] = []
                financial_by_ticker = _group(financial, "ticker")
                selected = {row["ticker"]: row["localization_candidates"][0] for row in cells}
                for ticker, candidate in selected.items():
                    wrong = next((value for key, value in selected.items() if key != ticker and int(value["layer"]) == int(candidate["layer"])), candidate)
                    random_neuron = select_matched_random_neuron(stats, layer=int(candidate["layer"]), target_neuron=int(candidate["neuron"]))
                    rows_out.extend(run_amnesia_curve(model, tokenizer, prompt_rows=financial_by_ticker[ticker], candidate=candidate, wrong_candidate=wrong, random_neuron=random_neuron, device=target_device))
                path = output_dir / "amnesia.jsonl"
                count = write_jsonl(path, rows_out, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_amnesia", stage="e1-amnesia", role="output", record_count=count)
                stage.count(count)
        if "analyze" in enabled:
            with run.stage("analyze") as stage:
                cells = read_jsonl(output_dir / "cells.jsonl") if (output_dir / "cells.jsonl").is_file() else []
                amnesia = read_jsonl(output_dir / "amnesia.jsonl") if (output_dir / "amnesia.jsonl").is_file() else []
                if not cells or not amnesia:
                    raise ValueError("analyze requires completed E1 localization and amnesia artifacts")
                candidates = {row["ticker"]: row["localization_candidates"] for row in cells}
                scores = {
                    row["ticker"]: {
                        (int(candidate["layer"]), int(candidate["neuron"])): float(candidate.get("score", 0.0))
                        for candidate in row["localization_candidates"]
                    }
                    for row in cells
                }
                summary = {"schema_version": 1, "artifact_type": "entity_cell_analysis", "candidate_label": "trusted candidate entity cell", "collision_selectivity": collision_selectivity_summary(candidates, scores_by_ticker=scores), "amnesia": summarize_amnesia(amnesia), "raw_runtime_payloads": False}
                path = run.run_directory / "analyze" / "summary.json"
                write_json(path, summary, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_analysis", stage="analyze", role="output")
                stage.count(len(summary["amnesia"]))
        run.finalize(required_stages=set(enabled))
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def analyze_e1(run_root: str | Path) -> Path:
    """Materialize the compact E1 summary from a completed run's outputs."""
    root = Path(run_root)
    cells = read_jsonl(root / "e1" / "cells.jsonl")
    amnesia = read_jsonl(root / "e1" / "amnesia.jsonl")
    candidates = {row["ticker"]: row["localization_candidates"] for row in cells}
    scores = {
        row["ticker"]: {
            (int(candidate["layer"]), int(candidate["neuron"])): float(candidate.get("score", 0.0))
            for candidate in row["localization_candidates"]
        }
        for row in cells
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "entity_cell_analysis",
        "candidate_label": "trusted candidate entity cell",
        "collision_selectivity": collision_selectivity_summary(candidates, scores_by_ticker=scores),
        "amnesia": summarize_amnesia(amnesia),
        "raw_runtime_payloads": False,
    }
    path = root / "analyze" / "summary.json"
    write_json(path, summary, overwrite=True)
    return path


__all__ = ["DATASET", "STAGES", "analyze_e1", "run_e1"]
