"""Compact E1 workflow stages using the shared artifact lifecycle."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids
from .analysis import (
    collision_selectivity_summary,
    frame_surface_control_summary,
    held_variant_metrics,
    select_wrong_entity_cell,
    summarize_amnesia,
    surface_control_summary,
    v2_candidate_eligibility,
)
from .mlp_cells import (
    CANDIDATE_LAYERS,
    collect_generic_stats,
    rank_absolute_activations,
    rank_stability_scores,
    record_post_swiglu,
    run_amnesia_curve,
    select_matched_random_neuron,
    surface_form_controls,
)
from .preparation import (
    FRAME_HELD_VARIANT_IDS,
    FRAME_LOCALIZATION_VARIANT_IDS,
    HELD_VARIANT_IDS,
    LOCALIZATION_FAMILY_V2,
    LOCALIZATION_VARIANT_IDS,
    parse_header,
    _positions_for_range,
    _token_offsets,
)
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


def _frame_control_vectors(
    tokenizer: Any, model: Any, rows: Sequence[Mapping[str, Any]], replacement: str, target_device: Any
) -> dict[int, torch.Tensor]:
    """Run re-tokenized frame control prompts with the name span replaced.

    Each control prompt is tokenized from scratch (never reusing the original
    ``input_ids``); the readout position is the last content token of the
    replacement name under the same contained-span rule as preparation.
    """
    vectors: dict[int, list[torch.Tensor]] = {layer: [] for layer in CANDIDATE_LAYERS}
    for row in rows:
        raw_prompt = str(row["prompt"])
        name = str(row["name"])
        start = raw_prompt.find(name)
        if raw_prompt.count(name) != 1 or start < 0:
            raise ValueError("frame control requires the company name exactly once")
        control_prompt = raw_prompt[:start] + replacement + raw_prompt[start + len(name):]
        formatted = format_prompt(tokenizer, control_prompt, use_chat_template=True, enable_thinking=False)
        raw_offset = formatted.find(control_prompt)
        absolute_start = raw_offset + start
        absolute_end = absolute_start + len(replacement)
        ids, offsets, specials = _token_offsets(tokenizer, formatted)
        positions = _positions_for_range(
            offsets, specials, absolute_start, absolute_end, limit=len(ids), contained=True
        )
        if not positions:
            raise ValueError("frame control replacement does not map to a non-empty token span")
        values = record_post_swiglu(
            model, torch.tensor([ids], device=target_device),
            layers=CANDIDATE_LAYERS, position=int(max(positions)),
        )
        for layer in CANDIDATE_LAYERS:
            vectors[layer].append(values[layer])
    return vectors


def _localize_frames_v2(
    *,
    model: Any,
    tokenizer: Any,
    stats: Mapping[int, Any],
    frames_by_ticker: Mapping[str, list[Mapping[str, Any]]],
    headers_by_ticker: Mapping[str, list[Mapping[str, Any]]],
    template_row: Mapping[str, Any],
    output_dir: Path,
    target_device: Any,
) -> list[dict[str, Any]]:
    """E1 V2 surface-varying frame localization plus the template-only signature."""
    import codecs

    def _row_vectors(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Any]]:
        vectors: dict[int, list[Any]] = {layer: [] for layer in CANDIDATE_LAYERS}
        for row in rows:
            ids = torch.tensor([_prepared_ids(tokenizer, row)], device=target_device)
            values = record_post_swiglu(model, ids, layers=CANDIDATE_LAYERS, position=int(row["final_company_name_content_token"]))
            for layer in CANDIDATE_LAYERS:
                vectors[layer].append(values[layer])
        return vectors

    records: list[dict[str, Any]] = []
    for ticker in sorted(frames_by_ticker):
        rows = frames_by_ticker[ticker]
        local = [row for row in rows if int(row["variant_number"]) in FRAME_LOCALIZATION_VARIANT_IDS]
        held = [row for row in rows if int(row["variant_number"]) in FRAME_HELD_VARIANT_IDS]
        local_candidates = rank_stability_scores(_row_vectors(local), stats)
        held_candidates = rank_stability_scores(_row_vectors(held), stats)
        name = str(rows[0]["name"])
        controls = {
            "anonymous_name_frames": rank_stability_scores(_frame_control_vectors(tokenizer, model, local, "Anonymous Company", target_device), stats),
            "name_form_control_frames": rank_stability_scores(_frame_control_vectors(tokenizer, model, local, codecs.decode(name, "rot_13"), target_device), stats),
        }
        header_candidates = rank_stability_scores(_row_vectors(headers_by_ticker[ticker]), stats)
        frame_top5 = {(int(row["layer"]), int(row["neuron"])) for row in local_candidates[:5]}
        header_top5 = {(int(row["layer"]), int(row["neuron"])) for row in header_candidates[:5]}
        records.append(
            {
                "ticker": ticker,
                "localization_candidates": local_candidates,
                "held_candidates": held_candidates,
                "held_variant_metrics": held_variant_metrics(local_candidates, held_candidates),
                "surface_control_summary": frame_surface_control_summary(local_candidates, controls),
                "header_family_candidates": header_candidates,
                "header_family_overlap": len(frame_top5.intersection(header_top5)),
            }
        )
    template_ids = torch.tensor([_prepared_ids(tokenizer, template_row)], device=target_device)
    template_values = record_post_swiglu(model, template_ids, layers=CANDIDATE_LAYERS, position=int(template_row["final_company_name_content_token"]))
    signature_path = output_dir / "template_signature.json"
    write_json(
        signature_path,
        {
            "schema_version": 1,
            "artifact_type": "entity_cell_template_signature",
            "template_control_sha256": template_row.get("prompt_sha256"),
            "cells": rank_absolute_activations(template_values, stats),
            "raw_runtime_payloads": False,
        },
        overwrite=True,
    )
    return records


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
    family = str(prepared_metadata.get("localization_family", "v1-header"))
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
    frames_by_ticker: dict[str, list[Mapping[str, Any]]] = {}
    template_row: Mapping[str, Any] | None = None
    if family == LOCALIZATION_FAMILY_V2:
        frames = read_jsonl(prepared / "prepare" / "frame_variants.jsonl")
        frames_by_ticker = _group([row for row in frames if row["ticker"] in tickers], "ticker")
        template_row = json.loads((prepared / "prepare" / "template_control.json").read_text(encoding="utf-8"))
        if set(frames_by_ticker) != set(tickers):
            raise ValueError("frame variants must cover exactly the selected tickers")
        if not isinstance(template_row, Mapping) or template_row.get("artifact_type") != "entity_cell_template_control":
            raise ValueError("prepared template control is missing or invalid")
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
        if family == LOCALIZATION_FAMILY_V2:
            run.manifest.register_artifact(prepared / "prepare" / "frame_variants.jsonl", artifact_type="entity_cell_frame_variant", stage="prepare", role="input")
            run.manifest.register_artifact(prepared / "prepare" / "template_control.json", artifact_type="entity_cell_template_control", stage="prepare", role="input")
        from llm_bias.core.model import load_model
        model, tokenizer, fallback_device = load_model(model_name)
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
                if family == LOCALIZATION_FAMILY_V2:
                    records = _localize_frames_v2(
                        model=model, tokenizer=tokenizer, stats=stats,
                        frames_by_ticker=frames_by_ticker, headers_by_ticker=_group(headers, "ticker"),
                        template_row=template_row, output_dir=output_dir, target_device=target_device,
                    )
                    run.manifest.register_artifact(output_dir / "template_signature.json", artifact_type="entity_cell_template_signature", stage="e1-localization", role="output")
                else:
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
                if family == LOCALIZATION_FAMILY_V2:
                    candidates_by_ticker = {row["ticker"]: row["localization_candidates"] for row in cells}
                    for ticker, candidate in selected.items():
                        wrong, degenerate = select_wrong_entity_cell(ticker, sorted(selected), candidates_by_ticker, candidate)
                        random_neuron = select_matched_random_neuron(stats, layer=int(candidate["layer"]), target_neuron=int(candidate["neuron"]))
                        rows_out.extend(run_amnesia_curve(model, tokenizer, prompt_rows=financial_by_ticker[ticker], candidate=candidate, wrong_candidate=wrong if not degenerate else candidate, random_neuron=random_neuron, device=target_device, wrong_degenerate=degenerate))
                else:
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
                amnesia_summary = summarize_amnesia(amnesia)
                summary = {"schema_version": 1, "artifact_type": "entity_cell_analysis", "candidate_label": "trusted candidate entity cell", "collision_selectivity": collision_selectivity_summary(candidates, scores_by_ticker=scores), "amnesia": amnesia_summary, "raw_runtime_payloads": False}
                if family == LOCALIZATION_FAMILY_V2:
                    signature_path = output_dir / "template_signature.json"
                    if not signature_path.is_file():
                        raise ValueError("v2-frames analyze requires the template signature")
                    signature = json.loads(signature_path.read_text(encoding="utf-8"))["cells"]
                    eligibility = {
                        row["ticker"]: v2_candidate_eligibility(
                            held_metrics=row["held_variant_metrics"],
                            surface_control_summary=row["surface_control_summary"],
                            candidate_cells=row["localization_candidates"],
                            template_signature=signature,
                            amnesia_summary=amnesia_summary[f"{row['ticker']}:all_positions"],
                        )
                        for row in cells
                    }
                    summary["localization_family"] = LOCALIZATION_FAMILY_V2
                    summary["v2_candidate_eligibility"] = eligibility
                    summary["trusted_ticker_count"] = sum(1 for entry in eligibility.values() if entry["eligible"])
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
    amnesia_summary = summarize_amnesia(amnesia)
    summary = {
        "schema_version": 1,
        "artifact_type": "entity_cell_analysis",
        "candidate_label": "trusted candidate entity cell",
        "collision_selectivity": collision_selectivity_summary(candidates, scores_by_ticker=scores),
        "amnesia": amnesia_summary,
        "raw_runtime_payloads": False,
    }
    signature_path = root / "e1" / "template_signature.json"
    if signature_path.is_file():
        signature = json.loads(signature_path.read_text(encoding="utf-8"))["cells"]
        eligibility = {
            row["ticker"]: v2_candidate_eligibility(
                held_metrics=row["held_variant_metrics"],
                surface_control_summary=row["surface_control_summary"],
                candidate_cells=row["localization_candidates"],
                template_signature=signature,
                amnesia_summary=amnesia_summary[f"{row['ticker']}:all_positions"],
            )
            for row in cells
        }
        summary["localization_family"] = LOCALIZATION_FAMILY_V2
        summary["v2_candidate_eligibility"] = eligibility
        summary["trusted_ticker_count"] = sum(1 for entry in eligibility.values() if entry["eligible"])
    path = root / "analyze" / "summary.json"
    write_json(path, summary, overwrite=True)
    return path


__all__ = ["DATASET", "STAGES", "analyze_e1", "run_e1"]
