"""E3 discovery lifecycle for upstream and downstream suppression."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import score_single_token_margin_fp32
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.prompt_input.encoding import format_prompt

from .analysis import denominator_eligibility
from .attention_attribution import (
    SOURCE_GROUPS,
    _attention_module,
    capture_attention_forward,
    direct_logit_attribution,
    frozen_margin_direction,
    rank_attention_heads,
    reconstruct_captured_attention,
    remove_hooks,
    resolve_single_token_pair,
    validate_attention_layers,
)
from .mlp_cells import OnlineVectorStats, mlp_hooks, select_matched_random_neuron
from .suppression import (
    E3_ALPHA_GRID,
    E3_BETA_GRID,
    DECISION_PREFIX,
    NEGATIVE_CANDIDATE,
    POSITIVE_CANDIDATE,
    e3_compact_record,
    preservation_metrics,
    attenuate_identity_sources,
    validate_dose_grid,
)

DATASET = "entity-cell-localization"
E3_STAGES = ("e3-upstream", "e3-downstream", "analyze")
E3_ANALYZE_STAGE = "analyze"


def _layers(model: Any) -> Any:
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("E3 model must expose decoder layers")
    return layers


def _forward(model: Any, ids: torch.Tensor) -> Any:
    try:
        return model.forward(ids)
    except TypeError:
        return model(ids)


def _ids(row: Mapping[str, Any], device: Any) -> torch.Tensor:
    values = row.get("input_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("prepared E3 row has no input_ids")
    return torch.tensor([values], dtype=torch.long, device=device)


def _margin(model: Any, tokenizer: Any, prompt: str, *, device: Any = None, score_fn: Callable[[str], float] | None = None) -> float:
    if score_fn is not None:
        value = float(score_fn(prompt))
    else:
        formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False) + DECISION_PREFIX
        value = float(score_single_token_margin_fp32(model, tokenizer, formatted, POSITIVE_CANDIDATE, NEGATIVE_CANDIDATE, device=device).value)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("E3 margin is non-finite")
    return value


def _frozen_direction(model: Any, tokenizer: Any, row: Mapping[str, Any], *, device: Any) -> torch.Tensor:
    ids = _ids(row, device)
    final_layer = int(getattr(model, "n_layers", len(_layers(model))) - 1)
    residual = record_residuals(model, ids, [final_layer])[final_layer][:, -1, :][0]
    prompt = format_prompt(tokenizer, str(row["prompt"]), use_chat_template=True, enable_thinking=False) + DECISION_PREFIX
    positive, negative = resolve_single_token_pair(tokenizer, prompt)
    return frozen_margin_direction(residual, model._final_norm, model._lm_head, positive, negative)


def _capture_dla(model: Any, tokenizer: Any, row: Mapping[str, Any], *, selected_heads: Sequence[tuple[int, int]], device: Any) -> dict[str, dict[str, float]]:
    """Recompute compact selected-head source contributions for one prompt."""
    ids = _ids(row, device)
    result: dict[str, dict[str, float]] = {}
    grouped: dict[int, list[int]] = {}
    for layer, head in selected_heads:
        grouped.setdefault(int(layer), []).append(int(head))
    for layer, heads in grouped.items():
        attention = _attention_module(_layers(model)[layer])
        capture, handles = capture_attention_forward(attention)
        try:
            with torch.no_grad():
                _forward(model, ids)
            reconstruction = reconstruct_captured_attention(_layers(model)[layer], capture, row["source_groups"], query_position=int(row["final_query_position"]))
            if not reconstruction.additive:
                raise ValueError(f"attention additivity failed at layer {layer}")
            final_layer = int(getattr(model, "n_layers", len(_layers(model))) - 1)
            residual = record_residuals(model, ids, [final_layer])[final_layer][:, -1, :][0]
            positive, negative = resolve_single_token_pair(tokenizer, format_prompt(tokenizer, row["prompt"], use_chat_template=True, enable_thinking=False) + DECISION_PREFIX)
            direction = frozen_margin_direction(residual, model._final_norm, model._lm_head, positive, negative)
            dla = direct_logit_attribution(reconstruction, direction)
            for head in heads:
                if int(head) not in dla:
                    raise ValueError(f"selected head {head} is outside layer L{layer}")
                values = dla[head]
                result[f"L{layer}H{head}"] = {group: float(values[group]["frozen_scale_margin"]) for group in values}
        finally:
            remove_hooks(handles)
    return result


def _flip(margin: float, clean_margin: float) -> bool:
    return (margin >= 0 > clean_margin) or (margin < 0 <= clean_margin)


def run_upstream_suppression_record(
    model: Any,
    tokenizer: Any,
    *,
    prompt_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    wrong_candidate: Mapping[str, Any] | None = None,
    random_neuron: int | None = None,
    selected_heads: Sequence[tuple[int, int]] = (),
    alpha_grid: Sequence[float] = E3_ALPHA_GRID,
    scopes: Sequence[str] = ("all_positions", "header_only"),
    device: Any = None,
    score_fn: Callable[[str], float] | None = None,
) -> list[dict[str, Any]]:
    """Run E3-A target and cell controls without retaining runtime tensors."""
    alphas = validate_dose_grid(alpha_grid, name="alpha_grid")
    ticker = str(prompt_row["ticker"])
    clean = _margin(model, tokenizer, str(prompt_row["prompt"]), device=device, score_fn=score_fn)
    anonymous_prompt = str(prompt_row["prompt"]).replace(f"Stock Ticker: [{ticker}]", "Stock Ticker: [ANON]").replace(f"Stock Name: [{prompt_row['name']}]", "Stock Name: [Anonymous Company]")
    anonymous = _margin(model, tokenizer, anonymous_prompt, device=device, score_fn=score_fn)
    eligible, exclusion = denominator_eligibility(clean, anonymous)
    groups = prompt_row.get("source_groups", {}).get("identity_header", {})
    positions = tuple(position for start, end in groups.get("ranges", ()) for position in range(int(start), int(end)))
    clean_contributions = _capture_dla(model, tokenizer, prompt_row, selected_heads=selected_heads, device=device) if selected_heads else {}
    controls = [("target", candidate)]
    if wrong_candidate is not None:
        controls.append(("wrong_entity", wrong_candidate))
    if random_neuron is not None:
        controls.append(("matched_random", {"layer": candidate["layer"], "neuron": random_neuron}))
    output: list[dict[str, Any]] = []
    for scope in scopes:
        if scope not in {"all_positions", "header_only"}:
            raise ValueError("E3-A scope must be all_positions or header_only")
        selected_positions = None if scope == "all_positions" else positions
        for alpha in alphas:
            for label, cell in controls:
                layer, neuron = int(cell["layer"]), int(cell["neuron"])
                with mlp_hooks(model, [layer], channel_scales={layer: {neuron: alpha}}, scope=scope, scope_positions=selected_positions):
                    margin = _margin(model, tokenizer, str(prompt_row["prompt"]), device=device, score_fn=score_fn)
                    contributions = _capture_dla(model, tokenizer, prompt_row, selected_heads=selected_heads, device=device) if selected_heads else {}
                mediation_deltas = {
                    head: {group: float(value) - float(clean_contributions.get(head, {}).get(group, 0.0)) for group, value in values.items()}
                    for head, values in contributions.items()
                }
                output.append(e3_compact_record(
                    ticker=ticker, prompt_id=str(prompt_row["prompt_id"]), phase="e3-a", scope=scope,
                    dose=alpha, margin=margin, clean_margin=clean, anonymous_margin=anonymous,
                    flip=_flip(margin, clean), contributions=contributions,
                    controls={"candidate": label, "eligible": eligible, "exclusion_reason": exclusion, "cell": {"layer": layer, "neuron": neuron}, "mediation_deltas": mediation_deltas},
                    provenance={"anonymous_prompt_id": f"{prompt_row['prompt_id']}:anonymous_identity", "selected_heads": [list(item) for item in selected_heads]},
                ))
    return output


def run_downstream_suppression_record(
    model: Any,
    tokenizer: Any,
    *,
    prompt_row: Mapping[str, Any],
    selected_heads: Sequence[tuple[int, int]],
    beta_grid: Sequence[float] = E3_BETA_GRID,
    grouping: str = "single",
    device: Any = None,
    score_fn: Callable[[str], float] | None = None,
) -> list[dict[str, Any]]:
    """Run E3-B identity, norm-matched source, and whole-head controls."""
    betas = validate_dose_grid(beta_grid, name="beta_grid", lower=0.0, upper=1.0)
    if not selected_heads:
        raise ValueError("E3-B requires selected full-attention heads")
    clean_prompt = str(prompt_row["prompt"])
    clean = _margin(model, tokenizer, clean_prompt, device=device, score_fn=score_fn)
    anonymous_prompt = clean_prompt.replace(f"Stock Ticker: [{prompt_row['ticker']}]", "Stock Ticker: [ANON]").replace(f"Stock Name: [{prompt_row['name']}]", "Stock Name: [Anonymous Company]")
    anonymous = _margin(model, tokenizer, anonymous_prompt, device=device, score_fn=score_fn)
    eligible, exclusion = denominator_eligibility(clean, anonymous)
    clean_contributions = _capture_dla(model, tokenizer, prompt_row, selected_heads=selected_heads, device=device)
    output: list[dict[str, Any]] = []
    for beta in betas:
        for mode in ("identity", "evidence", "random_subset", "whole_head"):
            if mode == "random_subset":
                # The live hook derives this direction from the same-token-count
                # subset; no vector enters the artifact.
                pass
            direction = None if score_fn is not None else _frozen_direction(model, tokenizer, prompt_row, device=device)
            with attenuate_identity_sources(
                model,
                {int(layer): prompt_row["source_groups"] for layer, _ in selected_heads},
                selected_heads,
                beta=beta,
                mode=mode,
                grouping=grouping,
                query_position=int(prompt_row["final_query_position"]),
                margin_direction=direction,
            ) as session:
                margin = _margin(model, tokenizer, clean_prompt, device=device, score_fn=score_fn)
            contributions = {head: dict(values) for head, values in clean_contributions.items()}
            for (layer, head), deltas in session.contribution_deltas.items():
                key = f"L{layer}H{head}"
                if key not in contributions:
                    continue
                for group, delta in deltas.items():
                    contributions[key][group] = contributions[key].get(group, 0.0) + delta
            clean_groups = {group: sum(values.get(group, 0.0) for values in clean_contributions.values()) for group in SOURCE_GROUPS}
            intervened_groups = {group: sum(values.get(group, 0.0) for values in contributions.values()) for group in SOURCE_GROUPS} if contributions else {}
            output.append(e3_compact_record(
                ticker=str(prompt_row["ticker"]), prompt_id=str(prompt_row["prompt_id"]), phase="e3-b", scope=mode,
                dose=beta, margin=margin, clean_margin=clean, anonymous_margin=anonymous, flip=_flip(margin, clean), contributions=contributions,
                controls={"mode": mode, "grouping": grouping, "eligible": eligible, "source_control_eligible": all(bool(item.get("eligible", False)) for item in session.controls.values()) if mode in {"evidence", "random_subset"} else True, "exclusion_reason": exclusion, "session": session.controls, "head_margin_deltas": {f"L{layer}H{head}": value for (layer, head), value in session.deltas.items()}, "source_margin_deltas": {f"L{layer}H{head}": values for (layer, head), values in session.contribution_deltas.items()}, "whole_head_upper_bound": mode == "whole_head", "preservation": preservation_metrics(clean_groups, intervened_groups)},
                provenance={"selected_heads": [list(item) for item in selected_heads], "source_groups": "prepared_ranges"},
            ))
    return output


def _trusted_cells(cells: Sequence[Mapping[str, Any]], amnesia: Mapping[str, Any] | None = None) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in cells:
        ticker = str(row["ticker"])
        held = row.get("held_variant_metrics", {})
        summary = (amnesia or {}).get(f"{ticker}:all_positions", {})
        candidates = row.get("localization_candidates", ())
        if (
            isinstance(held, Mapping)
            and int(held.get("top5_overlap", 0)) > 0
            and isinstance(summary, Mapping)
            and summary.get("trusted_candidate_entity_cell") is True
            and isinstance(candidates, Sequence)
            and candidates
            and isinstance(candidates[0], Mapping)
        ):
            result[ticker] = candidates[0]
    return result


def run_e3(
    *,
    prepared_dir: str | Path,
    model_name: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    stages: Iterable[str] = E3_STAGES,
    e1_run_root: str | Path | None = None,
    e2_run_root: str | Path | None = None,
    selected_heads: Iterable[tuple[int, int]] | None = None,
    max_tickers: int | None = None,
    alpha_grid: Sequence[float] = E3_ALPHA_GRID,
    beta_grid: Sequence[float] = E3_BETA_GRID,
    grouping: str = "single",
    device: Any = None,
) -> Path:
    enabled = tuple(dict.fromkeys(stages))
    if set(enabled) - set(E3_STAGES):
        raise ValueError(f"unknown E3 stages: {sorted(set(enabled) - set(E3_STAGES))}")
    prepared = Path(prepared_dir)
    financial = read_jsonl(prepared / "financial_prompts.jsonl")
    if e1_run_root is None:
        cells = []
    else:
        cells_path = Path(e1_run_root) / "e1" / "cells.jsonl"
        if not cells_path.is_file():
            raise ValueError("e1_run_root must contain e1/cells.jsonl")
        cells = read_jsonl(cells_path)
    amnesia_summary = {}
    if e1_run_root and (Path(e1_run_root) / "analyze" / "summary.json").is_file():
        amnesia_summary = json.loads((Path(e1_run_root) / "analyze" / "summary.json").read_text(encoding="utf-8")).get("amnesia", {})
    trusted = _trusted_cells(cells, amnesia_summary)
    if not e1_run_root:
        raise ValueError("E3 requires a completed E1 run root")
    if not trusted:
        raise ValueError("E3 found no trusted E1 candidate cells")
    if e2_run_root:
        attribution_path = Path(e2_run_root) / "e2" / "head_attribution.jsonl"
        if not attribution_path.is_file():
            raise ValueError("e2_run_root must contain e2/head_attribution.jsonl")
        attribution = read_jsonl(attribution_path)
    else:
        attribution = []
    if "e3-downstream" in enabled and not e2_run_root and not selected_heads:
        raise ValueError("E3-B requires an E2 attribution run or explicit selected heads")
    heads = tuple((int(layer), int(head)) for layer, head in (selected_heads or ((row["layer"], row["head"]) for row in rank_attention_heads(attribution) if row.get("selection_eligible"))))
    if heads:
        validate_attention_layers(layer for layer, _ in heads)
    if max_tickers is not None:
        if max_tickers < 1:
            raise ValueError("max_tickers must be positive")
        tickers = sorted(trusted)[:max_tickers]
        financial = [row for row in financial if str(row["ticker"]) in tickers]
    run = ArtifactRun.create(model_name, DATASET, run_id, artifact_root=artifact_root)
    out = run.run_directory / "e3"
    try:
        run.manifest.register_artifact(prepared / "metadata.json", artifact_type="entity_cell_prepare_metadata", stage="prepare", role="input")
        run.manifest.register_artifact(prepared / "financial_prompts.jsonl", artifact_type="entity_cell_financial_prompt", stage="prepare", role="input")
        e1_cells_path = Path(e1_run_root) / "e1" / "cells.jsonl"
        e1_summary_path = Path(e1_run_root) / "analyze" / "summary.json"
        run.manifest.register_artifact(e1_cells_path, artifact_type="entity_cell_candidates", stage="e1", role="input")
        if e1_summary_path.is_file():
            run.manifest.register_artifact(e1_summary_path, artifact_type="entity_cell_analysis", stage="e1", role="input")
        e2_attribution_path = Path(e2_run_root) / "e2" / "head_attribution.jsonl" if e2_run_root else None
        if e2_attribution_path is not None:
            run.manifest.register_artifact(e2_attribution_path, artifact_type="entity_cell_e2_attribution", stage="e2", role="input")
        from llm_bias.core.model import load_model
        model, tokenizer, fallback = load_model(model_name)
        target = torch.device(device or getattr(model, "input_device", fallback))
        if "e3-upstream" in enabled:
            with run.stage("e3-upstream") as stage:
                rows = []
                by_ticker = defaultdict_rows(financial)
                for ticker, prompt_rows in by_ticker.items():
                    if ticker not in trusted:
                        continue
                    candidate = trusted[ticker]
                    wrong = next((cell for name, cell in trusted.items() if name != ticker and int(cell["layer"]) == int(candidate["layer"])), None)
                    random = int(candidate["neuron"]) + 1
                    stats_path = Path(e1_run_root) / "e1" / "baseline_stats.json" if e1_run_root else None
                    if stats_path is not None and stats_path.is_file():
                        saved = json.loads(stats_path.read_text(encoding="utf-8"))
                        stats = {int(layer): OnlineVectorStats.from_compact(value) for layer, value in saved["layers"].items()}
                        random = select_matched_random_neuron(stats, layer=int(candidate["layer"]), target_neuron=int(candidate["neuron"]))
                    for row in prompt_rows:
                        rows.extend(run_upstream_suppression_record(model, tokenizer, prompt_row=row, candidate=candidate, wrong_candidate=wrong, random_neuron=random, selected_heads=heads, alpha_grid=alpha_grid, device=target))
                path = out / "suppression.jsonl"
                count = write_jsonl(path, rows, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_e3_suppression", stage="e3-upstream", role="output", record_count=count)
                stage.count(count)
        if "e3-downstream" in enabled:
            with run.stage("e3-downstream") as stage:
                rows = []
                for row in financial:
                    if str(row["ticker"]) in trusted and heads:
                        rows.extend(run_downstream_suppression_record(model, tokenizer, prompt_row=row, selected_heads=heads, beta_grid=beta_grid, grouping=grouping, device=target))
                path = out / "downstream.jsonl"
                count = write_jsonl(path, rows, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_e3_downstream", stage="e3-downstream", role="output", record_count=count)
                stage.count(count)
        if "analyze" in enabled:
            with run.stage("analyze") as stage:
                rows = []
                for name in ("suppression.jsonl", "downstream.jsonl"):
                    path = out / name
                    if path.is_file():
                        rows.extend(read_jsonl(path))
                if not rows:
                    raise ValueError("analyze requires completed E3 suppression output")
                summary = analyze_e3_records(rows)
                path = run.run_directory / "analyze" / "summary.json"
                write_json(path, summary, overwrite=True)
                write_metadata(run.run_directory / "analyze" / "metadata.json", {"schema_version": 1, "artifact_type": "entity_cell_e3_analysis_metadata", "selected_heads": [list(item) for item in heads], "alpha_grid": list(alpha_grid), "beta_grid": list(beta_grid), "raw_runtime_payloads": False}, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_e3_analysis", stage="analyze", role="output")
                stage.count(len(summary["groups"]))
        run.finalize(required_stages=set(enabled))
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def defaultdict_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["ticker"]), []).append(row)
    return result


def analyze_e3_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('ticker')}:{row.get('phase')}:{row.get('scope')}"
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        preservation = [row.get("controls", {}).get("preservation", {}) for row in values]
        evidence_deltas = [float(item["evidence"]["delta"]) for item in preservation if isinstance(item, Mapping) and isinstance(item.get("evidence"), Mapping)]
        identity_deltas = [float(item["identity_header"]["delta"]) for item in preservation if isinstance(item, Mapping) and isinstance(item.get("identity_header"), Mapping)]
        summaries.append({"group": key, "record_count": len(values), "mean_margin": sum(float(row["margin"]) for row in values) / len(values), "mean_anonymous_progress": sum(float(row["anonymous_progress"]) for row in values) / len(values), "flip_count": sum(bool(row["flip"]) for row in values), "eligible_count": sum(bool(row.get("controls", {}).get("eligible", True)) for row in values), "mean_evidence_dla_delta": sum(evidence_deltas) / len(evidence_deltas) if evidence_deltas else None, "mean_identity_dla_delta": sum(identity_deltas) / len(identity_deltas) if identity_deltas else None})
    return {"schema_version": 1, "artifact_type": "entity_cell_e3_analysis", "groups": summaries, "discovery_only": True, "raw_runtime_payloads": False}


def analyze_e3(run_root: str | Path) -> Path:
    root = Path(run_root)
    rows = []
    for path in (root / "e3" / "suppression.jsonl", root / "e3" / "downstream.jsonl"):
        if path.is_file():
            rows.extend(read_jsonl(path))
    path = root / "analyze" / "summary.json"
    write_json(path, analyze_e3_records(rows), overwrite=True)
    return path


__all__ = ["DATASET", "E3_STAGES", "E3_ANALYZE_STAGE", "analyze_e3", "analyze_e3_records", "run_downstream_suppression_record", "run_e3", "run_upstream_suppression_record"]
