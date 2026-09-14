"""Entity-to-dial phase pipelines (protocol docs/entity-to-dial/details/proposal-phase-abc.md Rev 1).

Phase A: token-group x layer sufficiency map (192 group patch forwards).
Phase B: handoff-interval block-level (MLP vs attention) patch.
Phase C: dial-path probe (observation, push, unexplained gap).

Stages per phase: prepare → forward → analyze, then ArtifactRun.finalize.
All upstream runs are verified fail-closed in prepare (SHA-256, record
counts, manifest status); transient states are never persisted.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import (
    mid_residual_interventions,
    record_block_states,
    residual_interventions,
)
from llm_bias.core.inference.mlp_addition import mlp_addition, mlp_summed_derivatives
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids, token_span

from .analysis import (
    bootstrap_ci,
    c1_descriptive,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_gate_d,
    evaluate_gate_e,
    evaluate_gate_f,
    normalized_transfer,
    spearman,
    toward_source_delta,
)
from .attribution import (
    differentiable_margin,
    mlp_all_positions_derivative,
    mlp_position_derivative,
    top_channel_stats,
)
from .block_patch import (
    identity_mapping,
    make_block_transform,
    make_position_transform,
    nearest_position_mapping,
)
from .dial_probe import (
    answer_token_ids,
    dial_value_capture,
    margin_from_log_probs,
    probe_forward,
    scoring_ids,
)
from .joint_patch import (
    dial_channel_transplant,
    dial_footprint_direction,
    directional_push_transform,
    dual_hook_interventions,
    load_pca_basis,
    make_delta_transform,
    make_full_transform,
    make_joint_transform,
    make_projected_transplant,
    make_projected_transform,
    pca_state_directions,
    project_delta,
    state_difference_rows,
    stack_state_difference,
)
from .spans import anonymous_prompt, entity_char_span, instruction_char_span, resolve_token_groups
from .template import (
    BOTTOM_GROUP,
    C1_RHO_WARNING,
    CROSS_CHECK_WARNING_NATS,
    DATASET,
    D2_CONTROLS_N,
    D2_CONTROLS_SEED_BASE,
    D2_PARTITION_TOLERANCE,
    DECISION_PREFIX,
    DEFAULT_PHASE2A_REV2_RUN,
    DIAL_LAYER,
    DIAL_NEURON,
    E1_SMOKE_FULL_BAND,
    E2A_RATIO_TARGET,
    E2_K_SWEEP,
    E2_LAYER,
    E2_PCA_DIM,
    F1_CONSISTENCY_TOLERANCE,
    F2_ALPHAS,
    F2_JITTER_BAND,
    F2_M_ANON_BAND,
    F2_M_ANON_REF,
    F2_PUSH_BOUND,
    F_LAYER,
    GATE_E1_RATIO_MIN,
    GATE_E2B_FALSIFIER_MAX,
    GATE_E2B_RATIO_MIN,
    GATE_F1_RATIO_MIN,
    GROUP_GAP_MIN,
    NOOP_TOLERANCE,
    PHASE_A_EARLY_LAYERS,
    PHASE_A_LAYERS,
    PHASE_B_LAYERS,
    PHASE_D_LAYERS,
    PHASE_E_LAYERS,
    PROTOCOL,
    PROTOCOL_D,
    PROTOCOL_D_REV,
    PROTOCOL_E,
    PROTOCOL_E_REV,
    PROTOCOL_F,
    PROTOCOL_F_REV,
    RATIO_MIN_FULL_DM,
    SCHEMA_VERSION,
    SMOKE_A_LAYERS,
    SMOKE_B_LAYERS,
    SMOKE_C_TICKERS,
    SMOKE_D2_TICKER,
    SMOKE_D_LAYERS,
    SMOKE_DIRECTIONS,
    SMOKE_E2_K_SWEEP,
    SMOKE_E_DIRECTIONS,
    SMOKE_E_LAYERS,
    SMOKE_F_DIRECTION,
    SMOKE_F2_ALPHAS,
    TOP_GROUP,
)

N_CANONICAL_ROWS = 16
N_2A_VARIANTS = 4


# ── upstream verification ─────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _verify_upstream(run_root: Path, files: dict[str, int | None]) -> dict:
    """Fail-closed upstream run check: manifest complete, SHA-256, counts."""
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"upstream manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    if status != "complete":
        raise ValueError(
            f"upstream run {run_root.name} manifest status={status!r}, expected 'complete'"
        )
    entry: dict[str, Any] = {
        "run_id": run_root.name,
        "root": str(run_root),
        "manifest_status": status,
        "files": {},
    }
    for rel, expected_count in files.items():
        path = run_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing upstream file: {path}")
        file_entry: dict[str, Any] = {"sha256": _sha256(path)}
        if expected_count is not None:
            count = _count_jsonl(path)
            if count != expected_count:
                raise ValueError(
                    f"record count mismatch for {path}: got {count}, expected {expected_count}"
                )
            file_entry["n_records"] = count
        entry["files"][rel] = file_entry
    return entry


def _canonical_2a_rows(phase2a_run: Path) -> dict[str, dict]:
    rows = read_jsonl(phase2a_run / "prepare" / "prompts.jsonl")
    canon = {
        r["ticker"]: r
        for r in rows
        if r.get("reverse") is False and r.get("order") == 0
    }
    if len(canon) != N_CANONICAL_ROWS:
        raise ValueError(f"expected {N_CANONICAL_ROWS} canonical 2A rows, found {len(canon)}")
    return canon


def _canonical_2a_margins(phase2a_run: Path) -> dict[str, float]:
    results = read_jsonl(phase2a_run / "forward" / "results.jsonl")
    margins = {
        r["ticker"]: float(r["margin"])
        for r in results
        if r.get("reverse") is False and r.get("order") == 0
    }
    if len(margins) != N_CANONICAL_ROWS:
        raise ValueError(
            f"expected {N_CANONICAL_ROWS} canonical 2A margins, found {len(margins)}"
        )
    return margins


def pure_entity_margins(phase2a_run: Path) -> dict[str, float]:
    """Per-ticker median margin over the 4 stored variants (2A definition)."""
    results = read_jsonl(phase2a_run / "forward" / "results.jsonl")
    by_ticker: dict[str, list[float]] = {}
    for row in results:
        by_ticker.setdefault(row["ticker"], []).append(float(row["margin"]))
    if len(by_ticker) != N_CANONICAL_ROWS:
        raise ValueError(f"expected {N_CANONICAL_ROWS} tickers, found {len(by_ticker)}")
    for ticker, values in by_ticker.items():
        if len(values) != N_2A_VARIANTS:
            raise ValueError(f"ticker {ticker} has {len(values)} variants, expected 4")
    return {t: statistics.median(v) for t, v in by_ticker.items()}


def cross_check_stored_margins(phase2a_run: Path, pure: dict[str, float]) -> None:
    """Recomputed medians must match the stored 2A analyze summary."""
    summary = json.loads(
        (phase2a_run / "analyze" / "summary.json").read_text(encoding="utf-8")
    )
    stored = summary["pure_entity_margin_median"]
    for ticker, value in pure.items():
        if ticker not in stored:
            raise ValueError(f"stored 2A margin missing ticker {ticker}")
        if abs(float(stored[ticker]) - value) > 1e-9:
            raise ValueError(
                f"pure entity margin mismatch for {ticker}: "
                f"recomputed {value!r} vs stored {stored[ticker]!r}"
            )


def verify_group_gap(pure: dict[str, float]) -> float:
    """Pre-check 1 (§4.1): group gap >= 0.5 nats, else fail-closed."""
    missing = [t for t in TOP_GROUP + BOTTOM_GROUP if t not in pure]
    if missing:
        raise ValueError(f"pure entity margins missing tickers: {missing}")
    gap = min(pure[t] for t in TOP_GROUP) - max(pure[t] for t in BOTTOM_GROUP)
    if gap < GROUP_GAP_MIN:
        raise ValueError(
            f"group gap {gap:.4f} nats < {GROUP_GAP_MIN}; pre-check 1 failed, aborting"
        )
    return gap


def frozen_directions(phase2b_run: Path) -> list[tuple[str, str]]:
    """The frozen 8 directions (top x bottom, both orders)."""
    pairs = json.loads((phase2b_run / "pairs" / "directions.json").read_text(encoding="utf-8"))
    directions = [tuple(d) for d in pairs["directions"]]
    expected = (
        {(s, t) for s in TOP_GROUP for t in BOTTOM_GROUP}
        | {(t, s) for s in TOP_GROUP for t in BOTTOM_GROUP}
    )
    if len(directions) != 8 or set(directions) != expected:
        raise ValueError("2B directions do not match the frozen top/bottom pairs")
    return directions


def verify_frozen_groups(rev2_summary: dict) -> None:
    groups = rev2_summary["descriptive"]["groups"]
    if set(groups["top"]) != set(TOP_GROUP) or set(groups["bottom"]) != set(BOTTOM_GROUP):
        raise ValueError("rev2 frozen groups mismatch with protocol constants")


def _forward_device(model: Any) -> Any:
    return (
        model.input_device
        if hasattr(model, "input_device")
        else "cuda" if torch.cuda.is_available() else "cpu"
    )


def _patched_final_margin(
    model: Any, input_tensor: torch.Tensor, transforms: dict[int, Any]
) -> torch.Tensor:
    """Final-position FP32 log probabilities under post-block interventions."""
    final_layer = int(model.n_layers) - 1
    with residual_interventions(model, transforms):
        residual = record_residuals(model, input_tensor, [final_layer])[final_layer]
    return fp32_next_token_log_probs(model, residual[:, -1, :])


def _live_margin(model: Any, residual_final: torch.Tensor, buy_id: int, sell_id: int) -> float:
    return margin_from_log_probs(
        fp32_next_token_log_probs(model, residual_final[:, -1, :]), buy_id, sell_id
    )


def _check_noop(value: float, live: float, where: str) -> None:
    if abs(value - live) > NOOP_TOLERANCE:
        raise ValueError(f"self-source no-op violated at {where}: ΔM={value - live:.3e}")


def _record(
    *,
    phase: str,
    direction: str,
    layer: int,
    key_field: str,
    key_value: str,
    patched: float,
    m_source: float,
    m_target: float,
    live_target_margin: float,
    **extra: Any,
) -> dict:
    return {
        "phase": phase,
        "direction": direction,
        "layer": layer,
        key_field: key_value,
        "patched_margin": patched,
        "toward_source_delta_m": toward_source_delta(patched, m_source, m_target),
        "normalized_transfer": normalized_transfer(patched, m_source, m_target),
        "m_source": m_source,
        "m_target": m_target,
        "live_target_margin": live_target_margin,
        **extra,
    }


# ── Phase A ───────────────────────────────────────────────────────────────────


def _phase_a_prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    phase2a_run: Path,
    phase2b_run: Path,
    phase2a_rev2_run: Path,
    directions: list[tuple[str, str]],
    smoke: bool,
) -> tuple[dict[str, dict], dict[str, float], list[tuple[str, str]]]:
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        rev2_summary = json.loads(
            (phase2a_rev2_run / "analyze" / "summary.json").read_text(encoding="utf-8")
        )
        verify_frozen_groups(rev2_summary)
        frozen = frozen_directions(phase2b_run)
        pure = pure_entity_margins(phase2a_run)
        cross_check_stored_margins(phase2a_run, pure)
        gap = verify_group_gap(pure)
        canon = _canonical_2a_rows(phase2a_run)

        tickers = sorted({t for d in directions for t in d})
        rows = []
        for ticker in tickers:
            row = dict(canon[ticker])
            ticker_span, name_span = resolve_token_groups(tokenizer, canon[ticker])
            row["ticker_span"] = list(ticker_span)
            row["name_span"] = list(name_span)
            row["pure_entity_margin"] = pure[ticker]
            rows.append(row)

        provenance = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_a_provenance",
            "protocol": PROTOCOL,
            "protocol_rev": 1,
            "smoke": smoke,
            "raw_runtime_payloads": False,
            "phase2a_run": _verify_upstream(
                phase2a_run,
                {
                    "prepare/prompts.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "forward/results.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "analyze/summary.json": None,
                },
            ),
            "phase2a_rev2_run": _verify_upstream(
                phase2a_rev2_run, {"analyze/summary.json": None}
            ),
            "phase2b_run": _verify_upstream(
                phase2b_run,
                {"pairs/directions.json": None, "sweep/records.jsonl": 1024},
            ),
            "frozen_groups": {"top": sorted(TOP_GROUP), "bottom": sorted(BOTTOM_GROUP)},
            "directions_frozen": [list(d) for d in frozen],
            "directions": [list(d) for d in directions],
            "pure_entity_margins": {t: pure[t] for t in sorted(pure)},
            "group_gap": gap,
            "group_gap_threshold": GROUP_GAP_MIN,
        }
        rows_path = out_dir / "rows.jsonl"
        count = write_jsonl(rows_path, rows, overwrite=True)
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_a_prepare",
                "n_rows": count,
                "group_gap": gap,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            rows_path, artifact_type="entity_to_dial_a_prepare", stage="prepare", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "provenance.json", artifact_type="entity_to_dial_a_provenance", stage="prepare", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_a_prepare_metadata", stage="prepare", role="output"
        )
        stage.count(count)
    return {r["ticker"]: r for r in rows}, pure, directions


def _phase_a_forward(
    run: ArtifactRun,
    model_path: str,
    rows: dict[str, dict],
    pure: dict[str, float],
    directions: list[tuple[str, str]],
    *,
    layers: list[int],
) -> None:
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with run.stage("forward") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        final_layer = int(model.n_layers) - 1
        records: list[dict] = []
        for index, (source_ticker, target_ticker) in enumerate(directions):
            source_row, target_row = rows[source_ticker], rows[target_ticker]
            m_source, m_target = pure[source_ticker], pure[target_ticker]
            source_tensor = torch.tensor(
                [scoring_ids(tokenizer, source_row["formatted"])], dtype=torch.long, device=device
            )
            target_tensor = torch.tensor(
                [scoring_ids(tokenizer, target_row["formatted"])], dtype=torch.long, device=device
            )
            buy_id, sell_id = answer_token_ids(
                tokenizer, target_row["formatted"] + DECISION_PREFIX
            )
            source_residuals = record_residuals(model, source_tensor, layers)
            target_residuals = record_residuals(model, target_tensor, [*layers, final_layer])
            live_target_margin = _live_margin(model, target_residuals[final_layer], buy_id, sell_id)
            direction = f"{source_ticker}->{target_ticker}"
            for layer in layers:
                for group in ("ticker", "name"):
                    mapping = nearest_position_mapping(
                        tuple(source_row[f"{group}_span"]), tuple(target_row[f"{group}_span"])
                    )
                    if not mapping:
                        raise ValueError(f"empty {group} span mapping at L{layer}")
                    transform = make_position_transform(source_residuals[layer], mapping)
                    patched = margin_from_log_probs(
                        _patched_final_margin(model, target_tensor, {layer: transform}),
                        buy_id,
                        sell_id,
                    )
                    records.append(
                        _record(
                            phase="a", direction=direction, layer=layer,
                            key_field="token_group", key_value=group,
                            patched=patched, m_source=m_source, m_target=m_target,
                            live_target_margin=live_target_margin,
                        )
                    )
                # Self no-op: full-entity self-patch (bit-exact expected).
                self_transform = make_position_transform(
                    target_residuals[layer], identity_mapping(tuple(target_row["entity_span"]))
                )
                self_patched = margin_from_log_probs(
                    _patched_final_margin(model, target_tensor, {layer: self_transform}),
                    buy_id,
                    sell_id,
                )
                _check_noop(self_patched, live_target_margin, f"phase A L{layer} {direction}")
                records.append(
                    _record(
                        phase="a", direction=direction, layer=layer,
                        key_field="token_group", key_value="self_noop",
                        patched=self_patched, m_source=m_source, m_target=m_target,
                        live_target_margin=live_target_margin,
                        noop_delta_m=self_patched - live_target_margin,
                    )
                )
            print(f"  direction {index + 1}/{len(directions)}: {direction}", flush=True)
        count = write_jsonl(path, records, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_a_forward",
                "n_records": count,
                "layers": layers,
                "self_source_noop_verified": True,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_a_forward", stage="forward", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_a_forward_metadata", stage="forward", role="output"
        )
        stage.count(count)


def _phase_a_analyze(
    run: ArtifactRun,
    *,
    phase2b_run: Path,
    layers: list[int],
    smoke: bool,
) -> dict:
    records = read_jsonl(run.run_directory / "forward" / "records.jsonl")
    curves: dict[str, dict[int, dict]] = {}
    for group in ("ticker", "name"):
        curves[group] = {}
        for layer in layers:
            group_records = [
                r for r in records if r["token_group"] == group and r["layer"] == layer
            ]
            if not group_records:
                raise ValueError(f"missing directions for {group} L{layer}")
            deltas = [float(r["toward_source_delta_m"]) for r in group_records]
            transfers = [float(r["normalized_transfer"]) for r in group_records]
            ci = bootstrap_ci(deltas) if len(deltas) >= 4 else None
            curves[group][layer] = {
                "mean_toward_source_delta_m": statistics.fmean(deltas),
                "mean_normalized_transfer": statistics.fmean(transfers),
                "delta_m_ci_95": list(ci) if ci is not None else None,
                "n_directions": len(deltas),
            }
    gate = evaluate_gate_a(records, early_layers=PHASE_A_EARLY_LAYERS)

    # Upper-bound reference: 2B entity-span per-layer mean T (descriptive).
    sweep = read_jsonl(phase2b_run / "sweep" / "records.jsonl")
    entity_rows = [r for r in sweep if r["span"] == "entity"]
    upper_bound = {
        layer: (
            statistics.fmean(float(r["normalized_transfer"]) for r in entity_rows if r["layer"] == layer)
            if any(r["layer"] == layer for r in entity_rows)
            else None
        )
        for layer in layers
    }

    early_records = [
        r
        for r in records
        if r["token_group"] == "name" and r["layer"] in set(PHASE_A_EARLY_LAYERS)
    ]
    name_early: dict[str, float] = {}
    for direction in sorted({r["direction"] for r in early_records}):
        values = [
            float(r["toward_source_delta_m"])
            for r in early_records
            if r["direction"] == direction
        ]
        name_early[direction] = statistics.fmean(values)
    name_early_ci = bootstrap_ci(list(name_early.values())) if len(name_early) >= 4 else None

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "entity_to_dial_a_analysis",
        "n_records": len(records),
        "layers": layers,
        "curves": curves,
        "gate_a": gate,
        "upper_bound_reference": {
            "source": str(phase2b_run),
            "span": "entity",
            "mean_normalized_transfer": {str(k): v for k, v in upper_bound.items()},
        },
        "descriptive": {
            "name_group_early": {
                "per_direction_mean": name_early,
                "ci_95": list(name_early_ci) if name_early_ci is not None else None,
            },
        },
        "smoke": smoke,
        "raw_runtime_payloads": False,
    }
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    with run.stage("analyze") as stage:
        write_json(path, summary, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_a_analysis_metadata",
                "gate_a_status": gate["status"],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_a_analysis", stage="analyze", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_a_analysis_metadata", stage="analyze", role="output"
        )
        stage.count(len(curves["ticker"]))
    return summary


def run_phase_a(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    phase2b_run: str | Path,
    phase2a_rev2_run: str | Path = DEFAULT_PHASE2A_REV2_RUN,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    phase2b_run = Path(phase2b_run)
    phase2a_rev2_run = Path(phase2a_rev2_run)
    directions = (
        [tuple(d) for d in SMOKE_DIRECTIONS] if smoke else frozen_directions(phase2b_run)
    )
    layers = list(SMOKE_A_LAYERS if smoke else PHASE_A_LAYERS)
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        rows, pure, _ = _phase_a_prepare(
            run, tokenizer,
            phase2a_run=phase2a_run, phase2b_run=phase2b_run,
            phase2a_rev2_run=phase2a_rev2_run,
            directions=directions, smoke=smoke,
        )
        _phase_a_forward(run, model_path, rows, pure, directions, layers=layers)
        summary = _phase_a_analyze(
            run, phase2b_run=phase2b_run, layers=layers, smoke=smoke
        )
        run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


# ── Phase B ───────────────────────────────────────────────────────────────────


def _phase_b_prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    phase2a_run: Path,
    phase2b_run: Path,
    phase2a_rev2_run: Path,
    directions: list[tuple[str, str]],
    smoke: bool,
) -> tuple[dict[str, dict], dict[str, float]]:
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        rev2_summary = json.loads(
            (phase2a_rev2_run / "analyze" / "summary.json").read_text(encoding="utf-8")
        )
        verify_frozen_groups(rev2_summary)
        frozen = frozen_directions(phase2b_run)
        pure = pure_entity_margins(phase2a_run)
        cross_check_stored_margins(phase2a_run, pure)
        gap = verify_group_gap(pure)
        canon = _canonical_2a_rows(phase2a_run)
        tickers = sorted({t for d in directions for t in d})
        rows = []
        for ticker in tickers:
            row = dict(canon[ticker])
            row["pure_entity_margin"] = pure[ticker]
            rows.append(row)
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_b_provenance",
            "protocol": PROTOCOL,
            "protocol_rev": 1,
            "smoke": smoke,
            "raw_runtime_payloads": False,
            "phase2a_run": _verify_upstream(
                phase2a_run,
                {
                    "prepare/prompts.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "forward/results.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "analyze/summary.json": None,
                },
            ),
            "phase2a_rev2_run": _verify_upstream(
                phase2a_rev2_run, {"analyze/summary.json": None}
            ),
            "phase2b_run": _verify_upstream(
                phase2b_run, {"pairs/directions.json": None}
            ),
            "frozen_groups": {"top": sorted(TOP_GROUP), "bottom": sorted(BOTTOM_GROUP)},
            "directions_frozen": [list(d) for d in frozen],
            "directions": [list(d) for d in directions],
            "pure_entity_margins": {t: pure[t] for t in sorted(pure)},
            "group_gap": gap,
            "group_gap_threshold": GROUP_GAP_MIN,
        }
        rows_path = out_dir / "rows.jsonl"
        count = write_jsonl(rows_path, rows, overwrite=True)
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_b_prepare",
                "n_rows": count,
                "group_gap": gap,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            rows_path, artifact_type="entity_to_dial_b_prepare", stage="prepare", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "provenance.json", artifact_type="entity_to_dial_b_provenance", stage="prepare", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_b_prepare_metadata", stage="prepare", role="output"
        )
        stage.count(count)
    return {r["ticker"]: r for r in rows}, pure


def _phase_b_forward(
    run: ArtifactRun,
    model_path: str,
    rows: dict[str, dict],
    pure: dict[str, float],
    directions: list[tuple[str, str]],
    *,
    layers: list[int],
) -> None:
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with run.stage("forward") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        final_layer = int(model.n_layers) - 1
        records: list[dict] = []
        for index, (source_ticker, target_ticker) in enumerate(directions):
            source_row, target_row = rows[source_ticker], rows[target_ticker]
            m_source, m_target = pure[source_ticker], pure[target_ticker]
            direction = f"{source_ticker}->{target_ticker}"
            source_tensor = torch.tensor(
                [scoring_ids(tokenizer, source_row["formatted"])], dtype=torch.long, device=device
            )
            target_tensor = torch.tensor(
                [scoring_ids(tokenizer, target_row["formatted"])], dtype=torch.long, device=device
            )
            buy_id, sell_id = answer_token_ids(
                tokenizer, target_row["formatted"] + DECISION_PREFIX
            )
            capture_layers = [*layers, final_layer]
            source_states = record_block_states(model, source_tensor, capture_layers)
            target_states = record_block_states(model, target_tensor, capture_layers)
            live_target_margin = _live_margin(
                model, target_states[final_layer]["post"], buy_id, sell_id
            )
            mapping = nearest_position_mapping(
                tuple(source_row["entity_span"]), tuple(target_row["entity_span"])
            )
            if not mapping:
                raise ValueError("empty entity span mapping")
            for layer in layers:
                for component in ("mlp", "attn"):
                    if component == "mlp":
                        transform = make_block_transform(
                            captured_base=target_states[layer]["mid"],
                            source_states=source_states[layer],
                            component="mlp",
                            mapping=mapping,
                        )
                        log_probs = _patched_final_margin(model, target_tensor, {layer: transform})
                    else:
                        transform = make_block_transform(
                            captured_base=target_states[layer]["pre"],
                            source_states=source_states[layer],
                            component="attn",
                            mapping=mapping,
                        )
                        with mid_residual_interventions(model, {layer: transform}):
                            residual = record_residuals(model, target_tensor, [final_layer])[final_layer]
                        log_probs = fp32_next_token_log_probs(model, residual[:, -1, :])
                    patched = margin_from_log_probs(log_probs, buy_id, sell_id)
                    records.append(
                        _record(
                            phase="b", direction=direction, layer=layer,
                            key_field="component", key_value=component,
                            patched=patched, m_source=m_source, m_target=m_target,
                            live_target_margin=live_target_margin,
                        )
                    )
                # Self no-op per component: copy the target's own state onto
                # itself (clone/assign, bit-exact by construction, as in 2B).
                # The fp32 block arithmetic is covered by unit tests instead.
                for component in ("mlp", "attn"):
                    self_state = target_states[layer]["post" if component == "mlp" else "mid"]
                    self_transform = make_position_transform(
                        self_state, identity_mapping(tuple(target_row["entity_span"]))
                    )
                    if component == "mlp":
                        self_log_probs = _patched_final_margin(model, target_tensor, {layer: self_transform})
                    else:
                        with mid_residual_interventions(model, {layer: self_transform}):
                            residual = record_residuals(model, target_tensor, [final_layer])[final_layer]
                        self_log_probs = fp32_next_token_log_probs(model, residual[:, -1, :])
                    self_patched = margin_from_log_probs(self_log_probs, buy_id, sell_id)
                    _check_noop(self_patched, live_target_margin, f"phase B L{layer}/{component} {direction}")
                    records.append(
                        _record(
                            phase="b", direction=direction, layer=layer,
                            key_field="component", key_value="self_noop",
                            patched=self_patched, m_source=m_source, m_target=m_target,
                            live_target_margin=live_target_margin,
                            noop_for=component,
                            noop_delta_m=self_patched - live_target_margin,
                        )
                    )
            print(f"  direction {index + 1}/{len(directions)}: {direction}", flush=True)
        count = write_jsonl(path, records, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_b_forward",
                "n_records": count,
                "layers": layers,
                "self_source_noop_verified": True,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_b_forward", stage="forward", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_b_forward_metadata", stage="forward", role="output"
        )
        stage.count(count)


def _phase_b_analyze(
    run: ArtifactRun,
    rows: dict[str, dict],
    *,
    layers: list[int],
    smoke: bool,
) -> dict:
    records = read_jsonl(run.run_directory / "forward" / "records.jsonl")
    curves: dict[str, dict[int, dict]] = {}
    for component in ("mlp", "attn"):
        curves[component] = {}
        for layer in layers:
            component_records = [
                r for r in records if r["component"] == component and r["layer"] == layer
            ]
            if not component_records:
                raise ValueError(f"missing directions for {component} L{layer}")
            deltas = [float(r["toward_source_delta_m"]) for r in component_records]
            transfers = [float(r["normalized_transfer"]) for r in component_records]
            ci = bootstrap_ci(deltas) if len(deltas) >= 4 else None
            curves[component][layer] = {
                "mean_toward_source_delta_m": statistics.fmean(deltas),
                "mean_normalized_transfer": statistics.fmean(transfers),
                "delta_m_ci_95": list(ci) if ci is not None else None,
                "n_directions": len(deltas),
            }
    sector_of = {t: rows[t]["sector"] for t in rows}
    companies = sorted(set(TOP_GROUP) | set(BOTTOM_GROUP))
    gate = evaluate_gate_b(records, layers=layers, sector_of=sector_of, companies=companies)
    strongest = gate.get("strongest_layer")
    descriptive: dict[str, Any] = {}
    if strongest is not None:
        descriptive["strongest_layer_mlp_vs_attn"] = {
            "mlp": curves["mlp"][strongest],
            "attn": curves["attn"][strongest],
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "entity_to_dial_b_analysis",
        "n_records": len(records),
        "layers": layers,
        "curves": curves,
        "gate_b": gate,
        "descriptive": descriptive,
        "smoke": smoke,
        "raw_runtime_payloads": False,
    }
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    with run.stage("analyze") as stage:
        write_json(path, summary, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_b_analysis_metadata",
                "gate_b_status": gate["status"],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_b_analysis", stage="analyze", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_b_analysis_metadata", stage="analyze", role="output"
        )
        stage.count(len(curves["mlp"]))
    return summary


def run_phase_b(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    phase2b_run: str | Path,
    phase2a_rev2_run: str | Path = DEFAULT_PHASE2A_REV2_RUN,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    phase2b_run = Path(phase2b_run)
    phase2a_rev2_run = Path(phase2a_rev2_run)
    directions = (
        [tuple(d) for d in SMOKE_DIRECTIONS] if smoke else frozen_directions(phase2b_run)
    )
    layers = list(SMOKE_B_LAYERS if smoke else PHASE_B_LAYERS)
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        rows, pure = _phase_b_prepare(
            run, tokenizer,
            phase2a_run=phase2a_run, phase2b_run=phase2b_run,
            phase2a_rev2_run=phase2a_rev2_run,
            directions=directions, smoke=smoke,
        )
        _phase_b_forward(run, model_path, rows, pure, directions, layers=layers)
        _phase_b_analyze(run, rows, layers=layers, smoke=smoke)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


# ── Phase C ───────────────────────────────────────────────────────────────────


def _anon_row(tokenizer: Any, named_row: dict[str, Any]) -> dict[str, Any]:
    prompt = anonymous_prompt(named_row["prompt"], named_row["ticker"], named_row["name"])
    formatted = format_prompt(
        tokenizer, prompt, use_chat_template=True, enable_thinking=False
    )
    prompt_ids = input_ids(tokenizer, formatted, add_special_tokens=True)
    body_start = formatted.find(prompt)
    if body_start < 0:
        raise ValueError("anonymous prompt body not found in formatted text")
    from llm_bias.core.prompt_input.encoding import token_span

    char_span = entity_char_span(prompt)
    entity_span = token_span(
        tokenizer,
        formatted,
        body_start + char_span[0],
        body_start + char_span[1],
        add_special_tokens=True,
    )
    if entity_span is None:
        raise ValueError("could not map anonymous entity span to tokens")
    return {
        "id": f"{named_row['id']}:anon",
        "ticker": named_row["ticker"],
        "name": named_row["name"],
        "sector": named_row["sector"],
        "prompt_type": "anon",
        "prompt": prompt,
        "formatted": formatted,
        "prompt_ids": prompt_ids,
        "entity_span": list(entity_span),
        "entity_position": entity_span[1] - 1,
        "final_position": len(prompt_ids) - 1,
    }


def _phase_c_prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    phase2a_run: Path,
    tickers: list[str],
    smoke: bool,
) -> tuple[dict[str, dict], dict[str, dict]]:
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        pure = pure_entity_margins(phase2a_run)
        cross_check_stored_margins(phase2a_run, pure)
        gap = verify_group_gap(pure)
        canon = _canonical_2a_rows(phase2a_run)
        canonical_margins = _canonical_2a_margins(phase2a_run)
        rows = []
        for ticker in tickers:
            named = dict(canon[ticker])
            named["prompt_type"] = "named"
            named["pure_entity_margin"] = pure[ticker]
            named["stored_2a_margin"] = canonical_margins[ticker]
            rows.append(named)
            rows.append(_anon_row(tokenizer, canon[ticker]))
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_c_provenance",
            "protocol": PROTOCOL,
            "protocol_rev": 1,
            "smoke": smoke,
            "raw_runtime_payloads": False,
            "phase2a_run": _verify_upstream(
                phase2a_run,
                {
                    "prepare/prompts.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "forward/results.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "analyze/summary.json": None,
                },
            ),
            "dial_coordinate": [DIAL_LAYER, DIAL_NEURON],
            "tickers": list(tickers),
            "pure_entity_margins": {t: pure[t] for t in sorted(pure)},
            "group_gap": gap,
            "group_gap_threshold": GROUP_GAP_MIN,
        }
        rows_path = out_dir / "rows.jsonl"
        count = write_jsonl(rows_path, rows, overwrite=True)
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_c_prepare",
                "n_rows": count,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            rows_path, artifact_type="entity_to_dial_c_prepare", stage="prepare", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "provenance.json", artifact_type="entity_to_dial_c_provenance", stage="prepare", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_c_prepare_metadata", stage="prepare", role="output"
        )
        stage.count(count)
    named = {r["ticker"]: r for r in rows if r["prompt_type"] == "named"}
    anon = {r["ticker"]: r for r in rows if r["prompt_type"] == "anon"}
    return named, anon


def _phase_c_forward(
    run: ArtifactRun,
    model_path: str,
    named: dict[str, dict],
    anon: dict[str, dict],
) -> None:
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    tickers = sorted(named)
    with run.stage("forward") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        clean: list[dict] = []
        for ticker in tickers:
            for prompt_type, row in (("named", named[ticker]), ("anon", anon[ticker])):
                scoring = scoring_ids(tokenizer, row["formatted"])
                tensor = torch.tensor([scoring], dtype=torch.long, device=device)
                buy_id, sell_id = answer_token_ids(
                    tokenizer, row["formatted"] + DECISION_PREFIX
                )
                margin, dials = probe_forward(
                    model, tensor,
                    buy_id=buy_id, sell_id=sell_id,
                    dial_positions=(row["entity_position"], row["final_position"]),
                )
                if row["entity_position"] not in dials or row["final_position"] not in dials:
                    raise RuntimeError("dial positions not captured")
                clean.append(
                    {
                        "ticker": ticker,
                        "prompt_type": prompt_type,
                        "sector": row["sector"],
                        "margin": margin,
                        "dial_entity": dials[row["entity_position"]],
                        "dial_final": dials[row["final_position"]],
                    }
                )
        clean_by = {(r["ticker"], r["prompt_type"]): r for r in clean}
        push_rows: list[dict] = []
        for ticker in tickers:
            a_named = clean_by[(ticker, "named")]["dial_entity"]
            a_anon = clean_by[(ticker, "anon")]["dial_entity"]
            delta = a_named - a_anon
            scoring = scoring_ids(tokenizer, anon[ticker]["formatted"])
            tensor = torch.tensor([scoring], dtype=torch.long, device=device)
            buy_id, sell_id = answer_token_ids(
                tokenizer, anon[ticker]["formatted"] + DECISION_PREFIX
            )
            with mlp_addition(model, DIAL_LAYER, DIAL_NEURON, delta):
                pushed_margin, _ = probe_forward(
                    model, tensor, buy_id=buy_id, sell_id=sell_id
                )
            clean_anon = clean_by[(ticker, "anon")]["margin"]
            push_rows.append(
                {
                    "ticker": ticker,
                    "delta": delta,
                    "pushed_margin": pushed_margin,
                    "clean_margin_anon": clean_anon,
                    "delta_m_dial": pushed_margin - clean_anon,
                }
            )
        # δ=0 no-op on the first anonymous prompt (bit-exact expected).
        noop_ticker = tickers[0]
        noop_tensor = torch.tensor(
            [scoring_ids(tokenizer, anon[noop_ticker]["formatted"])], dtype=torch.long, device=device
        )
        noop_buy, noop_sell = answer_token_ids(
            tokenizer, anon[noop_ticker]["formatted"] + DECISION_PREFIX
        )
        with mlp_addition(model, DIAL_LAYER, DIAL_NEURON, 0.0):
            noop_margin, _ = probe_forward(
                model, noop_tensor, buy_id=noop_buy, sell_id=noop_sell
            )
        _check_noop(noop_margin, clean_by[(noop_ticker, "anon")]["margin"], "phase C δ=0 no-op")
        results = clean + push_rows
        count = write_jsonl(path, results, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_c_forward",
                "n_records": count,
                "dial_coordinate": [DIAL_LAYER, DIAL_NEURON],
                "noop_ticker": noop_ticker,
                "self_source_noop_verified": True,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_c_forward", stage="forward", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_c_forward_metadata", stage="forward", role="output"
        )
        stage.count(count)


def _phase_c_analyze(
    run: ArtifactRun,
    named: dict[str, dict],
    *,
    smoke: bool,
) -> dict:
    results = read_jsonl(run.run_directory / "forward" / "results.jsonl")
    clean = [r for r in results if "prompt_type" in r]
    push = {r["ticker"]: r for r in results if "pushed_margin" in r}
    named_by = {r["ticker"]: r for r in clean if r["prompt_type"] == "named"}
    anon_by = {r["ticker"]: r for r in clean if r["prompt_type"] == "anon"}
    tickers = sorted(named_by)
    if not tickers:
        raise ValueError("no clean results")
    gap = {t: named_by[t]["margin"] - anon_by[t]["margin"] for t in tickers}
    delta_m = {t: push[t]["delta_m_dial"] for t in tickers}
    dial_delta_entity = {
        t: named_by[t]["dial_entity"] - anon_by[t]["dial_entity"] for t in tickers
    }
    dial_delta_final = {
        t: named_by[t]["dial_final"] - anon_by[t]["dial_final"] for t in tickers
    }
    pure = {t: named[t]["pure_entity_margin"] for t in tickers}
    descriptive = c1_descriptive(dial_delta_entity, dial_delta_final, pure)
    rho_entity = descriptive["rho_entity_position"]
    rho_warning = rho_entity is not None and rho_entity < C1_RHO_WARNING
    gate = evaluate_gate_c([delta_m[t] for t in tickers], [gap[t] for t in tickers])
    residual = {t: gap[t] - delta_m[t] for t in tickers}
    cross_check = {
        t: abs(named_by[t]["margin"] - named[t]["stored_2a_margin"]) for t in tickers
    }
    max_diff = max(cross_check.values()) if cross_check else None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "entity_to_dial_c_analysis",
        "n_records": len(results),
        "n_companies": len(tickers),
        "per_company": {
            t: {
                "m_named": named_by[t]["margin"],
                "m_anon": anon_by[t]["margin"],
                "gap": gap[t],
                "delta_m_dial": delta_m[t],
                "delta": push[t]["delta"],
                "unexplained_gap": residual[t],
            }
            for t in tickers
        },
        "c1_descriptive": {
            **descriptive,
            "rho_warning_below_0.3": rho_warning,
        },
        "gate_c": gate,
        "c3_unexplained_gap": {
            "per_company": residual,
            "mean": statistics.fmean(residual[t] for t in tickers) if tickers else None,
        },
        "cross_check_2a": {
            "per_company_abs_diff": cross_check,
            "max_abs_diff": max_diff,
            "warning_threshold": CROSS_CHECK_WARNING_NATS,
            "warning": bool(max_diff is not None and max_diff > CROSS_CHECK_WARNING_NATS),
        },
        "smoke": smoke,
        "raw_runtime_payloads": False,
    }
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    with run.stage("analyze") as stage:
        write_json(path, summary, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_c_analysis_metadata",
                "gate_c_status": gate["status"],
                "cross_check_warning": summary["cross_check_2a"]["warning"],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_c_analysis", stage="analyze", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_c_analysis_metadata", stage="analyze", role="output"
        )
        stage.count(len(tickers))
    return summary


def run_phase_c(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    tickers = list(SMOKE_C_TICKERS) if smoke else sorted(_canonical_2a_rows(phase2a_run))
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        named, anon = _phase_c_prepare(
            run, tokenizer, phase2a_run=phase2a_run, tickers=tickers, smoke=smoke
        )
        _phase_c_forward(run, model_path, named, anon)
        _phase_c_analyze(run, named, smoke=smoke)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


# ── Phase D ─────────────────────────────────────────────────────────────────


def _phase_d_prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    phase2a_run: Path,
    phase2b_run: Path,
    directions: list[tuple[str, str]],
    smoke: bool,
) -> tuple[dict[str, dict], dict[str, float]]:
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        frozen = frozen_directions(phase2b_run)
        pure = pure_entity_margins(phase2a_run)
        cross_check_stored_margins(phase2a_run, pure)
        gap = verify_group_gap(pure)
        canon = _canonical_2a_rows(phase2a_run)
        # Rows cover ALL 16 canonical companies (protocol §4.3: D2 needs the
        # full 2A population for the cross-company Spearman), not only the
        # 4 direction tickers (D1 consumes source/target rows by lookup).
        tickers = sorted(canon)
        rows = []
        for ticker in tickers:
            row = dict(canon[ticker])
            row["pure_entity_margin"] = pure[ticker]
            row["instruction_last_position"] = int(row["instruction_span"][1]) - 1
            rows.append(row)
        # 1:1 alignment invariant (protocol §4.1): the instruction span is
        # the same token sequence for all companies, hence identical span
        # length; the per-direction patch uses the offset-preserving
        # mapping (identity when absolute starts coincide). Fail closed if
        # any row breaks the length invariant.
        span_lengths = {
            int(r["instruction_span"][1]) - int(r["instruction_span"][0]) for r in rows
        }
        if len(span_lengths) != 1:
            raise ValueError(
                f"instruction span lengths not 1:1 aligned across companies: "
                f"{sorted(span_lengths)}"
            )
        instruction_span_length = span_lengths.pop()
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_d_provenance",
            "protocol": PROTOCOL_D,
            "protocol_rev": PROTOCOL_D_REV,
            "smoke": smoke,
            "raw_runtime_payloads": False,
            "phase2a_run": _verify_upstream(
                phase2a_run,
                {
                    "prepare/prompts.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "forward/results.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "analyze/summary.json": None,
                },
            ),
            "phase2b_run": _verify_upstream(
                phase2b_run, {"pairs/directions.json": None, "sweep/records.jsonl": 1024}
            ),
            "frozen_groups": {"top": sorted(TOP_GROUP), "bottom": sorted(BOTTOM_GROUP)},
            "directions_frozen": [list(d) for d in frozen],
            "directions": [list(d) for d in directions],
            "instruction_span_length": instruction_span_length,
            "pure_entity_margins": {t: pure[t] for t in sorted(pure)},
            "group_gap": gap,
            "group_gap_threshold": GROUP_GAP_MIN,
        }
        rows_path = out_dir / "rows.jsonl"
        count = write_jsonl(rows_path, rows, overwrite=True)
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_d_prepare",
                "n_rows": count,
                "instruction_span_length": instruction_span_length,
                "group_gap": gap,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            rows_path, artifact_type="entity_to_dial_d_prepare", stage="prepare", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "provenance.json", artifact_type="entity_to_dial_d_provenance", stage="prepare", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_d_prepare_metadata", stage="prepare", role="output"
        )
        stage.count(count)
    return {r["ticker"]: r for r in rows}, pure


def _phase_d_forward_d1(
    run: ArtifactRun,
    model_path: str,
    rows: dict[str, dict],
    pure: dict[str, float],
    directions: list[tuple[str, str]],
    *,
    layers: list[int],
) -> None:
    """D1 instruction-span block sweep (no-grad), protocol §4.2."""
    out_dir = run.run_directory / "forward_d1"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with run.stage("forward_d1") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        final_layer = int(model.n_layers) - 1
        capture_layers = sorted(set(list(layers) + [final_layer]))
        records: list[dict] = []
        for index, (source_ticker, target_ticker) in enumerate(directions):
            source_row, target_row = rows[source_ticker], rows[target_ticker]
            m_source, m_target = pure[source_ticker], pure[target_ticker]
            direction = f"{source_ticker}->{target_ticker}"
            source_tensor = torch.tensor(
                [scoring_ids(tokenizer, source_row["formatted"])], dtype=torch.long, device=device
            )
            target_tensor = torch.tensor(
                [scoring_ids(tokenizer, target_row["formatted"])], dtype=torch.long, device=device
            )
            buy_id, sell_id = answer_token_ids(
                tokenizer, target_row["formatted"] + DECISION_PREFIX
            )
            source_states = record_block_states(model, source_tensor, capture_layers)
            target_states = record_block_states(model, target_tensor, capture_layers)
            live_target_margin = _live_margin(
                model, target_states[final_layer]["post"], buy_id, sell_id
            )
            source_span = tuple(int(p) for p in source_row["instruction_span"])
            target_span = tuple(int(p) for p in target_row["instruction_span"])
            mapping = nearest_position_mapping(source_span, target_span)
            if not mapping:
                raise ValueError("empty instruction span mapping")
            final_position = target_tensor.shape[1] - 1

            def patched_margin(layer: int, transform: Any, at_mid: bool) -> float:
                if at_mid:
                    with mid_residual_interventions(model, {layer: transform}):
                        residual = record_residuals(model, target_tensor, [final_layer])[final_layer]
                    log_probs = fp32_next_token_log_probs(model, residual[:, -1, :])
                else:
                    log_probs = _patched_final_margin(model, target_tensor, {layer: transform})
                return margin_from_log_probs(log_probs, buy_id, sell_id)

            for layer in layers:
                for component in ("mlp", "attn"):
                    base = target_states[layer]["mid" if component == "mlp" else "pre"]
                    transform = make_block_transform(
                        captured_base=base,
                        source_states=source_states[layer],
                        component=component,
                        mapping=mapping,
                    )
                    patched = patched_margin(layer, transform, at_mid=(component == "attn"))
                    records.append(
                        _record(
                            phase="d1", direction=direction, layer=layer,
                            key_field="component", key_value=component,
                            patched=patched, m_source=m_source, m_target=m_target,
                            live_target_margin=live_target_margin,
                        )
                    )
                # Self no-op per component: copy the target's own state onto
                # itself (clone/assign, bit-exact by construction).
                for component in ("mlp", "attn"):
                    self_state = target_states[layer]["post" if component == "mlp" else "mid"]
                    self_transform = make_position_transform(
                        self_state, identity_mapping(target_span)
                    )
                    self_patched = patched_margin(
                        layer, self_transform, at_mid=(component == "attn")
                    )
                    _check_noop(self_patched, live_target_margin, f"phase D L{layer}/{component} {direction}")
                    records.append(
                        _record(
                            phase="d1", direction=direction, layer=layer,
                            key_field="component", key_value=f"self_noop_{component}",
                            patched=self_patched, m_source=m_source, m_target=m_target,
                            live_target_margin=live_target_margin,
                            noop_for=component, noop_delta_m=self_patched - live_target_margin,
                        )
                    )
            # Final-position sanity at the final layer (descriptive; expect
            # T ≈ 0). Each company's final position is its own
            # margin-readout position, so the source's final position maps
            # to the target's final position (1:1).
            final_mapping = {final_position: int(source_tensor.shape[1]) - 1}
            for component, key in (("mlp", "final_mlp"), ("attn", "final_attn")):
                base = target_states[final_layer]["mid" if component == "mlp" else "pre"]
                transform = make_block_transform(
                    captured_base=base,
                    source_states=source_states[final_layer],
                    component=component,
                    mapping=final_mapping,
                )
                patched = patched_margin(final_layer, transform, at_mid=(component == "attn"))
                records.append(
                    _record(
                        phase="d1", direction=direction, layer=final_layer,
                        key_field="component", key_value=key,
                        patched=patched, m_source=m_source, m_target=m_target,
                        live_target_margin=live_target_margin,
                    )
                )
            print(f"  direction {index + 1}/{len(directions)}: {direction}", flush=True)
        count = write_jsonl(path, records, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_d_forward_d1",
                "n_records": count,
                "layers": layers,
                "final_layer": final_layer,
                "self_source_noop_verified": True,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_d_forward_d1", stage="forward_d1", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_d_forward_d1_metadata", stage="forward_d1", role="output"
        )
        stage.count(count)


def _phase_d_forward_d2(
    run: ArtifactRun,
    model_path: str,
    rows: dict[str, dict],
    pure: dict[str, float],
    *,
    layers: list[int],
    smoke: bool,
) -> None:
    """D2 instruction-position signed attribution (grad mode), protocol §4.3."""
    out_dir = run.run_directory / "forward_d2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with run.stage("forward_d2") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        companies = [SMOKE_D2_TICKER] if smoke else sorted(rows)
        sector_of = {t: rows[t]["sector"] for t in rows}
        grads: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
        for ticker in companies:
            row = rows[ticker]
            position = int(row["instruction_last_position"])
            tensor = torch.tensor(
                [scoring_ids(tokenizer, row["formatted"])], dtype=torch.long, device=device
            )
            buy_id, sell_id = answer_token_ids(tokenizer, row["formatted"] + DECISION_PREFIX)
            with mlp_position_derivative(model, list(layers), position) as captured:
                margin = differentiable_margin(model, tensor, buy_id, sell_id)
                margin.backward()
                for layer in layers:
                    if layer not in captured:
                        raise RuntimeError(f"derivative not captured at L{layer}")
                    grads[layer][ticker] = captured[layer]
            del margin
        stats = top_channel_stats(
            grads,
            {t: pure[t] for t in rows},
            sector_of,
            controls_n=D2_CONTROLS_N,
            controls_seed_base=D2_CONTROLS_SEED_BASE,
        )
        records = []
        for layer in layers:
            records.append(
                {
                    "phase": "d2",
                    "layer": layer,
                    "top_channel_idx": stats[layer]["top_channel_idx"],
                    "top_channel_rho": stats[layer]["top_channel_rho"],
                    "top_channel_sector_agreement": stats[layer]["top_channel_sector_agreement"],
                    "n_sectors": stats[layer]["n_sectors"],
                    "control_channel_idxs": stats[layer]["control_channel_idxs"],
                    "max_control_rho": stats[layer]["max_control_rho"],
                    "n_companies": stats[layer]["n_companies"],
                    # Renamed from the draft `gradient_finite`: the core
                    # serializer reserves the word "gradient" for raw
                    # payloads. True = every captured vector was finite.
                    "vector_finite": True,
                }
            )
        count = write_jsonl(path, records, overwrite=True)
        # Smoke-only derivative acceptance (protocol Rev 1.1 §4.3). Rev 0
        # gated on sign agreement >= 0.8 between the position derivative
        # and the all-position summed derivative; that assumed the single
        # position dominates the sum, which the real model contradicts
        # (measured 0.519 at L12 / 0.479 at L15 — position contributions
        # are distributed). The acceptance is therefore re-anchored on
        # mathematical invariants of the mechanic itself:
        #   (a) structural zero: the final-layer derivative at the
        #       (non-final) instruction position is exactly zero, because
        #       no later computation reads that position's final-layer
        #       output (the margin reads the final position only);
        #   (b) connectivity: a non-final layer's derivative is nonzero
        #       (the graph reaches the margin);
        #   (c) partition identity: the fp32 sum of the all-position
        #       capture reproduces the core mlp_summed_derivatives output
        #       within bf16 accumulation tolerance (norm-relative).
        # The Rev 0 sign agreement is still recorded, descriptively.
        acceptance: dict[str, Any] | None = None
        if smoke:
            ticker = companies[0]
            row = rows[ticker]
            position = int(row["instruction_last_position"])
            tensor = torch.tensor(
                [scoring_ids(tokenizer, row["formatted"])], dtype=torch.long, device=device
            )
            buy_id, sell_id = answer_token_ids(tokenizer, row["formatted"] + DECISION_PREFIX)
            final_layer = int(model.n_layers) - 1
            non_final = [layer for layer in layers if layer != final_layer]
            if not non_final:
                raise ValueError("phase D grid has no non-final layer for the derivative acceptance")
            probe = non_final[0]
            structural_zero = True
            if final_layer in layers:
                vector = grads[final_layer][ticker]
                structural_zero = bool((vector == 0).all())
            nonzero_norm = float(grads[probe][ticker].abs().sum())
            with mlp_all_positions_derivative(model, probe) as partition:
                check_margin = differentiable_margin(model, tensor, buy_id, sell_id)
                check_margin.backward()
                all_positions = partition[probe]
            del check_margin
            with mlp_summed_derivatives(model, [probe]) as summed:
                check_margin = differentiable_margin(model, tensor, buy_id, sell_id)
                check_margin.backward()
                summed_vector = summed[probe]
            del check_margin
            partition_sum = all_positions.sum(dim=0)
            norm_a = float(partition_sum.norm())
            norm_b = float(summed_vector.norm())
            partition_rel_err = (
                float((partition_sum - summed_vector).norm() / max(norm_a, norm_b))
                if max(norm_a, norm_b) > 0
                else 0.0
            )
            g_position = grads[probe][ticker]
            sign_agreement = float(
                (torch.sign(g_position) == torch.sign(summed_vector)).float().mean()
            )
            acceptance = {
                "probe_layer": probe,
                "structural_zero_final_layer": structural_zero,
                "nonfinal_nonzero_norm": nonzero_norm,
                "partition_rel_err": partition_rel_err,
                "partition_tolerance": D2_PARTITION_TOLERANCE,
                "sign_agreement_descriptive": sign_agreement,
                "pass": (
                    structural_zero
                    and nonzero_norm > 0.0
                    and partition_rel_err <= D2_PARTITION_TOLERANCE
                ),
            }
            if not acceptance["pass"]:
                raise ValueError(f"D2 derivative acceptance failed: {acceptance}")
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_d_forward_d2",
                "n_records": count,
                "layers": layers,
                "companies": companies,
                "n_companies": len(companies),
                "all_vectors_finite": True,
                "derivative_acceptance": acceptance,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_d_forward_d2", stage="forward_d2", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_d_forward_d2_metadata", stage="forward_d2", role="output"
        )
        stage.count(count)


def _phase_d_analyze(
    run: ArtifactRun,
    rows: dict[str, dict],
    *,
    layers: list[int],
    smoke: bool,
) -> dict:
    d1_records = read_jsonl(run.run_directory / "forward_d1" / "records.jsonl")
    d2_records = read_jsonl(run.run_directory / "forward_d2" / "records.jsonl")
    curves: dict[str, dict[int, dict]] = {}
    for component in ("mlp", "attn"):
        curves[component] = {}
        for layer in layers:
            component_records = [
                r for r in d1_records if r["component"] == component and r["layer"] == layer
            ]
            if not component_records:
                raise ValueError(f"missing directions for {component} L{layer}")
            deltas = [float(r["toward_source_delta_m"]) for r in component_records]
            transfers = [float(r["normalized_transfer"]) for r in component_records]
            ci = bootstrap_ci(deltas) if len(deltas) >= 4 else None
            curves[component][layer] = {
                "mean_toward_source_delta_m": statistics.fmean(deltas),
                "mean_normalized_transfer": statistics.fmean(transfers),
                "delta_m_ci_95": list(ci) if ci is not None else None,
                "n_directions": len(deltas),
            }
    sector_of = {t: rows[t]["sector"] for t in rows}
    companies = sorted(set(TOP_GROUP) | set(BOTTOM_GROUP))
    gate = evaluate_gate_d(d1_records, layers=layers, sector_of=sector_of, companies=companies)
    final_sanity: dict[str, dict] = {}
    for component in ("final_mlp", "final_attn"):
        component_records = [r for r in d1_records if r["component"] == component]
        if not component_records:
            raise ValueError(f"missing final position sanity records for {component}")
        deltas = [float(r["toward_source_delta_m"]) for r in component_records]
        transfers = [float(r["normalized_transfer"]) for r in component_records]
        ci = bootstrap_ci(deltas) if len(deltas) >= 4 else None
        final_sanity[component] = {
            "mean_toward_source_delta_m": statistics.fmean(deltas),
            "mean_normalized_transfer": statistics.fmean(transfers),
            "delta_m_ci_95": list(ci) if ci is not None else None,
            "n_directions": len(deltas),
        }
    d2_by_layer = {int(r["layer"]): r for r in d2_records}
    if sorted(d2_by_layer) != sorted(layers):
        raise ValueError(f"D2 layer set mismatch: {sorted(d2_by_layer)} vs {sorted(layers)}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "entity_to_dial_d_analysis",
        "n_records_d1": len(d1_records),
        "n_records_d2": len(d2_records),
        "layers": layers,
        "curves": curves,
        "gate_d": gate,
        "final_position_sanity": final_sanity,
        "d2_descriptive": {str(k): d2_by_layer[k] for k in sorted(d2_by_layer)},
        "smoke": smoke,
        "raw_runtime_payloads": False,
    }
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    with run.stage("analyze") as stage:
        write_json(path, summary, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_d_analysis_metadata",
                "gate_d_status": gate["status"],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_d_analysis", stage="analyze", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_d_analysis_metadata", stage="analyze", role="output"
        )
        stage.count(len(curves["mlp"]))
    return summary


def run_phase_d(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    phase2b_run: str | Path,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    phase2b_run = Path(phase2b_run)
    directions = (
        [tuple(d) for d in SMOKE_DIRECTIONS] if smoke else frozen_directions(phase2b_run)
    )
    layers = list(SMOKE_D_LAYERS if smoke else PHASE_D_LAYERS)
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        rows, pure = _phase_d_prepare(
            run, tokenizer,
            phase2a_run=phase2a_run, phase2b_run=phase2b_run,
            directions=directions, smoke=smoke,
        )
        _phase_d_forward_d1(run, model_path, rows, pure, directions, layers=layers)
        _phase_d_forward_d2(run, model_path, rows, pure, layers=layers, smoke=smoke)
        _phase_d_analyze(run, rows, layers=layers, smoke=smoke)
        run.finalize(
            required_stages={"prepare", "forward_d1", "forward_d2", "analyze"}
        )
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


# ── Phase E ───────────────────────────────────────────────────────────────────


def _read_full_ref(phase2b_run: Path, layers: Sequence[int]) -> dict[str, dict[int, dict[str, float]]]:
    """Per-direction 2B instruction-span full-residual reference (ΔM, T).

    Reads the frozen 2B sweep archive; fails closed if any direction/layer
    pair is missing.
    """
    records = read_jsonl(phase2b_run / "sweep" / "records.jsonl")
    wanted = {int(layer) for layer in layers}
    ref: dict[str, dict[int, dict[str, float]]] = {}
    for r in records:
        if r.get("span") != "instruction" or int(r["layer"]) not in wanted:
            continue
        ref.setdefault(r["direction"], {})[int(r["layer"])] = {
            "toward_delta_m": float(r["toward_source_delta_m"]),
            "t": float(r["normalized_transfer"]),
        }
    for direction in sorted(ref):
        missing = sorted(wanted - set(ref[direction]))
        if missing:
            raise ValueError(f"2B full reference incomplete for {direction}: layers {missing}")
    return ref


def _phase_e_prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    phase2a_run: Path,
    phase2b_run: Path,
    directions: list[tuple[str, str]],
    smoke: bool,
) -> tuple[dict[str, dict], dict[str, float], dict[str, dict[int, dict[str, float]]]]:
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        frozen = frozen_directions(phase2b_run)
        pure = pure_entity_margins(phase2a_run)
        cross_check_stored_margins(phase2a_run, pure)
        gap = verify_group_gap(pure)
        canon = _canonical_2a_rows(phase2a_run)
        tickers = sorted(canon)
        rows = []
        for ticker in tickers:
            row = dict(canon[ticker])
            row["pure_entity_margin"] = pure[ticker]
            row["instruction_last_position"] = int(row["instruction_span"][1]) - 1
            rows.append(row)
        # 1:1 alignment invariant (protocol §4.1): equal span length,
        # offset-parallel; the patch uses the offset-preserving mapping
        # (identity-by-offset when starts coincide).
        span_lengths = {
            int(r["instruction_span"][1]) - int(r["instruction_span"][0]) for r in rows
        }
        if len(span_lengths) != 1:
            raise ValueError(
                f"instruction span lengths not 1:1 aligned across companies: "
                f"{sorted(span_lengths)}"
            )
        instruction_span_length = span_lengths.pop()
        # 2B full-residual reference: L11–18 (smoke acceptance band at
        # L12/L15; incremental ΔT reference for the E1 carryover curve).
        ref_layers = list(range(11, 19))
        full_ref = _read_full_ref(phase2b_run, ref_layers)
        for direction in frozen:
            name = f"{direction[0]}->{direction[1]}"
            if name not in full_ref:
                raise ValueError(f"2B full reference missing direction {name}")
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_e_provenance",
            "protocol": PROTOCOL_E,
            "protocol_rev": PROTOCOL_E_REV,
            "smoke": smoke,
            "raw_runtime_payloads": False,
            "phase2a_run": _verify_upstream(
                phase2a_run,
                {
                    "prepare/prompts.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "forward/results.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "analyze/summary.json": None,
                },
            ),
            "phase2b_run": _verify_upstream(
                phase2b_run, {"pairs/directions.json": None, "sweep/records.jsonl": 1024}
            ),
            "frozen_groups": {"top": sorted(TOP_GROUP), "bottom": sorted(BOTTOM_GROUP)},
            "directions_frozen": [list(d) for d in frozen],
            "directions": [list(d) for d in directions],
            "instruction_span_length": instruction_span_length,
            "pure_entity_margins": {t: pure[t] for t in sorted(pure)},
            "group_gap": gap,
            "group_gap_threshold": GROUP_GAP_MIN,
            "full_ref_layers": ref_layers,
        }
        rows_path = out_dir / "rows.jsonl"
        count = write_jsonl(rows_path, rows, overwrite=True)
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_json(
            out_dir / "full_ref.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_e_full_ref",
                "source_run": str(phase2b_run),
                "span": "instruction",
                "reference": {
                    d: {str(l): v for l, v in sorted(layers_map.items())}
                    for d, layers_map in sorted(full_ref.items())
                },
            },
            overwrite=True,
        )
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_e_prepare",
                "n_rows": count,
                "instruction_span_length": instruction_span_length,
                "group_gap": gap,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            rows_path, artifact_type="entity_to_dial_e_prepare", stage="prepare", role="output", record_count=count
        )
        run.manifest.register_artifact(
            out_dir / "provenance.json", artifact_type="entity_to_dial_e_provenance", stage="prepare", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "full_ref.json", artifact_type="entity_to_dial_e_full_ref", stage="prepare", role="output"
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_e_prepare_metadata", stage="prepare", role="output"
        )
        stage.count(count)
    return {r["ticker"]: r for r in rows}, pure, full_ref


def _dial_patched_margin(
    model: Any,
    input_tensor: torch.Tensor,
    hook_ctx: Any,
    final_layer: int,
    buy_id: int,
    sell_id: int,
) -> float:
    """Margin forward under an MLP-input hook context (no residual transform)."""
    with hook_ctx:
        residual = record_residuals(model, input_tensor, [final_layer])[final_layer]
    log_probs = fp32_next_token_log_probs(model, residual[:, -1, :])
    return margin_from_log_probs(log_probs, buy_id, sell_id)


def _phase_e_forward_e1(
    run: ArtifactRun,
    model_path: str,
    rows: dict[str, dict],
    pure: dict[str, float],
    directions: list[tuple[str, str]],
    *,
    layers: list[int],
    full_ref: dict[str, dict[int, dict[str, float]]],
    smoke: bool,
) -> dict[str, dict]:
    """E1 dual-block joint patch + full-swap reference arm (no-grad), §4.2."""
    out_dir = run.run_directory / "forward_e1"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    e1_ctx: dict[str, dict] = {}
    with run.stage("forward_e1") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        final_layer = int(model.n_layers) - 1
        capture_layers = sorted(set(list(layers) + [final_layer, E2_LAYER]))
        records: list[dict] = []
        for index, (source_ticker, target_ticker) in enumerate(directions):
            source_row, target_row = rows[source_ticker], rows[target_ticker]
            m_source, m_target = pure[source_ticker], pure[target_ticker]
            direction = f"{source_ticker}->{target_ticker}"
            source_tensor = torch.tensor(
                [scoring_ids(tokenizer, source_row["formatted"])], dtype=torch.long, device=device
            )
            target_tensor = torch.tensor(
                [scoring_ids(tokenizer, target_row["formatted"])], dtype=torch.long, device=device
            )
            buy_id, sell_id = answer_token_ids(
                tokenizer, target_row["formatted"] + DECISION_PREFIX
            )
            source_span = tuple(int(p) for p in source_row["instruction_span"])
            target_span = tuple(int(p) for p in target_row["instruction_span"])
            mapping = nearest_position_mapping(source_span, target_span)
            if not mapping:
                raise ValueError("empty instruction span mapping")
            identity = identity_mapping(target_span)
            with dial_value_capture(
                model, DIAL_LAYER, DIAL_NEURON, range(source_span[0], source_span[1])
            ) as dial_src:
                source_states = record_block_states(model, source_tensor, capture_layers)
            with dial_value_capture(
                model, DIAL_LAYER, DIAL_NEURON, range(target_span[0], target_span[1])
            ) as dial_tgt:
                target_states = record_block_states(model, target_tensor, capture_layers)
            live = _live_margin(model, target_states[final_layer]["post"], buy_id, sell_id)

            e1_ctx[direction] = {
                "delta_s": stack_state_difference(
                    state_difference_rows(
                        source_states[E2_LAYER]["post"],
                        target_states[E2_LAYER]["post"],
                        mapping=mapping,
                    )
                ).cpu(),
                "dial_src": dict(dial_src),
                "dial_tgt": dict(dial_tgt),
                "source_post_l15": source_states[E2_LAYER]["post"].detach(),
                "target_post_l15": target_states[E2_LAYER]["post"].detach(),
                "mapping": mapping,
                "full_dm_l15": None,
                "live_margin": live,
            }

            for layer in layers:
                joint = make_joint_transform(
                    source_states[layer]["pre"],
                    source_states[layer]["post"],
                    target_states[layer]["pre"],
                    target_states[layer]["post"],
                    mapping=mapping,
                )
                full = make_full_transform(source_states[layer]["post"], mapping=mapping)
                joint_patched = margin_from_log_probs(
                    _patched_final_margin(model, target_tensor, {layer: joint}), buy_id, sell_id
                )
                full_patched = margin_from_log_probs(
                    _patched_final_margin(model, target_tensor, {layer: full}), buy_id, sell_id
                )
                full_excluded = abs(toward_source_delta(full_patched, m_source, m_target)) < RATIO_MIN_FULL_DM
                records.append(
                    _record(
                        phase="e1", direction=direction, layer=layer,
                        key_field="arm", key_value="joint", patched=joint_patched,
                        m_source=m_source, m_target=m_target,
                        live_target_margin=live,
                        excluded=full_excluded,
                    )
                )
                records.append(
                    _record(
                        phase="e1", direction=direction, layer=layer,
                        key_field="arm", key_value="full", patched=full_patched,
                        m_source=m_source, m_target=m_target,
                        live_target_margin=live,
                        full_t_ref=normalized_transfer(full_patched, m_source, m_target),
                        excluded=full_excluded,
                    )
                )
                if layer == E2_LAYER:
                    e1_ctx[direction]["full_dm_l15"] = toward_source_delta(
                        full_patched, m_source, m_target
                    )
                # Self no-ops (bit-exact by construction: zero delta / self copy).
                self_joint = make_joint_transform(
                    target_states[layer]["pre"],
                    target_states[layer]["post"],
                    target_states[layer]["pre"],
                    target_states[layer]["post"],
                    mapping=identity,
                )
                self_joint_patched = margin_from_log_probs(
                    _patched_final_margin(model, target_tensor, {layer: self_joint}),
                    buy_id, sell_id,
                )
                _check_noop(self_joint_patched, live, f"phase E L{layer}/joint {direction}")
                self_full = make_full_transform(target_states[layer]["post"], mapping=identity)
                self_full_patched = margin_from_log_probs(
                    _patched_final_margin(model, target_tensor, {layer: self_full}),
                    buy_id, sell_id,
                )
                _check_noop(self_full_patched, live, f"phase E L{layer}/full {direction}")
                for arm, patched in (("joint_noop", self_joint_patched), ("full_noop", self_full_patched)):
                    records.append(
                        _record(
                            phase="e1", direction=direction, layer=layer,
                            key_field="arm", key_value=arm, patched=patched,
                            m_source=m_source, m_target=m_target,
                            live_target_margin=live,
                            noop_delta_m=patched - live,
                        )
                    )
            print(f"  direction {index + 1}/{len(directions)}: {direction}", flush=True)
        write_jsonl(path, records, overwrite=True)
        # Smoke acceptance #2: in-run full arm reproduces the 2B archive
        # per direction and per smoke layer (direction-matched; bit-exact
        # transform + deterministic forward, so the expected difference is
        # 0.0 and the band is a safety net).
        if smoke:
            for r in records:
                if r["arm"] != "full":
                    continue
                ref = full_ref.get(r["direction"], {}).get(r["layer"])
                if ref is None:
                    raise ValueError(f"no 2B reference for {r['direction']} L{r['layer']}")
                diff = r["toward_source_delta_m"] - ref["toward_delta_m"]
                if abs(diff) > E1_SMOKE_FULL_BAND:
                    raise ValueError(
                        f"smoke full-arm mismatch {r['direction']} L{r['layer']}: "
                        f"in-run {r['toward_source_delta_m']:+.4f} vs 2B archive "
                        f"{ref['toward_delta_m']:+.4f} (|diff| {abs(diff):.4f} > {E1_SMOKE_FULL_BAND})"
                    )
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_e_forward_e1",
                "n_records": len(records),
                "directions": [f"{s}->{t}" for s, t in directions],
                "layers": layers,
                "arms": ["joint", "full", "joint_noop", "full_noop"],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_e_forward_e1", stage="forward_e1", role="output", record_count=len(records)
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_e_forward_e1_metadata", stage="forward_e1", role="output"
        )
        stage.count(len(records))
    return e1_ctx


def _phase_e_forward_e2(
    run: ArtifactRun,
    model_path: str,
    rows: dict[str, dict],
    pure: dict[str, float],
    directions: list[tuple[str, str]],
    *,
    e1_ctx: dict[str, dict],
    smoke: bool,
) -> None:
    """E2 state-direction patch (PCA sweep + dial transplant), §4.3."""
    out_dir = run.run_directory / "forward_e2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    k_values = list(SMOKE_E2_K_SWEEP if smoke else E2_K_SWEEP)
    with run.stage("forward_e2") as stage:
        # PCA basis from the 8 (or 2 smoke) direction state differences.
        deltas = [e1_ctx[f"{s}->{t}"]["delta_s"] for s, t in directions]
        basis, singular = pca_state_directions(deltas, E2_PCA_DIM)
        basis2, singular2 = pca_state_directions(deltas, E2_PCA_DIM)
        # Smoke acceptance #5 (applied to all runs, fail-closed): SVD
        # determinism.
        if not (torch.equal(basis, basis2) and torch.equal(singular, singular2)):
            raise RuntimeError("PCA SVD is not deterministic on identical input")
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        final_layer = int(model.n_layers) - 1
        footprint = (
            None
            if smoke
            else dial_footprint_direction(model, DIAL_LAYER, DIAL_NEURON)
        )
        records: list[dict] = []
        for index, (source_ticker, target_ticker) in enumerate(directions):
            ctx = e1_ctx[f"{source_ticker}->{target_ticker}"]
            m_source, m_target = pure[source_ticker], pure[target_ticker]
            direction = f"{source_ticker}->{target_ticker}"
            full_dm = ctx["full_dm_l15"]
            if full_dm is None:
                raise ValueError(f"missing L15 full-arm delta for {direction}")
            excluded = abs(full_dm) < RATIO_MIN_FULL_DM
            target_tensor = torch.tensor(
                [scoring_ids(tokenizer, rows[target_ticker]["formatted"])],
                dtype=torch.long, device=device,
            )
            buy_id, sell_id = answer_token_ids(
                tokenizer, rows[target_ticker]["formatted"] + DECISION_PREFIX
            )
            mapping = ctx["mapping"]
            identity = {p: p for p in mapping}  # self-source over the same positions
            live = ctx["live_margin"]

            def ratio(patched: float) -> float | None:
                dm = toward_source_delta(patched, m_source, m_target)
                return None if excluded else dm / full_dm

            # E2-2: PCA sweep (projected arms + exact full swap).
            for k in k_values:
                arm = f"pca_k{k}"
                basis_k = basis[:, :k]
                transform = make_projected_transform(
                    ctx["source_post_l15"], ctx["target_post_l15"], basis_k, mapping=mapping
                )
                patched = margin_from_log_probs(
                    _patched_final_margin(model, target_tensor, {E2_LAYER: transform}),
                    buy_id, sell_id,
                )
                records.append(
                    _record(
                        phase="e2", direction=direction, layer=E2_LAYER,
                        key_field="arm", key_value=arm, patched=patched,
                        m_source=m_source, m_target=m_target,
                        live_target_margin=live,
                        effect_ratio=ratio(patched), excluded=excluded,
                    )
                )
            full_transform = make_full_transform(ctx["source_post_l15"], mapping=mapping)
            full_patched = margin_from_log_probs(
                _patched_final_margin(model, target_tensor, {E2_LAYER: full_transform}),
                buy_id, sell_id,
            )
            full_dm_inrun = toward_source_delta(full_patched, m_source, m_target)
            # Consistency (smoke acceptance #4, applied to all runs): the
            # exact-swap arm must reproduce the E1 L15 full arm bit-exactly.
            if full_dm_inrun != full_dm:
                raise RuntimeError(
                    f"E2 pca_full inconsistent with E1 L15 full arm for {direction}: "
                    f"{full_dm_inrun} != {full_dm}"
                )
            records.append(
                _record(
                    phase="e2", direction=direction, layer=E2_LAYER,
                    key_field="arm", key_value="pca_full", patched=full_patched,
                    m_source=m_source, m_target=m_target,
                    live_target_margin=live,
                    effect_ratio=None if excluded else 1.0, excluded=excluded,
                )
            )
            # pca_full self no-op (self-source exact copy, bit-exact).
            self_full = make_full_transform(ctx["target_post_l15"], mapping=identity)
            self_patched = margin_from_log_probs(
                _patched_final_margin(model, target_tensor, {E2_LAYER: self_full}),
                buy_id, sell_id,
            )
            _check_noop(self_patched, live, f"phase E L{E2_LAYER}/pca_full_noop {direction}")
            records.append(
                _record(
                    phase="e2", direction=direction, layer=E2_LAYER,
                    key_field="arm", key_value="pca_full_noop", patched=self_patched,
                    m_source=m_source, m_target=m_target,
                    live_target_margin=live,
                    effect_ratio=None, excluded=excluded,
                    noop_delta_m=self_patched - live,
                )
            )
            # E2-3: dial channel transplant (position-restricted, MLP space).
            dial_deltas = {
                target_pos: ctx["dial_src"][mapping[target_pos]] - ctx["dial_tgt"][target_pos]
                for target_pos in sorted(mapping)
            }
            dial_patched = _dial_patched_margin(
                model, target_tensor,
                dial_channel_transplant(model, DIAL_LAYER, DIAL_NEURON, dial_deltas),
                final_layer, buy_id, sell_id,
            )
            records.append(
                _record(
                    phase="e2", direction=direction, layer=E2_LAYER,
                    key_field="arm", key_value="dial_transplant", patched=dial_patched,
                    m_source=m_source, m_target=m_target,
                    live_target_margin=live,
                    effect_ratio=ratio(dial_patched), excluded=excluded,
                    dial_delta_per_position=[dial_deltas[p] for p in sorted(dial_deltas)],
                )
            )
            dial_noop_patched = _dial_patched_margin(
                model, target_tensor,
                dial_channel_transplant(model, DIAL_LAYER, DIAL_NEURON, {}),
                final_layer, buy_id, sell_id,
            )
            _check_noop(dial_noop_patched, live, f"phase E L{E2_LAYER}/dial_noop {direction}")
            records.append(
                _record(
                    phase="e2", direction=direction, layer=E2_LAYER,
                    key_field="arm", key_value="dial_noop", patched=dial_noop_patched,
                    m_source=m_source, m_target=m_target,
                    live_target_margin=live,
                    effect_ratio=None, excluded=excluded,
                    noop_delta_m=dial_noop_patched - live,
                )
            )
            # E2-4: dial residual footprint (descriptive; formal only).
            if footprint is not None:
                projected = project_delta(ctx["delta_s"], footprint.view(-1, 1))
                positions = sorted(mapping)
                foot_transform = make_delta_transform({pos: projected[i] for i, pos in enumerate(positions)})
                foot_patched = margin_from_log_probs(
                    _patched_final_margin(model, target_tensor, {E2_LAYER: foot_transform}),
                    buy_id, sell_id,
                )
                records.append(
                    _record(
                        phase="e2", direction=direction, layer=E2_LAYER,
                        key_field="arm", key_value="footprint", patched=foot_patched,
                        m_source=m_source, m_target=m_target,
                        live_target_margin=live,
                        effect_ratio=ratio(foot_patched), excluded=excluded,
                    )
                )
            print(f"  direction {index + 1}/{len(directions)}: {direction}", flush=True)
        write_jsonl(path, records, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_e_forward_e2",
                "n_records": len(records),
                "directions": [f"{s}->{t}" for s, t in directions],
                "layer": E2_LAYER,
                "k_values": k_values + ["full"],
                "pca_dim": E2_PCA_DIM,
                "singular_values": [float(v) for v in singular.tolist()],
                "basis_vectors": [[float(v) for v in row] for row in basis.T.tolist()],
                "svd_deterministic": True,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_e_forward_e2", stage="forward_e2", role="output", record_count=len(records)
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="entity_to_dial_e_forward_e2_metadata", stage="forward_e2", role="output"
        )
        stage.count(len(records))


def _phase_e_analyze(
    run: ArtifactRun,
    rows: dict[str, dict],
    *,
    layers: list[int],
    smoke: bool,
) -> dict:
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("analyze") as stage:
        e1_records = read_jsonl(run.run_directory / "forward_e1" / "records.jsonl")
        e2_records = read_jsonl(run.run_directory / "forward_e2" / "records.jsonl")
        e2_meta = json.loads(
            (run.run_directory / "forward_e2" / "metadata.json").read_text(encoding="utf-8")
        )
        full_ref = json.loads(
            (run.run_directory / "prepare" / "full_ref.json").read_text(encoding="utf-8")
        )["reference"]
        gate = evaluate_gate_e(
            e1_records,
            e2_records,
            layers=layers,
            gate_layer=E2_LAYER,
            k_values=list(SMOKE_E2_K_SWEEP if smoke else E2_K_SWEEP),
            min_full_dm=RATIO_MIN_FULL_DM,
            e1_min=GATE_E1_RATIO_MIN,
            e2b_min=GATE_E2B_RATIO_MIN,
            falsifier_max=GATE_E2B_FALSIFIER_MAX,
            smoke=smoke,
        )
        # Descriptive carryover reference: 2B archive 8-direction mean
        # incremental ΔM, full(L) − full(L−1).
        if gate.get("status") == "evaluated":
            ref_layers = set(full_ref.get(next(iter(full_ref), ()), {}))
            for layer_key, curve in gate["e1_curves"].items():
                layer, prev = int(layer_key), int(layer_key) - 1
                if str(layer) in ref_layers and str(prev) in ref_layers:
                    now = statistics.fmean(
                        full_ref[d][str(layer)]["toward_delta_m"] for d in full_ref
                    )
                    before = statistics.fmean(
                        full_ref[d][str(prev)]["toward_delta_m"] for d in full_ref
                    )
                    curve["incremental_dm_ref"] = now - before
                    curve["full_dm_ref_mean"] = now
        if gate.get("status") == "evaluated" and "8" in gate["e2a_curve"]:
            gate["e2a_curve"]["8"]["target"] = E2A_RATIO_TARGET
        summary = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_e_summary",
            "smoke": smoke,
            "gate_e1": gate.get("gate_e1"),
            "e1_curves": gate.get("e1_curves"),
            "gate_e2b": gate.get("gate_e2b"),
            "e2a_curve": gate.get("e2a_curve"),
            "e2_pca_full": gate.get("e2_pca_full"),
            "e2_footprint": gate.get("e2_footprint"),
            "gate_e": gate.get("gate") if gate.get("status") == "evaluated" else gate,
            "pca_singular_values": e2_meta["singular_values"],
            "pca_basis_vectors": e2_meta["basis_vectors"],
            "raw_runtime_payloads": False,
        }
        path = out_dir / "summary.json"
        write_json(path, summary, overwrite=True)
        run.manifest.register_artifact(
            path, artifact_type="entity_to_dial_e_summary", stage="analyze", role="output"
        )
        stage.count(1)
    return summary


def run_phase_e(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    phase2b_run: str | Path,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    phase2b_run = Path(phase2b_run)
    directions = (
        [tuple(name.split("->")) for name in SMOKE_E_DIRECTIONS]
        if smoke
        else frozen_directions(phase2b_run)
    )
    layers = list(SMOKE_E_LAYERS if smoke else PHASE_E_LAYERS)
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        rows, pure, full_ref = _phase_e_prepare(
            run, tokenizer,
            phase2a_run=phase2a_run, phase2b_run=phase2b_run,
            directions=directions, smoke=smoke,
        )
        e1_ctx = _phase_e_forward_e1(
            run, model_path, rows, pure, directions,
            layers=layers, full_ref=full_ref, smoke=smoke,
        )
        _phase_e_forward_e2(
            run, model_path, rows, pure, directions,
            e1_ctx=e1_ctx, smoke=smoke,
        )
        _phase_e_analyze(run, rows, layers=layers, smoke=smoke)
        run.finalize(
            required_stages={"prepare", "forward_e1", "forward_e2", "analyze"}
        )
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


# ── Phase F ───────────────────────────────────────────────────────────────────

# e-01 record counts (upstream verification, protocol §10).
N_E01_E1_RECORDS = 224
N_E01_E2_RECORDS = 72
F_ANCHOR_TICKER = "NSC"


def _anonymous_row(tokenizer: Any, named_row: dict[str, Any]) -> dict[str, Any]:
    """Anonymous probe row (Phase F §4.1): header substitution + instruction span.

    Reuses the Phase A/C ``anonymous_prompt`` construction; the instruction
    region maps to tokens with the same char-span → token_span mechanic as
    the 2A ``resolve_row``.
    """
    prompt = anonymous_prompt(named_row["prompt"], named_row["ticker"], named_row["name"])
    formatted = format_prompt(
        tokenizer, prompt, use_chat_template=True, enable_thinking=False
    )
    prompt_ids = input_ids(tokenizer, formatted, add_special_tokens=True)
    body_start = formatted.find(prompt)
    if body_start < 0:
        raise ValueError("anonymous prompt body not found in formatted text")
    char_start, char_end = instruction_char_span(prompt)
    instruction_span = token_span(
        tokenizer,
        formatted,
        body_start + char_start,
        body_start + char_end,
        add_special_tokens=True,
    )
    if instruction_span is None:
        raise ValueError("could not map anonymous instruction span to tokens")
    return {
        "id": "anonymous",
        "prompt_type": "anon",
        "prompt": prompt,
        "formatted": formatted,
        "prompt_ids": list(prompt_ids),
        "instruction_span": [int(p) for p in instruction_span],
        "final_position": len(prompt_ids) - 1,
        "source_anchor": named_row["ticker"],
    }


def _phase_f_prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    phase2a_run: Path,
    phase2b_run: Path,
    phasee_run: Path,
    directions: list[tuple[str, str]],
    smoke: bool,
) -> tuple[dict[str, dict], dict[str, float], torch.Tensor, torch.Tensor, dict[str, dict[str, float]], dict[str, Any]]:
    """Phase F prepare: upstream + PCA basis + anonymous identity check, §4.1."""
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        frozen = frozen_directions(phase2b_run)
        pure = pure_entity_margins(phase2a_run)
        cross_check_stored_margins(phase2a_run, pure)
        gap = verify_group_gap(pure)
        canon = _canonical_2a_rows(phase2a_run)
        tickers = sorted(canon)
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            row = dict(canon[ticker])
            row["pure_entity_margin"] = pure[ticker]
            row["instruction_last_position"] = int(row["instruction_span"][1]) - 1
            rows.append(row)
        span_lengths = {
            int(r["instruction_span"][1]) - int(r["instruction_span"][0]) for r in rows
        }
        if len(span_lengths) != 1:
            raise ValueError(
                f"instruction span lengths not 1:1 aligned across companies: {sorted(span_lengths)}"
            )
        instruction_span_length = span_lengths.pop()
        # PCA basis from the e-01 artifact (fail-closed provenance, §4.1).
        e01_summary = phasee_run / "analyze" / "summary.json"
        basis, singular = load_pca_basis(e01_summary, expected_k=E2_PCA_DIM)
        # e-01 per-direction reference (consistency check, §4.2).
        e01_ref: dict[str, dict[str, float]] = {}
        for r in read_jsonl(phasee_run / "forward_e1" / "records.jsonl"):
            if r["phase"] == "e1" and r["layer"] == E2_LAYER and r["arm"] == "full":
                e01_ref.setdefault(r["direction"], {})["full"] = float(r["toward_source_delta_m"])
        arm_map = {"pca_k1": "v1", "pca_k8": "k8", "dial_transplant": "dial"}
        for r in read_jsonl(phasee_run / "forward_e2" / "records.jsonl"):
            if r["phase"] == "e2" and r["arm"] in arm_map:
                e01_ref.setdefault(r["direction"], {})[arm_map[r["arm"]]] = float(r["toward_source_delta_m"])
        for source, target in frozen:
            name = f"{source}->{target}"
            have = set(e01_ref.get(name, {}))
            if have != {"full", "v1", "k8", "dial"}:
                raise ValueError(f"e-01 reference incomplete for {name}: {sorted(have)}")
        # Anonymous prompt: NSC-derived anchor + 16/16 identity check.
        anon = _anonymous_row(tokenizer, canon[F_ANCHOR_TICKER])
        anon_string = anon["prompt"]
        for ticker in tickers:
            candidate = anonymous_prompt(canon[ticker]["prompt"], ticker, canon[ticker]["name"])
            if candidate != anon_string:
                raise ValueError(
                    f"anonymous prompt for {ticker} differs from the anchor string "
                    "(template not unified; protocol §4.1 requires a new version)"
                )
        anon_sha = hashlib.sha256(anon_string.encode("utf-8")).hexdigest()
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_f_provenance",
            "protocol": PROTOCOL_F,
            "protocol_rev": PROTOCOL_F_REV,
            "smoke": smoke,
            "raw_runtime_payloads": False,
            "phase2a_run": _verify_upstream(
                phase2a_run,
                {
                    "prepare/prompts.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "forward/results.jsonl": N_CANONICAL_ROWS * N_2A_VARIANTS,
                    "analyze/summary.json": None,
                },
            ),
            "phase2b_run": _verify_upstream(
                phase2b_run, {"pairs/directions.json": None, "sweep/records.jsonl": 1024}
            ),
            "phasee_run": _verify_upstream(
                phasee_run,
                {
                    "forward_e1/records.jsonl": N_E01_E1_RECORDS,
                    "forward_e2/records.jsonl": N_E01_E2_RECORDS,
                    "analyze/summary.json": None,
                },
            ),
            "pca_basis": {
                "source_run": phasee_run.name,
                "k": int(basis.shape[1]),
                "d": int(basis.shape[0]),
                "orthonormal_atol": 1e-6,
                "singular_range": [float(singular[0]), float(singular[-1])],
            },
            "anonymous_prompt_sha256": anon_sha,
            "anonymous_anchor": F_ANCHOR_TICKER,
            "anonymous_identity_check": f"{len(tickers)}/{len(tickers)} identical",
            "frozen_groups": {"top": sorted(TOP_GROUP), "bottom": sorted(BOTTOM_GROUP)},
            "directions_frozen": [list(d) for d in frozen],
            "directions": [list(d) for d in directions],
            "instruction_span_length": instruction_span_length,
            "pure_entity_margins": {t: pure[t] for t in sorted(pure)},
            "group_gap": gap,
            "group_gap_threshold": GROUP_GAP_MIN,
        }
        rows_path = out_dir / "rows.jsonl"
        count = write_jsonl(rows_path, rows + [anon], overwrite=True)
        write_json(
            out_dir / "e01_ref.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_f_e01_ref",
                "source_run": str(phasee_run),
                "layer": E2_LAYER,
                "reference": {d: {a: v for a, v in sorted(vs.items())} for d, vs in sorted(e01_ref.items())},
            },
            overwrite=True,
        )
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_f_prepare",
                "n_rows": count,
                "instruction_span_length": instruction_span_length,
                "group_gap": gap,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(rows_path, artifact_type="entity_to_dial_f_prepare", stage="prepare", role="output", record_count=count)
        run.manifest.register_artifact(out_dir / "provenance.json", artifact_type="entity_to_dial_f_provenance", stage="prepare", role="output")
        run.manifest.register_artifact(out_dir / "e01_ref.json", artifact_type="entity_to_dial_f_e01_ref", stage="prepare", role="output")
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="entity_to_dial_f_prepare_metadata", stage="prepare", role="output")
    rows_by_ticker = {r["ticker"]: r for r in rows}
    return rows_by_ticker, pure, basis, singular, e01_ref, anon


def _f1_record(
    *,
    direction: str,
    arm: str,
    patched: float,
    m_source: float,
    m_target: float,
    live: float,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "phase": "f1",
        "direction": direction,
        "arm": arm,
        "patched_margin": patched,
        "toward_source_delta_m": toward_source_delta(patched, m_source, m_target),
        "normalized_transfer": normalized_transfer(patched, m_source, m_target),
        "m_source": m_source,
        "m_target": m_target,
        "live_target_margin": live,
        "noop": arm.endswith("_noop"),
        **extra,
    }


def _phase_f_forward_f1(
    run: ArtifactRun,
    model_path: str,
    rows: dict[str, dict],
    pure: dict[str, float],
    directions: list[tuple[str, str]],
    *,
    basis: torch.Tensor,
    singular: torch.Tensor,
    e01_ref: dict[str, dict[str, float]],
    smoke: bool,
) -> dict[str, float]:
    """F1 dual-hook combined patch (L15, §4.2)."""
    out_dir = run.run_directory / "forward_f1"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with run.stage("forward_f1") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        final_layer = int(model.n_layers) - 1
        capture_layers = sorted({F_LAYER, final_layer})
        records: list[dict[str, Any]] = []
        per_direction_base: list[float] = []
        for index, (source_ticker, target_ticker) in enumerate(directions):
            source_row, target_row = rows[source_ticker], rows[target_ticker]
            m_source, m_target = pure[source_ticker], pure[target_ticker]
            direction = f"{source_ticker}->{target_ticker}"
            source_tensor = torch.tensor(
                [scoring_ids(tokenizer, source_row["formatted"])], dtype=torch.long, device=device
            )
            target_tensor = torch.tensor(
                [scoring_ids(tokenizer, target_row["formatted"])], dtype=torch.long, device=device
            )
            buy_id, sell_id = answer_token_ids(
                tokenizer, target_row["formatted"] + DECISION_PREFIX
            )
            source_span = tuple(int(p) for p in source_row["instruction_span"])
            target_span = tuple(int(p) for p in target_row["instruction_span"])
            mapping = nearest_position_mapping(source_span, target_span)
            identity = identity_mapping(target_span)
            if not mapping:
                raise ValueError("empty instruction span mapping")
            with dial_value_capture(
                model, F_LAYER, DIAL_NEURON, range(source_span[0], source_span[1])
            ) as dial_src:
                source_states = record_block_states(model, source_tensor, capture_layers)
            with dial_value_capture(
                model, F_LAYER, DIAL_NEURON, range(target_span[0], target_span[1])
            ) as dial_tgt:
                target_states = record_block_states(model, target_tensor, capture_layers)
            live = _live_margin(model, target_states[final_layer]["post"], buy_id, sell_id)
            positions = sorted(mapping)
            delta_s = stack_state_difference(
                state_difference_rows(
                    source_states[F_LAYER]["post"],
                    target_states[F_LAYER]["post"],
                    mapping=mapping,
                )
            )
            dial_deltas = {p: float(dial_src[mapping[p]] - dial_tgt[p]) for p in positions}
            # Push base input (B2): per-direction median_p |Δs(p)·v₁|.
            v1_coeffs = delta_s @ basis[:, 0].to(device=device, dtype=delta_s.dtype)
            per_direction_base.append(float(torch.median(v1_coeffs.abs().cpu()).item()))

            v1_transform = make_projected_transplant(delta_s, basis[:, :1], positions)
            k8_transform = make_projected_transplant(delta_s, basis[:, :8], positions)
            full_transform = make_full_transform(source_states[F_LAYER]["post"], mapping=mapping)
            zero_rows = {
                p: torch.zeros(delta_s.shape[1], dtype=torch.float32) for p in positions
            }

            full_patched = margin_from_log_probs(
                _patched_final_margin(model, target_tensor, {F_LAYER: full_transform}),
                buy_id, sell_id,
            )
            v1_patched = margin_from_log_probs(
                _patched_final_margin(model, target_tensor, {F_LAYER: v1_transform}),
                buy_id, sell_id,
            )
            k8_patched = margin_from_log_probs(
                _patched_final_margin(model, target_tensor, {F_LAYER: k8_transform}),
                buy_id, sell_id,
            )
            dial_patched = _dial_patched_margin(
                model, target_tensor,
                dial_channel_transplant(model, F_LAYER, DIAL_NEURON, dial_deltas),
                final_layer, buy_id, sell_id,
            )
            # Combined arm: dual-hook (v₁ residual + dial) in one forward.
            with dual_hook_interventions(
                model, F_LAYER, v1_transform, DIAL_NEURON, dial_deltas
            ) as fires:
                residual = record_residuals(model, target_tensor, [final_layer])[final_layer]
            combined_patched = margin_from_log_probs(
                fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
            )
            if smoke and (fires["residual"] != 1 or fires["dial"] != 1):
                raise ValueError(f"dual-hook fire count unexpected: {fires}")

            # No-ops (bit-exact expected).
            full_noop_patched = margin_from_log_probs(
                _patched_final_margin(
                    model, target_tensor,
                    {F_LAYER: make_full_transform(target_states[F_LAYER]["post"], mapping=identity)},
                ),
                buy_id, sell_id,
            )
            _check_noop(full_noop_patched, live, f"phase F F1/full_noop {direction}")
            zero_transform = make_delta_transform(zero_rows)
            v1_noop_patched = margin_from_log_probs(
                _patched_final_margin(model, target_tensor, {F_LAYER: zero_transform}),
                buy_id, sell_id,
            )
            _check_noop(v1_noop_patched, live, f"phase F F1/v1_noop {direction}")
            k8_noop_patched = margin_from_log_probs(
                _patched_final_margin(model, target_tensor, {F_LAYER: zero_transform}),
                buy_id, sell_id,
            )
            _check_noop(k8_noop_patched, live, f"phase F F1/k8_noop {direction}")
            dial_noop_patched = _dial_patched_margin(
                model, target_tensor,
                dial_channel_transplant(model, F_LAYER, DIAL_NEURON, {}),
                final_layer, buy_id, sell_id,
            )
            _check_noop(dial_noop_patched, live, f"phase F F1/dial_noop {direction}")
            with dual_hook_interventions(
                model, F_LAYER, zero_transform, DIAL_NEURON, {}
            ) as _:
                residual = record_residuals(model, target_tensor, [final_layer])[final_layer]
            combined_noop_patched = margin_from_log_probs(
                fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
            )
            _check_noop(combined_noop_patched, live, f"phase F F1/combined_noop {direction}")

            records.append(_f1_record(direction=direction, arm="full", patched=full_patched, m_source=m_source, m_target=m_target, live=live))
            records.append(_f1_record(direction=direction, arm="v1", patched=v1_patched, m_source=m_source, m_target=m_target, live=live))
            records.append(_f1_record(direction=direction, arm="k8", patched=k8_patched, m_source=m_source, m_target=m_target, live=live))
            records.append(_f1_record(direction=direction, arm="dial", patched=dial_patched, m_source=m_source, m_target=m_target, live=live, dial_delta_per_position=[dial_deltas[p] for p in positions]))
            records.append(_f1_record(direction=direction, arm="combined", patched=combined_patched, m_source=m_source, m_target=m_target, live=live, additivity_ratio=(
                toward_source_delta(combined_patched, m_source, m_target)
                / toward_source_delta(full_patched, m_source, m_target)
            ) if abs(toward_source_delta(full_patched, m_source, m_target)) >= RATIO_MIN_FULL_DM else None))
            records.append(_f1_record(direction=direction, arm="full_noop", patched=full_noop_patched, m_source=m_source, m_target=m_target, live=live, noop_delta_m=full_noop_patched - live))
            records.append(_f1_record(direction=direction, arm="v1_noop", patched=v1_noop_patched, m_source=m_source, m_target=m_target, live=live, noop_delta_m=v1_noop_patched - live))
            records.append(_f1_record(direction=direction, arm="k8_noop", patched=k8_noop_patched, m_source=m_source, m_target=m_target, live=live, noop_delta_m=k8_noop_patched - live))
            records.append(_f1_record(direction=direction, arm="dial_noop", patched=dial_noop_patched, m_source=m_source, m_target=m_target, live=live, noop_delta_m=dial_noop_patched - live))
            records.append(_f1_record(direction=direction, arm="combined_noop", patched=combined_noop_patched, m_source=m_source, m_target=m_target, live=live, noop_delta_m=combined_noop_patched - live))

            if smoke:
                # Composition property (acceptance #2): zeroing one side of the
                # dual hook reproduces the corresponding single arm bit-exactly.
                with dual_hook_interventions(
                    model, F_LAYER, v1_transform, DIAL_NEURON, {}
                ) as _:
                    residual = record_residuals(model, target_tensor, [final_layer])[final_layer]
                combined_dialzero = margin_from_log_probs(
                    fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
                )
                if combined_dialzero != v1_patched:
                    raise ValueError(
                        f"dual-hook composition violated (dial=0): {combined_dialzero} != v1 {v1_patched}"
                    )
                with dual_hook_interventions(
                    model, F_LAYER, zero_transform, DIAL_NEURON, dial_deltas
                ) as _:
                    residual = record_residuals(model, target_tensor, [final_layer])[final_layer]
                combined_v1zero = margin_from_log_probs(
                    fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
                )
                if combined_v1zero != dial_patched:
                    raise ValueError(
                        f"dual-hook composition violated (v1=0): {combined_v1zero} != dial {dial_patched}"
                    )
                records.append(_f1_record(direction=direction, arm="combined_dialzero", patched=combined_dialzero, m_source=m_source, m_target=m_target, live=live))
                records.append(_f1_record(direction=direction, arm="combined_v1zero", patched=combined_v1zero, m_source=m_source, m_target=m_target, live=live))
                # Cross-run consistency (acceptance #3, fail-closed in smoke).
                for arm, patched in (("full", full_patched), ("v1", v1_patched), ("k8", k8_patched), ("dial", dial_patched)):
                    diff = abs(
                        toward_source_delta(patched, m_source, m_target) - e01_ref[direction][arm]
                    )
                    if diff > F1_CONSISTENCY_TOLERANCE:
                        raise ValueError(
                            f"smoke consistency mismatch {direction}/{arm}: in-run vs e-01 "
                            f"|diff| {diff:.4f} > {F1_CONSISTENCY_TOLERANCE}"
                        )
            print(f"  direction {index + 1}/{len(directions)}: {direction}", flush=True)
        # Push base (B2): median over directions of the per-direction medians.
        push_base = float(torch.median(torch.tensor(per_direction_base)).item())
        write_jsonl(path, records, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_f_forward_f1",
                "n_records": len(records),
                "directions": [f"{s}->{t}" for s, t in directions],
                "layer": F_LAYER,
                "arms": ["full", "v1", "k8", "dial", "combined",
                          "full_noop", "v1_noop", "k8_noop", "dial_noop", "combined_noop"],
                "per_direction_push_base": {f"{s}->{t}": v for (s, t), v in zip(directions, per_direction_base)},
                "push_base": push_base,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="entity_to_dial_f_forward_f1", stage="forward_f1", role="output", record_count=len(records))
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="entity_to_dial_f_forward_f1_metadata", stage="forward_f1", role="output")
    return {"push_base": push_base}


def _phase_f_forward_f2(
    run: ArtifactRun,
    model_path: str,
    anon_row: dict[str, Any],
    *,
    basis: torch.Tensor,
    push_base: float,
    smoke: bool,
) -> None:
    """F2 neutral-context directional push (anonymous prompt, §4.3)."""
    out_dir = run.run_directory / "forward_f2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with run.stage("forward_f2") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        final_layer = int(model.n_layers) - 1
        anon_tensor = torch.tensor(
            [scoring_ids(tokenizer, anon_row["formatted"])], dtype=torch.long, device=device
        )
        buy_id, sell_id = answer_token_ids(
            tokenizer, anon_row["formatted"] + DECISION_PREFIX
        )
        positions = tuple(int(p) for p in anon_row["instruction_span"])
        clean_residual = record_residuals(model, anon_tensor, [final_layer])[final_layer]
        m_anon = margin_from_log_probs(
            fp32_next_token_log_probs(model, clean_residual[:, -1, :]), buy_id, sell_id
        )
        if not math.isfinite(m_anon):
            raise ValueError("non-finite anonymous clean margin")
        m_anon_band_ok = abs(m_anon - F2_M_ANON_REF) <= F2_M_ANON_BAND
        if smoke and not m_anon_band_ok:
            raise ValueError(
                f"smoke anonymous clean margin {m_anon:+.4f} outside "
                f"{F2_M_ANON_REF} ± {F2_M_ANON_BAND}"
            )
        footprint = dial_footprint_direction(model, F_LAYER, DIAL_NEURON)
        alphas = list(SMOKE_F2_ALPHAS if smoke else F2_ALPHAS)
        records: list[dict[str, Any]] = []
        for arm, direction in (("v1", basis[:, 0]), ("dial_fp", footprint)):
            direction = direction.to(device=device, dtype=torch.float32)
            for sign in (1, -1):
                for alpha in alphas:
                    push_vector = sign * alpha * push_base * direction
                    transform = directional_push_transform(push_vector, positions)
                    pushed = margin_from_log_probs(
                        _patched_final_margin(model, anon_tensor, {F_LAYER: transform}),
                        buy_id, sell_id,
                    )
                    delta_m = pushed - m_anon
                    if smoke and abs(delta_m) > F2_PUSH_BOUND:
                        raise ValueError(
                            f"smoke push {arm} sign={sign} α={alpha}: |ΔM| {abs(delta_m):.4f} > {F2_PUSH_BOUND}"
                        )
                    records.append(
                        {
                            "phase": "f2",
                            "arm": arm,
                            "sign": sign,
                            "alpha": alpha,
                            "push_norm": abs(alpha * push_base),
                            "clean_margin": m_anon,
                            "pushed_margin": pushed,
                            "delta_m": delta_m,
                            "jitter_band": abs(delta_m) <= F2_JITTER_BAND,
                        }
                    )
        # Zero push no-op (bit-exact expected).
        zero_transform = directional_push_transform(
            torch.zeros_like(basis[:, 0], dtype=torch.float32), positions
        )
        noop_patched = margin_from_log_probs(
            _patched_final_margin(model, anon_tensor, {F_LAYER: zero_transform}),
            buy_id, sell_id,
        )
        _check_noop(noop_patched, m_anon, "phase F F2/push_noop")
        records.append(
            {
                "phase": "f2",
                "arm": "noop",
                "sign": 0,
                "alpha": 0.0,
                "push_norm": 0.0,
                "clean_margin": m_anon,
                "pushed_margin": noop_patched,
                "delta_m": noop_patched - m_anon,
                "noop_delta_m": noop_patched - m_anon,
                "jitter_band": False,
            }
        )
        write_jsonl(path, records, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "entity_to_dial_f_forward_f2",
                "n_records": len(records),
                "layer": F_LAYER,
                "push_base": push_base,
                "m_anon": m_anon,
                "m_anon_ref": F2_M_ANON_REF,
                "m_anon_band": F2_M_ANON_BAND,
                "m_anon_band_ok": m_anon_band_ok,
                "alphas": alphas,
                "n_positions": len(positions),
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="entity_to_dial_f_forward_f2", stage="forward_f2", role="output", record_count=len(records))
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="entity_to_dial_f_forward_f2_metadata", stage="forward_f2", role="output")


def _phase_f_analyze(
    run: ArtifactRun,
    *,
    directions: list[tuple[str, str]],
    smoke: bool,
) -> None:
    """Phase F analyze: Gate F1 + F2 verdicts + consistency, §4.2–4.3."""
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("analyze") as stage:
        f1_records = read_jsonl(run.run_directory / "forward_f1" / "records.jsonl")
        f2_records = read_jsonl(run.run_directory / "forward_f2" / "records.jsonl")
        f2_meta = json.loads(
            (run.run_directory / "forward_f2" / "metadata.json").read_text(encoding="utf-8")
        )
        e01_ref = json.loads(
            (run.run_directory / "prepare" / "e01_ref.json").read_text(encoding="utf-8")
        )["reference"]
        gate = evaluate_gate_f(
            f1_records,
            f2_records,
            directions=[f"{s}->{t}" for s, t in directions],
            top_group=list(TOP_GROUP),
            bottom_group=list(BOTTOM_GROUP),
            e01_ref=e01_ref,
            min_full_dm=RATIO_MIN_FULL_DM,
            f1_min=GATE_F1_RATIO_MIN,
            jitter_band=F2_JITTER_BAND,
            consistency_tolerance=F1_CONSISTENCY_TOLERANCE,
            smoke=smoke,
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "entity_to_dial_f_summary",
            "smoke": smoke,
            "gate_f1": gate.get("gate_f1"),
            "f1_secondary": gate.get("f1_secondary"),
            "f1_consistency": gate.get("f1_consistency"),
            "f2_verdict": gate.get("f2_verdict"),
            "f2_dose_response": gate.get("f2_dose_response"),
            "gate_f": gate.get("gate") if gate.get("status") == "evaluated" else gate,
            "push_base": f2_meta["push_base"],
            "m_anon": f2_meta["m_anon"],
            "m_anon_band_ok": f2_meta["m_anon_band_ok"],
            "raw_runtime_payloads": False,
        }
        path = out_dir / "summary.json"
        write_json(path, summary, overwrite=True)
        run.manifest.register_artifact(path, artifact_type="entity_to_dial_f_summary", stage="analyze", role="output")


def run_phase_f(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    phase2b_run: str | Path,
    phasee_run: str | Path,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    phase2b_run = Path(phase2b_run)
    phasee_run = Path(phasee_run)
    directions = (
        [tuple(SMOKE_F_DIRECTION)] if smoke else frozen_directions(phase2b_run)
    )
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        rows, pure, basis, singular, e01_ref, anon_row = _phase_f_prepare(
            run, tokenizer,
            phase2a_run=phase2a_run, phase2b_run=phase2b_run, phasee_run=phasee_run,
            directions=directions, smoke=smoke,
        )
        f1_ctx = _phase_f_forward_f1(
            run, model_path, rows, pure, directions,
            basis=basis, singular=singular, e01_ref=e01_ref, smoke=smoke,
        )
        _phase_f_forward_f2(
            run, model_path, anon_row,
            basis=basis, push_base=f1_ctx["push_base"], smoke=smoke,
        )
        _phase_f_analyze(run, directions=directions, smoke=smoke)
        run.finalize(
            required_stages={"prepare", "forward_f1", "forward_f2", "analyze"}
        )
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
