"""Phase 2B/2C pipeline: entity-state layer sweep and component attribution.

Stages for 2B: prepare-pairs → sweep → analyze.
Stages for 2C: prepare-arms → attention/MLP → analyze.
Protocol: docs/balanced-evidence-gap/proposal-phase2.md §4.4–§4.5.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any, Mapping

import torch

from scipy.stats import rankdata
import numpy as np

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import input_ids

from .analysis import (
    GATE_2C,
    bootstrap_ci,
    detect_handoff,
    exact_sign_flip_p,
    normalized_transfer,
    spearman,
    toward_source_delta,
)
from .intervention import (
    AttentionEdgeZeroing,
    _attention_module,
    answer_token_ids,
    capture_residuals,
    make_span_transform,
    margin_from_log_probs,
    mlp_margin_attribution,
    nearest_position_mapping,
    patched_final_margin,
    random_match_positions,
)
from .template import DECISION_PREFIX, DATASET, SCHEMA_VERSION

SPAN_NAMES = ("entity", "evidence", "instruction", "final")


# ── shared helpers ───────────────────────────────────────────────────────────

def _load_rows(run_root: Path) -> dict[str, dict]:
    rows = read_jsonl(run_root / "prepare" / "prompts.jsonl")
    return {f"{r['ticker']}": r for r in rows if r["reverse"] is False and r["order"] == 0}


def _clean_margins(run_root: Path) -> dict[str, float]:
    results = read_jsonl(run_root / "forward" / "results.jsonl")
    margins: dict[str, list[float]] = {}
    for row in results:
        margins.setdefault(row["ticker"], []).append(float(row["margin"]))
    return {t: statistics.median(v) for t, v in margins.items()}


def _scoring_ids(tokenizer: Any, row: dict) -> list[int]:
    return input_ids(tokenizer, row["formatted"] + DECISION_PREFIX, add_special_tokens=True)


def _span_mapping(source_row: dict, target_row: dict, span: str) -> dict[int, int]:
    if span == "final":
        return {target_row["final_position"]: source_row["final_position"]}
    src_key = f"{span}_span" if span != "instruction" else "instruction_span"
    tgt_key = f"{span}_span" if span != "instruction" else "instruction_span"
    source_span = tuple(source_row[src_key])
    target_span = tuple(target_row[tgt_key])
    if span == "entity":
        return nearest_position_mapping(source_span, target_span)
    # evidence / instruction share identical text: offset identity.
    if source_span[1] - source_span[0] != target_span[1] - target_span[0]:
        return nearest_position_mapping(source_span, target_span)
    offset = target_span[0] - source_span[0]
    return {
        source_pos + offset: source_pos
        for source_pos in range(source_span[0], source_span[1])
    }


def select_directions(pure_margins: dict[str, float]) -> list[tuple[str, str]]:
    """Top-2 vs bottom-2 tickers, 4 ordered pairs, both orders = 8 directions."""
    ordered = sorted(pure_margins, key=lambda t: pure_margins[t])
    bottom = ordered[:2]
    top = ordered[-2:]
    if len(set(top + bottom)) < 4:
        raise ValueError("need four distinct tickers for top/bottom pairs")
    directions: list[tuple[str, str]] = []
    for source in top:
        for target in bottom:
            directions.append((source, target))
            directions.append((target, source))
    return directions


# ── 2B ───────────────────────────────────────────────────────────────────────

def _sweep_one_direction(
    model: Any,
    tokenizer: Any,
    *,
    source_row: dict,
    target_row: dict,
    m_source: float,
    m_target: float,
    layers: list[int],
) -> list[dict]:
    device = model.input_device if hasattr(model, "input_device") else "cuda" if torch.cuda.is_available() else "cpu"
    source_ids = _scoring_ids(tokenizer, source_row)
    target_ids = _scoring_ids(tokenizer, target_row)
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
    _, buy_id, sell_id = answer_token_ids(tokenizer, target_row["formatted"] + DECISION_PREFIX)

    source_residuals = capture_residuals(model, source_tensor, layers)
    target_residuals = capture_residuals(model, target_tensor, layers)
    # Live clean margins anchor the exact self-source no-op check; the stored
    # 2A margins (medians over variants) drive the toward-source statistics.
    live_target_margin = margin_from_log_probs(
        patched_final_margin(model, target_tensor, {}), buy_id, sell_id
    )
    records: list[dict] = []
    for layer in layers:
        for span in SPAN_NAMES:
            mapping = _span_mapping(source_row, target_row, span)
            transform = make_span_transform(source_residuals[layer], mapping)
            log_probs = patched_final_margin(model, target_tensor, {layer: transform})
            patched = margin_from_log_probs(log_probs, buy_id, sell_id)
            records.append(
                {
                    "phase": "2b",
                    "layer": layer,
                    "span": span,
                    "direction": f"{source_row['ticker']}->{target_row['ticker']}",
                    "patched_margin": patched,
                    "toward_source_delta_m": toward_source_delta(patched, m_source, m_target),
                    "normalized_transfer": normalized_transfer(patched, m_source, m_target),
                }
            )
            # self-source no-op check on the same (layer, span)
            self_mapping = {t: t for t in mapping}
            self_transform = make_span_transform(target_residuals[layer], self_mapping)
            self_log_probs = patched_final_margin(model, target_tensor, {layer: self_transform})
            self_patched = margin_from_log_probs(self_log_probs, buy_id, sell_id)
            if abs(self_patched - live_target_margin) > 1e-12:
                raise ValueError(
                    f"self-source no-op violated at L{layer}/{span}: "
                    f"ΔM={self_patched - live_target_margin:.3e}"
                )
    return records


def run_phase2b(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a = Path(phase2a_run)
    rows = _load_rows(phase2a)
    clean = _clean_margins(phase2a)
    if smoke:
        top = sorted(clean, key=lambda t: clean[t], reverse=True)[:2]
        bottom = sorted(clean, key=lambda t: clean[t])[:2]
        directions = [(top[0], bottom[0]), (bottom[0], top[0])]
        layers = [3, 7, 15]
    else:
        directions = select_directions(clean)
        layers = list(range(32))

    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    out_dir = run.run_directory
    try:
        with run.stage("prepare-pairs") as stage:
            pairs_dir = out_dir / "pairs"
            pairs_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "directions": [list(d) for d in directions],
                "layers": layers,
                "spans": list(SPAN_NAMES),
                "pure_entity_margins": clean,
                "smoke": smoke,
            }
            pairs_path = pairs_dir / "directions.json"
            write_json(pairs_path, payload, overwrite=True)
            run.manifest.register_artifact(pairs_path, artifact_type="balanced_evidence_gap_phase2b_pairs", stage="prepare-pairs", role="output")
            stage.count(len(directions))

        model, tokenizer, device = load_model(model_path, dtype=None)
        records: list[dict] = []
        with run.stage("sweep") as stage:
            for index, (source, target) in enumerate(directions):
                source_row, target_row = rows[source], rows[target]
                records.extend(
                    _sweep_one_direction(
                        model, tokenizer,
                        source_row=source_row, target_row=target_row,
                        m_source=clean[source], m_target=clean[target],
                        layers=layers,
                    )
                )
                print(f"  direction {index + 1}/{len(directions)}: {source}->{target}", flush=True)
            sweep_dir = out_dir / "sweep"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            sweep_path = sweep_dir / "records.jsonl"
            count = write_jsonl(sweep_path, records, overwrite=True)
            write_metadata(
                sweep_dir / "metadata.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "balanced_evidence_gap_phase2b_sweep_metadata",
                    "n_records": count,
                    "self_source_noop_verified": True,
                    "raw_runtime_payloads": False,
                },
                overwrite=True,
            )
            run.manifest.register_artifact(sweep_path, artifact_type="balanced_evidence_gap_phase2b_sweep", stage="sweep", role="output", record_count=count)
            run.manifest.register_artifact(sweep_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2b_sweep_metadata", stage="sweep", role="output")
            stage.count(count)

        with run.stage("analyze") as stage:
            summary = analyze_2b_records(records, layers=layers)
            analyze_dir = out_dir / "analyze"
            analyze_dir.mkdir(parents=True, exist_ok=True)
            summary_path = analyze_dir / "summary.json"
            write_json(summary_path, summary, overwrite=True)
            write_metadata(
                analyze_dir / "metadata.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "balanced_evidence_gap_phase2b_analysis_metadata",
                    "handoff": summary["handoff"]["crossover"],
                    "raw_runtime_payloads": False,
                },
                overwrite=True,
            )
            run.manifest.register_artifact(summary_path, artifact_type="balanced_evidence_gap_phase2b_analysis", stage="analyze", role="output")
            run.manifest.register_artifact(analyze_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2b_analysis_metadata", stage="analyze", role="output")
            stage.count(len(summary["curves"]))
        run.finalize(required_stages={"prepare-pairs", "sweep", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


def analyze_2b_records(records: list[dict], *, layers: list[int]) -> dict:
    """Per (span, layer) equal-direction means + handoff detection."""
    curves: dict[str, dict[int, dict]] = {}
    for span in SPAN_NAMES:
        curves[span] = {}
    for layer in layers:
        for span in SPAN_NAMES:
            deltas = [r["toward_source_delta_m"] for r in records if r["layer"] == layer and r["span"] == span]
            transfers = [r["normalized_transfer"] for r in records if r["layer"] == layer and r["span"] == span]
            if len(deltas) < 2:
                raise ValueError(f"missing directions for L{layer}/{span}")
            ci = bootstrap_ci(deltas) if len(deltas) >= 4 else None
            curves[span][layer] = {
                "mean_toward_source_delta_m": statistics.fmean(deltas),
                "mean_normalized_transfer": statistics.fmean(transfers),
                "delta_m_ci_95": list(ci) if ci is not None else None,
                "n_directions": len(deltas),
            }
    entity_t = {l: curves["entity"][l]["mean_normalized_transfer"] for l in layers}
    context_t = {l: curves["instruction"][l]["mean_normalized_transfer"] for l in layers}
    handoff = detect_handoff(layers, entity_t, context_t)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "balanced_evidence_gap_phase2b_analysis",
        "spans": list(SPAN_NAMES),
        "curves": curves,
        "handoff": handoff,
        "phase2c_arms": {
            "attention_layers": handoff.get("attention_layers", []),
            "mlp_layers": handoff.get("mlp_layers", []),
        },
        "discovery_only": True,
        "raw_runtime_payloads": False,
    }


# ── 2C ───────────────────────────────────────────────────────────────────────

def run_phase2c(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    phase2b_run: str | Path,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a = Path(phase2a_run)
    phase2b_summary = json.loads(
        (Path(phase2b_run) / "analyze" / "summary.json").read_text(encoding="utf-8")
    )
    rows = _load_rows(phase2a)
    clean = _clean_margins(phase2a)
    arms = phase2b_summary["phase2c_arms"]
    attention_layers = arms["attention_layers"]
    mlp_layers = arms["mlp_layers"]
    if smoke:
        attention_layers = [l for l in (3, 7, 15) if l in attention_layers][:1] or []
        mlp_layers = mlp_layers[:1]
    if not attention_layers and not mlp_layers:
        raise ValueError("no 2C layers available from the 2B handoff")

    directions = select_directions(clean) if not smoke else [
        (sorted(clean, key=lambda t: clean[t], reverse=True)[0],
         sorted(clean, key=lambda t: clean[t])[0])
    ]

    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    out_dir = run.run_directory
    try:
        with run.stage("prepare-arms") as stage:
            arms_dir = out_dir / "arms"
            arms_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "attention_layers": attention_layers,
                "mlp_layers": mlp_layers,
                "attention_status": "run" if attention_layers else "not_applicable",
                "directions": [list(d) for d in directions],
                "matched_control_samples": GATE_2C["matched_control_samples"],
                "control_seed": GATE_2C["control_seed"],
                "smoke": smoke,
            }
            arms_path = arms_dir / "arms.json"
            write_json(arms_path, payload, overwrite=True)
            run.manifest.register_artifact(arms_path, artifact_type="balanced_evidence_gap_phase2c_arms", stage="prepare-arms", role="output")
            stage.count(len(attention_layers) + len(mlp_layers))

        model, tokenizer, device = load_model(model_path, dtype=None)
        attention_records: list[dict] = []
        mlp_records: list[dict] = []
        mlp_layer_summaries: list[dict] = []

        if attention_layers:
            with run.stage("attention") as stage:
                for layer in attention_layers:
                    attention_records.extend(
                        _attention_arm(
                            model, tokenizer, layer=layer,
                            directions=directions, rows=rows, clean=clean,
                        )
                    )
                att_dir = out_dir / "attention"
                att_dir.mkdir(parents=True, exist_ok=True)
                att_path = att_dir / "records.jsonl"
                count = write_jsonl(att_path, attention_records, overwrite=True)
                run.manifest.register_artifact(att_path, artifact_type="balanced_evidence_gap_phase2c_attention", stage="attention", role="output", record_count=count)
                stage.count(count)

        with run.stage("mlp") as stage:
            for layer in mlp_layers:
                mlp_records, layer_summary = _mlp_arm(
                    model, tokenizer, layer=layer, rows=rows, clean=clean
                )
                mlp_layer_summaries.append(layer_summary)
            mlp_dir = out_dir / "mlp"
            mlp_dir.mkdir(parents=True, exist_ok=True)
            mlp_path = mlp_dir / "records.jsonl"
            count = write_jsonl(mlp_path, mlp_records, overwrite=True)
            write_json(
                mlp_dir / "layer_summaries.json",
                {"schema_version": SCHEMA_VERSION, "layers": mlp_layer_summaries},
                overwrite=True,
            )
            run.manifest.register_artifact(mlp_path, artifact_type="balanced_evidence_gap_phase2c_mlp", stage="mlp", role="output", record_count=count)
            run.manifest.register_artifact(mlp_dir / "layer_summaries.json", artifact_type="balanced_evidence_gap_phase2c_mlp_layer_summaries", stage="mlp", role="output")
            stage.count(count)

        with run.stage("analyze") as stage:
            summary = analyze_2c_records(
                attention_records, mlp_layer_summaries,
                attention_layers=attention_layers, mlp_layers=mlp_layers,
            )
            analyze_dir = out_dir / "analyze"
            analyze_dir.mkdir(parents=True, exist_ok=True)
            summary_path = analyze_dir / "summary.json"
            write_json(summary_path, summary, overwrite=True)
            run.manifest.register_artifact(summary_path, artifact_type="balanced_evidence_gap_phase2c_analysis", stage="analyze", role="output")
            stage.count(1)
        run.finalize(
            required_stages={"prepare-arms", "mlp", "analyze"}
            | ({"attention"} if attention_layers else set())
        )
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


def _attention_arm(
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    directions: list[tuple[str, str]],
    rows: dict[str, dict],
    clean: dict[str, float],
) -> list[dict]:
    device = model.input_device if hasattr(model, "input_device") else "cuda" if torch.cuda.is_available() else "cpu"
    attention = _attention_module(model, layer)
    head_dim = int(getattr(attention, "head_dim", 0) or attention.q_proj.weight.shape[0] // 32)
    num_heads = attention.q_proj.weight.shape[0] // (2 * head_dim)
    records: list[dict] = []

    for source, target in directions:
        target_row = rows[target]
        m_source, m_target = clean[source], clean[target]
        scoring_ids = _scoring_ids(tokenizer, target_row)
        tensor = torch.tensor([scoring_ids], dtype=torch.long, device=device)
        _, buy_id, sell_id = answer_token_ids(tokenizer, target_row["formatted"] + DECISION_PREFIX)
        query = len(scoring_ids) - 1
        entity_positions = tuple(range(target_row["entity_span"][0], target_row["entity_span"][1]))
        controls = random_match_positions(
            query, entity_positions,
            n_samples=GATE_2C["matched_control_samples"],
            seed=GATE_2C["control_seed"] + layer,
        )

        def patched_margin_with(zero: tuple[int, ...], head: int) -> tuple[float, float | None]:
            with AttentionEdgeZeroing(
                model, layer, head, zero, query_position=query
            ) as session:
                log_probs = patched_final_margin(model, tensor, {})
            margin = margin_from_log_probs(log_probs, buy_id, sell_id)
            return margin, session.reconstruction_error

        for head in range(num_heads):
            entity_margin, entity_recon = patched_margin_with(entity_positions, head)
            control_margins = [patched_margin_with(ctrl, head)[0] for ctrl in controls]
            records.append(
                {
                    "phase": "2c-attention",
                    "layer": layer,
                    "head": head,
                    "direction": f"{source}->{target}",
                    "entity_zeroed": entity_margin,
                    "entity_toward_source_delta_m": toward_source_delta(entity_margin, m_source, m_target),
                    "control_toward_source_delta_ms": [
                        toward_source_delta(m, m_source, m_target) for m in control_margins
                    ],
                    "reconstruction_error": entity_recon,
                }
            )
    return records


def _mlp_arm(
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    rows: dict[str, dict],
    clean: dict[str, float],
) -> tuple[list[dict], dict]:
    """Per-layer attribution sweep; returns (compact records, layer summary).

    Matched-control correlations are computed here, where the full 9216-dim
    attribution vector exists in memory; only statistics are persisted.
    """
    device = model.input_device if hasattr(model, "input_device") else "cuda" if torch.cuda.is_available() else "cpu"
    records: list[dict] = []
    vectors: dict[str, torch.Tensor] = {}
    for ticker in sorted(rows):
        row = rows[ticker]
        scoring_ids = _scoring_ids(tokenizer, row)
        tensor = torch.tensor([scoring_ids], dtype=torch.long, device=device)
        _, buy_id, sell_id = answer_token_ids(tokenizer, row["formatted"] + DECISION_PREFIX)
        result = mlp_margin_attribution(
            model, tensor,
            layer=layer, position=row["entity_position"],
            buy_id=buy_id, sell_id=sell_id,
        )
        vectors[ticker] = result["attribution"]
        records.append(
            {
                "phase": "2c-mlp",
                "layer": layer,
                "ticker": ticker,
                "sector": row["sector"],
                "entity_position": row["entity_position"],
                "margin": result["margin"],
                "input_norm": result["input_norm"],
                "derivative_norm": result["derivative_norm"],
                # Compact top-k only; the 9216-dim vector is never persisted.
                "attribution_top_k": _top_k(result["attribution"], 20),
            }
        )

    tickers = sorted(vectors)
    stacked = torch.stack([vectors[t] for t in tickers])  # [n_tickers, width]
    margins = torch.tensor([clean[t] for t in tickers])

    all_values = stacked.numpy()
    all_margins = margins.numpy()
    with np.errstate(invalid="ignore"):
        rank_vals = rankdata(all_values, axis=0)
        rank_margs = rankdata(all_margins)
        # per-neuron spearman = pearson of ranks
        centered_v = rank_vals - rank_vals.mean(axis=0, keepdims=True)
        centered_m = rank_margs - rank_margs.mean()
        denom = np.sqrt((centered_v ** 2).sum(axis=0)) * np.sqrt((centered_m ** 2).sum())
        neuron_rho = np.where(denom > 0, centered_v.T @ centered_m / np.where(denom == 0, 1, denom), 0.0)
    top_neuron = int(np.argmax(np.abs(neuron_rho)))
    top_rho = float(neuron_rho[top_neuron])

    rng = random.Random(GATE_2C["control_seed"] + layer)
    n_controls = min(GATE_2C["matched_control_samples"], stacked.shape[1] // 2)
    control_neurons = rng.sample(range(stacked.shape[1]), n_controls)
    control_rhos = [float(neuron_rho[n]) for n in control_neurons]

    # Sector agreement: sign of the top neuron's per-sector spearman vs global.
    sector_map = {r["ticker"]: r["sector"] for r in records}
    sector_agreement = compute_sector_agreement(
        neuron_values={t: float(stacked[i, top_neuron]) for i, t in enumerate(tickers)},
        top_rho=top_rho,
        sectors=sector_map,
        margins={t: clean[t] for t in tickers},
    )

    # Sign-flip test: per-ticker deviation of the top neuron's attribution
    # from its mean, in the direction of the margin deviation.
    neuron_values = stacked[:, top_neuron].tolist()
    neuron_mean = statistics.fmean(neuron_values)
    margin_mean = statistics.fmean(margins.tolist())
    signs = [
        (v - neuron_mean) * (m - margin_mean)
        for v, m in zip(neuron_values, margins.tolist())
    ]
    nonzero = [s for s in signs if abs(s) > 1e-12]
    sign_flip_p = exact_sign_flip_p(nonzero) if nonzero else 1.0

    layer_summary = {
        "layer": layer,
        "top_neuron": top_neuron,
        "top_spearman": top_rho,
        "abs_top_spearman": abs(top_rho),
        "control_rhos": control_rhos,
        "control_mean_rho": statistics.fmean(control_rhos),
        "control_max_abs_rho": max(abs(c) for c in control_rhos),
        "sector_agreement": sector_agreement,
        "sign_flip_p": sign_flip_p,
    }
    return records, layer_summary


def compute_sector_agreement(
    *,
    neuron_values: Mapping[str, float],
    top_rho: float,
    sectors: Mapping[str, str],
    margins: Mapping[str, float],
) -> float:
    """Fraction of sectors whose within-sector spearman sign matches the global.

    A sector with fewer than 3 tickers is skipped. A degenerate sector (the
    neuron or the margin is constant within it) carries no within-sector
    evidence and fails closed: it counts as non-agreement.
    """
    tickers = sorted(neuron_values)
    agreements: list[bool] = []
    for sector in sorted(set(sectors.values())):
        sector_tickers = [t for t in tickers if sectors[t] == sector]
        if len(sector_tickers) < 3:
            continue
        sector_vals = [neuron_values[t] for t in sector_tickers]
        sector_margins = [margins[t] for t in sector_tickers]
        if len(set(sector_vals)) == 1 or len(set(sector_margins)) == 1:
            agreements.append(False)
            continue
        sector_rho = spearman(sector_vals, sector_margins)
        agreements.append(sector_rho * top_rho >= 0)
    return sum(agreements) / len(agreements) if agreements else 0.0


def _top_k(vector: torch.Tensor, k: int) -> list[dict]:
    indices = torch.topk(vector, k=k).indices.tolist()
    return [
        {"neuron": int(i), "value": float(vector[i])} for i in sorted(indices)
    ]


def analyze_2c_records(
    attention_records: list[dict],
    mlp_layer_summaries: list[dict],
    *,
    attention_layers: list[int],
    mlp_layers: list[int],
) -> dict:
    """Gate 2C over the two arms (proposal §4.5).

    Attention arm: per-head paired difference = ΔM(entity zeroing) −
    mean ΔM(matched position zeroing), averaged over directions; the top
    head must show a positive main effect whose direction-wise paired
    differences pass the Holm-adjusted exact sign-flip test.
    MLP arm: per-layer precomputed summaries (top neuron |spearman| vs
    matched controls, sector agreement, sign-flip p).
    """
    from .analysis import evaluate_gate_2c, holm_adjusted

    attention_arm: dict | None = None
    if attention_layers:
        head_effects: dict[str, dict] = {}
        for layer in attention_layers:
            layer_records = [r for r in attention_records if r["layer"] == layer]
            for head in sorted({r["head"] for r in layer_records}):
                key = f"L{layer}H{head}"
                per_direction = [
                    (
                        r["entity_toward_source_delta_m"],
                        r["control_toward_source_delta_ms"],
                    )
                    for r in layer_records
                    if r["head"] == head
                ]
                paired = [
                    entity - statistics.fmean(controls)
                    for entity, controls in per_direction
                ]
                head_effects[key] = {
                    "direction_deltas": paired,
                    "mean_delta": statistics.fmean(paired),
                    "control_mean_delta": statistics.fmean(
                        [c for _, controls in per_direction for c in controls]
                    ),
                }
        p_values = [exact_sign_flip_p(e["direction_deltas"]) for e in head_effects.values()]
        adjusted = holm_adjusted(p_values)
        adjusted_map = dict(zip(head_effects.keys(), adjusted))
        keys = sorted(head_effects, key=lambda k: head_effects[k]["mean_delta"], reverse=True)
        attention_arm = {
            "status": "run",
            "head_effects": head_effects,
            "holm_adjusted_p": adjusted_map,
            "top_head": keys[0] if keys else None,
        }

    if mlp_layer_summaries:
        mlp_arm = {
            "top_attribution": max(s["abs_top_spearman"] for s in mlp_layer_summaries),
            "control_mean": max(s["control_max_abs_rho"] for s in mlp_layer_summaries),
            "sector_agreement": min(s["sector_agreement"] for s in mlp_layer_summaries),
            "sign_flip_p": max(s["sign_flip_p"] for s in mlp_layer_summaries),
            "per_layer": {
                str(s["layer"]): {
                    "top_neuron": s["top_neuron"],
                    "top_spearman": s["top_spearman"],
                    "sector_agreement": s["sector_agreement"],
                    "sign_flip_p": s["sign_flip_p"],
                }
                for s in mlp_layer_summaries
            },
        }
    else:
        mlp_arm = {
            "top_attribution": 0.0, "control_mean": 0.0,
            "sector_agreement": 0.0, "sign_flip_p": 1.0, "per_layer": {},
        }

    gate = evaluate_gate_2c(attention_arm=attention_arm, mlp_arm=mlp_arm)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "balanced_evidence_gap_phase2c_analysis",
        "attention_layers": attention_layers,
        "mlp_layers": mlp_layers,
        "gate_2c": gate,
        "formal": True,
        "raw_runtime_payloads": False,
    }


