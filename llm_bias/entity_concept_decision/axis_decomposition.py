"""Bounded development probe: decompose the local causal weight of the L15
instruction-span state on the buy/sell margin by content axis.

Development only, no gates.  After the selected block (default L15), an additive
push (signed dose x unit axis) is applied to the instruction-span positions and
the final-position buy/sell margin is re-scored.  Axes: the top-k principal
components of the 16 company states (the between-company / entity-contrast
subspace), the stance axis (same formula as the characterization and causal
probe runs), the shared mean-state direction (content common to all companies),
and seeded random unit directions (control + overall-weight estimate: for a
random unit direction r, E[(w.r)^2] = ||w||^2 / d_model).  Per-axis local
slopes (margin change per unit state push) show where the decision's causal
sensitivity concentrates.  No raw states are persisted: compact statistics and
axis hashes only.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import write_json, write_jsonl
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.entity_concept_decision.development import (
    _answer_ids,
    _json_number,
    _margin,
    _register_reference,
)
from llm_bias.entity_concept_decision.stance_causal import (
    _additive_transform,
    _fit_stance_direction,
    _tensor_sha256,
    _unit,
)

DEFAULT_DOSES = (0.08, 0.40)


def _build_axes(
    company_states: list[torch.Tensor],
    stance: torch.Tensor,
    n_pcs: int,
    n_random: int,
    random_seed: int,
) -> list[tuple[str, torch.Tensor]]:
    """Top-k PCs of the centered company states, the stance axis, the shared
    mean-state direction, and seeded random unit directions."""
    X = torch.stack(company_states)
    Xc = X - X.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    axes: list[tuple[str, torch.Tensor]] = []
    for i in range(min(n_pcs, Vt.shape[0])):
        axes.append((f"pc{i + 1}", _unit(Vt[i].float())))
    axes.append(("stance", _unit(stance.float())))
    mean_state = X.mean(dim=0)
    axes.append(("mean_state", _unit(mean_state.float())))
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    for i in range(n_random):
        axes.append((f"random_{i}", _unit(torch.randn(X.shape[1], generator=generator))))
    return axes


def _analyze(
    records: list[dict[str, Any]],
    company_ids: list[str],
    doses: Sequence[float],
    d_model: int,
    axis_names: list[str],
) -> dict[str, Any]:
    """Per-axis local slopes, significance vs matched randoms, and the
    squared-weight decomposition (structured axes vs overall ||w||)."""
    structured = [a for a in axis_names if not a.startswith("random_")]
    random_names = [a for a in axis_names if a.startswith("random_")]
    # per (axis, signed dose): list of (company, delta/dose) — slope draws
    draws: defaultdict[tuple[str, float], list[float]] = defaultdict(list)
    for r in records:
        if r["role"] != "intervention" or r.get("status") != "ok":
            continue
        draws[(r["axis"], r["dose"])].append(r["delta_margin"] / r["dose"])
    per_axis: dict[str, Any] = {}
    for axis in axis_names:
        slope_draws: list[float] = []
        by_dose: dict[str, Any] = {}
        for dose in doses:
            for sign in (1.0, -1.0):
                sd = draws.get((axis, sign * dose), [])
                if not sd:
                    continue
                mean = sum(sd) / len(sd)
                by_dose[f"{sign * dose:+g}"] = {
                    "slope": _json_number(mean, name="slope"),
                    "sd": _json_number(_sd(sd), name="sd"),
                    "n": len(sd),
                }
                slope_draws.extend(sd)
        if not slope_draws:
            raise ValueError(f"no measurements for axis {axis}")
        slope = sum(slope_draws) / len(slope_draws)
        n = len(slope_draws)
        se = _sd(slope_draws) / math.sqrt(n) if n > 1 else 0.0
        per_axis[axis] = {
            "slope": _json_number(slope, name="slope"),
            "slope_se": _json_number(se, name="slope_se"),
            "slope_t": _json_number(slope / se, name="slope_t") if se > 0 else None,
            "by_signed_dose": by_dose,
            "n_draws": n,
        }
    # overall ||w|| estimate from random directions: E[(w.r)^2] = ||w||^2 / d
    random_draws = [v for (axis, _), vs in draws.items() if axis in random_names for v in vs]
    if not random_draws:
        raise ValueError("no random-direction measurements")
    mean_sq = sum(v * v for v in random_draws) / len(random_draws)
    w_norm_sq = d_model * mean_sq
    w_norm = math.sqrt(w_norm_sq)
    decomposition: dict[str, Any] = {}
    for axis in structured:
        slope = per_axis[axis]["slope"]
        decomposition[axis] = _json_number((slope * slope) / w_norm_sq, name="weight_fraction")
    pc_fraction = sum(decomposition[a] for a in per_axis if a.startswith("pc"))
    decomposition["sum_pc_subspace"] = _json_number(pc_fraction, name="weight_fraction")
    decomposition["sum_all_structured"] = _json_number(
        sum(decomposition[a] for a in structured if a in decomposition), name="weight_fraction")
    return {
        "claim_status": "descriptive_probe",
        "scientific_status": "not_evaluated",
        "purpose": "development",
        "per_axis": per_axis,
        "w_norm_estimate": {
            "w_norm_sq": _json_number(w_norm_sq, name="w_norm_sq"),
            "w_norm": _json_number(w_norm, name="w_norm"),
            "n_random_draws": len(random_draws),
            "random_mean_sq_draw": _json_number(mean_sq, name="random_mean_sq_draw"),
        },
        "weight_fractions": decomposition,
        "n_companies": len(company_ids),
    }


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _score_with_guard(
    *,
    model: Any,
    tensor: torch.Tensor,
    layer: int,
    last: int,
    buy_id: int,
    sell_id: int,
    transform: Any | None = None,
) -> dict[str, Any]:
    if transform is None:
        captured = record_residuals(model, tensor, [last])
    else:
        with residual_interventions(model, {layer: transform}):
            captured = record_residuals(model, tensor, [last])
    final = captured[last][:, -1, :].float()
    try:
        margin = _margin(model, final, buy_id, sell_id)
    except ValueError:
        return {"status": "non_finite"}
    del captured
    return {"status": "ok", "margin": float(margin), "p_buy_2way": 1.0 / (1.0 + math.exp(-margin))}


def run_axis_decomposition(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    layer: int,
    run_id: str,
    model_name: str,
    artifact_root: str | Path,
    provenance: Mapping[str, Any],
    doses: Sequence[float] = DEFAULT_DOSES,
    n_pcs: int = 8,
    n_random: int = 12,
    random_seed: int = 1729,
) -> Path:
    """prepare -> forward (axes from states, per-axis dose sweeps) -> analyze."""
    last = int(model.n_layers) - 1
    if not (0 <= int(layer) < int(model.n_layers)):
        raise ValueError(f"layer {layer} out of range for {model.n_layers} layers")
    dose_list = sorted({abs(float(d)) for d in doses})
    if not dose_list or any(d <= 0 for d in dose_list):
        raise ValueError("doses must be positive magnitudes")
    material = [dict(p) for p in prompts if p.get("kind") != "company"]
    companies = [dict(p) for p in prompts if p.get("kind") == "company"]
    if len(material) < 2 or len(companies) < 3:
        raise ValueError("need material rows and at least 3 companies")
    d_model = int(model.d_model)

    run = ArtifactRun.create(model_name, "entity-concept-axis-decomposition", run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            prepare_dir = run.run_directory / "prepare"
            write_jsonl(prepare_dir / "prompts.jsonl", [*material, *companies], overwrite=False)
            metadata = {
                "purpose": "development",
                "claim_status": "descriptive_probe",
                "provenance": dict(provenance),
                "layer": int(layer),
                "final_layer": last,
                "doses": dose_list,
                "n_pcs": n_pcs,
                "n_random_directions": n_random,
                "random_seed": random_seed,
                "d_model": d_model,
            }
            write_json(prepare_dir / "metadata.json", metadata, overwrite=False)
            _register_reference(run, provenance["upstream"], role="upstream", stage="prepare")
            if provenance.get("lens"):
                _register_reference(run, provenance["lens"], role="lens", stage="prepare")
            stage.count(len(material) + len(companies))

        with run.stage("forward") as stage:
            # 1) stance axis (same formula as the characterization/causal runs)
            states: dict[str, torch.Tensor] = {}
            for item in material:
                tensor = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                captured = record_residuals(model, tensor, [layer])
                span = item["instruction_span"]
                block = captured[layer][:, span[0]:span[1], :]
                states[item["id"]] = block.float().mean(dim=1).detach().cpu().reshape(-1)
                del captured, block
            stance, pair_ids = _fit_stance_direction(states, comparisons)
            del states
            # 2) company baselines + states -> axes
            company_states: list[torch.Tensor] = []
            baselines: dict[str, float] = {}
            records: list[dict[str, Any]] = []
            company_ids: list[str] = []
            for item in companies:
                company_ids.append(item["id"])
                tensor = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                captured = record_residuals(model, tensor, [layer, last])
                span = item["instruction_span"]
                company_states.append(captured[layer][:, span[0]:span[1], :].float().mean(dim=1).detach().cpu().reshape(-1))
                final = captured[last][:, -1, :].float()
                del captured
                ids, buy_id, sell_id = _answer_ids(tokenizer, item["formatted"])
                baseline = _margin(model, final, buy_id, sell_id)
                baselines[item["id"]] = float(baseline)
                records.append({
                    "id": item["id"], "role": "baseline", "axis": "none", "dose": 0.0,
                    "margin": _json_number(float(baseline), name="margin"), "status": "ok",
                })
            axes = _build_axes(company_states, stance, n_pcs, n_random, random_seed)
            axis_names = [name for name, _ in axes]
            # 3) per-axis signed-dose sweeps
            total = len(companies) * len(axes) * (2 * len(dose_list))
            done = 0
            for i, (item, cid) in enumerate(zip(companies, company_ids), start=1):
                tensor = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                span = item["instruction_span"]
                ids, buy_id, sell_id = _answer_ids(tokenizer, item["formatted"])
                baseline = baselines[cid]
                for name, direction in axes:
                    for dose in dose_list:
                        for sign in (1.0, -1.0):
                            transform = _additive_transform(direction, sign * dose, span[0], span[1], device)
                            t0 = time.perf_counter()
                            scored = _score_with_guard(
                                model=model, tensor=tensor, layer=layer, last=last,
                                buy_id=buy_id, sell_id=sell_id, transform=transform,
                            )
                            done += 1
                            row = {
                                "id": cid, "role": "intervention", "axis": name,
                                "dose": _json_number(float(sign * dose), name="dose"),
                                "status": scored["status"],
                                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                            }
                            if scored["status"] == "ok":
                                row["margin"] = _json_number(scored["margin"], name="margin")
                                row["delta_margin"] = _json_number(scored["margin"] - baseline, name="delta_margin")
                            records.append(row)
                            del transform
                print(f"company {i}/{len(companies)} ({done}/{total}): {cid}", flush=True)
                del tensor
            n_non_finite = sum(1 for r in records if r.get("status") != "ok")
            records_path = run.run_directory / "forward" / "records.jsonl"
            write_jsonl(records_path, records, overwrite=False)
            forward_meta = {
                "n_records": len(records),
                "n_non_finite": n_non_finite,
                "axes": {name: _tensor_sha256(direction) for name, direction in axes},
                "stance_n_pairs": len(pair_ids),
                "stance_pairs": pair_ids,
                "pc_variance_fractions": _pc_variance_fractions(company_states, n_pcs),
                "baseline_margins": {cid: _json_number(baselines[cid], name="margin") for cid in company_ids},
                "total_forward_seconds": round(sum(r.get("elapsed_seconds", 0.0) for r in records), 1),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0,
            }
            write_json(run.run_directory / "forward" / "metadata.json", forward_meta, overwrite=False)
            run.manifest.register_artifact(records_path, artifact_type="axis_decomposition_forward_records", stage="forward", record_count=len(records))
            run.manifest.register_artifact(run.run_directory / "forward" / "metadata.json", artifact_type="axis_decomposition_forward_metadata", stage="forward")
            stage.count(len(records))

        with run.stage("analyze") as stage:
            summary = _analyze(records, company_ids, dose_list, d_model, axis_names)
            summary.update({
                "schema_version": 1,
                "layer": int(layer),
                "doses": dose_list,
                "stance_direction_sha256": _tensor_sha256(stance),
            })
            summary_path = run.run_directory / "analyze" / "summary.json"
            write_json(summary_path, summary, overwrite=False)
            run.manifest.register_artifact(summary_path, artifact_type="axis_decomposition_summary", stage="analyze")
            stage.count(1)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def _pc_variance_fractions(company_states: list[torch.Tensor], n_pcs: int) -> list[float]:
    X = torch.stack(company_states)
    Xc = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    eig = (S ** 2) / (len(X) - 1)
    total = float(eig.sum())
    return [round(float(eig[i]) / total, 4) for i in range(min(n_pcs, eig.shape[0]))]


__all__ = ["run_axis_decomposition", "DEFAULT_DOSES"]
