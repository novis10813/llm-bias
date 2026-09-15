"""Bounded multi-layer concept-vs-stance separability scan (development only).

At each candidate layer, the material concept directions (G, C) and the general
stance direction are fitted in the FULL layer state space (no k=8 restriction),
the concept direction is orthogonalized against stance, and the four companies
are scored raw and orthogonalized.  This localizes whether any company-specific
concept component survives stance removal at any layer.  Development only: no
formal gates, no raw activations persisted.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import write_json, write_jsonl
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.forward import record_residuals
from llm_bias.entity_concept_decision.development import (
    _answer_ids,
    _company_records,
    _json_number,
    _margin,
    _material_prompt_records,
    _register_reference,
)


def _unit(v: torch.Tensor) -> torch.Tensor | None:
    n = float(v.norm())
    if n < 1e-9 or not torch.isfinite(v).all():
        return None
    return (v / n).contiguous()


def _fit_mean_direction(differences: Sequence[torch.Tensor], min_norm: float) -> tuple[torch.Tensor | None, str]:
    if not differences:
        return None, "empty"
    stacked = torch.stack([d.float() for d in differences], dim=0)
    mean = stacked.mean(dim=0)
    n = float(mean.norm())
    if n < min_norm or not torch.isfinite(mean).all():
        return None, "degenerate"
    return _unit(mean), "ok"


def _orthogonalize(direction: torch.Tensor, stance: torch.Tensor, min_norm: float) -> tuple[torch.Tensor | None, str, float]:
    proj = float(direction @ stance)
    remainder = direction - proj * stance
    n = float(remainder.norm())
    if n < min_norm or not torch.isfinite(remainder).all():
        return None, "degenerate_after_orthogonalization", proj
    return _unit(remainder), "ok", proj


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    q = lambda p: None if not ordered else ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]
    return {"n": len(ordered), "q05": q(0.05), "q50": q(0.50), "q95": q(0.95)}


def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return _json_number(sxy / (sxx ** 0.5 * syy ** 0.5), name="pearson_r")


def _pearson_r2(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    r = _pearson_r(xs, ys)
    return None if r is None else _json_number(r * r, name="pearson_r2")


def _fit_stance_direction(vectors: Mapping[str, torch.Tensor], comparisons: list[dict[str, Any]], min_norm: float) -> tuple[torch.Tensor | None, int]:
    eval_diffs = [
        vectors[c["positive_id"]].float() - vectors[c["negative_id"]].float()
        for c in comparisons
        if c["kind"] in ("evaluation_at_positive_concept", "evaluation_at_negative_concept")
    ]
    stance_dir, _ = _fit_mean_direction(eval_diffs, min_norm)
    return stance_dir, len(eval_diffs)


def _ols_r2(X: torch.Tensor, y: torch.Tensor) -> float | None:
    """R^2 of OLS regression y ~ X (with intercept). X: [n, p], y: [n]."""
    n, p = X.shape
    if n < p + 2:
        return None
    Xb = torch.cat([torch.ones((n, 1)), X], dim=1)
    beta, *_ = torch.linalg.lstsq(Xb, y)
    yhat = Xb @ beta
    ss_res = float(torch.sum((y - yhat) ** 2))
    ss_tot = float(torch.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return None
    return _json_number(1.0 - ss_res / ss_tot, name="ols_r2")


def _analyze_layer(
    vectors: Mapping[str, torch.Tensor],
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    company_ids: list[str],
    concept_ids: Sequence[str],
    *,
    random_seed: int,
    random_count: int,
    min_norm: float,
) -> dict[str, Any]:
    by_id = {r["id"]: r for r in rows}
    stance_dir, stance_n_pairs = _fit_stance_direction(vectors, comparisons, min_norm)
    stance_status = "ok" if stance_dir is not None else "degenerate"

    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    random_dirs: list[torch.Tensor] = []
    if random_count:
        raw = torch.randn((random_count, next(iter(vectors.values())).numel()), generator=generator, dtype=torch.float32)
        random_dirs = [_unit(row) for row in raw]

    concepts: dict[str, Any] = {}
    for concept_id in concept_ids:
        primary = [c for c in comparisons if c["concept_id"] == concept_id and c["kind"] == "primary"]
        if not primary:
            continue
        concept_diffs = [vectors[c["positive_id"]].float() - vectors[c["negative_id"]].float() for c in primary]
        concept_dir, c_status = _fit_mean_direction(concept_diffs, min_norm)
        cos_with_stance = _json_number(float(concept_dir @ stance_dir), name="cos_with_stance") if (concept_dir is not None and stance_dir is not None) else None
        ortho_dir: torch.Tensor | None = None
        ortho_status = "skipped"
        if concept_dir is not None and stance_dir is not None:
            ortho_dir, ortho_status, _ = _orthogonalize(concept_dir, stance_dir, min_norm)
        company_raw = {cid: _json_number(float(vectors[cid].float() @ concept_dir), name="company_raw_score") for cid in company_ids} if concept_dir is not None else {}
        company_ortho = {cid: _json_number(float(vectors[cid].float() @ ortho_dir), name="company_orthogonalized_score") for cid in company_ids} if ortho_dir is not None else {}
        raw_values = list(company_raw.values())
        ortho_values = list(company_ortho.values())

        # leave-one-group-out (full space)
        groups = sorted({by_id[c["positive_id"]]["group_id"] for c in primary})
        loo: list[dict[str, Any]] = []
        for excluded in groups:
            fit_diffs = [vectors[c["positive_id"]].float() - vectors[c["negative_id"]].float() for c in primary if by_id[c["positive_id"]]["group_id"] != excluded]
            loo_dir, loo_status = _fit_mean_direction(fit_diffs, min_norm)
            deltas = []
            if loo_dir is not None:
                for c in primary:
                    if by_id[c["positive_id"]]["group_id"] == excluded:
                        deltas.append(_json_number(float(vectors[c["positive_id"]].float() @ loo_dir) - float(vectors[c["negative_id"]].float() @ loo_dir), name="loo_delta"))
            loo.append({"excluded_group": excluded, "status": loo_status, "group_mean_delta": _json_number(sum(deltas) / len(deltas), name="loo_mean") if deltas else None})

        # random baseline (full-space unit directions on this concept's primary pairs)
        random_deltas = []
        for d in random_dirs:
            for c in primary:
                random_deltas.append(float((vectors[c["positive_id"]].float() - vectors[c["negative_id"]].float()) @ d))
        random_q = _quantiles(random_deltas)

        concepts[concept_id] = {
            "status": c_status,
            "n_primary_pairs": len(primary),
            "cos_with_stance": cos_with_stance,
            "orthogonalized_status": ortho_status,
            "company_raw_scores": company_raw,
            "company_orthogonalized_scores": company_ortho,
            "company_raw_spread": _json_number(max(raw_values) - min(raw_values), name="raw_spread") if raw_values else None,
            "company_orthogonalized_spread": _json_number(max(ortho_values) - min(ortho_values), name="ortho_spread") if ortho_values else None,
            "leave_one_group_out": loo,
            "random_primary_q95": random_q["q95"],
        }

    stance_company_scores = {
        cid: _json_number(float(vectors[cid].float() @ stance_dir), name="stance_company_score") for cid in company_ids
    } if stance_dir is not None else {}

    return {
        "stance_status": stance_status,
        "stance_n_pairs": stance_n_pairs,
        "stance_company_scores": stance_company_scores,
        "concepts": concepts,
    }


def run_layer_scan(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    materials: Mapping[str, Any],
    instruction_suffix: str,
    answer_prefix: str,
    company_prompts: Any,
    layers: Sequence[int],
    run_id: str,
    model_name: str,
    artifact_root: str | Path,
    provenance: Mapping[str, Any],
    random_seed: int = 1729,
    random_count: int = 16,
    min_norm: float = 1e-6,
    k8_basis: torch.Tensor | None = None,
    k8_layer: int = 15,
) -> Path:
    """Run prepare → forward (multi-layer) → analyze (per-layer), then finalize."""
    concept_ids = [str(c) for c in materials.get("concepts", [])]
    if not concept_ids:
        raise ValueError("materials must declare concepts")
    layer_list = sorted({int(layer) for layer in layers})
    if not layer_list:
        raise ValueError("layers must be non-empty")
    last = int(model.n_layers) - 1
    for layer in layer_list:
        if layer < 0 or layer >= int(model.n_layers):
            raise ValueError(f"layer {layer} out of range for {model.n_layers} layers")
    rows = [dict(r) for r in materials["rows"]]
    comparisons = [dict(c) for c in materials["comparisons"]]
    prepared_rows = _material_prompt_records(tokenizer, rows, instruction_suffix=instruction_suffix, answer_prefix=answer_prefix)
    expected_instruction_ids = list(prepared_rows[0]["instruction_ids"])
    prepared_companies = _company_records(tokenizer, company_prompts, instruction_suffix=instruction_suffix, answer_prefix=answer_prefix, expected_instruction_ids=expected_instruction_ids)
    company_ids = [c["id"] for c in prepared_companies]
    capture_layers = sorted(set(layer_list) | {last})

    run = ArtifactRun.create(model_name, "entity-concept-layer-scan", run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            prepare_dir = run.run_directory / "prepare"
            write_json(prepare_dir / "materials.json", dict(materials), overwrite=False)
            write_jsonl(prepare_dir / "prompts.jsonl", [*prepared_rows, *prepared_companies], overwrite=False)
            metadata = {"purpose": "development", "provenance": dict(provenance), "sources": {k: materials.get(k) for k in ("source_paths", "source_sha256", "concepts") if materials.get(k) is not None}, "seed": random_seed, "layer_list": layer_list, "final_layer": last, "min_norm": min_norm}
            write_json(prepare_dir / "metadata.json", metadata, overwrite=False)
            _register_reference(run, provenance["upstream"], role="upstream", stage="prepare")
            _register_reference(run, provenance["lens"], role="lens", stage="prepare")
            stage.count(len(prepared_rows) + len(prepared_companies))

        vectors_by_layer: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layer_list}
        margins: dict[str, float] = {}
        records: list[dict[str, Any]] = []
        forward_inputs = [*prepared_rows, *prepared_companies]
        with run.stage("forward") as stage:
            for item in forward_inputs:
                started = time.perf_counter()
                ids, buy_id, sell_id = _answer_ids(tokenizer, item["formatted"])
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                captured = record_residuals(model, tensor, capture_layers)
                for layer in capture_layers:
                    if layer not in captured:
                        raise ValueError(f"missing capture for layer {layer} on {item['id']}")
                for layer in layer_list:
                    block = captured[layer][:, item["instruction_span"][0]:item["instruction_span"][1], :]
                    if block.ndim != 3 or block.shape[0] != 1 or block.shape[1] <= 0:
                        raise ValueError(f"invalid capture shape for {item['id']} layer {layer}")
                    vectors_by_layer[layer][item["id"]] = block.float().mean(dim=1).detach().cpu().reshape(-1)
                final = captured[last][:, -1, :]
                margin = _margin(model, final.float(), buy_id, sell_id)
                if item["kind"] == "company":
                    margins[item["id"]] = margin
                records.append({"id": item["id"], "kind": item["kind"], "margin": margin, "elapsed_seconds": time.perf_counter() - started})
                print(f"forward {len(records)}/{len(forward_inputs) + 1}: {item['id']}", flush=True)
                del captured, block, final
            first = prepared_rows[0]
            ids, buy_id, sell_id = _answer_ids(tokenizer, first["formatted"])
            captured = record_residuals(model, torch.tensor([ids], dtype=torch.long, device=device), capture_layers)
            repeat_max = 0.0
            for layer in layer_list:
                repeated = captured[layer][:, first["instruction_span"][0]:first["instruction_span"][1], :].float().mean(dim=1).detach().cpu().reshape(-1)
                repeat_max = max(repeat_max, float((repeated - vectors_by_layer[layer][first["id"]]).abs().max().item()))
            repeat_margin = _margin(model, captured[last][:, -1, :].float(), buy_id, sell_id)
            records.append({"id": first["id"] + "__repeat", "kind": "material_repeat", "repeat_max_abs_mean_diff": repeat_max, "margin_diff": abs(repeat_margin - records[0]["margin"])})
            records_path = run.run_directory / "forward" / "records.jsonl"
            count = write_jsonl(records_path, records, overwrite=False)
            write_json(run.run_directory / "forward" / "metadata.json", {"layer_list": layer_list, "final_layer": last, "record_count": count, "repeat_max_abs_mean_diff": repeat_max, "total_forward_seconds": sum(r["elapsed_seconds"] for r in records if "elapsed_seconds" in r), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0}, overwrite=False)
            run.manifest.register_artifact(records_path, artifact_type="layerscan_forward_records", stage="forward", record_count=count)
            run.manifest.register_artifact(run.run_directory / "forward" / "metadata.json", artifact_type="layerscan_forward_metadata", stage="forward")
            stage.count(count)

        with run.stage("analyze") as stage:
            per_layer = {str(layer): _analyze_layer(vectors_by_layer[layer], rows, comparisons, company_ids, concept_ids, random_seed=random_seed, random_count=random_count, min_norm=min_norm) for layer in layer_list}
            stance_characterization: dict[str, Any] = {}
            for layer in layer_list:
                scores = per_layer[str(layer)].get("stance_company_scores", {})
                pairs = [(scores[c], margins[c]) for c in company_ids if c in scores and margins.get(c) is not None]
                if len(pairs) < 3:
                    stance_characterization[str(layer)] = None
                    continue
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                stance_characterization[str(layer)] = {
                    "stance_vs_margin_r2": _pearson_r2(xs, ys),
                    "stance_base_rate": {"mean": _json_number(sum(xs) / len(xs), name="stance_mean"), "min": _json_number(min(xs), name="stance_min"), "max": _json_number(max(xs), name="stance_max")},
                }
            k8_characterization: dict[str, Any] | None = None
            if k8_basis is not None and k8_layer in layer_list:
                kb = k8_basis.detach().to(dtype=torch.float32).cpu().contiguous()
                if kb.ndim != 2 or kb.shape[0] != next(iter(vectors_by_layer[k8_layer].values())).numel():
                    raise ValueError(f"k8_basis shape mismatch: {tuple(kb.shape)} vs dim {next(iter(vectors_by_layer[k8_layer].values())).numel()}")
                k8_stance_dir, _ = _fit_stance_direction(vectors_by_layer[k8_layer], comparisons, min_norm)
                if k8_stance_dir is not None:
                    proj = kb.T @ k8_stance_dir
                    k8_coords = torch.stack([vectors_by_layer[k8_layer][c].float() @ kb for c in company_ids])
                    stance_scores = torch.stack([torch.tensor(vectors_by_layer[k8_layer][c].float() @ k8_stance_dir, dtype=torch.float32) for c in company_ids]).reshape(-1)
                    margin_vec = torch.tensor([margins[c] for c in company_ids], dtype=torch.float32)
                    top = int(torch.argmax(proj.abs()))
                    # Stance-vs-non-stance decomposition of k8's decision content.
                    proj_norm = float(proj.norm())
                    proj_hat = proj / proj_norm if proj_norm > 1e-9 else None
                    s_k8 = (k8_coords @ proj_hat) if proj_hat is not None else None
                    stance_share_r2 = _pearson_r2(s_k8.tolist(), margin_vec.tolist()) if s_k8 is not None else None
                    k8_share_r2 = _ols_r2(k8_coords, margin_vec)
                    non_stance_share_r2 = None
                    if stance_share_r2 is not None and k8_share_r2 is not None:
                        non_stance_share_r2 = _json_number(k8_share_r2 - stance_share_r2, name="non_stance_share_r2")
                    # Top non-stance direction (PCA of the k8 component orthogonal to stance).
                    ns_metrics: dict[str, Any] = {}
                    if proj_hat is not None:
                        k_ortho = k8_coords - torch.outer(s_k8, proj_hat)
                        _, _, vh = torch.linalg.svd(k_ortho, full_matrices=False)
                        ns_scores = k_ortho @ vh[0]
                        ns_metrics["top_dir_r2_vs_margin"] = _pearson_r2(ns_scores.tolist(), margin_vec.tolist())
                        g_scores = per_layer[str(k8_layer)]["concepts"].get("G", {}).get("company_raw_scores", {})
                        c_scores = per_layer[str(k8_layer)]["concepts"].get("C", {}).get("company_raw_scores", {})
                        for label, score_map in (("g", g_scores), ("c", c_scores)):
                            aligned = [(float(ns_scores[i]), score_map[company_ids[i]]) for i in range(len(company_ids)) if company_ids[i] in score_map]
                            if len(aligned) >= 3:
                                ns_metrics[f"top_dir_r_vs_{label}"] = _json_number(_pearson_r([a[0] for a in aligned], [a[1] for a in aligned]), name=f"ns_vs_{label}")
                    k8_characterization = {
                        "layer": k8_layer,
                        "stance_in_k8_fraction": _json_number(float(torch.sum(proj ** 2)), name="stance_in_k8_fraction"),
                        "stance_k8_coords": [ _json_number(float(v), name="k8_coord") for v in proj.tolist() ],
                        "stance_aligns_with_k8_axis": top,
                        "stance_coord_abs_max": _json_number(float(proj.abs().max()), name="stance_coord_abs_max"),
                        "r2_stance_1d_fullspace": _pearson_r2(stance_scores.tolist(), margin_vec.tolist()),
                        "r2_stance_in_k8_1d": stance_share_r2,
                        "r2_k8_ols": k8_share_r2,
                        "non_stance_share_r2": non_stance_share_r2,
                        "non_stance_top_dir": ns_metrics,
                        "company_k8_coords": [[_json_number(float(v), name="coord") for v in row.tolist()] for row in k8_coords],
                        "company_stance_in_k8": [ _json_number(float(v), name="s_k8") for v in s_k8.tolist() ] if s_k8 is not None else [],
                        "n_companies": len(company_ids),
                    }
            summary = {
                "schema_version": 1,
                "scientific_status": "not_evaluated",
                "purpose": "development",
                "layer_list": layer_list,
                "final_layer": last,
                "concept_ids": concept_ids,
                "company_margins": {cid: margins.get(cid) for cid in company_ids},
                "stance_characterization": stance_characterization,
                "k8_characterization": k8_characterization,
                "layers": per_layer,
                "claim_status": "descriptive",
            }
            summary_path = run.run_directory / "analyze" / "summary.json"
            write_json(summary_path, summary, overwrite=False)
            run.manifest.register_artifact(summary_path, artifact_type="layerscan_summary", stage="analyze")
            stage.count(len(layer_list))
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = ["run_layer_scan"]
