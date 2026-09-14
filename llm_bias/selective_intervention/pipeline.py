"""Selective-intervention V1 pipeline: prepare → forward → analyze.

Protocol: docs/selective-intervention/proposal-v1.md (Rev 1, frozen
2026-09-13). One run covers the full arm matrix: clean bit-exact
reference, strength (dose) sweep, centering / position / layer /
random-subspace controls, and the anonymous + dial probes. No raw
states are persisted; centers are transient with SHA-256 digests.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.model import load_model, load_tokenizer

from . import template as T
from .analysis import decision_flips, evaluate_gates, group_gap, iqr, per_ticker_margins
from .scoring import answer_token_ids, margin_forward, scoring_ids
from .spans import anonymous_prompt, _anonymous_row
from .subspace import (
    align_grid,
    calibration_centers,
    load_e01_basis,
    random_orthonormal_basis,
    subspace_removal_transform,
    tensor_sha256,
)


# ── shared helpers ────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


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


def _forward_device(model: Any) -> Any:
    return (
        model.input_device
        if hasattr(model, "input_device")
        else "cuda" if torch.cuda.is_available() else "cpu"
    )


def _decision(margin: float) -> str:
    return "buy" if margin > 0 else "sell"


def _record(
    *,
    row: dict,
    arm: str,
    alpha: float,
    centering: str | None,
    layer: int | None,
    position_scope: str | None,
    margin: float,
    noop: bool,
    extra: dict | None = None,
) -> dict:
    rec = {
        "prompt_id": row["id"],
        "ticker": row["ticker"],
        "sector": row["sector"],
        "reverse": int(bool(row["reverse"])),
        "order": int(row["order"]),
        "arm": arm,
        "alpha": alpha,
        "centering": centering,
        "layer": layer,
        "position_scope": position_scope,
        "margin": margin,
        "decision": _decision(margin),
        "noop": noop,
    }
    if extra:
        rec.update(extra)
    return rec


def _ref_prompt_id(records: list[dict], tag: str) -> str:
    """The dial-probe reference prompt id for one dial tag."""
    for r in records:
        if r["arm"] == f"dial_{tag}_clean":
            return r["prompt_id"]
    raise ValueError(f"dial probe records missing for tag {tag}")


# ── prepare ───────────────────────────────────────────────────────────────────


def _prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    phase2a_run: Path,
    e01_run: Path,
    smoke: bool,
) -> dict:
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        rows = read_jsonl(phase2a_run / "prepare" / "prompts.jsonl")
        archive = {
            r["id"]: float(r["margin"])
            for r in read_jsonl(phase2a_run / "forward" / "results.jsonl")
        }
        if len(rows) != T.N_PROMPTS or len(archive) != T.N_PROMPTS:
            raise ValueError(
                f"2A population must be {T.N_PROMPTS} prompts, got {len(rows)}/{len(archive)}"
            )
        if smoke:
            wanted = set(T.SMOKE_COMPANIES)
            rows = [r for r in rows if r["ticker"] in wanted]
            if {r["ticker"] for r in rows} != wanted:
                raise ValueError(f"smoke companies missing from 2A population: {sorted(wanted)}")

        # Population invariants (fail-closed).
        tickers = sorted({r["ticker"] for r in rows})
        if len(tickers) * T.N_2A_VARIANTS != len(rows):
            raise ValueError("each ticker must have exactly 4 (reverse, order) variants")
        for ticker in tickers:
            variants = {
                (bool(r["reverse"]), int(r["order"])) for r in rows if r["ticker"] == ticker
            }
            if len(variants) != T.N_2A_VARIANTS:
                raise ValueError(f"{ticker}: expected 4 variants, got {sorted(variants)}")
            lengths = {len(r["formatted"]) for r in rows if r["ticker"] == ticker}
            spans = {tuple(r["instruction_span"]) for r in rows if r["ticker"] == ticker}
            if len(lengths) != 1 or len(spans) != 1:
                raise ValueError(f"{ticker}: variant lengths/spans are not identical")
        span_lengths = {
            int(r["instruction_span"][1]) - int(r["instruction_span"][0]) for r in rows
        }
        if len(span_lengths) != 1:
            raise ValueError(
                f"instruction span lengths not 1:1 aligned across companies: {sorted(span_lengths)}"
            )
        instruction_span_length = span_lengths.pop()
        if instruction_span_length <= 0:
            raise ValueError("instruction span is empty")

        ref_row = next(r for r in rows if r["reverse"] is False and r["order"] == 0)

        # Basis (e-01 artifact, fail-closed) + frozen-seed random control basis.
        e01_summary = e01_run / "analyze" / "summary.json"
        basis16, singular = load_e01_basis(e01_summary, expected_k=T.E01_PCA_DIM)
        d = basis16.shape[0]
        basis_k = basis16[:, : T.K_PRIMARY]
        basis_random = random_orthonormal_basis(d, T.K_PRIMARY, T.RANDOM_SEED)

        # Anonymous probe row (unified-header identity check, fail-closed).
        # Canonical rows only: evidence-order / option-order variants change
        # the prompt body below the header and must not enter this check.
        canon_rows = [
            r for r in rows if r["reverse"] is False and r["order"] == 0
        ]
        anon = _anonymous_row(tokenizer, ref_row)
        anon_string = anon["prompt"]
        for r in canon_rows:
            if r["ticker"] == ref_row["ticker"] and r["reverse"] is False and r["order"] == 0:
                continue
            candidate = anonymous_prompt(r["prompt"], r["ticker"], r["name"])
            if candidate != anon_string:
                raise ValueError(
                    f"anonymous prompt for {r['ticker']} differs from the anchor string "
                    "(template not unified; protocol requires a new version)"
                )
        anon_sha = hashlib.sha256(anon_string.encode("utf-8")).hexdigest()

        provenance = {
            "schema_version": T.SCHEMA_VERSION,
            "artifact_type": "selective_intervention_v1_provenance",
            "protocol": T.PROTOCOL,
            "protocol_rev": T.PROTOCOL_REV,
            "smoke": smoke,
            "raw_runtime_payloads": False,
            "phase2a_run": _verify_upstream(
                phase2a_run,
                {
                    "prepare/prompts.jsonl": T.N_PROMPTS,
                    "forward/results.jsonl": T.N_PROMPTS,
                    "analyze/summary.json": None,
                },
            ),
            "e01_run": _verify_upstream(e01_run, {"analyze/summary.json": None}),
            "subspace_basis": {
                "source_run": e01_run.name,
                "k": int(basis_k.shape[1]),
                "d": int(basis_k.shape[0]),
                "sha256": tensor_sha256(basis_k),
                "singular_range": [float(singular[0]), float(singular[-1])],
            },
            "random_control_basis": {
                "k": int(basis_random.shape[1]),
                "seed": T.RANDOM_SEED,
                "sha256": tensor_sha256(basis_random),
            },
            "anonymous_prompt_sha256": anon_sha,
            "anonymous_anchor": ref_row["ticker"],
            "frozen_groups": {"top": sorted(T.TOP_GROUP), "bottom": sorted(T.BOTTOM_GROUP)},
            "intervention_layer": T.INTERVENTION_LAYER,
            "control_layers": list(T.CONTROL_LAYERS if not smoke else T.SMOKE_CONTROL_LAYERS),
            "alpha_grid": list(T.ALPHA_GRID if not smoke else T.SMOKE_ALPHA_GRID),
            "instruction_span_length": instruction_span_length,
        }
        rows_path = out_dir / "rows.jsonl"
        count = write_jsonl(rows_path, rows + [anon], overwrite=True)
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": T.SCHEMA_VERSION,
                "artifact_type": "selective_intervention_v1_prepare",
                "n_rows": count,
                "instruction_span_length": instruction_span_length,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            rows_path, artifact_type="selective_intervention_v1_prepare", stage="prepare",
            role="output", record_count=count,
        )
        run.manifest.register_artifact(
            out_dir / "provenance.json", artifact_type="selective_intervention_v1_provenance",
            stage="prepare", role="output",
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="selective_intervention_v1_prepare_metadata",
            stage="prepare", role="output",
        )

    return {
        "rows": rows,
        "archive": archive,
        "ref_row": ref_row,
        "anon_row": anon,
        "basis_k": basis_k,
        "basis_random": basis_random,
        "instruction_span_length": instruction_span_length,
    }


# ── forward ───────────────────────────────────────────────────────────────────


def _forward(
    run: ArtifactRun,
    model_path: str,
    ctx: dict,
    *,
    smoke: bool,
) -> None:
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    started = time.time()
    with run.stage("forward") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        rows = ctx["rows"]
        archive = ctx["archive"]
        ref_row = ctx["ref_row"]
        canon = [r for r in rows if r["reverse"] is False and r["order"] == 0]
        basis_k = ctx["basis_k"].to(device=device, dtype=torch.float32)
        basis_random = ctx["basis_random"].to(device=device, dtype=torch.float32)
        alpha_grid = T.SMOKE_ALPHA_GRID if smoke else T.ALPHA_GRID
        control_layers = T.SMOKE_CONTROL_LAYERS if smoke else T.CONTROL_LAYERS
        layers = tuple(sorted({T.INTERVENTION_LAYER, *control_layers}))

        # Sequences + answer token ids (per row; anon included).
        seqs: dict[str, tuple[torch.Tensor, tuple[int, int]]] = {}
        for row in rows + [ctx["anon_row"]]:
            ids = scoring_ids(tokenizer, row["formatted"])
            seqs[row["id"]] = (
                torch.tensor([ids], dtype=torch.long, device=device),
                answer_token_ids(tokenizer, row["formatted"] + T.DECISION_PREFIX),
            )

        records: list[dict] = []
        counters = {"arms": 0}

        def run_margin(row: dict, transforms: dict[int, Any] | None = None, **kw) -> float:
            tensor, (buy_id, sell_id) = seqs[row["id"]]
            margin = margin_forward(model, tensor, transforms, buy_id, sell_id, **kw)
            counters["arms"] += 1
            if not torch.isfinite(torch.tensor(margin)):
                raise ValueError(f"non-finite margin for {row['id']}")
            return margin

        # 1) Clean reference arm: bit-exact against the 2A archive. The per-prompt
        #    check runs before any intervention forward (fail-closed pre-check).
        n_clean_bit_exact = 0
        clean_max_abs_delta = 0.0
        for row in rows:
            margin = run_margin(row)
            stored = archive[row["id"]]
            if abs(margin - stored) > T.BITEXACT_TOLERANCE:
                raise ValueError(
                    f"clean margin for {row['id']} is not bit-exact vs 2A archive: "
                    f"{margin:.8f} vs {stored:.8f}"
                )
            n_clean_bit_exact += 1
            clean_max_abs_delta = max(clean_max_abs_delta, abs(margin - stored))
            records.append(
                _record(row=row, arm="clean", alpha=0.0, centering=None, layer=None,
                        position_scope=None, margin=margin, noop=True)
            )

        # 2) Calibration centers (transient; digests recorded in metadata).
        ref_seq_len = len(scoring_ids(tokenizer, ref_row["formatted"]))
        calib = calibration_centers(
            model,
            tokenizer,
            canon,
            ref_row,
            ctx["anon_row"],
            layers=list(layers),
            anon_layer=T.INTERVENTION_LAYER,
            ref_seq_len=ref_seq_len,
            device=device,
        )
        n_calib = len(canon) + 1

        # Per-company center materialization (shared across arms/alphas/variants).
        centers: dict[str, dict] = {}
        for row in canon:
            span = (int(row["instruction_span"][0]), int(row["instruction_span"][1]))
            seq_len = len(seqs[row["id"]][0][0].tolist())
            entry: dict[str, Any] = {"span": span, "seq_len": seq_len}
            for layer in layers:
                entry[f"instr_cloud_{layer}"] = align_grid(
                    calib["mu_bar"][layer], calib["ref_span"], span
                )
            entry["instr_anon"] = align_grid(calib["mu_anon"], calib["anon_span"], span)
            entry["full_cloud"] = align_grid(
                calib["mu_full_ref"], (0, ref_seq_len), (0, seq_len)
            )
            centers[row["ticker"]] = entry

        anon = ctx["anon_row"]
        anon_span = (int(anon["instruction_span"][0]), int(anon["instruction_span"][1]))
        centers["__anon__"] = {
            "span": anon_span,
            "seq_len": len(seqs[anon["id"]][0][0].tolist()),
            "instr_cloud": align_grid(
                calib["mu_bar"][T.INTERVENTION_LAYER], calib["ref_span"], anon_span
            ),
        }

        def arm_forward(
            row: dict,
            *,
            arm: str,
            alpha: float,
            basis: torch.Tensor,
            centering: str,
            center_grid: dict[int, torch.Tensor] | None,
            layer: int = T.INTERVENTION_LAYER,
            scope: str = "instruction",
            dial_delta: float | None = None,
        ) -> None:
            positions = (
                tuple(range(int(row["instruction_span"][0]), int(row["instruction_span"][1])))
                if scope == "instruction"
                else tuple(range(centers[row["ticker"]]["seq_len"]))
            )
            if scope == "full" and center_grid is None:
                raise ValueError("full scope requires a center grid")
            transform = subspace_removal_transform(basis, alpha, center_grid, positions)
            margin = run_margin(
                row,
                {layer: transform} if alpha > 0 else None,
                **({"mlp_delta": dial_delta, "mlp_layer": T.DIAL_LAYER,
                    "mlp_neuron": T.DIAL_NEURON} if dial_delta is not None else {}),
            )
            records.append(
                _record(row=row, arm=arm, alpha=alpha, centering=centering, layer=layer,
                        position_scope=scope, margin=margin, noop=False,
                        extra=({"dial_delta": dial_delta} if dial_delta is not None else None))
            )

        instr_key = f"instr_cloud_{T.INTERVENTION_LAYER}"

        # 3) Dose sweep (primary centering: entity-contrast cloud means).
        for alpha in alpha_grid:
            for row in rows:
                arm_forward(row, arm=f"dose_{round(alpha * 100)}", alpha=alpha,
                            basis=basis_k, centering="cloud",
                            center_grid=centers[row["ticker"]][instr_key])

        # 4) Centering controls (full strength).
        for row in rows:
            arm_forward(row, arm="center_zero", alpha=1.0, basis=basis_k,
                        centering="zero", center_grid=None)
        for row in rows:
            arm_forward(row, arm="center_anon", alpha=1.0, basis=basis_k,
                        centering="anon", center_grid=centers[row["ticker"]]["instr_anon"])

        # 5) Position scope control (full sequence, full strength).
        for row in rows:
            arm_forward(row, arm="scope_fullseq", alpha=1.0, basis=basis_k,
                        centering="cloud", center_grid=centers[row["ticker"]]["full_cloud"],
                        scope="full")

        # 6) Random-subspace specificity control (full strength).
        for row in rows:
            arm_forward(row, arm="ctrl_random", alpha=1.0, basis=basis_random,
                        centering="cloud", center_grid=centers[row["ticker"]][instr_key])

        # 7) Layer controls (canonical variants only, full strength).
        for layer in control_layers:
            for row in canon:
                arm_forward(row, arm=f"ctrl_layer_{layer}", alpha=1.0, basis=basis_k,
                            centering="cloud",
                            center_grid=centers[row["ticker"]][f"instr_cloud_{layer}"],
                            layer=layer)

        # 8) Anonymous probe (clean + full-strength intervention).
        margin_anon_clean = run_margin(anon)
        records.append(
            _record(row=anon, arm="anon_clean", alpha=0.0, centering=None, layer=None,
                    position_scope=None, margin=margin_anon_clean, noop=True)
        )
        transform_anon = subspace_removal_transform(
            basis_k, 1.0, centers["__anon__"]["instr_cloud"],
            tuple(range(anon_span[0], anon_span[1])),
        )
        margin_anon_int = run_margin(anon, {T.INTERVENTION_LAYER: transform_anon})
        records.append(
            _record(row=anon, arm="anon_int", alpha=1.0, centering="cloud",
                    layer=T.INTERVENTION_LAYER, position_scope="instruction",
                    margin=margin_anon_int, noop=False)
        )

        # 9) Dial probe (reference row; ±4 native units, clean + full-strength).
        dial_kw = {"mlp_layer": T.DIAL_LAYER, "mlp_neuron": T.DIAL_NEURON}
        for delta in T.DIAL_PROBE_DELTAS:
            tag = f"{delta:+.0f}"
            margin_clean = run_margin(ref_row, mlp_delta=delta, **dial_kw)
            records.append(
                _record(row=ref_row, arm=f"dial_{tag}_clean", alpha=0.0, centering=None,
                        layer=None, position_scope=None, margin=margin_clean, noop=False,
                        extra={"dial_delta": delta})
            )
            transform_ref = subspace_removal_transform(
                basis_k, 1.0, centers[ref_row["ticker"]][instr_key],
                tuple(range(int(ref_row["instruction_span"][0]), int(ref_row["instruction_span"][1]))),
            )
            margin_int = run_margin(ref_row, {T.INTERVENTION_LAYER: transform_ref},
                                    mlp_delta=delta, **dial_kw)
            records.append(
                _record(row=ref_row, arm=f"dial_{tag}_int", alpha=1.0, centering="cloud",
                        layer=T.INTERVENTION_LAYER, position_scope="instruction",
                        margin=margin_int, noop=False, extra={"dial_delta": delta})
            )

        write_jsonl(path, records, overwrite=True)
        n_forwards = counters["arms"] + n_calib
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": T.SCHEMA_VERSION,
                "artifact_type": "selective_intervention_v1_forward",
                "n_records": len(records),
                "n_forwards": n_forwards,
                "n_arm_forwards": counters["arms"],
                "n_calib_forwards": n_calib,
                "n_clean_bit_exact": n_clean_bit_exact,
                "clean_max_abs_delta_m": clean_max_abs_delta,
                "smoke": smoke,
                "raw_runtime_payloads": False,
                "centers": {str(layer): tensor_sha256(calib["mu_bar"][layer]) for layer in layers},
                "mu_anon_sha256": tensor_sha256(calib["mu_anon"]),
                "mu_full_ref_sha256": tensor_sha256(calib["mu_full_ref"]),
                "instruction_span_length": ctx["instruction_span_length"],
                "m_anon_clean": margin_anon_clean,
                "m_anon_int": margin_anon_int,
                "runtime_seconds": round(time.time() - started, 1),
            },
            overwrite=True,
        )
        run.manifest.register_artifact(
            path, artifact_type="selective_intervention_v1_forward", stage="forward",
            role="output", record_count=len(records),
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json", artifact_type="selective_intervention_v1_forward_metadata",
            stage="forward", role="output",
        )
        stage.count(len(records))


# ── analyze ───────────────────────────────────────────────────────────────────


def _arm_records(records: list[dict], arm: str) -> list[dict]:
    return [r for r in records if r["arm"] == arm]


def _per_ticker(records: list[dict]) -> dict[str, float]:
    by_ticker: dict[str, list[float]] = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(float(r["margin"]))
    return per_ticker_margins(by_ticker)


def _analyze(run: ArtifactRun, *, smoke: bool) -> dict:
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(run.run_directory / "forward" / "records.jsonl")
    forward_meta = json.loads(
        (run.run_directory / "forward" / "metadata.json").read_text(encoding="utf-8")
    )
    alpha_grid = T.SMOKE_ALPHA_GRID if smoke else T.ALPHA_GRID
    with run.stage("analyze") as stage:
        clean = _arm_records(records, "clean")
        main_arm = _arm_records(records, f"dose_{round(max(alpha_grid) * 100)}")
        random_arm = _arm_records(records, "ctrl_random")
        if len(clean) != len(main_arm) or len(main_arm) != len(random_arm):
            raise ValueError("clean/main/random arm record counts differ")

        clean_pt = _per_ticker(clean)
        main_pt = _per_ticker(main_arm)
        random_pt = _per_ticker(random_arm)
        m_anon_clean = float(forward_meta["m_anon_clean"])
        m_anon_int = float(forward_meta["m_anon_int"])
        clean_mean = statistics.fmean(r["margin"] for r in clean)

        if smoke:
            gate: Any = {
                "status": "not_evaluated",
                "reason": "smoke run: gates are evaluated on formal data only",
            }
        else:
            gate = evaluate_gates(
                {"per_ticker": clean_pt, "all_margins": [r["margin"] for r in clean]},
                {"per_ticker": main_pt, "all_margins": [r["margin"] for r in main_arm]},
                {"per_ticker": random_pt, "all_margins": [r["margin"] for r in random_arm]},
                m_anon_clean,
                m_anon_int,
            )

        # Dose response (primary arm family).
        clean_decisions = {r["prompt_id"]: r["decision"] for r in clean}
        dose_response = []
        for alpha in alpha_grid:
            dose = _arm_records(records, f"dose_{round(alpha * 100)}")
            pt = _per_ticker(dose)
            decisions = {r["prompt_id"]: r["decision"] for r in dose}
            dose_response.append({
                "alpha": alpha,
                "group_gap": group_gap(pt),
                "iqr": iqr(pt.values()),
                "mean_margin": statistics.fmean(r["margin"] for r in dose),
                "flips_sell_to_buy": decision_flips(clean_decisions, decisions),
                "entity_contrast_iqr": iqr([pt[t] - m_anon_clean for t in pt]),
            })

        # Per-company table.
        per_company = []
        for ticker in sorted(clean_pt):
            clean_recs = [r for r in clean if r["ticker"] == ticker]
            int_recs = [r for r in main_arm if r["ticker"] == ticker]
            clean_vals = [r["margin"] for r in clean_recs]
            int_vals = [r["margin"] for r in int_recs]
            per_company.append({
                "ticker": ticker,
                "sector": clean_recs[0]["sector"],
                "clean_mean": statistics.fmean(clean_vals),
                "int_mean": statistics.fmean(int_vals),
                "clean_contrast": statistics.fmean(clean_vals) - m_anon_clean,
                "int_contrast": statistics.fmean(int_vals) - m_anon_clean,
                "flips": decision_flips(
                    {r["prompt_id"]: r["decision"] for r in clean_recs},
                    {r["prompt_id"]: r["decision"] for r in int_recs},
                ),
            })

        # Centering comparison (full strength).
        centering_comparison = []
        for arm, label in (("dose_100", "cloud"), ("center_zero", "zero"), ("center_anon", "anon")):
            arm_recs = _arm_records(records, arm)
            pt = _per_ticker(arm_recs)
            centering_comparison.append({
                "centering": label,
                "group_gap": group_gap(pt),
                "mean_shift": statistics.fmean(r["margin"] for r in arm_recs) - clean_mean,
            })

        # Position scope comparison.
        full_recs = _arm_records(records, "scope_fullseq")
        position_scope = {
            "instruction": {
                "group_gap": group_gap(main_pt),
                "mean_shift": statistics.fmean(r["margin"] for r in main_arm) - clean_mean,
            },
            "full": {
                "group_gap": group_gap(_per_ticker(full_recs)),
                "mean_shift": statistics.fmean(r["margin"] for r in full_recs) - clean_mean,
            },
        }

        # Layer control (canonical variants; L15 reference = main arm restricted).
        control_layers = T.SMOKE_CONTROL_LAYERS if smoke else T.CONTROL_LAYERS
        canon_ids = {r["prompt_id"] for r in clean if r["reverse"] == 0 and r["order"] == 0}
        main_sub = [r for r in main_arm if r["prompt_id"] in canon_ids]
        layer_control = [{"layer": T.INTERVENTION_LAYER,
                          "group_gap": group_gap(_per_ticker(main_sub))}]
        for layer in control_layers:
            recs = _arm_records(records, f"ctrl_layer_{layer}")
            if not recs:
                raise ValueError(f"layer control records missing for L{layer}")
            layer_control.append({"layer": layer, "group_gap": group_gap(_per_ticker(recs))})
        layer_control.sort(key=lambda item: item["layer"])

        # Random-subspace specificity control (descriptive; G2 is computed in
        # evaluate_gates from the same records).
        random_recs = _arm_records(records, "ctrl_random")
        if not random_recs:
            raise ValueError("ctrl_random records missing")
        random_control = {
            "group_gap": group_gap(_per_ticker(random_recs)),
            "mean_shift": statistics.fmean(r["margin"] for r in random_recs) - clean_mean,
        }

        # Dial probe (ΔM vs the same-row clean / main-arm margin).
        dial_probe = {}
        for delta in T.DIAL_PROBE_DELTAS:
            tag = f"{delta:+.0f}"
            prompt_id = _ref_prompt_id(records, tag)
            clean_ref = float(next(r["margin"] for r in clean if r["prompt_id"] == prompt_id))
            main_ref = float(next(r["margin"] for r in main_arm if r["prompt_id"] == prompt_id))
            dial_probe[tag] = {
                "clean_delta_m": float(next(r["margin"] for r in records
                                            if r["arm"] == f"dial_{tag}_clean")) - clean_ref,
                "int_delta_m": float(next(r["margin"] for r in records
                                          if r["arm"] == f"dial_{tag}_int")) - main_ref,
            }

        # Order consistency (per-company ord1 − ord0, max |shift| clean→int).
        def order_diff(recs: list[dict]) -> dict[str, float]:
            out: dict[str, dict[int, list[float]]] = {}
            for r in recs:
                out.setdefault(r["ticker"], {}).setdefault(r["order"], []).append(r["margin"])
            return {
                t: statistics.fmean(v[1]) - statistics.fmean(v[0])
                for t, v in out.items() if 0 in v and 1 in v
            }

        clean_od = order_diff(clean)
        int_od = order_diff(main_arm)
        order_consistency = {
            "max_abs_shift": max(abs(int_od[t] - clean_od[t]) for t in clean_od),
        }

        summary = {
            "schema_version": T.SCHEMA_VERSION,
            "artifact_type": "selective_intervention_v1_summary",
            "smoke": smoke,
            "gate_v1": gate,
            "clean_stats": {
                "group_gap": group_gap(clean_pt),
                "iqr": iqr(clean_pt.values()),
                "mean_margin": clean_mean,
                "n_prompts": len(clean),
                "decisions": {
                    d: sum(1 for r in clean if r["decision"] == d) for d in ("buy", "sell")
                },
            },
            "dose_response": dose_response,
            "per_company": per_company,
            "centering_comparison": centering_comparison,
            "position_scope": position_scope,
            "layer_control": layer_control,
            "random_control": random_control,
            "dial_probe": dial_probe,
            "anon_probe": {
                "clean": m_anon_clean,
                "intervention": m_anon_int,
                "delta": m_anon_int - m_anon_clean,
            },
            "order_consistency": order_consistency,
            "n_forwards": forward_meta["n_forwards"],
            "runtime_seconds": forward_meta["runtime_seconds"],
            "raw_runtime_payloads": False,
        }
        path = out_dir / "summary.json"
        write_json(path, summary, overwrite=True)
        run.manifest.register_artifact(
            path, artifact_type="selective_intervention_v1_summary", stage="analyze", role="output"
        )
    return summary


# ── entry point ───────────────────────────────────────────────────────────────


def run_selective_intervention_v1(
    *,
    model_path: str,
    run_id: str,
    phase2a_run: str | Path,
    e01_run: str | Path,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    e01_run = Path(e01_run)
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, T.DATASET, run_id, artifact_root=artifact_root)
    try:
        ctx = _prepare(run, tokenizer, phase2a_run=phase2a_run, e01_run=e01_run, smoke=smoke)
        _forward(run, model_path, ctx, smoke=smoke)
        _analyze(run, smoke=smoke)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
