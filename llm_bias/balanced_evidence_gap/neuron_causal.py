"""Phase 3: causal validation of the 2C entity-specific MLP coordinates.

Frozen protocol: docs/balanced-evidence-gap/details/proposal-phase3.md (Rev 2).
Additively perturbs three MLP down-projection coordinates (L19/n6334,
L20/n6520, L26/n2394) and 30 matched control coordinates on the 16-ticker
2A canonical prompt set, and tests gate 3A (predicted direction, ticker
consistency under Holm, control superiority).

Stages: prepare (provenance + control re-derivation) -> pilot (2C
reproduction check, activation scale s, local-derivative diagnostic) ->
intervene (mlp_addition grid + margin scoring) -> analyze (gate 3A).
Only compact derived values are persisted (margins, deltas, percentiles,
test statistics); raw activations are never written.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from scipy.stats import rankdata

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.core.inference.mlp_addition import mlp_addition, mlp_summed_derivatives
from llm_bias.core.model import load_model

from .analysis import holm_adjusted
from .intervention import answer_token_ids, clean_margin, mlp_margin_attribution
from .template import ALL_TICKERS, DIAL_LAYER, DIAL_NEURON, DECISION_PREFIX, SECTOR_OF

SCHEMA_VERSION = "balanced-evidence-gap-phase3-v1"
DATASET = "balanced-evidence-gap-phase3"

# proposal §4.1 — frozen candidate coordinates (2C gate 2C passing layers,
# run phase2c-gate-reanalysis-01). predicted_sign: sign of the 2C spearman;
# delta > 0 should move the margin in this direction.
CANDIDATES: tuple[dict, ...] = (
    {"layer": 19, "neuron": 6334, "rho_2c": -0.8970588445663452, "predicted_sign": -1},
    {"layer": 20, "neuron": 6520, "rho_2c": 0.8941176533699036, "predicted_sign": +1},
    {"layer": 26, "neuron": 2394, "rho_2c": -0.8588235378265381, "predicted_sign": -1},
)

# proposal §4.2 — matched-control sampling rule identical to 2C.
CONTROL_SEED = 42
N_CONTROLS = 10
MOE_WIDTH = 9216

GATE_3A = {
    "holm_alpha": 0.05,
    "gate_multiple": 4.0,          # gate delta = predicted_sign * 4s
    "min_confirmed": 2,
    # Rev 2 (frozen 2026-09-10): 2C reproduction tolerances calibrated to the
    # measured cross-run bf16-backward jitter (max control |dRho| 0.0412,
    # max candidate |dRho| 0.0177). Wrong construct/position/index would
    # produce systematic deviations >> 0.05.
    "rho_check_tolerance": 0.05,
    "top_strength_tolerance": 0.05,
    "baseline_check_tolerance": 1e-6,
    "pilot_percentile": 90.0,
}
GRID_MULTIPLES = (1.0, 2.0, 4.0)   # proposal §4.5: {0, ±s, ±2s, ±4s}

CANDIDATE_LAYERS = tuple(sorted({c["layer"] for c in CANDIDATES}))


def coord_name(layer: int, neuron: int) -> str:
    return f"L{layer}_n{neuron}"


def derive_control_neurons(layer: int) -> list[int]:
    """proposal §4.2 — same RNG rule as 2C (seed 42+layer, range 9216)."""
    rng = random.Random(CONTROL_SEED + layer)
    return rng.sample(range(MOE_WIDTH), N_CONTROLS)


def one_sign_flip_p(n: int, k: int) -> float:
    """One-sided exact binomial: P(Binomial(n, 0.5) >= k)."""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def evaluate_gate_3a(*, candidates: list[dict]) -> dict:
    """Gate 3A (proposal §4.6) over per-candidate gate-delta statistics.

    candidates: [{"name", "predicted_sign", "mean_delta_at_gate",
    "n_same_direction", "n_tickers", "control_layer_max_abs_mean_delta"}].
    """
    if not candidates:
        raise ValueError("gate 3A requires at least one candidate")
    p_values = [one_sign_flip_p(c["n_tickers"], c["n_same_direction"]) for c in candidates]
    adjusted = holm_adjusted(p_values)
    evaluated = []
    for c, p, adj in zip(candidates, p_values, adjusted):
        mean = float(c["mean_delta_at_gate"])
        direction_ok = mean != 0.0 and math.copysign(1.0, mean) == c["predicted_sign"]
        consistency_ok = adj < GATE_3A["holm_alpha"]
        superiority_ok = abs(mean) > float(c["control_layer_max_abs_mean_delta"])
        evaluated.append(
            {
                **c,
                "sign_flip_p": p,
                "sign_flip_p_adjusted": adj,
                "criteria": {
                    "direction": direction_ok,
                    "consistency": consistency_ok,
                    "control_superiority": superiority_ok,
                },
                "pass": bool(direction_ok and consistency_ok and superiority_ok),
            }
        )
    confirmed = sum(1 for e in evaluated if e["pass"])
    return {
        "gate": "3A",
        "candidates": evaluated,
        "confirmed": confirmed,
        "min_confirmed": GATE_3A["min_confirmed"],
        "pass": confirmed >= GATE_3A["min_confirmed"],
        "formal": True,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_rows(phase2a_run: Path) -> dict[str, dict]:
    """2A canonical variant (reverse=False, order=0) per ticker."""
    rows = read_jsonl(phase2a_run / "forward" / "results.jsonl")
    canon = {
        r["ticker"]: r
        for r in rows
        if r.get("reverse") is False and int(r.get("order", -1)) == 0
    }
    missing = [t for t in ALL_TICKERS if t not in canon]
    if missing:
        raise ValueError(f"2A canonical rows missing tickers: {missing}")
    return canon


def _variant_margins(phase2a_run: Path) -> dict[str, float]:
    """2C-identical construct: per-ticker median over all 4 variants."""
    rows = read_jsonl(phase2a_run / "forward" / "results.jsonl")
    margins: dict[str, list[float]] = {}
    for row in rows:
        margins.setdefault(row["ticker"], []).append(float(row["margin"]))
    return {t: statistics.median(v) for t, v in margins.items()}


def _per_neuron_spearman(values: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """2C-identical per-neuron spearman (pearson of rankdata), [n, w] -> [w]."""
    with np.errstate(invalid="ignore"):
        rank_vals = rankdata(values, axis=0)
        rank_margs = rankdata(margins)
        centered_v = rank_vals - rank_vals.mean(axis=0, keepdims=True)
        centered_m = rank_margs - rank_margs.mean()
        denom = np.sqrt((centered_v**2).sum(axis=0)) * np.sqrt((centered_m**2).sum())
        return np.where(denom > 0, centered_v.T @ centered_m / np.where(denom == 0, 1.0, denom), 0.0)


@contextmanager
def _summed_margin_derivative(
    model: Any, input_tensor: torch.Tensor, layers: list[int], buy_id: int, sell_id: int
) -> Iterator[dict[int, torch.Tensor]]:
    """Summed dM/da per layer across all positions (transient, one forward)."""
    with mlp_summed_derivatives(model, layers) as records:
        final_box: dict[str, torch.Tensor] = {}
        final_layer = int(model.n_layers) - 1
        handle = model.layers[final_layer].register_forward_hook(
            lambda _module, _inputs, output: final_box.update(
                residual=(output if torch.is_tensor(output) else output[0])
            )
        )
        try:
            attention_mask = torch.ones_like(input_tensor)
            with torch.enable_grad():
                try:
                    model.forward(input_tensor, attention_mask=attention_mask)
                except TypeError:
                    model.forward(input_tensor)
                residual = final_box["residual"]
                h = residual[0, -1, :].float()
                h_norm = model._final_norm(h)
                weight_delta = (
                    model._lm_head.weight[buy_id].float()
                    - model._lm_head.weight[sell_id].float()
                )
                margin = torch.dot(h_norm, weight_delta.to(h_norm.device))
                margin.backward()
        finally:
            handle.remove()
        for layer in layers:
            if layer not in records:
                raise RuntimeError(f"derivative hook did not fire for layer {layer}")
            if not torch.isfinite(records[layer]).all():
                raise ValueError(f"non-finite summed derivative at layer {layer}")
        yield records


def activation_scales(
    abs_values: dict[int, list[torch.Tensor]],
    coord_keys: list[tuple[int, int]],
) -> dict[str, float]:
    """Per-coordinate percentile-|a| across prompts x positions.

    Each ``abs_values[layer]`` is a list of ``[seq, n_neurons]`` tensors
    (one per prompt; prompts differ in length). Chunks are concatenated
    into ``[total_seq, n_neurons]`` and the percentile is taken over all
    (prompt, position) values.
    """
    s: dict[str, float] = {}
    for layer, chunks in abs_values.items():
        matrix = torch.cat(chunks, dim=0)  # [total_seq, n_neurons]
        if matrix.dim() != 2:
            raise ValueError(f"expected [total_seq, n_neurons], got {tuple(matrix.shape)}")
        for j, neuron in enumerate(
            [n for (l, n) in coord_keys if l == layer]
        ):
            col = matrix[:, j]
            s[coord_name(layer, neuron)] = float(
                torch.quantile(col.float(), GATE_3A["pilot_percentile"] / 100.0)
            )
    return s


@contextmanager
def _capture_abs_values(
    model: Any, coords_by_layer: dict[int, list[int]]
) -> Iterator[dict[int, list[torch.Tensor]]]:
    """Capture |a| at given (layer, neurons) across all positions (transient)."""
    records: dict[int, list[torch.Tensor]] = {}
    handles = []
    for layer, neurons in sorted(coords_by_layer.items()):
        def hook(_module: Any, args: tuple[Any, ...], layer: int, neurons: list[int]) -> None:
            values = args[0]
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError("require [1, sequence, width] MLP input")
            if any(n >= values.shape[-1] for n in neurons):
                raise ValueError("neuron index out of range")
            selected = values[0, :, neurons].detach().float().abs().cpu()
            if not torch.isfinite(selected).all():
                raise ValueError("non-finite MLP values")
            records.setdefault(layer, []).append(selected)
            return None

        handles.append(
            dense_down_projection(model.layers[layer]).register_forward_pre_hook(
                lambda _module, args, layer=layer, neurons=neurons: hook(_module, args, layer, neurons)
            )
        )
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


# ── stages ───────────────────────────────────────────────────────────────────


def check_c2c_reproduction(rho: np.ndarray, ref_layer: dict, candidate_neuron: int) -> dict:
    """Rev 2 2C reproduction check for one layer (proposal §4.2, frozen).

    ``rho``: recomputed per-neuron spearman [width]. ``ref_layer``: the 2C
    stored layer summary. Returns the check results; caller fails closed on
    any ``checks`` value being False.
    """
    controls = [
        n for n in derive_control_neurons(int(ref_layer["layer"]))
    ]
    control_diffs = [abs(float(rho[n]) - c) for n, c in zip(controls, ref_layer["control_rhos"])]
    top = int(np.argmax(np.abs(rho)))
    top_strength = float(np.abs(rho)[top])
    candidate_rho = float(rho[candidate_neuron])
    checks = {
        "controls": max(control_diffs) <= GATE_3A["rho_check_tolerance"],
        "candidate": abs(candidate_rho - float(ref_layer["top_spearman"]))
        <= GATE_3A["rho_check_tolerance"],
        "top_strength": top_strength
        >= abs(float(ref_layer["top_spearman"])) - GATE_3A["top_strength_tolerance"],
    }
    return {
        "top_neuron": top,
        "top_abs_rho": top_strength,
        "c2c_top_neuron": ref_layer["top_neuron"],
        "c2c_top_abs_rho": abs(float(ref_layer["top_spearman"])),
        "candidate_rho": candidate_rho,
        "candidate_rho_abs_diff": abs(candidate_rho - float(ref_layer["top_spearman"])),
        "control_rho_max_abs_diff": max(control_diffs),
        "checks": checks,
        "pass": all(checks.values()),
    }


def prepare_stage(
    run: ArtifactRun,
    *,
    phase2a_run: Path,
    phase2c_run: Path,
    phase2c_reanalysis_run: Path,
) -> None:
    reanalysis_summary = json.loads(
        (phase2c_reanalysis_run / "analyze" / "summary.json").read_text(encoding="utf-8")
    )
    mlp_arm = reanalysis_summary["gate_2c"]["mlp_arm"]
    per_layer = {int(k): v for k, v in mlp_arm["per_layer"].items()}
    expected_layers = sorted(c["layer"] for c in CANDIDATES)
    if mlp_arm["passing_layers"] != expected_layers:
        raise ValueError(
            f"2C passing layers {mlp_arm['passing_layers']} != frozen candidates {expected_layers}"
        )
    for c in CANDIDATES:
        d = per_layer[c["layer"]]
        if d["top_neuron"] != c["neuron"]:
            raise ValueError(
                f"2C top neuron at L{c['layer']} = {d['top_neuron']} != frozen {c['neuron']}"
            )
        if abs(d["top_spearman"] - c["rho_2c"]) > 1e-12:
            raise ValueError(f"2C rho at L{c['layer']} drifted from frozen value")

    controls: dict[int, list[int]] = {}
    for layer in CANDIDATE_LAYERS:
        neurons = derive_control_neurons(layer)
        top = next(c["neuron"] for c in CANDIDATES if c["layer"] == layer)
        if top in neurons:
            raise ValueError(f"re-derived controls for L{layer} contain the top neuron")
        controls[layer] = neurons

    canon = _canonical_rows(phase2a_run)
    baseline = {t: float(canon[t]["margin"]) for t in ALL_TICKERS}

    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "balanced_evidence_gap_phase3_prepare_provenance",
        "raw_runtime_payloads": False,
        "source_runs": {
            "phase2a": {
                "run_id": phase2a_run.name,
                "analyze_summary_sha256": _sha256(phase2a_run / "analyze" / "summary.json"),
                "forward_results_sha256": _sha256(phase2a_run / "forward" / "results.jsonl"),
            },
            "phase2c": {
                "run_id": phase2c_run.name,
                "mlp_layer_summaries_sha256": _sha256(
                    phase2c_run / "mlp" / "layer_summaries.json"
                ),
            },
            "phase2c_reanalysis": {
                "run_id": phase2c_reanalysis_run.name,
                "analyze_summary_sha256": _sha256(
                    phase2c_reanalysis_run / "analyze" / "summary.json"
                ),
            },
        },
        "candidates": [dict(c) for c in CANDIDATES],
        "control_seed": CONTROL_SEED,
        "n_controls": N_CONTROLS,
        "moe_width": MOE_WIDTH,
        "controls": {str(k): v for k, v in sorted(controls.items())},
        "baseline_margins": {t: baseline[t] for t in ALL_TICKERS},
        "protocol": "docs/balanced-evidence-gap/details/proposal-phase3.md (Rev 2, frozen)",
    }
    with run.stage("prepare") as stage:
        path = out_dir / "provenance.json"
        write_json(path, provenance, overwrite=True)
        run.manifest.register_artifact(
            path, artifact_type="balanced_evidence_gap_phase3_prepare_provenance",
            stage="prepare", role="output",
        )
        stage.count(len(ALL_TICKERS))


def pilot_stage(
    run: ArtifactRun,
    model_path: str,
    *,
    phase2a_run: Path,
    phase2c_run: Path,
    smoke: bool,
) -> dict:
    model, tokenizer, device = load_model(model_path, dtype=None)
    out_dir = run.run_directory / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with run.stage("pilot") as stage:
            # proposal §4.2 — sample-space assertion.
            for layer in (*CANDIDATE_LAYERS, DIAL_LAYER):
                width = int(dense_down_projection(model.layers[layer]).in_features)
                if width != MOE_WIDTH:
                    raise ValueError(
                        f"L{layer} down-projection in_features={width} != {MOE_WIDTH}"
                    )

            controls: dict[int, list[int]] = {}
            for c in CANDIDATES:
                controls.setdefault(c["layer"], derive_control_neurons(c["layer"]))

            # proposal §4.4 — coordinates: 3 candidates + 30 controls + dial.
            coords_by_layer: dict[int, list[int]] = {}
            for c in CANDIDATES:
                coords_by_layer.setdefault(c["layer"], []).append(c["neuron"])
            for layer, neurons in controls.items():
                coords_by_layer.setdefault(layer, []).extend(neurons)
            coords_by_layer.setdefault(DIAL_LAYER, []).append(DIAL_NEURON)
            coord_keys: list[tuple[int, int]] = [
                (layer, n) for layer, ns in sorted(coords_by_layer.items()) for n in ns
            ]

            tickers = ALL_TICKERS[:4] if smoke else ALL_TICKERS
            canon = _canonical_rows(phase2a_run)
            variant_margins = _variant_margins(phase2a_run)

            # (a) 2C reproduction check (skipped for smoke: 16-ticker construct).
            reproduction: dict = {"status": "skipped_smoke"} if smoke else {}
            if not smoke:
                c2c = json.loads(
                    (phase2c_run / "mlp" / "layer_summaries.json").read_text(encoding="utf-8")
                )["layers"]
                stored = {int(x["layer"]): x for x in c2c}
                reproduction = {"status": "run", "layers": {}}
                for layer in CANDIDATE_LAYERS:
                    stacked = []
                    for ticker in ALL_TICKERS:
                        row = canon[ticker]
                        scoring_text = row["formatted"] + DECISION_PREFIX
                        prompt_ids, buy_id, sell_id = answer_token_ids(tokenizer, scoring_text)
                        tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                        result = mlp_margin_attribution(
                            model, tensor,
                            layer=layer, position=int(row["entity_position"]),
                            buy_id=buy_id, sell_id=sell_id,
                        )
                        stacked.append(result["attribution"].cpu().numpy())
                    stacked = np.stack(stacked)
                    margins = np.array([variant_margins[t] for t in ALL_TICKERS])
                    rho = _per_neuron_spearman(stacked, margins)
                    ref = stored[layer]
                    candidate = next(c for c in CANDIDATES if c["layer"] == layer)
                    result = check_c2c_reproduction(rho, ref, candidate["neuron"])
                    reproduction["layers"][str(layer)] = result
                    if not result["pass"]:
                        raise ValueError(
                            f"2C reproduction failed at L{layer}: "
                            f"top={result['top_neuron']} (|rho|={result['top_abs_rho']:.4f}, "
                            f"2C top |rho|={result['c2c_top_abs_rho']:.4f}), "
                            f"candidate |dRho|={result['candidate_rho_abs_diff']:.4f}, "
                            f"control max |dRho|={result['control_rho_max_abs_diff']:.4f}, "
                            f"checks={result['checks']}"
                        )

            # (b) activation scale s + summed-derivative diagnostic.
            abs_values: dict[int, list[torch.Tensor]] = {}
            derivatives: dict[tuple[int, int], list[float]] = {}
            margin_diffs: list[float] = []
            for i, ticker in enumerate(tickers):
                row = canon[ticker]
                scoring_text = row["formatted"] + DECISION_PREFIX
                prompt_ids, buy_id, sell_id = answer_token_ids(tokenizer, scoring_text)
                tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                with _capture_abs_values(model, coords_by_layer) as cap:
                    with _summed_margin_derivative(model, tensor, list(CANDIDATE_LAYERS), buy_id, sell_id) as records:
                        pass
                for layer, chunks in cap.items():
                    abs_values.setdefault(layer, []).extend(chunks)
                for c in CANDIDATES:
                    if c["layer"] in CANDIDATE_LAYERS:
                        derivatives.setdefault((c["layer"], c["neuron"]), []).append(
                            float(records[c["layer"]][c["neuron"]])
                        )
                # margin cross-check: the differentiable forward's margin must
                # equal the stored 2A baseline for this ticker.
                with torch.no_grad():
                    margin_live = clean_margin(model, tokenizer, scoring_text, device=device)
                margin_diffs.append(abs(margin_live - float(row["margin"])))
                if (i + 1) % 4 == 0:
                    print(f"  pilot {i + 1}/{len(tickers)}", flush=True)

            if max(margin_diffs) > GATE_3A["baseline_check_tolerance"]:
                raise ValueError(
                    f"live margin deviates from 2A baseline by {max(margin_diffs):.3e} "
                    f"> {GATE_3A['baseline_check_tolerance']:.0e}"
                )

            s = activation_scales(abs_values, coord_keys)

            grid: dict[str, dict] = {}
            for c in CANDIDATES:
                scale = s[coord_name(c["layer"], c["neuron"])]
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"non-positive activation scale for {coord_name(c['layer'], c['neuron'])}")
                grid[coord_name(c["layer"], c["neuron"])] = {
                    "s": scale,
                    "deltas": [
                        (mult if sign > 0 else -mult) * scale
                        for mult in GRID_MULTIPLES
                        for sign in (1, -1)
                    ],
                    "gate_delta": c["predicted_sign"] * GATE_3A["gate_multiple"] * scale,
                }

            derivative_diag = {
                coord_name(c["layer"], c["neuron"]): {
                    "mean_summed_dM_da": statistics.fmean(
                        derivatives[(c["layer"], c["neuron"])]
                    ),
                    "predicted_sign": c["predicted_sign"],
                    "sign_match": (
                        statistics.fmean(derivatives[(c["layer"], c["neuron"])])
                        * c["predicted_sign"]
                    ) > 0,
                }
                for c in CANDIDATES
            }

            summary = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase3_pilot",
                "n_prompts": len(tickers),
                "smoke": smoke,
                "width": MOE_WIDTH,
                "scale_s": s,
                "grid": grid,
                "dial_scale_s": s[coord_name(DIAL_LAYER, DIAL_NEURON)],
                "derivative_diagnostic": derivative_diag,
                "margin_baseline_max_abs_diff": max(margin_diffs),
                "c2c_reproduction": reproduction,
                "raw_runtime_payloads": False,
            }
            path = out_dir / "summary.json"
            write_json(path, summary, overwrite=True)
            run.manifest.register_artifact(
                path, artifact_type="balanced_evidence_gap_phase3_pilot",
                stage="pilot", role="output",
            )
            write_metadata(
                out_dir / "metadata.json",
                {"schema_version": SCHEMA_VERSION,
                 "artifact_type": "balanced_evidence_gap_phase3_pilot_metadata",
                 "n_coordinates": len(coord_keys), "smoke": smoke},
                overwrite=True,
            )
            run.manifest.register_artifact(
                out_dir / "metadata.json",
                artifact_type="balanced_evidence_gap_phase3_pilot_metadata",
                stage="pilot", role="output",
            )
            stage.count(len(tickers))
            return summary
    finally:
        del model


def intervene_stage(
    run: ArtifactRun,
    model_path: str,
    *,
    phase2a_run: Path,
    smoke: bool,
) -> None:
    model, tokenizer, device = load_model(model_path, dtype=None)
    pilot = json.loads((run.run_directory / "pilot" / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((run.run_directory / "prepare" / "provenance.json").read_text(encoding="utf-8"))
    out_dir = run.run_directory / "intervene"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with run.stage("intervene") as stage:
            tickers = ALL_TICKERS[:4] if smoke else ALL_TICKERS
            canon = _canonical_rows(phase2a_run)
            grid = pilot["grid"]
            controls: dict[str, list[int]] = {
                int(k): v for k, v in provenance["controls"].items()
            }

            def score(ticker: str, layer: int | None, neuron: int | None, delta: float) -> float:
                text = canon[ticker]["formatted"] + DECISION_PREFIX
                if layer is None or delta == 0.0:
                    return clean_margin(model, tokenizer, text, device=device)
                with mlp_addition(model, int(layer), int(neuron), float(delta)):
                    return clean_margin(model, tokenizer, text, device=device)

            records: list[dict] = []
            total = (
                len(tickers)
                + len(tickers) * 6 * len(CANDIDATES)
                + len(tickers) * 2 * sum(len(v) for v in controls.values())
                + len(tickers) * 2  # dial descriptive
            )
            done = 0

            # baseline M(0): recompute, cross-check against the frozen 2A value.
            m0: dict[str, float] = {}
            for ticker in tickers:
                value = score(ticker, None, None, 0.0)
                stored = float(provenance["baseline_margins"][ticker])
                if abs(value - stored) > GATE_3A["baseline_check_tolerance"]:
                    raise ValueError(
                        f"baseline drift for {ticker}: {value} vs 2A {stored}"
                    )
                m0[ticker] = value
                records.append({
                    "phase": "3a", "kind": "baseline", "layer": None, "neuron": None,
                    "delta": 0.0, "ticker": ticker, "sector": SECTOR_OF[ticker],
                    "margin": value, "delta_m": 0.0,
                    "decision": "buy" if value > 0 else "sell",
                })
                done += 1

            for c in CANDIDATES:
                name = coord_name(c["layer"], c["neuron"])
                for delta in grid[name]["deltas"]:
                    for ticker in tickers:
                        value = score(ticker, c["layer"], c["neuron"], delta)
                        records.append({
                            "phase": "3a", "kind": "candidate", "layer": c["layer"],
                            "neuron": c["neuron"], "delta": delta, "ticker": ticker,
                            "sector": SECTOR_OF[ticker], "margin": value,
                            "delta_m": value - m0[ticker],
                            "decision": "buy" if value > 0 else "sell",
                        })
                        done += 1
                        if done % 32 == 0:
                            print(f"  intervene {done}/{total}", flush=True)

            for layer in sorted(controls):
                scale = grid[coord_name(layer, next(c["neuron"] for c in CANDIDATES if c["layer"] == layer))]["s"]
                for neuron in controls[layer]:
                    for delta in (GATE_3A["gate_multiple"] * scale, -GATE_3A["gate_multiple"] * scale):
                        for ticker in tickers:
                            value = score(ticker, layer, neuron, delta)
                            records.append({
                                "phase": "3a", "kind": "control", "layer": layer,
                                "neuron": neuron, "delta": delta, "ticker": ticker,
                                "sector": SECTOR_OF[ticker], "margin": value,
                                "delta_m": value - m0[ticker],
                                "decision": "buy" if value > 0 else "sell",
                            })
                            done += 1
                            if done % 32 == 0:
                                print(f"  intervene {done}/{total}", flush=True)

            # dial descriptive arm (proposal §4.7): known model-level prior.
            dial_scale = pilot["dial_scale_s"]
            for delta in (GATE_3A["gate_multiple"] * dial_scale, -GATE_3A["gate_multiple"] * dial_scale):
                for ticker in tickers:
                    value = score(ticker, DIAL_LAYER, DIAL_NEURON, delta)
                    records.append({
                        "phase": "3a", "kind": "dial", "layer": DIAL_LAYER,
                        "neuron": DIAL_NEURON, "delta": delta, "ticker": ticker,
                        "sector": SECTOR_OF[ticker], "margin": value,
                        "delta_m": value - m0[ticker],
                        "decision": "buy" if value > 0 else "sell",
                    })
                    done += 1

            path = out_dir / "records.jsonl"
            count = write_jsonl(path, records, overwrite=True)
            write_metadata(
                out_dir / "metadata.json",
                {"schema_version": SCHEMA_VERSION,
                 "artifact_type": "balanced_evidence_gap_phase3_intervene_metadata",
                 "n_records": count, "smoke": smoke,
                 "baseline_m0": m0},
                overwrite=True,
            )
            run.manifest.register_artifact(
                path, artifact_type="balanced_evidence_gap_phase3_intervene",
                stage="intervene", role="output", record_count=count,
            )
            run.manifest.register_artifact(
                out_dir / "metadata.json",
                artifact_type="balanced_evidence_gap_phase3_intervene_metadata",
                stage="intervene", role="output",
            )
            stage.count(count)
    finally:
        del model


def analyze_3a_records(
    records: list[dict], pilot: dict, provenance: dict
) -> dict:
    """Pure gate-3A analysis over intervene records (no artifact IO)."""
    grid = pilot["grid"]

    by_ticker = {r["ticker"]: float(r["margin"]) for r in records if r["kind"] == "baseline"}
    n_tickers = len(by_ticker)
    if n_tickers == 0:
        raise ValueError("no baseline records")

    def deltas(kind: str, layer: int, neuron: int) -> dict[float, list[float]]:
        out: dict[float, list[float]] = {}
        for r in records:
            if r["kind"] == kind and r["layer"] == layer and r["neuron"] == neuron:
                out.setdefault(r["delta"], []).append(float(r["delta_m"]))
        return out

    candidate_specs = []
    for c in CANDIDATES:
        name = coord_name(c["layer"], c["neuron"])
        gate_delta = float(grid[name]["gate_delta"])
        per_ticker = []
        for r in records:
            if (r["kind"] == "candidate" and r["layer"] == c["layer"]
                    and r["neuron"] == c["neuron"] and abs(r["delta"] - gate_delta) < 1e-9):
                per_ticker.append(float(r["delta_m"]))
        if len(per_ticker) != n_tickers:
            raise ValueError(f"gate-delta records incomplete for {name}: {len(per_ticker)}/{n_tickers}")
        n_same = sum(
            1 for d in per_ticker if d != 0.0 and math.copysign(1.0, d) == c["predicted_sign"]
        )
        candidate_specs.append({
            "name": name,
            "layer": c["layer"],
            "neuron": c["neuron"],
            "predicted_sign": c["predicted_sign"],
            "gate_delta": gate_delta,
            "mean_delta_at_gate": statistics.fmean(per_ticker),
            "n_same_direction": n_same,
            "n_tickers": n_tickers,
        })

    # control layer stats: per control, max |mean delta_m| over its two points.
    control_layer_max: dict[int, float] = {}
    for layer in sorted({c["layer"] for c in CANDIDATES}):
        layer_max = 0.0
        for neuron in provenance["controls"][str(layer)]:
            by_delta = deltas("control", layer, neuron)
            means = [statistics.fmean(v) for v in by_delta.values()]
            layer_max = max(layer_max, max(abs(m) for m in means))
        control_layer_max[layer] = layer_max
    for spec in candidate_specs:
        spec["control_layer_max_abs_mean_delta"] = control_layer_max[spec["layer"]]

    gate = evaluate_gate_3a(candidates=candidate_specs)

    # ── descriptive blocks (proposal §4.7) ──
    flips: dict[str, dict[str, int]] = {}
    for c in CANDIDATES:
        name = coord_name(c["layer"], c["neuron"])
        flips[name] = {
            f"{d:+g}": sum(
                1 for r in records
                if r["kind"] == "candidate" and r["layer"] == c["layer"]
                and r["neuron"] == c["neuron"] and abs(r["delta"] - d) < 1e-9
                and r["decision"] == "buy"
            )
            for d in grid[name]["deltas"]
        }

    sector_gate: dict[str, dict[str, float]] = {}
    for c in CANDIDATES:
        name = coord_name(c["layer"], c["neuron"])
        per_sector: dict[str, list[float]] = {}
        for r in records:
            if (r["kind"] == "candidate" and r["layer"] == c["layer"]
                    and r["neuron"] == c["neuron"]
                    and abs(r["delta"] - float(grid[name]["gate_delta"])) < 1e-9):
                per_sector.setdefault(r["sector"], []).append(float(r["delta_m"]))
        sector_gate[name] = {s: statistics.fmean(v) for s, v in sorted(per_sector.items())}

    linearity: dict[str, list[dict]] = {}
    deriv = pilot["derivative_diagnostic"]
    for c in CANDIDATES:
        name = coord_name(c["layer"], c["neuron"])
        d_local = deriv[name]["mean_summed_dM_da"]
        by_delta = deltas("candidate", c["layer"], c["neuron"])
        linearity[name] = [
            {
                "delta": d,
                "mean_delta_m": statistics.fmean(by_delta[d]),
                "first_order_ratio": (
                    statistics.fmean(by_delta[d]) / (d * d_local)
                    if d_local != 0 else None
                ),
            }
            for d in grid[name]["deltas"]
            if d in by_delta and by_delta[d]
        ]

    dial_stats = {
        d: {
            "mean_delta_m": statistics.fmean(deltas("dial", DIAL_LAYER, DIAL_NEURON)[d]),
        }
        for d in deltas("dial", DIAL_LAYER, DIAL_NEURON)
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "balanced_evidence_gap_phase3_analysis",
        "n_tickers": n_tickers,
        "baseline_m0": by_ticker,
        "gate_3a": gate,
        "descriptive": {
            "decision_flips": flips,
            "sector_mean_at_gate": sector_gate,
            "linearity_vs_first_order": linearity,
            "dial_comparison": {
                "coordinate": [DIAL_LAYER, DIAL_NEURON],
                "deltas": dial_stats,
                "descriptive_only": True,
            },
            "pilot": {
                "derivative_diagnostic": pilot["derivative_diagnostic"],
                "c2c_reproduction": pilot["c2c_reproduction"],
                "margin_baseline_max_abs_diff": pilot["margin_baseline_max_abs_diff"],
            },
        },
        "descriptive_only": not gate["pass"],
        "raw_runtime_payloads": False,
    }
    return summary


def analyze_stage(run: ArtifactRun) -> None:
    records = read_jsonl(run.run_directory / "intervene" / "records.jsonl")
    pilot = json.loads((run.run_directory / "pilot" / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((run.run_directory / "prepare" / "provenance.json").read_text(encoding="utf-8"))
    summary = analyze_3a_records(records, pilot, provenance)
    n_tickers = summary["n_tickers"]
    gate = summary["gate_3a"]
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("analyze") as stage:
        path = out_dir / "summary.json"
        write_json(path, summary, overwrite=True)
        run.manifest.register_artifact(
            path, artifact_type="balanced_evidence_gap_phase3_analysis",
            stage="analyze", role="output",
        )
        write_metadata(
            out_dir / "metadata.json",
            {"schema_version": SCHEMA_VERSION,
             "artifact_type": "balanced_evidence_gap_phase3_analysis_metadata",
             "gate_3a_pass": gate["pass"],
             "confirmed": gate["confirmed"]},
            overwrite=True,
        )
        run.manifest.register_artifact(
            out_dir / "metadata.json",
            artifact_type="balanced_evidence_gap_phase3_analysis_metadata",
            stage="analyze", role="output",
        )
        stage.count(n_tickers)


def run_phase3(
    *,
    model_path: str,
    phase2a_run: str | Path,
    phase2c_run: str | Path,
    phase2c_reanalysis_run: str | Path,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    smoke: bool = False,
) -> Path:
    phase2a_run = Path(phase2a_run)
    phase2c_run = Path(phase2c_run)
    phase2c_reanalysis_run = Path(phase2c_reanalysis_run)
    for p in (phase2a_run, phase2c_run, phase2c_reanalysis_run):
        if not p.is_dir():
            raise FileNotFoundError(f"run root not found: {p}")

    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        prepare_stage(
            run,
            phase2a_run=phase2a_run,
            phase2c_run=phase2c_run,
            phase2c_reanalysis_run=phase2c_reanalysis_run,
        )
        pilot_stage(
            run, model_path,
            phase2a_run=phase2a_run, phase2c_run=phase2c_run, smoke=smoke,
        )
        intervene_stage(run, model_path, phase2a_run=phase2a_run, smoke=smoke)
        analyze_stage(run)
        run.finalize(required_stages={"prepare", "pilot", "intervene", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
