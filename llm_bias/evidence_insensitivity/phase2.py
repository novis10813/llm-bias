"""Evidence-insensitivity Phase 2: capture-layer group contrast (descriptive).

Frozen protocol: ``docs/evidence-insensitivity/details/proposal-phase2.md``
(Rev 1.1). Two-model execution: Qwen3.5-4B at capture layer L15 (frozen
anchor from entity-to-dial / BEG 2B), Gemma-4-E2B at L* localized by the
pre-registered Step A rule. Pure-forward state capture at the prompt's final
token (== instruction-span final token, verified at prepare), in-memory only,
compact derived outputs (stance projections, PCA loadings, norms). No raw
activations persisted. Descriptive: no intervention, no causal claims.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.forward import capture_position_residuals, encode_batch
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import input_ids

import torch

from .pipeline import DATASET, MODEL_SLUG, _model_identity

PHASE2_CONDITIONS = ("zero", "N15", "P15")
CONDITION_V = {"zero": 0.0, "N15": -4.0, "P15": 4.0}
DETERMINISM_COUNT = 20
DETERMINISM_STRATUM = {"N15": 7, "P15": 7, "zero": 6}
DETERMINISM_TOL = 0.05
STANCE_R2_MIN = 0.30
GROUP_POWER_MIN = 10
STEP_A_SIZE = 16
STEP_A_MIN_PER_GROUP = 8
PCA_TOP_K = 5
DEFAULT_CAPTURE_LAYER = {"qwen3.5-4b": 15}
BATCH_SIZE = 8
PHASE1_DEFAULT_RUN = "phase1-gpu-bf16-01"
PROTOCOL = "proposal-phase2 Rev 1.1 frozen"


def _finite_float(value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("non-finite float in Phase 2 output")
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _welch(a: list[float], b: list[float]) -> dict[str, Any]:
    if len(a) < 2 or len(b) < 2:
        return {"t": 0.0, "df": 0.0, "p": 1.0, "cohen_d": 0.0}
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    if va == 0.0 and vb == 0.0:
        return {"t": 0.0, "df": 0.0, "p": 1.0, "cohen_d": 0.0}
    se = np.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return {"t": 0.0, "df": 0.0, "p": 1.0, "cohen_d": 0.0}
    t = (np.mean(a) - np.mean(b)) / se
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    pooled = np.sqrt((va + vb) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), df))
    return {"t": _finite_float(t), "df": _finite_float(df), "p": _finite_float(p), "cohen_d": _finite_float(d)}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_capture_companies(groups: list[dict[str, Any]], split: str = "discovery") -> list[dict[str, str]]:
    """Discovery-split capture list, fail-closed on group-table integrity."""
    if len(groups) != 503 or len({g["ticker"] for g in groups}) != 503:
        raise ValueError(f"group table must have 503 unique companies, got {len(groups)}")
    labeled = sum(1 for g in groups if g.get("group") is not None)
    if labeled < 0.95 * len(groups):
        raise ValueError(f"too many unlabeled companies: {labeled}/{len(groups)}")
    selected = [
        {"ticker": g["ticker"], "group": g["group"], "sector": g["gics_sector"]}
        for g in groups
        if g.get("split") == split and g.get("group") is not None
    ]
    if not selected:
        raise ValueError(f"no {split} companies")
    return selected


def select_step_a_subsample(
    groups: list[dict[str, Any]],
    split: str = "discovery",
    seed: int = 20260916,
    size: int = STEP_A_SIZE,
) -> list[str]:
    """Pre-registered Step A subsample: stratified by group, half/half when
    both groups have >= 8 discovery companies; otherwise take the smaller
    group fully and top up from the larger (seeded)."""
    discovery = [g for g in groups if g.get("split") == split and g.get("group") is not None]
    by_group: dict[str, list[str]] = {}
    for g in discovery:
        by_group.setdefault(g["group"], []).append(g["ticker"])
    for tickers in by_group.values():
        tickers.sort()
    if len(by_group) == 0:
        raise ValueError("no discovery companies for Step A")
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for _name, tickers in sorted(by_group.items(), key=lambda kv: len(kv[1])):
        take = len(tickers) if len(tickers) < STEP_A_MIN_PER_GROUP else size // len(by_group)
        shuffled = list(tickers)
        rng.shuffle(shuffled)
        picked.extend(shuffled[:take])
    if len(picked) < size:
        pool = sorted(t for tickers in by_group.values() for t in tickers if t not in picked)
        rng.shuffle(pool)
        picked.extend(pool[: size - len(picked)])
    if len(picked) != size or len(set(picked)) != size:
        raise ValueError(f"Step A subsample malformed: {len(picked)} rows, {len(set(picked))} unique")
    return sorted(picked)


# ---------------------------------------------------------------------------
# State-space math (in-memory; nothing raw is persisted)
# ---------------------------------------------------------------------------


def derive_stance_axis(p15: np.ndarray, n15: np.ndarray) -> np.ndarray:
    """d_stance = normalize(mean_c [s(c,P15) - s(c,N15)]), inputs (n, d)."""
    if p15.shape != n15.shape or p15.ndim != 2:
        raise ValueError("stance axis inputs must be equal (n, d) matrices")
    mean = (p15 - n15).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("stance axis has zero norm")
    return (mean / norm).astype(np.float32)


def fit_offset_gain(r_zero: float, r_n15: float, r_p15: float) -> tuple[float, float, float]:
    """Least squares on v/4 in {-1, 0, +1}: r = offset + gain * (v/4).

    Returns (offset, gain, nonlinearity); nonlinearity is the maximum
    absolute residual over the three points.
    """
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([r_n15, r_zero, r_p15], dtype=np.float64)
    design = np.vstack([x, np.ones(3)]).T
    sol, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    gain, offset = (float(v) for v in sol)
    resid = y - (offset + gain * x)
    return offset, gain, float(np.max(np.abs(resid)))


def layer_localization(
    states_by_layer: dict[int, dict[str, dict[str, np.ndarray]]],
    c_c_by_ticker: dict[str, float],
) -> tuple[int, dict[int, float]]:
    """Pre-registered Step A rule: per layer L, d_L = normalize of the mean
    polarity difference over the subsample; per-company scalar
    x_{c,L} = <s(c,P15)-s(c,N15), d_L>; L* = argmax_L |pearson(x_L, C_c)|,
    ties to the shallower layer; undefined correlations count as 0.0."""
    if not states_by_layer:
        raise ValueError("empty layer sweep")
    first_layer = next(iter(states_by_layer))
    tickers = sorted(states_by_layer[first_layer]["P15"])
    corrs: dict[int, float] = {}
    for layer in sorted(states_by_layer):
        p15, n15 = states_by_layer[layer]["P15"], states_by_layer[layer]["N15"]
        if set(p15) != set(tickers) or set(n15) != set(tickers):
            raise ValueError(f"layer {layer} sweep incomplete")
        diffs = np.stack([p15[t] - n15[t] for t in tickers])
        d = diffs.mean(axis=0)
        norm = float(np.linalg.norm(d))
        if norm == 0.0 or not np.isfinite(norm):
            corrs[layer] = 0.0
            continue
        x = diffs @ (d / norm)
        y = np.array([c_c_by_ticker[t] for t in tickers])
        corrs[layer] = _finite_float(_pearson(x, y))
    best = max(corrs, key=lambda L: (abs(corrs[L]), -L))
    return int(best), {L: _finite_float(c) for L, c in sorted(corrs.items())}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _phase1_run_directory(phase2_run_dir: Path, phase1_run_id: str) -> Path:
    """Phase 1 run directory: sibling of the Phase 2 run under the same model dataset."""
    return phase2_run_dir.parent / phase1_run_id


def prepare_stage(
    run: ArtifactRun,
    *,
    model_path: str,
    tokenizer: Any | None = None,
    model_slug: str | None = None,
    phase1_run_id: str = PHASE1_DEFAULT_RUN,
) -> Path:
    slug = model_slug or MODEL_SLUG
    tokenizer = tokenizer or load_tokenizer(model_path)
    phase1_dir = _phase1_run_directory(run.run_directory, phase1_run_id)
    prompts = read_jsonl(phase1_dir / "prepare" / "prompts.jsonl")
    summary_path = phase1_dir / "analyze" / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Phase 1 summary not found: {summary_path}")
    phase1 = json.loads(summary_path.read_text(encoding="utf-8"))
    groups = phase1["groups"]
    capture = select_capture_companies(groups, "discovery")
    step_a = [] if slug in DEFAULT_CAPTURE_LAYER else select_step_a_subsample(groups)
    default_layer = DEFAULT_CAPTURE_LAYER.get(slug)

    # prompt selection from Phase 1 primary arm, byte-identical prompts.
    index = {(p["ticker"], p["condition"]): p for p in prompts if p["arm"] == "primary"}
    by_ticker = {c["ticker"]: c for c in capture}
    selected: dict[tuple[str, str], str] = {}
    missing = []
    for ticker, c in by_ticker.items():
        for cond in PHASE2_CONDITIONS:
            row = index.get((ticker, cond))
            if row is None:
                missing.append(f"{ticker}:{cond}")
                continue
            selected[(ticker, cond)] = row["prompt_id"]
    if missing:
        raise ValueError(f"Phase 1 prompts missing for Phase 2 conditions: {missing[:5]}")

    # capture position = instruction-span final token (index span_end - 1).
    # Chat-template suffixes (assistant prefix / empty think block) may follow
    # the user turn, so the span end is generally NOT the prompt's final token;
    # verify it lies strictly inside the encoded sequence for every prompt.
    positions: dict[tuple[str, str], int] = {}
    for (ticker, cond), prompt_id in sorted(selected.items()):
        row = next(p for p in prompts if p["prompt_id"] == prompt_id)
        ids = input_ids(tokenizer, row["prompt_text"], add_special_tokens=True)
        span_end = row["instruction_span"][1]
        if not (1 <= span_end <= len(ids)):
            raise ValueError(f"capture position out of bounds for {ticker}:{cond}: span_end={span_end}, seq={len(ids)}")
        positions[(ticker, cond)] = span_end - 1

    selection = {
        "protocol": PROTOCOL,
        "model_slug": slug,
        "phase1_run_id": phase1_run_id,
        "split": "discovery",
        "conditions": list(PHASE2_CONDITIONS),
        "n_capture": len(capture),
        "capture": capture,
        "prompt_ids": {f"{t}:{c}": pid for (t, c), pid in sorted(selected.items())},
        "positions": {f"{t}:{c}": pos for (t, c), pos in sorted(positions.items())},
        "step_a": {"enabled": default_layer is None, "companies": step_a},
        "capture_layer": default_layer,
        "splits": {g: sum(1 for c in capture if c["group"] == g) for g in sorted({c["group"] for c in capture})},
    }
    if default_layer is None and not step_a:
        raise ValueError("model requires Step A but subsample is empty")
    output = run.run_directory / "prepare"
    output.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        sel_path = write_json(output / "selection.json", selection, overwrite=True)
        run.manifest.register_artifact(sel_path, artifact_type="evidence_insensitivity_phase2_selection", stage="prepare", record_count=len(capture))
        stage.count(len(selected))
    return sel_path


def _chunked(rows: list[tuple[str, str]], prompts_by_key: dict[tuple[str, str], str], positions_by_key: dict[tuple[str, str], int]):
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start : start + BATCH_SIZE]
        pos = torch.tensor([positions_by_key[k] for k in chunk], dtype=torch.long)
        yield chunk, [prompts_by_key[k] for k in chunk], pos


def forward_stage(
    run: ArtifactRun,
    *,
    model_path: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: Any = "cpu",
    model_slug: str | None = None,
) -> Path:
    slug = model_slug or MODEL_SLUG
    if model is None or tokenizer is None:
        model, tokenizer, device = load_model(model_path, dtype=None)
    selection = json.loads((run.run_directory / "prepare" / "selection.json").read_text(encoding="utf-8"))
    phase1_dir = _phase1_run_directory(run.run_directory, selection["phase1_run_id"])
    prompts = {p["prompt_id"]: p for p in read_jsonl(phase1_dir / "prepare" / "prompts.jsonl")}
    prompts_by_key = {
        (k.split(":")[0], k.split(":")[1]): prompts[pid]["prompt_text"]
        for k, pid in selection["prompt_ids"].items()
    }
    positions_by_key = {
        (k.split(":")[0], k.split(":")[1]): int(pos)
        for k, pos in selection["positions"].items()
    }
    capture_layer = selection.get("capture_layer")
    output = run.run_directory / "forward"
    output.mkdir(parents=True, exist_ok=True)

    # ---- Step A: layer localization (models without a frozen anchor) ----
    layer_sweep = None
    if capture_layer is None:
        phase1 = json.loads((phase1_dir / "analyze" / "summary.json").read_text(encoding="utf-8"))
        c_c = {g["ticker"]: g["contrast_c"] for g in phase1["groups"] if g.get("contrast_c") is not None}
        companies = selection["step_a"]["companies"]
        sweep_rows = [(t, c) for t in companies for c in ("P15", "N15")]
        states_by_layer: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        for chunk, prompts_text, pos in _chunked(sweep_rows, prompts_by_key, positions_by_key):
            ids = [input_ids(tokenizer, text, add_special_tokens=True) for text in prompts_text]
            residuals = capture_position_residuals(model, encode_batch(ids, device), pos, layers=list(range(model.n_layers)))
            for i, (t, c) in enumerate(chunk):
                for L, mat in residuals.items():
                    states_by_layer.setdefault(L, {}).setdefault(c, {})[t] = np.asarray(mat[i], dtype=np.float32)
        l_star, corrs = layer_localization(states_by_layer, c_c)
        capture_layer = l_star
        layer_sweep = {
            "rule": "argmax_L |pearson(<s(c,P15)-s(c,N15), d_L>, C_c)| over 16-company subsample; ties shallower",
            "companies": companies,
            "layer_star": l_star,
            "corrs": corrs,
        }
        del states_by_layer

    # ---- main capture pass ----
    rows = [(c["ticker"], cond) for c in selection["capture"] for cond in PHASE2_CONDITIONS]
    states: dict[tuple[str, str], np.ndarray] = {}
    for chunk, prompts_text, pos in _chunked(rows, prompts_by_key, positions_by_key):
        ids = [input_ids(tokenizer, text, add_special_tokens=True) for text in prompts_text]
        residuals = capture_position_residuals(model, encode_batch(ids, device), pos, layers=[capture_layer])
        for (t, cond), mat_row in zip(chunk, residuals[capture_layer]):
            states[(t, cond)] = np.asarray(mat_row, dtype=np.float32)
    missing = set(rows) - set(states)
    if missing:
        raise ValueError(f"missing capture states: {sorted(missing)[:5]}")

    capture = selection["capture"]
    p15 = np.stack([states[(c["ticker"], "P15")] for c in capture])
    n15 = np.stack([states[(c["ticker"], "N15")] for c in capture])
    zero = np.stack([states[(c["ticker"], "zero")] for c in capture])
    d_stance = derive_stance_axis(p15, n15)

    diffs = p15 - n15
    r_diff = diffs @ d_stance
    r2 = float(np.sum(r_diff**2) / np.sum(diffs**2)) if float(np.sum(diffs**2)) > 0 else 0.0
    if not np.isfinite(r2):
        raise ValueError("non-finite stance R2")

    stacked = np.vstack([zero, n15, p15]).astype(np.float64)
    centered = stacked - stacked.mean(axis=0)
    if PCA_TOP_K < min(centered.shape):
        _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
        explained = (s_vals**2 / np.sum(s_vals**2))[:PCA_TOP_K]
        top = vt[:PCA_TOP_K].T
        loadings = centered @ top
    else:
        explained = []
        loadings = np.zeros((len(stacked), PCA_TOP_K))
    del stacked, centered

    records: list[dict[str, Any]] = []
    for i, c in enumerate(capture):
        for j, cond in enumerate(PHASE2_CONDITIONS):
            state = states[(c["ticker"], cond)]
            idx = j * len(capture) + i
            records.append(
                {
                    "ticker": c["ticker"],
                    "condition": cond,
                    "v": _finite_float(CONDITION_V[cond]),
                    "group": c["group"],
                    "sector": c["sector"],
                    "split": "discovery",
                    "r_stance": _finite_float(float(state @ d_stance)),
                    "norm_state": _finite_float(float(np.linalg.norm(state))),
                    "top5_loadings": [_finite_float(v) for v in loadings[idx]],
                }
            )
    write_jsonl(output / "records.jsonl", records, overwrite=True)

    axis = {
        "d_stance": [_finite_float(v) for v in d_stance],
        "per_company_polarity_diff": {
            c["ticker"]: {"norm2": _finite_float(float(np.sum(diffs[i] ** 2))), "r_diff": _finite_float(float(r_diff[i]))}
            for i, c in enumerate(capture)
        },
        "r2_stance": _finite_float(r2),
        "pca_explained_top5": [_finite_float(v) for v in explained],
    }
    write_json(output / "axis.json", axis, overwrite=True)
    if layer_sweep is not None:
        write_json(output / "layer_sweep.json", layer_sweep, overwrite=True)

    # ---- determinism: 20 re-run records (seeded stratified 7/7/6) ----
    rng = np.random.default_rng(20260916)
    selected_pairs: list[tuple[str, str]] = []
    for cond, n in DETERMINISM_STRATUM.items():
        pool = [c["ticker"] for c in capture]
        rng.shuffle(pool)
        selected_pairs.extend((t, cond) for t in pool[:n])
    assert len(selected_pairs) == DETERMINISM_COUNT
    rerun_states: dict[tuple[str, str], np.ndarray] = {}
    for chunk, prompts_text, pos in _chunked(selected_pairs, prompts_by_key, positions_by_key):
        ids = [input_ids(tokenizer, text, add_special_tokens=True) for text in prompts_text]
        residuals = capture_position_residuals(model, encode_batch(ids, device), pos, layers=[capture_layer])
        for (t, cond), mat_row in zip(chunk, residuals[capture_layer]):
            rerun_states[(t, cond)] = np.asarray(mat_row, dtype=np.float32)
    max_delta = 0.0
    mismatches = 0
    for t, cond in selected_pairs:
        delta = abs(float(rerun_states[(t, cond)] @ d_stance) - float(states[(t, cond)] @ d_stance))
        max_delta = max(max_delta, delta)
        if not np.allclose(rerun_states[(t, cond)], states[(t, cond)], rtol=0, atol=DETERMINISM_TOL):
            mismatches += 1
    metadata = {
        "model": _model_identity(model_path),
        "model_slug": slug,
        "capture_layer": int(capture_layer),
        "capture_layer_basis": "frozen anchor L15" if selection.get("capture_layer") else f"Step A (L*={capture_layer})",
        "capture_position": "instruction-span final token (per-row position from Phase 1 span; verified in bounds at prepare; chat-template suffix may follow)",
        "n_states": len(states),
        "batch_size": BATCH_SIZE,
        "determinism_check": {
            "n_prompts": len(selected_pairs),
            "stratum": dict(DETERMINISM_STRATUM),
            "max_abs_delta_r_stance": _finite_float(max_delta),
            "state_mismatch_count": mismatches,
            "pairs": [f"{t}:{c}" for t, c in sorted(selected_pairs)],
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "phase2",
    }
    write_metadata(output / "metadata.json", metadata, overwrite=True)
    with run.stage("forward") as stage:
        run.manifest.register_artifact(output / "records.jsonl", artifact_type="evidence_insensitivity_phase2_records", stage="forward", record_count=len(records))
        run.manifest.register_artifact(output / "axis.json", artifact_type="evidence_insensitivity_phase2_axis", stage="forward")
        if layer_sweep is not None:
            run.manifest.register_artifact(output / "layer_sweep.json", artifact_type="evidence_insensitivity_phase2_layer_sweep", stage="forward")
        run.manifest.register_artifact(output / "metadata.json", artifact_type="evidence_insensitivity_phase2_metadata", stage="forward")
        stage.count(len(states))
    del states, rerun_states
    return output / "records.jsonl"


def analyze_stage(run: ArtifactRun, *, model_slug: str | None = None) -> Path:
    slug = model_slug or MODEL_SLUG
    selection = json.loads((run.run_directory / "prepare" / "selection.json").read_text(encoding="utf-8"))
    metadata = json.loads((run.run_directory / "forward" / "metadata.json").read_text(encoding="utf-8"))
    axis = json.loads((run.run_directory / "forward" / "axis.json").read_text(encoding="utf-8"))
    records = read_jsonl(run.run_directory / "forward" / "records.jsonl")
    phase1_dir = _phase1_run_directory(run.run_directory, selection["phase1_run_id"])
    phase1 = json.loads((phase1_dir / "analyze" / "summary.json").read_text(encoding="utf-8"))
    c_c = {g["ticker"]: g["contrast_c"] for g in phase1["groups"] if g.get("contrast_c") is not None}

    per_company: dict[str, dict[str, Any]] = {}
    for rec in records:
        entry = per_company.setdefault(
            rec["ticker"],
            {"group": rec["group"], "sector": rec["sector"], "split": rec["split"], "c_c": c_c.get(rec["ticker"]), "loadings": rec["top5_loadings"], "norm_state": rec["norm_state"]},
        )
        entry[f"r_{rec['condition']}"] = rec["r_stance"]
    for entry in per_company.values():
        entry["offset"], entry["gain"], entry["nonlinearity"] = fit_offset_gain(entry["r_zero"], entry["r_N15"], entry["r_P15"])
        entry["r_diff"] = entry["r_P15"] - entry["r_N15"]

    by_group: dict[str, list[str]] = {}
    for t, e in per_company.items():
        by_group.setdefault(e["group"], []).append(t)
    groups_sorted = sorted(by_group)

    det = metadata["determinism_check"]
    gates: dict[str, dict[str, Any]] = {
        "G-2A": {
            "name": "determinism",
            "threshold": f"max_abs_delta_r_stance <= {DETERMINISM_TOL} and state_mismatch_count == 0",
            "value": det["max_abs_delta_r_stance"],
            "state_mismatch_count": det["state_mismatch_count"],
            "pass": det["max_abs_delta_r_stance"] <= DETERMINISM_TOL and det["state_mismatch_count"] == 0,
        },
        "G-2B": {
            "name": "stance axis validity",
            "threshold": f"r2_stance >= {STANCE_R2_MIN}",
            "value": axis["r2_stance"],
            "pass": axis["r2_stance"] >= STANCE_R2_MIN,
        },
        "G-2C": {
            "name": "group power (discovery)",
            "threshold": f"all groups >= {GROUP_POWER_MIN}",
            "value": {g: len(by_group[g]) for g in groups_sorted},
            "pass": all(len(v) >= GROUP_POWER_MIN for v in by_group.values()),
        },
    }
    gates_passed = all(g["pass"] for g in gates.values())

    def contrast(field: str) -> dict[str, Any]:
        out: dict[str, Any] = {"n": {g: len(by_group[g]) for g in groups_sorted}}
        if len(groups_sorted) == 2 and gates["G-2C"]["pass"]:
            a = [per_company[t][field] for t in by_group[groups_sorted[0]]]
            b = [per_company[t][field] for t in by_group[groups_sorted[1]]]
            out["welch"] = _welch(a, b)
            out["group_means"] = {g: _finite_float(np.mean([per_company[t][field] for t in by_group[g]])) for g in groups_sorted}
            per_sector: dict[str, dict[str, Any]] = {}
            for sector in sorted({per_company[t]["sector"] for t in per_company}):
                a_s = [per_company[t][field] for t in by_group[groups_sorted[0]] if per_company[t]["sector"] == sector]
                b_s = [per_company[t][field] for t in by_group[groups_sorted[1]] if per_company[t]["sector"] == sector]
                if len(a_s) >= 3 and len(b_s) >= 3:
                    w = _welch(a_s, b_s)
                    per_sector[sector] = {"n": [len(a_s), len(b_s)], "cohen_d": w["cohen_d"]}
            out["per_sector_cohen_d"] = per_sector
        return out

    cross_counts: dict[str, dict[str, int]] = {g: {"positive": 0, "negative": 0} for g in groups_sorted}
    cross_sectors: dict[str, dict[str, dict[str, int]]] = {g: {"positive": {}, "negative": {}} for g in groups_sorted}
    for t, e in per_company.items():
        direction = "positive" if e["r_P15"] >= 0 else "negative"
        cross_counts[e["group"]][direction] += 1
        cell = cross_sectors[e["group"]][direction]
        cell[e["sector"]] = cell.get(e["sector"], 0) + 1

    load = np.array([per_company[t]["loadings"] for t in sorted(per_company)])
    total_var = float(np.var(load, axis=0).sum())
    explained_saved = axis.get("pca_explained_top5", [])
    group_loadings = {g: [_finite_float(v) for v in np.mean([per_company[t]["loadings"] for t in by_group[g]], axis=0)] for g in groups_sorted}

    with_cc = [t for t in per_company if per_company[t]["c_c"] is not None]
    r_diff_arr = np.array([per_company[t]["r_diff"] for t in with_cc])
    c_c_arr = np.array([per_company[t]["c_c"] for t in with_cc])

    summary = {
        "schema_version": "evidence-insensitivity-phase2-v1",
        "protocol": PROTOCOL,
        "model_slug": slug,
        "run_id": run.run_directory.name,
        "phase1_run_id": selection["phase1_run_id"],
        "capture_layer": metadata["capture_layer"],
        "capture_layer_basis": metadata["capture_layer_basis"],
        "n_companies": len(per_company),
        "gates": gates,
        "gates_passed": gates_passed,
        "stance": {
            "r2": axis["r2_stance"],
            "corr_rdiff_cc": _finite_float(_pearson(r_diff_arr, c_c_arr)) if len(with_cc) >= 3 else 0.0,
            "mean_r_by_condition": {cond: _finite_float(np.mean([per_company[t][f"r_{cond}"] for t in per_company])) for cond in PHASE2_CONDITIONS},
        },
        "contrast": {field: contrast(field) for field in ("offset", "gain", "r_P15", "r_zero")},
        "cross_tab": cross_counts,
        "cross_tab_sectors": cross_sectors,
        "pca": {"variance_share_top5_saved": explained_saved, "loading_space_share": [_finite_float(v) for v in (np.var(load, axis=0) / total_var if total_var > 0 else [0.0] * 5)], "group_mean_loadings": group_loadings},
        "per_company": {
            t: {
                "group": e["group"],
                "sector": e["sector"],
                "split": e["split"],
                "c_c": e["c_c"],
                "r_zero": _finite_float(e["r_zero"]),
                "r_n15": _finite_float(e["r_N15"]),
                "r_p15": _finite_float(e["r_P15"]),
                "offset": _finite_float(e["offset"]),
                "gain": _finite_float(e["gain"]),
                "nonlinearity": _finite_float(e["nonlinearity"]),
            }
            for t, e in sorted(per_company.items())
        },
    }
    if (run.run_directory / "forward" / "layer_sweep.json").exists():
        summary["layer_sweep"] = json.loads((run.run_directory / "forward" / "layer_sweep.json").read_text(encoding="utf-8"))
    output = run.run_directory / "analyze"
    output.mkdir(parents=True, exist_ok=True)
    path = write_json(output / "summary.json", summary, overwrite=True)
    with run.stage("analyze") as stage:
        run.manifest.register_artifact(path, artifact_type="evidence_insensitivity_phase2_summary", stage="analyze", record_count=len(per_company))
        stage.count(len(per_company))
    return path


# ---------------------------------------------------------------------------
# Public run entrypoints (mirror pipeline.py)
# ---------------------------------------------------------------------------


def create_phase2_run(run_id: str, *, artifact_root: str | Path = "artifacts", model_slug: str | None = None) -> ArtifactRun:
    return ArtifactRun.create(model_slug or MODEL_SLUG, DATASET, run_id, artifact_root=artifact_root)


def open_phase2_run(run_id: str, *, artifact_root: str | Path = "artifacts", model_slug: str | None = None) -> ArtifactRun:
    return ArtifactRun.open(Path(artifact_root) / (model_slug or MODEL_SLUG) / DATASET / "runs" / run_id / "manifest.json")


def run_phase2_prepare(
    run_id: str,
    *,
    artifact_root: str | Path = "artifacts",
    model_path: str = MODEL_SLUG,
    tokenizer: Any | None = None,
    model_slug: str | None = None,
    phase1_run_id: str = PHASE1_DEFAULT_RUN,
) -> Path:
    run = create_phase2_run(run_id, artifact_root=artifact_root, model_slug=model_slug)
    try:
        return prepare_stage(run, model_path=model_path, tokenizer=tokenizer, model_slug=model_slug, phase1_run_id=phase1_run_id)
    except BaseException as exc:
        run.fail(exc)
        raise


def run_phase2_forward(
    run_id: str,
    *,
    artifact_root: str | Path = "artifacts",
    model_path: str = MODEL_SLUG,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: Any = "cpu",
    model_slug: str | None = None,
) -> Path:
    run = open_phase2_run(run_id, artifact_root=artifact_root, model_slug=model_slug)
    try:
        return forward_stage(run, model_path=model_path, model=model, tokenizer=tokenizer, device=device, model_slug=model_slug)
    except BaseException as exc:
        run.fail(exc)
        raise


def run_phase2_analyze(
    run_id: str,
    *,
    artifact_root: str | Path = "artifacts",
    model_slug: str | None = None,
) -> Path:
    run = open_phase2_run(run_id, artifact_root=artifact_root, model_slug=model_slug)
    try:
        result = analyze_stage(run, model_slug=model_slug)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return result
    except BaseException as exc:
        run.fail(exc)
        raise
