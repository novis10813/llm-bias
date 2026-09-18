"""Evidence-insensitivity Phase 3: upstream causal localization (patching).

Frozen protocol: ``docs/evidence-insensitivity/details/proposal-phase3.md``
(Rev 1.1). Dual-model execution (Qwen3.5-4B + Gemma-4-E2B). Within-company
polarity transfer patching: T1 (P15 state -> N15 run) and T2 (N15 state ->
P15 run) at (layer x span-position) coordinates; two readouts: Stage 1
decision-position margin scan (all coordinates) and Stage 2 greedy
generation at rule-selected active coordinates. Core discriminable question:
is the behavioral group difference state-mediated (pro-buy signal only in
responsive groups' P15 states) or readout-mediated (both groups' states
carry the signal; the insensitive group's readout ignores it).

Raw states are transient (GPU/CPU memory inside the run only); persisted
outputs are compact decisions, margins, flip flags, and tables.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from llm_bias.core.analysis import decision_flip_summary
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import continuation_token_ids, fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.generation import GenerationConfig, generate_tokens
from llm_bias.core.inference.interventions import record_block_states, residual_interventions
from llm_bias.core.model import load_model, load_tokenizer_for_inference
from llm_bias.core.prompt_input.encoding import input_ids, token_span

from .pipeline import DATASET, MODEL_SLUG, _model_identity
from .screening import _generation_target, parse_decision
from .template import DECISION_PREFIX, prompt_char_spans

SEED = 20260916
PROTOCOL = "proposal-phase3 Rev 1.1 frozen"
PHASE1_DEFAULT_RUN = "phase1-gpu-bf16-01"
PHASE2_DEFAULT_RUN = "phase2-gpu-bf16-01"

POSITIONS = ("entity", "evidence", "instruction", "prompt_end")
DIRECTIONS = ("T1", "T2")  # T1: P15 -> N15 ; T2: N15 -> P15
DIRECTION_TARGET = {"T1": "N15", "T2": "P15"}
DIRECTION_SOURCE = {"T1": "P15", "T2": "N15"}

SAMPLE_PER_GROUP = 42
ACTIVE_TOP_K = 3
MAX_NEW_TOKENS = 128
DETERMINISM_MARGIN_N = 10
DETERMINISM_GEN_N = 10
DETERMINISM_MARGIN_TOL = 0.01
PARSE_RATE_MIN = 0.95
FOCUS_CELL_N_MIN = 30

LAYER_GRIDS = {
    # Gemma grid per proposal-phase3 Rev 1.2 (35-layer text tower; 36/41
    # of the original frozen grid do not exist; nearest-valid correction).
    "qwen3.5-4b": (0, 4, 8, 12, 15, 19, 23, 26, 30, 31),
    "gemma4-e2b-it": (0, 5, 10, 15, 18, 23, 28, 32, 33, 34),
}


def _finite(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("non-finite Phase 3 value")
    return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_sample(groups: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Seeded SAMPLE_PER_GROUP companies from each discovery group (Qwen:
    the 42 responsive companies are exactly the full group; Gemma: seeded
    subsamples of both groups). Deterministic under SEED."""
    by_group: dict[str, list[dict[str, str]]] = {}
    for g in groups:
        if g.get("split") == "discovery" and g.get("group"):
            by_group.setdefault(g["group"], []).append(
                {"ticker": g["ticker"], "group": g["group"], "sector": g["gics_sector"]}
            )
    if len(by_group) != 2:
        raise ValueError(f"expected exactly 2 discovery groups, got {sorted(by_group)}")
    sample: list[dict[str, str]] = []
    for name in sorted(by_group):
        members = sorted(by_group[name], key=lambda r: r["ticker"])
        if len(members) < SAMPLE_PER_GROUP:
            raise ValueError(f"group {name} has {len(members)} < {SAMPLE_PER_GROUP}")
        rng = np.random.default_rng(SEED * 1000 + sorted(by_group).index(name))
        idx = rng.permutation(len(members))[:SAMPLE_PER_GROUP]
        sample.extend(members[i] for i in sorted(idx))
    if len(sample) != 2 * SAMPLE_PER_GROUP or len({r["ticker"] for r in sample}) != len(sample):
        raise ValueError(f"sample malformed: {len(sample)} rows")
    return sample


def _prompt_positions(tokenizer: Any, seq_text: str, raw_prompt: str, stored_spans: dict[str, list[int]]) -> tuple[dict[str, int], int]:
    """The four frozen coordinates in the token space of ``seq_text`` (either
    the raw formatted prompt or raw + DECISION_PREFIX).

    All three span positions are re-derived from the frozen template's
    character ranges inside the actual sequence, so the result is correct in
    whatever coordinate space this tokenizer encodes (e.g. with the forced
    BOS that jlens applies at inference). ``stored_spans`` (Phase 1 token
    spans) are cross-checked: they must agree exactly or differ by a uniform
    +1 (BOS offset); anything else fails closed. Returns the position dict
    and the observed stored-span offset (0 or 1).
    """
    ids = input_ids(tokenizer, seq_text, add_special_tokens=True)
    body_start = seq_text.find(raw_prompt)
    if body_start < 0:
        raise ValueError("prompt body not found in sequence")
    char_spans = prompt_char_spans(raw_prompt)
    positions: dict[str, int] = {}
    # entity = end of the `Stock Name: ...` line (protocol §2), i.e. the
    # header minus the trailing `--- Evidence ---` delimiter line
    entity_end_char = raw_prompt.index("--- Evidence ---")
    ent_span = token_span(tokenizer, seq_text, body_start, body_start + entity_end_char, add_special_tokens=True)
    if ent_span is None:
        raise ValueError("entity span unmappable in sequence")
    positions["entity"] = int(ent_span[1]) - 1
    for name in ("evidence", "instruction"):
        chars = (body_start + char_spans[name][0], body_start + char_spans[name][1])
        span = token_span(tokenizer, seq_text, chars[0], chars[1], add_special_tokens=True)
        if span is None:
            raise ValueError(f"{name} span unmappable in sequence")
        positions[name] = int(span[1]) - 1
    positions["prompt_end"] = len(ids) - 1
    offset = -1
    for name in ("evidence", "instruction"):
        stored_end = int(stored_spans[name][1])
        derived_end = positions[name] + 1
        candidate = derived_end - stored_end
        if candidate not in (0, 1):
            raise ValueError(
                f"{name} span cross-check failed: derived end {derived_end}, stored {stored_end} "
                "(expected exact or uniform +1 BOS offset)"
            )
        offset = candidate if offset < 0 else offset
        if offset != candidate:
            raise ValueError(f"inconsistent stored-span offsets (evidence {derived_end - int(stored_spans['evidence'][1])}, {name} {candidate})")
    if offset < 0:
        offset = 0
    if not all(0 <= p < len(ids) for p in positions.values()):
        raise ValueError(f"position out of bounds: {positions} (seq={len(ids)})")
    if not (positions["entity"] < positions["evidence"] < positions["instruction"] <= positions["prompt_end"]):
        raise ValueError(f"positions not ordered: {positions}")
    return positions, offset


def prepare_stage(
    run: ArtifactRun,
    *,
    model_path: str,
    tokenizer: Any | None = None,
    model_slug: str | None = None,
    phase1_run_id: str = PHASE1_DEFAULT_RUN,
    phase2_run_id: str = PHASE2_DEFAULT_RUN,
) -> Path:
    slug = model_slug or MODEL_SLUG
    if slug not in LAYER_GRIDS:
        raise ValueError(f"no frozen layer grid for {slug}")
    # Inference-matched tokenizer: load_model wraps the model with
    # jlens force_bos=True, which prepends BOS for tokenizers with a
    # bos_token_id (Gemma); position tables must live in that coordinate
    # space or every patch coordinate is off by one. No-op for Qwen
    # (bos_token_id=None).
    tokenizer = tokenizer or load_tokenizer_for_inference(model_path)
    run_root = run.run_directory.parent
    phase1 = json.loads((run_root / phase1_run_id / "analyze" / "summary.json").read_text(encoding="utf-8"))
    phase2_meta = json.loads((run_root / phase2_run_id / "forward" / "metadata.json").read_text(encoding="utf-8"))
    l_anchor = int(phase2_meta["capture_layer"])
    prompts = read_jsonl(run_root / phase1_run_id / "prepare" / "prompts.jsonl")
    primary = {
        (p["ticker"], p["condition"]): p
        for p in prompts
        if p["arm"] == "primary" and p["condition"] in ("N15", "P15")
    }
    sample = select_sample(phase1["groups"])
    positions: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    span_offset: int | None = None
    for row in sample:
        t = row["ticker"]
        per_cond: dict[str, dict[str, dict[str, int]]] = {}
        for cond in ("N15", "P15"):
            record = primary.get((t, cond))
            if record is None:
                raise ValueError(f"Phase 1 prompt missing: {t}:{cond}")
            raw_prompt = build_raw_prompt(record)
            stored = {"evidence": record["evidence_span"], "instruction": record["instruction_span"]}
            raw_pos, offset = _prompt_positions(tokenizer, record["prompt_text"], raw_prompt, stored)
            prefix_pos, offset_pre = _prompt_positions(tokenizer, record["prompt_text"] + DECISION_PREFIX, raw_prompt, stored)
            if offset_pre != offset:
                raise ValueError(f"{t}:{cond} inconsistent stored-span offset between raw/prefix sequences")
            if span_offset is not None and span_offset != offset:
                raise ValueError(f"{t}:{cond} stored-span offset {offset} != {span_offset} across companies")
            span_offset = offset
            per_cond[cond] = {"raw": raw_pos, "prefix": prefix_pos}
        positions[t] = per_cond
    selection = {
        "protocol": PROTOCOL,
        "model_slug": slug,
        "phase1_run_id": phase1_run_id,
        "phase2_run_id": phase2_run_id,
        "seed": SEED,
        "sample": sample,
        "layer_grid": list(LAYER_GRIDS[slug]),
        "l_anchor": l_anchor,
        "anchor_combo": [l_anchor, "instruction"],
        "positions": positions,
        # 0 = Phase 1 stored spans in the same coordinate space as this
        # tokenizer; 1 = stored spans predate a forced BOS (Gemma).
        "stored_span_offset": span_offset if span_offset is not None else 0,
    }
    output = run.run_directory / "prepare"
    output.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        sel_path = write_json(output / "selection.json", selection, overwrite=True)
        run.manifest.register_artifact(sel_path, artifact_type="evidence_insensitivity_phase3_selection", stage="prepare", record_count=len(sample))
        stage.count(len(sample) * 2 * len(POSITIONS))
    return sel_path


def build_raw_prompt(record: dict[str, Any]) -> str:
    """Recover the raw (unformatted) prompt body from a Phase 1 record."""
    from .template import build_prompt

    return build_prompt(record["ticker"], record["company_name"], record["condition"])


# ---------------------------------------------------------------------------
# Patched scoring primitives
# ---------------------------------------------------------------------------


def _swap_transform(position: int, source_state: torch.Tensor, seq_len: int):
    """Post-block swap of one position, guarded to the prefill forward only.

    During ``model.generate`` the same hook fires on every decode step,
    whose hidden state has shape ``[1, 1, d]``; the guard passes those
    through untouched so only the prefill forward of length ``seq_len``
    is patched (the decode steps then ride on the patched KV cache).
    """

    def transform(output: torch.Tensor) -> torch.Tensor:
        if output.shape[1] != seq_len:
            return output
        out = output.clone()
        out[:, position, :] = source_state.to(out.device, out.dtype)
        return out

    return transform


def _margin_at_end(model: Any, input_tensor: torch.Tensor, buy_id: int, sell_id: int, transforms: dict[int, Any] | None) -> float:
    """Decision-position buy/sell margin (FP32 tail) at the sequence's final
    position, optionally under post-block residual swaps."""
    final_layer = int(model.n_layers) - 1
    with torch.no_grad():
        if transforms:
            with residual_interventions(model, transforms):
                residual = record_residuals(model, input_tensor, [final_layer])[final_layer]
        else:
            residual = record_residuals(model, input_tensor, [final_layer])[final_layer]
    log_probs = fp32_next_token_log_probs(model, residual[:, -1, :])[0]
    return _finite(log_probs[buy_id].item() - log_probs[sell_id].item())


def active_combos(scan_rows: list[dict[str, Any]], anchor_combo: tuple[int, str]) -> list[tuple[int, str]]:
    """Pre-registered Stage 2 selection: top-3 (layer, position) by median
    |delta_m| (both directions, pooled), union the anchor coordinate,
    deduplicated (<= 4)."""
    by_combo: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in scan_rows:
        by_combo[(row["layer"], row["position"])].append(abs(row["delta_m"]))
    ranked = sorted(by_combo, key=lambda combo: (-float(np.median(by_combo[combo])), combo[0], combo[1]))
    combos = list(ranked[:ACTIVE_TOP_K])
    if anchor_combo not in combos:
        combos.append(anchor_combo)
    return combos[: 2 * ACTIVE_TOP_K]


def _ids(tokenizer: Any, text: str, device: Any) -> torch.Tensor:
    return torch.tensor([input_ids(tokenizer, text, add_special_tokens=True)], dtype=torch.long, device=device)


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
    grid = list(selection["layer_grid"])
    l_anchor = int(selection["l_anchor"])
    run_root = run.run_directory.parent
    prompts = read_jsonl(run_root / selection["phase1_run_id"] / "prepare" / "prompts.jsonl")
    prompts_by_key = {
        (p["ticker"], p["condition"]): p
        for p in prompts
        if p["arm"] == "primary" and p["condition"] in ("N15", "P15")
    }

    # buy/sell single-token ids at the decision position (Phase 1 contract)
    first_t = selection["sample"][0]["ticker"]
    probe_text = prompts_by_key[(first_t, "N15")]["prompt_text"]
    _, buy_ids = continuation_token_ids(tokenizer, probe_text + DECISION_PREFIX, "buy")
    _, sell_ids = continuation_token_ids(tokenizer, probe_text + DECISION_PREFIX, "sell")
    if len(buy_ids) != 1 or len(sell_ids) != 1:
        raise ValueError("buy/sell must be single tokens at the decision position")
    buy_id, sell_id = buy_ids[0], sell_ids[0]

    pos = selection["positions"]
    # Coordinate-space guard: the stored position table must index the
    # forward-time encodings (prepare and forward may load the tokenizer
    # through different paths; e.g. jlens force_bos prepends BOS for
    # tokenizers with a bos_token_id). Fail closed on any drift.
    for row in selection["sample"]:
        t = row["ticker"]
        for cond in ("N15", "P15"):
            raw_len = len(input_ids(tokenizer, prompts_by_key[(t, cond)]["prompt_text"], add_special_tokens=True))
            prefix_len = len(input_ids(tokenizer, prompts_by_key[(t, cond)]["prompt_text"] + DECISION_PREFIX, add_special_tokens=True))
            if pos[t][cond]["raw"]["prompt_end"] != raw_len - 1 or pos[t][cond]["prefix"]["prompt_end"] != prefix_len - 1:
                raise ValueError(
                    f"{t}:{cond} position table does not match forward-time encoding "
                    f"(raw {raw_len} vs stored prompt_end {pos[t][cond]['raw']['prompt_end']}, "
                    f"prefix {prefix_len} vs {pos[t][cond]['prefix']['prompt_end']}); "
                    "tokenizer coordinate drift (e.g. forced BOS) - re-run prepare "
                    "with the inference-matched tokenizer"
                )
    scan_rows: list[dict[str, Any]] = []
    baseline_m: dict[tuple[str, str], float] = {}
    gen_rows: list[dict[str, Any]] = []
    counts = {"source": 0, "scan": 0, "gen": 0, "baseline_gen": 0}

    # ================= Stage 1: margin scan =================
    for row in selection["sample"]:
        t = row["ticker"]
        # source states (prefix sequences, all grid layers, four coordinates)
        src: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        for cond in ("N15", "P15"):
            text = prompts_by_key[(t, cond)]["prompt_text"] + DECISION_PREFIX
            ids = _ids(tokenizer, text, device)
            with torch.no_grad():
                states = record_block_states(model, ids, layers=grid)
            src[cond] = {
                L: {p_name: states[L]["post"][0, pos[t][cond]["prefix"][p_name], :].detach().clone() for p_name in POSITIONS}
                for L in grid
            }
            del states
            counts["source"] += 1
        # baselines + patches
        for direction in DIRECTIONS:
            target_cond, source_cond = DIRECTION_TARGET[direction], DIRECTION_SOURCE[direction]
            target_text = prompts_by_key[(t, target_cond)]["prompt_text"] + DECISION_PREFIX
            target_ids = _ids(tokenizer, target_text, device)
            with torch.no_grad():
                base_m = _margin_at_end(model, target_ids, buy_id, sell_id, None)
            baseline_m[(t, direction)] = base_m
            for L in grid:
                for p_name in POSITIONS:
                    patched = _margin_at_end(
                        model, target_ids, buy_id, sell_id,
                        {L: _swap_transform(pos[t][target_cond]["prefix"][p_name], src[source_cond][L][p_name], target_ids.shape[1])},
                    )
                    scan_rows.append(
                        {
                            "ticker": t,
                            "group": row["group"],
                            "sector": row["sector"],
                            "direction": direction,
                            "layer": L,
                            "position": p_name,
                            "baseline_m": base_m,
                            "patched_m": patched,
                            "delta_m": _finite(patched - base_m),
                        }
                    )
                    counts["scan"] += 1
        del src

    # ---- G-3B efficacy probe (fail-fast before the focus pass) ----
    l_final = grid[-1]
    probe = [r["delta_m"] for r in scan_rows if r["layer"] == l_final and r["position"] == "prompt_end" and r["direction"] == "T1"]
    probe_median = _finite(float(np.median(probe))) if probe else 0.0
    if probe_median <= 0.0:
        raise ValueError(f"G-3B intervention efficacy failed: probe median dM = {probe_median} (must be > 0)")

    # ================= Stage 2: generation at active coordinates =================
    combos = active_combos(scan_rows, tuple(selection["anchor_combo"]))
    active_layers = sorted({L for L, _ in combos})
    baseline_text: dict[tuple[str, str], str] = {}

    def _baseline_generated_text(t: str, cond: str) -> str:
        key = (t, cond)
        if key not in baseline_text:
            text = prompts_by_key[(t, cond)]["prompt_text"]
            ids = _ids(tokenizer, text, device)
            gen = generate_tokens(_generation_target(model), ids, GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, pad_token_id=getattr(tokenizer, "eos_token_id", None)))
            baseline_text[key] = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=False)
            counts["baseline_gen"] += 1
        return baseline_text[key]

    for row in selection["sample"]:
        t = row["ticker"]
        # source states at active layers for raw and prefix sequences
        src_raw: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        src_prefix: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        for cond in ("N15", "P15"):
            raw_text = prompts_by_key[(t, cond)]["prompt_text"]
            ids_raw = _ids(tokenizer, raw_text, device)
            with torch.no_grad():
                states = record_block_states(model, ids_raw, layers=active_layers)
            src_raw[cond] = {L: {p: states[L]["post"][0, pos[t][cond]["raw"][p], :].detach().clone() for p in POSITIONS} for L in active_layers}
            del states
            ids_pre = _ids(tokenizer, raw_text + DECISION_PREFIX, device)
            with torch.no_grad():
                states = record_block_states(model, ids_pre, layers=active_layers)
            src_prefix[cond] = {L: {p: states[L]["post"][0, pos[t][cond]["prefix"][p], :].detach().clone() for p in POSITIONS} for L in active_layers}
            del states
            counts["source"] += 2
        for L, p_name in combos:
            for direction in DIRECTIONS:
                target_cond, source_cond = DIRECTION_TARGET[direction], DIRECTION_SOURCE[direction]
                target_text = prompts_by_key[(t, target_cond)]["prompt_text"]
                prompt_ids = _ids(tokenizer, target_text, device)
                transform = _swap_transform(pos[t][target_cond]["raw"][p_name], src_raw[source_cond][L][p_name], prompt_ids.shape[1])
                with residual_interventions(model, {L: transform}):
                    gen = generate_tokens(_generation_target(model), prompt_ids, GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, pad_token_id=getattr(tokenizer, "eos_token_id", None)))
                patched_decision = parse_decision(tokenizer.decode(gen[0][prompt_ids.shape[1]:], skip_special_tokens=False))
                base_decision = parse_decision(_baseline_generated_text(t, target_cond))
                prefix_ids = _ids(tokenizer, target_text + DECISION_PREFIX, device)
                patched_m = _margin_at_end(
                    model, prefix_ids, buy_id, sell_id,
                    {L: _swap_transform(pos[t][target_cond]["prefix"][p_name], src_prefix[source_cond][L][p_name], prefix_ids.shape[1])},
                )
                gen_rows.append(
                    {
                        "ticker": t,
                        "group": row["group"],
                        "sector": row["sector"],
                        "direction": direction,
                        "layer": L,
                        "position": p_name,
                        "baseline_decision": base_decision,
                        "patched_decision": patched_decision,
                        "flip": bool(patched_decision is not None and base_decision is not None and patched_decision != base_decision),
                        "baseline_m": baseline_m[(t, direction)],
                        "patched_m": patched_m,
                    }
                )
                counts["gen"] += 1
        del src_raw, src_prefix

    # ---- determinism: 10 margin + 10 generation re-runs ----
    rng = np.random.default_rng(SEED)
    tickers = [r["ticker"] for r in selection["sample"]]
    max_delta_m = 0.0
    margin_mismatches = 0
    for i in range(DETERMINISM_MARGIN_N):
        t = tickers[int(rng.integers(len(tickers)))]
        direction = DIRECTIONS[i % 2]
        target_text = prompts_by_key[(t, DIRECTION_TARGET[direction])]["prompt_text"] + DECISION_PREFIX
        with torch.no_grad():
            rerun_m = _margin_at_end(model, _ids(tokenizer, target_text, device), buy_id, sell_id, None)
        max_delta_m = max(max_delta_m, abs(rerun_m - baseline_m[(t, direction)]))
    L0, p0 = combos[0]
    gen_mismatches = 0
    for i in range(DETERMINISM_GEN_N):
        t = tickers[int(rng.integers(len(tickers)))]
        direction = DIRECTIONS[i % 2]
        target_cond, source_cond = DIRECTION_TARGET[direction], DIRECTION_SOURCE[direction]
        target_text = prompts_by_key[(t, target_cond)]["prompt_text"]
        source_text = prompts_by_key[(t, source_cond)]["prompt_text"]
        with torch.no_grad():
            source_post = record_block_states(model, _ids(tokenizer, source_text, device), layers=[L0])[L0]["post"]
        src_state = source_post[0, pos[t][source_cond]["raw"][p0], :].detach().clone()
        rerun_ids = _ids(tokenizer, target_text, device)
        transform = _swap_transform(pos[t][target_cond]["raw"][p0], src_state, rerun_ids.shape[1])
        with residual_interventions(model, {L0: transform}):
            gen = generate_tokens(_generation_target(model), rerun_ids, GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, pad_token_id=getattr(tokenizer, "eos_token_id", None)))
        rerun_decision = parse_decision(tokenizer.decode(gen[0][rerun_ids.shape[1]:], skip_special_tokens=False))
        original = next((g for g in gen_rows if g["ticker"] == t and g["direction"] == direction and g["layer"] == L0 and g["position"] == p0), None)
        if original is not None and rerun_decision != original["patched_decision"]:
            gen_mismatches += 1

    # ---- outputs ----
    output = run.run_directory / "forward"
    output.mkdir(parents=True, exist_ok=True)
    by_combo: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in scan_rows:
        by_combo[(r["layer"], r["position"])].append(abs(r["delta_m"]))
    ranking = [
        {"combo": [L, p], "median_abs_delta_m": _finite(float(np.median(vals)))}
        for (L, p), vals in sorted(by_combo.items(), key=lambda kv: (-float(np.median(kv[1])), kv[0][0], kv[0][1]))
    ]
    focus_data = {
        "rule": f"top-{ACTIVE_TOP_K} (layer, position) by median |delta_m| (both directions, pooled) + anchor {[l_anchor, 'instruction']}, dedup",
        "ranking": ranking[: 2 * ACTIVE_TOP_K],
        "selected": [[L, p] for L, p in combos],
    }
    write_jsonl(output / "scan_records.jsonl", scan_rows, overwrite=True)
    write_json(output / "focus_combos.json", focus_data, overwrite=True)
    write_jsonl(output / "generation_records.jsonl", gen_rows, overwrite=True)
    metadata = {
        "model": _model_identity(model_path),
        "model_slug": slug,
        "layer_grid": grid,
        "l_anchor": l_anchor,
        "n_source_forwards": counts["source"],
        "n_scan_forwards": counts["scan"],
        "n_generation_runs": counts["gen"],
        "n_baseline_generation_runs": counts["baseline_gen"],
        "buy_token_id": buy_id,
        "sell_token_id": sell_id,
        "g3b_probe": {"layer": l_final, "position": "prompt_end", "direction": "T1", "median_delta_m": probe_median},
        "determinism_check": {
            "n_margin_reruns": DETERMINISM_MARGIN_N,
            "n_generation_reruns": DETERMINISM_GEN_N,
            "max_abs_delta_m": _finite(max_delta_m),
            "margin_mismatch_count": margin_mismatches,
            "generation_mismatch_count": gen_mismatches,
            "tolerance": DETERMINISM_MARGIN_TOL,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "phase3",
    }
    write_metadata(output / "metadata.json", metadata, overwrite=True)
    with run.stage("forward") as stage:
        run.manifest.register_artifact(output / "scan_records.jsonl", artifact_type="evidence_insensitivity_phase3_scan", stage="forward", record_count=len(scan_rows))
        run.manifest.register_artifact(output / "focus_combos.json", artifact_type="evidence_insensitivity_phase3_focus", stage="forward")
        run.manifest.register_artifact(output / "generation_records.jsonl", artifact_type="evidence_insensitivity_phase3_generation", stage="forward", record_count=len(gen_rows))
        run.manifest.register_artifact(output / "metadata.json", artifact_type="evidence_insensitivity_phase3_metadata", stage="forward")
        stage.count(len(scan_rows) + len(gen_rows))
    return output / "scan_records.jsonl"


def analyze_stage(run: ArtifactRun, *, model_slug: str | None = None) -> Path:
    slug = model_slug or MODEL_SLUG
    selection = json.loads((run.run_directory / "prepare" / "selection.json").read_text(encoding="utf-8"))
    metadata = json.loads((run.run_directory / "forward" / "metadata.json").read_text(encoding="utf-8"))
    scan_rows = read_jsonl(run.run_directory / "forward" / "scan_records.jsonl")
    gen_rows = read_jsonl(run.run_directory / "forward" / "generation_records.jsonl")
    focus = json.loads((run.run_directory / "forward" / "focus_combos.json").read_text(encoding="utf-8"))

    det = metadata["determinism_check"]
    g3a = det["max_abs_delta_m"] <= DETERMINISM_MARGIN_TOL and det["margin_mismatch_count"] == 0 and det["generation_mismatch_count"] == 0
    g3b = metadata["g3b_probe"]["median_delta_m"] > 0.0
    parsed = sum(1 for g in gen_rows if g["patched_decision"] is not None and g["baseline_decision"] is not None)
    cell_ns: dict[str, int] = {}
    for combo in focus["selected"]:
        L, p = combo
        for group in sorted({g["group"] for g in gen_rows}):
            cell_ns[f"{L}:{p}:{group}"] = sum(1 for g in gen_rows if g["layer"] == L and g["position"] == p and g["group"] == group)
    g3c = bool(gen_rows) and (parsed / len(gen_rows) >= PARSE_RATE_MIN) and all(n >= FOCUS_CELL_N_MIN for n in cell_ns.values())
    gates = {
        "G-3A": {"name": "baseline integrity", "threshold": f"max_delta_m <= {DETERMINISM_MARGIN_TOL} and 0 mismatches", "value": det["max_abs_delta_m"], "mismatches": det["margin_mismatch_count"] + det["generation_mismatch_count"], "pass": bool(g3a)},
        "G-3B": {"name": "intervention efficacy", "threshold": "probe median delta_m > 0", "value": metadata["g3b_probe"]["median_delta_m"], "pass": bool(g3b)},
        "G-3C": {"name": "stage-2 parse power", "threshold": f"parse >= {PARSE_RATE_MIN} and cell n >= {FOCUS_CELL_N_MIN}", "value": {"parse_rate": _finite(parsed / len(gen_rows)) if gen_rows else 0.0, "min_cell_n": min(cell_ns.values()) if cell_ns else 0}, "pass": g3c},
    }

    groups = sorted({r["group"] for r in scan_rows})

    def median_map(direction: str | None, group: str | None, position: str | None = None) -> dict[str, float]:
        by_key: dict[str, list[float]] = defaultdict(list)
        for r in scan_rows:
            if (direction is None or r["direction"] == direction) and (group is None or r["group"] == group) and (position is None or r["position"] == position):
                by_key[f"{r['layer']}:{r['position']}"].append(r["delta_m"])
        return {k: _finite(float(np.median(v))) for k, v in sorted(by_key.items(), key=lambda kv: (int(kv[0].split(":")[0]), kv[0]))}

    s_vs_r: dict[str, Any] = {}
    for combo in focus["selected"]:
        L, p = combo
        entry: dict[str, Any] = {}
        if len(groups) == 2:
            a = [r["delta_m"] for r in scan_rows if r["layer"] == L and r["position"] == p and r["direction"] == "T1" and r["group"] == groups[0]]
            b = [r["delta_m"] for r in scan_rows if r["layer"] == L and r["position"] == p and r["direction"] == "T1" and r["group"] == groups[1]]
            if len(a) >= 2 and len(b) >= 2:
                va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
                se = math.sqrt(va / len(a) + vb / len(b))
                if se > 0:
                    pooled = math.sqrt((va + vb) / 2)
                    entry["t1_margin_welch"] = {
                        "t": _finite((float(np.mean(a)) - float(np.mean(b))) / se),
                        "mean": {groups[0]: _finite(float(np.mean(a))), groups[1]: _finite(float(np.mean(b)))},
                        "cohen_d": _finite((float(np.mean(a)) - float(np.mean(b))) / pooled) if pooled > 0 else 0.0,
                    }
            flip_summaries = {}
            for group in groups:
                rows = [
                    g for g in gen_rows
                    if g["layer"] == L and g["position"] == p
                    and g["direction"] == "T1" and g["group"] == group
                ]
                flip_summaries[group] = decision_flip_summary(
                    {str(g["ticker"]): g["baseline_decision"] for g in rows},
                    {str(g["ticker"]): g["patched_decision"] for g in rows},
                )
            entry["t1_flip_rate"] = flip_summaries
            flip_a = flip_summaries[groups[0]]["flip_count"]
            flip_b = flip_summaries[groups[1]]["flip_count"]
            n_a = flip_summaries[groups[0]]["valid_pair_count"]
            n_b = flip_summaries[groups[1]]["valid_pair_count"]
            if n_a and n_b and flip_a + flip_b > 0:
                _, pval = stats.fisher_exact([[flip_a, n_a - flip_a], [flip_b, n_b - flip_b]])
                entry["t1_flip_fisher"] = {"flip": {groups[0]: flip_a, groups[1]: flip_b}, "n": {groups[0]: n_a, groups[1]: n_b}, "p": _finite(float(pval))}
            else:
                entry["t1_flip_fisher"] = {"flip": {groups[0]: flip_a, groups[1]: flip_b}, "n": {groups[0]: n_a, groups[1]: n_b}, "p": None, "note": "no flips; odds ratio undefined"}
        s_vs_r[f"T1@{L}:{p}"] = entry

    t1_t2_profile = {
        direction: {
            str(L): _finite(float(np.median([r["delta_m"] for r in scan_rows if r["layer"] == L and r["direction"] == direction])))
            for L in selection["layer_grid"]
        }
        for direction in DIRECTIONS
    }
    entity_control = {
        direction: _finite(float(np.median([r["delta_m"] for r in scan_rows if r["position"] == "entity" and r["direction"] == direction])))
        for direction in DIRECTIONS
    }
    sector_table: dict[str, Any] = {}
    for combo in focus["selected"]:
        L, p = combo
        combo_rows = [r for r in scan_rows if r["layer"] == L and r["position"] == p and r["direction"] == "T1"]
        sector_table[f"{L}:{p}"] = {
            sector: {
                group: _finite(float(np.median([r["delta_m"] for r in combo_rows if r["group"] == group and r["sector"] == sector])))
                for group in groups
                if any(r["group"] == group and r["sector"] == sector for r in combo_rows)
            }
            for sector in sorted({r["sector"] for r in combo_rows})
        }

    summary = {
        "schema_version": "evidence-insensitivity-phase3-v1",
        "protocol": PROTOCOL,
        "model_slug": slug,
        "run_id": run.run_directory.name,
        "n_companies": len(selection["sample"]),
        "group_sizes": {g: sum(1 for r in selection["sample"] if r["group"] == g) for g in groups},
        "active_combos": focus["selected"],
        "gates": gates,
        "gates_passed": all(g["pass"] for g in gates.values()),
        "effect_map_median_delta_m": {
            direction: {group: median_map(direction, group) for group in groups}
            for direction in DIRECTIONS
        },
        "s_vs_r": s_vs_r,
        "t1_vs_t2_layer_profile": t1_t2_profile,
        "entity_control_median_delta_m": entity_control,
        "sector_table_t1": sector_table,
    }
    output = run.run_directory / "analyze"
    output.mkdir(parents=True, exist_ok=True)
    path = write_json(output / "summary.json", summary, overwrite=True)
    with run.stage("analyze") as stage:
        run.manifest.register_artifact(path, artifact_type="evidence_insensitivity_phase3_summary", stage="analyze", record_count=len(selection["sample"]))
        stage.count(len(selection["sample"]))
    return path


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def run_phase3_prepare(
    run_id: str,
    *,
    artifact_root: str | Path = "artifacts",
    model_path: str = MODEL_SLUG,
    tokenizer: Any | None = None,
    model_slug: str | None = None,
    phase1_run_id: str = PHASE1_DEFAULT_RUN,
    phase2_run_id: str = PHASE2_DEFAULT_RUN,
) -> Path:
    run = ArtifactRun.create(model_slug or MODEL_SLUG, DATASET, run_id, artifact_root=artifact_root)
    try:
        return prepare_stage(run, model_path=model_path, tokenizer=tokenizer, model_slug=model_slug, phase1_run_id=phase1_run_id, phase2_run_id=phase2_run_id)
    except BaseException as exc:
        run.fail(exc)
        raise


def run_phase3_forward(
    run_id: str,
    *,
    artifact_root: str | Path = "artifacts",
    model_path: str = MODEL_SLUG,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: Any = "cpu",
    model_slug: str | None = None,
) -> Path:
    run = ArtifactRun.open(Path(artifact_root) / (model_slug or MODEL_SLUG) / DATASET / "runs" / run_id / "manifest.json")
    try:
        return forward_stage(run, model_path=model_path, model=model, tokenizer=tokenizer, device=device, model_slug=model_slug)
    except BaseException as exc:
        run.fail(exc)
        raise


def run_phase3_analyze(
    run_id: str,
    *,
    artifact_root: str | Path = "artifacts",
    model_slug: str | None = None,
) -> Path:
    run = ArtifactRun.open(Path(artifact_root) / (model_slug or MODEL_SLUG) / DATASET / "runs" / run_id / "manifest.json")
    try:
        result = analyze_stage(run, model_slug=model_slug)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return result
    except BaseException as exc:
        run.fail(exc)
        raise
