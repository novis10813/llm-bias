"""E1 V3 fact-level amnesia gate (proposed; see docs/entity-cell-localization/details/proposal-v3.md).

Closes the factual recall loop at the protocol level: for each top-5
localization candidate, measure the collapse of the model's own clean greedy
gold sequences (first 3 tokens) on the fact frames F0/F2/F3 under full
suppression (alpha = -3.0, all_positions), with wrong-entity and matched-random
controls and cross-entity F0 checks that classify a passing neuron as an
entity cell or a shared fact channel.

Gold validity is fail-closed: a frame only counts toward the gate when an
explicit operator decision (fact_gold_verifications.json, written via
``entity-cell verify-fact-gold``) marks its gold as a correct, specific fact.
Unverified or rejected golds are excluded from the gate and recorded.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.continuation_scoring import _extract_logits, score_token_ids
from llm_bias.core.prompt_input.encoding import input_ids

from .analysis import select_wrong_entity_cell
from .mlp_cells import MLPHookSession, OnlineVectorStats, select_matched_random_neuron

FACT_FRAME_SPECS: tuple[tuple[str, str], ...] = (
    ("F0", "The headquarters of {name} is located in"),
    ("F2", "The stock ticker of {name} is"),
    ("F3", "{name} was founded in"),
)
FACT_FRAME_IDS: tuple[str, ...] = tuple(frame for frame, _ in FACT_FRAME_SPECS)
FACT_GOLD_TOKENS = 3
FACT_END_DOSE = -3.0
FACT_EFFECT_THRESHOLD = 0.5
FACT_CONTROL_THRESHOLD = 0.3

V3_CLASSIFICATIONS = (
    "entity_cell",
    "shared_fact_channel",
    "fact_carrier_unrobust",
    "not_eligible",
    "fact_gate_not_applicable",
    "fact_gate_unavailable",
)


def fact_frame_prompt(frame_id: str, name: str) -> str:
    for frame, template in FACT_FRAME_SPECS:
        if frame == frame_id:
            return template.format(name=name)
    raise ValueError(f"unknown fact frame: {frame_id}")


def _cell_key(cell: Mapping[str, Any] | Sequence[int]) -> str:
    if isinstance(cell, Mapping):
        return f"{int(cell['layer'])}:{int(cell['neuron'])}"
    return f"{int(cell[0])}:{int(cell[1])}"


def _next_logit_distribution(model: Any, ids: Sequence[int], device: Any) -> torch.Tensor:
    tensor = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.forward(tensor)
        logits = _extract_logits(output, model, tensor).float()
    return torch.log_softmax(logits[0, -1], dim=-1)


def greedy_gold_token_ids(
    model: Any, tokenizer: Any, device: Any, prompt: str, steps: int = FACT_GOLD_TOKENS
) -> list[int]:
    """Clean greedy continuation: the frozen gold sequence definition."""
    current = list(input_ids(tokenizer, prompt, add_special_tokens=True))
    out: list[int] = []
    for _ in range(steps):
        dist = _next_logit_distribution(model, current, device)
        token = int(dist.argmax())
        out.append(token)
        current.append(token)
    return out


def _gold_joint_logp(
    model: Any, tokenizer: Any, device: Any, prompt: str, gold_ids: Sequence[int]
) -> float:
    return float(score_token_ids(model, tokenizer, prompt, list(gold_ids), device=device).log_probability)


def _suppressed_gold_logp(
    model: Any, tokenizer: Any, device: Any, prompt: str, gold_ids: Sequence[int],
    cell: Sequence[int], dose: float,
) -> float:
    with MLPHookSession(
        model, [int(cell[0])],
        channel_scales={int(cell[0]): {int(cell[1]): float(dose)}},
        scope="all_positions", scope_positions=None,
    ):
        return _gold_joint_logp(model, tokenizer, device, prompt, gold_ids)


def run_fact_amnesia_stage(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    stats: Mapping[int, OnlineVectorStats],
    fact_frames_by_ticker: Mapping[str, Sequence[Mapping[str, Any]]],
    cells_by_ticker: Mapping[str, Sequence[Mapping[str, Any]]],
    tickers: Sequence[str],
) -> list[dict[str, Any]]:
    """Measure all fact-gate quantities for the top-5 candidates of each ticker.

    Emits one compact row per (ticker, candidate, frame, condition) for the
    own frames and per (ticker, candidate, other ticker) for the cross F0
    checks. Golds are clean (no intervention); verifications stay pending
    (gold_verified: null) until the operator records decisions.
    """
    rows: list[dict[str, Any]] = []
    tickers = sorted(str(value) for value in tickers)
    for ticker in tickers:
        frames = {str(row["frame_id"]): row for row in fact_frames_by_ticker.get(ticker, ())}
        if set(frames) != set(FACT_FRAME_IDS):
            raise ValueError(f"{ticker}: fact frames must cover exactly {list(FACT_FRAME_IDS)}")
        candidates = list(cells_by_ticker[ticker])
        if not candidates:
            raise ValueError(f"{ticker}: no localization candidates for the fact stage")

        # Own frames: clean golds are shared across candidates.
        golds: dict[str, dict[str, Any]] = {}
        for frame_id in FACT_FRAME_IDS:
            row = frames[frame_id]
            prompt = str(row["prompt"])
            gold_ids = greedy_gold_token_ids(model, tokenizer, device, prompt)
            gold_text = tokenizer.decode(gold_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            clean_logp = _gold_joint_logp(model, tokenizer, device, prompt, gold_ids)
            golds[frame_id] = {
                "prompt": prompt,
                "prompt_id": str(row.get("prompt_id", "")),
                "gold_token_ids": list(gold_ids),
                "gold_text": gold_text,
                "clean_logp": clean_logp,
            }

        # Cross F0 golds for every other ticker (shared across candidates).
        cross_golds: dict[str, dict[str, Any]] = {}
        for other in tickers:
            if other == ticker:
                continue
            other_frames = {str(row["frame_id"]): row for row in fact_frames_by_ticker.get(other, ())}
            if "F0" not in other_frames:
                raise ValueError(f"{other}: missing F0 fact frame for cross checks")
            row = other_frames["F0"]
            prompt = str(row["prompt"])
            gold_ids = greedy_gold_token_ids(model, tokenizer, device, prompt)
            gold_text = tokenizer.decode(gold_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            clean_logp = _gold_joint_logp(model, tokenizer, device, prompt, gold_ids)
            cross_golds[other] = {
                "prompt": prompt,
                "prompt_id": str(row.get("prompt_id", "")),
                "gold_token_ids": list(gold_ids),
                "gold_text": gold_text,
                "clean_logp": clean_logp,
            }

        for candidate in candidates:
            cell = (int(candidate["layer"]), int(candidate["neuron"]))
            wrong, degenerate = select_wrong_entity_cell(ticker, tickers, cells_by_ticker, candidate)
            random_neuron = select_matched_random_neuron(
                stats, layer=cell[0], target_neuron=cell[1]
            )
            for frame_id in FACT_FRAME_IDS:
                gold = golds[frame_id]
                target_logp = _suppressed_gold_logp(model, tokenizer, device, gold["prompt"], gold["gold_token_ids"], cell, FACT_END_DOSE)
                wrong_logp = None if degenerate else _suppressed_gold_logp(
                    model, tokenizer, device, gold["prompt"], gold["gold_token_ids"],
                    (int(wrong["layer"]), int(wrong["neuron"])), FACT_END_DOSE,
                )
                random_logp = _suppressed_gold_logp(
                    model, tokenizer, device, gold["prompt"], gold["gold_token_ids"],
                    (cell[0], int(random_neuron)), FACT_END_DOSE,
                )
                base = {
                    "schema_version": 1,
                    "artifact_type": "entity_cell_fact_amnesia",
                    "check": "own",
                    "ticker": ticker,
                    "frame_id": frame_id,
                    "prompt_id": gold["prompt_id"],
                    "prompt": gold["prompt"],
                    "candidate": {"layer": cell[0], "neuron": cell[1]},
                    "dose": FACT_END_DOSE,
                    "scope": "all_positions",
                    "gold_token_ids": gold["gold_token_ids"],
                    "gold_text": gold["gold_text"],
                    "gold_verified": None,
                    "wrong_entity_degenerate": bool(degenerate),
                }
                rows.append({**base, "condition": "clean", "gold_joint_logp": gold["clean_logp"], "collapse": None})
                rows.append({**base, "condition": "target", "gold_joint_logp": target_logp, "collapse": target_logp - gold["clean_logp"]})
                rows.append({
                    **base, "condition": "wrong_entity",
                    "wrong_entity_cell": [int(wrong["layer"]), int(wrong["neuron"])],
                    "gold_joint_logp": wrong_logp,
                    "collapse": None if wrong_logp is None else wrong_logp - gold["clean_logp"],
                })
                rows.append({
                    **base, "condition": "matched_random",
                    "matched_random_neuron": int(random_neuron),
                    "gold_joint_logp": random_logp,
                    "collapse": random_logp - gold["clean_logp"],
                })
            for other in sorted(cross_golds):
                gold = cross_golds[other]
                suppressed_logp = _suppressed_gold_logp(model, tokenizer, device, gold["prompt"], gold["gold_token_ids"], cell, FACT_END_DOSE)
                rows.append({
                    "schema_version": 1,
                    "artifact_type": "entity_cell_fact_amnesia",
                    "check": "cross",
                    "ticker": ticker,
                    "other_ticker": other,
                    "frame_id": "F0",
                    "prompt_id": gold["prompt_id"],
                    "prompt": gold["prompt"],
                    "candidate": {"layer": cell[0], "neuron": cell[1]},
                    "dose": FACT_END_DOSE,
                    "scope": "all_positions",
                    "gold_token_ids": gold["gold_token_ids"],
                    "gold_text": gold["gold_text"],
                    "gold_verified": None,
                    "wrong_entity_degenerate": False,
                    "condition": "target",
                    "gold_joint_logp": suppressed_logp,
                    "collapse": suppressed_logp - gold["clean_logp"],
                })
    return rows


def list_gold_decisions(fact_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Unique gold rows needing an operator decision, in deterministic order."""
    seen: dict[str, dict[str, Any]] = {}
    for row in fact_rows:
        if str(row.get("check", "own")) == "cross":
            key = f"{row['other_ticker']}:F0"
        else:
            key = f"{row['ticker']}:{row['frame_id']}"
        if key in seen or row.get("gold_text") in (None, ""):
            continue
        seen[key] = {
            "key": key,
            "ticker": row["other_ticker"] if str(row.get("check", "own")) == "cross" else row["ticker"],
            "frame_id": row["frame_id"],
            "prompt_id": row["prompt_id"],
            "gold_text": row["gold_text"],
        }
    return sorted(seen.values(), key=lambda value: value["key"])


def _frame_outcome(
    frame_id: str,
    own_rows: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    clean = own_rows["clean"]
    target = own_rows["target"]
    wrong = own_rows["wrong_entity"]
    random = own_rows["matched_random"]
    verified = bool(decision is not None and decision.get("verified"))
    collapse = float(target["collapse"])
    degenerate = bool(wrong.get("wrong_entity_degenerate", False))
    wrong_collapse = None if (degenerate or wrong.get("collapse") is None) else float(wrong["collapse"])
    random_collapse = float(random["collapse"])
    specificity_pass = abs(random_collapse) <= FACT_CONTROL_THRESHOLD and (
        wrong_collapse is None or abs(wrong_collapse) <= FACT_CONTROL_THRESHOLD
    )
    effect_pass = verified and collapse <= -FACT_EFFECT_THRESHOLD
    return {
        "gold_text": clean["gold_text"],
        "gold_verified": verified,
        "clean_logp": float(clean["gold_joint_logp"]),
        "target_collapse": collapse,
        "wrong_entity_collapse": wrong_collapse,
        "wrong_entity_degenerate": degenerate,
        "matched_random_collapse": random_collapse,
        "effect_pass": effect_pass,
        "specificity_pass": specificity_pass,
    }


def _candidate_fact_gate(
    ticker: str,
    cell: Mapping[str, Any],
    own_rows: Mapping[str, dict[str, Mapping[str, Any]]],
    cross_rows: Sequence[Mapping[str, Any]],
    verifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    key = _cell_key(cell)
    frames: dict[str, dict[str, Any]] = {}
    for frame_id in FACT_FRAME_IDS:
        decision = verifications.get(f"{ticker}:{frame_id}")
        frames[frame_id] = _frame_outcome(frame_id, own_rows[frame_id], decision)
    passing_frames = sorted(
        frame_id for frame_id, outcome in frames.items() if outcome["effect_pass"] and outcome["specificity_pass"]
    )
    members = sorted({
        str(row["other_ticker"])
        for row in cross_rows
        if _cell_key(row["candidate"]) == key
        and row.get("collapse") is not None
        and float(row["collapse"]) <= -FACT_EFFECT_THRESHOLD
        and bool(verifications.get(f"{row['other_ticker']}:F0", {}).get("verified"))
    })
    return {
        "candidate": {"layer": int(cell["layer"]), "neuron": int(cell["neuron"])},
        "frames": frames,
        "passing_frames": passing_frames,
        "fact_gate_pass": bool(passing_frames),
        "shared": bool(members),
        "shared_member_tickers": members,
    }


def v3_candidate_eligibility(
    *,
    ticker: str,
    candidate_cells: Sequence[Mapping[str, Any]],
    fact_rows: Sequence[Mapping[str, Any]],
    verifications: Mapping[str, Mapping[str, Any]],
    gates_1_3_pass: bool,
) -> dict[str, Any]:
    """Bind the E1 V3 classification for one ticker (fail-closed).

    The ticker's primary state follows its highest-ranked fact-gate-passing
    candidate: shared -> shared_fact_channel; not shared + Gates 1-3 ->
    entity_cell (trusted cell, with its rank); not shared without Gates 1-3
    -> fact_carrier_unrobust. Lower-ranked shared passing candidates are
    reported in channel_cells regardless of the primary state. Classifications
    without any passing candidate: not_eligible (verified golds present) or
    fact_gate_not_applicable (no verified gold); fact_gate_unavailable when no
    fact measurements exist.
    """
    if not fact_rows:
        return {
            "ticker": ticker,
            "classification": "fact_gate_unavailable",
            "eligible": False,
            "trusted_cell": None,
            "channel_cells": None,
            "gates_1_3_pass": bool(gates_1_3_pass),
            "candidates": {},
            "exclusion_reasons": ["fact_measurements_missing"],
        }
    own = [row for row in fact_rows if str(row.get("check", "own")) == "own"]
    cross = [row for row in fact_rows if str(row.get("check", "own")) == "cross"]
    own_by_cell: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for row in own:
        cell_key = _cell_key(row["candidate"])
        frame = str(row["frame_id"])
        condition = str(row["condition"])
        own_by_cell.setdefault(cell_key, {}).setdefault(frame, {})[condition] = row
    per_candidate: dict[str, dict[str, Any]] = {}
    for cell in candidate_cells:
        cell_key = _cell_key(cell)
        if cell_key not in own_by_cell:
            continue
        frames_rows = own_by_cell[cell_key]
        if any(condition not in frames_rows[frame_id] for frame_id in FACT_FRAME_IDS for condition in ("clean", "target", "wrong_entity", "matched_random")):
            raise ValueError(f"{ticker}: incomplete fact measurements for {cell_key}")
        per_candidate[cell_key] = _candidate_fact_gate(ticker, cell, frames_rows, cross, verifications)
    if not per_candidate:
        raise ValueError(f"{ticker}: no fact measurements for any candidate")
    passing = [key for key, entry in per_candidate.items() if entry["fact_gate_pass"]]
    # candidate_cells is in rank order, so `passing` is rank-ordered as well.
    shared_passing = [key for key in passing if per_candidate[key]["shared"]]
    any_verified = any(
        outcome["gold_verified"] for entry in per_candidate.values() for outcome in entry["frames"].values()
    )
    channel_cells = [
        {
            "cell": [int(part) for part in key.split(":")],
            "member_tickers": per_candidate[key]["shared_member_tickers"],
        }
        for key in shared_passing
    ]
    if not passing:
        if not any_verified:
            classification = "fact_gate_not_applicable"
            reasons = ["gold_verification_pending_or_rejected"]
        else:
            classification = "not_eligible"
            reasons = ["fact_gate_no_frame_pass"]
        return {
            "ticker": ticker,
            "classification": classification,
            "eligible": False,
            "trusted_cell": None,
            "trusted_cell_rank": None,
            "channel_cells": channel_cells,
            "gates_1_3_pass": bool(gates_1_3_pass),
            "passing_candidates": passing,
            "candidates": per_candidate,
            "exclusion_reasons": sorted(set(reasons)),
        }
    # The ticker's primary state follows its highest-ranked passing candidate;
    # lower-ranked shared channels are reported in channel_cells.
    primary = passing[0]
    if per_candidate[primary]["shared"]:
        classification = "shared_fact_channel"
        reasons = []
    elif gates_1_3_pass:
        classification = "entity_cell"
        reasons = []
    else:
        classification = "fact_carrier_unrobust"
        reasons = ["gates_1_3_not_passed"]
    return {
        "ticker": ticker,
        "classification": classification,
        "eligible": classification == "entity_cell",
        "trusted_cell": [int(part) for part in primary.split(":")] if classification == "entity_cell" else None,
        "trusted_cell_rank": int(candidate_cells.index(next(c for c in candidate_cells if _cell_key(c) == primary))) + 1 if classification == "entity_cell" else None,
        "channel_cells": channel_cells,
        "gates_1_3_pass": bool(gates_1_3_pass),
        "passing_candidates": passing,
        "candidates": per_candidate,
        "exclusion_reasons": sorted(set(reasons)),
    }


def read_verifications(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    decisions = payload.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("fact gold verifications must contain a decisions mapping")
    for key, decision in decisions.items():
        if not isinstance(decision, Mapping) or not isinstance(decision.get("verified"), bool):
            raise ValueError(f"fact gold verification for {key} must carry a boolean verified flag")
    return dict(decisions)


def write_verifications(path: str | Path, decisions: Mapping[str, Mapping[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "entity_cell_fact_gold_verifications",
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decisions": {key: dict(value) for key, value in sorted(decisions.items())},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


__all__ = [
    "FACT_CONTROL_THRESHOLD", "FACT_EFFECT_THRESHOLD", "FACT_END_DOSE", "FACT_FRAME_IDS",
    "FACT_FRAME_SPECS", "FACT_GOLD_TOKENS", "V3_CLASSIFICATIONS", "fact_frame_prompt",
    "greedy_gold_token_ids", "list_gold_decisions", "read_verifications",
    "run_fact_amnesia_stage", "v3_candidate_eligibility", "write_verifications",
]
