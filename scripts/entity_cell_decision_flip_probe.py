"""Entity-cell suppression and decision-flip probe (proposed; not a frozen protocol).

Generic extension of the AMZN factual amnesia / decision flip probes. Tests
whether a localized entity cell, when suppressed:
  1. collapses the entity's factual recall (optional fact frame; gold = clean
     greedy first-3-token continuation, human-verified), and
  2. moves / flips the buy-sell decision on prepared financial prompts,
     including prompts where the clean entity decision is OPPOSITE to the
     anonymous-header baseline.

Decision readout = the frozen amnesia-endpoint margin: logP(buy) - logP(sell)
after DECISION_PREFIX, computed with the v2 FP32 final-norm tail instrument,
plus the greedy next token as the observed output. Greedy continuations are
display-only diagnostics. Margin movement is reported both as nats and as the
frozen anonymous_progress toward the anonymous margin.

Conditions: frozen ALPHA_GRID dose sweep for the target cell (all_positions
and header_only), wrong-cell and deterministic matched-random controls at the
end dose, and a cross-ticker control (same target cell on other tickers'
prompts at the end dose).

Usage (from the repo root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/entity_cell_decision_flip_probe.py \
        --output /path/to/probe.json \
        [--flip-ticker NVDA] [--flip-cell 2,1786] [--wrong-cell 0,1476] \
        [--financial-prompts /path/to/financial_prompts.jsonl] \
        [--reuse-scan /path/to/previous_probe.json] \
        [--fact-frame 'The headquarters of {name} is located in'] \
        [--fact-name 'Nvidia Corp.'] \
        [--model .cache/models/qwen3.5-4b]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.continuation_scoring import (
    fp32_next_token_log_probs,
    score_single_token_margin_fp32,
    score_token_ids,
)
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import decode_token, format_prompt, input_ids
from llm_bias.entity_cell.mlp_cells import (
    ALPHA_GRID,
    DECISION_PREFIX,
    MLPHookSession,
    OnlineVectorStats,
    anonymous_progress,
    select_matched_random_neuron,
)

DISCOVERY_RUN = "artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e1-hfm-discovery-v1"
DEFAULT_PROMPTS = (
    "artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-prepare-hfm-v3"
    "/prepare/financial_prompts.jsonl"
)
GREEDY_STEPS = 8
GOLD_TOKENS = 3


def _fp32_dist(model: Any, ids: list[int], device: Any) -> torch.Tensor:
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    final_layer = int(model.n_layers) - 1
    with torch.no_grad():
        residual = record_residuals(model, tensor, [final_layer])[final_layer]
        return fp32_next_token_log_probs(model, residual[:, -1, :])[0]


def _margin_and_output(model: Any, tokenizer: Any, prompt: str, device: Any) -> dict[str, Any]:
    formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False) + DECISION_PREFIX
    ids = input_ids(tokenizer, formatted, add_special_tokens=True)
    margin = score_single_token_margin_fp32(
        model, tokenizer, formatted, "buy", "sell", device=device
    )
    dist = _fp32_dist(model, ids, device)
    top1 = int(dist.argmax())
    out: dict[str, Any] = {
        "buy_logp": margin.positive.log_probability,
        "sell_logp": margin.negative.log_probability,
        "margin": float(margin.value),
        "greedy_token": decode_token(tokenizer, top1),
        "greedy_logp": float(dist[top1]),
    }
    current = list(ids)
    tokens: list[int] = []
    for _ in range(GREEDY_STEPS):
        d = _fp32_dist(model, current, device)
        token = int(d.argmax())
        tokens.append(token)
        current.append(token)
    out["greedy_text"] = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return out


def _greedy_tokens(model: Any, ids: list[int], device: Any, tokenizer: Any, steps: int) -> list[int]:
    current = list(ids)
    out: list[int] = []
    for _ in range(steps):
        d = _fp32_dist(model, current, device)
        token = int(d.argmax())
        out.append(token)
        current.append(token)
    return out


def _name_positions(tokenizer: Any, prompt: str, name: str) -> tuple[int, ...]:
    start = prompt.index(name)
    end = start + len(name)
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    return tuple(i for i, (s, e) in enumerate(offsets) if e > start and s < end and (s, e) != (0, 0))


def _header_positions(row: dict[str, Any]) -> tuple[int, ...]:
    identity = row.get("source_groups", {}).get("identity_header", {})
    return tuple(
        pos
        for start, end in identity.get("ranges", ())
        for pos in range(int(start), int(end))
    )


def _anonymous_prompt(prompt: str) -> str:
    prompt = re.sub(r"(?m)^(Stock Ticker: \[)[^\]\r\n]+(\])$", r"\g<1>ANON\g<2>", prompt, count=1)
    return re.sub(r"(?m)^(Stock Name: \[)[^\]\r\n]+(\])$", r"\g<1>Anonymous Company\g<2>", prompt, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--financial-prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--flip-ticker", default="AMZN")
    parser.add_argument("--flip-cell", default="0,1476")
    parser.add_argument("--wrong-cell", default="2,770")
    parser.add_argument("--reuse-scan", default=None, help="previous probe JSON whose direction_scan to reuse")
    parser.add_argument("--fact-frame", default=None, help="factual cloze template with {name}")
    parser.add_argument("--fact-name", default=None, help="entity name filling the fact frame")
    parser.add_argument("--baseline-stats", default=f"{DISCOVERY_RUN}/e1/baseline_stats.json")
    args = parser.parse_args()

    cell = tuple(int(part) for part in args.flip_cell.split(","))
    wrong = tuple(int(part) for part in args.wrong_cell.split(","))

    started = time.time()
    model, tokenizer, device = load_model(args.model)

    with open(args.baseline_stats) as fh:
        stats_payload = json.load(fh)
    stats = {int(layer): OnlineVectorStats.from_compact(value) for layer, value in stats_payload["layers"].items()}
    random_neuron = select_matched_random_neuron(stats, layer=cell[0], target_neuron=cell[1])
    rows_in = [json.loads(line) for line in open(args.financial_prompts)]
    end = ALPHA_GRID[-1]

    # ---- Optional fact-level specificity block. ----
    fact: dict[str, Any] | None = None
    if args.fact_frame and args.fact_name:
        prompt = args.fact_frame.format(name=args.fact_name)
        ids = input_ids(tokenizer, prompt, add_special_tokens=True)
        gold = _greedy_tokens(model, ids, device, tokenizer, GOLD_TOKENS)
        gold_text = tokenizer.decode(gold, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print(f"fact frame {args.fact_name!r} clean gold={gold_text!r} (ids={gold})", flush=True)
        fact_rows: list[dict[str, Any]] = []

        def fact_row(label: str, sup_cell: tuple[int, int] | None, scope: str,
                     positions: tuple[int, ...] | None) -> None:
            ctx = (
                MLPHookSession(model, [sup_cell[0]], channel_scales={sup_cell[0]: {sup_cell[1]: end}},
                               scope=scope, scope_positions=positions)
                if sup_cell is not None else nullcontext()
            )
            with ctx:
                logp = float(score_token_ids(model, tokenizer, prompt, gold, device=device).log_probability)
            fact_rows.append({"label": label, "cell": list(sup_cell) if sup_cell else None,
                              "scope": scope, "alpha": end if sup_cell else None, "gold_logp": logp})
            print(f"  fact {label:15s} {scope:14s} gold_logp={logp:+.3f}", flush=True)

        fact_row("clean", None, "all_positions", None)
        fact_row("target", cell, "all_positions", None)
        fact_row("target_name_only", cell, "header_only", _name_positions(tokenizer, prompt, args.fact_name))
        fact_row("wrong_entity", wrong, "all_positions", None)
        fact_row("matched_random", (cell[0], random_neuron), "all_positions", None)
        clean_logp = fact_rows[0]["gold_logp"]
        fact = {
            "frame": args.fact_frame, "name": args.fact_name,
            "gold_tokens": gold, "gold_text": gold_text, "rows": fact_rows,
            "target_collapse_nats": fact_rows[1]["gold_logp"] - clean_logp,
            "name_only_collapse_nats": fact_rows[2]["gold_logp"] - clean_logp,
            "wrong_entity_delta_nats": fact_rows[3]["gold_logp"] - clean_logp,
            "matched_random_delta_nats": fact_rows[4]["gold_logp"] - clean_logp,
        }

    # ---- Phase A: direction scan (clean entity vs anonymous header). ----
    if args.reuse_scan:
        with open(args.reuse_scan) as fh:
            previous = json.load(fh)
        scan = previous["direction_scan"]
        print(f"reused direction scan from {args.reuse_scan} ({len(scan)} rows)", flush=True)
    else:
        scan: list[dict[str, Any]] = []
        for row in rows_in:
            prompt = str(row["prompt"])
            clean = _margin_and_output(model, tokenizer, prompt, device)
            anon = _margin_and_output(model, tokenizer, _anonymous_prompt(prompt), device)
            gap = anon["margin"] - clean["margin"]
            opposite = bool(clean["margin"] and anon["margin"] and (clean["margin"] > 0) != (anon["margin"] > 0))
            scan.append({
                "ticker": row["ticker"], "name": row["name"], "prompt_id": row["prompt_id"],
                "clean": clean, "anonymous": anon, "anonymous_gap": gap,
                "opposite_direction": opposite, "eligible_gap_ge_0.1": abs(gap) >= 0.1,
            })
            flag = "OPPOSITE" if opposite else "same-dir"
            print(
                f"{row['ticker']} {row['prompt_id'][:12]} clean={clean['margin']:+.3f} "
                f"({clean['greedy_token']!r}) anon={anon['margin']:+.3f} ({anon['greedy_token']!r}) "
                f"gap={gap:+.3f} {flag}",
                flush=True,
            )

    # ---- Phase B: flip test on the flip-ticker's prompts. ----
    flip: list[dict[str, Any]] = []
    for s in (x for x in scan if x["ticker"] == args.flip_ticker):
        row = next(r for r in rows_in if r["prompt_id"] == s["prompt_id"])
        prompt = str(row["prompt"])
        header = _header_positions(row)
        clean_margin = s["clean"]["margin"]
        anon_margin = s["anonymous"]["margin"]
        clean_token = s["clean"]["greedy_token"]

        def sweep(label: str, sup_cell: tuple[int, int] | None, scope: str,
                  positions: tuple[int, ...] | None, display: bool) -> None:
            alphas = (ALPHA_GRID[0],) if sup_cell is None else ALPHA_GRID
            for alpha in alphas:
                ctx = (
                    MLPHookSession(model, [sup_cell[0]], channel_scales={sup_cell[0]: {sup_cell[1]: alpha}},
                                   scope=scope, scope_positions=positions)
                    if sup_cell is not None else nullcontext()
                )
                with ctx:
                    measured = _margin_and_output(model, tokenizer, prompt, device)
                flipped = bool(measured["margin"] and clean_margin and (measured["margin"] > 0) != (clean_margin > 0))
                flip.append({
                    "prompt_id": s["prompt_id"], "condition": label,
                    "cell": list(sup_cell) if sup_cell else None, "scope": scope, "alpha": alpha,
                    **measured, "clean_margin": clean_margin, "anonymous_margin": anon_margin,
                    "clean_token": clean_token,
                    "anonymous_progress": float(anonymous_progress(measured["margin"], clean_margin, anon_margin)),
                    "decision_flip": flipped,
                })
                if display or alpha == end:
                    mark = " <== FLIP" if flipped else ""
                    print(
                        f"{label:16s} {scope:14s} alpha={alpha:+.1f} margin={measured['margin']:+.3f} "
                        f"token={measured['greedy_token']!r} ap={flip[-1]['anonymous_progress']:+.3f}{mark}",
                        flush=True,
                    )

        sweep("clean", None, "all_positions", None, display=True)
        sweep("target", cell, "all_positions", None, display=False)
        sweep("target", cell, "header_only", header, display=False)
        sweep("wrong_entity", wrong, "all_positions", None, display=False)
        sweep("matched_random", (cell[0], random_neuron), "all_positions", None, display=False)

    # ---- Cross-ticker control: same target cell at end dose, other tickers. ----
    cross: list[dict[str, Any]] = []
    for s in (x for x in scan if x["ticker"] != args.flip_ticker):
        row = next(r for r in rows_in if r["prompt_id"] == s["prompt_id"])
        prompt = str(row["prompt"])
        clean_margin = s["clean"]["margin"]
        with MLPHookSession(model, [cell[0]], channel_scales={cell[0]: {cell[1]: end}},
                            scope="all_positions", scope_positions=None):
            measured = _margin_and_output(model, tokenizer, prompt, device)
        flipped = bool(measured["margin"] and clean_margin and (measured["margin"] > 0) != (clean_margin > 0))
        cross.append({
            "ticker": s["ticker"], "prompt_id": s["prompt_id"], "alpha": end,
            "clean_margin": clean_margin, **measured, "decision_flip": flipped,
        })
        print(
            f"cross {s['ticker']:6s} alpha={end:+.1f} clean={clean_margin:+.3f} "
            f"perturbed={measured['margin']:+.3f} token={measured['greedy_token']!r} "
            f"{'<== FLIP' if flipped else ''}",
            flush=True,
        )

    output = {
        "schema_version": 2,
        "artifact_type": "entity_cell_suppression_flip_probe",
        "status": "proposed_probe_not_frozen",
        "financial_prompts": args.financial_prompts,
        "discovery_run": DISCOVERY_RUN,
        "model": args.model,
        "device": str(device),
        "flip_ticker": args.flip_ticker,
        "target_cell": list(cell),
        "wrong_cell": list(wrong),
        "matched_random_neuron": random_neuron,
        "alpha_grid": list(ALPHA_GRID),
        "margin_definition": "logP(buy) - logP(sell) after DECISION_PREFIX, FP32 final-norm tail",
        "fact_block": fact,
        "direction_scan": scan,
        "flip_curves": flip,
        "cross_ticker_control": cross,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=1)
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
