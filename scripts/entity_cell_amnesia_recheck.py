"""Targeted E1 V2 amnesia recheck under the v2 (corrected) FP32 instrument.

Recomputes, for the E1 V2 discovery candidates, the frozen amnesia endpoint
gate (alpha=-3, all_positions): target A_p > 0 and above the non-degenerate
wrong-entity / matched-random controls on >= 2 eligible prompts
(|anon - clean| >= 0.1). Tickers marked in --full-curve also get the full
dose curve (both scopes, all ALPHA_GRID doses).

Candidates (target / wrong / random) are the frozen E1 V2 discovery values
from entity-cell-e1-discovery-v2 (see CANDIDATES below); this operator is
bound to that frozen run and must not be pointed at other candidate sets
without a new version of the recheck contract. Uses the repo's own frozen
functions (mlp_hooks, anonymous_progress, _replace_header) so the gate
semantics are identical to the pipeline.

Usage (from the repo root):
    uv run python scripts/entity_cell_amnesia_recheck.py \
        --output /path/to/amnesia_recheck.json \
        [--tickers FTNT AKAM ...] [--full-curve FTNT] \
        [--prepared-dir ...] [--model ...]
"""
from __future__ import annotations

import argparse
import json

import torch

from llm_bias.core.continuation_scoring import score_single_token_margin_fp32
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import format_prompt
from llm_bias.entity_cell.mlp_cells import (
    ALPHA_GRID,
    DECISION_PREFIX,
    NEGATIVE_CANDIDATE,
    POSITIVE_CANDIDATE,
    _replace_header,
    anonymous_progress,
    mlp_hooks,
)

# Frozen E1 V2 discovery candidates (entity-cell-e1-discovery-v2):
# ticker -> (target (layer,neuron), wrong (layer,neuron), random (layer,neuron), degenerate)
CANDIDATES = {
    "FTNT": ((0, 104), (0, 5101), (0, 1927), False),
    "AKAM": ((0, 104), (0, 37), (0, 1927), False),
    "AMAT": ((0, 104), (0, 62), (0, 1927), False),
    "CTSH": ((0, 5101), (0, 104), (0, 3135), False),
    "FFIV": ((0, 3416), (0, 104), (0, 3981), False),
    "IBM": ((0, 4485), (0, 104), (0, 7878), False),
    "JKHY": ((0, 104), (0, 62), (0, 1927), False),
    "MU": ((0, 104), (0, 37), (0, 1927), False),
    "TXN": ((0, 4485), (0, 104), (0, 7878), False),
}


def header_positions(ranges):
    return tuple(pos for start, end in ranges for pos in range(int(start), int(end)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-prepare-discovery-v2/prepare")
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tickers", nargs="+", default=sorted(CANDIDATES), choices=sorted(CANDIDATES))
    parser.add_argument("--full-curve", nargs="*", default=["FTNT"], choices=sorted(CANDIDATES))
    parser.add_argument("--min-eligible-prompts", type=int, default=2)
    args = parser.parse_args()

    # CPU-only operator: the corrected FP32 tail is the whole point of the recheck.
    torch.cuda.is_available = lambda: False
    model, tokenizer, device = load_model(args.model)

    def margin(prompt: str) -> float:
        formatted = format_prompt(tokenizer, prompt=prompt, use_chat_template=True, enable_thinking=False) + DECISION_PREFIX
        return float(score_single_token_margin_fp32(model, tokenizer, formatted, POSITIVE_CANDIDATE, NEGATIVE_CANDIDATE, device=device).value)

    def anon_prompt(p: str) -> str:
        return _replace_header(p, "ANON", "Anonymous Company")

    prompts: dict[str, list[tuple[str, str, list]]] = {}
    with open(f"{args.prepared_dir}/financial_prompts.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            prompts.setdefault(r["ticker"], []).append(
                (r["prompt_id"], r["prompt"], r.get("source_groups", {}).get("identity_header", {}).get("ranges", []))
            )

    out: dict[str, dict] = {}
    for ticker in sorted(args.tickers):
        target, wrong, random_cell, degenerate = CANDIDATES[ticker]
        rows = []
        for pid, p, hdr_ranges in sorted(prompts[ticker]):
            clean = margin(p)
            an = margin(anon_prompt(p))
            gap = an - clean
            eligible = abs(gap) >= 0.1
            hdr = header_positions(hdr_ranges)
            scopes = {"all_positions": None, "header_only": hdr} if ticker in args.full_curve else {"all_positions": None}
            for scope, positions in scopes.items():
                for alpha in ALPHA_GRID:
                    rec = {"prompt_id": pid, "scope": scope, "alpha": alpha, "clean": clean, "anon": an, "gap": gap, "eligible": eligible}
                    for label, cell in (("target", target), ("wrong_entity", wrong), ("matched_random", random_cell)):
                        with mlp_hooks(model, [cell[0]], channel_scales={cell[0]: {cell[1]: float(alpha)}}, scope=scope, scope_positions=positions):
                            m = margin(p)
                        rec[label] = m
                        rec[label + "_progress"] = anonymous_progress(m, clean, an)
                    rows.append(rec)
        endpoint = [r for r in rows if r["scope"] == "all_positions" and r["alpha"] == -3.0 and r["eligible"]]
        passed = []
        for r in endpoint:
            controls = {"wrong_entity": r["wrong_entity_progress"], "matched_random": r["matched_random_progress"]}
            if degenerate:
                controls.pop("wrong_entity")
            if r["target_progress"] > 0 and all(r["target_progress"] > v for v in controls.values()):
                passed.append(r["prompt_id"])
        out[ticker] = {
            "eligible_prompt_count": len(endpoint),
            "endpoint_pass_prompt_ids": passed,
            "endpoint_pass_count": len(passed),
            "gate_pass": len(passed) >= args.min_eligible_prompts,
            "endpoint_rows": [
                {
                    "prompt_id": r["prompt_id"],
                    "clean": round(r["clean"], 4), "anon": round(r["anon"], 4),
                    "target_progress": round(r["target_progress"], 4),
                    "wrong_progress": round(r["wrong_entity_progress"], 4),
                    "random_progress": round(r["matched_random_progress"], 4),
                    "target_margin": round(r["target"], 4),
                }
                for r in sorted(endpoint, key=lambda x: x["prompt_id"])
            ],
            "dose_curve": (
                [
                    {
                        "prompt_id": r["prompt_id"], "scope": r["scope"], "alpha": r["alpha"],
                        "target": round(r["target"], 4),
                        "wrong": round(r["wrong_entity"], 4),
                        "random": round(r["matched_random"], 4),
                    }
                    for r in rows
                ]
                if ticker in args.full_curve
                else None
            ),
        }
        print(f"done {ticker}: gate_pass={out[ticker]['gate_pass']} ({out[ticker]['endpoint_pass_count']}/{len(endpoint)} prompts)", flush=True)

    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
