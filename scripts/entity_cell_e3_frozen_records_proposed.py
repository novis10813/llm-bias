"""Entity-cell E3 frozen-format records with explicit (proposed) selection.

The frozen E3 CLI (entity-cell run-intervention) selects intervention targets
from E1's formal trusted-candidate eligibility. Discovery runs with zero
formally trusted tickers (e.g. entity-cell-e1-hfm2-discovery-v1) therefore
cannot drive the frozen CLI, even when the fact-level probes identify a
genuine entity cell.

This operator calls the same frozen E3 record functions
(llm_bias.entity_cell.e3.run_upstream_suppression_record /
run_downstream_suppression_record) directly with explicitly selected
cells/heads, producing byte-compatible compact E3-A / E3-B records
(phase, dose, margin, clean/anonymous margin, flip, controls, provenance).
The selection itself is a proposed extension and is recorded in the output
provenance; no frozen protocol files are modified.

Usage (from the repo root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/entity_cell_e3_frozen_records_proposed.py \
        --output /path/to/e3_records.json \
        --ticker JNJ --cell 4,7676 --wrong-cell 0,1476 \
        --financial-prompts /path/to/financial_prompts.jsonl \
        [--heads 23,5 27,2] \
        [--baseline-stats /path/to/baseline_stats.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from llm_bias.core.model import load_model
from llm_bias.entity_cell.e3 import (
    run_downstream_suppression_record,
    run_upstream_suppression_record,
)
from llm_bias.entity_cell.mlp_cells import OnlineVectorStats, select_matched_random_neuron
from llm_bias.entity_cell.suppression import E3_ALPHA_GRID, E3_BETA_GRID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--cell", required=True, help="target cell L,N")
    parser.add_argument("--wrong-cell", required=True, help="wrong-entity control cell L,N")
    parser.add_argument("--financial-prompts", required=True)
    parser.add_argument("--baseline-stats", required=True)
    parser.add_argument("--heads", nargs="*", default=[], help="selected heads as L,H pairs (enables mediation + E3-B)")
    parser.add_argument("--grouping", default="single", choices=("single", "group"))
    args = parser.parse_args()

    cell = {"layer": int(args.cell.split(",")[0]), "neuron": int(args.cell.split(",")[1])}
    wrong = {"layer": int(args.wrong_cell.split(",")[0]), "neuron": int(args.wrong_cell.split(",")[1])}
    heads = [(int(part.split(",")[0]), int(part.split(",")[1])) for part in args.heads]

    started = time.time()
    model, tokenizer, device = load_model(args.model)

    with open(args.baseline_stats) as fh:
        stats_payload = json.load(fh)
    stats = {int(layer): OnlineVectorStats.from_compact(value) for layer, value in stats_payload["layers"].items()}
    random_neuron = select_matched_random_neuron(stats, layer=cell["layer"], target_neuron=cell["neuron"])

    rows = [json.loads(line) for line in open(args.financial_prompts) if json.loads(line)["ticker"] == args.ticker]
    if not rows:
        raise SystemExit(f"no financial prompts for ticker {args.ticker}")

    records: list[dict[str, Any]] = []
    for row in rows:
        # E3-A upstream (frozen readout), both scopes, with mediation when heads are given.
        upstream = run_upstream_suppression_record(
            model, tokenizer,
            prompt_row=row, candidate=cell, wrong_candidate=wrong, random_neuron=random_neuron,
            selected_heads=tuple(heads), alpha_grid=E3_ALPHA_GRID,
            scopes=("all_positions", "header_only"), device=device,
        )
        for record in upstream:
            record["provenance"]["selection"] = "proposed_explicit_selection"
            record["provenance"]["source_ticker_note"] = f"{args.ticker} cell selected from fact-level probe evidence, not formal E1 eligibility"
        records.extend(upstream)
        print(f"{row['prompt_id'][:16]} E3-A {len(upstream)} records", flush=True)

        if heads:
            downstream = run_downstream_suppression_record(
                model, tokenizer,
                prompt_row=row, selected_heads=tuple(heads), beta_grid=E3_BETA_GRID,
                grouping=args.grouping, device=device,
            )
            for record in downstream:
                record["provenance"]["selection"] = "proposed_explicit_selection"
            records.extend(downstream)
            print(f"{row['prompt_id'][:16]} E3-B {len(downstream)} records", flush=True)

    output = {
        "schema_version": 1,
        "artifact_type": "entity_cell_e3_frozen_records_proposed",
        "status": "proposed_selection_frozen_readout",
        "model": args.model,
        "device": str(device),
        "ticker": args.ticker,
        "target_cell": [cell["layer"], cell["neuron"]],
        "wrong_cell": [wrong["layer"], wrong["neuron"]],
        "matched_random_neuron": random_neuron,
        "selected_heads": [list(item) for item in heads],
        "alpha_grid": list(E3_ALPHA_GRID),
        "beta_grid": list(E3_BETA_GRID),
        "prompt_ids": [str(row["prompt_id"]) for row in rows],
        "records": records,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=1)
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
