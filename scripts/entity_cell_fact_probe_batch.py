"""Batch runner for the entity-cell factual amnesia probe (proposed).

Loads the model once and runs multiple (entity, cell) probe targets
sequentially via the shared run_probe helper; intended to be split into
several workers (one --indices subset each) so multiple models run in
parallel across free GPUs.

Usage (from the repo root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/entity_cell_fact_probe_batch.py \
        --targets /tmp/fact_probe_batch_targets.json \
        --indices 0,1,2,3 \
        --baseline-stats artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e1-hfm2-discovery-v1/e1/baseline_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import entity_cell_factual_amnesia_probe as probe
from llm_bias.core.model import load_model
from llm_bias.entity_cell.mlp_cells import OnlineVectorStats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="JSON array of probe target configs")
    parser.add_argument("--indices", required=True, help="comma-separated target indices for this worker")
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--baseline-stats", required=True)
    args = parser.parse_args()

    targets = json.load(open(args.targets))
    idxs = [int(part) for part in args.indices.split(",")]
    for i in idxs:
        if not (0 <= i < len(targets)):
            raise SystemExit(f"target index {i} out of range (0..{len(targets) - 1})")

    model, tokenizer, device = load_model(args.model)
    with open(args.baseline_stats) as fh:
        stats_payload = json.load(fh)
    stats = {int(layer): OnlineVectorStats.from_compact(value) for layer, value in stats_payload["layers"].items()}

    for i in idxs:
        target = targets[i]
        print(f"=== target {i}: {target['label']} ({target['entity']} @ {target['cell']}) ===", flush=True)
        output = probe.run_probe(
            model, tokenizer, device, stats,
            entity=target["entity"],
            target_cell=tuple(int(part) for part in target["cell"].split(",")),
            wrong_cell=tuple(int(part) for part in target["wrong_cell"].split(",")),
            cross=target["cross"],
            source_run=target.get("source_run", probe.RUN),
            label=target["label"],
        )
        output["model"] = args.model
        with open(target["output"], "w") as fh:
            json.dump(output, fh, indent=1)
        print(f"written {target['output']} in {output['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
