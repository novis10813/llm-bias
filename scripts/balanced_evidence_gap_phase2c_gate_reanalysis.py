"""Balanced Evidence Gap — Phase 2C gate re-analysis operator.

CPU-only re-evaluation of a completed 2C run under the frozen protocol's
per-arm existence semantics (proposal §4.5). No model load, no GPU.

Usage
-----
    uv run --no-sync python scripts/balanced_evidence_gap_phase2c_gate_reanalysis.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2c-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2c-gpu-bf16-05 \\
        --run-id phase2c-gate-reanalysis-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.balanced_evidence_gap.gate_reanalysis import run_2c_gate_reanalysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path (artifact slug only)")
    parser.add_argument("--phase2c-run", required=True, help="Completed 2C run root to re-analyze")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")

    run_id = args.run_id or f"phase2c-gate-reanalysis-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    started = time.time()
    print(f"[2c-gate-reanalysis] run_id={run_id}", flush=True)
    print(f"[2c-gate-reanalysis] source run: {args.phase2c_run}", flush=True)
    run_root = run_2c_gate_reanalysis(
        model_name=Path(args.model).name,
        phase2c_run=args.phase2c_run,
        run_id=run_id,
        artifact_root=args.artifact_root,
    )
    elapsed = time.time() - started

    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_2c"]
    print(f"[2c-gate-reanalysis] done in {elapsed:.1f}s")
    print(f"  run root: {run_root}")
    print("── gate 2C (existence semantics) ──")
    att = gate["attention_arm"]
    if att["status"] == "run":
        print(f"  attention: {att['n_heads']} heads; top={att['top_head']} "
              f"effect={att['top_effect']:+.4f} adj_p={att['sign_flip_p_adjusted']:.4f} "
              f"passing={att['passing_heads'][:5]} pass={att['pass']}")
    else:
        print(f"  attention: {att['status']}")
    mlp = gate["mlp_arm"]
    print(f"  mlp: {mlp['n_layers']} layers; passing={mlp['passing_layers']} pass={mlp['pass']}")
    print(f"  gate 2C pass  : {gate['pass']}")


if __name__ == "__main__":
    main()
