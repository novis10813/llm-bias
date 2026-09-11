"""Balanced Evidence Gap — Phase 3: causal validation of 2C coordinates.

Frozen protocol: docs/balanced-evidence-gap/proposal-phase3.md (Rev 1).
Additive mlp_addition grid on L19/n6334, L20/n6520, L26/n2394 (+ 30
matched controls, dial descriptive arm) over the 16-ticker 2A canonical
prompt set; gate 3A.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    uv run --no-sync python scripts/balanced_evidence_gap_phase3.py \
        --model .cache/models/qwen3.5-4b \
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
        --phase2c-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2c-gpu-bf16-05 \
        --phase2c-reanalysis-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2c-gate-reanalysis-01 \
        --run-id phase3-gpu-bf16-01

Smoke (4 tickers, all stages; gate is descriptive at n=4):
    ... --run-id phase3-smoke-01 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.balanced_evidence_gap.neuron_causal import run_phase3


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path (artifact slug only)")
    parser.add_argument("--phase2a-run", required=True, help="Completed 2A run root")
    parser.add_argument("--phase2c-run", required=True, help="Completed 2C run root (control rho source)")
    parser.add_argument("--phase2c-reanalysis-run", required=True, help="2C gate re-analysis run root (candidate source)")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="4 tickers, skip intervene/analyze")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")

    run_id = args.run_id or f"phase3-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    started = time.time()
    print(f"[phase3] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase3(
        model_path=args.model,
        phase2a_run=args.phase2a_run,
        phase2c_run=args.phase2c_run,
        phase2c_reanalysis_run=args.phase2c_reanalysis_run,
        run_id=run_id,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    elapsed = time.time() - started
    print(f"[phase3] done in {elapsed:.1f}s; run root: {run_root}", flush=True)

    summary_path = Path(run_root) / "analyze" / "summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = summary["gate_3a"]
    print("── gate 3A " + ("(smoke n=4, descriptive only)" if args.smoke else "──"), flush=True)
    for c in gate["candidates"]:
        crit = c["criteria"]
        print(
            f"  {c['name']}: mean ΔM@gate={c['mean_delta_at_gate']:+.4f} "
            f"({c['n_same_direction']}/{c['n_tickers']} same dir, "
            f"adj p={c['sign_flip_p_adjusted']:.4f}, "
            f"ctl max={c['control_layer_max_abs_mean_delta']:.4f}) "
            f"dir={crit['direction']} cons={crit['consistency']} "
            f"ctl={crit['control_superiority']} -> {c['pass']}"
        )
    print(f"  confirmed={gate['confirmed']}/{len(gate['candidates'])} "
          f"(min {gate['min_confirmed']})  gate 3A pass: {gate['pass']}")


if __name__ == "__main__":
    main()
