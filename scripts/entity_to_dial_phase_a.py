"""Entity-to-Dial Phase A — token-group x layer sufficiency map.

Runs the 3-stage pipeline (prepare → forward → analyze) defined in
docs/entity-to-dial/proposal.md §4.2 (Rev 1): 8 directions x 12 layers x
2 token-groups (ticker/name) patch forwards, self-source no-op discipline,
gate A (ticker-group L0–5 sufficiency), and the 2B entity-span
upper-bound reference.

Usage
-----
Smoke preflight (2 directions x 4 layers x 2 groups + no-ops):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_a.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --smoke

Formal run (8 directions x 12 layers x 2 groups):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_a.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --run-id entity-to-dial-a-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.entity_to_dial.pipeline import run_phase_a
from llm_bias.entity_to_dial.template import (
    DEFAULT_PHASE2A_REV2_RUN,
    DEFAULT_PHASE2A_RUN,
    DEFAULT_PHASE2B_RUN,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--phase2a-run", default=DEFAULT_PHASE2A_RUN)
    parser.add_argument("--phase2b-run", default=DEFAULT_PHASE2B_RUN)
    parser.add_argument("--phase2a-rev2-run", default=DEFAULT_PHASE2A_REV2_RUN)
    parser.add_argument("--run-id", default=None, help="Run identifier (required for formal runs)")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="Preflight (2 directions x 4 layers)")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")
    for label, value in (
        ("phase2a-run", args.phase2a_run),
        ("phase2b-run", args.phase2b_run),
        ("phase2a-rev2-run", args.phase2a_rev2_run),
    ):
        if not Path(value, "manifest.json").is_file():
            sys.exit(f"ERROR: {label} not found or missing manifest: {value}")
    if args.smoke:
        run_id = f"entity-to-dial-a-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        run_id = args.run_id
        if not run_id:
            sys.exit("ERROR: --run-id is required for formal runs")

    started = time.time()
    print(f"[entity-to-dial-a] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase_a(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=args.phase2a_run,
        phase2b_run=args.phase2b_run,
        phase2a_rev2_run=args.phase2a_rev2_run,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(f"[entity-to-dial-a] done in {time.time() - started:.0f}s")
    print(f"  run root: {run_root}")
    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_a"]
    print("── gate A ──")
    if gate["status"] == "evaluated":
        a1 = gate["a1"]
        print(f"  A1 ticker L0–5  mean CI 95% : [{a1['ci_95'][0]:+.4f}, {a1['ci_95'][1]:+.4f}]  pass={a1['pass']}")
    else:
        print(f"  {gate['reason']}")
    if args.smoke:
        records = (Path(run_root) / "forward" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
        print(f"  [smoke] {len(records)} forward records; no-op discipline verified")
        print("[smoke] PASS")


if __name__ == "__main__":
    main()
