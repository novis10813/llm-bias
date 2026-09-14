"""Balanced Evidence Gap — Phase 2 Rev 2 gate re-evaluation operator.

CPU-only re-analysis of an existing 2A forward run under the Rev 2 protocol
(docs/balanced-evidence-gap/details/proposal-phase2-rev2.md): gate 2A with the
Phase 1 gap (named-vs-anonymous) as the construct reference, plus the
group construct check. No model load, no GPU, no new inference.

Usage
-----
    uv run --no-sync python scripts/balanced_evidence_gap_phase2_rev2.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --run-id phase2a-rev2-gate-01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from llm_bias.balanced_evidence_gap.pipeline import PHASE1_SUMMARY_PATH
from llm_bias.balanced_evidence_gap.rev2 import DEFAULT_PHASE2A_RUN, run_rev2_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path (artifact slug only)")
    parser.add_argument("--phase2a-run", default=str(DEFAULT_PHASE2A_RUN), help="2A run root to re-analyze")
    parser.add_argument("--phase1-summary", default=str(PHASE1_SUMMARY_PATH), help="Phase 1 analyze/summary.json")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")

    run_id = args.run_id or f"phase2a-rev2-gate-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    started = time.time()
    print(f"[rev2-gate] run_id={run_id}", flush=True)
    print(f"[rev2-gate] 2A run: {args.phase2a_run}", flush=True)
    run_root = run_rev2_gate(
        model_name=Path(args.model).name,
        phase2a_run=args.phase2a_run,
        phase1_summary=args.phase1_summary,
        run_id=run_id,
        artifact_root=args.artifact_root,
    )
    elapsed = time.time() - started

    import json

    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_2a_rev2"]
    print(f"[rev2-gate] done in {elapsed:.1f}s")
    print(f"  run root: {run_root}")
    print("── gate 2A Rev 2 ──")
    for name, crit in gate["criteria"].items():
        if name == "group_construct_check":
            print(f"  {name:22s} top={','.join(crit['top'])} bottom={','.join(crit['bottom'])}  "
                  f"{crit['n_positive']}/{crit['n_pairs']} pairs positive  pass={crit['pass']}")
        else:
            print(f"  {name:22s} value={crit['value']:+.4f}  threshold={crit['threshold']}  pass={crit['pass']}")
    print(f"  gate pass              : {gate['pass']}")
    print(f"  phase2b authorized     : {gate['phase2b_authorized']}")
    desc = summary["descriptive"]
    print("── descriptive ──")
    print(f"  spearman vs phase1 named : {desc['spearman_vs_phase1_named']:+.4f}")
    print(f"  spearman vs phase1 gap   : {desc['spearman_vs_phase1_gap']:+.4f}")


if __name__ == "__main__":
    main()
