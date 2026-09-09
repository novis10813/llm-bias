"""Balanced Evidence Gap — Phase 2A cross-entity probe operator.

Runs the 3-stage pipeline (prepare → forward → analyze) defined in
docs/balanced-evidence-gap/proposal-phase2.md §4.1: 16 tickers × 2 reverse
options × 2 evidence orders = 64 clean forwards, per-company pure entity
margins, gate 2A (IQR / Spearman vs Phase 1 / framing stability), and the
descriptive H4 readout of the investment-dial L15/n8490 coordinate.

Usage
-----
Smoke preflight (1 ticker, 2 prompts, writes a smoke run directory):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \\
        --model .cache/models/qwen3.5-4b --smoke

Formal run:
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \\
        --model .cache/models/qwen3.5-4b \\
        --run-id phase2a-gpu-bf16-01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from llm_bias.balanced_evidence_gap.pipeline import run_phase2a


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="Preflight (1 ticker, 2 prompts)")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")

    if args.smoke:
        run_id = f"phase2a-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        run_id = args.run_id or f"phase2a-gpu-bf16-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"

    started = time.time()
    print(f"[phase2a] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase2a(
        model_path=args.model,
        run_id=run_id,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    elapsed = time.time() - started

    import json

    elapsed = time.time() - started
    print(f"[phase2a] done in {elapsed:.0f}s")
    print(f"  run root: {run_root}")
    if args.smoke:
        rows = (Path(run_root) / "forward" / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        for line in rows:
            row = json.loads(line)
            print(f"  [smoke] {row['id']}  margin={row['margin']:+.3f}  [{row['decision']}]")
        print("[smoke] PASS")
        return
    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_2a"]
    print("── gate 2A ──")
    for name, crit in gate["criteria"].items():
        print(f"  {name:22s} value={crit['value']:+.4f}  threshold={crit['threshold']}  pass={crit['pass']}")
    print(f"  gate_2a pass           : {gate['pass']}")
    print(f"  phase2b authorized     : {gate['phase2b_authorized']}")
    h4 = summary["h4_dial"]
    print("── H4 (descriptive) ──")
    print(f"  dial L{h4['coordinate'][0]}/n{h4['coordinate'][1]}")
    print(f"  entity-position ρ vs pure entity margin: {h4['entity_position_pearson']:+.3f}")
    print(f"  final-position ρ   vs pure entity margin: {h4['final_position_pearson']:+.3f}")


if __name__ == "__main__":
    main()
