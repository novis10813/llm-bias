"""Entity-to-Dial Held-out Transfer V1 runner.

Smoke only validates one deterministic, ticker-sorted pair through the frozen
L15 V8 transplant operator and its controls. It never runs the 200-company
formal evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.entity_to_dial.heldout_transfer import run_heldout_transfer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--phase-e-run", required=True, help="Completed entity-to-dial E-01 run")
    parser.add_argument("--m6-manifest", required=True, help="Frozen M6 external-company manifest")
    parser.add_argument("--population-csv", required=True, help="Canonical 2024 S&P 500 CSV")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--run-id", default=None, help="Required for a formal 200-company run")
    parser.add_argument("--smoke", action="store_true", help="Run one fixed pair only; no formal graphs or analysis")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")
    if not Path(args.phase_e_run, "manifest.json").is_file():
        sys.exit(f"ERROR: phase-e-run not found or missing manifest: {args.phase_e_run}")
    if not Path(args.m6_manifest).is_file():
        sys.exit(f"ERROR: m6-manifest not found: {args.m6_manifest}")
    if not Path(args.population_csv).is_file():
        sys.exit(f"ERROR: population-csv not found: {args.population_csv}")

    if args.smoke:
        run_id = f"entity-to-dial-heldout-transfer-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        if not args.run_id:
            sys.exit("ERROR: --run-id is required for a formal run")
        run_id = args.run_id
    started = time.time()
    print(f"[entity-to-dial-heldout-transfer] run_id={run_id} smoke={args.smoke}", flush=True)
    root = run_heldout_transfer(
        model_path=args.model,
        run_id=run_id,
        phase_e_run=args.phase_e_run,
        m6_manifest=args.m6_manifest,
        population_csv=args.population_csv,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(f"[entity-to-dial-heldout-transfer] done in {time.time() - started:.0f}s")
    print(f"  run root: {root}")
    if args.smoke:
        smoke = json.loads((Path(root) / "forward" / "smoke.json").read_text(encoding="utf-8"))
        print(f"  smoke pair: {smoke['source_ticker']} -> {smoke['target_ticker']}")
        print(f"  margins: {smoke['margins']}")
        print("  formal pair graphs and analysis were not run")


if __name__ == "__main__":
    main()
