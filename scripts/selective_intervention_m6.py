"""Selective-intervention M6-V2 current-runtime compatibility validation.

This operator consumes a frozen external-company manifest and the existing
V1/e-01 artifacts. It does not fit or tune the subspace on the external rows;
V2 records the current deterministic center digest separately from the V1 digest.

Formal command (after real-model smoke authorization):
    uv run python scripts/selective_intervention_m6.py \
      --model .cache/models/qwen3.5-4b \
      --external-manifest <frozen-m6-external-manifest> \
      --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
      --e01-run artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01 \
      --v1-run artifacts/qwen3.5-4b/selective-intervention/runs/selective-intervention-v1-gpu-bf16-01 \
      --run-id <m6-run-id>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.selective_intervention.m6_pipeline import (
    run_selective_intervention_m6,
    run_selective_intervention_m6_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--external-manifest", required=True, help="Frozen M6 external-company manifest")
    parser.add_argument("--phase2a-run", required=True, help="Frozen Phase 2A prompt-template run")
    parser.add_argument("--e01-run", required=True, help="Frozen entity-to-dial Phase E run")
    parser.add_argument("--v1-run", required=True, help="Frozen selective-intervention V1 run")
    parser.add_argument("--run-id", required=True, help="M6 run identifier")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one real-model external prompt through clean/main/random arms; do not compute formal M6 analysis",
    )
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    for label, value in (
        ("model", args.model),
        ("external-manifest", args.external_manifest),
        ("phase2a-run", args.phase2a_run),
        ("e01-run", args.e01_run),
        ("v1-run", args.v1_run),
    ):
        path = Path(value)
        if label == "model":
            if not path.is_dir():
                sys.exit(f"ERROR: {label} path not found: {value}")
        elif label == "external-manifest":
            if not path.is_file():
                sys.exit(f"ERROR: {label} not found: {value}")
        elif not path.joinpath("manifest.json").is_file():
            sys.exit(f"ERROR: {label} not found or missing manifest: {value}")

    started = time.time()
    print(f"[selective-intervention-m6] run_id={args.run_id}", flush=True)
    runner = run_selective_intervention_m6_smoke if args.smoke else run_selective_intervention_m6
    run_root = runner(
        model_path=args.model,
        run_id=args.run_id,
        external_manifest=args.external_manifest,
        phase2a_run=args.phase2a_run,
        e01_run=args.e01_run,
        v1_run=args.v1_run,
        artifact_root=args.artifact_root,
    )
    print(f"[selective-intervention-m6] done in {time.time() - started:.0f}s")
    print(f"  run root: {run_root}")
    if args.smoke:
        smoke = json.loads((Path(run_root) / "forward" / "smoke.json").read_text(encoding="utf-8"))
        print(f"  smoke prompt={smoke['prompt_id']} center_digest={smoke['center_digest_l15']}")
        print(f"  smoke margins={smoke['margins']}")
        print("  formal M6 analysis was not run")
    else:
        summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
        primary = summary["primary"]
        print(f"  primary: {primary['interpretation']['status']} ratio={primary['spread']['ratio']:.4f} ci={primary['spread']['bootstrap']['ci']}")
        print(f"  specificity pass={summary['specificity']['pass']}")
        print(f"  G3' pass={summary['safety']['g3']['pass']} G4' pass={summary['safety']['g4']['pass']}")
        print(f"  generation main={summary['generation']['main']}")


if __name__ == "__main__":
    main()
