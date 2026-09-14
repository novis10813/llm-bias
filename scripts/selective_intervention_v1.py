"""Selective-intervention V1 — L15 k=8 subspace removal (protocol-v1 Rev 1).

Runs the 3-stage pipeline (prepare → forward → analyze) defined in
docs/selective-intervention/proposal-v1.md (Rev 1, frozen): clean
bit-exact 2A reference, strength sweep (dose), centering / position /
layer / random-subspace controls, anonymous + dial probes; gates
G1a/G1b (efficacy), G2 (specificity), G3/G4 (task preservation).

Usage
-----
Smoke preflight (4 group companies, reduced grid; mechanism validation):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/selective_intervention_v1.py \\
        --model .cache/models/qwen3.5-4b \\
        --smoke

Formal run (64 prompts, ~663 forwards, ~60-70 min):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/selective_intervention_v1.py \\
        --model .cache/models/qwen3.5-4b \\
        --run-id selective-intervention-v1-gpu-bf16-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.selective_intervention.pipeline import run_selective_intervention_v1
from llm_bias.selective_intervention.template import DEFAULT_E01_RUN, DEFAULT_PHASE2A_RUN


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--phase2a-run", default=DEFAULT_PHASE2A_RUN)
    parser.add_argument("--e01-run", default=DEFAULT_E01_RUN)
    parser.add_argument("--run-id", default=None, help="Run identifier (required for formal runs)")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="Preflight (4 group companies, reduced grid)")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")
    for label, value in (
        ("phase2a-run", args.phase2a_run),
        ("e01-run", args.e01_run),
    ):
        if not Path(value, "manifest.json").is_file():
            sys.exit(f"ERROR: {label} not found or missing manifest: {value}")
    if args.smoke:
        run_id = f"selective-intervention-v1-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        run_id = args.run_id
        if not run_id:
            sys.exit("ERROR: --run-id is required for formal runs")

    started = time.time()
    print(f"[selective-intervention-v1] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_selective_intervention_v1(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=args.phase2a_run,
        e01_run=args.e01_run,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(f"[selective-intervention-v1] done in {time.time() - started:.0f}s")
    print(f"  run root: {run_root}")
    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_v1"]
    print("── gate V1 ──")
    if "pass" in gate:
        for name in ("g1a", "g1b", "g2", "g3", "g4"):
            item = gate[name]
            value = f"{item['value']:+.4f}" if "value" in item else "n/a"
            limit = f"(limit {item['limit']:.4f})" if "limit" in item else ""
            print(f"  {name:5s} pass={str(item['pass']):5s} value={value} {limit}".rstrip())
        print(f"  group gap: clean {gate['group_gap_clean']:+.4f} → "
              f"intervention {gate['group_gap_intervention']:+.4f} → "
              f"random {gate['group_gap_random']:+.4f}")
        print(f"  iqr: clean {gate['iqr_clean']:.4f} → intervention {gate['iqr_intervention']:.4f}")
        print(f"  gate_v1 pass: {gate['pass']}")
    else:
        print(f"  {gate.get('reason', 'not evaluated')}")
    print(f"  clean stats: gap {summary['clean_stats']['group_gap']:+.4f}  "
          f"iqr {summary['clean_stats']['iqr']:.4f}  "
          f"mean {summary['clean_stats']['mean_margin']:+.4f}  "
          f"{summary['clean_stats']['decisions']}")
    for point in summary["dose_response"]:
        print(f"  dose α={point['alpha']:.2f}: gap {point['group_gap']:+.4f}  "
              f"iqr {point['iqr']:.4f}  mean {point['mean_margin']:+.4f}  "
              f"flips {point['flips_sell_to_buy']}")
    print(f"  anon probe: {summary['anon_probe']['clean']:+.4f} → "
          f"{summary['anon_probe']['intervention']:+.4f} (Δ {summary['anon_probe']['delta']:+.4f})")
    if args.smoke:
        n_records = len((Path(run_root) / "forward" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
        print(f"  [smoke] {n_records} forward records; acceptance enforced in-pipeline:")
        print("    #1 clean vs 2A archive bit-exact (per prompt, fail-closed)")
        print("    #2 calibration centers finite + digests recorded")
        print("    #3 records finite, decisions valid")
        print("    #4 summary schema passes core serializer reserved-key guard")
        print("    #5 manifest complete (prepare/forward/analyze)")
        print("[smoke] PASS")


if __name__ == "__main__":
    main()
