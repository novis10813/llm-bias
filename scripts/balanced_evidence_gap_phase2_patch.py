"""Balanced Evidence Gap — Phase 2B/2C patching operator.

Experiment 2B (entity-state layer sweep) and 2C (component attribution),
per docs/balanced-evidence-gap/details/proposal-phase2.md §4.4–§4.5.

2B requires a completed 2A run (gate 2A pass authorizes it). 2C requires
a completed 2B run (its handoff interval selects the layers).

Usage
-----
2B formal (8 directions × 32 layers × 4 spans):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py \\
        sweep --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/<2A run> \\
        --run-id phase2b-gpu-bf16-01

2C formal (attention arm on full-attention layers in the handoff interval,
MLP attribution on all layers in the interval):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py \\
        attribute --model .cache/models/qwen3.5-4b \\
        --phase2a-run ... --phase2b-run ... \\
        --run-id phase2c-gpu-bf16-01

Smoke variants (--smoke) use reduced layers/directions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.balanced_evidence_gap.patch_pipeline import run_phase2b, run_phase2c


def _sweep(args: argparse.Namespace) -> None:
    phase2a = Path(args.phase2a_run)
    gate_pass = False
    gate_name = None
    if not args.smoke:
        if args.gate_run:
            gate_root = Path(args.gate_run)
            gate_summary = json.loads((gate_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
            gate = gate_summary["gate_2a_rev2"]
            gate_name = f"gate 2A Rev 2 ({gate_root.name})"
        else:
            summary = json.loads((phase2a / "analyze" / "summary.json").read_text(encoding="utf-8"))
            gate = summary["gate_2a"]
            gate_name = "gate 2A (Rev 1)"
        gate_pass = bool(gate["pass"])
        if not gate_pass:
            if not args.gate_override:
                sys.exit(
                    f"ERROR: {gate_name} did not pass; 2B is not authorized "
                    "(rerun the gate or use a passing run)"
                )
            print(
                f"[phase2b] WARNING: {gate_name} did not pass; proceeding under "
                f"development override: {args.gate_override}",
                flush=True,
            )
        else:
            print(f"[phase2b] authorization: {gate_name} pass=True", flush=True)
    run_id = args.run_id or (
        f"phase2b-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        if args.smoke else
        f"phase2b-gpu-bf16-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    started = time.time()
    print(f"[phase2b] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase2b(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=phase2a,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
        device_map=args.device_map,
        dtype=args.dtype,
        gate_pass=gate_pass if not args.smoke else None,
        gate_override=(args.gate_override if (not args.smoke and not gate_pass) else None),
        gate_name=gate_name,
        family=args.family,
        direction_min_gap=args.direction_min_gap,
    )
    elapsed = time.time() - started
    out = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    print(f"[phase2b] done in {elapsed:.0f}s")
    print(f"  run root: {run_root}")
    handoff = out["handoff"]
    print(f"  crossover band       : {handoff.get('crossover_band', [])}")
    print(f"  2C attention layers  : {out['phase2c_arms']['attention_layers']}")
    print(f"  2C mlp layers        : {out['phase2c_arms']['mlp_layers']}")


def _attribute(args: argparse.Namespace) -> None:
    phase2a = Path(args.phase2a_run)
    phase2b = Path(args.phase2b_run)
    if not (phase2b / "analyze" / "summary.json").exists():
        sys.exit(f"ERROR: 2B summary not found under {phase2b}")
    run_id = args.run_id or (
        f"phase2c-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        if args.smoke else
        f"phase2c-gpu-bf16-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    started = time.time()
    print(f"[phase2c] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase2c(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=phase2a,
        phase2b_run=phase2b,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
        dtype=args.dtype,
    )
    elapsed = time.time() - started
    out = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    print(f"[phase2c] done in {elapsed:.0f}s")
    print(f"  run root: {run_root}")
    gate = out["gate_2c"]
    attention = gate["attention_arm"]
    print(f"  attention arm        : {attention['status']}", end="")
    if attention["status"] == "run":
        print(f"  top={attention['top_head']}  effect={attention['top_effect']:+.4f}  p={attention['sign_flip_p_adjusted']:.4g}  pass={attention['pass']}")
    else:
        print()
    mlp = gate["mlp_arm"]
    print(f"  mlp arm              : top={mlp['top_attribution']:+.3f}  control={mlp['control_mean']:+.3f}  sector_agreement={mlp['sector_agreement']:.2f}  p={mlp['sign_flip_p']:.4g}  pass={mlp['pass']}")
    print(f"  gate 2C pass         : {gate['pass']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sweep = sub.add_parser("sweep", help="2B entity-state layer sweep")
    p_sweep.add_argument("--model", required=True)
    p_sweep.add_argument("--phase2a-run", required=True)
    p_sweep.add_argument("--gate-run", default=None,
                         help="Rev 2 gate re-evaluation run root (uses gate_2a_rev2); default: Rev 1 gate in the 2A run")
    p_sweep.add_argument("--run-id", default=None)
    p_sweep.add_argument("--artifact-root", default="artifacts")
    p_sweep.add_argument("--smoke", action="store_true")
    p_sweep.add_argument("--device-map", default=None,
                         help="Multi-GPU sharded load map (e.g. qwen27b_two_gpu); default: single device")
    p_sweep.add_argument("--dtype", default=None,
                         help="load_model dtype: 'native' keeps the checkpoint's stored dtypes "
                              "(e.g. gpt-oss-20b packed MXFP4 experts); default: CUDA-bf16 / CPU-fp32")
    p_sweep.add_argument("--gate-override", default=None,
                         help="Proceed with 2B even when gate 2A did not pass; the reason is "
                              "recorded in the run pairs artifact (development use only)")
    p_sweep.add_argument("--family", choices=("v1", "v2"), default="v1",
                         help="Prompt family of the 2A run. v2 = within-company "
                              "pos/neg condition-flip directions (proposal-phase2-v2.md)")
    p_sweep.add_argument("--direction-min-gap", type=float, default=0.1,
                         help="v2: skip companies whose |M_pos - M_neg| is below this (nats)")
    p_sweep.set_defaults(func=_sweep)

    p_attr = sub.add_parser("attribute", help="2C component attribution")
    p_attr.add_argument("--model", required=True)
    p_attr.add_argument("--phase2a-run", required=True)
    p_attr.add_argument("--phase2b-run", required=True)
    p_attr.add_argument("--run-id", default=None)
    p_attr.add_argument("--artifact-root", default="artifacts")
    p_attr.add_argument("--smoke", action="store_true")
    p_attr.add_argument("--dtype", default=None,
                        help="load_model dtype: 'native' keeps the checkpoint's stored dtypes "
                             "(e.g. gpt-oss-20b packed MXFP4 experts); default: CUDA-bf16 / CPU-fp32")
    p_attr.set_defaults(func=_attribute)

    args = parser.parse_args()
    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")
    args.func(args)


if __name__ == "__main__":
    main()
