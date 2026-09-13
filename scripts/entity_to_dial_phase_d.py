"""Entity-to-Dial Phase D — instruction-span block sweep + signed attribution.

Runs the 4-stage pipeline (prepare → forward_d1 → forward_d2 → analyze)
defined in docs/entity-to-dial/proposal-phase-d.md (Rev 1): 8 directions
x 20 layers (L12–31) x 2 components (MLP block vs attention block)
instruction-span patch forwards with fp32 block arithmetic, self-source
no-op discipline, final-position sanity, gate D (MLP path existence +
cross-sector sign agreement), and the D2 instruction-position signed
attribution (descriptive, not gated).

Usage
-----
Smoke preflight (2 directions x 3 layers x 2 components + no-ops +
1-company D2 gradient check):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_d.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --smoke

Formal run (8 directions x 20 layers x 2 components + 16 backward passes):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_d.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --run-id entity-to-dial-d-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.entity_to_dial.pipeline import run_phase_d
from llm_bias.entity_to_dial.template import (
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
    parser.add_argument("--run-id", default=None, help="Run identifier (required for formal runs)")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="Preflight (2 directions x 3 layers)")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")
    for label, value in (
        ("phase2a-run", args.phase2a_run),
        ("phase2b-run", args.phase2b_run),
    ):
        if not Path(value, "manifest.json").is_file():
            sys.exit(f"ERROR: {label} not found or missing manifest: {value}")
    if args.smoke:
        run_id = f"entity-to-dial-d-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        run_id = args.run_id
        if not run_id:
            sys.exit("ERROR: --run-id is required for formal runs")

    started = time.time()
    print(f"[entity-to-dial-d] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase_d(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=args.phase2a_run,
        phase2b_run=args.phase2b_run,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(f"[entity-to-dial-d] done in {time.time() - started:.0f}s")
    print(f"  run root: {run_root}")
    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_d"]
    print("── gate D ──")
    if gate["status"] == "evaluated":
        d1 = gate["d1"]
        print(f"  D1 qualifying MLP layers     : {d1['qualifying_layers']}  pass={d1['pass']}")
        print(f"  strongest layer              : {gate['strongest_layer']}")
        if gate["d3"] is not None:
            sectors = ", ".join(f"{k}={v:+.4f}" for k, v in sorted(gate["d3"]["sector_means"].items()))
            print(f"  D3 sector means (strongest)  : {sectors}  pass={gate['d3']['pass']}")
        print(f"  gate_d pass                  : {gate['pass']}")
    else:
        print(f"  {gate['reason']}")
    sanity = summary["final_position_sanity"]
    for component in ("final_mlp", "final_attn"):
        entry = sanity[component]
        print(f"  final-position sanity {component}: mean T = {entry['mean_normalized_transfer']:+.4f}")
    print("── D2 descriptive (top channel per layer) ──")
    for layer_key in sorted(summary["d2_descriptive"], key=int):
        entry = summary["d2_descriptive"][layer_key]
        print(
            f"  L{layer_key}: top channel {entry['top_channel_idx']} "
            f"rho={entry['top_channel_rho']:+.3f} "
            f"sector {entry['top_channel_sector_agreement']}/{entry['n_sectors']} "
            f"max control |rho|={entry['max_control_rho']:.3f}"
        )
    if args.smoke:
        d1_records = (Path(run_root) / "forward_d1" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
        d2_metadata = json.loads(
            (Path(run_root) / "forward_d2" / "metadata.json").read_text(encoding="utf-8")
        )
        acceptance = d2_metadata["derivative_acceptance"]
        print(
            f"  [smoke] D2 derivative acceptance: structural_zero={acceptance['structural_zero_final_layer']} "
            f"nonzero_norm={acceptance['nonfinal_nonzero_norm']:.4f} "
            f"partition_rel_err={acceptance['partition_rel_err']:.3e} "
            f"(tol {acceptance['partition_tolerance']}) "
            f"sign_agreement(descriptive)={acceptance['sign_agreement_descriptive']:.3f} "
            f"pass={acceptance['pass']}"
        )
        print(f"  [smoke] {len(d1_records)} D1 records; no-op discipline verified")
        print("[smoke] PASS")


if __name__ == "__main__":
    main()
