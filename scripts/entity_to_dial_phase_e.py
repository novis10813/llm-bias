"""Entity-to-Dial Phase E — dual-block joint discrimination + state-direction patch.

Runs the 4-stage pipeline (prepare → forward_e1 → forward_e2 → analyze)
defined in docs/entity-to-dial/details/proposal-phase-e.md (Rev 1): E1 dual-block
joint patch + full-swap reference arm (L12–18, gate E1 = joint ratio at
L15) and E2 state-direction patch at L15 (PCA sweep, dial channel
transplant gate E2b, residual footprint descriptive).

Usage
-----
Smoke preflight (2 directions x 2 layers + E2 k {1,8,full} + dial;
5 mechanism acceptance checks):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_e.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --smoke

Formal run (8 directions x 7 layers x 4 arms + E2 ~72 forwards):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_e.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --run-id entity-to-dial-e-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.entity_to_dial.pipeline import run_phase_e
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
    parser.add_argument("--smoke", action="store_true", help="Preflight (2 directions x 2 layers + E2 subset)")
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
        run_id = f"entity-to-dial-e-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        run_id = args.run_id
        if not run_id:
            sys.exit("ERROR: --run-id is required for formal runs")

    started = time.time()
    print(f"[entity-to-dial-e] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase_e(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=args.phase2a_run,
        phase2b_run=args.phase2b_run,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(f"[entity-to-dial-e] done in {time.time() - started:.0f}s")
    print(f"  run root: {run_root}")
    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_e"]
    print("── gate E ──")
    if gate.get("status") == "evaluated" or "e1" in gate:
        e1 = summary["gate_e1"]
        print(f"  E1 L15 joint ratio median  : {e1['median_ratio']:+.4f} "
              f"(n_effective {e1['n_effective']}/8, threshold {e1['threshold']})  pass={e1['pass']}")
        for p in e1["per_direction"]:
            ratio = "n/a (excluded)" if p["ratio"] is None else f"{p['ratio']:+.3f}"
            print(f"    {p['direction']:10s} joint ΔM={p['joint_dm']:+.4f} full ΔM={p['full_dm']:+.4f} ratio={ratio}")
        e2b = summary["gate_e2b"]
        print(f"  E2b dial ratio median      : {e2b['median_ratio']:+.4f} "
              f"(threshold {e2b['threshold']}, falsifier < {e2b['falsifier_max']})  "
              f"pass={e2b['pass']} falsified={e2b['falsified']}")
        print(f"  gate_e pass                : {gate['pass']} (E1={gate['e1']}, E2b={gate['e2b']})")
    else:
        print(f"  {gate.get('reason', 'not evaluated')}")
    print("── E1 curves (joint vs full, per layer) ──")
    for layer_key in sorted(summary["e1_curves"] or {}, key=int):
        entry = summary["e1_curves"][layer_key]
        incr = f"  incremental ref {entry['incremental_dm_ref']:+.4f}" if "incremental_dm_ref" in entry else ""
        ratio = "n/a" if entry["median_ratio"] is None else f"{entry['median_ratio']:+.3f}"
        print(f"  L{layer_key}: joint ratio median={ratio} mean joint ΔM={entry['mean_joint_dm']:+.4f} "
              f"mean full ΔM={entry['mean_full_dm']:+.4f}{incr}")
    print("── E2 descriptive ──")
    for k in sorted((summary["e2a_curve"] or {}), key=lambda x: (x == "full", int(x) if x != "full" else 99)):
        entry = summary["e2a_curve"][k]
        ratio = "n/a" if entry["median_ratio"] is None else f"{entry['median_ratio']:+.3f}"
        target = f"  (H_E2a target {entry['target']})" if "target" in entry else ""
        print(f"  k={k:4s} effect ratio median={ratio}{target}")
    if summary.get("e2_footprint") is not None:
        foot = summary["e2_footprint"]
        ratio = "n/a" if foot["median_ratio"] is None else f"{foot['median_ratio']:+.3f}"
        print(f"  footprint (dial residual direction, descriptive): effect ratio median={ratio}")
    print(f"  PCA singular values (top 8): {[f'{v:.3f}' for v in summary['pca_singular_values'][:8]]}")
    if args.smoke:
        e1_count = len((Path(run_root) / "forward_e1" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
        e2_count = len((Path(run_root) / "forward_e2" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
        print(f"  [smoke] {e1_count} E1 records + {e2_count} E2 records; acceptance enforced in-pipeline:")
        print("    #1 no-ops bit-exact 0.0  #2 full arm vs 2B archive (per direction, ±0.05 nats)")
        print("    #3 dial capture complete  #4 k=full == E1 L15 full arm  #5 SVD deterministic")
        print("[smoke] PASS")


if __name__ == "__main__":
    main()
