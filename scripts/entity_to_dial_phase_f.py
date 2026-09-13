"""Entity-to-Dial Phase F — dual-path additivity + directional push.

Runs the 4-stage pipeline (prepare → forward_f1 → forward_f2 → analyze)
defined in docs/entity-to-dial/details/proposal-phase-f.md (Rev 1): F1 dual-hook
combined patch (v₁ PCA transplant + dial channel transplant in one
forward, L15, 8 directions; gate F1 = additivity ratio ≥ 0.85) and F2
neutral-context directional push (anonymous prompt, ±α × base × {v₁,
dial footprint}; descriptive four-way verdict).

Usage
-----
Smoke preflight (1 direction + anonymous prompt; mechanism validation,
6 acceptance checks):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_f.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --phasee-run artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01 \\
        --smoke

Formal run (8 directions x 12 forwards + 14 push forwards = 110):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_f.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \\
        --phasee-run artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01 \\
        --run-id entity-to-dial-f-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.entity_to_dial.pipeline import run_phase_f
from llm_bias.entity_to_dial.template import (
    DEFAULT_PHASE2A_RUN,
    DEFAULT_PHASE2B_RUN,
    DEFAULT_PHASEE_RUN,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--phase2a-run", default=DEFAULT_PHASE2A_RUN)
    parser.add_argument("--phase2b-run", default=DEFAULT_PHASE2B_RUN)
    parser.add_argument("--phasee-run", default=DEFAULT_PHASEE_RUN)
    parser.add_argument("--run-id", default=None, help="Run identifier (required for formal runs)")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="Preflight (1 direction + anonymous prompt)")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")
    for label, value in (
        ("phase2a-run", args.phase2a_run),
        ("phase2b-run", args.phase2b_run),
        ("phasee-run", args.phasee_run),
    ):
        if not Path(value, "manifest.json").is_file():
            sys.exit(f"ERROR: {label} not found or missing manifest: {value}")
    if args.smoke:
        run_id = f"entity-to-dial-f-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        run_id = args.run_id
        if not run_id:
            sys.exit("ERROR: --run-id is required for formal runs")

    started = time.time()
    print(f"[entity-to-dial-f] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase_f(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=args.phase2a_run,
        phase2b_run=args.phase2b_run,
        phasee_run=args.phasee_run,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(f"[entity-to-dial-f] done in {time.time() - started:.0f}s")
    print(f"  run root: {run_root}")
    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_f"]
    print("── gate F ──")
    if gate.get("status") == "evaluated" or "f1" in gate:
        f1 = summary["gate_f1"]
        print(f"  F1 additivity ratio median : {f1['median_ratio']:+.4f} "
              f"(n_effective {f1['n_effective']}/8, threshold {f1['threshold']})  pass={f1['pass']}")
        for p in f1["per_direction"]:
            ratio = "n/a (excluded)" if p["additivity_ratio"] is None else f"{p['additivity_ratio']:+.3f}"
            print(f"    {p['direction']:10s} combined ΔM={p['combined_dm']:+.4f} "
                  f"v1={p['v1_dm']:+.4f} dial={p['dial_dm']:+.4f} "
                  f"residual={p['interaction_delta_m']:+.4f} ratio={ratio}")
        secondary = summary["f1_secondary"]
        print(f"  secondary: bottom→top {secondary['bottom_to_top']} / top→bottom {secondary['top_to_bottom']}")
        consistency = summary["f1_consistency"]
        print(f"  consistency vs e-01 (tol {consistency['tolerance']}): "
              + " ".join(f"{a}={v:+.2e}" for a, v in consistency.items() if a != "tolerance"))
        print(f"  F2 verdicts: v1={summary['f2_verdict']['v1']}  dial_fp={summary['f2_verdict']['dial_fp']}")
        print(f"  push_base={summary['push_base']:.4f}  m_anon={summary['m_anon']:+.4f} (band ok={summary['m_anon_band_ok']})")
        print(f"  gate_f pass                : {gate['pass']} (F1={gate['f1']})")
    else:
        print(f"  {gate.get('reason', 'not evaluated')}")
    if args.smoke:
        f1_count = len((Path(run_root) / "forward_f1" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
        f2_count = len((Path(run_root) / "forward_f2" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
        print(f"  [smoke] {f1_count} F1 records + {f2_count} F2 records; acceptance enforced in-pipeline:")
        print("    #1 no-ops bit-exact (incl. dual-hook)  #2 composition property (zero-side == single arm)")
        print("    #3 consistency vs e-01/2B ≤ 0.01  #4 PCA provenance  #5 anonymous identity + m_anon band")
        print("    #6 push margins finite, |ΔM| ≤ 5")
        print("[smoke] PASS")


if __name__ == "__main__":
    main()
