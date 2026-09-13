"""Entity-to-Dial Phase C — dial-path probe.

Runs the 3-stage pipeline (prepare → forward → analyze) defined in
docs/entity-to-dial/details/proposal-phase-abc.md §4.4 (Rev 1): 16 named + 16 anonymous
clean forwards with dual-position dial readout, 16 dial push forwards
(mlp_addition on L15/n8490 with per-company measured δ), the δ=0 no-op
check, gate C (direction + ≥25% of the named-vs-anonymous gap), the
descriptive Step C1 rho, and the Step C3 unexplained gap.

Usage
-----
Smoke preflight (2 tickers: 4 clean + 2 pushed + 1 no-op):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_c.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --smoke

Formal run (16 tickers: 32 clean + 16 pushed + 1 no-op):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/entity_to_dial_phase_c.py \\
        --model .cache/models/qwen3.5-4b \\
        --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \\
        --run-id entity-to-dial-c-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_bias.entity_to_dial.pipeline import run_phase_c
from llm_bias.entity_to_dial.template import DEFAULT_PHASE2A_RUN


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--phase2a-run", default=DEFAULT_PHASE2A_RUN)
    parser.add_argument("--run-id", default=None, help="Run identifier (required for formal runs)")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="Preflight (2 tickers)")
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")
    if not Path(args.phase2a_run, "manifest.json").is_file():
        sys.exit(f"ERROR: phase2a-run not found or missing manifest: {args.phase2a_run}")
    if args.smoke:
        run_id = f"entity-to-dial-c-smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        run_id = args.run_id
        if not run_id:
            sys.exit("ERROR: --run-id is required for formal runs")

    started = time.time()
    print(f"[entity-to-dial-c] run_id={run_id} smoke={args.smoke}", flush=True)
    run_root = run_phase_c(
        model_path=args.model,
        run_id=run_id,
        phase2a_run=args.phase2a_run,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(f"[entity-to-dial-c] done in {time.time() - started:.0f}s")
    print(f"  run root: {run_root}")
    summary = json.loads((Path(run_root) / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_c"]
    c1 = summary["c1_descriptive"]
    print("── C1 descriptive ──")
    for key, label in (("rho_entity_position", "dial Δ (entity pos) ρ"), ("rho_final_position", "dial Δ (final pos) ρ")):
        value = c1.get(key)
        print(f"  {label:22s}: {value:+.3f}" if value is not None else f"  {label:22s}: n/a")
    if c1.get("rho_warning_below_0.3"):
        print("  warning: entity-position ρ < 0.3")
    cross = summary["cross_check_2a"]
    print(f"  2A cross-check max |Δ|      : {cross['max_abs_diff']:.4f} nats (warn > {cross['warning_threshold']})")
    print("── gate C ──")
    if gate["status"] == "evaluated":
        print(f"  C1 mean gap / ΔM_dial CI : {gate['c1']}")
        print(f"  C2 mean|ΔM_dial|/mean|gap|: {gate['c2']['ratio']:.4f} (threshold {0.25})  pass={gate['c2']['pass']}")
        print(f"  gate_c pass              : {gate['pass']}")
    else:
        print(f"  {gate['reason']}")
    if args.smoke:
        results = (Path(run_root) / "forward" / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        print(f"  [smoke] {len(results)} forward records; δ=0 no-op verified")
        print("[smoke] PASS")


if __name__ == "__main__":
    main()
