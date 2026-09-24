"""Balanced Evidence Gap — Phase 2A cross-entity probe operator.

Runs the 3-stage pipeline (prepare → forward → analyze) defined in
docs/balanced-evidence-gap/details/proposal-phase2.md §4.1: 16 tickers × 2 reverse
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
    parser.add_argument("--no-dial", action="store_true",
                        help="Skip the Qwen3.5-4B L15/n8490 H4 dial readout (cross-model runs)")
    parser.add_argument("--phase1-summary", default=None,
                        help="Phase 1 named-margin summary for the Spearman gate criterion. "
                             "Default: the protocol Qwen3.5-4B reference. Pass 'none' for "
                             "cross-model runs to skip the reference (gate covers IQR, framing, schema only)")
    parser.add_argument("--device-map", default=None,
                        help="Multi-GPU sharded load map (e.g. qwen27b_two_gpu); default: single device")
    parser.add_argument("--dtype", default=None,
                        help="load_model dtype: 'native' keeps the checkpoint's stored dtypes "
                             "(e.g. gpt-oss-20b packed MXFP4 experts); default: CUDA-bf16 / CPU-fp32")
    parser.add_argument("--companies-file", default=None,
                        help="JSON with a 'companies' list ({ticker, name}, e.g. "
                             "data/baseline/investment-dial/exploratory-v1.json); uses all its "
                             "tickers instead of the frozen 16-ticker v1 universe (v2 development runs)")
    parser.add_argument("--family", choices=("v1", "v2"), default="v1",
                        help="Prompt family. v2 = two-sentence evidence conditions (pos/neg), "
                             "within-company condition-flip design (proposal-phase2-v2.md); "
                             "force-disables the dial readout and Phase 1 Spearman reference")
    args = parser.parse_args()

    tickers = None
    if args.companies_file is not None:
        import json as _json
        companies = _json.loads(Path(args.companies_file).read_text(encoding="utf-8"))["companies"]
        tickers = [c["ticker"] for c in companies]
        if len(tickers) < 8:
            sys.exit(f"ERROR: need >=8 tickers for the gate, got {len(tickers)}")
        print(f"[phase2a] custom universe: {len(tickers)} tickers from {args.companies_file}", flush=True)

    if not Path(args.model).is_dir():
        sys.exit(f"ERROR: model path not found: {args.model}")

    phase1_summary = None
    if args.phase1_summary is not None and args.phase1_summary.lower() != "none":
        phase1_summary = Path(args.phase1_summary)

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
        no_dial=args.no_dial,
        phase1_summary=phase1_summary,
        device_map=args.device_map,
        dtype=args.dtype,
        tickers=tickers,
        family=args.family,
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
        if crit["value"] is None:
            print(f"  {name:22s} value=skipped  threshold={crit['threshold']}  pass=None")
        else:
            print(f"  {name:22s} value={crit['value']:+.4f}  threshold={crit['threshold']}  pass={crit['pass']}")
    print(f"  gate_2a pass           : {gate['pass']}")
    print(f"  phase2b authorized     : {gate['phase2b_authorized']}")
    h4 = summary.get("h4_dial")
    print("── H4 (descriptive) ──")
    if h4 is None or h4.get("skipped"):
        reason = h4.get("skipped") if h4 else "not_run (v2 family)"
        print(f"  skipped ({reason})")
    else:
        print(f"  dial L{h4['coordinate'][0]}/n{h4['coordinate'][1]}")
        print(f"  entity-position ρ vs pure entity margin: {h4['entity_position_pearson']:+.3f}")
        print(f"  final-position ρ   vs pure entity margin: {h4['final_position_pearson']:+.3f}")


if __name__ == "__main__":
    main()
