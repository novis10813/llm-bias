"""CLI for the single-sector header-span sensitivity pilot."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="prepare, score, and analyze header identity conditions")
    run.add_argument(
        "--input",
        type=Path,
        default=Path("data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv"),
    )
    run.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/qwen3.5-4b/jspace-intervention/splits.json"),
    )
    run.add_argument("--model", default=".cache/models/qwen3.5-4b")
    run.add_argument("--run-id", required=True)
    run.add_argument("--artifact-root", default="artifacts")
    run.add_argument("--dataset", default="technology-header-span-sensitivity")
    run.add_argument("--sector", default="Technology")
    run.add_argument(
        "--split", choices=("discovery", "calibration", "test"), default="discovery"
    )
    run.add_argument("--prompt-column", action="append", dest="prompt_columns")
    run.add_argument("--max-records", type=int, default=None)
    run.add_argument("--max-seq-len", type=int, default=1024)
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--device-map",
        choices=("qwen27b_two_gpu", "qwen27b_two_gpu_24", "qwen27b_two_gpu_16"),
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from llm_bias.span_sensitivity.pipeline import run_pipeline

    root = run_pipeline(
        input_path=args.input,
        split_manifest=args.split_manifest,
        model_name=args.model,
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        dataset=args.dataset,
        sector=args.sector,
        split=args.split,
        prompt_columns=set(args.prompt_columns) if args.prompt_columns else None,
        max_records=args.max_records,
        max_seq_len=args.max_seq_len,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        device_map=args.device_map,
    )
    print(root)


if __name__ == "__main__":
    main()
