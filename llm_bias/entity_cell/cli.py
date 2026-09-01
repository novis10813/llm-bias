"""CLI for preparation, E1 localization, and E2 attribution controls."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate and materialize tokenizer-only inputs")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--baseline", type=Path, required=True)
    prepare.add_argument("--baseline-identity", required=True)
    prepare.add_argument("--model", required=True, help="model identity for provenance; no model is loaded")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    prepare.add_argument("--dataset", default="entity-cell-localization")
    prepare.add_argument("--split", choices=("discovery", "calibration", "test"), default="discovery")
    prepare.add_argument("--sector", default="Technology")
    prepare.add_argument("--baseline-count", type=int, default=399)
    prepare.add_argument("--tokenizer", default=None, help="tokenizer identity/path; defaults to --model")
    run = commands.add_parser("run", help="run E1 localization/amnesia or E2 attribution from prepared inputs")
    run.add_argument("--prepared-dir", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    run.add_argument("--stage", action="append", choices=("e1-baseline", "e1-localization", "e1-amnesia", "e2-attribution", "e2-readout", "e2-patching", "analyze"), dest="stages")
    run.add_argument("--e2-layers", nargs="+", type=int, default=None, help="E2 full-attention layers; defaults to L3/L7/L11/L15/L19/L23/L27/L31")
    run.add_argument("--max-tickers", type=int, default=None, help="one-ticker smoke cap when set to 1")
    run.add_argument("--lens-path", type=Path, default=None, help="explicit validated canonical lens path")
    run.add_argument("--expected-lens-sha256", default=None)
    run.add_argument("--expected-model-revision", default=None)
    run.add_argument("--expected-tokenizer-identity", default=None)
    analyze = commands.add_parser("analyze", help="analyze completed compact E1 or E2 outputs")
    analyze.add_argument("--run-root", type=Path, required=True)
    analyze.add_argument("--experiment", choices=("e1", "e2"), default="e1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        from llm_bias.core.model import load_tokenizer
        from llm_bias.entity_cell.preparation import prepare_artifacts
        root = prepare_artifacts(
            input_path=args.input, split_manifest=args.split_manifest, baseline_source=args.baseline,
            baseline_identity=args.baseline_identity, tokenizer=load_tokenizer(args.tokenizer or args.model),
            model=args.model, run_id=args.run_id, artifact_root=args.artifact_root, dataset=args.dataset,
            split=args.split, sector=args.sector, baseline_expected_count=args.baseline_count,
        )
    elif args.command == "run":
        stages = tuple(args.stages) if args.stages else ("e1-baseline", "e1-localization", "e1-amnesia", "analyze")
        if any(stage.startswith("e2-") for stage in stages):
            from llm_bias.entity_cell.e2 import run_e2
            root = run_e2(prepared_dir=args.prepared_dir, model_name=args.model, run_id=args.run_id, artifact_root=args.artifact_root, stages=stages, layers=args.e2_layers or (3, 7, 11, 15, 19, 23, 27, 31), max_tickers=args.max_tickers, lens_path=args.lens_path, expected_lens_sha256=args.expected_lens_sha256, expected_model_revision=args.expected_model_revision, expected_tokenizer_identity=args.expected_tokenizer_identity)
        else:
            from llm_bias.entity_cell.pipeline import run_e1
            root = run_e1(prepared_dir=args.prepared_dir, model_name=args.model, run_id=args.run_id, artifact_root=args.artifact_root, stages=stages, max_tickers=args.max_tickers)
    elif args.command == "analyze":
        if args.experiment == "e2":
            from llm_bias.entity_cell.e2 import analyze_e2
            root = analyze_e2(args.run_root)
        else:
            from llm_bias.entity_cell.pipeline import analyze_e1
            root = analyze_e1(args.run_root)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(root)


if __name__ == "__main__":
    main()
