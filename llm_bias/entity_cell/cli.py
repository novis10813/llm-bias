"""Preparation-only CLI for the entity-cell localization protocol."""
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "prepare":
        raise ValueError(f"unsupported command: {args.command}")
    from llm_bias.core.model import load_tokenizer
    from llm_bias.entity_cell.preparation import prepare_artifacts

    tokenizer = load_tokenizer(args.tokenizer or args.model)
    root = prepare_artifacts(
        input_path=args.input,
        split_manifest=args.split_manifest,
        baseline_source=args.baseline,
        baseline_identity=args.baseline_identity,
        tokenizer=tokenizer,
        model=args.model,
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        dataset=args.dataset,
        split=args.split,
        sector=args.sector,
        baseline_expected_count=args.baseline_count,
    )
    print(root)


if __name__ == "__main__":
    main()
