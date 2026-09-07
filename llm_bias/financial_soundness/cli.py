"""Separate entry points for preparation, localization and causal validation."""
from __future__ import annotations

import argparse
from llm_bias.core.artifacts.io import write_json
from .prompts import build_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-soundness")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-prompts")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--companies", nargs="+", required=True)
    localize = commands.add_parser("run-localization")
    localize.add_argument("--prompts", required=True)
    localize.add_argument("--model", required=True)
    localize.add_argument("--run-id", required=True)
    localize.add_argument("--artifact-root", default="artifacts")
    localize.add_argument("--layers", nargs="+", type=int)
    localize.add_argument("--top-k", type=int, default=3)
    causal = commands.add_parser("run-causal-validation")
    causal.add_argument("--source-run", required=True)
    causal.add_argument("--model", required=True)
    causal.add_argument("--run-id", required=True)
    causal.add_argument("--artifact-root", default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-prompts":
        print(write_json(args.output, build_prompts(args.companies)))
    elif args.command == "run-localization":
        from .pipeline import run_localization
        print(run_localization(args.prompts, args.model, args.run_id, artifact_root=args.artifact_root,
                               layers=args.layers, top_k=args.top_k))
    elif args.command == "run-causal-validation":
        from .pipeline import run_causal_validation
        print(run_causal_validation(args.source_run, args.model, args.run_id, artifact_root=args.artifact_root))
    return 0
