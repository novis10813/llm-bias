"""CLI for the counterfactual residual activation patching experiment."""

from __future__ import annotations

import argparse

from llm_bias.core.model import DEFAULT_MODEL
from llm_bias.counterfactual_patching.experiment import (
    prepare_data,
    run_patch,
    summarize,
    visualize,
)

DEFAULT_ROOT = "artifacts/counterfactual_patching"


def _model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data")
    _model_argument(prepare)
    prepare.add_argument("--output", default=f"{DEFAULT_ROOT}/pairs.jsonl")
    prepare.add_argument("--max-pairs", type=int)

    run = commands.add_parser("run")
    _model_argument(run)
    run.add_argument("--pairs", default=f"{DEFAULT_ROOT}/pairs.jsonl")
    run.add_argument("--lens", required=True)
    run.add_argument("--output", default=f"{DEFAULT_ROOT}/patch_results.jsonl")
    run.add_argument("--max-pairs", type=int)

    summary = commands.add_parser("summarize")
    summary.add_argument("--input", default=f"{DEFAULT_ROOT}/patch_results.jsonl")
    summary.add_argument("--output-dir", default=DEFAULT_ROOT)

    visual = commands.add_parser("visualize")
    visual.add_argument("--input", default=f"{DEFAULT_ROOT}/patch_results.jsonl")
    visual.add_argument("--output-dir", default=DEFAULT_ROOT)

    serve = commands.add_parser("serve")
    _model_argument(serve)
    serve.add_argument(
        "--lens",
        help="defaults to artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt",
    )
    serve.add_argument("--pairs", default=f"{DEFAULT_ROOT}/pairs.jsonl")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8321)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        prepare_data(args.model, args.output, args.max_pairs)
    elif args.command == "run":
        run_patch(args.model, args.pairs, args.lens, args.output, args.max_pairs)
    elif args.command == "summarize":
        summarize(args.input, args.output_dir)
    elif args.command == "visualize":
        visualize(args.input, args.output_dir)
    elif args.command == "serve":
        import uvicorn

        from llm_bias.counterfactual_patching.visualization import build_app

        app = build_app(args.model, args.lens, args.pairs)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
