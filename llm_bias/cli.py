"""Command-line entry points for the entity control experiment."""

from __future__ import annotations

import argparse

from llm_bias.analysis import fit_lens, prepare_data, run_patch, summarize, visualize
from llm_bias.model import DEFAULT_MODEL


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data")
    _common(prepare)
    prepare.add_argument("--output", default="artifacts/entity_control/pairs.jsonl")
    prepare.add_argument("--max-pairs", type=int)

    fit = commands.add_parser("fit-lens")
    _common(fit)
    fit.add_argument("--output", default="artifacts/entity_control/jacobian_lens.pt")
    fit.add_argument("--calibration-prompts", type=int, default=16)
    fit.add_argument("--layer-stride", type=int, default=2)

    patch = commands.add_parser("run-patch")
    _common(patch)
    patch.add_argument("--pairs", default="artifacts/entity_control/pairs.jsonl")
    patch.add_argument("--lens", default="artifacts/entity_control/jacobian_lens.pt")
    patch.add_argument("--output", default="artifacts/entity_control/patch_results.jsonl")
    patch.add_argument("--max-pairs", type=int)

    summary = commands.add_parser("summarize")
    summary.add_argument("--input", default="artifacts/entity_control/patch_results.jsonl")
    summary.add_argument("--output-dir", default="artifacts/entity_control")

    visual = commands.add_parser("visualize")
    visual.add_argument("--input", default="artifacts/entity_control/patch_results_full.jsonl")
    visual.add_argument("--output-dir", default="artifacts/entity_control")

    serve = commands.add_parser("serve-viz")
    _common(serve)
    serve.add_argument("--lens", default="artifacts/entity_control/jacobian_lens.pt")
    serve.add_argument("--pairs", default="artifacts/entity_control/pairs.jsonl")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8321)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        prepare_data(args.model, args.output, args.max_pairs)
    elif args.command == "fit-lens":
        fit_lens(args.model, args.output, args.calibration_prompts, args.layer_stride)
    elif args.command == "run-patch":
        run_patch(args.model, args.pairs, args.lens, args.output, args.max_pairs)
    elif args.command == "summarize":
        summarize(args.input, args.output_dir)
    elif args.command == "visualize":
        visualize(args.input, args.output_dir)
    elif args.command == "serve-viz":
        from llm_bias.visualization import build_app

        import uvicorn

        app = build_app(args.model, args.lens, args.pairs)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
