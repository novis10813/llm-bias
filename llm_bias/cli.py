"""Command-line entry points for the entity control experiment."""

from __future__ import annotations

import argparse

from llm_bias.analysis import fit_lens, prepare_data, run_patch, summarize, visualize
from llm_bias.model import DEFAULT_MODEL
from llm_bias.model import QWEN35_MODEL
from llm_bias.prompt_outputs import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STRIDE1_LENS,
    analyze_prompt_outputs,
)
from llm_bias.generated_attribution import analyze_generated_attribution


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
    fit.add_argument("--dim-batch", type=int, default=16)
    fit.add_argument("--max-seq-len", type=int, default=128)
    fit.add_argument("--skip-first", type=int, default=0)

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

    prompt_outputs = commands.add_parser(
        "analyze-prompt-outputs",
        help="average J-space next-token distributions from prompt CSV columns",
    )
    _common(prompt_outputs)
    prompt_outputs.set_defaults(model=QWEN35_MODEL)
    prompt_outputs.add_argument("--input", default=DEFAULT_INPUT)
    prompt_outputs.add_argument("--lens", default=DEFAULT_STRIDE1_LENS)
    prompt_outputs.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    prompt_outputs.add_argument("--top-k", type=int, default=15)
    prompt_outputs.add_argument("--batch-size", type=int, default=32)
    prompt_outputs.add_argument("--max-seq-len", type=int, default=256)
    prompt_outputs.add_argument("--max-rows", type=int)
    prompt_outputs.add_argument(
        "--prompt-column",
        action="append",
        dest="prompt_columns",
        help="analyze only this prompt column (repeatable)",
    )

    generated = commands.add_parser(
        "analyze-generated-attribution",
        help="sample generated output tokens and attribute them to prompt tokens",
    )
    generated.set_defaults(model=QWEN35_MODEL)
    generated.add_argument("--model", default=QWEN35_MODEL)
    generated.add_argument("--input", default=DEFAULT_INPUT)
    generated.add_argument("--output-dir", default="artifacts/qwen_generated_attribution")
    generated.add_argument("--sample-per-condition", type=int, default=32)
    generated.add_argument("--max-new-tokens", type=int, default=16)
    generated.add_argument("--input-top-k", type=int, default=15)
    generated.add_argument("--max-seq-len", type=int, default=256)
    generated.add_argument("--prompt-column", action="append", dest="prompt_columns")
    prompt_outputs.add_argument(
        "--no-save-prompt-topk",
        action="store_false",
        dest="save_prompt_topk",
        help="skip the per-prompt/per-layer compact JSONL",
    )
    prompt_outputs.add_argument(
        "--no-save-prompt-uncertainty",
        action="store_false",
        dest="save_prompt_uncertainty",
        help="skip per-prompt/per-layer uncertainty JSONL",
    )
    prompt_outputs.add_argument(
        "--no-input-attribution",
        action="store_false",
        dest="compute_input_attribution",
        help="skip output-token attribution to input positions",
    )
    prompt_outputs.add_argument("--attribution-batch-size", type=int, default=8)
    prompt_outputs.add_argument("--input-top-k", type=int, default=15)
    prompt_outputs.add_argument(
        "--attribution-output-top-k",
        type=int,
        help="number of output top-k tokens to attribute (default: all --top-k)",
    )
    prompt_outputs.add_argument(
        "--attribution-max-rows",
        type=int,
        help="deterministically spread attribution over at most this many dates",
    )
    prompt_outputs.add_argument(
        "--raw-prompt",
        action="store_false",
        dest="use_chat_template",
        help="do not wrap each CSV prompt as a user message",
    )
    prompt_outputs.add_argument(
        "--enable-thinking",
        action="store_true",
        help="leave Qwen's thinking mode enabled (default: disabled)",
    )

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
        fit_lens(
            args.model,
            args.output,
            args.calibration_prompts,
            args.layer_stride,
            args.dim_batch,
            args.max_seq_len,
            args.skip_first,
        )
    elif args.command == "run-patch":
        run_patch(args.model, args.pairs, args.lens, args.output, args.max_pairs)
    elif args.command == "summarize":
        summarize(args.input, args.output_dir)
    elif args.command == "visualize":
        visualize(args.input, args.output_dir)
    elif args.command == "analyze-prompt-outputs":
        analyze_prompt_outputs(
            input_path=args.input,
            model_name=args.model,
            lens_path=args.lens,
            output_dir=args.output_dir,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_rows=args.max_rows,
            prompt_columns=args.prompt_columns,
            save_prompt_topk=args.save_prompt_topk,
            save_prompt_uncertainty=args.save_prompt_uncertainty,
            compute_input_attribution=args.compute_input_attribution,
            attribution_batch_size=args.attribution_batch_size,
            input_top_k=args.input_top_k,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
            attribution_output_top_k=args.attribution_output_top_k,
            attribution_max_rows=args.attribution_max_rows,
        )
    elif args.command == "analyze-generated-attribution":
        analyze_generated_attribution(
            input_path=args.input,
            model_name=args.model,
            output_dir=args.output_dir,
            sample_per_condition=args.sample_per_condition,
            max_new_tokens=args.max_new_tokens,
            input_top_k=args.input_top_k,
            max_seq_len=args.max_seq_len,
            prompt_columns=args.prompt_columns,
        )
    elif args.command == "serve-viz":
        from llm_bias.visualization import build_app

        import uvicorn

        app = build_app(args.model, args.lens, args.pairs)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
