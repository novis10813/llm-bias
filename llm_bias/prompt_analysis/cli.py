"""CLI for prompt representation readout, attribution, and validation."""

from __future__ import annotations

import argparse

from llm_bias.prompt_analysis.attribution import (
    DEFAULT_OUTPUT_DIR as DEFAULT_ATTRIBUTION_OUTPUT,
    analyze_generated_attribution,
)
from llm_bias.prompt_analysis.readout import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR as DEFAULT_READOUT_OUTPUT,
    analyze_prompt_outputs,
)
from llm_bias.prompt_analysis.validation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_VALIDATION_OUTPUT,
    evaluate_semantic_scope,
)
from llm_bias.prompt_analysis.visualization import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_INPUT as DEFAULT_PRICE_INPUT,
    DEFAULT_OUTPUT_DIR as DEFAULT_VISUALIZATION_OUTPUT,
    DEFAULT_PRICE_DISTRIBUTION_OUTPUT,
    DEFAULT_UNCERTAINTY_DISTRIBUTION_OUTPUT,
    uncertainty_paths_from_root,
    visualize_price_distributions,
    visualize_uncertainty_distributions,
    visualize_prompt_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    readout = commands.add_parser(
        "readout", help="compute per-layer next-token readouts for prompt columns"
    )
    readout.add_argument("--model", required=True)
    readout.add_argument("--input", default=DEFAULT_INPUT)
    readout.add_argument("--lens", required=True)
    readout.add_argument("--output-dir", default=DEFAULT_READOUT_OUTPUT)
    readout.add_argument("--top-k", type=int, default=15)
    readout.add_argument("--batch-size", type=int, default=32)
    readout.add_argument("--max-seq-len", type=int, default=256)
    readout.add_argument("--max-rows", type=int)
    readout.add_argument("--prompt-column", action="append", dest="prompt_columns")
    readout.add_argument(
        "--no-save-prompt-topk",
        action="store_false",
        dest="save_prompt_topk",
    )
    readout.add_argument(
        "--no-save-prompt-uncertainty",
        action="store_false",
        dest="save_prompt_uncertainty",
    )
    readout.add_argument(
        "--no-input-attribution",
        action="store_false",
        dest="compute_input_attribution",
    )
    readout.add_argument("--attribution-batch-size", type=int, default=8)
    readout.add_argument("--input-top-k", type=int, default=15)
    readout.add_argument("--attribution-output-top-k", type=int)
    readout.add_argument("--attribution-max-rows", type=int)
    readout.add_argument(
        "--raw-prompt",
        action="store_false",
        dest="use_chat_template",
    )
    readout.add_argument("--enable-thinking", action="store_true")
    readout.add_argument("--dataset-format", choices=("auto", "legacy-wide", "return-pairs"), default="auto")

    attribute = commands.add_parser(
        "attribute", help="generate output tokens and attribute them to prompt tokens"
    )
    attribute.add_argument("--model", required=True)
    attribute.add_argument("--input", default=DEFAULT_INPUT)
    attribute.add_argument("--output-dir", default=DEFAULT_ATTRIBUTION_OUTPUT)
    attribute.add_argument("--sample-per-condition", type=int, default=32)
    attribute.add_argument("--max-new-tokens", type=int, default=64)
    attribute.add_argument("--input-top-k", type=int)
    attribute.add_argument("--max-seq-len", type=int, default=256)
    attribute.add_argument("--prompt-column", action="append", dest="prompt_columns")
    attribute.add_argument("--date", action="append", dest="dates")
    attribute.add_argument("--runs", type=int, default=1)
    attribute.add_argument("--temperature", type=float, default=0.0)
    attribute.add_argument("--seed", type=int)
    attribute.add_argument("--top-p", type=float, default=1.0)
    attribute.add_argument("--top-k", type=int, default=0)
    attribute.add_argument("--dataset-format", choices=("auto", "legacy-wide", "return-pairs"), default="auto")

    validate = commands.add_parser(
        "validate-attribution", help="evaluate Semantic Scope with input ablations"
    )
    validate.add_argument("--attribution", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--output-dir", default=DEFAULT_VALIDATION_OUTPUT)
    validate.add_argument("--seed", type=int, default=0)

    visual = commands.add_parser(
        "visualize", help="build uncertainty plots and an attribution dashboard"
    )
    visual.add_argument("--uncertainty-root", default=DEFAULT_ARTIFACT_ROOT)
    visual.add_argument("--attribution", required=True)
    visual.add_argument("--validation")
    visual.add_argument("--prices", default=DEFAULT_PRICE_INPUT)
    visual.add_argument("--output-dir", default=DEFAULT_VISUALIZATION_OUTPUT)
    visual.add_argument("--input-top-k", type=int, default=15)
    visual.add_argument("--tokenizer", required=True)
    visual.add_argument("--max-seq-len", type=int, default=256)

    price_distributions = commands.add_parser(
        "plot-price-distributions",
        help="plot actual closes, sampled price distributions, and errors",
    )
    price_distributions.add_argument("--sampling-root", required=True)
    price_distributions.add_argument("--prices", default=DEFAULT_PRICE_INPUT)
    price_distributions.add_argument(
        "--output-dir", default=DEFAULT_PRICE_DISTRIBUTION_OUTPUT
    )

    uncertainty_distributions = commands.add_parser(
        "plot-uncertainty-distributions",
        help="plot final-layer entropy and effective-temperature distributions",
    )
    uncertainty_distributions.add_argument("--uncertainty-root", required=True)
    uncertainty_distributions.add_argument(
        "--output-dir", default=DEFAULT_UNCERTAINTY_DISTRIBUTION_OUTPUT
    )

    serve = commands.add_parser(
        "serve", help="serve an interactive Jacobian-lens prompt explorer"
    )
    serve.add_argument("--model", required=True)
    serve.add_argument(
        "--lens",
        help="defaults to artifacts/lenses/<model>/jacobian_lens.pt",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8322)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "readout":
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
            dataset_format=args.dataset_format,
        )
    elif args.command == "attribute":
        analyze_generated_attribution(
            input_path=args.input,
            model_name=args.model,
            output_dir=args.output_dir,
            sample_per_condition=args.sample_per_condition,
            max_new_tokens=args.max_new_tokens,
            input_top_k=args.input_top_k,
            max_seq_len=args.max_seq_len,
            prompt_columns=args.prompt_columns,
            dates=args.dates,
            runs=args.runs,
            temperature=args.temperature,
            seed=args.seed,
            top_p=args.top_p,
            top_k=args.top_k,
            dataset_format=args.dataset_format,
        )
    elif args.command == "validate-attribution":
        evaluate_semantic_scope(
            attribution_path=args.attribution,
            model_name=args.model,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.command == "visualize":
        visualize_prompt_results(
            uncertainty_paths=uncertainty_paths_from_root(args.uncertainty_root),
            attribution_path=args.attribution,
            validation_path=args.validation,
            prices_path=args.prices,
            output_dir=args.output_dir,
            input_top_k=args.input_top_k,
            tokenizer_path=args.tokenizer,
            max_seq_len=args.max_seq_len,
        )
    elif args.command == "plot-price-distributions":
        visualize_price_distributions(
            sampling_root=args.sampling_root,
            prices_path=args.prices,
            output_dir=args.output_dir,
        )
    elif args.command == "plot-uncertainty-distributions":
        visualize_uncertainty_distributions(
            uncertainty_paths=uncertainty_paths_from_root(args.uncertainty_root),
            output_dir=args.output_dir,
        )
    elif args.command == "serve":
        import uvicorn

        from llm_bias.prompt_analysis.interactive import build_app

        app = build_app(args.model, args.lens)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
