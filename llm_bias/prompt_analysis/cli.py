"""CLI for prompt representation readout, attribution, and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_bias.prompt_analysis.readout import DEFAULT_INPUT, analyze_prompt_outputs
from llm_bias.prompt_analysis.input_inspection import inspect_input_to_json
from llm_bias.prompt_analysis.return_evaluation import evaluate_return_predictions
from llm_bias.prompt_analysis.validation import evaluate_semantic_scope
from llm_bias.prompt_analysis.visualization import (
    DEFAULT_INPUT as DEFAULT_PRICE_INPUT,
    uncertainty_paths_from_root,
    visualize_price_distributions,
    visualize_uncertainty_distributions,
    visualize_prompt_results,
)
from llm_bias.prompt_analysis.return_visualization import visualize_return_predictions


def _missing_stage_api(stage: str, error: Exception) -> RuntimeError:
    message = RuntimeError(
        f"prompt-analysis {stage} stage API is unavailable; "
        "install or merge the generation/backward implementation"
    )
    message.__cause__ = error
    return message


def _run_generation_stage(args: argparse.Namespace) -> Path:
    try:
        from llm_bias.prompt_analysis.generation import generate_prompt_outputs
    except ImportError as error:
        raise _missing_stage_api("generation", error) from error
    return generate_prompt_outputs(
        input_path=args.input,
        model_name=args.model,
        output_path=args.output,
        sample_per_condition=(
            None
            if args.full_generation or args.sample_per_condition == 0
            else args.sample_per_condition
        ),
        full_generation=args.full_generation or args.sample_per_condition == 0,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        top_k=args.top_k,
        runs=args.runs,
        max_seq_len=args.max_seq_len,
        prompt_columns=args.prompt_columns,
        dates=args.dates,
        dataset_format=args.dataset_format,
    )


def _run_generated_attribution_stage(args: argparse.Namespace) -> Path:
    try:
        from llm_bias.prompt_analysis.generated_attribution import (
            attribute_generated_outputs,
        )
    except ImportError as error:
        raise _missing_stage_api("generated attribution", error) from error
    forward_artifact = Path(args.forward_artifact)
    if not forward_artifact.is_file():
        raise FileNotFoundError(
            f"forward artifact does not exist: {forward_artifact}"
        )
    return attribute_generated_outputs(
        forward_artifact=args.forward_artifact,
        model_name=args.model,
        output_path=Path(args.output),
        input_top_k=args.input_top_k,
        max_seq_len=args.max_seq_len,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect-input", help="validate prompt CSV compatibility without model inference"
    )
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--output", help="write the machine-readable JSON report here")

    readout = commands.add_parser(
        "readout", help="compute per-layer next-token readouts for prompt columns"
    )
    readout.add_argument("--model", required=True)
    readout.add_argument("--input", default=DEFAULT_INPUT)
    readout.add_argument("--lens", required=True)
    readout.add_argument("--output-dir", required=True)
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
        "--save-hidden",
        action="store_true",
        dest="save_prompt_hidden",
        help="Save final-layer normalized_hidden vectors (float16) to prompt_output_hidden.jsonl for cosine similarity analysis.",
    )
    readout.add_argument(
        "--raw-prompt",
        action="store_false",
        dest="use_chat_template",
    )
    readout.add_argument("--enable-thinking", action="store_true")
    readout.add_argument("--dataset-format", choices=("auto", "legacy-wide", "return-pairs"), default="auto")

    generate = commands.add_parser(
        "generate", help="generate and persist complete forward outputs"
    )
    generate.add_argument("--model", required=True)
    generate.add_argument("--input", default=DEFAULT_INPUT)
    generate.add_argument("--output", required=True)
    generate.add_argument("--sample-per-condition", type=int, default=32)
    generate.add_argument(
        "--full-generation",
        action="store_true",
        help="process every return-pair instead of the legacy sample limit",
    )
    generate.add_argument("--runs", type=int, default=1)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--max-seq-len", type=int, default=256)
    generate.add_argument("--prompt-column", action="append", dest="prompt_columns")
    generate.add_argument("--date", action="append", dest="dates")
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument(
        "--dataset-format",
        choices=("auto", "legacy-wide", "return-pairs"),
        default="auto",
    )

    generated_attribution = commands.add_parser(
        "attribute-generated",
        help="backpropagate over an existing generated forward artifact",
    )
    generated_attribution.add_argument("--model", required=True)
    generated_attribution.add_argument("--forward-artifact", required=True)
    generated_attribution.add_argument("--output", required=True)
    generated_attribution.add_argument("--input-top-k", type=int)
    generated_attribution.add_argument("--max-seq-len", type=int, default=256)

    evaluate_return = commands.add_parser(
        "evaluate-return-predictions", help="score five-class return predictions from a forward JSONL artifact"
    )
    evaluate_return.add_argument("--forward", required=True)
    evaluate_return.add_argument("--output-dir", required=True)

    validate = commands.add_parser(
        "validate-attribution", help="evaluate Semantic Scope with input ablations"
    )
    validate.add_argument("--attribution", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--max-seq-len", type=int)

    visual = commands.add_parser(
        "visualize", help="build uncertainty plots and an attribution dashboard"
    )
    visual.add_argument("--uncertainty-root", required=True)
    visual.add_argument("--forward", required=True)
    visual.add_argument("--backward")
    visual.add_argument("--validation")
    visual.add_argument("--prices", default=DEFAULT_PRICE_INPUT)
    visual.add_argument("--output-dir", required=True)
    visual.add_argument("--input-top-k", type=int, default=15)
    visual.add_argument("--tokenizer", required=True)
    visual.add_argument("--max-seq-len", type=int, default=256)

    price_distributions = commands.add_parser(
        "plot-price-distributions",
        help="plot actual closes, sampled price distributions, and errors",
    )
    price_distributions.add_argument("--sampling-root", required=True)
    price_distributions.add_argument("--prices", default=DEFAULT_PRICE_INPUT)
    price_distributions.add_argument("--output-dir", required=True)

    uncertainty_distributions = commands.add_parser(
        "plot-uncertainty-distributions",
        help="plot final-layer entropy and effective-temperature distributions",
    )
    uncertainty_distributions.add_argument("--uncertainty-root", required=True)
    uncertainty_distributions.add_argument("--output-dir", required=True)

    return_predictions = commands.add_parser(
        "visualize-return-predictions",
        help="plot five-class original/counterfactual return predictions and uncertainty",
    )
    return_predictions.add_argument("--forward", required=True)
    return_predictions.add_argument("--uncertainty", required=True)
    return_predictions.add_argument("--output-dir", required=True)

    serve = commands.add_parser(
        "serve", help="serve an interactive Jacobian-lens prompt explorer"
    )
    serve.add_argument("--model", required=True)
    serve.add_argument(
        "--lens",
        help="defaults to artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8322)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect-input":
        report = inspect_input_to_json(args.input, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["errors"]:
            raise SystemExit(1)
    elif args.command == "readout":
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
            save_prompt_hidden=args.save_prompt_hidden,
            compute_input_attribution=False,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
            dataset_format=args.dataset_format,
        )
    elif args.command == "generate":
        _run_generation_stage(args)
    elif args.command == "attribute-generated":
        _run_generated_attribution_stage(args)
    elif args.command == "evaluate-return-predictions":
        evaluate_return_predictions(args.forward, args.output_dir)
    elif args.command == "validate-attribution":
        evaluate_semantic_scope(
            attribution_path=args.attribution,
            model_name=args.model,
            output_dir=args.output_dir,
            seed=args.seed,
            max_seq_len=args.max_seq_len,
        )
    elif args.command == "visualize":
        visualize_prompt_results(
            uncertainty_paths=uncertainty_paths_from_root(args.uncertainty_root),
            forward_path=args.forward,
            backward_path=args.backward,
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
    elif args.command == "visualize-return-predictions":
        visualize_return_predictions(
            forward_path=args.forward,
            uncertainty_path=args.uncertainty,
            output_dir=args.output_dir,
        )
    elif args.command == "serve":
        import uvicorn

        from llm_bias.prompt_analysis.interactive import build_app

        app = build_app(args.model, args.lens)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
