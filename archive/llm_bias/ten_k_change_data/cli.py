"""Build a prompt-agnostic CSV from extracted EDGAR 10-K metadata-change windows."""

from __future__ import annotations

import argparse
import json

from llm_bias.ten_k_change_data.pipeline import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    build_change_dataset,
    validate_change_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build year,cik,item rows around metadata changes")
    build.add_argument("--input", default=DEFAULT_INPUT)
    build.add_argument("--output", default=DEFAULT_OUTPUT)
    build.add_argument("--max-files", type=int)
    build.add_argument("--fail-on-input-issues", action="store_true")
    validate = commands.add_parser("validate", help="validate a published dataset")
    validate.add_argument("--input", default=DEFAULT_OUTPUT)
    generate = commands.add_parser(
        "generate", help="generate answers for change-window CSV rows"
    )
    generate.add_argument("--input", default=f"{DEFAULT_OUTPUT}/change_window_items.csv")
    generate.add_argument("--model", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=16)
    generate.add_argument("--max-seq-len", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--full-generation", action="store_true", default=True)
    structured = commands.add_parser(
        "generate-structured", help="generate schema-constrained answers via llama.cpp"
    )
    structured.add_argument("--input", default=f"{DEFAULT_OUTPUT}/change_window_items.csv")
    structured.add_argument("--output", required=True)
    structured.add_argument("--base-url", default="http://127.0.0.1:11433/v1")
    structured.add_argument("--model", default="qwen3.5-9b-mtp")
    structured.add_argument("--max-tokens", type=int, default=64)
    structured.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        result = build_change_dataset(
            args.input,
            args.output,
            max_files=args.max_files,
            fail_on_input_issues=args.fail_on_input_issues,
        )
    elif args.command == "validate":
        result = validate_change_dataset(args.input)
    elif args.command == "generate":
        from llm_bias.prompt_analysis.generation import generate_prompt_outputs

        result = generate_prompt_outputs(
            input_path=args.input,
            model_name=args.model,
            output_path=args.output,
            sample_per_condition=None,
            full_generation=True,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
            seed=args.seed,
            top_p=args.top_p,
            top_k=args.top_k,
            dataset_format="ten-k-change",
        )
        result = {"forward_artifact": str(result)}
    else:
        from llm_bias.ten_k_change_data.server_generation import generate_structured_answers

        result = generate_structured_answers(
            input_path=args.input,
            output_dir=args.output,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        result = {"structured_artifact": str(result)}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
