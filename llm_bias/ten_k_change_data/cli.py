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
    else:
        result = validate_change_dataset(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
