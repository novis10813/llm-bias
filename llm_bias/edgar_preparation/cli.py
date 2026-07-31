"""Prepare an auditable JSONL corpus from extracted EDGAR 8-K filings."""

from __future__ import annotations

import argparse
import json

from llm_bias.edgar_preparation.pipeline import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    clean_filings,
    validate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    clean = commands.add_parser("clean", help="clean extracted filing JSON files")
    clean.add_argument("--input", default=DEFAULT_INPUT)
    clean.add_argument("--output", default=DEFAULT_OUTPUT)
    clean.add_argument("--max-files", type=int)

    validate = commands.add_parser("validate", help="validate a staged dataset")
    validate.add_argument("--input", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "clean":
        result = clean_filings(args.input, args.output, max_files=args.max_files)
    else:
        result = validate_dataset(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
