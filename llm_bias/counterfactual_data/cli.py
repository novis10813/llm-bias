"""Build and validate the 8-K counterfactual entity-bias dataset."""

from __future__ import annotations

import argparse
import json

from llm_bias.counterfactual_data.annotation import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_CHAR_BUFFER,
    DEFAULT_MODEL,
    annotate_events,
)
from llm_bias.counterfactual_data.pipeline import (
    DEFAULT_CLEANED,
    DEFAULT_INDICES,
    DEFAULT_METADATA,
    DEFAULT_OUTPUT,
    DEFAULT_SEED,
    build_company_history,
    build_pairs,
    create_review_bundle,
    promote_reviewed_annotations,
    render_pairs,
    sample_events,
    validate_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    entities = commands.add_parser("entities", help="build point-in-time CIK history")
    entities.add_argument("--metadata", default=DEFAULT_METADATA)
    entities.add_argument("--indices", default=DEFAULT_INDICES)
    entities.add_argument("--output", default=DEFAULT_OUTPUT)

    sample = commands.add_parser("sample", help="sample eligible earnings events")
    sample.add_argument("--cleaned", default=DEFAULT_CLEANED)
    sample.add_argument("--output", default=DEFAULT_OUTPUT)
    sample.add_argument("--count", type=int, default=500)
    sample.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sample.add_argument("--max-per-cik", type=int, default=2)

    annotate = commands.add_parser("annotate", help="run local LangExtract annotation")
    annotate.add_argument("--output", default=DEFAULT_OUTPUT)
    annotate.add_argument("--model", default=DEFAULT_MODEL)
    annotate.add_argument("--base-url", default=DEFAULT_BASE_URL)
    annotate.add_argument("--api-key", default="local")
    annotate.add_argument("--max-events", type=int)
    annotate.add_argument(
        "--max-char-buffer", type=int, default=DEFAULT_MAX_CHAR_BUFFER
    )
    annotate.add_argument("--no-resume", action="store_true")

    review = commands.add_parser("review-bundle", help="prepare the manual gold review")
    review.add_argument("--output", default=DEFAULT_OUTPUT)
    review.add_argument("--count", type=int, default=200)
    review.add_argument("--seed", type=int, default=DEFAULT_SEED)

    promote = commands.add_parser("promote", help="enforce review gates and promote data")
    promote.add_argument("--output", default=DEFAULT_OUTPUT)
    promote.add_argument("--review", required=True)
    promote.add_argument("--minimum-reviews", type=int, default=200)

    pairs = commands.add_parser("build-pairs", help="build all entity contrasts")
    pairs.add_argument("--output", default=DEFAULT_OUTPUT)
    pairs.add_argument("--target-use-cap", type=int, default=5)

    render = commands.add_parser("render", help="render token spans for a local model")
    render.add_argument("--output", default=DEFAULT_OUTPUT)
    render.add_argument("--model", required=True)

    validate = commands.add_parser("validate", help="validate available stage artifacts")
    validate.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "entities":
        result = build_company_history(args.metadata, args.output, args.indices)
    elif args.command == "sample":
        result = sample_events(
            args.cleaned,
            args.output,
            count=args.count,
            seed=args.seed,
            max_per_cik=args.max_per_cik,
        )
    elif args.command == "annotate":
        result = annotate_events(
            args.output,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_events=args.max_events,
            max_char_buffer=args.max_char_buffer,
            resume=not args.no_resume,
        )
    elif args.command == "review-bundle":
        result = create_review_bundle(args.output, count=args.count, seed=args.seed)
    elif args.command == "promote":
        result = promote_reviewed_annotations(
            args.review, args.output, minimum_reviews=args.minimum_reviews
        )
    elif args.command == "build-pairs":
        result = build_pairs(args.output, target_use_cap=args.target_use_cap)
    elif args.command == "render":
        result = render_pairs(args.model, args.output)
    else:
        result = validate_outputs(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
