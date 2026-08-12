"""CLI for the synthetic entity-bias pilot."""

from __future__ import annotations

import argparse
import json


def _experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--constituents", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    _experiment_arguments(validate)

    run = subparsers.add_parser("run")
    _experiment_arguments(run)
    run.add_argument("--artifact-root", default="artifacts")
    run.add_argument("--dataset", default="synthetic-entity-bias-2020-2025")
    run.add_argument("--run-id", required=True)

    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("--run-root", required=True)
    visualize.add_argument("--output-dir")
    visualize.add_argument("--replace-existing", action="store_true")
    return parser


def _load(args):
    from llm_bias.core.model import load_model

    return load_model(args.model)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "visualize":
        from .visualization import visualize_run

        output = visualize_run(
            args.run_root,
            output_dir=args.output_dir,
            replace_existing=args.replace_existing,
        )
        print(output)
        return

    from .entities import load_entity_pool
    from .prompts import render_prompt, validate_token_contract
    from .spec import BASELINE_ENTITY, TEMPLATES

    pool = load_entity_pool(args.constituents, seed=args.seed)
    model, tokenizer, device = _load(args)
    if args.command == "validate":
        from llm_bias.core.lens_loader import load_validated_lens

        loaded_lens = load_validated_lens(
            model=model,
            model_name=args.model,
            lens_path=args.lens,
            require_complete=True,
        )
        rendered = [
            render_prompt(
                tokenizer,
                template,
                entity=entity.company_name,
                ticker=entity.ticker,
                max_seq_len=args.max_seq_len,
            )
            for entity in pool
            for template in TEMPLATES
        ]
        rendered += [
            render_prompt(
                tokenizer,
                template,
                entity=BASELINE_ENTITY,
                max_seq_len=args.max_seq_len,
            )
            for template in TEMPLATES
        ]
        result = validate_token_contract(tokenizer, rendered)
        result.update(
            pool_count=len(pool),
            anomaly_count=sum(bool(entity.anomalies) for entity in pool),
            model=args.model,
            lens=str(loaded_lens.path),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    from .pipeline import run_pipeline

    root = run_pipeline(
        constituents=args.constituents,
        model_path=args.model,
        lens_path=args.lens,
        artifact_root=args.artifact_root,
        dataset=args.dataset,
        run_id=args.run_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )
    print(root)
