"""CLI for causal J-space steering and sector-coordinate replacement."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.artifacts.io import write_json
from llm_bias.jspace_intervention.analysis import grouped_effects
from llm_bias.jspace_intervention.candidates import select_prototype
from llm_bias.jspace_intervention.schemas import InterventionConfig
from llm_bias.jspace_intervention.splits import (
    assign_balanced_ticker_splits,
    assign_ticker_splits,
    split_counts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-splits", help="create ticker-grouped splits")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument(
        "--forward-artifact", type=Path, default=None,
        help="optional generated outputs used to balance baseline company buy rates",
    )

    config = commands.add_parser(
        "prepare-config", help="freeze sector prototypes from a discovery-only analysis"
    )
    config.add_argument("--discovery-run", required=True, type=Path)
    config.add_argument("--model", required=True)
    config.add_argument("--source-sector", required=True)
    config.add_argument("--target-sector", required=True)
    config.add_argument("--score-type", choices=("tfidf", "logodds_z"), default="logodds_z")
    config.add_argument("--top-n", type=int, default=4)
    config.add_argument("--layers", default="14,15,16,17,18,19,20,21,22,23,24,25,26")
    config.add_argument("--alphas", default="0,0.5,1,2")
    config.add_argument(
        "--coordinate-modes",
        default="swap,source_ablation,target_addition",
    )
    config.add_argument("--output", required=True, type=Path)

    validate = commands.add_parser("validate-config", help="validate an intervention JSON config")
    validate.add_argument("--config", required=True, type=Path)

    run = commands.add_parser("run-swap", help="run one directional sector-prototype swap")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--split-manifest", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--lens", default=None)
    run.add_argument("--run-id", required=True)
    run.add_argument("--dataset", default="jspace-sector-intervention")
    run.add_argument("--artifact-root", default="artifacts")
    run.add_argument("--split", choices=("discovery", "calibration", "test"), default="test")
    run.add_argument("--max-records", type=int, default=None)
    run.add_argument("--max-seq-len", type=int, default=1024)
    run.add_argument("--prompt-column", action="append", dest="prompt_columns")

    analyze = commands.add_parser("analyze", help="summarize compact intervention JSONL")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=0)
    return parser


def _prepare_splits(args: argparse.Namespace) -> None:
    ticker_to_sector = {}
    ticker_marketcap = {}
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ticker = row.get("ticker") or ""
            sector = row.get("sector") or ""
            previous = ticker_to_sector.setdefault(ticker, sector)
            if previous != sector:
                raise ValueError(f"ticker {ticker!r} has conflicting sectors")
            ticker_marketcap[ticker] = float(row.get("marketcap") or 0.0)
    balance_metadata = {"method": "sector_stratified_hash"}
    if args.forward_artifact is None:
        assignments = assign_ticker_splits(ticker_to_sector, seed=args.seed)
    else:
        decisions: dict[str, Counter] = defaultdict(Counter)
        pattern = re.compile(r'"decision"\s*:\s*"(buy|sell)"', re.IGNORECASE)
        with args.forward_artifact.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                match = pattern.search(str(row.get("generated_text") or ""))
                if match is None:
                    raise ValueError(f"could not parse decision for {row.get('record_id')}")
                decisions[str(row["ticker"])][match.group(1).lower()] += 1
        ticker_buy_rate = {
            ticker: counts["buy"] / sum(counts.values())
            for ticker, counts in decisions.items()
        }
        assignments = assign_balanced_ticker_splits(
            ticker_to_sector,
            ticker_buy_rate,
            ticker_marketcap,
            seed=args.seed,
        )
        balance_metadata = {
            "method": "sector_local_buy_rate_blocks_with_marketcap_order",
            "forward_artifact": str(args.forward_artifact),
            "forward_sha256": sha256_file(args.forward_artifact),
        }
    write_json(
        args.output,
        {
            "artifact_type": "jspace_intervention_splits",
            "schema_version": 1,
            "input": str(args.input),
            "input_sha256": sha256_file(args.input),
            "seed": args.seed,
            "ratios": {"discovery": 3, "calibration": 1, "test": 1},
            "balance": balance_metadata,
            "counts": split_counts(assignments, ticker_to_sector),
            "assignments": assignments,
        },
        overwrite=True,
    )
    print(args.output)


def _prepare_config(args: argparse.Namespace) -> None:
    from llm_bias.core.model import load_tokenizer

    summary = json.loads((args.discovery_run / "summary.json").read_text())
    if summary.get("params", {}).get("split_name") != "discovery":
        raise ValueError("candidate config requires a discovery-only analysis run")
    filename = "sector_keywords.jsonl" if args.score_type == "tfidf" else "sector_logodds.jsonl"
    rows = [
        json.loads(line)
        for line in (args.discovery_run / filename).open()
        if line.strip()
    ]
    tokenizer = load_tokenizer(args.model)
    source = select_prototype(
        rows, tokenizer=tokenizer, sector=args.source_sector,
        score_type=args.score_type, top_n=args.top_n,
        contrast_sector=(args.target_sector if args.score_type == "tfidf" else None),
    )
    target = select_prototype(
        rows, tokenizer=tokenizer, sector=args.target_sector,
        score_type=args.score_type, top_n=args.top_n,
        contrast_sector=(args.source_sector if args.score_type == "tfidf" else None),
    )
    config = InterventionConfig.from_dict(
        {
            "source": source.to_dict(),
            "target": target.to_dict(),
            "layers": [int(value) for value in args.layers.split(",") if value.strip()],
            "alphas": [float(value) for value in args.alphas.split(",") if value.strip()],
            "coordinate_modes": [
                value for value in args.coordinate_modes.split(",") if value.strip()
            ],
        }
    )
    write_json(
        args.output,
        {
            **config.to_dict(),
            "artifact_type": "jspace_intervention_config",
            "schema_version": 1,
            "discovery_run": str(args.discovery_run),
            "discovery_summary_sha256": sha256_file(args.discovery_run / "summary.json"),
            "split_manifest_sha256": summary["params"]["split_manifest_sha256"],
            "candidate_table_sha256": sha256_file(args.discovery_run / filename),
            "model": args.model,
        },
        overwrite=True,
    )
    print(args.output)


def _validate_config(args: argparse.Namespace) -> None:
    config = InterventionConfig.from_dict(json.loads(args.config.read_text()))
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))


def _run_swap(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.pipeline import run_swap_pipeline

    run_root = run_swap_pipeline(
        input_path=args.input,
        split_manifest=args.split_manifest,
        config_path=args.config,
        model_name=args.model,
        lens_path=args.lens,
        run_id=args.run_id,
        dataset=args.dataset,
        artifact_root=args.artifact_root,
        split_name=args.split,
        max_records=args.max_records,
        max_seq_len=args.max_seq_len,
        prompt_columns=set(args.prompt_columns) if args.prompt_columns else None,
    )
    print(run_root)


def _analyze(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.input.open() if line.strip()]
    summary = grouped_effects(
        rows, seed=args.seed, bootstrap_samples=args.bootstrap_samples
    )
    write_json(
        args.output,
        {
            "artifact_type": "jspace_intervention_analysis",
            "schema_version": 1,
            "input": str(args.input),
            "input_sha256": sha256_file(args.input),
            "groups": summary,
        },
        overwrite=True,
    )
    print(args.output)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-splits":
        _prepare_splits(args)
    elif args.command == "prepare-config":
        _prepare_config(args)
    elif args.command == "validate-config":
        _validate_config(args)
    elif args.command == "run-swap":
        _run_swap(args)
    else:
        _analyze(args)


if __name__ == "__main__":
    main()
