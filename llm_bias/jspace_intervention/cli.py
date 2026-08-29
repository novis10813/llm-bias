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
from llm_bias.jspace_intervention.schemas import (
    GainConfig,
    InterventionConfig,
    OutcomeFlipConfig,
    PriorProbeConfig,
    PrototypeSpec,
    TokenScreenCandidate,
    TokenScreenConfig,
)
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
    config.add_argument(
        "--swap-fractions", "--alphas", dest="swap_fractions",
        default="0,0.25,0.5,0.75,1",
        help="replacement fractions in [0,1]; --alphas is a deprecated alias",
    )
    config.add_argument(
        "--coordinate-modes",
        default="swap,source_removal_component,target_installation_component",
    )
    config.add_argument("--position-controls", default="evidence")
    config.add_argument("--direction-controls", default="prototype")
    config.add_argument("--output", required=True, type=Path)

    gain_config = commands.add_parser(
        "prepare-gain-config", help="derive one gain config from a frozen swap config"
    )
    gain_config.add_argument("--swap-config", required=True, type=Path)
    gain_config.add_argument("--prototype", choices=("source", "target"), required=True)
    gain_config.add_argument(
        "--token", default=None,
        help="optional concept token from the selected prototype for an exploratory gain arm",
    )
    gain_config.add_argument("--gains", default="0,0.5,1,1.5,2")
    gain_config.add_argument("--output", required=True, type=Path)

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

    gain_run = commands.add_parser("run-gain", help="run one concept-coordinate gain sweep")
    gain_run.add_argument("--input", required=True, type=Path)
    gain_run.add_argument("--split-manifest", required=True, type=Path)
    gain_run.add_argument("--config", required=True, type=Path)
    gain_run.add_argument("--model", required=True)
    gain_run.add_argument("--lens", default=None)
    gain_run.add_argument("--run-id", required=True)
    gain_run.add_argument("--dataset", default="jspace-concept-gain")
    gain_run.add_argument("--artifact-root", default="artifacts")
    gain_run.add_argument("--split", choices=("discovery", "calibration", "test"), default="test")
    gain_run.add_argument("--max-records", type=int, default=None)
    gain_run.add_argument("--max-seq-len", type=int, default=1024)
    gain_run.add_argument("--prompt-column", action="append", dest="prompt_columns")

    valence = commands.add_parser(
        "run-valence-readout",
        help="positive-vs-negative J-space vocabulary readout (prepare/forward/analyze/finalize)",
    )
    valence.add_argument("--input", required=True, type=Path)
    valence.add_argument("--split-manifest", required=True, type=Path)
    valence.add_argument("--model", required=True)
    valence.add_argument("--lens", default=None)
    valence.add_argument("--run-id", required=True)
    valence.add_argument("--dataset", default="jspace-valence-readout")
    valence.add_argument("--artifact-root", default="artifacts")
    valence.add_argument("--sector", default="Technology")
    valence.add_argument("--split", choices=("discovery", "calibration", "test"), default="discovery")
    valence.add_argument("--trials-per-ticker", type=int, default=3)
    valence.add_argument("--layers", default=",".join(str(layer) for layer in range(14, 27)))
    valence.add_argument("--top-k", type=int, default=30)
    valence.add_argument("--max-seq-len", type=int, default=1024)
    valence.add_argument("--seed", type=int, default=0)

    token_screen_config = commands.add_parser(
        "prepare-token-screen-config",
        help="freeze a token screen config from a completed valence candidate artifact",
    )
    token_screen_config.add_argument(
        "--candidates", required=True, type=Path,
        help="completed valence frozen_candidate_suggestions.json",
    )
    token_screen_config.add_argument("--model", required=True)
    token_screen_config.add_argument("--source-sector", required=True)
    token_screen_config.add_argument("--split-manifest", required=True, type=Path)
    token_screen_config.add_argument(
        "--layers", default=",".join(str(layer) for layer in range(14, 27))
    )
    token_screen_config.add_argument(
        "--alphas", default="-1,0,1",
        help="symmetric dose pair plus 0, e.g. -1,0,1",
    )
    token_screen_config.add_argument("--top-positions", type=int, default=3)
    token_screen_config.add_argument("--loading-threshold", type=float, default=0.0)
    token_screen_config.add_argument("--controls", default="token,matched_random")
    token_screen_config.add_argument("--output", required=True, type=Path)

    token_screen_run = commands.add_parser(
        "run-token-screen", help="run the frozen token causal screen"
    )
    token_screen_run.add_argument("--input", required=True, type=Path)
    token_screen_run.add_argument("--split-manifest", required=True, type=Path)
    token_screen_run.add_argument("--config", required=True, type=Path)
    token_screen_run.add_argument("--model", required=True)
    token_screen_run.add_argument("--lens", default=None)
    token_screen_run.add_argument("--run-id", required=True)
    token_screen_run.add_argument("--dataset", default="jspace-token-screen")
    token_screen_run.add_argument("--artifact-root", default="artifacts")
    token_screen_run.add_argument(
        "--split", choices=("discovery", "calibration", "test"), default="discovery"
    )
    token_screen_run.add_argument("--max-records", type=int, default=None)
    token_screen_run.add_argument("--max-seq-len", type=int, default=1024)
    token_screen_run.add_argument("--prompt-column", action="append", dest="prompt_columns")

    outcome_config = commands.add_parser(
        "prepare-outcome-flip-config",
        help="freeze a V2 outcome-conditioned decision-flip config (Draft 1)",
    )
    outcome_config.add_argument("--model", required=True)
    outcome_config.add_argument("--source-sector", required=True)
    outcome_config.add_argument("--split-manifest", required=True, type=Path)
    outcome_config.add_argument(
        "--fitted-layers", default=None,
        help="comma-separated layers; default is the union of --bands",
    )
    outcome_config.add_argument(
        "--bands", default="14-26,12-28,10-30",
        help="candidate layer bands as start-end pairs",
    )
    outcome_config.add_argument(
        "--position-rules", default="evidence_item_end,evidence_span_all"
    )
    outcome_config.add_argument("--dose-grid", default="0.05,0.10,0.20,0.40")
    outcome_config.add_argument("--safety-bound", type=float, default=0.50)
    outcome_config.add_argument("--scale-floor", type=float, default=1.0)
    outcome_config.add_argument(
        "--clean-margin-edges", default="0.5,1.5",
        help="clean margin |M| stratum edges, strictly increasing",
    )
    outcome_config.add_argument("--min-flip-rate", type=float, default=0.10)
    outcome_config.add_argument("--parse-success-gate", type=float, default=0.90)
    outcome_config.add_argument("--agreement-gate", type=float, default=0.70)
    outcome_config.add_argument("--max-new-tokens", type=int, default=256)
    outcome_config.add_argument("--fitting-seed", type=int, default=0)
    outcome_config.add_argument("--output", required=True, type=Path)

    outcome_run = commands.add_parser(
        "run-outcome-flip",
        help="run one V2 outcome-conditioned decision-flip split "
        "(discovery/calibration/test)",
    )
    outcome_run.add_argument("--input", required=True, type=Path)
    outcome_run.add_argument("--split-manifest", required=True, type=Path)
    outcome_run.add_argument("--config", required=True, type=Path)
    outcome_run.add_argument("--model", required=True)
    outcome_run.add_argument("--run-id", required=True)
    outcome_run.add_argument("--dataset", default="jspace-outcome-direction-flip")
    outcome_run.add_argument("--artifact-root", default="artifacts")
    outcome_run.add_argument(
        "--split", choices=("discovery", "calibration", "test"), default="discovery"
    )
    outcome_run.add_argument(
        "--direction-identity", type=Path, default=None,
        help="discovery direction identity artifact (required for calibration/test)",
    )
    outcome_run.add_argument(
        "--calibration-selection", type=Path, default=None,
        help="calibration selection artifact (required for test)",
    )
    outcome_run.add_argument("--max-records", type=int, default=None)
    outcome_run.add_argument("--max-seq-len", type=int, default=1024)
    outcome_run.add_argument("--prompt-column", action="append", dest="prompt_columns")

    geometry_run = commands.add_parser(
        "run-outcome-geometry",
        help="auxiliary geometric projection of the sector state difference "
        "onto the frozen V2 outcome directions (descriptive, non-causal)",
    )
    geometry_run.add_argument("--input", required=True, type=Path)
    geometry_run.add_argument("--split-manifest", required=True, type=Path)
    geometry_run.add_argument(
        "--config", required=True, type=Path,
        help="frozen V2 outcome_flip_config",
    )
    geometry_run.add_argument(
        "--direction-identity", required=True, type=Path,
        help="V2 discovery direction identity artifact",
    )
    geometry_run.add_argument("--model", required=True)
    geometry_run.add_argument("--run-id", required=True)
    geometry_run.add_argument("--dataset", default="jspace-outcome-direction-geometry")
    geometry_run.add_argument("--artifact-root", default="artifacts")
    geometry_run.add_argument("--contrast-sector", default="Financial Services")
    geometry_run.add_argument(
        "--lens", default=None,
        help="canonical lens (required with --tfidf-config)",
    )
    geometry_run.add_argument(
        "--tfidf-config", type=Path, default=None,
        help="frozen contrastive TF-IDF sector config (optional geometry arm)",
    )
    geometry_run.add_argument("--max-seq-len", type=int, default=1024)
    geometry_run.add_argument("--prompt-column", action="append", dest="prompt_columns")

    prior_probe_config = commands.add_parser(
        "prepare-prior-probe-config",
        help="freeze a V2 zero-evidence header-only prior probe config",
    )
    prior_probe_config.add_argument("--model", required=True)
    prior_probe_config.add_argument("--input", required=True, type=Path)
    prior_probe_config.add_argument("--split-manifest", required=True, type=Path)
    prior_probe_config.add_argument(
        "--outcome-flip-config", required=True, type=Path,
        help="frozen V2 outcome_flip_config that defines the direction fitting",
    )
    prior_probe_config.add_argument(
        "--direction-identity", required=True, type=Path,
        help="frozen V2 discovery direction identity artifact",
    )
    prior_probe_config.add_argument(
        "--condition", action="append", required=True,
        help="one frozen probe condition TICKER:SECTOR (repeatable)",
    )
    prior_probe_config.add_argument(
        "--position-rule", default="evidence_item_end",
        help="V2 position rule whose frozen direction is probed",
    )
    prior_probe_config.add_argument(
        "--neutral-item", default="No evidence provided.",
        help="single-line neutral evidence body, identical for all conditions",
    )
    prior_probe_config.add_argument(
        "--contrast-tickers", default="NVDA:JPM",
        help="two distinct tickers for the frozen pair contrast, TICKER1:TICKER2",
    )
    prior_probe_config.add_argument(
        "--contrast-sectors", default="Technology:Financial Services",
        help="frozen sector-label contrast orientation, LABEL1:LABEL2 (LABEL1 minus LABEL2)",
    )
    prior_probe_config.add_argument("--scale-floor", type=float, default=1.0)
    prior_probe_config.add_argument("--output", required=True, type=Path)

    prior_probe_run = commands.add_parser(
        "run-prior-probe",
        help="run the frozen V2 zero-evidence header-only prior probe",
    )
    prior_probe_run.add_argument("--config", required=True, type=Path)
    prior_probe_run.add_argument("--model", required=True)
    prior_probe_run.add_argument("--run-id", required=True)
    prior_probe_run.add_argument("--dataset", default="jspace-outcome-direction-flip")
    prior_probe_run.add_argument("--artifact-root", default="artifacts")

    outcome_decode = commands.add_parser(
        "decode-outcome-direction",
        help="V2 auxiliary diagnostic: decode the frozen outcome direction d_l "
        "with the canonical Jacobian lens (prepare/forward/analyze/finalize)",
    )
    outcome_decode.add_argument("--input", required=True, type=Path)
    outcome_decode.add_argument("--split-manifest", required=True, type=Path)
    outcome_decode.add_argument("--config", required=True, type=Path)
    outcome_decode.add_argument("--direction-identity", required=True, type=Path)
    outcome_decode.add_argument("--calibration-selection", required=True, type=Path)
    outcome_decode.add_argument("--model", required=True)
    outcome_decode.add_argument("--lens", default=None)
    outcome_decode.add_argument("--run-id", required=True)
    outcome_decode.add_argument("--dataset", default="jspace-outcome-direction-decode")
    outcome_decode.add_argument("--artifact-root", default="artifacts")
    outcome_decode.add_argument("--top-k", type=int, default=30)
    outcome_decode.add_argument("--max-seq-len", type=int, default=1024)

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
            "swap_fractions": [
                float(value) for value in args.swap_fractions.split(",") if value.strip()
            ],
            "coordinate_modes": [
                value for value in args.coordinate_modes.split(",") if value.strip()
            ],
            "position_controls": [
                value for value in args.position_controls.split(",") if value.strip()
            ],
            "direction_controls": [
                value for value in args.direction_controls.split(",") if value.strip()
            ],
        }
    )
    write_json(
        args.output,
        {
            **config.to_dict(),
            "artifact_type": "jspace_intervention_config",
            "schema_version": 2,
            "discovery_run": str(args.discovery_run),
            "discovery_summary_sha256": sha256_file(args.discovery_run / "summary.json"),
            "split_manifest_sha256": summary["params"]["split_manifest_sha256"],
            "candidate_table_sha256": sha256_file(args.discovery_run / filename),
            "model": args.model,
        },
        overwrite=True,
    )
    print(args.output)


def _prepare_gain_config(args: argparse.Namespace) -> None:
    payload = json.loads(args.swap_config.read_text())
    swap_config = InterventionConfig.from_dict(payload)
    prototype = (
        swap_config.source if args.prototype == "source" else swap_config.target
    )
    if args.token is not None:
        matches = [token for token in prototype.tokens if token.token == args.token]
        if len(matches) != 1:
            raise ValueError(
                f"token {args.token!r} is not present exactly once in {prototype.name}"
            )
        prototype = PrototypeSpec.from_dict(
            {
                "name": f"{prototype.sector}:{args.token}:gain",
                "sector": prototype.sector,
                "score_type": prototype.score_type,
                "tokens": [{
                    "token": matches[0].token,
                    "token_id": matches[0].token_id,
                    "weight": 1.0,
                    "selection_score": matches[0].selection_score,
                }],
            }
        )
    gain_config = GainConfig.from_dict(
        {
            "prototype": prototype.to_dict(),
            "layers": list(swap_config.layers),
            "gains": [float(value) for value in args.gains.split(",") if value.strip()],
            "position_controls": list(swap_config.position_controls),
            "direction_controls": list(swap_config.direction_controls),
            "top_positions": swap_config.top_positions,
            "loading_threshold": swap_config.loading_threshold,
            "decision_prefix": swap_config.decision_prefix,
            "positive_candidate": swap_config.positive_candidate,
            "negative_candidate": swap_config.negative_candidate,
        }
    )
    provenance = {
        key: value
        for key, value in payload.items()
        if key in {
            "model",
            "discovery_run",
            "discovery_summary_sha256",
            "split_manifest_sha256",
            "candidate_table_sha256",
        }
    }
    write_json(
        args.output,
        {
            **gain_config.to_dict(),
            **provenance,
            "artifact_type": "jspace_gain_config",
            "schema_version": 2,
            "parent_swap_config": str(args.swap_config),
            "parent_swap_config_sha256": sha256_file(args.swap_config),
        },
        overwrite=True,
    )
    print(args.output)


def _validate_config(args: argparse.Namespace) -> None:
    payload = json.loads(args.config.read_text())
    if payload.get("artifact_type") == "outcome_flip_config" or (
        "fitted_layers" in payload and "candidate_bands" in payload
    ):
        config = OutcomeFlipConfig.from_dict(payload)
    elif "candidates" in payload and "candidate_artifact_sha256" in payload:
        config = TokenScreenConfig.from_dict(payload)
    elif "prototype" in payload:
        config = GainConfig.from_dict(payload)
    else:
        config = InterventionConfig.from_dict(payload)
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))


def _prepare_token_screen_config(args: argparse.Namespace) -> None:
    payload = json.loads(args.candidates.read_text())
    if payload.get("artifact_type") != "frozen_candidate_suggestions":
        raise ValueError(
            "candidate input must be a completed valence "
            "frozen_candidate_suggestions artifact"
        )
    candidates = [
        TokenScreenCandidate.from_dict(row).to_dict()
        for row in payload.get("candidates", [])
    ]
    if not candidates:
        raise ValueError("candidate artifact contains no frozen candidates")
    config = TokenScreenConfig.from_dict(
        {
            "candidates": candidates,
            "model": args.model,
            "source_sector": args.source_sector,
            "layers": [int(value) for value in args.layers.split(",") if value.strip()],
            "alphas": [float(value) for value in args.alphas.split(",") if value.strip()],
            "top_positions": args.top_positions,
            "loading_threshold": args.loading_threshold,
            "controls": [value for value in args.controls.split(",") if value.strip()],
            "candidate_artifact_path": str(args.candidates),
            "candidate_artifact_sha256": sha256_file(args.candidates),
            "split_manifest_sha256": sha256_file(args.split_manifest),
        }
    )
    write_json(
        args.output,
        {
            **config.to_dict(),
            "artifact_type": "jspace_token_screen_config",
            "schema_version": 1,
            "model": args.model,
        },
        overwrite=True,
    )
    print(args.output)


def _prepare_outcome_flip_config(args: argparse.Namespace) -> None:
    bands: list[list[int]] = []
    for part in args.bands.split(","):
        part = part.strip()
        if not part:
            continue
        start_text, end_text = part.split("-")
        bands.append([int(start_text), int(end_text)])
    if not bands:
        raise ValueError("--bands must contain at least one start-end band")
    if args.fitted_layers is None:
        fitted = sorted(
            {layer for band in bands for layer in range(band[0], band[1] + 1)}
        )
    else:
        fitted = [int(value) for value in args.fitted_layers.split(",") if value.strip()]
    config = OutcomeFlipConfig.from_dict(
        {
            "model": args.model,
            "source_sector": args.source_sector,
            "fitted_layers": fitted,
            "candidate_bands": bands,
            "position_rules": [
                value for value in args.position_rules.split(",") if value.strip()
            ],
            "dose_grid": [
                float(value) for value in args.dose_grid.split(",") if value.strip()
            ],
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "safety_bound": args.safety_bound,
            "scale_floor": args.scale_floor,
            "clean_margin_edges": [
                float(value)
                for value in args.clean_margin_edges.split(",")
                if value.strip()
            ],
            "min_flip_rate": args.min_flip_rate,
            "parse_success_gate": args.parse_success_gate,
            "agreement_gate": args.agreement_gate,
            "max_new_tokens": args.max_new_tokens,
            "fitting_seed": args.fitting_seed,
        }
    )
    write_json(
        args.output,
        {
            **config.to_dict(),
            "artifact_type": "outcome_flip_config",
            "schema_version": 1,
            "model": args.model,
        },
        overwrite=True,
    )
    print(args.output)


def _run_outcome_flip(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.outcome_flip import run_outcome_flip_pipeline

    run_root = run_outcome_flip_pipeline(
        input_path=args.input,
        split_manifest=args.split_manifest,
        config_path=args.config,
        model_name=args.model,
        run_id=args.run_id,
        dataset=args.dataset,
        artifact_root=args.artifact_root,
        split_name=args.split,
        direction_identity_path=args.direction_identity,
        calibration_selection_path=args.calibration_selection,
        max_records=args.max_records,
        max_seq_len=args.max_seq_len,
        prompt_columns=set(args.prompt_columns) if args.prompt_columns else None,
    )
    print(run_root)


def _run_outcome_geometry(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.outcome_geometry import (
        run_outcome_geometry_pipeline,
    )

    run_root = run_outcome_geometry_pipeline(
        input_path=args.input,
        split_manifest=args.split_manifest,
        config_path=args.config,
        direction_identity_path=args.direction_identity,
        model_name=args.model,
        run_id=args.run_id,
        dataset=args.dataset,
        artifact_root=args.artifact_root,
        contrast_sector=args.contrast_sector,
        lens_path=args.lens,
        tfidf_config_path=args.tfidf_config,
        max_seq_len=args.max_seq_len,
        prompt_columns=set(args.prompt_columns) if args.prompt_columns else None,
    )
    print(run_root)


def _prepare_prior_probe_config(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention import prior_probe

    conditions: list[dict[str, str]] = []
    for raw in args.condition:
        ticker, separator, sector = raw.partition(":")
        if not separator or not ticker or not sector:
            raise ValueError(
                f"invalid --condition {raw!r}; expected TICKER:SECTOR"
            )
        conditions.append({"ticker": ticker, "sector": sector})

    paths = {
        "input": args.input,
        "split_manifest": args.split_manifest,
        "outcome_flip_config": args.outcome_flip_config,
        "direction_identity": args.direction_identity,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    v2_payload = json.loads(paths["outcome_flip_config"].read_text(encoding="utf-8"))
    if v2_payload.get("artifact_type") != "outcome_flip_config":
        raise ValueError(
            "--outcome-flip-config must be a frozen outcome_flip_config artifact"
        )
    v2_config = OutcomeFlipConfig.from_dict(v2_payload)
    if v2_config.model != args.model:
        raise ValueError(
            f"model {args.model!r} does not match the outcome flip config model"
        )

    identity = json.loads(paths["direction_identity"].read_text(encoding="utf-8"))
    if identity.get("artifact_type") != "outcome_flip_direction_identity":
        raise ValueError(
            "--direction-identity must be an outcome_flip_direction_identity artifact"
        )
    if identity.get("split") != "discovery":
        raise ValueError("direction identity must come from a discovery run")
    if identity.get("model") != args.model:
        raise ValueError("direction identity model does not match the run model")
    if identity.get("config_sha256") != sha256_file(paths["outcome_flip_config"]):
        raise ValueError(
            "direction identity binds a different outcome flip config"
        )
    if identity.get("split_manifest_sha256") != sha256_file(paths["split_manifest"]):
        raise ValueError("direction identity binds a different split manifest")
    if args.position_rule not in identity.get("position_rules", []):
        raise ValueError(
            f"position rule {args.position_rule!r} is not in the direction identity"
        )

    split_payload = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    if "assignments" not in split_payload:
        raise ValueError("split manifest is missing ticker assignments")

    ticker_index = prior_probe._read_ticker_index(paths["input"])
    known_sectors = {row["sector"] for row in ticker_index.values()}
    for condition in conditions:
        if condition["ticker"] not in ticker_index:
            raise ValueError(
                f"probe ticker {condition['ticker']!r} is not in the input CSV"
            )
        if condition["sector"] not in known_sectors:
            raise ValueError(
                f"probe sector {condition['sector']!r} is not a canonical sector "
                "label in the input CSV"
            )
    first_ticker, separator, second_ticker = args.contrast_tickers.partition(":")
    if not separator or not first_ticker or not second_ticker:
        raise ValueError(
            "invalid --contrast-tickers; expected TICKER1:TICKER2"
        )
    first_sector, separator, second_sector = args.contrast_sectors.partition(":")
    if not separator or not first_sector or not second_sector:
        raise ValueError(
            "invalid --contrast-sectors; expected LABEL1:LABEL2"
        )

    config = PriorProbeConfig.from_dict(
        {
            "model": args.model,
            "input": str(args.input),
            "input_sha256": sha256_file(paths["input"]),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": sha256_file(paths["split_manifest"]),
            "outcome_flip_config": str(args.outcome_flip_config),
            "outcome_flip_config_sha256": sha256_file(paths["outcome_flip_config"]),
            "direction_identity": str(args.direction_identity),
            "direction_identity_sha256": sha256_file(paths["direction_identity"]),
            "position_rule": args.position_rule,
            "conditions": conditions,
            "contrast_tickers": [first_ticker, second_ticker],
            "contrast_sectors": [first_sector, second_sector],
            "neutral_evidence_item": args.neutral_item,
            "scale_floor": args.scale_floor,
        }
    )
    write_json(
        args.output,
        {
            **config.to_dict(),
            "artifact_type": prior_probe.PROBE_CONFIG_ARTIFACT_TYPE,
            "schema_version": 1,
        },
        overwrite=True,
    )
    print(args.output)


def _run_prior_probe(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.prior_probe import run_prior_probe_pipeline

    run_root = run_prior_probe_pipeline(
        config_path=args.config,
        model_name=args.model,
        run_id=args.run_id,
        dataset=args.dataset,
        artifact_root=args.artifact_root,
    )
    print(run_root)


def _decode_outcome_direction(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.outcome_decode import run_outcome_decode_pipeline

    run_root = run_outcome_decode_pipeline(
        input_path=args.input,
        split_manifest=args.split_manifest,
        config_path=args.config,
        model_name=args.model,
        run_id=args.run_id,
        direction_identity_path=args.direction_identity,
        calibration_selection_path=args.calibration_selection,
        dataset=args.dataset,
        artifact_root=args.artifact_root,
        lens_path=args.lens,
        top_k=args.top_k,
        max_seq_len=args.max_seq_len,
    )
    print(run_root)


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


def _run_gain(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.pipeline import run_gain_pipeline

    run_root = run_gain_pipeline(
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


def _run_valence_readout(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.valence_readout import run_valence_readout_pipeline

    run_root = run_valence_readout_pipeline(
        input_path=args.input,
        split_manifest=args.split_manifest,
        model_name=args.model,
        lens_path=args.lens,
        run_id=args.run_id,
        dataset=args.dataset,
        artifact_root=args.artifact_root,
        sector=args.sector,
        split_name=args.split,
        trials_per_ticker=args.trials_per_ticker,
        layers=[int(value) for value in args.layers.split(",") if value.strip()],
        top_k=args.top_k,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    print(run_root)


def _run_token_screen(args: argparse.Namespace) -> None:
    from llm_bias.jspace_intervention.pipeline import run_token_screen_pipeline

    run_root = run_token_screen_pipeline(
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
            "schema_version": 2,
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
    elif args.command == "prepare-gain-config":
        _prepare_gain_config(args)
    elif args.command == "prepare-token-screen-config":
        _prepare_token_screen_config(args)
    elif args.command == "prepare-outcome-flip-config":
        _prepare_outcome_flip_config(args)
    elif args.command == "validate-config":
        _validate_config(args)
    elif args.command == "run-swap":
        _run_swap(args)
    elif args.command == "run-gain":
        _run_gain(args)
    elif args.command == "run-valence-readout":
        _run_valence_readout(args)
    elif args.command == "run-token-screen":
        _run_token_screen(args)
    elif args.command == "run-outcome-flip":
        _run_outcome_flip(args)
    elif args.command == "run-outcome-geometry":
        _run_outcome_geometry(args)
    elif args.command == "prepare-prior-probe-config":
        _prepare_prior_probe_config(args)
    elif args.command == "run-prior-probe":
        _run_prior_probe(args)
    elif args.command == "decode-outcome-direction":
        _decode_outcome_direction(args)
    else:
        _analyze(args)


if __name__ == "__main__":
    main()
