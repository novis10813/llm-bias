"""Phase 2A pipeline: cross-entity probe with H4 dial readout.

Stages: prepare → forward → analyze. Reuses the shared ArtifactRun
manifest lifecycle and core prompt/scoring mechanics. The frozen protocol
is docs/balanced-evidence-gap/details/proposal-phase2.md.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids

from .analysis import GATE_2A, bootstrap_ci, evaluate_gate_2a, spearman
from .intervention import answer_token_ids, capture_mlp_channel, clean_margin
from .spans import resolve_row
from .template import (
    ALL_TICKERS,
    DIAL_LAYER,
    DIAL_NEURON,
    DATASET,
    DECISION_PREFIX,
    REVERSE_OPTIONS,
    SCHEMA_VERSION,
    SECTOR_OF,
    variant_id,
)

INPUT_DATA_PATH = Path("data/baseline/investment-dial/exploratory-v1.json")
PHASE1_SUMMARY_PATH = Path(
    "artifacts/qwen3.5-4b/balanced-evidence-gap/runs/balanced-gap-gpu-bf16-01/analyze/summary.json"
)


def company_names(data_path: Path) -> dict[str, str]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    names = {c["ticker"]: c["name"] for c in payload["companies"]}
    missing = [t for t in ALL_TICKERS if t not in names]
    if missing:
        raise ValueError(f"tickers missing from input data: {missing}")
    return {t: names[t] for t in ALL_TICKERS}


def phase1_named_margins(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    margins = {
        ticker: float(info["named_margin_median"])
        for ticker, info in payload["per_company"].items()
    }
    missing = [t for t in ALL_TICKERS if t not in margins]
    if missing:
        raise ValueError(f"Phase 1 margins missing tickers: {missing}")
    return margins


def build_prepared_rows(names: dict[str, str], tokenizer: Any, *, smoke: bool = False) -> list[dict]:
    # Smoke uses 4 tickers x 1 variant so downstream 2B/2C smokes have
    # distinct top/bottom companies.
    tickers = ALL_TICKERS[:4] if smoke else ALL_TICKERS
    orders = (0,) if smoke else (0, 1)
    reverses = (False,) if smoke else REVERSE_OPTIONS
    rows = []
    for ticker in tickers:
        for order in orders:
            for reverse in reverses:
                rows.append(
                    resolve_row(
                        tokenizer, ticker, names[ticker], SECTOR_OF[ticker],
                        reverse=reverse, order=order, format_fn=format_prompt,
                    )
                )
    expected = len(tickers) * len(orders) * len(reverses)
    if len(rows) != expected:
        raise ValueError(f"prepared {len(rows)} rows, expected {expected}")
    return rows


def prepare_stage(run: ArtifactRun, names: dict[str, str], tokenizer: Any, *, smoke: bool = False) -> Path:
    rows = build_prepared_rows(names, tokenizer, smoke=smoke)
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prompts.jsonl"
    with run.stage("prepare") as stage:
        count = write_jsonl(path, rows, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2_prepare",
                "n_rows": count,
                "tickers": ALL_TICKERS,
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="balanced_evidence_gap_phase2_prepare", stage="prepare", role="output", record_count=count)
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2_prepare_metadata", stage="prepare", role="output")
        stage.count(count)
    return path


def forward_stage(run: ArtifactRun, model_path: str, *, smoke: bool = False) -> Path:
    prompts = read_jsonl(run.run_directory / "prepare" / "prompts.jsonl")
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    with run.stage("forward") as stage:
        model, tokenizer, device = load_model(model_path, dtype=None)
        results = []
        for index, row in enumerate(prompts):
            scoring_text = row["formatted"] + DECISION_PREFIX
            margin = clean_margin(model, tokenizer, scoring_text, device=device)
            if smoke:
                dial_entity = None
                dial_final = None
            else:
                scoring_tensor = torch.tensor(
                    [input_ids(tokenizer, scoring_text)], dtype=torch.long, device=device
                )
                dial_entity = capture_mlp_channel(
                    model, scoring_tensor, DIAL_LAYER, DIAL_NEURON, row["entity_position"]
                )
                dial_final = capture_mlp_channel(
                    model, scoring_tensor, DIAL_LAYER, DIAL_NEURON, row["final_position"]
                )
            results.append(
                {k: v for k, v in row.items() if k not in ("prompt_ids",)}
                | {
                    "margin": margin,
                    "decision": "buy" if margin > 0 else "sell",
                    "dial_channel_entity": dial_entity,
                    "dial_channel_final": dial_final,
                }
            )
            if (index + 1) % 16 == 0:
                print(f"  forward {index + 1}/{len(prompts)}", flush=True)
        count = write_jsonl(path, results, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2_forward",
                "n_rows": count,
                "dial_coordinate": [DIAL_LAYER, DIAL_NEURON],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="balanced_evidence_gap_phase2_forward", stage="forward", role="output", record_count=count)
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2_forward_metadata", stage="forward", role="output")
        stage.count(count)
    return path


def analyze_stage(run: ArtifactRun, *, phase1_summary: Path) -> Path:
    results = read_jsonl(run.run_directory / "forward" / "results.jsonl")
    by_ticker: dict[str, list[dict]] = {}
    for row in results:
        by_ticker.setdefault(row["ticker"], []).append(row)

    pure_entity_margins: dict[str, float] = {}
    per_variant: dict[str, list[float]] = {}
    framing_deltas: list[float] = []
    for ticker in sorted(by_ticker):
        rows = by_ticker[ticker]
        margins = [float(r["margin"]) for r in rows]
        pure_entity_margins[ticker] = statistics.median(margins)
        per_variant[ticker] = margins
        # reverse pair delta: same ticker, same order, reverse flipped.
        for order in {int(r["order"]) for r in rows}:
            by_reverse = {
                bool(r["reverse"]): float(r["margin"])
                for r in rows if int(r["order"]) == order
            }
            if True in by_reverse and False in by_reverse:
                framing_deltas.append(by_reverse[True] - by_reverse[False])

    phase1 = phase1_named_margins(phase1_summary)
    gate = evaluate_gate_2a(
        pure_entity_margins=pure_entity_margins,
        phase1_named_margins=phase1,
        framing_pair_deltas=framing_deltas,
        valid_rate=1.0,
    )

    # H4: dial activation vs pure entity margin (descriptive, not gated).
    dial_entity = {
        t: statistics.median(float(r["dial_channel_entity"]) for r in by_ticker[t])
        for t in sorted(by_ticker)
    }
    dial_final = {
        t: statistics.median(float(r["dial_channel_final"]) for r in by_ticker[t])
        for t in sorted(by_ticker)
    }
    tickers_sorted = sorted(pure_entity_margins)
    h4 = {
        "coordinate": [DIAL_LAYER, DIAL_NEURON],
        "entity_position_pearson": spearman(
            [dial_entity[t] for t in tickers_sorted],
            [pure_entity_margins[t] for t in tickers_sorted],
        ) if len(tickers_sorted) >= 3 else None,
        "final_position_pearson": spearman(
            [dial_final[t] for t in tickers_sorted],
            [pure_entity_margins[t] for t in tickers_sorted],
        ) if len(tickers_sorted) >= 3 else None,
        "descriptive_only": True,
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "balanced_evidence_gap_phase2_analysis",
        "n_prompts": len(results),
        "pure_entity_margin_median": pure_entity_margins,
        "iqr": gate["criteria"]["iqr"]["value"],
        "spearman_vs_phase1": gate["criteria"]["spearman_vs_phase1"]["value"],
        "margin_ci_95": list(bootstrap_ci(
            [pure_entity_margins[t] for t in tickers_sorted],
            n=GATE_2A["bootstrap_samples"],
            seed=GATE_2A["bootstrap_seed"],
        )),
        "h4_dial": h4,
        "gate_2a": gate,
        "per_ticker_variants": per_variant,
        "descriptive_only": not gate["pass"],
        "raw_runtime_payloads": False,
    }
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    with run.stage("analyze") as stage:
        write_json(path, summary, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2_analysis_metadata",
                "gate_2a_pass": gate["pass"],
                "phase2b_authorized": gate["phase2b_authorized"],
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="balanced_evidence_gap_phase2_analysis", stage="analyze", role="output")
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2_analysis_metadata", stage="analyze", role="output")
        stage.count(len(pure_entity_margins))
    return path


def run_phase2a(
    *,
    model_path: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    input_data: Path = INPUT_DATA_PATH,
    phase1_summary: Path = PHASE1_SUMMARY_PATH,
    smoke: bool = False,
) -> Path:
    names = company_names(input_data)
    tokenizer = load_tokenizer(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        prepare_stage(run, names, tokenizer, smoke=smoke)
        forward_stage(run, model_path, smoke=smoke)
        if smoke:
            run.finalize(required_stages={"prepare", "forward"})
        else:
            analyze_stage(run, phase1_summary=phase1_summary)
            run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
