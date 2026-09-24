"""Phase 2A pipeline: cross-entity probe with H4 dial readout.

Stages: prepare → forward → analyze. Reuses the shared ArtifactRun
manifest lifecycle and core prompt/scoring mechanics. The frozen protocol
is docs/balanced-evidence-gap/details/proposal-phase2.md.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.model import load_model, load_tokenizer_for_inference
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids

from .analysis import (
    GATE_2A,
    bootstrap_ci,
    evaluate_gate_2a,
    evaluate_gate_2a_reference_free,
    spearman,
)
from .intervention import answer_token_ids, capture_mlp_channel, clean_margin
from .spans import resolve_row, resolve_row_v2
from .template import (
    ALL_TICKERS,
    DIAL_LAYER,
    DIAL_NEURON,
    DATASET,
    DECISION_PREFIX,
    REVERSE_OPTIONS,
    SCHEMA_VERSION,
    SECTOR_OF,
    V2_CONDITIONS,
    V2_SCHEMA_VERSION,
    variant_id,
)

INPUT_DATA_PATH = Path("data/baseline/investment-dial/exploratory-v1.json")
PHASE1_SUMMARY_PATH = Path(
    "artifacts/qwen3.5-4b/balanced-evidence-gap/runs/balanced-gap-gpu-bf16-01/analyze/summary.json"
)


def company_names(data_path: Path, tickers: Sequence[str] | None = None) -> dict[str, str]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    names = {c["ticker"]: c["name"] for c in payload["companies"]}
    wanted = list(tickers) if tickers is not None else ALL_TICKERS
    missing = [t for t in wanted if t not in names]
    if missing:
        raise ValueError(f"tickers missing from input data: {missing[:5]}")
    return {t: names[t] for t in wanted}


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


def build_prepared_rows(
    names: dict[str, str],
    tokenizer: Any,
    *,
    smoke: bool = False,
    tickers: Sequence[str] | None = None,
) -> list[dict]:
    # Smoke uses 4 tickers x 1 variant so downstream 2B/2C smokes have
    # distinct top/bottom companies.
    if smoke:
        tickers = list(tickers or ALL_TICKERS)[:4]
    else:
        tickers = list(tickers or ALL_TICKERS)
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


def build_prepared_rows_v2(
    names: dict[str, str],
    tokenizer: Any,
    tickers: Sequence[str] | None = None,
    *,
    smoke: bool = False,
) -> list[dict]:
    """v2 rows: every ticker × {pos, neg} condition × reverse variants.

    The sentence-order axis does not exist (a same-sign pair has one
    canonical order); smoke keeps one reverse per condition.
    """
    tickers = list(tickers or ALL_TICKERS)
    if smoke:
        tickers = tickers[:4]
    reverses = (False,) if smoke else REVERSE_OPTIONS
    rows = []
    for ticker in tickers:
        for condition in V2_CONDITIONS:
            for reverse in reverses:
                rows.append(
                    resolve_row_v2(
                        tokenizer, ticker, names[ticker], SECTOR_OF.get(ticker, ""),
                        condition=condition, reverse=reverse, format_fn=format_prompt,
                    )
                )
    expected = len(tickers) * len(V2_CONDITIONS) * len(reverses)
    if len(rows) != expected:
        raise ValueError(f"prepared {len(rows)} v2 rows, expected {expected}")
    return rows


def prepare_stage(
    run: ArtifactRun,
    names: dict[str, str],
    tokenizer: Any,
    *,
    smoke: bool = False,
    tickers: Sequence[str] | None = None,
    family: str = "v1",
) -> Path:
    if family == "v2":
        rows = build_prepared_rows_v2(names, tokenizer, tickers, smoke=smoke)
    else:
        rows = build_prepared_rows(names, tokenizer, smoke=smoke, tickers=tickers)
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prompts.jsonl"
    with run.stage("prepare") as stage:
        count = write_jsonl(path, rows, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": V2_SCHEMA_VERSION if family == "v2" else SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2_prepare",
                "n_rows": count,
                "family": family,
                "tickers": [row["ticker"] for row in rows],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="balanced_evidence_gap_phase2_prepare", stage="prepare", role="output", record_count=count)
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2_prepare_metadata", stage="prepare", role="output")
        stage.count(count)
    return path


def forward_stage(run: ArtifactRun, model_path: str, *, smoke: bool = False, no_dial: bool = False, device_map: str | None = None, dtype: torch.dtype | str | None = None) -> Path:
    prompts = read_jsonl(run.run_directory / "prepare" / "prompts.jsonl")
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    with run.stage("forward") as stage:
        model, tokenizer, device = load_model(model_path, dtype=dtype, device_map=device_map)
        results = []
        for index, row in enumerate(prompts):
            scoring_text = row["formatted"] + DECISION_PREFIX
            margin = clean_margin(model, tokenizer, scoring_text, device=device)
            if smoke or no_dial:
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
                "dial_coordinate": None if no_dial else [DIAL_LAYER, DIAL_NEURON],
                "smoke": smoke,
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="balanced_evidence_gap_phase2_forward", stage="forward", role="output", record_count=count)
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2_forward_metadata", stage="forward", role="output")
        stage.count(count)
    return path


def analyze_stage(run: ArtifactRun, *, phase1_summary: Path | None, no_dial: bool = False, family: str = "v1") -> Path:
    results = read_jsonl(run.run_directory / "forward" / "results.jsonl")
    if family == "v2":
        return analyze_stage_v2(run, results)
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

    if phase1_summary is None:
        gate = evaluate_gate_2a_reference_free(
            pure_entity_margins=pure_entity_margins,
            framing_pair_deltas=framing_deltas,
            valid_rate=1.0,
        )
    else:
        phase1 = phase1_named_margins(phase1_summary)
        gate = evaluate_gate_2a(
            pure_entity_margins=pure_entity_margins,
            phase1_named_margins=phase1,
            framing_pair_deltas=framing_deltas,
            valid_rate=1.0,
        )

    tickers_sorted = sorted(pure_entity_margins)
    # H4: dial activation vs pure entity margin (descriptive, not gated).
    if no_dial or all(r["dial_channel_entity"] is None for r in results):
        h4 = {"skipped": "no_dial", "descriptive_only": True}
    else:
        dial_entity = {
            t: statistics.median(float(r["dial_channel_entity"]) for r in by_ticker[t])
            for t in sorted(by_ticker)
        }
        dial_final = {
            t: statistics.median(float(r["dial_channel_final"]) for r in by_ticker[t])
            for t in sorted(by_ticker)
        }
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
        "phase1_comparison": "skipped" if phase1_summary is None else "protocol_default",
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


def analyze_stage_v2(run: ArtifactRun, results: list[dict]) -> Path:
    """v2 analysis: per-company margins per condition + condition-flip gate.

    Development semantics (proposal-phase2-v2.md): the gate runs reference-free
    on the per-company CONDITION DIFFERENCE ``M_pos − M_neg`` (the spread of
    how strongly the two two-sentence conditions pull the decision apart),
    with framing stability over reverse-pair deltas within each condition.
    """
    rev_margin: dict[str, dict[str, dict[bool, float]]] = {}
    for row in results:
        rev_margin.setdefault(row["ticker"], {}).setdefault(row["condition"], {})[
            bool(row["reverse"])
        ] = float(row["margin"])

    margin_pos: dict[str, float] = {}
    margin_neg: dict[str, float] = {}
    diff: dict[str, float] = {}
    framing_deltas: list[float] = []
    per_condition: dict[str, dict[str, list[float]]] = {}
    for ticker in sorted(rev_margin):
        conds = rev_margin[ticker]
        missing = [c for c in V2_CONDITIONS if c not in conds]
        if missing:
            raise ValueError(f"v2 results missing conditions for {ticker}: {missing}")
        margin_pos[ticker] = statistics.median(conds["pos"].values())
        margin_neg[ticker] = statistics.median(conds["neg"].values())
        diff[ticker] = margin_pos[ticker] - margin_neg[ticker]
        per_condition[ticker] = {c: sorted(conds[c].values()) for c in V2_CONDITIONS}
        for cond in V2_CONDITIONS:
            if True in conds[cond] and False in conds[cond]:
                framing_deltas.append(conds[cond][True] - conds[cond][False])

    gate = evaluate_gate_2a_reference_free(
        pure_entity_margins=diff,
        framing_pair_deltas=framing_deltas,
        valid_rate=1.0,
    )
    tickers_sorted = sorted(diff)
    summary = {
        "schema_version": V2_SCHEMA_VERSION,
        "artifact_type": "balanced_evidence_gap_phase2_analysis",
        "family": "v2",
        "conditions": list(V2_CONDITIONS),
        "universe": "investment-dial",
        "n_tickers": len(tickers_sorted),
        "n_prompts": len(results),
        "margin_pos_median": margin_pos,
        "margin_neg_median": margin_neg,
        # M_pos − M_neg per company; the reference-free gate's IQR runs on
        # this difference (v2 development semantics, see proposal-phase2-v2.md).
        "condition_diff_median": diff,
        "pure_entity_margin_median": diff,
        "iqr": gate["criteria"]["iqr"]["value"],
        "spearman_vs_phase1": None,
        "margin_ci_95": list(bootstrap_ci(
            [diff[t] for t in tickers_sorted],
            n=GATE_2A["bootstrap_samples"],
            seed=GATE_2A["bootstrap_seed"],
        )),
        "gate_2a": gate,
        "per_ticker_variants": per_condition,
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
                "schema_version": V2_SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2_analysis_metadata",
                "family": "v2",
                "gate_2a_pass": gate["pass"],
                "phase2b_authorized": gate["phase2b_authorized"],
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="balanced_evidence_gap_phase2_analysis", stage="analyze", role="output")
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="balanced_evidence_gap_phase2_analysis_metadata", stage="analyze", role="output")
        stage.count(len(diff))
    return path


def run_phase2a(
    *,
    model_path: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    input_data: Path = INPUT_DATA_PATH,
    phase1_summary: Path | None = PHASE1_SUMMARY_PATH,
    smoke: bool = False,
    no_dial: bool = False,
    device_map: str | None = None,
    dtype: torch.dtype | str | None = None,
    tickers: Sequence[str] | None = None,
    family: str = "v1",
) -> Path:
    """Run the 2A cross-entity probe.

    ``phase1_summary=None`` skips the Qwen3.5-4B Phase 1 Spearman reference
    (cross-model runs); the gate then covers IQR, framing stability, and
    schema validity only, and the cross-model Spearman is not evaluated.
    ``no_dial=True`` skips the Qwen-specific L15/n8490 H4 readout.
    ``device_map`` passes through to ``load_model`` (multi-GPU sharded runs).
    ``dtype`` passes through to ``load_model`` (e.g. ``"native"`` keeps a
    checkpoint's stored dtypes, such as gpt-oss-20b's packed MXFP4 experts).
    ``tickers=None`` uses the frozen 16-ticker v1 universe; passing an explicit
    list (e.g. the full investment-dial 427-company universe) drives a v2
    development run against the same shared-evidence template.
    ``family="v2"`` runs the two-sentence condition family (proposal-phase2-v2.md);
    the Qwen-specific dial readout and the Phase 1 Spearman reference do not
    apply and are force-disabled.

    Prepare uses the inference-time tokenizer (BOS-forced, matching
    ``load_model``) so token spans index the same sequences the forward pass
    consumes.
    """
    if family == "v2":
        no_dial = True
        phase1_summary = None
    names = company_names(input_data, tickers)
    tokenizer = load_tokenizer_for_inference(model_path)
    run = ArtifactRun.create(Path(model_path).name, DATASET, run_id, artifact_root=artifact_root)
    try:
        prepare_stage(run, names, tokenizer, smoke=smoke, tickers=tickers, family=family)
        forward_stage(run, model_path, smoke=smoke, no_dial=no_dial, device_map=device_map, dtype=dtype)
        if smoke:
            run.finalize(required_stages={"prepare", "forward"})
        else:
            analyze_stage(run, phase1_summary=phase1_summary, no_dial=no_dial, family=family)
            run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
