"""Phase 2 Rev 2 gate re-evaluation (docs/balanced-evidence-gap/proposal-phase2-rev2.md).

CPU-only re-analysis of an existing 2A forward run: no model load, no new
inference. The 2A margin is a deterministic logit computation, so the gate
is evaluated from the stored forward records with provenance to the source
run. Stages: prepare (provenance) → analyze (gate 2A Rev 2 + descriptive).
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun

from .analysis import (
    bootstrap_ci,
    evaluate_gate_2a_rev2,
    select_margin_groups,
    spearman,
)
from .pipeline import PHASE1_SUMMARY_PATH, phase1_named_margins
from .template import DATASET, SECTOR_OF, SCHEMA_VERSION

DEFAULT_PHASE2A_RUN = Path(
    "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_variant_margins(rows: list[dict]) -> tuple[dict[str, float], list[float]]:
    """Per-ticker median margin over variants + reverse-pair framing deltas."""
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)
    pure: dict[str, float] = {}
    framing: list[float] = []
    for ticker, ticker_rows in by_ticker.items():
        margins = [float(r["margin"]) for r in ticker_rows]
        pure[ticker] = statistics.median(margins)
        for order in {int(r["order"]) for r in ticker_rows}:
            by_reverse = {
                bool(r["reverse"]): float(r["margin"])
                for r in ticker_rows if int(r["order"]) == order
            }
            if True in by_reverse and False in by_reverse:
                framing.append(by_reverse[True] - by_reverse[False])
    return pure, framing


def phase1_gaps(summary_path: Path) -> dict[str, float]:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return {t: float(v["gap_mean"]) for t, v in payload["per_company"].items()}


def run_rev2_gate(
    *,
    model_name: str,
    phase2a_run: str | Path,
    phase1_summary: str | Path,
    run_id: str,
    artifact_root: str | Path = "artifacts",
) -> Path:
    phase2a_run = Path(phase2a_run)
    phase1_summary = Path(phase1_summary)
    results_path = phase2a_run / "forward" / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"2A forward records not found: {results_path}")
    if not phase1_summary.is_file():
        raise FileNotFoundError(f"Phase 1 summary not found: {phase1_summary}")
    rows = read_jsonl(results_path)

    run = ArtifactRun.create(model_name, DATASET, run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            provenance = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2_rev2_provenance",
                "reanalysis_only": True,
                "raw_runtime_payloads": False,
                "phase2a_run": {
                    "run_id": phase2a_run.name,
                    "root": str(phase2a_run),
                    "results_path": str(results_path),
                    "results_sha256": _sha256(results_path),
                    "n_records": len(rows),
                },
                "phase1_summary": {
                    "path": str(phase1_summary),
                    "sha256": _sha256(phase1_summary),
                },
                "gate_protocol": "docs/balanced-evidence-gap/proposal-phase2-rev2.md",
            }
            prov_path = run.run_directory / "prepare" / "provenance.json"
            write_json(prov_path, provenance)
            run.manifest.register_artifact(
                prov_path, artifact_type="balanced_evidence_gap_phase2_rev2_provenance", stage="prepare", role="output"
            )
            write_metadata(
                run.run_directory / "prepare" / "metadata.json",
                {"artifact_type": "balanced_evidence_gap_phase2_rev2_prepare", "n_records": len(rows)},
            )
            run.manifest.register_artifact(
                run.run_directory / "prepare" / "metadata.json",
                artifact_type="balanced_evidence_gap_phase2_rev2_prepare_metadata",
                stage="prepare", role="output",
            )
            stage.count(len(rows))

        with run.stage("analyze") as stage:
            pure, framing = summarize_variant_margins(rows)
            gaps = phase1_gaps(phase1_summary)
            named = phase1_named_margins(phase1_summary)
            gate = evaluate_gate_2a_rev2(
                pure_entity_margins=pure,
                phase1_gaps=gaps,
                framing_pair_deltas=framing,
                valid_rate=1.0,
            )
            tickers_sorted = sorted(pure)
            by_sector: dict[str, list[float]] = {}
            for t in tickers_sorted:
                by_sector.setdefault(SECTOR_OF[t], []).append(pure[t])
            bottom, top = select_margin_groups(pure)
            summary = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2_rev2_analysis",
                "protocol": "docs/balanced-evidence-gap/proposal-phase2-rev2.md",
                "n_prompts": len(rows),
                "gate_2a_rev2": gate,
                "descriptive": {
                    "spearman_vs_phase1_named": spearman(
                        [pure[t] for t in tickers_sorted],
                        [named[t] for t in tickers_sorted],
                    ),
                    "spearman_vs_phase1_gap": gate["criteria"]["spearman_vs_phase1_gap"]["value"],
                    "margin_ci_95": list(bootstrap_ci(
                        [pure[t] for t in tickers_sorted], n=2000, seed=42
                    )),
                    "sector_pure_margin_mean": {
                        s: statistics.fmean(v) for s, v in sorted(by_sector.items())
                    },
                    "groups": {"top": list(top), "bottom": list(bottom)},
                },
                "pure_entity_margin_median": pure,
                "iqr": gate["criteria"]["iqr"]["value"],
            }
            summary_path = run.run_directory / "analyze" / "summary.json"
            write_json(summary_path, summary)
            run.manifest.register_artifact(
                summary_path, artifact_type="balanced_evidence_gap_phase2_rev2_analysis", stage="analyze", role="output"
            )
            write_metadata(
                run.run_directory / "analyze" / "metadata.json",
                {"artifact_type": "balanced_evidence_gap_phase2_rev2_analyze", "n_prompts": len(rows)},
            )
            run.manifest.register_artifact(
                run.run_directory / "analyze" / "metadata.json",
                artifact_type="balanced_evidence_gap_phase2_rev2_analyze_metadata",
                stage="analyze", role="output",
            )
            stage.count(1)

        run.finalize(required_stages={"prepare", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
