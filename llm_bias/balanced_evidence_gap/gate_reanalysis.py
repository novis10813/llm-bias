"""Phase 2C gate re-analysis after the gate-aggregation fix.

CPU-only re-evaluation of a completed 2C run's persisted records under the
frozen protocol's per-arm existence semantics (proposal §4.5: "至少 1 個
head 或 1 個神經元"). The original analyze stage aggregated the MLP arm
across layers (max top / max control / min sector-agreement / max
sign-flip-p), which mixed statistics from different layers and let the
structurally zero final layer (L31: no downstream reader of the entity
position after the final block) contaminate the verdict. No GPU, no new
inference: all inputs are the source run's persisted attention records and
MLP layer summaries, with SHA-256 provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun

from .patch_pipeline import analyze_2c_records
from .template import DATASET, SCHEMA_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_2c_gate_reanalysis(
    *,
    model_name: str,
    phase2c_run: str | Path,
    run_id: str,
    artifact_root: str | Path = "artifacts",
) -> Path:
    phase2c_run = Path(phase2c_run)
    att_path = phase2c_run / "attention" / "records.jsonl"
    mlp_path = phase2c_run / "mlp" / "layer_summaries.json"
    src_summary_path = phase2c_run / "analyze" / "summary.json"
    for p in (att_path, mlp_path, src_summary_path):
        if not p.is_file():
            raise FileNotFoundError(f"2C run input not found: {p}")

    source_summary = json.loads(src_summary_path.read_text(encoding="utf-8"))
    attention_layers = [int(l) for l in source_summary["attention_layers"]]
    mlp_layers = [int(l) for l in source_summary["mlp_layers"]]
    attention_records = read_jsonl(att_path)
    mlp_layer_summaries = json.loads(mlp_path.read_text(encoding="utf-8"))["layers"]

    run = ArtifactRun.create(model_name, DATASET, run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            provenance = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "balanced_evidence_gap_phase2c_gate_reanalysis_provenance",
                "reanalysis_only": True,
                "raw_runtime_payloads": False,
                "reason": (
                    "gate 2C MLP arm was aggregated across layers (max top / max "
                    "control / min sector-agreement / max sign-flip-p), mixing "
                    "statistics from different layers; frozen protocol §4.5 "
                    "defines a per-arm existence test (>=1 head or >=1 neuron)"
                ),
                "source_run": {
                    "run_id": phase2c_run.name,
                    "root": str(phase2c_run),
                    "attention_records_path": str(att_path),
                    "attention_records_sha256": _sha256(att_path),
                    "attention_n_records": len(attention_records),
                    "mlp_layer_summaries_path": str(mlp_path),
                    "mlp_layer_summaries_sha256": _sha256(mlp_path),
                    "mlp_n_layers": len(mlp_layer_summaries),
                },
                "protocol": "docs/balanced-evidence-gap/proposal-phase2.md §4.5",
            }
            prov_path = run.run_directory / "prepare" / "provenance.json"
            write_json(prov_path, provenance)
            run.manifest.register_artifact(
                prov_path, artifact_type="balanced_evidence_gap_phase2c_gate_reanalysis_provenance",
                stage="prepare", role="output",
            )
            write_metadata(
                run.run_directory / "prepare" / "metadata.json",
                {"artifact_type": "balanced_evidence_gap_phase2c_gate_reanalysis_prepare",
                 "attention_n_records": len(attention_records)},
            )
            run.manifest.register_artifact(
                run.run_directory / "prepare" / "metadata.json",
                artifact_type="balanced_evidence_gap_phase2c_gate_reanalysis_prepare_metadata",
                stage="prepare", role="output",
            )
            stage.count(len(attention_records))

        with run.stage("analyze") as stage:
            summary = analyze_2c_records(
                attention_records, mlp_layer_summaries,
                attention_layers=attention_layers, mlp_layers=mlp_layers,
            )
            summary["gate_semantics_fix"] = provenance["reason"]
            summary_path = run.run_directory / "analyze" / "summary.json"
            write_json(summary_path, summary)
            run.manifest.register_artifact(
                summary_path, artifact_type="balanced_evidence_gap_phase2c_gate_reanalysis",
                stage="analyze", role="output",
            )
            write_metadata(
                run.run_directory / "analyze" / "metadata.json",
                {"artifact_type": "balanced_evidence_gap_phase2c_gate_reanalysis_analyze",
                 "n_records": len(attention_records)},
            )
            run.manifest.register_artifact(
                run.run_directory / "analyze" / "metadata.json",
                artifact_type="balanced_evidence_gap_phase2c_gate_reanalysis_analyze_metadata",
                stage="analyze", role="output",
            )
            stage.count(1)

        run.finalize(required_stages={"prepare", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
