"""Orchestrate an atomic, artifact-only visualization bundle."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import file_sha256

from .contract import SUMMARY_FILES, VISUALIZATION_SCHEMA_VERSION
from .dashboard import write_dashboard
from .plots import make_plots
from .reader import validate_run
from .summaries import summarize_all


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        raise ValueError(f"refusing to write empty summary: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent summary schema: {path.name}")
    forbidden = ("activation", "residual", "gradient", "hidden_state")
    if any(any(word in field.lower() for word in forbidden) for field in fields):
        raise ValueError(f"forbidden summary field: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _output_record(path: Path, root: Path, record_count: int | None = None) -> dict[str, Any]:
    item = {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if record_count is not None:
        item["record_count"] = record_count
    return item


def visualize_run(
    run_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    replace_existing: bool = False,
) -> Path:
    run = validate_run(run_root)
    destination = Path(output_dir).resolve() if output_dir else run.root / "visualization"
    if destination == run.root or run.root not in destination.parents:
        if output_dir is None:
            raise ValueError("visualization output must be below the run root")
    if destination.exists() and any(destination.iterdir()):
        if not replace_existing:
            raise FileExistsError(f"visualization output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        summaries = summarize_all(run)
        outputs = []
        for name, filename in SUMMARY_FILES.items():
            path = staging / filename
            count = _write_csv(path, summaries[name])
            outputs.append(_output_record(path, staging, count))
        plots, plot_metadata = make_plots(run, staging)
        for formats in plots.values():
            outputs.extend(_output_record(path, staging) for path in formats.values())
        dashboard = write_dashboard(run, staging / "dashboard.html", plot_names=list(plots))
        outputs.append(_output_record(dashboard, staging))
        manifest_hash = file_sha256(run.root / "manifest.json")
        metadata = {
            "schema_version": VISUALIZATION_SCHEMA_VERSION,
            "visualization_type": "synthetic_entity_bias_single_run",
            "source_run": {
                "run_root": run.root.as_posix(),
                "model": run.manifest["model"],
                "dataset": run.manifest["dataset"],
                "run_id": run.manifest["run_id"],
                "manifest_sha256": manifest_hash,
                "template_hash": run.config["template_hash"],
                "label_hash": run.config["label_hash"],
                "lens_binary_sha256": run.config["lens_binary_sha256"],
            },
            "source_artifacts": list(run.source_artifacts),
            "validation": {
                "status": "passed",
                "checks": ["manifest", "stages", "paths", "sha256", "schemas", "counts", "probabilities", "numeric_domains", "cross_file_identity"],
                "model_loading_performed": False,
            },
            "records": {
                "entity_pool": len(run.entity_pool),
                "no_entity_baselines": len(run.baselines),
                "entity_template_results": len(run.results),
                "localization": len(run.localization),
            },
            "aggregation": {
                "quantiles": "linear interpolation over sorted values",
                "sector_membership": "pipe-delimited sectors are exploded; missing sectors are Unknown",
                **plot_metadata,
            },
            "interpretation": {
                "delta_expected_score": "restricted-nine-label entity expected score minus matched no-entity baseline expected score",
                "localization": "Jacobian-transported representation statistics; not chain-of-thought or standalone causal proof",
                "membership_years": "source provenance, not independently verified historical membership evidence",
                "lens_source": "lens calibration source remains an experimental condition",
            },
            "outputs": sorted(outputs, key=lambda item: item["path"]),
        }
        metadata_path = staging / "visualization_metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
