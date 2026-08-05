"""Visualizations for five-class 8-K return predictions and readout uncertainty.

This module deliberately owns a small JSONL artifact contract and does not depend
on the numeric-price visualizer. Pairing is always by ``pair_id``.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from llm_bias.prompt_analysis.artifact_io import read_jsonl, sha256_file
from llm_bias.prompt_analysis.return_evaluation import (
    CONDITIONS,
    FORWARD_ARTIFACT_TYPE,
    LABELS,
    parse_return_prediction,
)

FORWARD_FIELDS = {
    "pair_id", "condition", "target_label", "generated_text",
    "ticker", "peer_ticker", "filing_date",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {source}:{number}")
            rows.append(row)
    return rows


def _validate_forward(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize forward generated outputs and derive prediction fields."""
    result = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        missing = FORWARD_FIELDS - row.keys()
        if missing:
            raise ValueError(f"forward record missing fields: {sorted(missing)}")
        pair_id, condition = str(row["pair_id"]), str(row["condition"])
        if not pair_id or condition not in CONDITIONS:
            raise ValueError(f"invalid pair_id/condition: {pair_id!r}/{condition!r}")
        key = (pair_id, condition)
        if key in seen:
            raise ValueError(f"duplicate forward record for {pair_id}/{condition}")
        seen.add(key)
        if row["target_label"] not in LABELS:
            raise ValueError(f"unknown target_label for {pair_id}: {row['target_label']!r}")
        normalized = dict(row)
        normalized.update({"pair_id": pair_id, "condition": condition})
        normalized.update(parse_return_prediction(row.get("generated_text")))
        result.append(normalized)
    if not result:
        raise ValueError("forward artifact is empty")
    return result


def _validate_attribution(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate already-normalized forward records for plotting."""
    return _validate_forward(rows)


def _final_uncertainty(row: dict[str, Any]) -> dict[str, float | int | None]:
    layers = row.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"uncertainty record {row.get('pair_id')} has no layers list")
    outputs = [layer for layer in layers if isinstance(layer, dict) and layer.get("is_output") is True]
    if len(outputs) != 1:
        raise ValueError(f"uncertainty record {row.get('pair_id')} must have exactly one output layer")
    layer = outputs[0]
    try:
        entropy = float(layer["entropy_nats"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"uncertainty output missing entropy_nats for {row.get('pair_id')}") from exc
    if "effective_temperature" in layer:
        temperature = float(layer["effective_temperature"])
    elif "effective_inverse_temperature" in layer:
        inverse = float(layer["effective_inverse_temperature"])
        temperature = 1.0 / inverse if inverse else float("nan")
    else:
        raise ValueError(f"uncertainty output missing effective temperature for {row.get('pair_id')}")
    if not math.isfinite(entropy) or entropy < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"uncertainty metrics must be finite and positive for {row.get('pair_id')}")
    return {"layer": layer.get("layer"), "entropy_nats": entropy, "effective_temperature": temperature}


def _validate_uncertainty(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        for field in ("pair_id", "condition", "layers"):
            if field not in row:
                raise ValueError(f"uncertainty record missing field: {field}")
        pair_id, condition = str(row["pair_id"]), str(row["condition"])
        if not pair_id or condition not in CONDITIONS:
            raise ValueError(f"invalid uncertainty pair_id/condition: {pair_id!r}/{condition!r}")
        key = (pair_id, condition)
        if key in seen:
            raise ValueError(f"duplicate uncertainty record for {pair_id}/{condition}")
        seen.add(key)
        result.append({**row, "pair_id": pair_id, "condition": condition, **_final_uncertainty(row)})
    if not result:
        raise ValueError("uncertainty artifact is empty")
    return result


def _pair_rows(rows: Iterable[dict[str, Any]], *, value_fields: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[str(row["pair_id"])][str(row["condition"])] = row
    complete, missing = [], Counter()
    for pair_id, conditions in groups.items():
        if not all(condition in conditions for condition in CONDITIONS):
            missing["missing_condition"] += 1
            continue
        original, counterfactual = conditions["original"], conditions["counterfactual"]
        complete.append({
            "pair_id": pair_id,
            "filing_date": original.get("filing_date", counterfactual.get("filing_date")),
            **{f"original_{field}": original.get(field) for field in value_fields},
            **{f"counterfactual_{field}": counterfactual.get(field) for field in value_fields},
        })
    return complete, dict(missing)


def build_return_prediction_rows(attribution_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return validated compact classification records, retaining invalid rows."""
    return _validate_attribution(attribution_rows)


def build_prediction_flip_rows(attribution_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _validate_attribution(attribution_rows)
    pairs, _ = _pair_rows(rows, value_fields=("target_label", "predicted_label", "parse_status", "predicted_confidence"))
    for row in pairs:
        row["valid_original"] = row["original_parse_status"] == "valid"
        row["valid_counterfactual"] = row["counterfactual_parse_status"] == "valid"
        row["prediction_flip"] = (
            row["valid_original"] and row["valid_counterfactual"]
            and row["original_predicted_label"] != row["counterfactual_predicted_label"]
        )
    return pairs


def build_paired_uncertainty_delta_rows(uncertainty_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _validate_uncertainty(uncertainty_rows)
    pairs, _ = _pair_rows(rows, value_fields=("layer", "entropy_nats", "effective_temperature"))
    for row in pairs:
        if row["original_layer"] != row["counterfactual_layer"]:
            raise ValueError(f"paired output layers differ for {row['pair_id']}")
        row["entropy_delta_nats"] = row["counterfactual_entropy_nats"] - row["original_entropy_nats"]
        row["effective_temperature_delta"] = row["counterfactual_effective_temperature"] - row["original_effective_temperature"]
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _style(axis: Any) -> None:
    axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def _confusion(axis: Any, rows: list[dict[str, Any]], condition: str, labels: list[str]) -> None:
    valid = [r for r in rows if r["condition"] == condition and r["parse_status"] == "valid"]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    positions = {label: i for i, label in enumerate(labels)}
    for row in valid:
        if row["target_label"] in positions and row["predicted_label"] in positions:
            matrix[positions[row["target_label"]], positions[row["predicted_label"]]] += 1
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#0b0b0b")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Target label")
    axis.set_title(f"{condition.title()} · valid predictions (n={len(valid)})", loc="left", fontweight="bold")
    return image


def plot_return_prediction_figures(rows: Iterable[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = list(rows)
    labels = list(LABELS)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, condition in zip(axes, CONDITIONS, strict=True):
        _confusion(axis, rows, condition, labels)
    figure.suptitle("8-K return-class prediction confusion matrices", fontsize=15, fontweight="bold")
    path = output_dir / "return_confusion_matrices.png"; figure.savefig(path, dpi=220, facecolor="#fcfcfb"); plt.close(figure); paths.append(path)

    counts = {condition: Counter(r["predicted_label"] for r in rows if r["condition"] == condition and r["parse_status"] == "valid") for condition in CONDITIONS}
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    positions = np.arange(len(labels)); width = 0.36
    for offset, condition, color in ((-width / 2, "original", "#2a78d6"), (width / 2, "counterfactual", "#eb6834")):
        axis.bar(positions + offset, [counts[condition][label] for label in labels], width, label=condition.title(), color=color)
    axis.set_xticks(positions, labels, rotation=35, ha="right"); axis.set_ylabel("Valid predictions"); axis.set_title("Predicted-label distribution", loc="left", fontweight="bold"); _style(axis); axis.legend(frameon=False)
    path = output_dir / "return_label_distribution.png"; figure.savefig(path, dpi=220, facecolor="#fcfcfb"); plt.close(figure); paths.append(path)

    flips = build_prediction_flip_rows(rows)
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    matrix = Counter((r["original_predicted_label"], r["counterfactual_predicted_label"]) for r in flips if r["prediction_flip"])
    names = [f"{a} → {b}" for (a, b) in sorted(matrix)]
    axis.barh(names, [matrix[key] for key in sorted(matrix)], color="#7c3aed"); axis.set_xlabel("Pairs"); axis.set_title(f"Prediction flips (n={sum(matrix.values())})", loc="left", fontweight="bold"); _style(axis)
    path = output_dir / "return_prediction_flips.png"; figure.savefig(path, dpi=220, facecolor="#fcfcfb"); plt.close(figure); paths.append(path)

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    confidence = [[r["predicted_confidence"] for r in rows if r["condition"] == condition and r["parse_status"] == "valid"] for condition in CONDITIONS]
    axis.boxplot(confidence, tick_labels=[condition.title() for condition in CONDITIONS], patch_artist=True, boxprops={"facecolor": "#d9e8f7"}); axis.set_ylabel("Predicted confidence (%)"); axis.set_ylim(0, 102); axis.set_title("Confidence by condition", loc="left", fontweight="bold"); _style(axis)
    path = output_dir / "return_prediction_confidence.png"; figure.savefig(path, dpi=220, facecolor="#fcfcfb"); plt.close(figure); paths.append(path)
    return paths


def plot_paired_uncertainty_delta_figures(rows: Iterable[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    rows = list(rows)
    if not rows:
        raise ValueError("no complete paired uncertainty rows")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for axis, field, label, color in zip(axes, ("entropy_delta_nats", "effective_temperature_delta"), ("Entropy delta (nats)", "Effective-temperature delta"), ("#2a78d6", "#eb6834"), strict=True):
        values = [float(row[field]) for row in rows]
        axis.axhline(0, color="#898781", linewidth=1)
        axis.scatter(np.arange(len(values)), values, color=color, edgecolor="#fcfcfb", linewidth=0.8, s=46)
        axis.set_xlabel("Complete pair (pair_id order)"); axis.set_ylabel("Counterfactual − original"); axis.set_title(label, loc="left", fontweight="bold"); _style(axis)
    figure.suptitle("Paired final-layer uncertainty deltas", fontsize=15, fontweight="bold")
    path = output_dir / "return_paired_uncertainty_delta.png"; figure.savefig(path, dpi=220, facecolor="#fcfcfb"); plt.close(figure)
    return [path]


def visualize_return_predictions(*, attribution_path: str | Path, uncertainty_path: str | Path, output_dir: str | Path) -> Path:
    """Visualize predictions from a forward artifact and final-layer uncertainty."""
    forward_source = Path(attribution_path)
    uncertainty_source = Path(uncertainty_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forward_rows = read_jsonl(forward_source)
    forward_metadata_path = forward_source.parent / "metadata.json"
    forward_metadata = (
        json.loads(forward_metadata_path.read_text(encoding="utf-8"))
        if forward_metadata_path.is_file() else {}
    )
    artifact_type = forward_metadata.get("artifact_type") or next(
        (row.get("artifact_type") for row in forward_rows if row.get("artifact_type")), None
    )
    if artifact_type != FORWARD_ARTIFACT_TYPE:
        raise ValueError("return visualization requires a forward generated-output artifact")
    attribution = build_return_prediction_rows(forward_rows)
    uncertainty = _validate_uncertainty(read_jsonl(uncertainty_source))
    flip_rows = build_prediction_flip_rows(attribution)
    delta_rows = build_paired_uncertainty_delta_rows(uncertainty)
    figure_paths = plot_return_prediction_figures(attribution, output) + plot_paired_uncertainty_delta_figures(delta_rows, output)
    fields = sorted(FORWARD_FIELDS | {"predicted_label", "predicted_confidence", "parse_status"})
    _write_csv(output / "return_prediction_records.csv", attribution, fields)
    _write_csv(output / "return_prediction_flips.csv", flip_rows, ["pair_id", "filing_date", "original_predicted_label", "counterfactual_predicted_label", "original_target_label", "counterfactual_target_label", "original_predicted_confidence", "counterfactual_predicted_confidence", "valid_original", "valid_counterfactual", "prediction_flip"])
    _write_csv(output / "return_paired_uncertainty_delta.csv", delta_rows, ["pair_id", "filing_date", "original_entropy_nats", "counterfactual_entropy_nats", "entropy_delta_nats", "original_effective_temperature", "counterfactual_effective_temperature", "effective_temperature_delta"])
    missing_prediction = _pair_rows(attribution, value_fields=("target_label",))[1]
    missing_uncertainty = _pair_rows(uncertainty, value_fields=("entropy_nats",))[1]
    forward_hash = sha256_file(forward_source)
    uncertainty_hash = sha256_file(uncertainty_source)
    coverage = {
        "forward_records": len(attribution),
        "uncertainty_records": len(uncertainty),
        "prediction_pairs": len(flip_rows),
        "uncertainty_pairs": len(delta_rows),
        "invalid_predictions": sum(r["parse_status"] != "valid" for r in attribution),
    }
    metadata = {
        "artifact_contract": {"source_stage": "forward", "artifact_type": artifact_type, "pairing_key": "pair_id", "forward_required_fields": sorted(FORWARD_FIELDS), "uncertainty_required_fields": ["pair_id", "condition", "layers"], "final_layer_rule": "exactly one layers entry with is_output=true"},
        "source": {"forward": {"artifact": str(forward_source), "sha256": forward_hash, "coverage": {"records": len(attribution), "pairs": len(flip_rows)}}, "uncertainty": {"artifact": str(uncertainty_source), "sha256": uncertainty_hash, "coverage": {"records": len(uncertainty), "pairs": len(delta_rows)}}},
        "source_artifact": str(forward_source), "source_artifact_sha256": forward_hash,
        "coverage": coverage,
        "records": {"forward": len(attribution), "uncertainty": len(uncertainty), "valid_predictions": sum(r["parse_status"] == "valid" for r in attribution), "invalid_predictions": coverage["invalid_predictions"], "prediction_pairs": len(flip_rows), "uncertainty_pairs": len(delta_rows)},
        "labels": sorted({str(r["target_label"]) for r in attribution}),
        "missing_or_incomplete": {"prediction": missing_prediction, "uncertainty": missing_uncertainty},
        "interpretation": "Prediction flips and Jacobian readout uncertainty deltas are descriptive; uncertainty is not a causal proof.",
        "outputs": ["return_prediction_records.csv", "return_prediction_flips.csv", "return_paired_uncertainty_delta.csv", *(path.name for path in figure_paths)],
    }
    (output / "return_visualization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output
