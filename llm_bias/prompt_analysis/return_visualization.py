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
        ticker = original.get("ticker") or counterfactual.get("ticker")
        peer_ticker = original.get("peer_ticker") or counterfactual.get("peer_ticker")
        pair_label = f"{ticker} vs {peer_ticker}" if ticker and peer_ticker else (ticker or "Unknown")
        complete.append({
            "pair_id": pair_id,
            "filing_date": original.get("filing_date", counterfactual.get("filing_date")),
            "ticker": ticker,
            "peer_ticker": peer_ticker,
            "pair_label": pair_label,
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

    pair_labels = list(dict.fromkeys(r.get("pair_label") for r in rows if r.get("pair_label")))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    if pair_labels:
        cmap = plt.get_cmap("tab10")
        label_colors = {pl: cmap(i % 10) for i, pl in enumerate(pair_labels)}

        for axis, field, label in zip(
            axes,
            ("entropy_delta_nats", "effective_temperature_delta"),
            ("Entropy delta (nats)", "Effective-temperature delta"),
            strict=True,
        ):
            axis.axhline(0, color="#898781", linewidth=1)
            for pl in pair_labels:
                indices = [i for i, r in enumerate(rows) if r.get("pair_label") == pl]
                vals = [float(rows[i][field]) for i in indices]
                axis.scatter(
                    indices,
                    vals,
                    color=label_colors[pl],
                    label=pl,
                    edgecolor="#fcfcfb",
                    linewidth=0.6,
                    s=40,
                )
            axis.set_xlabel("Complete pair (pair_id order)")
            axis.set_ylabel("Counterfactual − original")
            axis.set_title(label, loc="left", fontweight="bold")
            _style(axis)
            axis.legend(title="Entity Pair", loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e1e0d9", fontsize=9)
    else:
        for axis, field, label, color in zip(
            axes,
            ("entropy_delta_nats", "effective_temperature_delta"),
            ("Entropy delta (nats)", "Effective-temperature delta"),
            ("#2a78d6", "#eb6834"),
            strict=True,
        ):
            values = [float(row[field]) for row in rows]
            axis.axhline(0, color="#898781", linewidth=1)
            axis.scatter(np.arange(len(values)), values, color=color, edgecolor="#fcfcfb", linewidth=0.8, s=46)
            axis.set_xlabel("Complete pair (pair_id order)")
            axis.set_ylabel("Counterfactual − original")
            axis.set_title(label, loc="left", fontweight="bold")
            _style(axis)

    figure.suptitle("Paired final-layer uncertainty deltas", fontsize=15, fontweight="bold")
    path = output_dir / "return_paired_uncertainty_delta.png"
    figure.savefig(path, dpi=220, facecolor="#fcfcfb")
    plt.close(figure)
    return [path]


def plot_confidence_vs_entropy_scatter_figure(
    attribution_rows: Iterable[dict[str, Any]],
    uncertainty_rows: Iterable[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    attr_list = list(attribution_rows)
    unc_list = list(uncertainty_rows)
    if not attr_list or not unc_list:
        return []

    unc_map = {(str(r["pair_id"]), str(r["condition"])): float(r["entropy_nats"]) for r in unc_list if "entropy_nats" in r}
    merged = []
    for r in attr_list:
        key = (str(r["pair_id"]), str(r["condition"]))
        if key in unc_map and r.get("parse_status") == "valid" and r.get("predicted_confidence") is not None:
            ticker = str(r.get("ticker", ""))
            peer = str(r.get("peer_ticker", ""))
            pair_label = f"{ticker} vs {peer}" if ticker and peer else (ticker or "Unknown")
            merged.append({
                "pair_id": str(r["pair_id"]),
                "condition": str(r["condition"]),
                "confidence": float(r["predicted_confidence"]),
                "entropy": unc_map[key],
                "pair_label": pair_label,
            })

    if not merged:
        return []

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax1, ax2 = axes

    cond_colors = {"original": "#2a78d6", "counterfactual": "#eb6834"}
    for cond in CONDITIONS:
        sub = [m for m in merged if m["condition"] == cond]
        if not sub:
            continue
        xs = [m["entropy"] for m in sub]
        ys = [m["confidence"] for m in sub]
        ax1.scatter(xs, ys, color=cond_colors.get(cond, "#898781"), label=cond.title(), alpha=0.6, edgecolor="#ffffff", linewidth=0.5, s=36)
        if len(sub) > 1:
            arr_x = np.array(xs)
            arr_y = np.array(ys)
            slope, intercept = np.polyfit(arr_x, arr_y, 1)
            x_seq = np.linspace(arr_x.min(), arr_x.max(), 100)
            r_val = float(np.corrcoef(arr_x, arr_y)[0, 1])
            ax1.plot(x_seq, slope * x_seq + intercept, color=cond_colors.get(cond, "#898781"), linewidth=2, linestyle="--", label=f"{cond.title()} trend (r={r_val:.2f})")

    ax1.set_xlabel("Final-layer Entropy (nats)")
    ax1.set_ylabel("Predicted Confidence (%)")
    ax1.set_title("Predicted Confidence vs. Final-layer Entropy (by Condition)", loc="left", fontweight="bold")
    _style(ax1)
    ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e1e0d9", fontsize=9)

    pair_labels = list(dict.fromkeys(m["pair_label"] for m in merged if m.get("pair_label")))
    if pair_labels:
        cmap = plt.get_cmap("tab10")
        label_colors = {pl: cmap(i % 10) for i, pl in enumerate(pair_labels)}
        for pl in pair_labels:
            sub = [m for m in merged if m["pair_label"] == pl]
            xs = [m["entropy"] for m in sub]
            ys = [m["confidence"] for m in sub]
            ax2.scatter(xs, ys, color=label_colors[pl], label=pl, alpha=0.65, edgecolor="#ffffff", linewidth=0.5, s=36)
        ax2.set_xlabel("Final-layer Entropy (nats)")
        ax2.set_ylabel("Predicted Confidence (%)")
        ax2.set_title("Predicted Confidence vs. Final-layer Entropy (by Entity Pair)", loc="left", fontweight="bold")
        _style(ax2)
        ax2.legend(title="Entity Pair", loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e1e0d9", fontsize=9)

    figure.suptitle("Scatter Analysis: Model Output Confidence vs. Final-Layer Readout Entropy", fontsize=15, fontweight="bold")
    path = output_dir / "return_confidence_vs_entropy_scatter.png"
    figure.savefig(path, dpi=220, facecolor="#fcfcfb")
    plt.close(figure)
    return [path]


def _decompose_entropy_beta(
    processed: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, float | str]], float, float]:
    rows = list(processed)
    if not rows:
        raise ValueError("processed entropy-beta rows must not be empty")

    x_all = np.array([row["beta_delta"] for row in rows], dtype=float)
    y_all = np.array([row["entropy_delta"] for row in rows], dtype=float)
    gamma_global, alpha_global = np.polyfit(x_all, y_all, 1)

    summaries: list[dict[str, float | str]] = []
    pair_labels = list(dict.fromkeys(row["pair_label"] for row in rows))
    for pair_label in pair_labels:
        group = [row for row in rows if row["pair_label"] == pair_label]
        total_delta = float(np.mean([row["entropy_delta"] for row in group]))
        fitted_effect = float(
            alpha_global + gamma_global * np.mean([row["beta_delta"] for row in group])
        )
        summaries.append(
            {
                "pair_label": pair_label,
                "total_delta": total_delta,
                "fitted_effect": fitted_effect,
                "directional_residual": total_delta - fitted_effect,
            }
        )
    return summaries, float(gamma_global), float(alpha_global)


def plot_entropy_beta_regression_figure(delta_rows: Iterable[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = list(delta_rows)
    if not rows:
        return []

    processed = []
    for r in rows:
        orig_t = float(r.get("original_effective_temperature", 0))
        cf_t = float(r.get("counterfactual_effective_temperature", 0))
        if orig_t > 0 and cf_t > 0:
            beta_orig = 1.0 / orig_t
            beta_cf = 1.0 / cf_t
            processed.append({
                "pair_id": str(r["pair_id"]),
                "pair_label": str(r.get("pair_label", "Unknown")),
                "entropy_delta": float(r["entropy_delta_nats"]),
                "beta_delta": beta_cf - beta_orig,
            })

    if not processed:
        return []

    df = pd.DataFrame(processed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    pair_labels = list(dict.fromkeys(df["pair_label"]))
    cmap = plt.get_cmap("tab10")
    label_colors = {pl: cmap(i % 10) for i, pl in enumerate(pair_labels)}

    ax1.axhline(0, color="#898781", linewidth=0.8, linestyle=":")
    ax1.axvline(0, color="#898781", linewidth=0.8, linestyle=":")

    for pl in pair_labels:
        sub = df[df["pair_label"] == pl]
        x = sub["beta_delta"]
        y = sub["entropy_delta"]
        ax1.scatter(x, y, color=label_colors[pl], label=pl, alpha=0.6, edgecolor="#ffffff", linewidth=0.5, s=36)
        if len(sub) > 1:
            g_gamma, g_alpha = np.polyfit(x, y, 1)
            r_val = float(np.corrcoef(x, y)[0, 1])
            r2_val = r_val ** 2
            x_seq = np.linspace(x.min(), x.max(), 100)
            ax1.plot(x_seq, g_gamma * x_seq + g_alpha, color=label_colors[pl], linewidth=1.8, linestyle="--", label=f"{pl} (γ_group={g_gamma:+.3f}, R²={r2_val:.2f})")

    ax1.set_xlabel("Inverse Temperature Delta (Δβ = β_cf - β_orig)")
    ax1.set_ylabel("Entropy Delta (ΔH, nats)")
    ax1.set_title("Cross-Sample Regression: ΔH vs. Δβ (Grouped)", loc="left", fontweight="bold")
    _style(ax1)
    ax1.legend(title="Entity Pair & Group Fit", loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e1e0d9", fontsize=8.5)

    # Global regression for baseline decomposition.
    group_summary, gamma_global, alpha_global = _decompose_entropy_beta(processed)

    sum_df = pd.DataFrame(group_summary)
    pos = np.arange(len(pair_labels))
    bar_width = 0.25

    ax2.axhline(0, color="#898781", linewidth=0.8)
    b1 = ax2.bar(pos - bar_width, sum_df["total_delta"], bar_width, label="Total Observed ΔH", color="#4a5568")
    b2 = ax2.bar(pos, sum_df["fitted_effect"], bar_width, label="Global Fitted Effect (α_global + γ_global · Δβ)", color="#3182ce")
    b3 = ax2.bar(pos + bar_width, sum_df["directional_residual"], bar_width, label="Unexplained Residual (ε)", color="#dd6b20")

    ax2.set_xticks(pos)
    ax2.set_xticklabels(sum_df["pair_label"], rotation=15, ha="right")
    ax2.set_ylabel("Entropy Delta (nats)")
    ax2.set_title("Linear Decomposition: Global Fitted Effect vs. Unexplained Residual", loc="left", fontweight="bold")
    _style(ax2)
    ax2.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e1e0d9", fontsize=9)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            offset = 0.003 if h >= 0 else -0.007
            va = "bottom" if h >= 0 else "top"
            ax2.text(bar.get_x() + bar.get_width()/2., h + offset, f"{h:+.3f}", ha="center", va=va, fontsize=8)

    fig.suptitle("Entropy Delta (ΔH) Regression & Decomposition: Length Scaling (Δβ) vs. Directional Shift Residual (ε)", fontsize=14, fontweight="bold")
    path = output_dir / "return_entropy_beta_regression_decomposition.png"
    fig.savefig(path, dpi=220, facecolor="#fcfcfb")
    plt.close(fig)
    return [path]


def visualize_return_predictions(*, forward_path: str | Path, uncertainty_path: str | Path, output_dir: str | Path) -> Path:
    """Visualize predictions from a forward artifact and final-layer uncertainty."""
    forward_source = Path(forward_path)
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
    figure_paths = (
        plot_return_prediction_figures(attribution, output)
        + plot_paired_uncertainty_delta_figures(delta_rows, output)
        + plot_confidence_vs_entropy_scatter_figure(attribution, uncertainty, output)
        + plot_entropy_beta_regression_figure(delta_rows, output)
    )
    fields = sorted(FORWARD_FIELDS | {"predicted_label", "predicted_confidence", "parse_status"})
    _write_csv(output / "return_prediction_records.csv", attribution, fields)
    _write_csv(output / "return_prediction_flips.csv", flip_rows, ["pair_id", "filing_date", "original_predicted_label", "counterfactual_predicted_label", "original_target_label", "counterfactual_target_label", "original_predicted_confidence", "counterfactual_predicted_confidence", "valid_original", "valid_counterfactual", "prediction_flip"])
    _write_csv(output / "return_paired_uncertainty_delta.csv", delta_rows, ["pair_id", "filing_date", "ticker", "peer_ticker", "pair_label", "original_entropy_nats", "counterfactual_entropy_nats", "entropy_delta_nats", "original_effective_temperature", "counterfactual_effective_temperature", "effective_temperature_delta"])
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
