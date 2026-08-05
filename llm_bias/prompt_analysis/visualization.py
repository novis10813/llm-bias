"""Static and interactive visualizations for prompt-analysis artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from llm_bias.prompt_analysis.artifact_io import read_jsonl, sha256_file

FORWARD_ARTIFACT_TYPE = "generated_outputs"
BACKWARD_ARTIFACT_TYPES = {"generated_token_attribution"}

DEFAULT_INPUT = "sp500_r1k_r2k_entityBiasPrompt.csv"
DEFAULT_TOKENIZER = ".cache/models/llama-3.2-1b-instruct"

INDEX_LABELS = {
    "sp500": "S&P 500",
    "russell1000": "Russell 1000",
    "russell2000": "Russell 2000",
}
INDEX_ORDER = ("sp500", "russell1000", "russell2000")
CONTEXT_ORDER = ("without", "with")
UNCERTAINTY_METRICS = {
    "entropy_nats": {
        "label": "Entropy (nats)",
        "delta_label": "Entropy difference (nats)",
        "file_stem": "entropy",
    },
    "effective_temperature": {
        "label": "Effective temperature",
        "delta_label": "Effective-temperature difference",
        "file_stem": "effective_temperature",
    },
}
UNCERTAINTY_KEYS = tuple(
    (index, context) for index in INDEX_ORDER for context in CONTEXT_ORDER
)


def uncertainty_paths_from_root(root: str | Path) -> dict[tuple[str, str], Path]:
    """Resolve the combined readout uncertainty artifact below a run root."""
    root_path = Path(root)
    combined_candidates = (
        root_path / "readout" / "prompt_layer_uncertainty.jsonl",
        root_path / "prompt_layer_uncertainty.jsonl",
    )
    for combined in combined_candidates:
        if combined.is_file():
            return {key: combined for key in UNCERTAINTY_KEYS}
    raise FileNotFoundError(
        f"prompt_layer_uncertainty.jsonl not found below {root_path}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            rows.append(value)
    return rows


def load_final_layer_uncertainty(
    paths: dict[tuple[str, str], str | Path],
) -> list[dict[str, Any]]:
    """Load one final-layer uncertainty record per date and condition."""
    # A runner writes one combined JSONL for all prompt columns.  The historical
    # ``paths`` API represents that file six times (once per index/context), so
    # detect that shape and discover the conditions from the records instead of
    # filtering against the old fixed S&P/Russell condition set.
    unique_sources = {Path(source) for source in paths.values()}
    combined_source = next(iter(unique_sources)) if len(unique_sources) == 1 else None
    discover_conditions = combined_source is not None and len(paths) > 1
    result: list[dict[str, Any]] = []
    seen_dates_by_condition: dict[tuple[str, str], set[str]] = defaultdict(set)

    def append_row(
        row: dict[str, Any],
        *,
        index: str,
        context: str,
        source_path: Path,
    ) -> None:
        row_date = str(row.get("date", ""))
        layers = row.get("layers")
        if not row_date or not isinstance(layers, list):
            raise ValueError(f"uncertainty row is missing date/layers: {source_path}")
        output_layers = [layer for layer in layers if layer.get("is_output")]
        if len(output_layers) != 1:
            raise ValueError(
                f"expected exactly one output layer for {index}/{context} "
                f"date {row_date}, found {len(output_layers)}"
            )
        condition = (index, context)
        if row_date in seen_dates_by_condition[condition]:
            raise ValueError(
                f"duplicate uncertainty date {row_date} for {index}/{context} "
                f"in {source_path}"
            )
        seen_dates_by_condition[condition].add(row_date)
        output = output_layers[0]
        if "entropy_nats" not in output:
            raise ValueError(f"output layer has no entropy_nats: {source_path}")
        result.append(
            {
                "row_index": row.get("row_index"),
                "date": row_date,
                "index": index,
                "context": context,
                "layer": output.get("layer"),
                "entropy_nats": float(output["entropy_nats"]),
                "normalized_entropy": float(output.get("normalized_entropy", 0.0)),
                "perplexity": float(output.get("perplexity", 0.0)),
                "top1_probability": float(output.get("top1_probability", 0.0)),
                "topk_mass": float(output.get("topk_mass", 0.0)),
                "effective_inverse_temperature": float(
                    output.get("effective_inverse_temperature", 0.0)
                ),
                "effective_temperature": float(
                    output.get("effective_temperature", 0.0)
                ),
            }
        )

    if discover_conditions and combined_source is not None:
        for row in _read_jsonl(combined_source):
            index = str(row.get("index", ""))
            context = str(row.get("context", ""))
            if not index or not context:
                raise ValueError(
                    f"combined uncertainty row has no index/context: {combined_source}"
                )
            append_row(row, index=index, context=context, source_path=combined_source)
    else:
        for (index, context), source in paths.items():
            source_path = Path(source)
            for row in _read_jsonl(source_path):
                row_index_name = str(row.get("index", ""))
                row_context = str(row.get("context", ""))
                if (row_index_name or row_context) and (
                    row_index_name != index or row_context != context
                ):
                    continue
                append_row(row, index=index, context=context, source_path=source_path)
    return sorted(result, key=lambda row: (row["date"], row["index"], row["context"]))


def build_uncertainty_distribution_rows(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert final-layer uncertainty records into a metric-long table."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    layers: set[Any] = set()
    for record in records:
        date = str(record.get("date", ""))
        index = str(record.get("index", ""))
        context = str(record.get("context", ""))
        key = (date, index, context)
        if not all(key):
            raise ValueError("uncertainty record is missing date/index/context")
        if key in seen:
            raise ValueError(f"duplicate uncertainty condition: {date}/{index}/{context}")
        seen.add(key)
        layers.add(record.get("layer"))
        for metric in UNCERTAINTY_METRICS:
            value = float(record[metric])
            if not math.isfinite(value):
                raise ValueError(f"uncertainty metric must be finite: {key}/{metric}")
            if metric == "entropy_nats" and value < 0:
                raise ValueError(f"entropy must be non-negative: {key}")
            if metric == "effective_temperature" and value <= 0:
                raise ValueError(f"effective temperature must be positive: {key}")
            result.append(
                {
                    "date": date,
                    "index": index,
                    "context": context,
                    "layer": record.get("layer"),
                    "metric": metric,
                    "value": value,
                }
            )
    if len(layers) != 1 or None in layers:
        raise ValueError(
            f"uncertainty distribution requires one consistent output layer, found {layers}"
        )
    index_positions = {index: position for position, index in enumerate(INDEX_ORDER)}
    context_positions = {context: position for position, context in enumerate(CONTEXT_ORDER)}
    metric_positions = {
        metric: position for position, metric in enumerate(UNCERTAINTY_METRICS)
    }
    return sorted(
        result,
        key=lambda row: (
            metric_positions[row["metric"]],
            index_positions.get(row["index"], len(index_positions)),
            context_positions.get(row["context"], len(context_positions)),
            row["date"],
        ),
    )


def build_paired_uncertainty_deltas(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair with/without readouts by date within each index."""
    conditions: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    layers: set[Any] = set()
    for record in records:
        date = str(record.get("date", ""))
        index = str(record.get("index", ""))
        context = str(record.get("context", ""))
        if context not in CONTEXT_ORDER:
            raise ValueError(f"unsupported uncertainty context: {context!r}")
        layers.add(record.get("layer"))
        entropy = float(record["entropy_nats"])
        temperature = float(record["effective_temperature"])
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError(f"entropy must be finite and non-negative: {date}/{index}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                f"effective temperature must be finite and positive: {date}/{index}"
            )
        by_context = conditions[(index, date)]
        if context in by_context:
            raise ValueError(f"duplicate uncertainty condition: {date}/{index}/{context}")
        by_context[context] = record

    if len(layers) != 1 or None in layers:
        raise ValueError(
            f"paired uncertainty requires one consistent output layer, found {layers}"
        )
    result: list[dict[str, Any]] = []
    for (index, date), by_context in conditions.items():
        if not all(context in by_context for context in CONTEXT_ORDER):
            continue
        without = by_context["without"]
        with_context = by_context["with"]
        entropy_without = float(without["entropy_nats"])
        entropy_with = float(with_context["entropy_nats"])
        temperature_without = float(without["effective_temperature"])
        temperature_with = float(with_context["effective_temperature"])
        values = (
            entropy_without,
            entropy_with,
            temperature_without,
            temperature_with,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"paired uncertainty metrics must be finite: {date}/{index}")
        result.append(
            {
                "date": date,
                "index": index,
                "entropy_without": entropy_without,
                "entropy_with": entropy_with,
                "entropy_delta_nats": entropy_with - entropy_without,
                "effective_temperature_without": temperature_without,
                "effective_temperature_with": temperature_with,
                "effective_temperature_delta": temperature_with - temperature_without,
            }
        )
    index_positions = {index: position for position, index in enumerate(INDEX_ORDER)}
    return sorted(
        result,
        key=lambda row: (
            index_positions.get(row["index"], len(index_positions)),
            row["date"],
        ),
    )


def _distribution_statistics(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty uncertainty distribution")
    if not np.isfinite(array).all():
        raise ValueError("uncertainty distribution values must be finite")
    quantiles = np.percentile(
        array,
        [1, 5, 25, 50, 75, 95, 99],
        method="linear",
    )
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(array.max()),
    }


def summarize_uncertainty_distributions(
    raw_rows: Iterable[dict[str, Any]],
    paired_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize raw conditions and paired with-minus-without deltas."""
    raw_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_groups[(row["metric"], row["index"], row["context"])].append(row)

    result: list[dict[str, Any]] = []
    for (metric, index, context), rows in raw_groups.items():
        dates = sorted(str(row["date"]) for row in rows)
        result.append(
            {
                "distribution": "raw",
                "metric": metric,
                "index": index,
                "context": context,
                "date_min": dates[0],
                "date_max": dates[-1],
                **_distribution_statistics(float(row["value"]) for row in rows),
            }
        )

    paired_metric_fields = {
        "entropy_nats": "entropy_delta_nats",
        "effective_temperature": "effective_temperature_delta",
    }
    paired_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        paired_groups[row["index"]].append(row)
    for index, rows in paired_groups.items():
        dates = sorted(str(row["date"]) for row in rows)
        for metric, field in paired_metric_fields.items():
            result.append(
                {
                    "distribution": "paired_delta",
                    "metric": metric,
                    "index": index,
                    "context": "with_minus_without",
                    "date_min": dates[0],
                    "date_max": dates[-1],
                    **_distribution_statistics(float(row[field]) for row in rows),
                }
            )

    distribution_positions = {"raw": 0, "paired_delta": 1}
    metric_positions = {
        metric: position for position, metric in enumerate(UNCERTAINTY_METRICS)
    }
    index_positions = {index: position for position, index in enumerate(INDEX_ORDER)}
    context_positions = {
        context: position for position, context in enumerate(
            (*CONTEXT_ORDER, "with_minus_without")
        )
    }
    return sorted(
        result,
        key=lambda row: (
            distribution_positions[row["distribution"]],
            metric_positions[row["metric"]],
            index_positions.get(row["index"], len(index_positions)),
            context_positions.get(row["context"], len(context_positions)),
        ),
    )


def _uncertainty_index_order(rows: Iterable[dict[str, Any]]) -> list[str]:
    available = {str(row["index"]) for row in rows}
    result = [index for index in INDEX_ORDER if index in available]
    result.extend(sorted(available - set(result)))
    return result


def plot_uncertainty_distribution_figures(
    raw_rows: Iterable[dict[str, Any]],
    paired_rows: Iterable[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot raw ECDFs and paired-delta violins for each uncertainty metric."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    raw = list(raw_rows)
    paired = list(paired_rows)
    indices = _uncertainty_index_order(raw)
    if not indices:
        raise ValueError("uncertainty distribution contains no indices")
    output_dir.mkdir(parents=True, exist_ok=True)
    context_colors = {"without": "#eb6834", "with": "#2a78d6"}
    context_styles = {"without": "--", "with": "-"}
    violin_color = "#2a78d6"
    paths: list[Path] = []

    for metric, config in UNCERTAINTY_METRICS.items():
        figure, axes = plt.subplots(
            len(indices),
            1,
            figsize=(12.5, 3.0 * len(indices) + 1.6),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for position, index in enumerate(indices):
            axis = axes[position][0]
            for context in CONTEXT_ORDER:
                values = np.sort(
                    np.asarray(
                        [
                            float(row["value"])
                            for row in raw
                            if row["metric"] == metric
                            and row["index"] == index
                            and row["context"] == context
                        ],
                        dtype=float,
                    )
                )
                if values.size == 0:
                    continue
                cumulative = np.arange(1, values.size + 1) / values.size
                axis.plot(
                    values,
                    cumulative,
                    color=context_colors[context],
                    linestyle=context_styles[context],
                    linewidth=2.0,
                    solid_capstyle="round",
                    label=(
                        f"Without context (n={values.size:,})"
                        if context == "without"
                        else f"With context (n={values.size:,})"
                    ),
                )
            axis.set_title(_index_label(index), loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Cumulative proportion")
            axis.set_ylim(0, 1.01)
            axis.grid(color="#e1e0d9", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, loc="lower right")
        axes[-1][0].set_xlabel(config["label"])
        figure.suptitle(
            f"Final-layer {config['label'].lower()} distribution across dates",
            fontsize=15,
            fontweight="bold",
        )
        figure.text(
            0.01,
            0.012,
            "Each curve is an empirical cross-date distribution; no model inference is rerun.",
            fontsize=8.5,
            color="#52514e",
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.965))
        raw_path = output_dir / f"final_layer_{config['file_stem']}_raw_ecdf.png"
        figure.savefig(raw_path, dpi=300, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(figure)
        paths.append(raw_path)

        delta_field = (
            "entropy_delta_nats"
            if metric == "entropy_nats"
            else "effective_temperature_delta"
        )
        distributions = [
            np.asarray(
                [float(row[delta_field]) for row in paired if row["index"] == index],
                dtype=float,
            )
            for index in indices
        ]
        empty_indices = [
            index for index, values in zip(indices, distributions, strict=True)
            if values.size == 0
        ]
        if empty_indices:
            raise ValueError(
                "paired uncertainty has no common dates for: "
                + ", ".join(empty_indices)
            )
        delta_figure, delta_axis = plt.subplots(figsize=(10.5, 6.5))
        violin_items = [
            (position, values)
            for position, values in enumerate(distributions, start=1)
            if values.size >= 5
        ]
        if violin_items:
            violin = delta_axis.violinplot(
                [values for _position, values in violin_items],
                positions=[position for position, _values in violin_items],
                widths=0.68,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(violin_color)
                body.set_edgecolor("none")
                body.set_alpha(0.30)
        for position, (index, values) in enumerate(
            zip(indices, distributions, strict=True), start=1
        ):
            if values.size < 5:
                offsets = np.linspace(-0.12, 0.12, values.size)
                delta_axis.scatter(
                    position + offsets,
                    values,
                    s=36,
                    color=violin_color,
                    edgecolor="#fcfcfb",
                    linewidth=1.0,
                    alpha=0.65,
                    zorder=2,
                )
            q25, median, q75 = np.percentile(
                values, [25, 50, 75], method="linear"
            )
            delta_axis.vlines(position, q25, q75, color="#0b0b0b", linewidth=6)
            delta_axis.scatter(
                position,
                median,
                s=56,
                color=violin_color,
                edgecolor="#fcfcfb",
                linewidth=1.5,
                zorder=3,
            )
            delta_axis.text(
                position,
                0.98,
                f"n={values.size:,}",
                transform=delta_axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="#52514e",
            )
        delta_axis.axhline(0, color="#898781", linewidth=1.0)
        delta_axis.set_xticks(
            np.arange(1, len(indices) + 1),
            [_index_label(index) for index in indices],
        )
        delta_axis.set_ylabel(config["delta_label"] + " · with − without")
        delta_axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        delta_axis.spines[["top", "right"]].set_visible(False)
        delta_axis.set_title(
            f"Final-layer {config['label'].lower()} paired differences",
            fontsize=15,
            fontweight="bold",
            pad=16,
        )
        delta_axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#0b0b0b",
                    markerfacecolor="#fcfcfb",
                    linewidth=6,
                    markersize=7,
                    label="Median/IQR; sparse groups show observations",
                )
            ],
            frameon=False,
            loc="lower right",
        )
        delta_figure.text(
            0.01,
            0.012,
            (
                "Paired within each index on dates present in both contexts; "
                "with − without is descriptive, not a causal estimate."
            ),
            fontsize=8.5,
            color="#52514e",
        )
        delta_figure.tight_layout(rect=(0, 0.045, 1, 1))
        delta_path = (
            output_dir / f"final_layer_{config['file_stem']}_paired_delta_violin.png"
        )
        delta_figure.savefig(
            delta_path, dpi=300, bbox_inches="tight", facecolor="#fcfcfb"
        )
        plt.close(delta_figure)
        paths.append(delta_path)
    return paths


def visualize_uncertainty_distributions(
    *,
    uncertainty_paths: dict[tuple[str, str], str | Path],
    output_dir: str | Path,
) -> Path:
    """Create final-layer uncertainty distribution research artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = load_final_layer_uncertainty(uncertainty_paths)
    raw_rows = build_uncertainty_distribution_rows(records)
    paired_rows = build_paired_uncertainty_deltas(records)
    indices = set(_uncertainty_index_order(records))
    paired_indices = {str(row["index"]) for row in paired_rows}
    if missing_indices := sorted(indices - paired_indices):
        raise ValueError(
            "paired uncertainty has no common dates for: "
            + ", ".join(missing_indices)
        )
    summaries = summarize_uncertainty_distributions(raw_rows, paired_rows)

    raw_fields = ["date", "index", "context", "layer", "metric", "value"]
    paired_fields = [
        "date",
        "index",
        "entropy_without",
        "entropy_with",
        "entropy_delta_nats",
        "effective_temperature_without",
        "effective_temperature_with",
        "effective_temperature_delta",
    ]
    summary_fields = [
        "distribution",
        "metric",
        "index",
        "context",
        "n",
        "date_min",
        "date_max",
        "mean",
        "std",
        "min",
        "q01",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "q99",
        "max",
    ]
    _write_csv(
        output / "final_layer_uncertainty_distribution_raw.csv",
        raw_rows,
        raw_fields,
    )
    _write_csv(
        output / "final_layer_uncertainty_paired_delta.csv",
        paired_rows,
        paired_fields,
    )
    _write_csv(
        output / "final_layer_uncertainty_distribution_summary.csv",
        summaries,
        summary_fields,
    )
    figure_paths = plot_uncertainty_distribution_figures(raw_rows, paired_rows, output)

    source_paths = sorted({Path(path) for path in uncertainty_paths.values()})
    counts_by_condition: dict[str, int] = defaultdict(int)
    dates_by_condition: dict[str, list[str]] = defaultdict(list)
    for record in records:
        condition = f"{record['index']}/{record['context']}"
        counts_by_condition[condition] += 1
        dates_by_condition[condition].append(str(record["date"]))
    paired_counts: dict[str, int] = defaultdict(int)
    for row in paired_rows:
        paired_counts[row["index"]] += 1
    paired_metadata = {}
    for index in _uncertainty_index_order(records):
        without_dates = {
            str(record["date"])
            for record in records
            if record["index"] == index and record["context"] == "without"
        }
        with_dates = {
            str(record["date"])
            for record in records
            if record["index"] == index and record["context"] == "with"
        }
        paired_metadata[index] = {
            "n_pairs": paired_counts[index],
            "unmatched_dates": len(without_dates ^ with_dates),
            "date_rule": "intersection within index",
        }
    metadata = {
        "sources": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in source_paths
        ],
        "records": len(records),
        "output_layer": next(iter({record["layer"] for record in records})),
        "raw_distribution_rows": len(raw_rows),
        "paired_rows": len(paired_rows),
        "counts_by_condition": dict(sorted(counts_by_condition.items())),
        "date_ranges_by_condition": {
            condition: {"min": min(dates), "max": max(dates)}
            for condition, dates in sorted(dates_by_condition.items())
        },
        "paired": {
            "definition": "with_context - without_context",
            "by_index": paired_metadata,
        },
        "metrics": {
            "entropy_nats": {
                "unit": "nats",
                "definition": "Shannon entropy of the final-layer full-vocabulary softmax",
            },
            "effective_temperature": {
                "definition": "reciprocal L2 norm of the final-normalized residual",
                "sampling_temperature": False,
            },
        },
        "quantiles": {
            "values": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
            "method": "numpy.percentile(method=linear)",
        },
        "plots": {
            "raw": "empirical cumulative distribution function",
            "paired_delta": "violin with median and interquartile range",
            "dual_axis": False,
        },
        "interpretation": (
            "Paired with-minus-without differences are descriptive associations, "
            "not causal estimates."
        ),
        "outputs": [
            "final_layer_uncertainty_distribution_raw.csv",
            "final_layer_uncertainty_paired_delta.csv",
            "final_layer_uncertainty_distribution_summary.csv",
            "final_layer_uncertainty_distribution_metadata.json",
            *(path.name for path in figure_paths),
        ],
    }
    (output / "final_layer_uncertainty_distribution_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _load_prices(
    path: Path,
    indices: Iterable[str] = INDEX_ORDER,
) -> list[dict[str, str]]:
    """Load rows containing the price columns used by the experiment."""
    if not path.is_file():
        raise FileNotFoundError(path)
    price_indices = tuple(dict.fromkeys(str(index) for index in indices))
    if not price_indices:
        raise ValueError("at least one price column is required")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Date", *price_indices}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"price CSV must contain {sorted(required)}")
    valid_rows = []
    for row in rows:
        try:
            for index in price_indices:
                float(row[index])
        except (TypeError, ValueError):
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise ValueError(f"price CSV contains no complete price rows: {path}")
    return valid_rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_generated_answer(generated_text: Any) -> dict[str, Any]:
    """Extract a finite numeric answer from a generated JSON object."""
    if not isinstance(generated_text, str):
        return {
            "parsed_answer": None,
            "confidence": None,
            "parse_status": "invalid",
            "parse_reason": "generated_text_not_string",
        }
    object_start = generated_text.find("{")
    if object_start < 0:
        return {
            "parsed_answer": None,
            "confidence": None,
            "parse_status": "invalid",
            "parse_reason": "json_object_not_found",
        }
    try:
        payload, _end = json.JSONDecoder().raw_decode(generated_text[object_start:])
    except json.JSONDecodeError:
        return {
            "parsed_answer": None,
            "confidence": None,
            "parse_status": "invalid",
            "parse_reason": "malformed_json",
        }
    if not isinstance(payload, dict):
        return {
            "parsed_answer": None,
            "confidence": None,
            "parse_status": "invalid",
            "parse_reason": "json_payload_not_object",
        }
    answer = payload.get("answer")
    if answer is None:
        reason = "answer_missing" if "answer" not in payload else "answer_null"
        return {
            "parsed_answer": None,
            "confidence": None,
            "parse_status": "invalid",
            "parse_reason": reason,
        }
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        return {
            "parsed_answer": None,
            "confidence": None,
            "parse_status": "invalid",
            "parse_reason": "answer_not_numeric",
        }
    parsed_answer = float(answer)
    if not math.isfinite(parsed_answer):
        return {
            "parsed_answer": None,
            "confidence": None,
            "parse_status": "invalid",
            "parse_reason": "answer_not_finite",
        }
    confidence = payload.get("confidence")
    parsed_confidence = (
        float(confidence)
        if not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(float(confidence))
        else None
    )
    return {
        "parsed_answer": parsed_answer,
        "confidence": parsed_confidence,
        "parse_status": "valid",
        "parse_reason": None,
    }


def _sampling_condition(prompt_column: str) -> tuple[str, str]:
    for context in CONTEXT_ORDER:
        prefix = f"prompt_{context}_context_"
        if prompt_column.startswith(prefix) and len(prompt_column) > len(prefix):
            return prompt_column[len(prefix) :], context
    raise ValueError(f"invalid sampling prompt column: {prompt_column!r}")


def _load_price_lookup(
    path: Path,
    indices: Iterable[str],
) -> dict[tuple[str, str], float]:
    rows = _load_prices(path, indices)
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        date = str(row["Date"])
        for index in indices:
            key = (date, index)
            if key in lookup:
                raise ValueError(f"duplicate close price for {date}/{index}: {path}")
            value = float(row[index])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"close price must be finite and positive for {date}/{index}")
            lookup[key] = value
    return lookup


def load_price_distribution_samples(
    sampling_root: str | Path,
    prices_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and normalize every record in a complete multi-run artifact."""
    root = Path(sampling_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"expected JSON object in {manifest_path}")

    runs = manifest.get("runs")
    run_indices = manifest.get("run_indices")
    if not isinstance(runs, int) or runs < 1:
        raise ValueError("sampling manifest must contain a positive runs value")
    if run_indices != list(range(runs)):
        raise ValueError("sampling manifest run_indices must be contiguous from zero")

    condition_counts = manifest.get("condition_counts")
    selected_dates = manifest.get("selected_dates")
    records_per_run = manifest.get("records_per_run")
    if not isinstance(condition_counts, dict) or not condition_counts:
        raise ValueError("sampling manifest has no condition_counts")
    if not isinstance(selected_dates, list) or not selected_dates:
        raise ValueError("sampling manifest has no selected_dates")
    if len(set(selected_dates)) != len(selected_dates):
        raise ValueError("sampling manifest contains duplicate selected_dates")
    if not isinstance(records_per_run, int) or records_per_run < 1:
        raise ValueError("sampling manifest has no valid records_per_run")

    conditions: dict[str, tuple[str, str]] = {}
    for prompt_column, count in condition_counts.items():
        if not isinstance(prompt_column, str) or count != len(selected_dates):
            raise ValueError("sampling condition counts must match selected_dates")
        conditions[prompt_column] = _sampling_condition(prompt_column)
    indices = tuple(dict.fromkeys(index for index, _context in conditions.values()))
    if records_per_run != len(selected_dates) * len(conditions):
        raise ValueError("sampling manifest records_per_run does not match its conditions")

    expected_directories = {f"run_{run_index:03d}" for run_index in run_indices}
    actual_directories = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    }
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        unexpected = sorted(actual_directories - expected_directories)
        raise ValueError(
            f"sampling run directories do not match manifest; missing={missing}, "
            f"unexpected={unexpected}"
        )

    run_metadata = manifest.get("run_directories")
    if not isinstance(run_metadata, list) or len(run_metadata) != runs:
        raise ValueError("sampling manifest run_directories does not match runs")
    run_seeds: dict[int, int | None] = {}
    run_directories: dict[int, str] = {}
    expected_written: dict[int, int] = {}
    for entry in run_metadata:
        if not isinstance(entry, dict):
            raise ValueError("sampling manifest run directory entry must be an object")
        run_index = entry.get("run_index")
        if run_index not in run_indices or run_index in run_seeds:
            raise ValueError("sampling manifest has invalid or duplicate run_index")
        directory = entry.get("directory")
        expected_directory = f"run_{run_index:03d}"
        if directory != expected_directory:
            raise ValueError(
                f"sampling manifest directory for run {run_index} must be "
                f"{expected_directory!r}"
            )
        run_seeds[run_index] = entry.get("run_seed")
        run_directories[run_index] = directory
        expected_written[run_index] = entry.get("records_written")
    if any(count != records_per_run for count in expected_written.values()):
        raise ValueError("sampling manifest records_written does not match records_per_run")

    prices = _load_price_lookup(Path(prices_path), indices)
    expected_keys = {
        (str(date), prompt_column)
        for date in selected_dates
        for prompt_column in conditions
    }
    samples: list[dict[str, Any]] = []
    for run_index in run_indices:
        run_path = root / run_directories[run_index] / "generated_token_attribution.jsonl"
        if not run_path.is_file():
            raise FileNotFoundError(run_path)
        seen_keys: set[tuple[str, str]] = set()
        row_count = 0
        with run_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {run_path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"expected JSON object in {run_path}:{line_number}")
                row_count += 1
                if row.get("run_index") != run_index:
                    raise ValueError(f"run_index mismatch in {run_path}:{line_number}")
                date = str(row.get("date", ""))
                prompt_column = str(row.get("prompt_column", ""))
                key = (date, prompt_column)
                if key not in expected_keys:
                    raise ValueError(f"unexpected date/condition in {run_path}:{line_number}")
                if key in seen_keys:
                    raise ValueError(f"duplicate date/condition in {run_path}:{line_number}")
                seen_keys.add(key)
                index, context = conditions[prompt_column]
                if row.get("index") != index or row.get("context") != context:
                    raise ValueError(f"condition fields disagree in {run_path}:{line_number}")
                price_key = (date, index)
                if price_key not in prices:
                    raise ValueError(f"missing close price for {date}/{index}")

                parsed = _parse_generated_answer(row.get("generated_text"))
                answer = parsed["parsed_answer"]
                actual_close = prices[price_key]
                absolute_percentage_error = (
                    abs(answer - actual_close) / actual_close * 100.0
                    if answer is not None
                    else None
                )
                samples.append(
                    {
                        "run_index": run_index,
                        "run_seed": run_seeds[run_index],
                        "sample_index": row.get("sample_index"),
                        "date": date,
                        "index": index,
                        "context": context,
                        "prompt_column": prompt_column,
                        "generated_text": row.get("generated_text"),
                        **parsed,
                        "actual_close": actual_close,
                        "absolute_percentage_error": absolute_percentage_error,
                    }
                )
        if row_count != records_per_run or seen_keys != expected_keys:
            raise ValueError(f"incomplete sampling records in {run_path}")

    return samples, manifest


def summarize_price_distributions(
    samples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize price samples by date, index, and context."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(sample["date"], sample["index"], sample["context"])].append(sample)

    index_positions = {index: position for position, index in enumerate(INDEX_ORDER)}
    context_positions = {context: position for position, context in enumerate(CONTEXT_ORDER)}
    result: list[dict[str, Any]] = []
    for (date, index, context), rows in groups.items():
        actual_values = {float(row["actual_close"]) for row in rows}
        if len(actual_values) != 1:
            raise ValueError(f"inconsistent close prices for {date}/{index}/{context}")
        values = [
            float(row["parsed_answer"])
            for row in rows
            if row["parsed_answer"] is not None
        ]
        errors = [
            float(row["absolute_percentage_error"])
            for row in rows
            if row["absolute_percentage_error"] is not None
        ]
        quantiles = (
            np.percentile(values, [5, 25, 50, 75, 95], method="linear").tolist()
            if values
            else [None] * 5
        )
        result.append(
            {
                "date": date,
                "index": index,
                "context": context,
                "actual_close": next(iter(actual_values)),
                "n_total": len(rows),
                "n_valid": len(values),
                "n_invalid": len(rows) - len(values),
                "validity_rate": len(values) / len(rows),
                "min": min(values) if values else None,
                "q05": quantiles[0],
                "q25": quantiles[1],
                "median": quantiles[2],
                "q75": quantiles[3],
                "q95": quantiles[4],
                "max": max(values) if values else None,
                "median_absolute_percentage_error": (
                    float(np.median(errors)) if errors else None
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["date"],
            index_positions.get(row["index"], len(index_positions)),
            context_positions.get(row["context"], len(context_positions)),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def plot_price_distributions(
    summaries: Iterable[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot actual closes, generated price bands, and MdAPE for each index."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = list(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"without": "#eb6834", "with": "#2a78d6"}
    line_styles = {"without": "--", "with": "-"}
    markers = {"without": "s", "with": "o"}
    paths: list[Path] = []
    available_indices = set(str(row["index"]) for row in rows)
    indices = [index for index in INDEX_ORDER if index in available_indices]
    indices.extend(sorted(available_indices - set(indices)))

    for index in indices:
        market_rows = [row for row in rows if row["index"] == index]
        if not market_rows:
            continue
        figure = plt.figure(figsize=(13, 10.5), facecolor="#fcfcfb")
        grid = figure.add_gridspec(3, 1, height_ratios=(1, 1, 0.78), hspace=0.12)
        without_axis = figure.add_subplot(grid[0, 0])
        with_axis = figure.add_subplot(
            grid[1, 0], sharex=without_axis, sharey=without_axis
        )
        error_axis = figure.add_subplot(grid[2, 0], sharex=without_axis)
        price_axes = {"without": without_axis, "with": with_axis}

        invalid_total = 0
        for context in CONTEXT_ORDER:
            axis = price_axes[context]
            series = sorted(
                (row for row in market_rows if row["context"] == context),
                key=lambda row: row["date"],
            )
            dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
            actual = np.asarray([row["actual_close"] for row in series], dtype=float)
            q05 = np.asarray([row["q05"] for row in series], dtype=float)
            q25 = np.asarray([row["q25"] for row in series], dtype=float)
            median = np.asarray([row["median"] for row in series], dtype=float)
            q75 = np.asarray([row["q75"] for row in series], dtype=float)
            q95 = np.asarray([row["q95"] for row in series], dtype=float)
            invalid_total += sum(int(row["n_invalid"]) for row in series)

            axis.fill_between(dates, q05, q95, color=colors[context], alpha=0.10)
            axis.fill_between(dates, q25, q75, color=colors[context], alpha=0.22)
            axis.plot(
                dates,
                median,
                color=colors[context],
                linewidth=2.0,
                solid_capstyle="round",
            )
            axis.plot(
                dates,
                actual,
                color="#0b0b0b",
                linewidth=2.3,
                solid_capstyle="round",
                zorder=4,
            )
            axis.set_title(
                "Without context" if context == "without" else "With context",
                loc="left",
                fontsize=11,
                fontweight="bold",
            )
            axis.set_ylabel("Close price")
            axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", labelbottom=False)

        for context in CONTEXT_ORDER:
            series = sorted(
                (row for row in market_rows if row["context"] == context),
                key=lambda row: row["date"],
            )
            dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
            errors = [row["median_absolute_percentage_error"] for row in series]
            error_axis.plot(
                dates,
                errors,
                color=colors[context],
                linestyle=line_styles[context],
                linewidth=2.0,
                marker=markers[context],
                markersize=5.5,
                markeredgecolor="#fcfcfb",
                markeredgewidth=1.0,
                label=(
                    "Without context MdAPE"
                    if context == "without"
                    else "With context MdAPE"
                ),
            )
        error_axis.set_ylim(bottom=0)
        error_axis.set_ylabel("MdAPE (%)")
        error_axis.set_xlabel("Date")
        error_axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        error_axis.spines[["top", "right"]].set_visible(False)
        error_axis.legend(frameon=False, ncol=2, loc="upper left")

        locator = mdates.AutoDateLocator(minticks=7, maxticks=12)
        error_axis.xaxis.set_major_locator(locator)
        error_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        without_axis.legend(
            handles=[
                Line2D([0], [0], color="#0b0b0b", linewidth=2.3, label="Actual close"),
                Line2D([0], [0], color=colors["without"], linewidth=2.0, label="LLM median"),
                Patch(facecolor=colors["without"], alpha=0.22, label="25–75%"),
                Patch(facecolor=colors["without"], alpha=0.10, label="5–95%"),
            ],
            frameon=False,
            ncol=4,
            loc="upper left",
        )
        with_axis.legend(
            handles=[
                Line2D([0], [0], color="#0b0b0b", linewidth=2.3, label="Actual close"),
                Line2D([0], [0], color=colors["with"], linewidth=2.0, label="LLM median"),
                Patch(facecolor=colors["with"], alpha=0.22, label="25–75%"),
                Patch(facecolor=colors["with"], alpha=0.10, label="5–95%"),
            ],
            frameon=False,
            ncol=4,
            loc="upper left",
        )
        figure.suptitle(
            f"{_index_label(index)} · Actual close and LLM price distribution",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )
        figure.text(
            0.01,
            0.012,
            (
                "Bands summarize the central 90% and 50% of valid generated prices; "
                f"{invalid_total} invalid outputs are excluded and retained in CSV. "
                "Descriptive sampling variability; not a causal estimate."
            ),
            fontsize=8.5,
            color="#52514e",
        )
        figure.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.09)
        output_path = output_dir / f"{index}_price_distribution.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(figure)
        paths.append(output_path)
    return paths


def visualize_price_distributions(
    *,
    sampling_root: str | Path,
    prices_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path,
) -> Path:
    """Create research figures and traceable tables for multi-run prices."""
    root = Path(sampling_root)
    prices = Path(prices_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples, manifest = load_price_distribution_samples(root, prices)
    summaries = summarize_price_distributions(samples)

    sample_fields = [
        "run_index",
        "run_seed",
        "sample_index",
        "date",
        "index",
        "context",
        "prompt_column",
        "generated_text",
        "parsed_answer",
        "confidence",
        "parse_status",
        "parse_reason",
        "actual_close",
        "absolute_percentage_error",
    ]
    summary_fields = [
        "date",
        "index",
        "context",
        "actual_close",
        "n_total",
        "n_valid",
        "n_invalid",
        "validity_rate",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
        "median_absolute_percentage_error",
    ]
    _write_csv(output / "price_distribution_samples.csv", samples, sample_fields)
    _write_csv(output / "price_distribution_summary.csv", summaries, summary_fields)
    figure_paths = plot_price_distributions(summaries, output)

    valid_count = sum(sample["parse_status"] == "valid" for sample in samples)
    metadata = {
        "sampling_root": str(root),
        "sampling_manifest": str(root / "manifest.json"),
        "sampling_manifest_sha256": _sha256_file(root / "manifest.json"),
        "prices": str(prices),
        "prices_sha256": _sha256_file(prices),
        "model": manifest.get("model"),
        "runs": manifest.get("runs"),
        "selected_dates": manifest.get("selected_dates"),
        "generation": manifest.get("generation"),
        "generation_config": manifest.get("generation_config"),
        "records_total": len(samples),
        "records_valid": valid_count,
        "records_invalid": len(samples) - valid_count,
        "conditions": sorted(manifest.get("condition_counts", {})),
        "parser": {
            "strategy": "first_json_object",
            "numeric_answer_only": True,
            "finite_answer_only": True,
            "string_coercion": False,
        },
        "quantiles": {
            "values": [0.05, 0.25, 0.5, 0.75, 0.95],
            "method": "numpy.percentile(method=linear)",
            "invalid_values_excluded": True,
        },
        "error_metric": {
            "name": "median_absolute_percentage_error",
            "formula": "median(abs(answer - actual_close) / actual_close * 100)",
            "aggregation": "per date, index, and context",
        },
        "interpretation": (
            "Figures describe generated-price sampling variability and prediction "
            "error; they are not causal estimates."
        ),
        "outputs": [
            "price_distribution_samples.csv",
            "price_distribution_summary.csv",
            "price_distribution_metadata.json",
            *(path.name for path in figure_paths),
        ],
    }
    (output / "price_distribution_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _market_returns(
    prices: Iterable[dict[str, str]],
    indices: Iterable[str] = INDEX_ORDER,
) -> dict[str, dict[str, Any]]:
    rows = list(prices)
    price_indices = tuple(dict.fromkeys(str(index) for index in indices))
    result: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        values = {index: float(row[index]) for index in price_indices}
        previous = rows[row_index - 1] if row_index else None
        returns = (
            {
                index: (values[index] / float(previous[index]) - 1.0) * 100.0
                for index in price_indices
            }
            if previous is not None
            else None
        )
        result[row["Date"]] = {
            "date": row["Date"],
            "prices": values,
            "returns_pct": returns,
            "mean_return_pct": sum(returns.values()) / len(returns) if returns else None,
        }
    return result


def select_attribution_dates(
    attribution_rows: Iterable[dict[str, Any]],
    prices: Iterable[dict[str, str]],
    *,
    condition_order: Iterable[tuple[str, str]],
    crash_count: int = 2,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Select common crash and normal dates deterministically."""
    required_order = list(condition_order)
    required_conditions = set(required_order)
    by_date: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in attribution_rows:
        key = (str(row.get("index", "")), str(row.get("context", "")))
        date_value = str(row.get("date", ""))
        if key in required_conditions and date_value:
            by_date[date_value].add(key)

    indices = tuple(dict.fromkeys(index for index, _context in required_order))
    market = _market_returns(prices, indices)
    common = [
        day
        for day, conditions in by_date.items()
        if conditions == required_conditions and market.get(day, {}).get("mean_return_pct") is not None
    ]
    if len(common) < crash_count + 1:
        raise ValueError(
            f"need at least {crash_count + 1} common attribution dates with prices; "
            f"found {len(common)}"
        )
    crash_dates = sorted(
        common,
        key=lambda day: (market[day]["mean_return_pct"], day),
    )[:crash_count]
    normal_date = min(
        (day for day in common if day not in crash_dates),
        key=lambda day: (abs(market[day]["mean_return_pct"]), day),
    )
    selected = sorted(crash_dates + [normal_date])
    return selected, {day: market[day] for day in selected}


def _condition_order(
    records: Iterable[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Return deterministic index/context order.

    With no records this preserves the historical S&P/Russell order.  When
    records are supplied, conditions are discovered from their ``index`` and
    ``context`` fields while keeping each index's without/with ordering.
    """
    if records is None:
        return [(index, context) for index in INDEX_ORDER for context in CONTEXT_ORDER]
    rows = list(records)
    indices: list[str] = []
    contexts_by_index: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        index = str(row.get("index", ""))
        context = str(row.get("context", ""))
        if not index or not context:
            continue
        if index not in indices:
            indices.append(index)
        if context not in contexts_by_index[index]:
            contexts_by_index[index].append(context)
    result: list[tuple[str, str]] = []
    for index in indices:
        contexts = contexts_by_index[index]
        for context in CONTEXT_ORDER:
            if context in contexts:
                result.append((index, context))
        # Preserve non-standard context values without silently dropping them.
        for context in contexts:
            if (index, context) not in result:
                result.append((index, context))
    return result


def _index_label(index: str) -> str:
    """Human-readable label for a condition, including arbitrary tickers."""
    return INDEX_LABELS.get(index, index.upper())


def _context_label(context: str) -> str:
    return {"without": "without context", "with": "with context"}.get(
        context, context
    )


def _prompt_tokens(
    prompt: str,
    tokenizer: Any,
    *,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_len,
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [
        {
            "position": position,
            "token_id": int(token_id),
            "token": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }
        for position, token_id in enumerate(token_ids)
    ]


def _attribution_panel(
    row: dict[str, Any],
    *,
    backward: dict[str, Any] | None,
    input_top_k: int,
    tokenizer: Any | None,
    max_seq_len: int,
    validation_by_position: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact panel from forward output, optionally adding backward data."""
    legacy_backward = row if backward is None and isinstance(row.get("generated_tokens"), list) else None
    effective_backward = backward or legacy_backward
    generated = (effective_backward or {}).get("generated_tokens") if effective_backward is not None else None
    if not isinstance(generated, list) or not generated:
        token_ids = row.get("generated_token_ids")
        if isinstance(token_ids, list) and token_ids:
            generated = [
                {"position": position, "token_id": int(token_id)}
                for position, token_id in enumerate(token_ids)
            ]
        else:
            generated = row.get("generated_tokens")
    if not isinstance(generated, list) or not generated:
        raise ValueError("forward record has no generated output tokens")

    by_output: list[dict[int, float]] = []
    input_metadata: dict[int, dict[str, Any]] = {}
    totals: dict[int, float] = defaultdict(float)
    attribution_enabled = backward is not None
    for output in generated:
        contributions = output.get("top_input_tokens")
        if not attribution_enabled:
            by_output.append({})
            continue
        if not isinstance(contributions, list):
            raise ValueError("backward record token has no top_input_tokens")
        values: dict[int, float] = {}
        for item in contributions:
            position = int(item.get("prompt_position", item["position"]))
            attribution = float(item["attribution"])
            values[position] = attribution
            totals[position] += attribution
            input_metadata.setdefault(position, {"position": position, "token_id": int(item["token_id"]), "token": str(item["token"])})
        by_output.append(values)

    if attribution_enabled and tokenizer is not None:
        prompt = str(row.get("prompt", ""))
        if prompt:
            full_input_tokens = _prompt_tokens(prompt, tokenizer, max_seq_len=max_seq_len)
            selected_positions = [item["position"] for item in full_input_tokens]
            input_metadata = {item["position"]: item for item in full_input_tokens}
        else:
            selected_positions = sorted(sorted(totals, key=lambda position: (-totals[position], position))[:input_top_k])
    elif attribution_enabled:
        selected_positions = sorted(sorted(totals, key=lambda position: (-totals[position], position))[:input_top_k])
    else:
        selected_positions = []
    matrix = [[round(values.get(position, 0.0), 8) for position in selected_positions] for values in by_output]
    input_attribution_complete = bool(attribution_enabled and selected_positions) and all(
        position in values for values in by_output for position in selected_positions
    )
    output_tokens = []
    for output in generated:
        output_record = {
            "position": int(output.get("position", len(output_tokens))),
            "token_id": int(output.get("token_id", -1)),
            "token": output.get("token") if isinstance(output.get("token"), str) else None,
            "log_probability": (float(output["log_probability"]) if output.get("log_probability") is not None else None),
        }
        if "logit" in output:
            output_record["target_logit"] = float(output["logit"])
        validation = (validation_by_position or {}).get(output_record["position"])
        if validation is not None:
            semantic = validation.get("semantic_scope", {})
            random_baseline = validation.get("random", {})
            output_record["semantic_scope_aopc"] = float(semantic["aopc"])
            output_record["random_aopc"] = float(random_baseline["aopc"])
            output_record["semantic_scope_log_probability_delta"] = list(semantic.get("log_probability_delta", []))
            output_record["random_log_probability_delta"] = list(random_baseline.get("log_probability_delta", []))
        output_tokens.append(output_record)
    return {
        "index": str(row.get("index", "")),
        "index_label": _index_label(str(row.get("index", ""))),
        "context": str(row.get("context", "")),
        "context_label": _context_label(str(row.get("context", ""))),
        "prompt_column": str(row.get("prompt_column", "")),
        "prompt": str(row.get("prompt", "")),
        "generated_text": str(row.get("generated_text", "")),
        "prediction": {key: row.get(key) for key in ("predicted_label", "predicted_confidence", "parse_status") if key in row},
        "output_tokens": output_tokens,
        "input_tokens": [input_metadata[position] for position in selected_positions],
        "matrix": matrix,
        "max_attribution": max((value for values in matrix for value in values), default=0.0),
        "input_attribution_complete": input_attribution_complete,
        "attribution_enabled": attribution_enabled,
        "validation_summary": row.get("validation_summary"),
    }


def build_attribution_data(
    attribution_rows: Iterable[dict[str, Any]],
    selected_dates: Iterable[str],
    *,
    input_top_k: int = 15,
    market: dict[str, dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    max_seq_len: int = 256,
    validation_rows: Iterable[dict[str, Any]] | None = None,
    backward_rows: Iterable[dict[str, Any]] | None = None,
    condition_order: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build dashboard data from forward outputs and optional backward scores."""
    if input_top_k < 1:
        raise ValueError("input_top_k must be positive")
    selected = list(selected_dates)
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in attribution_rows:
        key = (str(row.get("date", "")), str(row.get("index", "")), str(row.get("context", "")))
        if key[0] in selected:
            if key in by_key:
                raise ValueError(f"duplicate attribution record for {key}")
            by_key[key] = row
    backward_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for backward in backward_rows or ():
        key = (str(backward.get("date", "")), str(backward.get("index", "")), str(backward.get("context", "")))
        if key in backward_by_key:
            raise ValueError(f"duplicate backward record for {key}")
        backward_by_key[key] = backward
    validation_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for validation in validation_rows or ():
        key = (
            str(validation.get("date", "")),
            str(validation.get("index", "")),
            str(validation.get("context", "")),
        )
        if key in validation_by_key:
            raise ValueError(f"duplicate validation record for {key}")
        validation_by_key[key] = validation

    required_order = list(condition_order or _condition_order())
    dates = []
    for day in selected:
        conditions = []
        for index, context in required_order:
            row = by_key.get((day, index, context))
            if row is None:
                raise ValueError(f"missing attribution record for {day}/{index}/{context}")
            validation = validation_by_key.get((day, index, context))
            validation_by_position = {
                int(token["position"]): token
                for token in (validation or {}).get("generated_tokens", [])
            }
            row_for_panel = dict(row)
            if validation is not None:
                row_for_panel["validation_summary"] = validation.get("summary")
            conditions.append(
                _attribution_panel(
                    row_for_panel,
                    backward=backward_by_key.get((day, index, context)),
                    input_top_k=input_top_k,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    validation_by_position=validation_by_position,
                )
            )
        dates.append(
            {
                "date": day,
                "market": (market or {}).get(day),
                "conditions": conditions,
            }
        )
    return {
        "metric": "semantic_scope_target_logit_gradient_l2_norm",
        "normalization": "none",
        "input_top_k": input_top_k,
        "attribution_enabled": backward_rows is not None,
        "validation": "semantic_scope_aopc_with_random_baseline"
        if validation_rows is not None
        else None,
        "dates": dates,
    }


def _load_template(name: str) -> str:
    return (
        Path(__file__).resolve().parents[1] / "static" / name
    ).read_text(encoding="utf-8")


def render_attribution_html(data: dict[str, Any]) -> str:
    """Embed compact attribution data into the standalone dashboard template."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    template = _load_template("attribution_dashboard.html")
    script = _load_template("attribution_dashboard.js")
    return template.replace("__ATTRIBUTION_DATA__", payload).replace("__ATTRIBUTION_SCRIPT__", script)


def plot_uncertainty(
    records: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    condition_order: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Write Temperature Scope curves and entropy comparison curves."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = list(records)
    required_order = list(condition_order or _condition_order(rows))
    indices = list(dict.fromkeys(index for index, _context in required_order))
    contexts = list(dict.fromkeys(context for _index, context in required_order))
    if not indices or not contexts:
        raise ValueError("uncertainty records contain no index/context conditions")
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date", "index", "context", "layer", "entropy_nats",
        "normalized_entropy", "perplexity", "top1_probability", "topk_mass",
        "effective_inverse_temperature", "effective_temperature",
    ]
    with (output_dir / "final_layer_uncertainty.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)
    palette = (
        "#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c",
        "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#c026d3",
    )
    colors = {
        index: palette[position % len(palette)]
        for position, index in enumerate(indices)
    }
    for context in contexts:
        figure, axes = plt.subplots(
            len(indices),
            1,
            figsize=(15, 10),
            sharex=True,
            squeeze=False,
        )
        for index_position, index in enumerate(indices):
            axis = axes[index_position][0]
            series = sorted(
                (row for row in rows if row["index"] == index and row["context"] == context),
                key=lambda row: row["date"],
            )
            if not series:
                continue
            dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
            values = [row["effective_temperature"] for row in series]
            axis.plot(dates, values, linewidth=1.25, color=colors[index])
            axis.set_title(_index_label(index), loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Effective temperature")
            axis.grid(alpha=0.25)
        axes[-1][0].set_xlabel("Date")
        locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        axes[-1][0].xaxis.set_major_locator(locator)
        axes[-1][0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        figure.suptitle(
            f"Temperature Scope uncertainty · {_context_label(context)}", fontsize=14
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"final_layer_effective_temperature_{context}_context.png",
            dpi=180,
        )
        plt.close(figure)

        # Retain an entropy comparison plot under the historical name.
        entropy_figure, entropy_axes = plt.subplots(
            len(indices), 1, figsize=(15, 10), sharex=True, squeeze=False
        )
        for index_position, index in enumerate(indices):
            axis = entropy_axes[index_position][0]
            series = sorted(
                (row for row in rows if row["index"] == index and row["context"] == context),
                key=lambda row: row["date"],
            )
            if not series:
                continue
            dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
            axis.plot(
                dates,
                [row["entropy_nats"] for row in series],
                linewidth=1.25,
                color=colors[index],
            )
            axis.set_title(_index_label(index), loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Entropy (nats)")
            axis.grid(alpha=0.25)
        entropy_axes[-1][0].set_xlabel("Date")
        entropy_locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        entropy_axes[-1][0].xaxis.set_major_locator(entropy_locator)
        entropy_axes[-1][0].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(entropy_locator)
        )
        entropy_figure.suptitle(
            f"Final-layer entropy comparison · {_context_label(context)}", fontsize=14
        )
        entropy_figure.tight_layout()
        entropy_figure.savefig(
            output_dir / f"final_layer_entropy_{context}_context.png", dpi=180
        )
        plt.close(entropy_figure)


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    candidate = path.parent / "metadata.json"
    if not candidate.is_file():
        candidate = path.with_suffix(path.suffix + ".metadata.json")
    if not candidate.is_file():
        return {}
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact metadata must be an object: {candidate}")
    return value


def _artifact_type(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str | None:
    value = metadata.get("artifact_type")
    if isinstance(value, str):
        return value
    values = {row.get("artifact_type") for row in rows if row.get("artifact_type")}
    return next(iter(values)) if len(values) == 1 else None


def _parent_hash(metadata: dict[str, Any]) -> str | None:
    for key in ("parent_forward_artifact_sha256", "parent_sha256", "parent_artifact_sha256"):
        value = metadata.get(key)
        if isinstance(value, str):
            return value
    for key in ("parent_forward_artifact", "parent_artifact", "parent"):
        nested = metadata.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("sha256"), str):
            return nested["sha256"]
    return None


def visualize_prompt_results(
    *,
    forward_path: str | Path,
    uncertainty_paths: dict[tuple[str, str], str | Path],
    backward_path: str | Path | None = None,
    validation_path: str | Path | None = None,
    prices_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path,
    input_top_k: int = 15,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
    max_seq_len: int = 256,
) -> Path:
    """Create uncertainty figures and a forward-output dashboard.

    Generated output panels are always available from the forward artifact.
    Input attribution is enabled only when an explicit backward artifact whose
    parent hash matches the forward JSONL is supplied.
    """
    output = Path(output_dir)
    uncertainty = load_final_layer_uncertainty(uncertainty_paths)
    forward_source = Path(forward_path)
    forward_rows = read_jsonl(forward_source)
    forward_metadata = _sidecar_metadata(forward_source)
    if _artifact_type(forward_source, forward_rows, forward_metadata) != FORWARD_ARTIFACT_TYPE:
        raise ValueError("prompt visualization requires a forward generated-output artifact")
    forward_hash = sha256_file(forward_source)
    backward_rows = None
    backward_hash = None
    if backward_path is not None:
        backward_source = Path(backward_path)
        backward_rows = read_jsonl(backward_source)
        backward_metadata = _sidecar_metadata(backward_source)
        if _artifact_type(backward_source, backward_rows, backward_metadata) not in BACKWARD_ARTIFACT_TYPES:
            raise ValueError("attribution panel requires a backward artifact")
        parent = _parent_hash(backward_metadata)
        if parent != forward_hash:
            raise ValueError(f"backward artifact parent hash mismatch: expected {forward_hash}, got {parent}")
        backward_hash = sha256_file(backward_source)
    validation_rows = _read_jsonl(Path(validation_path)) if backward_path is not None and validation_path is not None and Path(validation_path).is_file() else None
    condition_order = _condition_order(uncertainty) or _condition_order(forward_rows)
    indices = tuple(dict.fromkeys(index for index, _context in condition_order))
    prices = _load_prices(Path(prices_path), indices) if Path(prices_path).is_file() else []
    if prices:
        selected_dates, market = select_attribution_dates(forward_rows, prices, condition_order=condition_order)
    else:
        available_dates = sorted({str(row.get("date", "")) for row in forward_rows if row.get("date")})
        selected_dates, market = available_dates[:3], {}
        if not selected_dates:
            raise ValueError("forward artifact has no dates for dashboard")
    tokenizer = None
    if backward_rows is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    data = build_attribution_data(
        forward_rows,
        selected_dates,
        input_top_k=input_top_k,
        market=market,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        validation_rows=validation_rows,
        backward_rows=backward_rows,
        condition_order=condition_order,
    )
    plot_uncertainty(uncertainty, output, condition_order=condition_order)
    output.mkdir(parents=True, exist_ok=True)
    (output / "attribution_dashboard.html").write_text(render_attribution_html(data), encoding="utf-8")
    (output / "attribution_selected_dates.json").write_text(json.dumps({"selection": "two largest negative equal-weight index returns and one closest-to-zero return" if prices else "first available forward dates", "dates": selected_dates, "market": market}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage = {"forward_records": len(forward_rows), "uncertainty_records": len(uncertainty), "backward_records": len(backward_rows or []), "dates": len(selected_dates), "attribution_enabled": backward_rows is not None}
    visualization_metadata = {"source": {"forward": {"artifact": str(forward_source), "sha256": forward_hash}, "backward": ({"artifact": str(backward_path), "sha256": backward_hash} if backward_path is not None else None)}, "source_artifact": str(forward_source), "source_artifact_sha256": forward_hash, "coverage": coverage, "attribution_panel": "enabled" if backward_rows is not None else "disabled: no backward artifact", "interpretation": "Generated outputs and prediction fields are descriptive; attribution is a local first-order sensitivity readout, not a causal proof."}
    (output / "prompt_visualization_metadata.json").write_text(json.dumps(visualization_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
