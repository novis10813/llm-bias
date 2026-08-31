#!/usr/bin/env python3
"""Render figures for the activation-patching causal-tracing result.

Reads only compact derived artifacts (no model, no raw activations):
  1. Phase 3 discovery localization summary  -> main panel (a) curves
  2. Phase 3 discovery patch records + Phase 0 baseline -> direction-split supplement
  3. Frozen held-out confirmation evaluation -> main panel (b) 3x3 matrix

Outputs PDF+PNG figures plus figures_provenance.json into the output directory.

Canonical document: docs/activation-patching-causal-tracing/proposal.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib import gridspec  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY = (
    "artifacts/qwen3.5-4b/jspace-causal-tracing/runs/"
    "causal-trace-phase3-layer-span-discovery-20260830"
)
DEFAULT_CONFIRMATION = (
    "artifacts/qwen3.5-4b/jspace-causal-tracing/runs/"
    "causal-trace-confirmation-test-20260830"
)
DEFAULT_OUTPUT = "docs/assets/jspace-causal-tracing"

SPAN_ORDER = ("all_evidence", "instruction_context", "final_position")
SPAN_LABELS = {
    "all_evidence": "All evidence",
    "instruction_context": "Instruction context",
    "final_position": "Final position",
}
SPAN_SHORT = {
    "all_evidence": "evidence",
    "instruction_context": "context",
    "final_position": "final",
}
SHORT_TO_SPAN = {
    "evidence": "all_evidence",
    "instruction_context": "instruction_context",
    "final_position": "final_position",
}
DIRECTIONS = ("positive_to_negative", "negative_to_positive")
DIRECTION_LABELS = {
    "positive_to_negative": "positive \u2192 negative",
    "negative_to_positive": "negative \u2192 positive",
}
LAYER_RE = re.compile(r"^\w+_L(\d+)$")
HANDOFF_RE = re.compile(r"^between_L(\d+)_and_L(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def layer_of(layer_condition: str) -> int:
    match = LAYER_RE.match(layer_condition)
    if match is None:
        raise ValueError(f"unexpected layer condition: {layer_condition!r}")
    return int(match.group(1))


def _quantile(values: list[float], probability: float) -> float:
    # Mirrors llm_bias.jspace_intervention.activation_patching._confirmation_quantile
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(values: list[float], *, seed: int, samples: int) -> list[float]:
    # Mirrors _confirmation_bootstrap_ci in activation_patching.py
    if not values or samples < 1:
        raise ValueError("cannot bootstrap an empty ticker collection")
    rng = random.Random(seed)
    count = len(values)
    bootstrap_means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return [_quantile(bootstrap_means, 0.025), _quantile(bootstrap_means, 0.975)]


def compute_direction_split(
    records: list[dict],
    baseline_records: list[dict],
    *,
    seed: int,
    samples: int,
) -> tuple[dict, dict]:
    """Aggregate direction-specific equal-ticker transfer from raw records.

    Unit is the ticker: pair (trial) values are averaged within
    (ticker, layer, span, direction), then tickers receive equal weight.
    Returns (direction_cells, bidirectional_cells), each keyed by
    (layer, span) or (layer, span, direction) with mean + CI + n_tickers.
    """
    eligible_pairs = {
        str(rec["pair_record_id"])
        for rec in baseline_records
        if bool(rec.get("clean_decision_match"))
    }
    trials: dict[tuple, list[float]] = defaultdict(list)
    for rec in records:
        if str(rec.get("pair_record_id")) not in eligible_pairs:
            continue
        source = float(rec["source_clean_margin"])
        target = float(rec["target_clean_margin"])
        denominator = source - target
        if denominator == 0.0:
            continue
        transfer = float(rec["delta_margin"]) / denominator
        if not math.isfinite(transfer):
            continue
        key = (
            str(rec["ticker"]),
            layer_of(str(rec["layer_condition"])),
            str(rec["span_condition"]),
            str(rec["patching_direction"]),
        )
        trials[key].append(transfer)
    if not trials:
        raise ValueError("no eligible phase3 records")
    ticker_means = {key: sum(values) / len(values) for key, values in trials.items()}
    cells: dict[tuple, set[str]] = defaultdict(set)
    for (ticker, layer, span, direction) in ticker_means:
        cells[(layer, span, direction)].add(ticker)
    direction_cells: dict[tuple, dict] = {}
    for (layer, span, direction), tickers in cells.items():
        values = [
            ticker_means[(t, layer, span, direction)] for t in sorted(tickers)
        ]
        direction_cells[(layer, span, direction)] = {
            "mean": sum(values) / len(values),
            "ci": bootstrap_ci(values, seed=seed, samples=samples),
            "n_tickers": len(values),
        }
    bidir_cells: dict[tuple, dict] = {}
    for (layer, span, _direction) in list(cells):
        if direction_cells.get((layer, span, DIRECTIONS[0])) is None:
            continue
        tickers = cells[(layer, span, DIRECTIONS[0])] & cells[(layer, span, DIRECTIONS[1])]
        values = [
            (
                ticker_means[(t, layer, span, DIRECTIONS[0])]
                + ticker_means[(t, layer, span, DIRECTIONS[1])]
            )
            / 2.0
            for t in sorted(tickers)
        ]
        bidir_cells[(layer, span)] = {
            "mean": sum(values) / len(values),
            "ci": bootstrap_ci(values, seed=seed, samples=samples),
            "n_tickers": len(values),
        }
    return direction_cells, bidir_cells


def check_localization(bidir_cells: dict, localization: dict) -> float:
    """Compare recomputed bidirectional means with the localized summary."""
    worst = 0.0
    for rec in localization["records"]:
        key = (int(rec["layer"]), str(rec["span_condition"]))
        if key not in bidir_cells:
            raise ValueError(f"localization key missing from recomputation: {key}")
        worst = max(
            worst,
            abs(bidir_cells[key]["mean"] - float(rec["equal_ticker_bidirectional_normalized_transfer"])),
        )
    if worst > 1e-8:
        raise ValueError(
            f"direction-split aggregation mismatch: max |diff| vs phase3_localization.json is {worst:.3e}"
        )
    return worst


def parse_handoffs(localization: dict) -> list[tuple[float, str, str]]:
    handoffs = []
    for key, value in localization["handoff_intervals"].items():
        match = HANDOFF_RE.match(value)
        if match is None:
            raise ValueError(f"unexpected handoff interval: {value!r}")
        first, second = key.split("_to_")
        handoffs.append(
            ((int(match.group(1)) + int(match.group(2))) / 2.0, SHORT_TO_SPAN[first], SHORT_TO_SPAN[second])
        )
    handoffs.sort()
    return handoffs


def style_setup() -> None:
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    SPAN_COLORS.clear()
    SPAN_COLORS.update(
        {
            "all_evidence": pal[1],
            "instruction_context": pal[3],
            "final_position": pal[5],
        }
    )


SPAN_COLORS: dict[str, str] = {}


def finish_panel(ax, *, frame: bool = True) -> None:
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax.grid(False)
    sns.despine(left=True, bottom=True)
    if frame:
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)


def span_series(
    cells: dict[tuple, dict],
    span: str,
    direction: str | None,
) -> tuple[list[int], list[float], list[float], list[float]]:
    layers: list[int] = []
    means: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for layer in sorted({key[0] for key in cells}):
        key = (layer, span, direction) if direction is not None else (layer, span)
        if key not in cells:
            continue
        entry = cells[key]
        layers.append(layer)
        means.append(entry["mean"])
        lo.append(entry["ci"][0])
        hi.append(entry["ci"][1])
    return layers, means, lo, hi


def draw_curves(
    ax,
    cells: dict[tuple, dict],
    *,
    direction: str | None,
    handoffs: list[tuple[float, str, str]] | None,
    annotate_peaks: bool,
    annotate_l16: bool,
    label_handoffs: bool = True,
) -> None:
    series = {span: span_series(cells, span, direction) for span in SPAN_ORDER}
    top = max(
        (value for span in SPAN_ORDER for value in list(series[span][1]) + list(series[span][3]))
        or [1.0]
    )
    ax.set_xlim(0, 30)
    ax.set_ylim(0.0, max(1.0, top) + 0.08)
    for span in SPAN_ORDER:
        layers, means, lo, hi = series[span]
        color = SPAN_COLORS[span]
        ax.fill_between(layers, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(layers, means, color=color, linewidth=2.2, zorder=4, label=SPAN_LABELS[span])
    if handoffs:
        for index, (x, first, second) in enumerate(handoffs):
            ax.axvline(x, color="dimgrey", linestyle=":", linewidth=1.0, zorder=1)
            if not label_handoffs:
                continue
            ax.text(
                x,
                ax.get_ylim()[1] - 0.015,
                f"{SPAN_SHORT[first]} \u2192 {SPAN_SHORT[second]}\n(L{int(x - 0.5)}\u2013L{int(x + 0.5)})",
                ha="right" if index % 2 == 0 else "left",
                va="top",
                fontsize=9,
                color="dimgrey",
            )
    if annotate_peaks:
        for span in SPAN_ORDER:
            layers, means, _, _ = span_series(cells, span, direction)
            peak = max(range(len(layers)), key=lambda i: means[i])
            x, y = layers[peak], means[peak]
            ha = "right" if x >= 28 else ("left" if x <= 2 else "center")
            ax.annotate(
                f"L{x}: {y:.2f}",
                xy=(x, y),
                xytext=(x, y + 0.045),
                ha=ha,
                fontsize=9,
                color="dimgrey",
            )
    if annotate_l16:
        layers, means, _, _ = span_series(cells, "instruction_context", direction)
        if 16 in layers:
            i = layers.index(16)
            ax.annotate(
                f"L16: {means[i]:.2f}",
                xy=(16, means[i]),
                xytext=(16, means[i] + 0.045),
                ha="center",
                fontsize=9,
                color="dimgrey",
            )
    ax.set_xticks(range(0, 31, 5))
    ax.set_xlabel("Layer", fontsize=12, labelpad=8, color="dimgrey")
    finish_panel(ax)


def legend_lines() -> list[Line2D]:
    return [
        Line2D([0], [0], color=SPAN_COLORS[span], linewidth=2.2, label=SPAN_LABELS[span])
        for span in SPAN_ORDER
    ]


def top_legend(ax) -> None:
    ax.legend(
        handles=legend_lines(),
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="lightgrey",
        labelcolor="dimgrey",
    )


def render_main(localization: dict, confirmation: dict, output_dir: Path) -> list[str]:
    handoffs = parse_handoffs(localization)
    cells: dict[tuple, dict] = {
        (int(rec["layer"]), rec["span_condition"]): {
            "mean": rec["equal_ticker_bidirectional_normalized_transfer"],
            "ci": rec["ticker_bootstrap_95_ci"],
            "n_tickers": rec["ticker_count"],
        }
        for rec in localization["records"]
    }
    fig = plt.figure(figsize=(14, 5.8), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1.0], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_curves(ax_a, cells, direction=None, handoffs=handoffs,
                annotate_peaks=True, annotate_l16=False)
    ax_a.set_ylabel("Normalized transfer T (bidirectional)", fontsize=12, labelpad=8, color="dimgrey")
    ax_a.set_title(r"$\bf{(a)}$  Discovery: source-outcome sufficiency by span and layer",
                   loc="left", fontsize=12, pad=40, color="dimgrey")
    top_legend(ax_a)

    # (b) held-out 3x3 confirmation matrix
    matrix = confirmation["matrix"]
    row_order = []
    col_order = []
    for cell in matrix:
        if cell["layer_condition"] not in row_order:
            row_order.append(cell["layer_condition"])
        if cell["span_condition"] not in col_order:
            col_order.append(cell["span_condition"])
    grid = [[0.0] * len(col_order) for _ in row_order]
    for cell in matrix:
        grid[row_order.index(cell["layer_condition"])][col_order.index(cell["span_condition"])] = (
            float(cell["equal_ticker_mean_transfer"])
        )
    diagonal = {
        (
            row_order.index(c["layer_condition"]),
            col_order.index(c["primary_span"]),
        )
        for c in confirmation["contrasts"]
    }
    COL_LABELS = {
        "all_evidence": "evidence",
        "instruction_context": "instruction\ncontext",
        "final_position": "final\nposition",
    }
    row_labels = [f"L{layer_of(row)}" for row in row_order]
    col_labels = [COL_LABELS[span] for span in col_order]
    sns.heatmap(
        grid,
        annot=True,
        fmt=".3f",
        cmap="rocket",
        vmin=0.0,
        vmax=1.0,
        ax=ax_b,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"shrink": 0.85, "label": "Normalized transfer T"},
        annot_kws={"size": 11},
    )
    for i, j in sorted(diagonal):
        ax_b.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="black", linewidth=1.8, zorder=5))
        ax_b.texts[i * len(col_order) + j].set_fontweight("bold")
    ax_b.set_title(r"$\bf{(b)}$  Held-out confirmation (frozen 3\u00d73)",
                   loc="left", fontsize=12, pad=7, color="dimgrey")
    finish_panel(ax_b)
    fig.text(
        0.012,
        0.012,
        f"Held-out: success=true \u2014 {confirmation['eligible_tickers']} tickers, "
        f"{confirmation['eligible_pairs']} pairs; black-outlined cells are the frozen "
        "primary diagonal conditions.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    return _save(fig, output_dir, "causal_tracing_position_transfer")


def render_direction_split(cells: dict, handoffs: list, output_dir: Path, seed: int, samples: int) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=150)
    for ax, direction in zip(axes, DIRECTIONS):
        draw_curves(ax, cells, direction=direction, handoffs=handoffs,
                    annotate_peaks=False, annotate_l16=True, label_handoffs=False)
        ax.set_ylabel("Normalized transfer T (equal-ticker)", fontsize=12, labelpad=8, color="dimgrey")
        ax.set_title(
            rf"$\bf{{({chr(ord('a') + DIRECTIONS.index(direction))})}}$  {DIRECTION_LABELS[direction]}",
            loc="left",
            fontsize=12,
            pad=40,
            color="dimgrey",
        )
    top_legend(axes[0])
    p2n = cells[(16, "instruction_context", "positive_to_negative")]["mean"]
    n2p = cells[(16, "instruction_context", "negative_to_positive")]["mean"]
    intervals = "; ".join(
        f"L{int(x - 0.5)}\u2013L{int(x + 0.5)}"
        for x, _first, _second in handoffs
    )
    fig.text(
        0.012,
        0.012,
        "positive \u2192 negative = patch Buy-condition states into the Sell prompt. "
        f"Ticker-bootstrap 95% CI (seed {seed}, {samples} samples). Dotted lines mark the "
        f"position-transfer intervals ({intervals}). Layer ordering holds in both directions; "
        f"instruction-context transfer is asymmetric at L16 ({p2n:.2f} vs {n2p:.2f}).",
        ha="left",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    return _save(fig, output_dir, "causal_tracing_direction_split")


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        names.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    plt.close(fig)
    return names


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-run", default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation-run", default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    discovery = resolve(args.discovery_run)
    confirmation = resolve(args.confirmation_run)
    output_dir = resolve(args.output_dir)

    style_setup()

    localization_path = discovery / "analyze/phase3_localization.json"
    records_path = discovery / "forward/phase3_records.jsonl"
    baseline_path = discovery / "forward/phase0_baseline.jsonl"
    confirmation_path = confirmation / "analyze/confirmation_evaluation_v1.json"

    localization = read_json(localization_path)
    records = read_jsonl(records_path)
    baseline_records = read_jsonl(baseline_path)
    confirmation = read_json(confirmation_path)
    if not confirmation.get("success"):
        raise SystemExit(f"confirmation evaluation is not success=true: {confirmation_path}")

    direction_cells, bidir_cells = compute_direction_split(
        records,
        baseline_records,
        seed=args.bootstrap_seed,
        samples=args.bootstrap_samples,
    )
    max_diff = check_localization(bidir_cells, localization)
    handoffs = parse_handoffs(localization)

    outputs = []
    outputs += render_main(localization, confirmation, output_dir)
    outputs += render_direction_split(direction_cells, handoffs, output_dir,
                                      args.bootstrap_seed, args.bootstrap_samples)

    inputs = {
        str(localization_path.relative_to(ROOT)): sha256_file(localization_path),
        str(records_path.relative_to(ROOT)): sha256_file(records_path),
        str(baseline_path.relative_to(ROOT)): sha256_file(baseline_path),
        str(confirmation_path.relative_to(ROOT)): sha256_file(confirmation_path),
    }
    provenance = {
        "artifact_type": "causal_tracing_figures_provenance",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/render_causal_tracing_figures.py",
        "git_commit": git_commit(),
        "inputs": inputs,
        "aggregation": {
            "direction_split": (
                "pair mean within (ticker, layer, span, direction); equal-ticker mean; "
                "ticker bootstrap 95% CI (same procedure as the frozen confirmation evaluation)"
            ),
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "sanity_check": {
            "bidirectional_max_abs_diff_vs_phase3_localization": max_diff,
        },
        "outputs": outputs,
        "interpretation_limit": (
            "descriptive resample-patching sufficiency across positions; "
            "not an attention map, mediation decomposition, or unique information route"
        ),
    }
    provenance_path = output_dir / "figures_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(output_dir)
    print(f"sanity max|diff| vs phase3_localization.json: {max_diff:.3e}")


if __name__ == "__main__":
    main()
