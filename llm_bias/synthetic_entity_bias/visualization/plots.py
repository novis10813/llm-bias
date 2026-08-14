"""Static PNG/SVG plots for synthetic entity-bias summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_bias.synthetic_entity_bias.analysis.diagnostics import (
    baseline_statistics,
    entity_distribution_diagnostics,
    localization_transition_diagnostics,
    temperature_null_diagnostics,
    temperature_null_entity_rows,
)

from .contract import TEMPLATE_ORDER, TIER_ORDER
from .summaries import summarize_sector, summarize_ticker
from .theme import (
    DASHES,
    HATCHES,
    LIGHT_COLORS,
    LINE_WIDTH,
    MARKERS,
    MARKER_SIZE,
    PAPER_DPI,
)


def _save(fig: Any, output_dir: Path, name: str) -> dict[str, Path]:
    paths = {suffix: output_dir / f"{name}.{suffix}" for suffix in ("png", "svg", "pdf")}
    for suffix, path in paths.items():
        fig.savefig(path, dpi=PAPER_DPI if suffix == "png" else None, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return paths


def _paper_header(fig: Any, title: str, subtitle: str, note: str) -> None:
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.945, subtitle, ha="left", va="top", fontsize=9, color="#52514e")
    fig.text(0.01, 0.012, f"Note: {note}", ha="left", va="bottom", fontsize=7.5, color="#52514e", wrap=True)


def _panel_label(ax: Any, label: str) -> None:
    ax.text(-0.08, 1.01, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom", clip_on=False)


def _zero_label(ax: Any, *, vertical: bool) -> None:
    if vertical:
        ax.annotate("No entity effect", xy=(0, 0.98), xycoords=("data", "axes fraction"), xytext=(4, -2), textcoords="offset points", rotation=90, va="top", fontsize=7, color="#52514e")
    else:
        ax.annotate("No entity effect", xy=(0.99, 0), xycoords=("axes fraction", "data"), xytext=(-2, 4), textcoords="offset points", ha="right", fontsize=7, color="#52514e")


def _style(ax: Any, title: str, ylabel: str = "Delta expected score") -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#c3c2b7")


def _line_kwargs(template: str, *, markers: bool = False) -> dict[str, Any]:
    return {
        "color": LIGHT_COLORS[template],
        "linewidth": LINE_WIDTH,
        "linestyle": DASHES[template],
        "marker": MARKERS[template] if markers else None,
        "markersize": MARKER_SIZE if markers else None,
        "markevery": 4 if markers else None,
        "solid_capstyle": "round",
        "label": template.title(),
    }


def plot_effect_distribution(run: Any, output_dir: Path) -> dict[str, Path]:
    n = len(run.entity_pool)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    for template in TEMPLATE_ORDER:
        values = np.array([float(row["delta_expected_score"]) for row in run.results if row["template"] == template])
        axes[0].hist(values, bins=30, histtype="step", color=LIGHT_COLORS[template], linewidth=LINE_WIDTH, linestyle=DASHES[template], label=template.title())
        ordered = np.sort(values)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), **_line_kwargs(template))
    for ax in axes:
        ax.axvline(0, color="#898781", linewidth=1)
        ax.set_xlabel("Entity minus baseline expected score")
        ax.legend(frameon=False)
    _style(axes[0], "Distribution", "Count")
    _style(axes[1], "Empirical cumulative distribution", "Cumulative fraction")
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
    _zero_label(axes[0], vertical=True)
    _zero_label(axes[1], vertical=True)
    _paper_header(fig, "Entity names shift expected scores differently across prompt contexts", f"ΔE = entity expected score − matched ‘The company’ baseline; n = {n:,} entities per context.", "Expected scores use the restricted nine-label distribution. ΔE is descriptive and is not, by itself, a standalone causal effect.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "entity_effect_distribution")


def plot_tail_diagnostics(run: Any, output_dir: Path) -> dict[str, Path]:
    summaries = {row["template"]: row for row in entity_distribution_diagnostics(run)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    for index, (ax, template) in enumerate(zip(axes, TEMPLATE_ORDER, strict=True)):
        values = np.asarray([float(row["delta_expected_score"]) for row in run.results if row["template"] == template])
        summary = summaries[template]
        ax.hist(values, bins=40, color=LIGHT_COLORS[template], alpha=0.18, edgecolor=LIGHT_COLORS[template], linewidth=1.2, hatch=HATCHES[template], label=f"{template.title()} ΔE")
        ax.set_yscale("symlog", linthresh=1)
        ax.axvline(0, color="#898781", linewidth=1)
        ax.axvline(summary["mean"], color=LIGHT_COLORS[template], linewidth=2, linestyle="-", label=f"Mean {summary['mean']:+.3f}")
        ax.axvline(summary["median"], color=LIGHT_COLORS[template], linewidth=2, linestyle="--", label=f"Median {summary['median']:+.3f}")
        ax.axvspan(summary["q05"], summary["q95"], color=LIGHT_COLORS[template], alpha=0.08, label="5th–95th percentile")
        ax.set_xlabel("Entity minus baseline expected score")
        ax.legend(loc="upper right", frameon=False, fontsize=7)
        _style(ax, f"{template.title()} context", "Count (symlog)")
        _panel_label(ax, f"({chr(97 + index)})")
        _zero_label(ax, vertical=True)
        shape = (
            f"skew={summary['skewness']:.2f}\nexcess kurtosis={summary['excess_kurtosis']:.2f}"
            if summary["skewness"] is not None
            else "shape unavailable"
        )
        ax.text(
            0.02,
            0.96,
            f"{shape}\nmin={summary['min']:+.3f}; max={summary['max']:+.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            color="#52514e",
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
    _paper_header(fig, "Entity-effect distributions contain asymmetric and heavy tails", f"Each panel contains n = {len(run.entity_pool):,} entity-level ΔE values; shaded spans show the 5th–95th percentiles.", "Symlog counts keep sparse tails visible. Skewness and kurtosis describe distribution shape; a histogram alone does not establish bimodality or a latent mixture.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "entity_effect_tail_diagnostics")


def plot_baseline_movement(run: Any, output_dir: Path) -> dict[str, Path]:
    rows = baseline_statistics(run)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, facecolor="#fcfcfb")
    for index, (ax, row) in enumerate(zip(axes, rows, strict=True)):
        template = row["template"]
        baseline = row["baseline_expected_score"]
        mean = row["entity_expected_score_mean"]
        median = row["entity_expected_score_median"]
        ax.plot([0, 1], [baseline, mean], color=LIGHT_COLORS[template], linewidth=LINE_WIDTH, linestyle=DASHES[template], marker=MARKERS[template], markersize=MARKER_SIZE, label="Baseline → entity mean")
        ax.errorbar(1, mean, yerr=[[mean - row["entity_expected_score_q05"]], [row["entity_expected_score_q95"] - mean]], color=LIGHT_COLORS[template], capsize=4, linewidth=1.5, label="Entity 5th–95th percentile")
        ax.scatter([1], [median], s=MARKER_SIZE ** 2, facecolor="#fcfcfb", edgecolor=LIGHT_COLORS[template], linewidth=2, marker=MARKERS[template], label=f"Entity median {median:+.3f}")
        ax.axhline(0, color="#898781", linewidth=1)
        ax.set_xticks([0, 1], ["The company\nbaseline", "Named entities"])
        ax.set_ylim(-4.2, 4.2)
        ax.legend(loc="lower left", frameon=False, fontsize=7)
        _style(ax, f"{template.title()} context", "Restricted expected score")
        _panel_label(ax, f"({chr(97 + index)})")
        ax.text(
            0.03,
            0.96,
            f"mean movement={row['mean_movement_from_baseline']:+.3f}\nmedian movement={row['median_movement_from_baseline']:+.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            color="#52514e",
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
    _paper_header(fig, "Named entities move expected scores away from matched generic baselines", "Points compare each prompt context’s single ‘The company’ baseline with the entity-level mean, median, and 5th–95th percentile.", "Scores use only the restricted labels mapped to −4…+4. Movement is a matched prompt contrast and is not, by itself, a standalone causal entity effect.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "baseline_entity_movement")


def plot_temperature_null(run: Any, output_dir: Path) -> dict[str, Path]:
    entity_rows = temperature_null_entity_rows(run)
    summaries = {row["template"]: row for row in temperature_null_diagnostics(run)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    for index, (ax, template) in enumerate(zip(axes, TEMPLATE_ORDER, strict=True)):
        rows = [row for row in entity_rows if row["template"] == template and row["fit_status"] == "ok"]
        observed = np.asarray([row["observed_delta_expected_score"] for row in rows])
        predicted = np.asarray([row["null_delta_expected_score"] for row in rows])
        summary = summaries[template]
        ax.scatter(predicted, observed, s=MARKER_SIZE ** 2, alpha=0.25, color=LIGHT_COLORS[template], marker=MARKERS[template], edgecolors="#fcfcfb", linewidths=1, label=f"{template.title()} entities")
        lower = min(float(np.min(predicted)), float(np.min(observed)))
        upper = max(float(np.max(predicted)), float(np.max(observed)))
        ax.plot([lower, upper], [lower, upper], color="#898781", linewidth=1, linestyle="--", label="Observed = null")
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.axvline(0, color="#c3c2b7", linewidth=1)
        ax.set_xlabel("Temperature-null predicted ΔE")
        ax.set_ylabel("Observed ΔE")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"{template.title()} context\nR²={summary['expected_score_null_r2']:.3f}; mean difference={summary['expected_score_difference_from_null_mean']:+.3f}", loc="left", fontweight="bold", fontsize=9.5)
        _panel_label(ax, f"({chr(97 + index)})")
        ax.spines[["top", "right"]].set_visible(False)
    _paper_header(fig, "A one-dimensional temperature null does not explain every entity effect", f"Each point is one entity; n = {len(run.entity_pool):,} per context. The null fits pᵢ(T) ∝ p₀ᵢ^(1/T) to the persisted nine-point probabilities.", "The null allows only sharpening or flattening of the matched baseline shape. Fitted probability temperature is not the artifact effective-temperature field, is not a recovered model logit temperature, and does not constitute a new model experiment.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "temperature_null_diagnostics")


def plot_tier(run: Any, output_dir: Path) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, facecolor="#fcfcfb")
    for ax, template in zip(axes, TEMPLATE_ORDER, strict=True):
        groups = [[float(row["delta_expected_score"]) for row in run.results if row["template"] == template and row["familiarity_tier"] == tier] for tier in TIER_ORDER]
        box = ax.boxplot(groups, tick_labels=TIER_ORDER, showfliers=False, patch_artist=True, widths=0.5)
        for patch in box["boxes"]:
            patch.set(facecolor=LIGHT_COLORS[template], alpha=0.18, edgecolor=LIGHT_COLORS[template], linewidth=LINE_WIDTH, hatch=HATCHES[template])
        for key in ("whiskers", "caps", "medians"):
            for mark in box[key]:
                mark.set(color=LIGHT_COLORS[template], linewidth=LINE_WIDTH)
        ax.plot([], [], color=LIGHT_COLORS[template], linewidth=LINE_WIDTH, marker=MARKERS[template], markersize=MARKER_SIZE, label=template.title())
        ax.axhline(0, color="#898781", linewidth=1)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False)
        _style(ax, f"{template.title()} context")
        _panel_label(ax, f"({chr(97 + TEMPLATE_ORDER.index(template))})")
        _zero_label(ax, vertical=False)
        ymax = ax.get_ylim()[1]
        for position, (tier, group) in enumerate(zip(TIER_ORDER, groups, strict=True), start=1):
            ax.text(position, ymax, f"n={len(group):,}", ha="center", va="bottom", fontsize=7, color="#52514e")
    _paper_header(fig, "Entity effects vary by familiarity tier and prompt context", "Boxes show entity-level ΔE distributions; labels report the number of entities in each tier.", "The train/eval field is a deterministic analysis split, not an independently repeated sample.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "entity_effect_by_tier")


def plot_template_relationships(run: Any, output_dir: Path) -> dict[str, Path]:
    rows = summarize_ticker(run)
    pairs = (("negative", "positive"), ("negative", "neutral"), ("positive", "neutral"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    for ax, (left, right) in zip(axes, pairs, strict=True):
        x = [row[f"{left}_delta_expected_score"] for row in rows]
        y = [row[f"{right}_delta_expected_score"] for row in rows]
        ax.scatter(x, y, s=MARKER_SIZE ** 2, alpha=0.3, color=LIGHT_COLORS[left], marker=MARKERS[left], edgecolors="#fcfcfb", linewidths=2, label=f"{left.title()} vs {right.title()}")
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.axvline(0, color="#c3c2b7", linewidth=1)
        ax.set_xlabel(left.title())
        ax.set_ylabel(right.title())
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
        ax.set_title(f"{left.title()} vs {right.title()}\nn={len(rows):,}; Pearson r={correlation:.2f}", loc="left", fontweight="bold", fontsize=10)
        _panel_label(ax, f"({chr(97 + pairs.index((left, right)))})")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    _paper_header(fig, "Entity effects are related across prompt contexts", f"Each point is one ticker; n = {len(rows):,} tickers in every panel.", "Axes show ΔE for the named contexts. Correlations are descriptive and do not establish causality.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "template_relationships")


def plot_localization_profiles(run: Any, output_dir: Path) -> dict[str, Path]:
    metrics = (("mean_cosine", "Mean cosine"), ("pearson_r", "Pearson r"), ("spearman_r", "Spearman rho"), ("linear_r2", "Linear R²"))
    transitions = {(row["template"], row["metric"]): row for row in localization_transition_diagnostics(run)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, facecolor="#fcfcfb")
    max_layer = max(int(row["layer"]) for row in run.localization)
    first = run.localization[0]
    for panel_index, (ax, (metric, label)) in enumerate(zip(axes.flat, metrics, strict=True)):
        for template in TEMPLATE_ORDER:
            rows = sorted((row for row in run.localization if row["template"] == template), key=lambda row: int(row["layer"]))
            x = [int(row["layer"]) / max_layer if max_layer else 0 for row in rows]
            y = [float(row[metric]) for row in rows]
            ax.plot(x, y, **_line_kwargs(template, markers=True))
            diagnostic = transitions[(template, metric)]
            peak_layer = diagnostic["absolute_peak_layer"]
            peak_index = next(index for index, row in enumerate(rows) if int(row["layer"]) == peak_layer)
            ax.scatter([x[peak_index]], [y[peak_index]], s=(MARKER_SIZE + 2) ** 2, facecolor="#fcfcfb", edgecolor=LIGHT_COLORS[template], marker=MARKERS[template], linewidth=2, zorder=4)
            if metric == "mean_cosine" and diagnostic["maximum_jump_to_layer"] is not None:
                jump_depth = diagnostic["maximum_jump_to_layer"] / max_layer if max_layer else 0
                ax.axvline(jump_depth, color=LIGHT_COLORS[template], linewidth=0.8, linestyle=DASHES[template], alpha=0.45)
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        _style(ax, label, label)
        ax.set_xlabel("Normalized layer depth")
        ax.legend(frameon=False, ncols=3, fontsize=8)
        _panel_label(ax, f"({chr(97 + panel_index)})")
        ax.axvline(1.0, color="#898781", linewidth=1)
        if metric == "mean_cosine":
            counts = ", ".join(f"{template.title()} {transitions[(template, metric)]['sign_change_count']}" for template in TEMPLATE_ORDER)
            ax.text(0.02, 0.03, f"Sign changes: {counts}", transform=ax.transAxes, fontsize=7, color="#52514e")
    _paper_header(fig, "Entity-sensitive localization changes across model depth", f"Depth is layer/final layer; train n = {int(first['n_train']):,}, eval n = {int(first['n_eval']):,} entities per context.", "Localization is Jacobian-transported representation evidence, not chain-of-thought or a standalone causal proof.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "localization_profiles")


def plot_sector_effects(run: Any, output_dir: Path, *, minimum_count: int = 20) -> tuple[dict[str, Path], int]:
    summary = summarize_sector(run)
    rows = [row for row in summary if row["count"] >= minimum_count]
    excluded = len(summary) - len(rows)
    sectors = sorted({row["sector"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, max(4, len(sectors) * 0.5)), facecolor="#fcfcfb")
    width = 0.22
    positions = np.arange(len(sectors))
    for index, template in enumerate(TEMPLATE_ORDER):
        values = {row["sector"]: row["mean_delta_expected_score"] for row in rows if row["template"] == template}
        ax.barh(positions + (index - 1) * (width + 0.04), [values.get(sector, 0) for sector in sectors], height=width, color=LIGHT_COLORS[template], hatch=HATCHES[template], edgecolor="#fcfcfb", linewidth=2, label=template.title())
    ax.axvline(0, color="#898781", linewidth=1)
    ax.set_yticks(positions, sectors)
    ax.set_xlabel("Mean delta expected score")
    ax.set_title("Included sector–context groups", loc="left", fontweight="bold")
    _panel_label(ax, "(a)")
    _zero_label(ax, vertical=True)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _paper_header(fig, "Mean entity effects differ across reported sectors", f"Only sector–context groups with n ≥ {minimum_count} are shown; {excluded} groups are excluded.", "Pipe-delimited sector memberships are exploded; missing sectors are Unknown. Source years are provenance, not independently verified historical membership evidence.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return _save(fig, output_dir, "sector_effects"), excluded


def make_plots(run: Any, output_dir: str | Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sector_paths, excluded = plot_sector_effects(run, directory)
    return {
        "entity_effect_distribution": plot_effect_distribution(run, directory),
        "entity_effect_tail_diagnostics": plot_tail_diagnostics(run, directory),
        "baseline_entity_movement": plot_baseline_movement(run, directory),
        "temperature_null_diagnostics": plot_temperature_null(run, directory),
        "entity_effect_by_tier": plot_tier(run, directory),
        "template_relationships": plot_template_relationships(run, directory),
        "localization_profiles": plot_localization_profiles(run, directory),
        "sector_effects": sector_paths,
    }, {"sector_minimum_count": 20, "excluded_sector_template_groups": excluded}
