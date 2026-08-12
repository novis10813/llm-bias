"""Static PNG/SVG plots for synthetic entity-bias summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contract import TEMPLATE_ORDER, TIER_ORDER
from .summaries import summarize_sector, summarize_ticker

_COLORS = {"negative": "#b85c5c", "positive": "#35608f", "neutral": "#5b8f72"}


def _save(fig: Any, output_dir: Path, name: str) -> dict[str, Path]:
    paths = {suffix: output_dir / f"{name}.{suffix}" for suffix in ("png", "svg")}
    for suffix, path in paths.items():
        fig.savefig(path, dpi=170 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return paths


def _style(ax, title, ylabel="Delta expected score"):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def plot_effect_distribution(run: Any, output_dir: Path) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for template in TEMPLATE_ORDER:
        values = np.array([float(row["delta_expected_score"]) for row in run.results if row["template"] == template])
        axes[0].hist(values, bins=30, histtype="step", linewidth=1.8, color=_COLORS[template], label=template)
        ordered = np.sort(values)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=_COLORS[template], label=template)
    for ax in axes:
        ax.axvline(0, color="#555", linewidth=1)
        ax.legend(frameon=False)
        ax.set_xlabel("Entity minus baseline expected score")
    _style(axes[0], "Distribution by template", "Count")
    _style(axes[1], "Empirical cumulative distribution", "Cumulative fraction")
    return _save(fig, output_dir, "entity_effect_distribution")


def plot_tier(run: Any, output_dir: Path) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, template in zip(axes, TEMPLATE_ORDER):
        groups = [[float(row["delta_expected_score"]) for row in run.results if row["template"] == template and row["familiarity_tier"] == tier] for tier in TIER_ORDER]
        ax.boxplot(groups, tick_labels=TIER_ORDER, showfliers=False)
        ax.axhline(0, color="#555", linewidth=1)
        ax.tick_params(axis="x", rotation=25)
        _style(ax, f"{template.title()} template")
    return _save(fig, output_dir, "entity_effect_by_tier")


def plot_template_relationships(run: Any, output_dir: Path) -> dict[str, Path]:
    rows = summarize_ticker(run)
    pairs = (("negative", "positive"), ("negative", "neutral"), ("positive", "neutral"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (left, right) in zip(axes, pairs):
        x = [row[f"{left}_delta_expected_score"] for row in rows]
        y = [row[f"{right}_delta_expected_score"] for row in rows]
        ax.scatter(x, y, s=8, alpha=0.35, color="#35608f")
        ax.axhline(0, color="#aaa", linewidth=0.8)
        ax.axvline(0, color="#aaa", linewidth=0.8)
        ax.set_xlabel(left)
        ax.set_ylabel(right)
        ax.set_title(f"{left} vs {right}", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, output_dir, "template_relationships")


def plot_localization_profiles(run: Any, output_dir: Path) -> dict[str, Path]:
    metrics = (("mean_cosine", "Mean cosine"), ("pearson_r", "Pearson r"), ("spearman_r", "Spearman rho"), ("linear_r2", "Linear R²"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    max_layer = max(int(row["layer"]) for row in run.localization)
    for ax, (metric, label) in zip(axes.flat, metrics):
        for template in TEMPLATE_ORDER:
            rows = sorted((row for row in run.localization if row["template"] == template), key=lambda row: int(row["layer"]))
            x = [int(row["layer"]) / max_layer if max_layer else 0 for row in rows]
            ax.plot(x, [float(row[metric]) for row in rows], color=_COLORS[template], label=template)
        ax.axhline(0, color="#aaa", linewidth=0.8)
        _style(ax, label, label)
        ax.set_xlabel("Normalized layer depth")
    axes[0, 0].legend(frameon=False)
    return _save(fig, output_dir, "localization_profiles")


def plot_sector_effects(run: Any, output_dir: Path, *, minimum_count: int = 20) -> tuple[dict[str, Path], int]:
    rows = [row for row in summarize_sector(run) if row["count"] >= minimum_count]
    excluded = len(summarize_sector(run)) - len(rows)
    sectors = sorted({row["sector"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, max(4, len(sectors) * 0.36)))
    width = 0.24
    positions = np.arange(len(sectors))
    for index, template in enumerate(TEMPLATE_ORDER):
        values = {row["sector"]: row["mean_delta_expected_score"] for row in rows if row["template"] == template}
        ax.barh(positions + (index - 1) * width, [values.get(sector, 0) for sector in sectors], height=width, color=_COLORS[template], label=template)
    ax.axvline(0, color="#555", linewidth=1)
    ax.set_yticks(positions, sectors)
    ax.set_xlabel("Mean delta expected score")
    ax.set_title(f"Sector effects (n ≥ {minimum_count})", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, output_dir, "sector_effects"), excluded


def make_plots(run: Any, output_dir: str | Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sector_paths, excluded = plot_sector_effects(run, directory)
    return {
        "entity_effect_distribution": plot_effect_distribution(run, directory),
        "entity_effect_by_tier": plot_tier(run, directory),
        "template_relationships": plot_template_relationships(run, directory),
        "localization_profiles": plot_localization_profiles(run, directory),
        "sector_effects": sector_paths,
    }, {"sector_minimum_count": 20, "excluded_sector_template_groups": excluded}
