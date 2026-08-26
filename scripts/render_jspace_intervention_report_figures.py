#!/usr/bin/env python3
"""Render tracked figures for the J-space intervention interim report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "artifacts/qwen3.5-4b/jspace-intervention-calibration/runs"
)
OUTPUT = ROOT / "docs/assets/jspace-sector-intervention"

COLORS = {
    "technology": "#2878B5",
    "financial": "#D95F02",
    "estimand": "#3A923A",
    "control": "#8A8A8A",
}


def read_rows(run_id: str) -> list[dict]:
    path = RUN_ROOT / run_id / "forward/intervention_results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def grouped_mean(rows: list[dict], key: str, value: str) -> dict[float, float]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        groups[float(row[key])].append(float(row[value]))
    return {dose: mean(values) for dose, values in sorted(groups.items())}


def render_dose_calibration() -> None:
    swap_tech = read_rows("jspace-v2-swap-logodds-tech-fin-20260826")
    swap_fin = read_rows("jspace-v2-swap-logodds-fin-tech-20260826")
    gain_tech = read_rows("jspace-v2-gain-fine-logodds-tech-20260826")
    gain_fin = read_rows("jspace-v2-gain-fine-logodds-financial-20260826")

    swap_t = grouped_mean(swap_tech, "swap_fraction", "delta_margin")
    swap_f = grouped_mean(swap_fin, "swap_fraction", "delta_margin")
    fractions = sorted(set(swap_t) & set(swap_f))
    estimand = [0.5 * (swap_f[x] - swap_t[x]) for x in fractions]

    gain_t = grouped_mean(gain_tech, "gain", "delta_margin")
    gain_f = grouped_mean(gain_fin, "gain", "delta_margin")
    gains = sorted(set(gain_t) & set(gain_f))

    def gain_relative_norm(rows: list[dict]) -> dict[float, float]:
        groups: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            groups[float(row["gain"])].append(
                float(row["delivered_dose"]["relative_perturbation_norm"]) * 100
            )
        return {dose: mean(values) for dose, values in groups.items()}

    gain_norm_t = gain_relative_norm(gain_tech)
    gain_norm_f = gain_relative_norm(gain_fin)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    axes[0].axhline(0, color="#BBBBBB", linewidth=1)
    axes[0].plot(fractions, [swap_t[x] for x in fractions], "o-", color=COLORS["technology"], label="Technology → Financial")
    axes[0].plot(fractions, [swap_f[x] for x in fractions], "o-", color=COLORS["financial"], label="Financial → Technology")
    axes[0].plot(fractions, estimand, "s--", color=COLORS["estimand"], label="Bidirectional E")
    axes[0].axvline(0.5, color="#222222", linestyle=":", linewidth=1)
    axes[0].set(title="Coordinate-swap dose response", xlabel="Swap fraction", ylabel="Mean ΔM")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(0, color="#BBBBBB", linewidth=1)
    axes[1].plot(gains, [gain_t[x] for x in gains], "o-", color=COLORS["technology"], label="Technology prototype")
    axes[1].plot(gains, [gain_f[x] for x in gains], "o-", color=COLORS["financial"], label="Financial prototype")
    axes[1].axvline(1.1, color="#222222", linestyle=":", linewidth=1)
    axes[1].set(title="Prototype-gain margin response", xlabel="Gain", ylabel="Mean ΔM")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(gains, [gain_norm_t[x] for x in gains], "o-", color=COLORS["technology"], label="Technology prototype")
    axes[2].plot(gains, [gain_norm_f[x] for x in gains], "o-", color=COLORS["financial"], label="Financial prototype")
    axes[2].axhline(5, color="#B22222", linestyle="--", linewidth=1, label="5% reference")
    axes[2].axvline(1.1, color="#222222", linestyle=":", linewidth=1)
    axes[2].set(title="Delivered perturbation under gain", xlabel="Gain", ylabel="Relative perturbation (%)", ylim=(-1, 15))
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("J-space intervention calibration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "dose_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def control_key(row: dict) -> tuple[str, str]:
    return row["direction_control"], row["position_control"]


def render_control_comparison() -> None:
    tech = [
        row for row in read_rows("jspace-v3-control-full-swap-tech-fin-20260826")
        if float(row["swap_fraction"]) == 0.5
    ]
    fin = [
        row for row in read_rows("jspace-v3-control-full-swap-fin-tech-20260826")
        if float(row["swap_fraction"]) == 0.5
    ]
    order = [
        ("prototype", "evidence"),
        ("prototype", "shuffled_evidence"),
        ("prototype", "final_position"),
        ("matched_random", "evidence"),
        ("matched_random", "shuffled_evidence"),
        ("matched_random", "final_position"),
    ]
    labels = [
        "Sector\nEvidence", "Sector\nShuffled", "Sector\nFinal",
        "Random\nEvidence", "Random\nShuffled", "Random\nFinal",
    ]

    def effects(rows: list[dict]) -> dict[tuple[str, str], float]:
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            groups[control_key(row)].append(float(row["delta_margin"]))
        return {key: mean(values) for key, values in groups.items()}

    tech_effect = effects(tech)
    fin_effect = effects(fin)
    x = np.arange(len(order))
    width = 0.25
    tech_values = [tech_effect[key] for key in order]
    fin_values = [fin_effect[key] for key in order]
    estimand = [0.5 * (f - t) for t, f in zip(tech_values, fin_values)]

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.axhline(0, color="#777777", linewidth=1)
    ax.bar(x - width, tech_values, width, color=COLORS["technology"], label="Technology → Financial ΔM")
    ax.bar(x, fin_values, width, color=COLORS["financial"], label="Financial → Technology ΔM")
    ax.bar(x + width, estimand, width, color=COLORS["estimand"], label="Bidirectional E")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean effect")
    ax.set_title("Primary and controls at matched delivered dose")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "control_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_dose_matching() -> None:
    rows = []
    for run_id, direction in (
        ("jspace-v3-control-full-swap-tech-fin-20260826", "Technology → Financial"),
        ("jspace-v3-control-full-swap-fin-tech-20260826", "Financial → Technology"),
    ):
        rows.extend(
            (row, direction)
            for row in read_rows(run_id)
            if float(row["swap_fraction"]) == 0.5
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for direction, color in (("Technology → Financial", COLORS["technology"]), ("Financial → Technology", COLORS["financial"])):
        selected = [row for row, label in rows if label == direction]
        target = np.array([row["paired_primary_perturbation_norm"] for row in selected])
        delivered = np.array([row["delivered_dose"]["perturbation_norm"] for row in selected])
        axes[0].scatter(target, delivered, s=18, alpha=0.55, color=color, label=direction)
        errors = np.array([row["dose_match_relative_error"] * 100 for row in selected])
        axes[1].hist(errors, bins=np.linspace(0, 3.5, 15), alpha=0.55, color=color, label=direction)

    limits = axes[0].get_xlim()
    lower = min(limits[0], axes[0].get_ylim()[0])
    upper = max(limits[1], axes[0].get_ylim()[1])
    axes[0].plot([lower, upper], [lower, upper], color="#222222", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Paired-primary target norm", ylabel="Delivered perturbation norm", title="Target versus delivered dose")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="Dose-matching error (%)", ylabel="Row count", title="Dose error after BF16 rounding")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "dose_matching.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    render_dose_calibration()
    render_control_comparison()
    render_dose_matching()
    print(OUTPUT)


if __name__ == "__main__":
    main()
