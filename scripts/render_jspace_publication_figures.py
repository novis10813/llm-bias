#!/usr/bin/env python
"""Render readable publication figures from normalized J-space token artifacts."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SECTORS = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def echo_colour(row: dict) -> str:
    lift = row.get("echo_lift")
    if lift is None:
        return "#7b3294"  # absent from prompt
    if lift > 1.5:
        return "#d95f02"  # amplified
    if lift < 0.67:
        return "#1b9e77"  # suppressed
    return "#777777"      # near prompt share


def render_sector_cards(run_dir: Path, output_dir: Path, top_n: int = 5) -> None:
    tfidf_rows = load_jsonl(run_dir / "sector_keywords.jsonl")
    lod_rows = load_jsonl(run_dir / "sector_logodds.jsonl")
    by_tfidf: dict[str, list[dict]] = defaultdict(list)
    by_lod: dict[str, list[dict]] = defaultdict(list)
    echo_lookup = {}
    for row in tfidf_rows:
        by_tfidf[row["document"]].append(row)
        echo_lookup[(row["document"], row["token"])] = row
    for row in lod_rows:
        if row["logodds_z"] > 0:
            by_lod[row["document"]].append(row)
    cards = output_dir / "sector_cards"
    cards.mkdir(parents=True, exist_ok=True)

    for sector in SECTORS:
        left = sorted(by_tfidf[sector], key=lambda r: r["tfidf"], reverse=True)[:top_n]
        right = sorted(by_lod[sector], key=lambda r: r["logodds_z"], reverse=True)[:top_n]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        fig.suptitle(sector, fontsize=16, fontweight="bold")

        left_max = max((r["tfidf"] for r in left), default=1.0)
        left_values = [100 * r["tfidf"] / left_max for r in left][::-1]
        left_labels = [r["token"] for r in left][::-1]
        left_colours = [echo_colour(r) for r in left][::-1]
        axes[0].barh(left_labels, left_values, color=left_colours)
        axes[0].set_title("Distinctive concepts (TF-IDF, max = 100)")
        axes[0].set_xlim(0, 108)
        axes[0].set_xlabel("within-sector relative TF-IDF")
        for i, row in enumerate(left[::-1]):
            lift = row.get("echo_lift")
            echo = "prompt absent" if lift is None else f"echo {lift:.2f}x"
            axes[0].text(left_values[i] + 1, i, echo, va="center", fontsize=8)

        right_values = [r["logodds_z"] for r in right][::-1]
        right_labels = [r["token"] for r in right][::-1]
        axes[1].barh(right_labels, right_values, color="#4472c4")
        axes[1].axvline(2, color="#999999", linestyle="--", linewidth=1)
        axes[1].set_title("Reliable overuse (descriptive log-odds z)")
        axes[1].set_xlabel("z vs all other sectors")
        for i, value in enumerate(right_values):
            axes[1].text(value + 0.05, i, f"z={value:.1f}", va="center", fontsize=8)

        legend = [
            plt.Line2D([], [], color="#d95f02", linewidth=6, label="amplified >1.5x"),
            plt.Line2D([], [], color="#777777", linewidth=6, label="near prompt share"),
            plt.Line2D([], [], color="#1b9e77", linewidth=6, label="suppressed <0.67x"),
            plt.Line2D([], [], color="#7b3294", linewidth=6, label="absent from prompt"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 0.93))
        fig.savefig(cards / f"{slug(sector)}.png", dpi=160)
        plt.close(fig)


def render_sector_summary(run_dir: Path, output: Path, top_n: int = 4) -> None:
    tfidf_rows = load_jsonl(run_dir / "sector_keywords.jsonl")
    lod_rows = load_jsonl(run_dir / "sector_logodds.jsonl")
    by_tfidf: dict[str, list[dict]] = defaultdict(list)
    by_lod: dict[str, list[dict]] = defaultdict(list)
    for row in tfidf_rows:
        by_tfidf[row["document"]].append(row)
    for row in lod_rows:
        if row["logodds_z"] > 0:
            by_lod[row["document"]].append(row)

    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(SECTORS) + 1.5)
    ax.axis("off")
    ax.text(0.01, len(SECTORS) + 0.9, "Sector", fontweight="bold", fontsize=12)
    ax.text(0.22, len(SECTORS) + 0.9, "Distinctive concepts (TF-IDF)", fontweight="bold", fontsize=12)
    ax.text(0.62, len(SECTORS) + 0.9, "Reliable overuse (log-odds z)", fontweight="bold", fontsize=12)
    ax.plot([0, 1], [len(SECTORS) + 0.65] * 2, color="black", linewidth=1)
    for index, sector in enumerate(SECTORS):
        y = len(SECTORS) - index
        if index % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.45), 1, 0.9, color="#f4f4f4", zorder=-1))
        left = sorted(by_tfidf[sector], key=lambda r: r["tfidf"], reverse=True)[:top_n]
        right = sorted(by_lod[sector], key=lambda r: r["logodds_z"], reverse=True)[:top_n]
        ax.text(0.01, y, sector, va="center", fontweight="bold", fontsize=9)
        ax.text(0.22, y, ", ".join(r["token"] for r in left), va="center", fontsize=9)
        ax.text(
            0.62, y,
            ", ".join(f"{r['token']} ({r['logodds_z']:.1f})" for r in right),
            va="center", fontsize=9,
        )
    ax.text(
        0.01, 0.15,
        "TF-IDF highlights sector-specific content; log-odds z highlights reliably overused narrative language.",
        fontsize=9, color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_echo_evidence(run_dir: Path, output: Path) -> None:
    rows = [
        row
        for row in load_jsonl(run_dir / "sector_keywords.jsonl")
        if row.get("echo_lift") is not None and row.get("logodds_z") is not None
    ]
    xs = np.array([math.log2(row["echo_lift"]) for row in rows])
    ys = np.array([row["logodds_z"] for row in rows])
    tfidf = np.array([row["tfidf"] for row in rows])
    sizes = 15 + 80 * np.sqrt(tfidf / max(tfidf.max(), 1e-12))
    echo_threshold = math.log2(1.5)
    z_threshold = 2.0

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axvspan(echo_threshold, max(xs.max() + 0.3, 4), color="#fff2cc", alpha=0.45)
    ax.axhspan(z_threshold, max(ys.max() + 0.2, 2.5), color="#d9ead3", alpha=0.55)
    ax.scatter(xs, ys, s=sizes, color="#777777", alpha=0.55, edgecolors="none")
    reliable = (xs >= echo_threshold) & (ys >= z_threshold)
    if reliable.any():
        ax.scatter(xs[reliable], ys[reliable], s=sizes[reliable] + 35,
                   color="#d95f02", edgecolors="black", linewidths=0.5)
    ax.axvline(echo_threshold, color="#b45f06", linestyle="--", linewidth=1)
    ax.axhline(z_threshold, color="#38761d", linestyle="--", linewidth=1)

    label_indices = set(np.argsort(xs)[-6:].tolist()) | set(np.argsort(ys)[-6:].tolist())
    abbreviations = {sector: "".join(word[0] for word in sector.split()) for sector in SECTORS}
    for i in label_indices:
        ax.annotate(
            f"{rows[i]['token']} [{abbreviations[rows[i]['document']]}]",
            (xs[i], ys[i]), fontsize=8, xytext=(3, 3), textcoords="offset points",
        )
    ax.text(
        echo_threshold + 0.08, max(ys.max() + 0.05, 2.15),
        "reliably amplified", color="#274e13", fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("log2 echo-lift: readout share / prompt share")
    ax.set_ylabel("descriptive log-odds z vs other sectors")
    ax.set_title("Prompt-echo check for top sector-distinctive concepts")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_company_confusion(npz_path: Path, output: Path, k: int = 10) -> None:
    bundle = np.load(npz_path, allow_pickle=False)
    matrix = bundle["matrix"]
    sectors = bundle["sectors"]
    normed = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    similarities = normed @ normed.T
    np.fill_diagonal(similarities, -1)
    predicted = []
    for i in range(len(sectors)):
        neighbours = np.argsort(-similarities[i])[:k]
        predicted.append(Counter(sectors[neighbours]).most_common(1)[0][0])
    predicted = np.array(predicted)
    labels = [s for s in SECTORS if s in sectors]
    confusion = np.zeros((len(labels), len(labels)))
    for actual, guess in zip(sectors, predicted):
        confusion[labels.index(actual), labels.index(guess)] += 1
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = confusion / np.clip(row_totals, 1, None)
    accuracy = float(np.mean(predicted == sectors))
    majority = max(Counter(sectors).values()) / len(sectors)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = normalized[i, j]
            if value >= 0.05:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if value > 0.55 else "black")
    ax.set_xlabel("predicted sector from 10 nearest companies")
    ax.set_ylabel("actual sector")
    ax.set_title(
        f"Company readout sector structure: 10-NN accuracy {accuracy:.1%} "
        f"(majority baseline {majority:.1%})"
    )
    fig.colorbar(im, label="fraction of actual-sector companies")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    render_sector_cards(args.run_dir, output_dir)
    render_sector_summary(args.run_dir, output_dir / "sector_summary.png")
    render_echo_evidence(args.run_dir, output_dir / "echo_evidence.png")
    render_company_confusion(
        args.run_dir / "company_matrix.npz",
        output_dir / "company_sector_confusion.png",
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
