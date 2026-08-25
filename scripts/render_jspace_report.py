#!/usr/bin/env python
"""Render jspace layer-band report (curves + CKA heatmap) from layer_stats.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.stats.read_text())
    stats = data["stats"]
    layers = sorted(int(k) for k in stats)

    def series(key: str, subkey: str | None = None) -> list[float]:
        values = []
        for layer in layers:
            value = stats[str(layer)].get(key)
            if value is not None and subkey is not None:
                value = value.get(subkey)
            values.append(float("nan") if value is None else float(value))
        return values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    depth = [l / max(layers) * 100 for l in layers]

    ax = axes[0][0]
    ax.plot(depth, series("next_token_accuracy_topk"), marker="o")
    ax.set_ylabel(f"top-{data['params']['top_k']} next-token acc")
    ax.set_title("(a) next-token accuracy")

    ax = axes[0][1]
    ax.plot(depth, series("mean_excess_kurtosis"), marker="o", color="tab:orange")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("(b) readout logit sharpness")

    ax = axes[1][0]
    for lag in data["params"]["lags"]:
        ax.plot(
            depth,
            [
                stats[str(l)]["top1_lag_agreement"][str(lag)]["excess_over_null"]
                for l in layers
            ],
            marker="o",
            label=f"lag {lag}",
        )
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("top-1 agreement − null")
    ax.set_xlabel("normalised depth (%)")
    ax.legend()
    ax.set_title("(c) cross-position autocorrelation")

    ax = axes[1][1]
    ax.plot(
        depth,
        series("jvector_effective_dim_pr"),
        marker="o",
        color="tab:green",
        label="participation ratio",
    )
    ax2 = ax.twinx()
    ax2.plot(
        depth,
        series("jvector_dims_for_90pct_energy"),
        marker="s",
        color="tab:red",
        alpha=0.6,
        label="dims for 90% energy",
    )
    ax.set_ylabel("effective dim (PR)")
    ax2.set_ylabel("dims @90% energy")
    ax.set_xlabel("normalised depth (%)")
    ax.set_title("(d) J-lens vector dimensionality")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Jacobian-lens per-layer statistics — {data['model']} "
        f"({data['params']['num_prompts']} wikitext docs)"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "layer_curves.png", dpi=150)

    cka_path = args.stats.parent / "cka.npz"
    if cka_path.is_file():
        bundle = np.load(cka_path)
        cka, cka_layers = bundle["cka"], bundle["layers"]
        fig2, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cka, vmin=0.0, vmax=1.0, cmap="magma")
        ticks = np.arange(len(cka_layers))
        step = max(len(ticks) // 16, 1)
        ax.set_xticks(ticks[::step])
        ax.set_xticklabels(cka_layers[::step])
        ax.set_yticks(ticks[::step])
        ax.set_yticklabels(cka_layers[::step])
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        ax.set_title("linear CKA of J-lens vector column spaces")
        fig2.colorbar(im, label="CKA")
        fig2.tight_layout()
        fig2.savefig(args.output_dir / "cka_matrix.png", dpi=150)

    print(f"wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
