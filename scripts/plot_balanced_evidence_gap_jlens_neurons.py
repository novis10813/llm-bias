# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Plot the J-lens neuron structural readout (balanced-evidence-gap diagnostic).

Reads the formal diagnostic run's analyze/summary.json and renders:
  (a) transported-direction buy/sell margin for the 4 primary coordinates
      against the per-layer matched-control distribution;
  (b) 4x4 pairwise cosine of the transported directions.

Usage:
    uv run --no-sync python scripts/plot_balanced_evidence_gap_jlens_neurons.py \
        [--run artifacts/qwen3.5-4b/balanced-evidence-gap-phase3/runs/phase3-jlens-neurons-01]

Saves: ./docs/assets/balanced-evidence-gap/jlens_neuron_structure.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

DEFAULT_RUN = Path(
    "artifacts/qwen3.5-4b/balanced-evidence-gap-phase3/runs/phase3-jlens-neurons-03"
)
OUT = Path("./docs/assets/balanced-evidence-gap")

PRIMARY = ["L19_n6334", "L20_n6520", "L26_n2394", "L15_n8490 (dial)"]
PRIMARY_KEYS = ["L19_n6334", "L20_n6520", "L26_n2394", "L15_n8490"]
COLORS = {"L19_n6334": "#d95f02", "L20_n6520": "#7570b3", "L26_n2394": "#1a9641",
          "L15_n8490": "#4575b4"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    summary = json.loads((args.run / "analyze" / "summary.json").read_text())

    sns.set_theme(style="whitegrid", font="DejaVu Sans", context="paper")
    readouts = summary["readouts"]
    cos = summary["cosine_primary"]

    fig = plt.figure(figsize=(12, 4.6), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    # ── (a) margin buy/sell vs controls ──
    x = np.arange(len(PRIMARY))
    vals = [readouts[k]["margin_buy_sell"] for k in PRIMARY_KEYS]
    bars = ax1.bar(x, vals, width=0.5,
                   color=[COLORS[k] for k in PRIMARY_KEYS], alpha=0.9)
    rng = np.random.default_rng(7)
    for k, key in enumerate(PRIMARY_KEYS):
        layer = readouts[key]["layer"]
        ctl = [v["margin_buy_sell"] for name, v in readouts.items()
               if v["role"] == "matched_control" and v["layer"] == layer]
        if not ctl:
            continue
        ax1.plot(x[k] + rng.uniform(-0.12, 0.12, len(ctl)), ctl,
                 "o", ms=3, color="0.55", alpha=0.7)
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(PRIMARY, fontsize=8)
    ax1.set_ylabel("transported-direction margin\nlogit(buy) - logit(sell)")
    ax1.set_title("(a) J-lens readout of the neuron injection direction",
                  fontsize=10)
    from matplotlib.lines import Line2D
    ax1.legend(handles=[Line2D([], [], marker="o", ms=3, color="0.55",
                               linestyle="none",
                               label="10 matched controls per layer")],
               fontsize=7.5, loc="lower left")
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # ── (b) cosine heatmap ──
    M = np.array([[cos[a][b] for b in PRIMARY_KEYS] for a in PRIMARY_KEYS])
    im = ax2.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels([p.replace(" (dial)", "") for p in PRIMARY], fontsize=8, rotation=30, ha="right")
    ax2.set_yticklabels(PRIMARY, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="white" if abs(M[i, j]) > 0.5 else "black")
    ax2.set_title("(b) cosine of transported directions", fontsize=10)
    fig.colorbar(im, ax=ax2, shrink=0.8)
    for spine in ("top", "right", "bottom", "left"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle("J-lens structural readout — Phase 3 entity channels vs dial "
                 "(descriptive, non-causal)", fontsize=10.5, y=1.03)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "jlens_neuron_structure.pdf", bbox_inches="tight")
    fig.savefig(OUT / "jlens_neuron_structure.png", bbox_inches="tight")
    print(f"saved {OUT}/jlens_neuron_structure.{{pdf,png}}")


if __name__ == "__main__":
    main()
