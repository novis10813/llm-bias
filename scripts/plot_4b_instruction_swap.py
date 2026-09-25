# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Qwen3.5-4B instruction-span state swap, layer sweep by direction orientation.

Re-aggregates the existing formal 2B run (phase2b-gpu-bf16-01) — no new forwards.
Sweep records (sweep/records.jsonl) contain per-direction, per-layer, per-span
normalized transfer T. This figure keeps the INSTRUCTION span only and splits the
8 directions into two orientations, defined by the 2A pure entity margins:

  top-2  (least sell-leaning margins) -> "buy-leaning" side
  bottom-2 (most sell-leaning margins) -> "sell-leaning" side

  (a) buy  -> sell : patch source in top-2,   target in bottom-2
  (b) sell -> buy  : patch source in bottom-2, target in top-2

Final figure style: per-layer MEAN line across the 4 directions of the panel
with a SHADED STANDARD-DEVIATION band (sample std, ddof=1); individual
directions remain visible as faint background lines.

For Qwen3.5-4B all 16 pure entity margins are negative, so "buy-leaning" means
least sell-leaning; the figure title states this.

Usage:
    uv run --no-sync python scripts/plot_4b_instruction_swap.py [--out DIR]

Saves: <out>/4b_instruction_swap.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RUN_DIR = Path("artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./docs/concept-cone-steering/c2-phase2b-16/figures")
    args = parser.parse_args()

    # --- Data ---
    pairs = json.loads((RUN_DIR / "pairs" / "directions.json").read_text(encoding="utf-8"))
    n_layers = len(pairs["layers"])
    margins = pairs["pure_entity_margins"]
    recs = [json.loads(line) for line in (RUN_DIR / "sweep" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    ins = {}  # direction -> {layer: T}
    for r in recs:
        if r["span"] != "instruction":
            continue
        ins.setdefault(r["direction"], {})[int(r["layer"])] = r["normalized_transfer"]

    ranked = sorted(margins, key=margins.get)
    bottom2, top2 = set(ranked[:2]), set(ranked[-2:])
    buy_to_sell = [d for d in ins if d.split("->")[0] in top2 and d.split("->")[1] in bottom2]
    sell_to_buy = [d for d in ins if d.split("->")[0] in bottom2 and d.split("->")[1] in top2]
    assert len(buy_to_sell) == 4 and len(sell_to_buy) == 4, (buy_to_sell, sell_to_buy)
    print(f"n_layers={n_layers}  top2={sorted(top2)}  bottom2={sorted(bottom2)}")
    print(f"buy->sell: {sorted(buy_to_sell)}")
    print(f"sell->buy: {sorted(sell_to_buy)}")

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    dir_colors = [pal[1], pal[2], pal[3], pal[4]]
    mean_color = pal[5]
    band_color = pal[5]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), dpi=150)
    fig.subplots_adjust(top=0.60, bottom=0.16, wspace=0.22)
    panels = [
        (axes[0], "buy  →  sell", buy_to_sell),
        (axes[1], "sell  →  buy", sell_to_buy),
    ]

    for panel_i, (ax, title, directions) in enumerate(panels, start=1):
        all_t = np.full((len(directions), n_layers), np.nan)
        for di, d in enumerate(sorted(directions)):
            layers = sorted(ins[d])
            t = np.array([ins[d][l] for l in layers])
            all_t[di, :len(t)] = t
            # faint per-direction background lines
            ax.plot(np.arange(len(layers)) / (n_layers - 1), t,
                    lw=0.9, color=dir_colors[di], alpha=0.30,
                    label=f"{d.split('->')[0]} → {d.split('->')[1]}")
        mean_t = np.nanmean(all_t, axis=0)
        std_t = np.nanstd(all_t, axis=0, ddof=1)
        depths = np.arange(n_layers) / (n_layers - 1)
        # mean + shaded std band (the panel's primary display)
        ax.fill_between(depths, mean_t - std_t, mean_t + std_t,
                        color=band_color, alpha=0.22, lw=0, label="mean ± 1 SD")
        ax.plot(depths, mean_t, lw=2.6, color=mean_color, label="mean of 4 directions")

        # insight annotation: peak of the mean curve (or absence of transfer)
        pk = int(np.nanargmax(mean_t))
        if mean_t[pk] < 0.05:
            note = r"no net transfer: $T \leq$" + f" {mean_t[pk]:+.2f} at all layers"
        else:
            note = (f"peak T = {mean_t[pk]:.2f} ± {std_t[pk]:.2f} "
                    f"@ depth {depths[pk]:.2f} (layer {pk})")
        ax.text(
            0.97, 0.04, note, transform=ax.transAxes,
            fontsize=9, color="dimgrey", verticalalignment="bottom", horizontalalignment="right",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "lightgrey", "pad": 4},
        )

        ax.axhline(0, color="0.4", lw=0.8, ls=":")
        ax.set_xlabel("relative depth  $\\ell$ / (n_layers $-$ 1)", fontsize=12)
        ax.set_title(r"$\bf{(" + chr(96 + panel_i) + ")}$" + f" {title}", loc="left", fontsize=12, pad=55)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(-0.1, 0.9)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        sns.despine(left=True, bottom=True)
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8,
            frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
            labelcolor="dimgrey", columnspacing=1.2, handletextpad=0.6,
        )

    axes[0].set_ylabel("normalized transfer T (instruction span)", fontsize=12)
    fig.suptitle(
        "Qwen3.5-4B — instruction-span state swap, layer sweep by direction orientation "
        "(frozen shared-evidence template, 16 companies; top-2 "
        f"({', '.join(sorted(top2))}) = least sell-leaning, bottom-2 "
        f"({', '.join(sorted(bottom2))}) = most sell-leaning — all 16 pure entity margins negative)",
        fontsize=11, y=0.99,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "4b_instruction_swap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "4b_instruction_swap.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/4b_instruction_swap.{{pdf,png}}")


if __name__ == "__main__":
    main()
