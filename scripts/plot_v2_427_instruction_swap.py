# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Qwen3.5-4B v2 condition-flip instruction swap, 427-company universe.

Reads the v2 2B run (phase2b-v2-427-01). Each of the 427 companies contributes
two directions — its pos condition (two positive shared evidence sentences)
swapped into its neg condition (two negative sentences) and vice versa. The
INSTRUCTION span only is plotted:

  (a) left  : all pos → neg directions (positive → negative condition)
  (b) right : all neg → pos directions (negative → positive condition)

Each panel shows the per-layer MEAN normalized transfer T across the panel's
directions with a SHADED STANDARD-DEVIATION band (sample std, ddof=1).

Usage:
    uv run --no-sync python scripts/plot_v2_427_instruction_swap.py [--out DIR]

Saves: <out>/v2_427_instruction_swap.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RUN_DIR = Path("artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./docs/concept-cone-steering/c2-v2-427/figures")
    args = parser.parse_args()

    pairs = json.loads((RUN_DIR / "pairs" / "directions.json").read_text(encoding="utf-8"))
    n_layers = len(pairs["layers"])
    recs = [json.loads(line) for line in (RUN_DIR / "sweep" / "records.jsonl").read_text(encoding="utf-8").splitlines()]

    pos_to_neg, neg_to_pos = {}, {}
    for r in recs:
        if r["span"] != "instruction":
            continue
        src, tgt = r["direction"].split("->")
        target = pos_to_neg if (src.endswith(":pos") and tgt.endswith(":neg")) else neg_to_pos
        target.setdefault(r["direction"], {})[int(r["layer"])] = r["normalized_transfer"]
    assert pos_to_neg, "no pos->neg instruction records"
    assert neg_to_pos, "no neg->pos instruction records"
    skipped = pairs.get("skipped_directions", [])
    print(f"n_layers={n_layers}  pos->neg={len(pos_to_neg)}  neg->pos={len(neg_to_pos)}  skipped={len(skipped)}")

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(4, rot=-0.25, light=0.7)
    mean_color = pal[3]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), dpi=150)
    fig.subplots_adjust(top=0.62, bottom=0.16, wspace=0.22)
    panels = [
        (axes[0], "positive → negative  (427 companies)", pos_to_neg),
        (axes[1], "negative → positive  (427 companies)", neg_to_pos),
    ]

    for panel_i, (ax, title, directions) in enumerate(panels, start=1):
        all_t = np.full((len(directions), n_layers), np.nan)
        for di, d in enumerate(sorted(directions)):
            layers = sorted(directions[d])
            all_t[di, :len(layers)] = [directions[d][l] for l in layers]
        mean_t = np.nanmean(all_t, axis=0)
        std_t = np.nanstd(all_t, axis=0, ddof=1)
        depths = np.arange(n_layers) / (n_layers - 1)
        ax.fill_between(depths, mean_t - std_t, mean_t + std_t,
                        color=mean_color, alpha=0.22, lw=0,
                        label=f"mean ± 1 SD over {len(directions)} directions")
        ax.plot(depths, mean_t, lw=2.6, color=mean_color, label="mean T")

        pk = int(np.nanargmax(mean_t))
        note = (f"peak T = {mean_t[pk]:.3f} ± {std_t[pk]:.3f} "
                f"@ depth {depths[pk]:.2f} (layer {pk})")
        ax.text(
            0.97, 0.04, note, transform=ax.transAxes,
            fontsize=9, color="dimgrey", verticalalignment="bottom", horizontalalignment="right",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "lightgrey", "pad": 4},
        )

        ax.axhline(0, color="0.4", lw=0.8, ls=":")
        lo = min(0.0, float(np.nanmin(mean_t - std_t)))
        hi = float(np.nanmax(mean_t + std_t))
        pad = 0.08 * max(1.0, hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.0, 1.02)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("relative depth  $\\ell$ / (n_layers $-$ 1)", fontsize=12)
        ax.set_title(r"$\bf{(" + chr(96 + panel_i) + ")}$" + f" {title}", loc="left", fontsize=12, pad=55)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        sns.despine(left=True, bottom=True)
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=9,
            frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
            labelcolor="dimgrey",
        )

    axes[0].set_ylabel("normalized transfer T (instruction span)", fontsize=12)
    fig.suptitle(
        "Qwen3.5-4B v2 — within-company condition-flip, instruction-span state swap "
        f"(427 companies; each company's 2-sentence positive vs negative shared-evidence condition; "
        f"{len(skipped)} companies skipped: |M_pos $-$ M_neg| < 0.1 nats)",
        fontsize=11, y=0.99,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "v2_427_instruction_swap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "v2_427_instruction_swap.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/v2_427_instruction_swap.{{pdf,png}}")


if __name__ == "__main__":
    main()
