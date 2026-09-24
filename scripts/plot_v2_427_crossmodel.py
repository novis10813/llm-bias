# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""v2 427-company condition-flip, cross-model comparison (instruction span).

One row per model; each row's panels:
  left  : all pos -> neg directions (positive -> negative condition)
  right : all neg -> pos directions (negative -> positive condition)

x = relative depth ell / (n_layers - 1); per-layer MEAN normalized transfer T
with a SHADED 1-SD band (sample std, ddof=1) over the row's directions.
Models whose phase2b-v2-427-01 run is missing are skipped (interim figure).

Usage:
    uv run --no-sync python scripts/plot_v2_427_crossmodel.py [--out DIR]

Saves: <out>/v2_427_crossmodel.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MODELS = [
    ("Qwen3.5-4B", "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01"),
    ("GPT-OSS-20B", "artifacts/gpt-oss-20b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01"),
    ("GLM-4-9B", "artifacts/glm4-9b-0414/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01"),
    ("Gemma-4-12B", "artifacts/gemma4-12b-it/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01"),
    ("Qwen3.6-27B", "artifacts/qwen3.6-27b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01"),
]


def load_model(run_dir: Path):
    pairs = json.loads((run_dir / "pairs" / "directions.json").read_text(encoding="utf-8"))
    n_layers = len(pairs["layers"])
    recs = [json.loads(line) for line in (run_dir / "sweep" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    pos_to_neg, neg_to_pos = {}, {}
    for r in recs:
        if r["span"] != "instruction":
            continue
        src, tgt = r["direction"].split("->")
        target = pos_to_neg if (src.endswith(":pos") and tgt.endswith(":neg")) else neg_to_pos
        target.setdefault(r["direction"], {})[int(r["layer"])] = r["normalized_transfer"]
    skipped = len(pairs.get("skipped_directions", []))
    return n_layers, pos_to_neg, neg_to_pos, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./docs/assets/concept-cone-steering")
    args = parser.parse_args()

    rows = []
    for name, run_dir in MODELS:
        run_dir = Path(run_dir)
        if not (run_dir / "sweep" / "records.jsonl").exists():
            print(f"skip {name}: no sweep records under {run_dir}")
            continue
        rows.append((name,) + load_model(run_dir))
    if not rows:
        raise SystemExit("no completed v2 2B runs found")

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(4, rot=-0.25, light=0.7)
    mean_color = pal[3]

    fig, axes = plt.subplots(len(rows), 2, figsize=(12.8, 3.35 * len(rows) + 1.4), dpi=150,
                             sharex=True)
    if len(rows) == 1:
        axes = axes[None, :]
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.10, right=0.985, hspace=0.52, wspace=0.14)

    for row_i, (name, n_layers, pos_to_neg, neg_to_pos, skipped) in enumerate(rows):
        for panel_i, (title, directions) in enumerate(
            [("positive → negative", pos_to_neg), ("negative → positive", neg_to_pos)]
        ):
            ax = axes[row_i, panel_i]
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
            ax.plot(depths, mean_t, lw=2.4, color=mean_color, label="mean T")

            pk = int(np.nanargmax(mean_t))
            note = (f"peak T = {mean_t[pk]:.3f} ± {std_t[pk]:.3f} "
                    f"@ depth {depths[pk]:.2f} (layer {pk})")
            ax.text(
                0.985, 0.04, note, transform=ax.transAxes,
                fontsize=8, color="dimgrey", verticalalignment="bottom", horizontalalignment="right",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "lightgrey", "pad": 3},
            )
            ax.axhline(0, color="0.4", lw=0.8, ls=":")
            lo = min(0.0, float(np.nanmin(mean_t - std_t)))
            hi = float(np.nanmax(mean_t + std_t))
            pad = 0.08 * max(1.0, hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(0.0, 1.02)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
            sns.despine(left=True, bottom=True)
            ax.patch.set_edgecolor("lightgrey")
            ax.patch.set_linewidth(0.8)
            if row_i == 0:
                ax.legend(
                    loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8,
                    frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
                    labelcolor="dimgrey",
                )
                ax.set_title(r"$\bf{(a)}$ positive → negative      $\bf{(b)}$ negative → positive",
                             loc="left", fontsize=11, pad=28)
            if panel_i == 0:
                ax.set_ylabel(f"{name}\nn_layers = {n_layers}", fontsize=10.5)
            if row_i == len(rows) - 1:
                ax.set_xlabel("relative depth  $\\ell$ / (n_layers $-$ 1)", fontsize=10.5)

    shown = ", ".join(r[0] for r in rows)
    fig.suptitle(
        "v2 — within-company condition-flip, instruction-span state swap "
        f"(427 companies; 2-sentence positive vs negative shared-evidence condition)\n"
        f"models: {shown}",
        fontsize=11.5, y=0.995,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "v2_427_crossmodel.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "v2_427_crossmodel.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/v2_427_crossmodel.{{pdf,png}}  (models: {shown})")


if __name__ == "__main__":
    main()
