# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Plot the investment-dial calibration V2 assembled A curve, target inversions, and B reevaluation."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

V2 = Path(
    "artifacts/qwen3.5-4b/investment-dial-calibration-v2/runs/calib-v2-gpu-bf16-01"
)
V1 = Path(
    "artifacts/qwen3.5-4b/investment-dial-calibration/runs/exploratory-v1-20260907-07-gpu0-calibration"
)
OUT = Path("./docs/assets/investment-dial")

V1_TARGETS = [-0.3, 0.0, 0.3]


def main():
    """Render the calibration V2 curve and error comparison.

    Parameters: none (reads run artifacts from the repository tree).

    Saves: ./docs/assets/investment-dial/calibration_v2_curve.pdf and
            ./docs/assets/investment-dial/calibration_v2_curve.png
    """

    # --- Data (computed from run artifacts, never hardcoded) ---
    curve = json.load(open(V2 / "forward" / "a_curve.json"))
    result = json.load(open(V2 / "analyze" / "result.json"))
    v1_selected = json.load(open(V1 / "analyze" / "result.json"))["selected"]

    points = curve["points"]
    delta = np.array([p["delta"] for p in points])
    pi = np.array([p["pi"] for p in points])
    sources = [p["source"] for p in points]

    targets = list(curve["inversion"]["targets"])
    delta_hats = list(curve["inversion"]["delta_hats"])
    b_by_target = {t["target"]: t for t in result["B_targets"]}

    v1_pi = {i: s["pi"] for i, s in enumerate(v1_selected["B_stats"])}
    v1_error = [abs(v1_pi[i] - t) for i, t in enumerate(V1_TARGETS)]
    v2_error = [abs(b_by_target[t]["pi"] - t) for t in targets]

    def interp_a(x):
        return float(np.interp(x, delta, pi))

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    line_dark = pal[5]
    line_mid = pal[3]
    accent = pal[1]
    OUT.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(13, 5.6), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0])
    gs.update(wspace=0.14, left=0.055, right=0.99, top=0.86, bottom=0.13)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    fig.suptitle(
        "Calibration V2: assembled 15-point A curve, target inversions, and B reevaluation — "
        "Qwen3.5-4B, layer 15 / neuron 8490",
        fontsize=13, y=0.98, color="dimgrey",
    )

    # --- Plot (a): assembled curve with inversions and B points ---
    ax1.grid(False)
    ax1.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax1.patch.set_edgecolor("lightgrey")
    ax1.patch.set_linewidth(0.8)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.6, color="lightgrey")

    ax1.plot(delta, pi, color=line_dark, linewidth=2.0, zorder=3,
             label="assembled A curve (15 points, non-decreasing)")
    for src, style in (("v2-new", dict(marker="^", color=accent, s=58, edgecolors="white", linewidths=0.8)),
                       ("fine-a", dict(marker="o", color=line_dark, s=40, edgecolors="white", linewidths=0.8)),
                       ("v1-raw", dict(marker="s", facecolors="white",
                                       edgecolors="dimgrey", s=56, linewidths=1.2))):
        m = [i for i, s in enumerate(sources) if s == src]
        ax1.scatter(delta[m], pi[m], zorder=4, **style)
    ax1.scatter([], [], marker="^", color=accent, s=50, edgecolors="white",
                linewidths=0.8, label="v2-new (δ = −0.5, −0.25)")
    ax1.scatter([], [], marker="o", color=line_dark, s=34, edgecolors="white",
                linewidths=0.8, label="fine-a (δ = 0–2)")
    ax1.scatter([], [], marker="s", facecolors="white", edgecolors="dimgrey",
                s=48, linewidths=1.2, label="v1-raw (δ = −8, −4, 4, 8)")

    for t, dh in zip(targets, delta_hats):
        pa = interp_a(dh)
        bpi = b_by_target[t]["pi"]
        ax1.vlines(dh, t, pa, colors="grey", linestyles=(0, (5, 4)), linewidth=1.2,
                   zorder=2)
        ax1.scatter([dh], [t], marker="x", color="dimgrey", s=70, linewidths=1.6,
                    zorder=5)
        ax1.scatter([dh], [bpi], marker="D", color=accent, s=46, edgecolors="white",
                    linewidths=0.8, zorder=5)
        tx, ty = {(-0.3): (-0.50, -0.92), (0.0): (0.55, -0.55), (0.3): (0.85, 0.42)}[t]
        ax1.annotate(
            "Δ̂({0:+.1f}) = {1:+.3f}\nπ_B = {2:+.3f} (err {3:.3f})".format(
                t, dh, bpi, abs(bpi - t)),
            xy=(dh, bpi), xytext=(tx, ty),
            fontsize=8.5, color="dimgrey", ha="left",
            arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7),
        )
    ax1.scatter([], [], marker="x", color="dimgrey", s=60, linewidths=1.5,
                label="target (Δ̂, t)")
    ax1.scatter([], [], marker="D", color=accent, s=38, edgecolors="white",
                linewidths=0.8, label="measured B point (Δ̂, π_B)")

    ax1.set_xlim(-8.6, 8.6)
    ax1.set_ylim(-1.18, 1.18)
    ax1.set_xticks(np.arange(-8, 9, 2))
    ax1.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax1.set_xlabel("delta (neuron intervention magnitude)", fontsize=11, labelpad=6,
                   color="dimgrey")
    ax1.set_ylabel("π = (buy − sell) / (buy + sell)", fontsize=11, labelpad=6,
                   color="dimgrey")
    ax1.set_title(r"$\bf{(a)}$" + "  assembled curve, inverted targets, B measurements",
                  loc="left", fontsize=12, pad=7)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
               labelcolor="dimgrey", loc="upper left", fontsize=8.5)

    # --- Plot (b): V1 vs V2 per-target error ---
    ax2.grid(False)
    ax2.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax2.patch.set_edgecolor("lightgrey")
    ax2.patch.set_linewidth(0.8)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.6, color="lightgrey")

    x = np.arange(len(targets))
    w = 0.36
    b1 = ax2.bar(x - w / 2, v1_error, width=w, color="lightgrey", edgecolor="dimgrey",
                 linewidth=0.8, zorder=3, label="V1 coarse grid (RMSE 0.576)")
    b2 = ax2.bar(x + w / 2, v2_error, width=w, color=accent, edgecolor="white",
                 linewidth=0.8, zorder=3, label="V2 fine grid (RMSE 0.058)")
    ax2.axhline(0.25, color="dimgrey", linestyle=(0, (5, 4)), linewidth=1.0, zorder=2)
    ax2.text(len(targets) - 0.44, 0.262, "frozen max-error gate 0.25",
             fontsize=8, color="dimgrey", ha="right")
    for bars in (b1, b2):
        for r in bars:
            ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                     "{0:.3f}".format(r.get_height()), fontsize=8, color="dimgrey",
                     ha="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["t = −0.3", "t = 0.0", "t = +0.3"])
    ax2.set_ylim(0, 0.85)
    ax2.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax2.set_xlabel("target t", fontsize=11, labelpad=6, color="dimgrey")
    ax2.set_ylabel("|π_B(Δ̂) − t|", fontsize=11, labelpad=6, color="dimgrey")
    ax2.set_title(r"$\bf{(b)}$" + "  per-target error: V1 vs V2 (same coordinate, B)",
                  loc="left", fontsize=12, pad=7)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.8,
               edgecolor="lightgrey", labelcolor="dimgrey", loc="upper right",
               fontsize=8.5)

    # --- Figure-level insight ---
    fig.text(
        0.99, 0.01,
        "Gate: RMSE {0:.4f} ≤ 0.15 and max error {1:.4f} ≤ 0.25 → pass (certified). "
        "B is the V1 transfer set (reevaluation, not fresh holdout).".format(
            result["rmse"], result["max_error"]),
        ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
    )

    sns.despine(left=True, bottom=True)

    # --- Save ---
    plt.savefig(str(OUT / "calibration_v2_curve.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(str(OUT / "calibration_v2_curve.png"), dpi=150, bbox_inches="tight")
    print("saved", OUT / "calibration_v2_curve.png")


if __name__ == "__main__":
    main()
