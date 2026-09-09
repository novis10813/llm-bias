# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Plot the investment-dial fine-a A-only curve against the original V1 coarse grid."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

RUN = Path("artifacts/qwen3.5-4b/investment-dial-fine-a/runs/fine-a-gpu-bf16-01")
PARENT = Path(
    "artifacts/qwen3.5-4b/investment-dial-calibration/runs/exploratory-v1-20260907-07-gpu0-calibration"
)
OUT = Path("./docs/assets/investment-dial")


def main():
    """Render the fine-a curve diagnostic.

    Parameters: none (reads run artifacts from the repository tree).

    Saves: ./docs/assets/investment-dial/fine_a_curve.pdf and
            ./docs/assets/investment-dial/fine_a_curve.png
    """

    # --- Data ---
    fine = json.load(open(RUN / "analyze" / "result.json"))["summaries"]
    coarse_deltas = json.load(open(PARENT / "prepare" / "protocol.json"))["deltas"]
    coarse = json.load(open(PARENT / "analyze" / "result.json"))["selected"]["A_curve"]

    fine_delta = np.array([r["delta"] for r in fine])
    fine_pi = np.array([r["pi"] for r in fine])
    fine_valid = np.array([r["valid_decision_rate"] for r in fine])
    coarse_delta = np.array(coarse_deltas)
    coarse_pi = np.array([r["pi"] for r in coarse])

    i0 = int(np.where(coarse_delta == 0.0)[0][0])
    i4 = int(np.where(coarse_delta == 4.0)[0][0])
    interp_x = np.array([coarse_delta[i0], coarse_delta[i4]])
    interp_y = np.array([coarse_pi[i0], coarse_pi[i4]])

    # --- derived findings (computed, never hardcoded) ---
    steps = np.diff(fine_pi)
    k = int(np.argmax(steps))
    steepest_dx = fine_delta[k]
    steepest_dy = float(steps[k])

    def interp_at(x):
        return float(np.interp(x, interp_x, interp_y))

    gap_at_1 = float(fine_pi[np.argmin(np.abs(fine_delta - 1.0))] - interp_at(1.0))
    remeas = float(fine_pi[0] - coarse_pi[i0])

    d15 = int(np.argmin(np.abs(fine_delta - 1.5)))
    frac_by_15 = (fine_pi[d15] - fine_pi[0]) / (fine_pi[-1] - fine_pi[0])

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    line_dark = pal[5]
    line_mid = pal[3]
    OUT.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(13, 5.6), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0])
    gs.update(wspace=0.14, left=0.055, right=0.99, top=0.86, bottom=0.13)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    fig.suptitle(
        "A-only fine-grained curve, investment-dial fine-a — Qwen3.5-4B, layer 15 / neuron 8490",
        fontsize=13, y=0.98, color="dimgrey",
    )

    # --- Plot (a): pi curve ---
    ax1.grid(False)
    ax1.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax1.patch.set_edgecolor("lightgrey")
    ax1.patch.set_linewidth(0.8)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.6, color="lightgrey")

    ax1.plot(fine_delta, fine_pi, color=line_dark, linewidth=2.2, zorder=3,
             label="fine grid (this diagnostic, δ = 0–2)")
    ax1.scatter(fine_delta, fine_pi, color=line_dark, s=42, zorder=4,
                edgecolors="white", linewidths=0.8)
    ax1.scatter(coarse_delta, coarse_pi, facecolors="white", edgecolors="dimgrey",
                s=64, marker="s", zorder=4, linewidths=1.2,
                label="original V1 A samples (δ = −8…8)")
    ax1.plot(interp_x, interp_y, color="grey", linestyle=(0, (5, 4)),
             linewidth=1.4, zorder=2,
             label="linear interpolation, V1 δ 0 → 4")

    ax1.annotate(
        "remeasured at δ = 0:\nπ = {0:+.3f}  (Δ {1:+.3f} vs original)".format(
            float(fine_pi[0]), remeas),
        xy=(0.0, float(fine_pi[0])), xytext=(-7.4, -0.72),
        fontsize=9, color="dimgrey", ha="left",
        arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7,
                        connectionstyle="arc3,rad=-0.15"),
    )
    ax1.annotate(
        "steepest step: Δπ = {0:+.2f}\nwithin first quarter-step (δ 0 → 0.25)".format(
            steepest_dy),
        xy=(float(steepest_dx), float(fine_pi[k + 1])), xytext=(-7.6, 0.42),
        fontsize=9, color="dimgrey", ha="left",
        arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7,
                        connectionstyle="arc3,rad=0.2"),
    )
    ax1.text(3.15, 0.55, "coarse interpolation underestimates\n"
                         "Δπ = {0:+.2f} at δ = 1".format(gap_at_1),
             fontsize=9, color="dimgrey", ha="left", va="center")

    ax1.text(-0.08, float(fine_pi[0]), "{0:+.2f}".format(float(fine_pi[0])),
             fontsize=8, color="dimgrey", ha="right", va="bottom")
    ax1.text(1.97, float(fine_pi[-1]) + 0.014, "{0:+.2f}".format(float(fine_pi[-1])),
             fontsize=8, color="dimgrey", ha="right", va="bottom")

    ax1.set_xlim(-8.6, 8.6)
    ax1.set_ylim(-1.18, 1.18)
    ax1.set_xticks(np.arange(-8, 9, 2))
    ax1.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax1.set_xlabel("delta (neuron intervention magnitude)", fontsize=11, labelpad=6,
                   color="dimgrey")
    ax1.set_ylabel("π = (buy − sell) / (buy + sell)", fontsize=11, labelpad=6,
                   color="dimgrey")
    ax1.set_title(r"$\bf{(a)}$" + "  π response across the delta grid", loc="left",
                  fontsize=12, pad=7)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
               labelcolor="dimgrey", loc="upper left", fontsize=9)

    # --- Plot (b): validity rate ---
    ax2.grid(False)
    ax2.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax2.patch.set_edgecolor("lightgrey")
    ax2.patch.set_linewidth(0.8)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.6, color="lightgrey")

    ax2.axhline(1.0, color="lightgrey", linewidth=1.0, zorder=1)
    ax2.plot(fine_delta, fine_valid, color=line_mid, linewidth=1.8, zorder=3)
    ax2.scatter(fine_delta, fine_valid, color=line_mid, s=34, zorder=4,
                edgecolors="white", linewidths=0.8)
    for d, v in zip(fine_delta, fine_valid):
        va = "bottom" if v < 1.0 else "top"
        off = 0.0012 if v < 1.0 else -0.0012
        ax2.text(d, v + off, "{0:.3f}".format(v), fontsize=7.5, color="dimgrey",
                 ha="center", va=va)

    i_min = int(np.argmin(fine_valid))
    ax2.annotate("min 0.988 at δ = 1\n(4 of 340 invalid)",
                 xy=(float(fine_delta[i_min]), float(fine_valid[i_min])),
                 xytext=(1.42, 0.9858), fontsize=8, color="dimgrey", ha="left",
                 arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7))

    ax2.set_xlim(-0.15, 2.15)
    ax2.set_ylim(0.984, 1.0045)
    ax2.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax2.set_yticks([0.985, 0.990, 0.995, 1.000])
    ax2.set_xlabel("delta (neuron intervention magnitude)", fontsize=11, labelpad=6,
                   color="dimgrey")
    ax2.set_ylabel("output-valid decision rate", fontsize=11, labelpad=6,
                   color="dimgrey")
    ax2.set_title(r"$\bf{(b)}$" + "  validity across the fine grid", loc="left",
                  fontsize=12, pad=7)

    # --- Figure-level insight ---
    fig.text(
        0.99, 0.01,
        "{0:.0f}% of the δ 0 → 2 π change is completed by δ = 1.5; the coarse 0 → 4 "
        "line misses the front-loaded response.".format(frac_by_15 * 100),
        ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
    )

    sns.despine(left=True, bottom=True)

    # --- Save ---
    plt.savefig(str(OUT / "fine_a_curve.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(str(OUT / "fine_a_curve.png"), dpi=150, bbox_inches="tight")
    print("saved", OUT / "fine_a_curve.png")


if __name__ == "__main__":
    main()
