# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Plot why the V1 coarse grid failed and what the V2 fine grid recovered (investment-dial)."""
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
    """Render the calibration V2 explanation figure (3 panels).

    Parameters: none (reads run artifacts from the repository tree).

    Saves: ./docs/assets/investment-dial/calibration_v2_explained.pdf and
            ./docs/assets/investment-dial/calibration_v2_explained.png
    """

    # --- Data ---
    curve = json.load(open(V2 / "forward" / "a_curve.json"))
    result = json.load(open(V2 / "analyze" / "result.json"))
    v1 = json.load(open(V1 / "analyze" / "result.json"))["selected"]

    points = curve["points"]
    delta = np.array([p["delta"] for p in points])
    pi = np.array([p["pi"] for p in points])

    targets = list(curve["inversion"]["targets"])
    delta_hats = list(curve["inversion"]["delta_hats"])
    b_by_t = {t["target"]: t for t in result["B_targets"]}
    b_base_pi = result["B_baseline"]["pi"]

    # V1's five coarse A samples: v1-raw points recomputed under the frozen rule
    # (stored in the V2 curve) + V1's stored delta=0 point (verified consistent
    # with raw records).
    v1_raw = {p["delta"]: p["pi"] for p in points if p["source"] == "v1-raw"}
    v1_zero_pi = v1["A_curve"][2]["pi"]
    v1d = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
    v1p = np.array([v1_raw[-8.0], v1_raw[-4.0], v1_zero_pi, v1_raw[4.0], v1_raw[8.0]])

    v1_b_pi = [s["pi"] for s in v1["B_stats"]]
    v1_err = [abs(p - t) for p, t in zip(v1_b_pi, V1_TARGETS)]
    v2_err = [abs(b_by_t[t]["pi"] - t) for t in targets]

    # Inverse lookup on the non-decreasing fine curve (pi as x): the delta at which
    # the fine curve equals V1's measured neutral B pi -> inferred V1 delta_hat(0).
    v1_neutral_pi = v1_b_pi[1]
    v1_dh0_inferred = float(np.interp(v1_neutral_pi, pi, delta))
    frac_by_15 = (pi[np.argmin(np.abs(delta - 1.5))] - pi[0]) / (pi[-1] - pi[0])
    ab_offset = b_base_pi - pi[0]

    def interp_a(x):
        return float(np.interp(x, delta, pi))

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    line_dark = pal[5]
    accent = pal[1]
    OUT.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 5.6), dpi=150)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.7, 1.35, 1.0])
    gs.update(wspace=0.16, left=0.04, right=0.99, top=0.85, bottom=0.14)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    fig.suptitle(
        "Investment-dial V2 — why the coarse grid failed, and what the fine grid recovered "
        "(Qwen3.5-4B, layer 15 / neuron 8490)",
        fontsize=13, y=0.98, color="dimgrey",
    )

    def base_ax(ax, xlabel):
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6, color="lightgrey")
        ax.set_xlabel(xlabel, fontsize=10.5, labelpad=6, color="dimgrey")

    # --- Plot (a): what the V1 grid saw ---
    base_ax(ax1, "delta (neuron intervention magnitude)")
    ax1.axvspan(0.0, 2.0, color="lightgrey", alpha=0.18, zorder=1)
    ax1.plot(delta, pi, color=line_dark, linewidth=2.0, zorder=3,
             label="true A curve (V2, 15 points)")
    ax1.scatter(v1d, v1p, ls="", marker="s", facecolors="white", edgecolors="dimgrey",
                s=58, linewidths=1.2, zorder=4,
                label="V1 A samples (5 points)")
    ax1.plot([0.0, 4.0], [v1_zero_pi, 1.0], color="grey", linestyle=(0, (5, 4)),
             linewidth=1.4, zorder=2, label="V1's only information: δ 0 → 4")

    dh0 = delta_hats[1]
    ax1.vlines(dh0, -0.85, interp_a(dh0), colors=line_dark, linewidth=1.5, zorder=3)
    ax1.vlines(v1_dh0_inferred, -0.85, v1_neutral_pi, colors="dimgrey",
               linestyle=(0, (5, 4)), linewidth=1.5, zorder=3)
    ax1.annotate("V2 Δ̂(0) = {0:+.3f}".format(dh0),
                 xy=(dh0, interp_a(dh0)), xytext=(-2.2, 0.35),
                 fontsize=9, color="dimgrey", ha="center",
                 arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7))
    ax1.annotate("V1 Δ̂(0) ≈ {0:+.2f} (inferred)\nfrom π_B = {1:+.3f} on the fine curve".format(
                     v1_dh0_inferred, v1_neutral_pi),
                 xy=(v1_dh0_inferred, v1_neutral_pi), xytext=(1.6, 0.80),
                 fontsize=9, color="dimgrey", ha="left", va="top",
                 arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7))

    ax1.text(1.0, -1.08, "V1: 0 samples in δ 0–2", fontsize=8.5, color="dimgrey",
             ha="center")
    ax1.text(-8.45, 1.10,
             "{0:.0f}% of the δ 0 → 2 π change\ncompletes by δ = 1.5 — the band\nV1 never sampled".format(
                 frac_by_15 * 100),
             fontsize=9, color="dimgrey", ha="left", va="top")
    ax1.legend(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
               labelcolor="dimgrey", loc="upper left", bbox_to_anchor=(0.0, 0.62),
               fontsize=8)

    ax1.set_xlim(-8.7, 8.7)
    ax1.set_ylim(-1.18, 1.18)
    ax1.set_xticks(np.arange(-8, 9, 2))
    ax1.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax1.set_ylabel("π = (buy − sell) / (buy + sell)", fontsize=10.5, labelpad=6,
                   color="dimgrey")
    ax1.set_title(r"$\bf{(a)}$" + "  V1's five A samples vs the true curve",
                  loc="left", fontsize=11.5, pad=7)

    # --- Plot (b): zoom on the steep region V2 sampled ---
    base_ax(ax2, "delta (zoom: the steep region)")
    mk = (delta >= -1.6) & (delta <= 1.6)
    ax2.plot(delta[mk], pi[mk], color=line_dark, linewidth=2.2, zorder=3)
    ax2.scatter(delta[mk], pi[mk], color=line_dark, s=40, zorder=4,
                edgecolors="white", linewidths=0.8)
    for t, dh in zip(targets, delta_hats):
        bpi = b_by_t[t]["pi"]
        ax2.vlines(dh, t, interp_a(dh), colors="grey", linestyle=(0, (5, 4)),
                   linewidth=1.2, zorder=2)
        ax2.scatter([dh], [t], marker="x", color="dimgrey", s=64, linewidths=1.6,
                    zorder=5)
        ax2.scatter([dh], [bpi], marker="D", color=accent, s=40, edgecolors="white",
                    linewidths=0.8, zorder=5)
    pos = {(-0.3): (-1.08, -0.95), (0.0): (0.50, -0.75), (0.3): (0.58, 0.30)}
    for t, dh in zip(targets, delta_hats):
        bpi = b_by_t[t]["pi"]
        tx, ty = pos[t]
        ax2.annotate("Δ̂({0:+.1f}) = {1:+.3f}\nπ_B = {2:+.3f}".format(t, dh, bpi),
                     xy=(dh, bpi), xytext=(tx, ty),
                     fontsize=8.5, color="dimgrey", ha="left",
                     arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7))

    ax2.set_xlim(-1.15, 1.65)
    ax2.set_ylim(-1.18, 1.18)
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ax2.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax2.set_title(r"$\bf{(b)}$" + "  V2 resolution: three targets land within 0.083",
                  loc="left", fontsize=11.5, pad=7)

    # --- Plot (c): per-target error V1 vs V2 ---
    ax3.grid(False)
    ax3.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    x = np.arange(len(targets))
    w = 0.36
    b1 = ax3.bar(x - w / 2, v1_err, width=w, color="lightgrey", edgecolor="dimgrey",
                 linewidth=0.8, zorder=3,
                 label="V1 coarse grid (RMSE {0:.3f})".format(
                     float(np.sqrt(np.mean(np.square(v1_err))))))
    b2 = ax3.bar(x + w / 2, v2_err, width=w, color=accent, edgecolor="white",
                 linewidth=0.8, zorder=3,
                 label="V2 fine grid (RMSE {0:.3f})".format(result["rmse"]))
    ax3.axhline(0.25, color="dimgrey", linestyle=(0, (5, 4)), linewidth=1.0, zorder=2)
    ax3.text(2.44, 0.262, "frozen max-error gate 0.25", fontsize=7.5,
             color="dimgrey", ha="right")
    for bars in (b1, b2):
        for r in bars:
            ax3.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                     "{0:.3f}".format(r.get_height()), fontsize=7.5, color="dimgrey",
                     ha="center")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["t = −0.3", "t = 0.0", "t = +0.3"])
    ax3.set_ylim(0, 0.85)
    ax3.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax3.set_xlabel("target t", fontsize=10.5, labelpad=6, color="dimgrey")
    ax3.set_ylabel("|π_B(Δ̂) − t|", fontsize=10.5, labelpad=6, color="dimgrey")
    ax3.set_title(r"$\bf{(c)}$" + "  per-target error: V1 vs V2",
                  loc="left", fontsize=11.5, pad=7)
    ax3.legend(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
               labelcolor="dimgrey", loc="upper right", fontsize=8)

    # --- Figure-level insight ---
    fig.text(
        0.99, 0.01,
        "Same coordinate, same B population; only the A grid changed: RMSE {0:.3f} → "
        "{1:.3f} (gate pass). A→B offset at δ=0: {2:+.3f}. B is the V1 transfer set "
        "(reevaluation, not fresh holdout).".format(
            float(np.sqrt(np.mean(np.square(v1_err)))), result["rmse"], ab_offset),
        ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
    )

    sns.despine(left=True, bottom=True)

    # --- Save ---
    plt.savefig(str(OUT / "calibration_v2_explained.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(str(OUT / "calibration_v2_explained.png"), dpi=150, bbox_inches="tight")
    print("saved", OUT / "calibration_v2_explained.png")


if __name__ == "__main__":
    main()
