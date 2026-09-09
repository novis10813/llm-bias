# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Plot the balanced-evidence-gap Phase 1 behavioral results.

Reads the run's analyze/summary.json (compact, no raw tensors) and renders
two panels:
  (a) per-company named median margin (sorted), colored by sector, with the
      anonymous baseline mean;
  (b) per-company named-vs-anonymous gap (paired), with the global mean.

Usage:
    uv run --no-sync python scripts/plot_balanced_evidence_gap.py

Saves: ./docs/assets/balanced-evidence-gap/balanced_evidence_gap.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

RUN = Path("artifacts/qwen3.5-4b/balanced-evidence-gap/runs/balanced-gap-gpu-bf16-01")
OUT = Path("./docs/assets/balanced-evidence-gap")

SECTOR_COLORS = {
    "Information Technology": "#4575b4",
    "Financials": "#72a5b4",
    "Health Care": "#fc8d59",
    "Industrials": "#90c1c6",
}


def main():
    s = json.load(open(RUN / "analyze" / "summary.json"))
    per = s["per_company"]

    tickers = sorted(per, key=lambda t: per[t]["named_margin_median"])
    medians = np.array([per[t]["named_margin_median"] for t in tickers])
    gaps = np.array([per[t]["gap_mean"] for t in tickers])
    sectors = [per[t]["sector"] for t in tickers]
    colors = [SECTOR_COLORS[x] for x in sectors]

    # anonymous baseline: mean of per-company anon means
    anon_means = [per[t]["anon_margin_mean"] for t in tickers]
    anon_mean = float(np.mean(anon_means))
    gap_mean = float(s["global_gap_mean"])

    fig = plt.figure(figsize=(14, 7), dpi=150)
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    # --- (a) per-company named median margin ---
    bars = ax1.barh(tickers, medians, color=colors, edgecolor="white", linewidth=0.5)
    for p, v in zip(bars, medians):
        _x = p.get_width() + (0.03 if v >= 0 else -0.03)
        ax1.text(_x, p.get_y() + p.get_height() / 2, f"{v:+.2f}",
                 ha="left" if v >= 0 else "right", va="center",
                 weight="medium", size=10, color="dimgrey")
    ax1.axvline(x=0, color="lightgrey", linewidth=0.8, zorder=0)
    ax1.axvline(x=anon_mean, color="#bd0c0c", linestyle="--", linewidth=1.2,
                label=f"anonymous baseline ({anon_mean:+.2f})")
    ax1.set_xlabel("median named margin (nats)")
    ax1.set_title(r"$\bf{(a)}$" + " Per-company named margin (balanced evidence, 2 pos + 2 neg)",
                  loc="left", fontsize=12, pad=7)
    ax1.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax1.set_yticks(range(len(tickers)), tickers, fontsize=10)
    ax1.grid(False)
    sns.despine(ax=ax1, left=True, bottom=True)

    # sector legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=SECTOR_COLORS[x])
               for x in SECTOR_COLORS]
    ax1.legend(handles, list(SECTOR_COLORS.keys()), loc="lower right", fontsize=9,
               frameon=True, facecolor="white", framealpha=0.8,
               edgecolor="lightgrey", labelcolor="dimgrey")
    ax1.patch.set_edgecolor("lightgrey")
    ax1.patch.set_linewidth(0.8)

    # --- (b) per-company named-vs-anonymous gap ---
    bars2 = ax2.barh(tickers, gaps, color="#58849f", edgecolor="white", linewidth=0.5)
    for p, v in zip(bars2, gaps):
        _x = p.get_width() + (0.015 if v >= 0 else -0.015)
        ax2.text(_x, p.get_y() + p.get_height() / 2, f"{v:+.2f}",
                 ha="left" if v >= 0 else "right", va="center",
                 weight="medium", size=10, color="dimgrey")
    ax2.axvline(x=0, color="lightgrey", linewidth=0.8, zorder=0)
    ci = s["global_gap_ci_95"]
    ax2.axvline(x=gap_mean, color="#bd0c0c", linestyle="--", linewidth=1.2,
                label=f"global mean ({gap_mean:+.2f})")
    ax2.axvspan(ci["lower"], ci["upper"], color="#bd0c0c", alpha=0.08,
                label=f"95% CI [{ci['lower']:+.2f}, {ci['upper']:+.2f}]")
    ax2.set_xlabel("named margin $-$ anonymous margin (nats)")
    ax2.set_xlim(min(gaps) - 0.25, max(gaps) + 0.25)
    ax2.set_title(r"$\bf{(b)}$" + " Named-vs-anonymous gap (paired, per company)",
                  loc="left", fontsize=12, pad=7)
    ax2.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax2.set_yticks(range(len(tickers)), tickers, fontsize=10)
    ax2.grid(False)
    ax2.legend(loc="lower right", fontsize=9,
               frameon=True, facecolor="white", framealpha=0.8,
               edgecolor="lightgrey", labelcolor="dimgrey")
    sns.despine(ax=ax2, left=True, bottom=True)
    ax2.patch.set_edgecolor("lightgrey")
    ax2.patch.set_linewidth(0.8)

    fig.suptitle("Balanced Evidence Gap — Phase 1 (Qwen3.5-4B, 16 test-split companies, "
                 "positive_count = 2)", fontsize=13, y=1.02)
    gs.update(wspace=0.35)

    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / "balanced_evidence_gap.pdf", bbox_inches="tight")
    plt.savefig(OUT / "balanced_evidence_gap.png", dpi=150, bbox_inches="tight")
    print(f"saved: {OUT}/balanced_evidence_gap.{{pdf,png}}")


if __name__ == "__main__":
    main()
