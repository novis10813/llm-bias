# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Plot the balanced-evidence-gap Phase 2B layer-sweep results.

Reads the 2B run's analyze/summary.json (compact, no raw tensors) and
renders two panels:
  (a) mean normalized transfer T per span vs layer (entity, evidence,
      instruction, final control), with the handoff crossover band shaded;
  (b) entity-span toward-source delta margin vs layer with bootstrap
      95% CI, showing where entity-span sufficiency decays.

Usage:
    uv run --no-sync python scripts/plot_balanced_evidence_gap_phase2b.py

Saves: ./docs/assets/balanced-evidence-gap/phase2b_layer_sweep.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

RUN = Path("artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01")
OUT = Path("./docs/assets/balanced-evidence-gap")

SPANS = {
    "entity": ("Entity span", "#d95f02"),
    "evidence": ("Evidence span", "#7570b3"),
    "instruction": ("Instruction span", "#1b9e77"),
    "final": ("Final (control)", "#999999"),
}


def main():
    sns.set_theme(style="whitegrid", font="DejaVu Sans", context="paper")
    s = json.load(open(RUN / "analyze" / "summary.json"))
    curves = s["curves"]
    handoff = s["handoff"]
    layers = np.arange(32)

    fig = plt.figure(figsize=(12, 5.2), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    # ── (a) normalized transfer per span ──
    band = handoff["crossover_band"]
    ax1.axvspan(band[0] - 0.5, band[-1] + 0.5, color="#fde0d0", zorder=0)
    for span, (label, color) in SPANS.items():
        t = np.array([curves[span][str(l)]["mean_normalized_transfer"] for l in layers])
        ax1.plot(layers, t, marker="o", ms=3.5, lw=1.6, color=color, label=label)
    # annotations
    e = {l: curves["entity"][str(l)]["mean_normalized_transfer"] for l in layers}
    i = {l: curves["instruction"][str(l)]["mean_normalized_transfer"] for l in layers}
    v = {l: curves["evidence"][str(l)]["mean_normalized_transfer"] for l in layers}
    ax1.annotate("entity T ≈ 1.0\n(L0–5)", xy=(2.5, e[2]), xytext=(4.2, 0.80),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    imax = max(i, key=i.get)
    ax1.annotate(f"instruction peak L{imax}\nT = {i[imax]:+.2f}", xy=(imax, i[imax]),
                 xytext=(imax + 2.4, i[imax] + 0.08),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    vmax = max(v, key=v.get)
    ax1.annotate(f"evidence peak L{vmax}\nT = {v[vmax]:+.2f}", xy=(vmax, v[vmax]),
                 xytext=(vmax - 7.6, v[vmax] + 0.07),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax1.text((band[0] + band[-1]) / 2, 1.045, "crossover band "
             f"L{band[0]}–L{band[-1]}\n(context T ≥ entity T)",
             ha="center", va="bottom", fontsize=8, color="#9c3d00")
    ax1.axhline(0, color="0.4", lw=0.8, ls="--")
    ax1.set_xlabel("layer ℓ (patch applied at block-ℓ output)")
    ax1.set_ylabel("mean normalized transfer T")
    ax1.set_title("(a) Entity-state layer sweep: normalized transfer by span", fontsize=10)
    ax1.set_xlim(-0.5, 31.5)
    ax1.set_ylim(-0.15, 1.18)
    ax1.set_xticks(range(0, 32, 4))
    ax1.legend(loc="center left", fontsize=7.5, framealpha=0.9)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # ── (b) entity-span toward-source delta margin with CI ──
    dm = np.array([curves["entity"][str(l)]["mean_toward_source_delta_m"] for l in layers])
    ci_lo = np.array([curves["entity"][str(l)]["delta_m_ci_95"][0]
                      if curves["entity"][str(l)]["delta_m_ci_95"] else np.nan for l in layers])
    ci_hi = np.array([curves["entity"][str(l)]["delta_m_ci_95"][1]
                      if curves["entity"][str(l)]["delta_m_ci_95"] else np.nan for l in layers])
    ax2.fill_between(layers, ci_lo, ci_hi, color="#d95f02", alpha=0.25, label="bootstrap 95% CI (8 directions)")
    ax2.plot(layers, dm, marker="o", ms=3.5, lw=1.6, color="#d95f02", label="mean ΔM toward source")
    ax2.axhline(0, color="0.4", lw=0.8, ls="--")
    ax2.axvspan(band[0] - 0.5, band[-1] + 0.5, color="#fde0d0", zorder=0)
    last_sig = max(
        l for l in layers
        if curves["entity"][str(l)]["delta_m_ci_95"]
        and curves["entity"][str(l)]["delta_m_ci_95"][0] > 0
    )
    ax2.annotate(f"CI excludes 0\nthrough L{last_sig}", xy=(last_sig, dm[last_sig]),
                 xytext=(last_sig - 9.5, 1.55), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax2.set_xlabel("layer ℓ (patch applied at block-ℓ output)")
    ax2.set_ylabel("toward-source ΔM (nats)")
    ax2.set_title("(b) Entity-span sufficiency: toward-source ΔM", fontsize=10)
    ax2.set_xlim(-0.5, 31.5)
    ax2.set_xticks(range(0, 32, 4))
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle("Phase 2B — entity-state layer sweep "
                 f"(8 directions; handoff interval L{handoff['interval'][0]}–L{handoff['interval'][-1]})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "phase2b_layer_sweep.pdf", bbox_inches="tight")
    fig.savefig(OUT / "phase2b_layer_sweep.png", bbox_inches="tight")
    print(f"saved {OUT}/phase2b_layer_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
