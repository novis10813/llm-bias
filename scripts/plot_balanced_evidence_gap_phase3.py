# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Plot the balanced-evidence-gap Phase 3 neuron causal-validation results.

Reads the formal run's compact intervene records and renders the
delta-M(δ) curves for the three 2C candidate coordinates against the
matched-control envelope, plus the dial comparison points.

Usage:
    uv run --no-sync python scripts/plot_balanced_evidence_gap_phase3.py

Saves: ./docs/assets/balanced-evidence-gap/phase3_neuron_causal.{pdf,png}
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

RUN = Path("artifacts/qwen3.5-4b/balanced-evidence-gap-phase3/runs/phase3-gpu-bf16-02")
OUT = Path("./docs/assets/balanced-evidence-gap")

CANDIDATES = [(19, 6334), (20, 6520), (26, 2394)]
PRED = {19: -1, 20: +1, 26: -1}  # 2C-rho predicted polarity (delta>0 effect)


def main():
    sns.set_theme(style="whitegrid", font="DejaVu Sans", context="paper")
    records = [json.loads(l) for l in open(RUN / "intervene" / "records.jsonl")]

    cand: dict[tuple, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    ctl_by_coord: dict[int, dict[int, dict[float, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    dial_vals: dict[float, list[float]] = defaultdict(list)
    for r in records:
        if r["kind"] == "candidate":
            cand[(r["layer"], r["neuron"])][r["delta"]].append(r["delta_m"])
        elif r["kind"] == "control":
            ctl_by_coord[r["layer"]][r["neuron"]][r["delta"]].append(r["delta_m"])
        elif r["kind"] == "dial":
            dial_vals[r["delta"]].append(r["delta_m"])

    # control envelope per layer: max over the 10 controls of
    # max |mean delta_m| over the control's two gate deltas
    env = {}
    for layer, by_neuron in ctl_by_coord.items():
        per_ctl = [
            max(abs(statistics.fmean(vals)) for vals in by_delta.values())
            for by_delta in by_neuron.values()
        ]
        env[layer] = max(per_ctl)

    dial_means = {d: statistics.fmean(v) for d, v in dial_vals.items()}

    fig = plt.figure(figsize=(12, 5.0), dpi=150)
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    colors = {(19, 6334): "#d95f02", (20, 6520): "#7570b3", (26, 2394): "#1a9641"}
    # ── (a) candidate curves ──
    for (layer, neuron) in CANDIDATES:
        name = f"L{layer}_n{neuron}"
        deltas = sorted(cand[(layer, neuron)])
        means = [statistics.fmean(cand[(layer, neuron)][d]) for d in deltas]
        ax1.plot(deltas, means, "o-", color=colors[(layer, neuron)], lw=1.5, ms=4, label=name)
    ax1.axhline(0, color="0.4", lw=0.8)
    for layer in (19, 20, 26):
        ax1.axhspan(-env[layer], env[layer], color="0.7", alpha=0.18, lw=0)
    ax1.text(-0.78, env[26] + 0.0012, "max |mean ΔM| over 10 matched random controls per layer",
             fontsize=7.5, color="0.35")
    ax1.annotate("L19: polarity matches 2C prediction\n(δ>0 → sell), |ΔM| below controls",
                 xy=(0.4785, -0.0110), xytext=(0.10, -0.0175), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax1.annotate("L26: polarity OPPOSITE 2C prediction\n(δ>0 → buy), |ΔM| below controls",
                 xy=(0.7812, 0.0111), xytext=(0.16, 0.0155), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax1.set_xlabel("additive Δ (native units, ±s/±2s/±4s)")
    ax1.set_ylabel("mean ΔM (buy−sell logit margin), 16 tickers")
    ax1.set_title("(a) Candidate coordinates — ΔM vs Δ (all positions)", fontsize=10)
    ax1.legend(fontsize=8, loc="center left")
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # ── (b) effect-size comparison (gate point) ──
    pilot = json.load(open(RUN / "pilot" / "summary.json"))
    labels, vals, ecolors = [], [], []
    for (layer, neuron) in CANDIDATES:
        gd = pilot["grid"][f"L{layer}_n{neuron}"]["gate_delta"]
        v = statistics.fmean(cand[(layer, neuron)][gd])
        labels.append(f"L{layer}_n{neuron}")
        vals.append(abs(v))
        ecolors.append(colors[(layer, neuron)])
    sdial = pilot["dial_scale_s"]
    dpos = dial_means.get(+4 * sdial, max(dial_means.values()))
    dneg = dial_means.get(-4 * sdial, min(dial_means.values()))
    labels.append("L15_n8490 (dial, +4s)")
    vals.append(abs(dpos))
    ecolors.append("#4575b4")
    labels.append("L15_n8490 (dial, −4s)")
    vals.append(abs(dneg))
    ecolors.append("#4575b4")
    # control maxes
    for layer in (19, 20, 26):
        labels.append(f"max random control L{layer}")
        vals.append(env[layer])
        ecolors.append("0.6")
    y = np.arange(len(labels))
    ax2.barh(y, vals, color=ecolors, alpha=0.9, height=0.62)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("|mean ΔM| at gate point (log scale)")
    ax2.set_title("(b) Effect size vs matched controls & dial", fontsize=10)
    ax2.axvline(1.0, color="0.3", ls=":", lw=1)
    ax2.text(1.02, 0.15, "1 nat = typical\ndecision margin", fontsize=7.5, color="0.35")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle("Phase 3 — additive mlp_addition on 2C entity-specific coordinates: "
                 "gate 3A fail (0/3 confirmed; effects ≤ matched controls)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "phase3_neuron_causal.pdf", bbox_inches="tight")
    fig.savefig(OUT / "phase3_neuron_causal.png", bbox_inches="tight")
    print(f"saved {OUT}/phase3_neuron_causal.{{pdf,png}}")


if __name__ == "__main__":
    main()
