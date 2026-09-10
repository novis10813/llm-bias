# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Plot the balanced-evidence-gap Phase 2C component-attribution results.

Reads the 2C run's persisted records (compact, no raw tensors) and the
gate re-analysis summary, and renders two panels:
  (a) MLP arm: per-layer top-neuron spearman vs the max matched control,
      with Holm-adjusted sign-flip p; passing layers annotated;
  (b) attention arm: per-head mean paired difference (entity zeroing -
      position-matched control zeroing) across the 5 full-attention
      layers; all effects near zero (null).

Usage:
    uv run --no-sync python scripts/plot_balanced_evidence_gap_phase2c.py

Saves: ./docs/assets/balanced-evidence-gap/phase2c_components.{pdf,png}
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

RUN = Path("artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2c-gpu-bf16-05")
REANALYSIS = Path(
    "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2c-gate-reanalysis-01"
)
OUT = Path("./docs/assets/balanced-evidence-gap")


def main():
    sns.set_theme(style="whitegrid", font="DejaVu Sans", context="paper")
    layers = json.load(open(RUN / "mlp" / "layer_summaries.json"))["layers"]
    gate = json.load(open(REANALYSIS / "analyze" / "summary.json"))["gate_2c"]
    per_layer = gate["mlp_arm"]["per_layer"]
    passing = set(gate["mlp_arm"]["passing_layers"])

    ls = sorted(layers, key=lambda d: d["layer"])
    lay = np.array([d["layer"] for d in ls])
    top_rho = np.array([d["top_spearman"] for d in ls])
    abs_top = np.array([d["abs_top_spearman"] for d in ls])
    ctl_max = np.array([d["control_max_abs_rho"] for d in ls])
    holm_p = np.array([per_layer[str(d["layer"])]["sign_flip_p_adjusted"] for d in ls])

    fig = plt.figure(figsize=(12, 5.2), dpi=150)
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    # ── (a) MLP arm ──
    colors = ["#d95f02" if d["layer"] in passing else "#999999" for d in ls]
    ax1.bar(lay - 0.18, abs_top, width=0.36, color=colors, label="top neuron |ρ| vs entity margin")
    ax1.plot(lay + 0.18, ctl_max, marker="s", ms=4, lw=1.2, color="#4575b4",
             label="max of 10 matched control |ρ|")
    for d in ls:
        l = d["layer"]
        if l in passing:
            ax1.annotate(f"L{l} n{per_layer[str(l)]['top_neuron']}\n"
                         f"ρ={d['top_spearman']:+.2f}, Holm p={holm_p[l - lay[0]]:.3f}",
                         xy=(l - 0.18, abs_top[l - lay[0]]),
                         xytext=(l - 2.6, 0.75), fontsize=7.5,
                         arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.text(12.1, 0.06, "top > all controls, Holm p<0.05,\nsector agreement 4/4 → pass",
             fontsize=7.5, color="#9c3d00")
    ax1.text(31.4, 0.06, "L31 structurally zero:\nno downstream reader of the\nentity position after final block",
             ha="right", fontsize=7.5, color="0.4")
    ax1.set_xlabel("layer ℓ (MLP down-projection at entity position)")
    ax1.set_ylabel("|Spearman ρ| vs pure entity margin")
    ax1.set_title("(a) MLP arm — per-layer top neuron vs matched controls", fontsize=10)
    ax1.set_xlim(11.3, 31.7)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(12, 32, 2))
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # ── (b) attention arm ──
    records = [json.loads(line) for line in open(RUN / "attention" / "records.jsonl")]
    by_head: dict[tuple[int, int], list[float]] = {}
    for r in records:
        paired = r["entity_toward_source_delta_m"] - statistics.fmean(r["control_toward_source_delta_ms"])
        by_head.setdefault((r["layer"], r["head"]), []).append(paired)
    heads = sorted(by_head)
    hx = np.arange(len(heads))
    hmean = np.array([statistics.fmean(v) for v in (by_head[k] for k in heads)])
    layer_of = [k[0] for k in heads]
    ax2.bar(hx, hmean, width=1.0, color=["#d95f02" if m > 0 else "#4575b4" for m in hmean], alpha=0.8)
    ax2.axhline(0, color="0.4", lw=0.8)
    imax = int(np.argmax(np.abs(hmean)))
    ax2.annotate(f"max |effect| = {hmean[imax]:+.4f} nats\n(Holm p = 1.0 → null)",
                 xy=(imax, hmean[imax]), xytext=(imax - 22, 0.009),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    # layer tick labels
    ticks, labels = [], []
    prev = None
    for i, l in enumerate(layer_of):
        if l != prev:
            ticks.append(i - 0.5)
            labels.append(f"L{l}")
            prev = l
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("head (grouped by full-attention layer)")
    ax2.set_ylabel("mean paired ΔM (entity − control zeroing)")
    ax2.set_title("(b) Attention arm — 80 heads, entity→final edge zeroing", fontsize=10)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle("Phase 2C — component attribution in the handoff interval "
                 "(gate 2C: MLP arm pass, attention arm null)", fontsize=11, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "phase2c_components.pdf", bbox_inches="tight")
    fig.savefig(OUT / "phase2c_components.png", bbox_inches="tight")
    print(f"saved {OUT}/phase2c_components.{{pdf,png}}")


if __name__ == "__main__":
    main()
