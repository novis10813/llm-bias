# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Cross-model Phase 2B: span sweep (v1 16-company) + condition-flip directions (v2 427-company).

1x3 panels, one color per model:
  (a) v1 16-company sweep: entity (solid) vs instruction (dashed) span curves,
      per-layer mean T over the 8 transfer directions
  (b) v2 427-company condition-flip, positive -> negative (427 directions),
      per-layer mean T over directions
  (c) v2 427-company condition-flip, negative -> positive (427 directions)

x = relative depth ell / (n_layers - 1). Models missing either run are skipped
(interim figure: 4/5 models, Qwen3.8-27B pending).

Usage:
    uv run --no-sync python scripts/plot_crossmodel_span_direction.py [--out DIR]

Saves: <out>/crossmodel_span_direction.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# (display name, artifact slug, v1 2B run id, v2 2B run id)
MODELS = [
    ("Qwen3.5-4B", "qwen3.5-4b", "phase2b-gpu-bf16-01", "phase2b-v2-427-01"),
    ("Gemma-4-12B", "gemma4-12b-it", "phase2b-crossmodel-01", "phase2b-v2-427-01"),
    ("GLM-4-9B", "glm4-9b-0414", "phase2b-crossmodel-01", "phase2b-v2-427-01"),
    ("GPT-OSS-20B", "gpt-oss-20b", "phase2b-crossmodel-01", "phase2b-v2-427-01"),
]
COLORS = {
    "Qwen3.5-4B": "#0072B2",
    "Gemma-4-12B": "#E69F00",
    "GLM-4-9B": "#999999",
    "GPT-OSS-20B": "#000000",
}
SHORT = {"Qwen3.5-4B": "4B", "Gemma-4-12B": "Gemma", "GLM-4-9B": "GLM", "GPT-OSS-20B": "gpt-oss"}


def load_v1(slug, run_id):
    p = Path(f"artifacts/{slug}/balanced-evidence-gap-phase2/runs/{run_id}/analyze/summary.json")
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    n = len(s["curves"]["entity"])
    spans = {
        sp: {int(l): v["mean_normalized_transfer"] for l, v in s["curves"][sp].items()}
        for sp in ("entity", "instruction")
    }
    return n, spans


def load_v2(slug, run_id):
    rd = Path(f"artifacts/{slug}/balanced-evidence-gap-phase2/runs/{run_id}")
    if not (rd / "sweep" / "records.jsonl").exists():
        return None
    pairs = json.loads((rd / "pairs" / "directions.json").read_text(encoding="utf-8"))
    n = len(pairs["layers"])
    recs = [json.loads(line) for line in (rd / "sweep" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = {"posneg": {}, "negpos": {}}
    for r in recs:
        if r["span"] != "instruction":
            continue
        src, tgt = r["direction"].split("->")
        g = "posneg" if (src.endswith(":pos") and tgt.endswith(":neg")) else "negpos"
        groups[g].setdefault(r["direction"], {})[int(r["layer"])] = r["normalized_transfer"]
    out = {}
    for g in ("posneg", "negpos"):
        d = groups[g]
        L = np.full((len(d), n), np.nan)
        for i, k in enumerate(sorted(d)):
            lv = d[k]
            L[i, : len(lv)] = [lv[l] for l in sorted(lv)]
        out[g] = (np.nanmean(L, axis=0), len(d))
    return n, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./docs/assets/concept-cone-steering")
    args = parser.parse_args()

    rows = []
    for name, slug, v1_id, v2_id in MODELS:
        v1, v2 = load_v1(slug, v1_id), load_v2(slug, v2_id)
        if v1 is None or v2 is None:
            print(f"skip {name}: v1={v1 is not None} v2={v2 is not None}")
            continue
        rows.append((name, v1[0], v1[1], v2[0], v2[1]))
    if len(rows) < 2:
        raise SystemExit("need at least two models with both v1 and v2 runs")

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    fig = plt.figure(figsize=(16.8, 5.0), dpi=150)
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.055, right=0.99, top=0.80, bottom=0.13, wspace=0.085)
    ax_a, ax_b, ax_c = (plt.subplot(gs[0, i]) for i in range(3))

    depths = lambda n: np.arange(n) / (n - 1)

    # --- shared (b)/(c) y-limits (same metric, both v2 direction groups) ---
    v2_vals = np.concatenate([v2[g][0] for _, _, _, _, v2 in rows for g in ("posneg", "negpos")])
    v2_lo, v2_hi = min(0.0, float(np.nanmin(v2_vals))), float(np.nanmax(v2_vals))
    pad = 0.08 * (v2_hi - v2_lo)
    v2_ylim = (v2_lo - pad, v2_hi + pad)

    # --- (a) v1 span sweep ---
    a_vals = [t for _, n1, spans, _, _ in rows
              for t in list(spans["entity"].values()) + list(spans["instruction"].values())]
    a_lo, a_hi = min(0.0, float(np.min(a_vals))), float(np.max(a_vals))
    a_pad = 0.06 * (a_hi - a_lo)
    ax_a.set_ylim(a_lo - a_pad, a_hi + a_pad)
    ins_peaks, ent_peaks = {}, {}
    for name, n1, spans, _, _ in rows:
        c = COLORS[name]
        ax_a.plot(depths(n1), [spans["entity"][l] for l in sorted(spans["entity"])], lw=1.8, color=c, zorder=3)
        ax_a.plot(depths(n1), [spans["instruction"][l] for l in sorted(spans["instruction"])],
                  lw=1.2, ls="--", color=c, alpha=0.75, zorder=2)

    # --- (b)/(c) v2 direction groups ---
    for ax, g in ((ax_b, "posneg"), (ax_c, "negpos")):
        for name, _, _, n2, v2 in rows:
            ax.plot(depths(n2), v2[g][0], lw=1.8, color=COLORS[name], zorder=3)
        ax.set_ylim(*v2_ylim)

    for ax in (ax_a, ax_b, ax_c):
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xlim(0.0, 1.02)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("relative depth  $\\ell$ / (n_layers $-$ 1)", fontsize=10)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        sns.despine(left=True, bottom=True)
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)

    ax_a.set_title(r"$\bf{(a)}$ span sweep — 16 cos, 8 dir (v1)", loc="left", fontsize=11.5, pad=7)
    ax_b.set_title(r"$\bf{(b)}$ positive → negative — 427 cos (v2)", loc="left", fontsize=11.5, pad=7)
    ax_c.set_title(r"$\bf{(c)}$ negative → positive — 427 cos (v2)", loc="left", fontsize=11.5, pad=7)
    ax_a.set_ylabel("mean normalized transfer T", fontsize=11)

    # --- legends (unified: upper right on all panels) ---
    legend_handles = [plt.Line2D([], [], color=COLORS[r[0]], lw=2.2) for r in rows]
    ax_a.legend(
        legend_handles + [plt.Line2D([], [], color="0.45", lw=1.8),
                          plt.Line2D([], [], color="0.45", lw=1.2, ls="--")],
        [r[0] for r in rows] + ["entity", "instruction"],
        loc="upper right", fontsize=8, title="color: model · style: span", title_fontsize=8,
        frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey",
    )
    for ax in (ax_b, ax_c):
        ax.legend(legend_handles, [r[0] for r in rows],
                  loc="upper right", fontsize=8,
                  frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey")

    # --- peak annotations: marker + label; only on (b)/(c) with arrows.
    # Panel (a) carries no annotations: its purpose is the entity-vs-instruction
    # shape contrast (its peak values live in the caption).
    # Label positions (tx, ty, ha) are hand-tuned per panel to avoid curves/legend;
    # unknown models fall back to directly above their peak.
    def annotate_peaks(ax, peaks, offsets):
        for name, (px, py, layer) in peaks.items():
            tx, ty, ha = offsets.get(name, (px, py + 0.08, "center"))
            ax.scatter([px], [py], s=16, color=COLORS[name], edgecolors="white",
                       linewidths=0.5, zorder=5)
            ax.annotate(
                f"{SHORT.get(name, name)} L{layer} {py:.3f}", xy=(px, py), xytext=(tx, ty),
                fontsize=7.5, color="dimgrey", ha=ha, va="center", zorder=6,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.8),
                arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7, shrinkA=1, shrinkB=2),
            )

    for ax, g in ((ax_b, "posneg"), (ax_c, "negpos")):
        peaks = {}
        for name, _, _, n2, v2 in rows:
            m = v2[g][0]
            pk = int(np.nanargmax(m))
            peaks[name] = (pk / (n2 - 1), m[pk], pk)
        if g == "posneg":
            offsets = {
                "Qwen3.5-4B": (0.30, 0.28, "center"),
                "Gemma-4-12B": (0.57, 0.07, "center"),
                "GLM-4-9B": (0.40, 0.55, "center"),
                "GPT-OSS-20B": (0.62, 0.56, "center"),
            }
        else:
            offsets = {
                "Qwen3.5-4B": (0.38, 0.53, "center"),
                "Gemma-4-12B": (0.57, 0.08, "center"),
                "GLM-4-9B": (0.35, 0.40, "center"),
                "GPT-OSS-20B": (0.46, 0.63, "center"),
            }
        annotate_peaks(ax, peaks, offsets)

    fig.suptitle(
        f"Cross-model Phase 2B — residual state transfer by relative depth "
        f"({len(rows)}/5 models; Qwen3.8-27B pending)",
        fontsize=12.5, y=0.965,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "crossmodel_span_direction.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "crossmodel_span_direction.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/crossmodel_span_direction.{{pdf,png}}  (models: "
          + ", ".join(r[0] for r in rows) + ")")


if __name__ == "__main__":
    main()
