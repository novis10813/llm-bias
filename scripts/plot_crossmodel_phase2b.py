# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Cross-model Phase 2B entity-state layer sweep comparison.

Renders the Qwen3.5-4B formal run alongside every cross-model 2B run that
exists under artifacts/<slug>/balanced-evidence-gap-phase2/runs/phase2b-crossmodel-01.

Single panel: mean normalized transfer T vs relative depth l/(n_layers - 1),
one line per model (entity span solid, instruction span dashed, same color per
model), so the entity -> instruction handoff can be compared across models of
different depth.

Usage:
    uv run --no-sync python scripts/plot_crossmodel_phase2b.py [--out DIR]

Saves: <out>/crossmodel_layer_sweep.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# (display name, artifact model slug, 2B run id)
MODELS = [
    ("Qwen3.5-4B", "qwen3.5-4b", "phase2b-gpu-bf16-01"),
    ("Qwen3.6-27B", "qwen3.6-27b", "phase2b-crossmodel-01"),
    ("Gemma-4-12B", "gemma4-12b-it", "phase2b-crossmodel-01"),
    ("GLM-4-9B", "glm4-9b-0414", "phase2b-crossmodel-01"),
    ("GPT-OSS-20B*", "gpt-oss-20b", "phase2b-crossmodel-01"),
]

# Colorblind-safe (Okabe-Ito extended with greys), fixed by display order.
COLORS = [
    "#0072B2",  # Qwen3.5-4B (baseline)
    "#009E73",  # Qwen3.6-27B
    "#E69F00",  # Gemma-4-12B
    "#999999",  # GLM-4-9B
    "#000000",  # GPT-OSS-20B
]


def load_run(slug: str, run_id: str):
    """Return (n_layers, {span: {layer: T}}) for an existing 2B run, else None."""
    path = Path(f"artifacts/{slug}/balanced-evidence-gap-phase2/runs/{run_id}/analyze/summary.json")
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    n_layers = len(s["curves"]["entity"])
    spans = {}
    for span in ("entity", "instruction"):
        spans[span] = {
            int(l): v["mean_normalized_transfer"]
            for l, v in s["curves"][span].items()
        }
    return n_layers, spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./docs/concept-cone-steering/c2-phase2b-16/figures")
    args = parser.parse_args()

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    data = []  # (name, n_layers, spans, color)
    for (name, slug, run_id), color in zip(MODELS, COLORS):
        loaded = load_run(slug, run_id)
        if loaded is None:
            print(f"  (missing) {name}: no 2B run at artifacts/{slug}/.../{run_id}")
            continue
        n_layers, spans = loaded
        data.append((name, n_layers, spans, color))
    if len(data) < 2:
        raise SystemExit("need at least two completed 2B runs to plot the comparison")
    print(f"plotting {len(data)} models: " + ", ".join(n for n, *_ in data))

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.4), dpi=150)

    for name, n_layers, spans, color in data:
        layers = sorted(spans["entity"])
        t_ent = np.array([spans["entity"][l] for l in layers])
        t_ins = np.array([spans["instruction"][l] for l in layers])
        rel = np.array(layers) / (n_layers - 1)
        ax.plot(rel, t_ent, lw=1.8, color=color, label=f"{name} ({n_layers}L)")
        ax.plot(rel, t_ins, lw=1.2, ls="--", color=color, alpha=0.75)

    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("relative depth  $\\ell$ / (n_layers $-$ 1)", fontsize=12)
    ax.set_ylabel("mean normalized transfer T", fontsize=12)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.25, 1.2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    sns.despine(left=True, bottom=True)
    ax.patch.set_edgecolor("lightgrey")
    ax.patch.set_linewidth(0.8)

    from matplotlib.lines import Line2D
    legend = ax.legend(
        loc="center left", fontsize=9,
        frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
        labelcolor="dimgrey",
        title="solid: entity span · dashed: instruction span",
        title_fontsize=9,
    )
    ax.add_artist(legend)

    ax.set_title(
        "Cross-model Phase 2B — entity-state layer sweep by relative depth "
        "(frozen shared-evidence template, 16 companies, 8 transfer directions; "
        "*GPT-OSS-20B is MoE, shown as requested exception)",
        loc="left", fontsize=12, pad=7,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "crossmodel_layer_sweep.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "crossmodel_layer_sweep.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/crossmodel_layer_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
