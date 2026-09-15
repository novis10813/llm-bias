"""Diagnostic: is the 1D stance direction (L15, decision-predicting) the same object as
the between-company (entity-contrast) signal, or a different direction?

Development only. No gates, no formal claim. Recomputes L15 instruction-span-mean
residuals for the 32 round-2 material rows + 16 companies (all from the frozen
stance-char run prompts), reconstructs the stance direction (mean of the 6 evaluation
pairs), and compares it to the entity-contrast subspace (PCA of the 16 company
residuals) and to the 8 2B entity-pair contrasts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json

import torch

from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.model import load_model

STANCE_CHAR_RUN = "artifacts/qwen3.5-4b/entity-concept-layer-scan/runs/phase1-v2-stance-char16-01"
LAYER = 15
EVAL_KINDS = {"evaluation_at_positive_concept", "evaluation_at_negative_concept"}
# 2B entity-pair directions (directed), from phase2b-gpu-bf16-01/pairs/directions.json
ENTITY_PAIRS = [
    ("NSC:rev0:ord0", "IT:rev0:ord0"),
    ("IT:rev0:ord0", "NSC:rev0:ord0"),
    ("NSC:rev0:ord0", "BDX:rev0:ord0"),
    ("BDX:rev0:ord0", "NSC:rev0:ord0"),
    ("BLK:rev0:ord0", "IT:rev0:ord0"),
    ("IT:rev0:ord0", "BLK:rev0:ord0"),
    ("BLK:rev0:ord0", "BDX:rev0:ord0"),
    ("BDX:rev0:ord0", "BLK:rev0:ord0"),
]


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = float(v.norm())
    return v / n if n > 1e-9 else v


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(_unit(a) @ _unit(b))



def main() -> None:
    model_path = ".cache/models/qwen3.5-4b"
    if not Path(model_path).is_dir():
        raise FileNotFoundError(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    prompts = [json.loads(l) for l in open(Path(STANCE_CHAR_RUN) / "prepare/prompts.jsonl")]
    materials = json.load(open(Path(STANCE_CHAR_RUN) / "prepare/materials.json"))
    comparisons = materials["comparisons"]
    company_ids = [p["id"] for p in prompts if p.get("kind") == "company"]

    model, tokenizer, device = load_model(model_path, dtype=torch.bfloat16)
    if model.n_layers != 32:
        raise ValueError("expected 32 layers")

    vectors: dict[str, torch.Tensor] = {}
    for item in prompts:
        ids = item["input_ids"]
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        captured = record_residuals(model, tensor, [LAYER])
        span = item["instruction_span"]
        block = captured[LAYER][:, span[0]:span[1], :]
        vectors[item["id"]] = block.float().mean(dim=1).detach().cpu().reshape(-1)
        print(f"captured {item['id']} {item.get('kind')}", flush=True)
        del captured, block

    # --- stance direction: mean of the 6 evaluation-pair diffs ---
    eval_diffs = [
        vectors[c["positive_id"]] - vectors[c["negative_id"]]
        for c in comparisons if c["kind"] in EVAL_KINDS
    ]
    stance = _unit(torch.stack(eval_diffs).mean(dim=0))
    n_eval = len(eval_diffs)

    # --- sanity: recompute stance-vs-margin R^2 (should be ~0.605) ---
    margins = json.load(open(Path(STANCE_CHAR_RUN) / "analyze/summary.json"))["company_margins"]
    xs = [float(vectors[c] @ stance) for c in company_ids]
    ys = [margins[c] for c in company_ids]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    r = sxy / (sxx ** 0.5 * syy ** 0.5)
    print(f"\n[sanity] stance n_eval_pairs={n_eval}  stance_vs_margin R^2 = {r*r:.4f} (expect ~0.605)")

    # --- entity-contrast subspace: PCA of the 16 company residuals ---
    X = torch.stack([vectors[c] for c in company_ids])  # [16, 2560]
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    eig = (S ** 2) / (len(X) - 1)  # per-PC variance
    total_var = float(eig.sum())
    pcs = Vt  # rows are PCs
    print(f"\n[entity subspace] total between-company variance = {total_var:.6f}")
    print("  PC   var_frac   cos(stance, PC)")
    cum = 0.0
    for i in range(min(8, len(pcs))):
        cum += float(eig[i])
        print(f"  PC{i+1:<2d}  {float(eig[i])/total_var:8.4f}  (cum {cum/total_var:6.3f})   {abs(_cos(stance, pcs[i])):8.4f}")

    # --- fraction of between-company variance along the stance axis ---
    proj = Xc @ stance
    stance_var = float(proj.pow(2).mean())
    frac = stance_var / total_var
    print(f"\n[fraction] stance-axis variance = {stance_var:.6f}  -> {frac*100:.2f}% of between-company variance")

    # --- cosines with the 8 2B entity-pair contrasts ---
    print("\n[entity-pair contrasts] cos(stance, A-B):")
    for a, b in ENTITY_PAIRS:
        d = vectors[a] - vectors[b]
        print(f"  {a[:12]:14s} - {b[:12]:14s}  cos = {_cos(stance, d):+8.4f}   (margin diff {margins[a]-margins[b]:+7.3f})")

    out = {
        "sanity_stance_vs_margin_r2": round(r * r, 4),
        "n_eval_pairs": n_eval,
        "total_between_company_variance": round(total_var, 6),
        "pc_variance_fraction": [round(float(eig[i]) / total_var, 4) for i in range(min(8, len(pcs)))],
        "pc_abs_cos_with_stance": [round(abs(_cos(stance, pcs[i])), 4) for i in range(min(8, len(pcs)))],
        "stance_axis_variance_fraction": round(frac, 4),
        "entity_pair_abs_cos": [round(abs(_cos(stance, vectors[a] - vectors[b])), 4) for a, b in ENTITY_PAIRS],
        "entity_pair_signed_cos": [round(_cos(stance, vectors[a] - vectors[b]), 4) for a, b in ENTITY_PAIRS],
    }
    out_path = Path(STANCE_CHAR_RUN) / "analyze" / "stance_vs_entity_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
