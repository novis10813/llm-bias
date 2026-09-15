"""Bounded causal probe operator: push the L15 stance axis (and matched random
directions) at the instruction span and re-score the buy/sell margin.

Development only, no gates.  Reuses the frozen prepared prompts and materials of
the stance characterization run, so the stance direction is refit from exactly
the same 6 evaluation pairs and prompt encodings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from llm_bias.core.artifact_paths import file_sha256
from llm_bias.core.lens_artifacts import model_slug
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model
from llm_bias.entity_concept_decision.stance_causal import run_stance_causal

DEFAULT_SOURCE_RUN = "artifacts/qwen3.5-4b/entity-concept-layer-scan/runs/phase1-v2-stance-char16-01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--source-run", type=Path, default=Path(DEFAULT_SOURCE_RUN))
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--doses", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = args.source_run
    prompts_path = source / "prepare" / "prompts.jsonl"
    materials_path = source / "prepare" / "materials.json"
    manifest_path = source / "manifest.json"
    for required in (prompts_path, materials_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"source run input missing: {required}")
    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines()]
    materials = json.loads(materials_path.read_text())
    n_companies = sum(1 for p in prompts if p.get("kind") == "company")
    if n_companies < 3:
        raise ValueError("source run must contain at least 3 company prompts")

    model_path = args.model
    if not Path(model_path).is_dir():
        raise FileNotFoundError("local model is required; no downloads")
    if not torch.cuda.is_available():
        raise RuntimeError("this operator requires the authorized CUDA smoke environment")
    model, tokenizer, device = load_model(model_path, dtype=torch.bfloat16)
    if model.n_layers != 32:
        raise ValueError("expected 32-layer Qwen3.5-4B")
    loaded = load_validated_lens(model=model, model_name=model_path, artifact_root=args.artifact_root)
    lens_ref = {"path": str(loaded.path), "sha256": file_sha256(loaded.path)}
    del loaded  # validated identity; this development run does not decode vocabulary

    two_a_prompts = (args.artifact_root / model_slug(model_path)
                     / "balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01/prepare/prompts.jsonl")
    refs = [{"path": str(p), "sha256": file_sha256(p)} for p in (
        manifest_path, prompts_path, materials_path, two_a_prompts)]
    torch.cuda.reset_peak_memory_stats(device)
    run = run_stance_causal(
        model=model, tokenizer=tokenizer, device=device,
        prompts=prompts, comparisons=materials["comparisons"],
        layer=args.layer, run_id=args.run_id, model_name=model_path,
        artifact_root=args.artifact_root,
        doses=args.doses, n_random=args.n_random, random_seed=args.seed,
        provenance={
            "mode": "model_smoke",
            "review_status": "ai_reviewed_development",
            "source_run": str(source),
            "n_companies": n_companies,
            "upstream": refs,
            "lens": lens_ref,
            "operator": {"path": __file__, "sha256": file_sha256(__file__)},
            "model_config_sha256": file_sha256(Path(model_path) / "config.json"),
        },
    )
    print(f"Completed stance causal probe: {run}", flush=True)


if __name__ == "__main__":
    main()
