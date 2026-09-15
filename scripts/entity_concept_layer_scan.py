"""Bounded multi-layer concept-vs-stance separability scan; development only, no formal gates."""
from __future__ import annotations

import argparse
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
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model
from llm_bias.entity_concept_decision.development_materials import build_round2_materials
from llm_bias.entity_concept_decision.layer_scan import run_layer_scan
from scripts.entity_concept_development import ANSWER_PREFIX, prepare_companies, verified_sources

ALL_16_TICKERS = ["ABT", "AMAT", "AXP", "BDX", "BLK", "C", "CSX", "DE", "DHR", "GLW", "GS", "HON", "HPE", "IT", "NSC", "SYK"]

# Early → late coverage: L8 (early), L12 (entity handoff), L15 (instruction peak, round-2 baseline),
# L19/L20/L26 (MLP-arm neuron layers), L23 (late).
DEFAULT_LAYERS = [8, 12, 15, 19, 20, 23, 26]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--companies", choices=["canonical4", "all16"], default="canonical4",
                        help="canonical4 = BDX/IT/BLK/NSC; all16 = full balanced-evidence-gap-2A company pool")
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        raise FileNotFoundError("local model is required; no downloads")
    if not torch.cuda.is_available():
        raise RuntimeError("this operator requires the authorized CUDA smoke environment")
    tickers = ALL_16_TICKERS if args.companies == "all16" else None
    basis, rows, suffix, refs = verified_sources(args.artifact_root, args.model, tickers=tickers)
    materials = build_round2_materials(
        ["docs/entity-concept-decision/details/materials-round2-g.md",
         "docs/entity-concept-decision/details/materials-round2-c.md"],
        expected_concepts=["C", "G"])
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    if model.n_layers != 32:
        raise ValueError("expected 32-layer Qwen3.5-4B")
    loaded = load_validated_lens(model=model, model_name=args.model, artifact_root=args.artifact_root)
    lens_ref = {"path": str(loaded.path), "sha256": file_sha256(loaded.path)}
    del loaded  # validated identity; this development run does not decode vocabulary
    companies = prepare_companies(tokenizer, rows)
    torch.cuda.reset_peak_memory_stats(device)
    run = run_layer_scan(model=model, tokenizer=tokenizer, device=device,
        materials=materials, instruction_suffix=suffix, answer_prefix=ANSWER_PREFIX,
        company_prompts=companies, layers=args.layers, run_id=args.run_id,
        model_name=args.model, artifact_root=args.artifact_root, k8_basis=basis, k8_layer=15,
        provenance={"mode": "model_smoke", "review_status": "ai_reviewed_development", "company_pool": args.companies,
                    "upstream": refs, "lens": lens_ref,
                    "operator": {"path": __file__, "sha256": file_sha256(__file__)},
                    "model_config_sha256": file_sha256(Path(args.model) / "config.json")})
    print(f"Completed layer scan: {run}", flush=True)


if __name__ == "__main__":
    main()
