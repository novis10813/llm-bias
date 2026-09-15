"""Bounded Phase 1 V2 development smoke; no formal gates or model fitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from llm_bias.core.analysis.subspaces import basis_from_rows
from llm_bias.core.artifact_paths import file_sha256
from llm_bias.core.artifacts.verification import verify_completed_inputs
from llm_bias.core.lens_artifacts import model_slug
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.entity_concept_decision.development import run_development
from llm_bias.entity_concept_decision.development_materials import build_development_materials, build_round2_materials

E_MANIFEST = "fe5149f9931c2aa0b9779f042fb84971c9ea31ef1f4c403fc701ea3098a70e00"
E_SUMMARY = "ce3c3a9cd443358894691a9681f04f6c1218a840724ebfb2bc98d529e3f5e358"
A_MANIFEST = "5b7f8161fec7807f57065a59410d84058964a7e41b636544f04ec558e3a32d85"
A_PROMPTS = "713e393a1da0c157f7019e672d3496f4c15a4bfe7793a39ca2f162f0b6f7647f"
ANSWER_PREFIX = '{"decision": "'


CANONICAL_TICKERS = {"BDX", "IT", "BLK", "NSC"}


def verified_sources(artifact_root: Path, model_name: str, tickers=None):
    if tickers is None:
        tickers = set(CANONICAL_TICKERS)
    tickers = set(tickers)
    if not tickers:
        raise ValueError("tickers must be non-empty")
    root = artifact_root / model_slug(model_name)
    e = root / "entity-to-dial/runs/entity-to-dial-e-01"
    a = root / "balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01"
    identity = model_slug(model_name)
    verify_completed_inputs(e, dataset="entity-to-dial", model=identity,
                            run_id="entity-to-dial-e-01", expected_manifest_sha256=E_MANIFEST,
                            required={"analyze/summary.json": (E_SUMMARY, None)})
    verify_completed_inputs(a, dataset="balanced-evidence-gap-phase2", model=identity,
                            run_id="phase2a-gpu-bf16-01", expected_manifest_sha256=A_MANIFEST,
                            required={"prepare/prompts.jsonl": (A_PROMPTS, 64)})
    summary = json.loads((e / "analyze/summary.json").read_text())
    basis = basis_from_rows(summary["pca_basis_vectors"][:8], dimension=2560, rank=8)
    rows = [json.loads(line) for line in (a / "prepare/prompts.jsonl").read_text().splitlines()]
    companies = [r for r in rows if r["ticker"] in tickers and r["order"] == 0 and r["reverse"] is False]
    if len(companies) != len(tickers) or len({r["ticker"] for r in companies}) != len(tickers):
        raise ValueError(f"require exactly {len(tickers)} canonical company rows (order 0, reverse False)")
    suffixes = {r["prompt"].split("\n\n—\n\n", 1)[1] for r in companies}
    if len(suffixes) != 1:
        raise ValueError("company instruction suffixes differ")
    refs = [{"path": str(p), "sha256": file_sha256(p)} for p in (
        e / "manifest.json", e / "analyze/summary.json", a / "manifest.json", a / "prepare/prompts.jsonl")]
    return basis, companies, suffixes.pop(), refs


def prepare_companies(tokenizer, rows):
    result = []
    for row in rows:
        original = input_ids(tokenizer, row["formatted"], add_special_tokens=False)
        if original != row["prompt_ids"]:
            raise ValueError("archived tokenizer IDs differ")
        formatted = row["formatted"] + ANSWER_PREFIX
        ids = input_ids(tokenizer, formatted, add_special_tokens=False)
        if ids[:len(original)] != original:
            raise ValueError("answer prefix changes archived IDs")
        start, end = row["instruction_span"]
        result.append({"id": row["id"], "company_id": row["ticker"],
                       "formatted": formatted, "input_ids": ids,
                       "instruction_span": [start, end], "instruction_ids": ids[start:end],
                       "partition": {"before_instruction": [0, start], "instruction": [start, end],
                                     "after_instruction": [end, len(ids)]},
                       "final_position": len(ids)-1, "source_formatted": row["formatted"],
                       "answer_prefix_appended": ANSWER_PREFIX})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--mode", choices=["phase1-v2", "round2"], default="phase1-v2",
                        help="phase1-v2 = original C+S materials; round2 = G + stance-balanced C")
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        raise FileNotFoundError("local model is required; no downloads")
    if not torch.cuda.is_available():
        raise RuntimeError("this operator requires the authorized CUDA smoke environment")
    basis, rows, suffix, refs = verified_sources(args.artifact_root, args.model)
    if args.mode == "round2":
        materials = build_round2_materials(
            ["docs/entity-concept-decision/details/materials-round2-g.md",
             "docs/entity-concept-decision/details/materials-round2-c.md"],
            expected_concepts=["C", "G"])
        concept_ids = ["C", "G"]
    else:
        materials = build_development_materials(
            "docs/entity-concept-decision/details/materials-phase1-v2-draft.md",
            "docs/entity-concept-decision/details/review-materials-phase1-v2.md")
        concept_ids = ["C", "S"]
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    if model.n_layers != 32:
        raise ValueError("expected 32-layer Qwen3.5-4B")
    loaded = load_validated_lens(model=model, model_name=args.model, artifact_root=args.artifact_root)
    lens_ref = {"path": str(loaded.path), "sha256": file_sha256(loaded.path)}
    del loaded  # validated identity; this development run does not decode vocabulary
    companies = prepare_companies(tokenizer, rows)
    torch.cuda.reset_peak_memory_stats(device)
    run = run_development(model=model, tokenizer=tokenizer, device=device, basis=basis,
        materials=materials, instruction_suffix=suffix, answer_prefix=ANSWER_PREFIX,
        company_prompts=companies, run_id=args.run_id, model_name=args.model,
        artifact_root=args.artifact_root, concept_ids=concept_ids,
        provenance={"mode": "model_smoke", "development_mode": args.mode, "review_status": "ai_reviewed_development",
                    "upstream": refs, "lens": lens_ref,
                    "operator": {"path": __file__, "sha256": file_sha256(__file__)},
                    "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
                    "company_scoring": "archive_formatted_plus_decision_prefix"})
    print(f"Completed development run: {run}", flush=True)


if __name__ == "__main__":
    main()
