"""T01 layer localization: compact data from the tagged runs, and the report tables from that data (CPU only).

  build   read artifacts/ (phase2b-v2-427-01, audit-v1-20260925, confirmation-v1-20260925-full-01) -> data/*.json
  tables  read data/*.json -> the Markdown tables in REPORT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from llm_bias.core.steering.protocol import MODEL_REGISTRY

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = (("qwen3.5-4b", "Qwen3.5-4B"), ("glm4-9b-0414", "GLM-4-9B"),
          ("gemma4-12b-it", "Gemma-4-12B"), ("gpt-oss-20b", "GPT-OSS-20B"))
STAGE1_RUN = "balanced-evidence-gap-phase2/runs/phase2b-v2-427-01"
AUDIT_RUN = "concept-cone-steering/runs/audit-v1-20260925"
CONFIRM_RUN = "concept-cone-steering/runs/confirmation-v1-20260925-full-01"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_band(curve: dict[int, float]) -> dict:
    peak = max(curve, key=curve.get)
    return {"peak": peak, "peak_T": curve[peak],
            "band": [layer for layer in sorted(curve) if curve[layer] >= 0.7 * curve[peak]]}


def build(artifacts: Path) -> None:
    stage1, overlap, steer = {}, {}, {}
    for slug, _ in MODELS:
        n_layers = MODEL_REGISTRY[slug].n_layers
        summary_path = artifacts / slug / STAGE1_RUN / "analyze" / "summary.json"
        if sha256(summary_path) != MODEL_REGISTRY[slug].c2_sha256:
            raise ValueError(f"{summary_path} differs from the summary bound in MODEL_REGISTRY")
        curves = json.loads(summary_path.read_text())["curves"]
        stage1[slug] = {"n_layers": n_layers, "spans": {
            span: {int(layer): v["mean_normalized_transfer"] for layer, v in c.items()} for span, c in curves.items()},
            "n_directions": {span: {int(layer): v["n_directions"] for layer, v in c.items()}
                             for span, c in curves.items()}}

        audit = json.loads((artifacts / slug / AUDIT_RUN / "c2_overlap_exclusion.json").read_text())
        overlap[slug] = {"overlap_companies": audit["overlap_companies"], "subsets": {
            name: {"n_directions": v["n_directions"],
                   **peak_band({int(l): x["mean_T"] for l, x in v["instruction"]["curve"].items()})}
            for name, v in audit["curves"].items()}}

        run = artifacts / slug / CONFIRM_RUN
        c2v3 = json.loads((run / "c2v3" / "result.json").read_text())["summary"]
        gen = json.loads((run / "c2v3_gen" / "result.json").read_text())["summary"]
        tf = {span: {int(l): v["mean_T"] for l, v in c.items() if v["mean_T"] is not None}
              for span, c in c2v3["curves"]["tf"].items()}
        steer[slug] = {"n_layers": n_layers, "injection_layer": MODEL_REGISTRY[slug].peak,
                       "teacher_forced": tf, "self_patch_max_abs": c2v3["self_patch_max_abs"],
                       "generation": [{k: c[k] for k in ("layer", "span", "toward_source_flips", "eligible_pairs",
                                                          "parsed", "n")} for c in gen["per_cell"]],
                       "generation_self_patch_identical": gen["self_patch_identical"]}

    DATA.mkdir(exist_ok=True)
    for name, obj in (("stage1_curves", stage1), ("stage1_overlap_exclusion", overlap),
                      ("steer_prompt_patching", steer)):
        (DATA / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    print(f"wrote {DATA}/stage1_curves.json, stage1_overlap_exclusion.json, steer_prompt_patching.json")


def layers(band: list[int]) -> str:
    """Layer list as ranges, e.g. [1, 8] -> 'L1, L8' and [14, 15, 16] -> 'L14–16'."""
    runs, start = [], band[0]
    for prev, cur in zip(band, band[1:] + [None]):
        if cur != prev + 1:
            runs.append(f"L{start}" if start == prev else f"L{start}–{prev}")
            start = cur
    return ", ".join(runs)


def tables() -> None:
    stage1 = json.loads((DATA / "stage1_curves.json").read_text())
    overlap = json.loads((DATA / "stage1_overlap_exclusion.json").read_text())
    steer = json.loads((DATA / "steer_prompt_patching.json").read_text())

    print("Table 1. Stage 1 instruction-span peak (all 854 directions, and the 676 without evaluation companies).\n")
    print("| Model | Layers | Peak | Depth | T | T without eval companies |")
    print("|---|---:|---:|---:|---:|---:|")
    for slug, name in MODELS:
        s = stage1[slug]
        pb = peak_band({int(l): t for l, t in s["spans"]["instruction"].items()})
        excl = overlap[slug]["subsets"]["excluding_eval"]
        print(f"| {name} | {s['n_layers']} | L{pb['peak']} | {pb['peak'] / (s['n_layers'] - 1):.2f} "
              f"| {pb['peak_T']:.3f} | L{excl['peak']}, {excl['peak_T']:.3f} |")

    print("\nTable 2. Steering-prompt patching (teacher-forced readout) and patch-under-generation.\n")
    print("| Model | Injection layer | Steer-suffix peak (T) | 70% band | Injection in band | "
          "Generated flips at peak | Entity peak (depth) |")
    print("|---|---:|---:|---|---|---:|---:|")
    for slug, name in MODELS:
        s = steer[slug]
        pb = peak_band({int(l): t for l, t in s["teacher_forced"]["steer_suffix"].items()})
        ent = peak_band({int(l): t for l, t in s["teacher_forced"]["entity"].items()})
        cell = next(c for c in s["generation"] if c["span"] == "steer_suffix" and c["layer"] == pb["peak"])
        flips = (f"{cell['toward_source_flips']}/{cell['eligible_pairs']}" if cell["eligible_pairs"]
                 else "n/a (0 comparable pairs)")
        print(f"| {name} | L{s['injection_layer']} | L{pb['peak']} ({pb['peak_T']:.3f}) | {layers(pb['band'])} "
              f"| {'yes' if s['injection_layer'] in pb['band'] else 'no'} | {flips} "
              f"| L{ent['peak']} ({ent['peak'] / (s['n_layers'] - 1):.2f}) |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "tables"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    build(args.artifacts) if args.command == "build" else tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
