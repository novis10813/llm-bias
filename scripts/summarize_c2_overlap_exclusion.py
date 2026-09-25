"""A2: recompute the C2-427 instruction curve after excluding the steering evaluation companies.

The 427-company C2 run (phase2b-v2-427-01) shares 89 companies with the 101 held-out steering targets.
This re-aggregates its stored per-direction records (no inference) three ways -- all directions, directions
of non-evaluation companies only, evaluation-overlap directions only -- and reports each curve's peak and
T >= 0.7 x peak band next to the frozen registry peak/band.

Writes artifacts/<slug>/concept-cone-steering/runs/audit-v1-<date>/c2_overlap_exclusion.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from llm_bias.core.steering import protocol as R


def curve(records: list[dict], span: str) -> dict[int, dict]:
    by_layer: dict[int, list[float]] = {}
    for r in records:
        if r["span"] == span:
            by_layer.setdefault(int(r["layer"]), []).append(float(r["normalized_transfer"]))
    return {layer: {"mean_T": fmean(v), "n_directions": len(v)} for layer, v in sorted(by_layer.items())}


def peak_band(c: dict[int, dict]) -> dict:
    if not c:
        return {"peak": None, "band": []}
    peak = max(c, key=lambda l: c[l]["mean_T"])
    return {"peak": peak, "peak_T": c[peak]["mean_T"],
            "band": [l for l, v in c.items() if v["mean_T"] >= 0.7 * c[peak]["mean_T"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="audit-v1-<date>")
    parser.add_argument("--models", nargs="+", default=list(R.MODEL_REGISTRY))
    args = parser.parse_args()
    _, _, evaluation = R.split_population(Path(R.POPULATION_CSV), R.SPLIT_SEED)
    held_out = set(evaluation)
    for slug in args.models:
        spec = R.model_spec(slug)
        run = Path("artifacts") / slug / "balanced-evidence-gap-phase2" / "runs" / "phase2b-v2-427-01"
        raw = (run / "sweep" / "records.jsonl").read_bytes()
        records = [json.loads(line) for line in raw.decode().splitlines() if line]
        ticker = lambda r: r["direction"].split("->")[0].split(":")[0]
        groups = {"all": records, "excluding_eval": [r for r in records if ticker(r) not in held_out],
                  "eval_overlap_only": [r for r in records if ticker(r) in held_out]}
        overlap = sorted({ticker(r) for r in groups["eval_overlap_only"]})
        out = {"schema": "audit-v1-c2-overlap-exclusion", "model_slug": slug,
               "source": str(run / "sweep" / "records.jsonl"), "source_sha256": R.sha256_bytes(raw),
               "registry": {"peak": spec.peak, "band": list(spec.band)}, "overlap_companies": len(overlap),
               "curves": {}}
        for name, subset in groups.items():
            out["curves"][name] = {"n_directions": len({r["direction"] for r in subset})}
            for span in ("instruction", "entity", "evidence", "final"):
                c = curve(subset, span)
                out["curves"][name][span] = {"curve": {str(k): v for k, v in c.items()}, **peak_band(c)}
        target = Path("artifacts") / slug / "concept-cone-steering" / "runs" / args.run_id / "c2_overlap_exclusion.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=1) + "\n")
        ins = {k: (v["instruction"]["peak"], v["instruction"]["band"], v["n_directions"]) for k, v in out["curves"].items()}
        print(slug, "overlap companies", len(overlap), ins, flush=True)


if __name__ == "__main__":
    main()
