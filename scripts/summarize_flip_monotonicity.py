"""flip-monotonicity-v1: smoothness of the generated-decision flip-rate curves (no inference).

Reads confirmation-v1 ``dim/result.json`` and ``ops/result.json`` per model and, for each operator and
sign, measures how monotone the ITT on-target flip-rate curve is over the dose grid. Protocol:
docs/concept-cone-steering/flip-monotonicity-v1/proposal.md. Writes
artifacts/<slug>/concept-cone-steering/runs/<out-run-id>/result.json.

Usage: python3 scripts/summarize_flip_monotonicity.py [--out-run-id flip-monotonicity-v1-20260930-01]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = "confirmation-v1-20260925-full-01"
MODELS = {"Qwen3.5-4B": "qwen3.5-4b", "Gemma-4-12B": "gemma4-12b-it",
          "GLM-4-9B": "glm4-9b-0414", "GPT-OSS-20B": "gpt-oss-20b"}
CRITERION_OPS = ("cone4", "dim_orth_rand4", "dim")
OTHER_OPS = ("cone2", "cone4_projection", "neuron")


def curve(per_alpha: Sequence[Mapping[str, Any]], sign: int) -> list[tuple[float, float]]:
    """(|alpha|, ITT on-target flip rate) over grid points of one sign that have a denominator."""
    pts = [(abs(p["alpha"]), p["on_flip_itt"]) for p in per_alpha
           if p["alpha"] * sign > 0 and p.get("on_class_n") and p["on_flip_itt"] is not None]
    return sorted(pts)


def monotonicity(ys: Sequence[float]) -> dict[str, Any]:
    steps = [b - a for a, b in zip(ys, ys[1:])]
    if not steps:
        return {"n_points": len(ys), "monotone_step_fraction": None, "drops": 0, "max_drop": 0.0}
    return {"n_points": len(ys),
            "monotone_step_fraction": sum(s >= 0 for s in steps) / len(steps),
            "drops": sum(s < 0 for s in steps),
            "max_drop": max(0.0, -min(steps))}


def sign_verdict(curves: Mapping[str, Sequence[tuple[float, float]]]) -> dict[str, Any]:
    """cone4 strictly above dim_orth_rand4 and not below dim in monotone fraction, with no more drops."""
    doses = set.intersection(*(set(a for a, _ in curves[op]) for op in CRITERION_OPS))
    if len(doses) < 2:
        return {"status": "untestable", "reason": "fewer than two shared grid points"}
    metrics = {op: monotonicity([y for a, y in curves[op] if a in doses]) for op in CRITERION_OPS}
    frac = {op: m["monotone_step_fraction"] for op, m in metrics.items()}
    drops = {op: m["drops"] for op, m in metrics.items()}
    holds = (frac["cone4"] > frac["dim_orth_rand4"] and frac["cone4"] >= frac["dim"]
             and drops["cone4"] <= min(drops["dim_orth_rand4"], drops["dim"]))
    return {"status": "holds" if holds else "fails", "shared_doses": sorted(doses), "metrics": metrics,
            "tie_with_dim_orth_rand4": frac["cone4"] == frac["dim_orth_rand4"]}


def model_verdict(signs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    testable = {s: v for s, v in signs.items() if v["status"] in ("holds", "fails")}
    if not testable:
        return {"status": "untestable"}
    return {"status": "holds" if all(v["status"] == "holds" for v in testable.values()) else "fails",
            "signs_evaluated": sorted(testable), "single_sign": len(testable) == 1}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_model(slug: str) -> dict[str, Any]:
    base = ROOT / "artifacts" / slug / "concept-cone-steering" / "runs" / SOURCE_RUN
    dim_path, ops_path = base / "dim" / "result.json", base / "ops" / "result.json"
    dim, ops = json.loads(dim_path.read_text(encoding="utf-8")), json.loads(ops_path.read_text(encoding="utf-8"))
    assert dim.get("complete") and ops.get("complete"), slug
    per_alpha = {"dim": dim["summary"]["dim"]["per_alpha"],
                 **{op: ops["summary"][op]["per_alpha"] for op in ops["summary"] if op != "c8"}}
    out: dict[str, Any] = {"inputs": {"dim": {"path": str(dim_path.relative_to(ROOT)), "sha256": sha256(dim_path)},
                                      "ops": {"path": str(ops_path.relative_to(ROOT)), "sha256": sha256(ops_path)}},
                           "signs": {}, "descriptive": {}}
    for label, sign in (("sign_+", 1), ("sign_-", -1)):
        curves = {op: curve(per_alpha[op], sign) for op in CRITERION_OPS}
        out["signs"][label] = sign_verdict(curves) if all(curves.values()) else \
            {"status": "untestable", "reason": "no source-class denominator"}
        out["descriptive"][label] = {
            op: {"curve": curve(per_alpha[op], sign), **monotonicity([y for _, y in curve(per_alpha[op], sign)])}
            for op in (*CRITERION_OPS, *OTHER_OPS) if op in per_alpha and curve(per_alpha[op], sign)}
    out["verdict"] = model_verdict(out["signs"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-run-id", default="flip-monotonicity-v1-20260930-01")
    args = parser.parse_args()
    for name, slug in MODELS.items():
        result = {"protocol": "docs/concept-cone-steering/flip-monotonicity-v1/proposal.md",
                  "source_run": SOURCE_RUN, "model": name, **analyze_model(slug)}
        out_dir = ROOT / "artifacts" / slug / "concept-cone-steering" / "runs" / args.out_run_id
        if (out_dir / "result.json").exists():
            raise SystemExit(f"refusing to overwrite {out_dir}/result.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"{name}: verdict={result['verdict']}")


if __name__ == "__main__":
    main()
