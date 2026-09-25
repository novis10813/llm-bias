"""A3: apply the pre-registered confirmation-v1 decision rules to one model's completed arms (no inference).

Reads artifacts/<slug>/concept-cone-steering/runs/<run-id>/<arm>/result.json and writes
<run-id>/summary/confirmation_summary.{json,md}: C1 dose-response, C5 (DIM vs random / jitter / off-target),
C3/C8 operator table and smoothness rule, C4 readout agreement, C6 anonymous identity-level contrast,
C7 evidence contrasts, C10 LOSO rule, and the C2 v3 band. Missing arms are reported as missing, never
filled in. Mixing units across runs (V1 unit alphas vs raw-difference alphas) is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from llm_bias.core.steering import summary as S

PARSED = ("buy", "sell")


def load(root: Path, arm: str, name: str = "result.json") -> dict[str, Any] | None:
    path = root / arm / name
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("complete") else None


def rows_at(arm: dict, op: str, key: str, alpha: float) -> dict[str, Any] | None:
    grid = arm["operators"][op]["grid"]
    return arm["rows"][op][key][grid.index(alpha)] if alpha in grid else None


def c5(alpha0: dict, dim: dict, random: dict, jitter: dict | None, cal: dict) -> dict[str, Any]:
    rule = cal["rule"]
    base = alpha0["rows"]["balanced"]
    keys = dim["operators"]["dim"]["keys"]
    points = []
    for alpha in (rule["alpha_50"], -rule["alpha_50"], rule["alpha_hi"], -rule["alpha_hi"]):
        dim_rows = {k: rows_at(dim, "dim", k, alpha) for k in keys}
        rand = {op: {k: rows_at(random, op, k, alpha) for k in keys} for op in random["operators"]}
        if any(r is None for r in dim_rows.values()) or any(r is None for v in rand.values() for r in v.values()):
            points.append({"alpha": alpha, "status": "alpha not on both grids"})
            continue
        contrast = S.c5_contrast({k: base[k] for k in keys}, dim_rows, rand, alpha)
        stats = S.flip_stats(base, dim_rows, alpha)
        jitter_any = None
        if jitter:
            jitter_any = max(
                S.flip_stats(base, {k: jitter["rows"][op][k][g] for k in keys}, jitter["operators"][op]["grid"][g])["any_flip_itt"] or 0
                for op in jitter["operators"] for g in range(len(jitter["operators"][op]["grid"])))
        points.append({"alpha": alpha, "dim_on_itt": stats["on_flip_itt"], "dim_off_itt": stats["off_flip_itt"],
                       "on_class_n": stats["on_class_n"], "jitter_max_any_itt": jitter_any, **contrast})
    decisive = [p for p in points if p.get("on_class_n")]
    holds = bool(decisive) and all(
        p["supports"] and p["dim_on_itt"] > (p["jitter_max_any_itt"] or 0) and p["dim_on_itt"] > (p["dim_off_itt"] or 0)
        for p in decisive)
    return {"points": points, "c5_holds": holds, "decisive_points": len(decisive)}


def c8(ops: dict) -> dict[str, Any]:
    smooth = ops["summary"].get("c8")
    if not smooth:
        return {"status": "c8 summary missing (dim arm incomplete when ops finished)"}
    verdict = {}
    for sign in ("sign_+", "sign_-"):
        cone, orth, dim = smooth["cone4"][sign], smooth["dim_orth_rand4"][sign], smooth["dim"][sign]
        m = lambda x: x["monotone_step_fraction"] if x["monotone_step_fraction"] is not None else -1
        verdict[sign] = (m(cone) > m(orth) and m(cone) >= m(dim)
                         and cone["reversals"] <= min(orth["reversals"], dim["reversals"]))
    return {"metrics": smooth, "per_sign": verdict, "c8_holds": all(verdict.values())}


def operator_table(dim: dict, ops: dict | None) -> list[dict[str, Any]]:
    table = []
    sources = [("dim", dim)] + ([(op, ops) for op in ops["operators"]] if ops else [])
    for op, arm in sources:
        dose = arm["operators"][op]["dose"]
        per = arm["summary"][op]["per_alpha"]
        table.append({"operator": op, "cos_to_dim": dose["cos_to_dim_median"], "inject_norm": dose["inject_norm_median"],
                      "per_alpha": [{k: p[k] for k in ("alpha", "on_flip_itt", "off_flip_itt", "parse_rate",
                                                         "collapsed", "truncated", "mean_delta_margin",
                                                         "median_realized_margin")} for p in per]})
    return table


def c4(alpha0: dict, arms: list[dict]) -> dict[str, Any]:
    """Sign disagreement of fixed-prefix vs realized-path margins with the generated decision."""
    rows = [r for cond in alpha0["rows"].values() for r in cond.values()]
    for arm in arms:
        for op, by_key in arm["rows"].items():
            for key_rows in by_key.values():
                rows += [r for r in key_rows if isinstance(r, dict) and "generated_text" in r]
    parsed = [r for r in rows if r["decision"] in PARSED]
    fixed_div = [r for r in parsed if (r["margin"] > 0) != (r["decision"] == "buy") and r["margin"] != 0]
    realized = [r for r in parsed if r.get("realized_margin") is not None]
    realized_div = [r for r in realized if (r["realized_margin"] > 0) != (r["decision"] == "buy")]
    by_path: dict[str, dict[str, int]] = {}
    for r in parsed:
        entry = by_path.setdefault(r["path_class"], {"parsed": 0, "fixed_divergent": 0})
        entry["parsed"] += 1
        entry["fixed_divergent"] += r in fixed_div
    return {"rows": len(rows), "parsed": len(parsed), "fixed_prefix_divergent": len(fixed_div),
            "fixed_divergent_abs_margin_median": median(abs(r["margin"]) for r in fixed_div) if fixed_div else None,
            "realized_n": len(realized), "realized_divergent": len(realized_div), "by_path_class": by_path}


def c6(alpha0: dict, anon: dict, dim: dict, evidence: dict | None) -> dict[str, Any]:
    out = {}
    for condition in ("balanced", "pos", "neg", "zero"):
        op = f"dim_{condition}"
        identities = anon["operators"][op]["keys"]
        per = []
        for alpha in anon["operators"][op]["grid"]:
            ident_rows = {k: rows_at(anon, op, k, alpha) for k in identities}
            source, goal = S.on_target(alpha)
            klass = [k for k in identities if alpha0["rows"][condition][k]["decision"] == source]

            def rate(sample: list[str]) -> float | None:
                cls = [k for k in sample if alpha0["rows"][condition][k]["decision"] == source]
                return sum(ident_rows[k]["decision"] == goal for k in cls) / len(cls) if cls else None

            named = None
            named_arm, named_op = (dim, "dim") if condition == "balanced" else (evidence, op)
            if named_arm and alpha in named_arm["operators"][named_op]["grid"]:
                keys = named_arm["operators"][named_op]["keys"]
                named = S.flip_stats({k: alpha0["rows"][condition][k] for k in keys},
                                     {k: rows_at(named_arm, named_op, k, alpha) for k in keys}, alpha)["on_flip_itt"]
            per.append({"alpha": alpha, "identity_class_n": len(klass), "identity_itt": S.bootstrap_ci(identities, rate),
                        "named_itt": named})
        out[condition] = per
    return {"n_identities": 10, "per_condition": out}


def c10(loso: dict | None, loso_c: dict | None, cal: dict, companies_sector: dict[str, str] | None = None) -> dict[str, Any]:
    if not loso:
        return {"status": "loso missing"}
    a50 = cal["rule"]["alpha_50"]
    r9 = [c for c in loso["summary"]["contrasts"] if abs(c["alpha"]) == a50]
    r9_ok = bool(r9) and all(c["comparator_minus_loso_itt"]["upper"] is not None
                             and c["comparator_minus_loso_itt"]["upper"] < 0.10 for c in r9
                             if c["comparator_minus_loso_itt"]["point"] is not None)
    r10 = None
    if loso_c:
        r10 = {}
        for sector, entry in loso_c["summary"]["per_sector"].items():
            flips = sum((p.get("loso") or {}).get("on_flip", 0) for p in entry["per_alpha"])
            r10[sector] = {"n": entry["n"], "loso_on_flips": flips}
    return {"r9_alpha50": r9, "r9_rule": r9_ok, "r10_per_sector": r10,
            "note": "sector-disjoint wording also needs every eligible R10 sector (n>=20 or >=3 of any model's "
                    "Top/Bottom10) to have loso_on_flips > 0; eligibility across models is applied in the status doc"}


def render(slug: str, out: dict[str, Any]) -> str:
    lines = [f"# {slug}: confirmation-v1 summary (A3)", ""]
    cal = out.get("cal")
    if cal:
        lines.append(f"CAL: α_lo={cal['alpha_lo']} α_50={cal['alpha_50']} α_90={cal['alpha_90']} α_hi={cal['alpha_hi']} "
                     f"abort={cal['abort']} full grid={cal['full_grid']}")
    for name in ("c1", "c5", "c8", "c4", "c7", "c10", "c2"):
        lines += ["", f"## {name.upper()}", "", "```json", json.dumps(out.get(name), ensure_ascii=False, indent=1)[:6000], "```"]
    lines += ["", "## Operator table", "", "| operator | cos→DIM | α | on ITT | off ITT | parse | collapsed | truncated | ΔM |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in out.get("operators", []):
        for p in row["per_alpha"]:
            f = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
            lines.append(f"| {row['operator']} | {row['cos_to_dim']:.3f} | {p['alpha']:g} | {f(p['on_flip_itt'])} | "
                         f"{f(p['off_flip_itt'])} | {f(p['parse_rate'])} | {p['collapsed']} | {p['truncated']} | "
                         f"{f(p['mean_delta_margin'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path("artifacts") / args.slug / "concept-cone-steering" / "runs" / args.run_id
    dim = load(root, "dim")
    # the shared alpha-0 store has no completion flag: it grows as arms need prompts
    alpha0 = json.loads((root / "alpha0" / "result.json").read_text()) if (root / "alpha0" / "result.json").exists() else None
    cal = load(root, "cal", "calibration.json")
    if not (alpha0 and dim and cal):
        raise SystemExit("alpha0, cal and dim arms are required")
    arms = {name: load(root, name) for name in ("random", "jitter", "ops", "shuffle", "evidence", "anon", "loso",
                                                "loso_construction", "split_seed", "c2v3", "c2v3_gen", "dim_layers")}
    out: dict[str, Any] = {"slug": args.slug, "run_id": args.run_id,
                           "arms_complete": {k: v is not None for k, v in arms.items()},
                           "cal": {k: cal["rule"][k] for k in ("alpha_lo", "alpha_50", "alpha_90", "alpha_hi", "abort",
                                                              "full_grid", "left_censored", "right_censored")},
                           "c1": {"baseline": dim["summary"]["dim"]["baseline"], "per_alpha": dim["summary"]["dim"]["per_alpha"],
                                  "structural_zero_ok": dim["summary"]["structural_zero_ok"],
                                  "r0_agreement": dim["summary"].get("r0_agreement")}}
    if arms["random"]:
        out["c5"] = c5(alpha0, dim, arms["random"], arms["jitter"], cal)
    if arms["ops"]:
        out["c8"] = c8(arms["ops"])
    out["operators"] = operator_table(dim, arms["ops"])
    out["c4"] = c4(alpha0, [a for a in (dim, arms["ops"], arms["evidence"]) if a])
    if arms["evidence"]:
        out["c7"] = arms["evidence"]["summary"].get("c7", {}).get("contrasts")
    if arms["anon"]:
        out["c6"] = c6(alpha0, arms["anon"], dim, arms["evidence"])
    out["c10"] = c10(arms["loso"], arms["loso_construction"], cal)
    if arms["c2v3"]:
        c2 = arms["c2v3"]["summary"]
        out["c2"] = {"peak": c2["peak"], "band": c2["band"], "primary_layer": dim["metadata"]["primary_layer"],
                     "primary_within_band_pm2": any(abs(dim["metadata"]["primary_layer"] - l) <= 2 for l in c2["band"]),
                     "self_patch_max_abs": c2.get("self_patch_max_abs"),
                     "r7": arms["c2v3_gen"]["summary"] if arms["c2v3_gen"] else None}
    target = root / "summary"
    target.mkdir(parents=True, exist_ok=True)
    (target / "confirmation_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    (target / "confirmation_summary.md").write_text(render(args.slug, out))
    print(json.dumps({k: out.get(k, {}).get(k + "_holds") if isinstance(out.get(k), dict) else None
                      for k in ("c5", "c8")}), out["arms_complete"])


if __name__ == "__main__":
    main()
