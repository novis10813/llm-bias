"""T03 control limits: compact data from the tagged runs, and the report tables from that data (CPU only).

  build   read artifacts/ (confirmation-v1-20260925-full-01, confirmation-v1-supp-20261006-full-01) -> data/*.json
  tables  read data/*.json -> the Markdown tables in REPORT.md

Validation cells are computed by scripts/summarize_confirmation_paper_tables.py, the summarizer behind the paper
tables; the pre-registered C5 rule and the readout disagreement count by scripts/summarize_confirmation.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from llm_bias.core.steering import summary as S

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
MODELS = (("qwen3.5-4b", "Qwen3.5-4B"), ("glm4-9b-0414", "GLM-4-9B"),
          ("gemma4-12b-it", "Gemma-4-12B"), ("gpt-oss-20b", "GPT-OSS-20B"))
SOURCE, SUPPLEMENT = "confirmation-v1-20260925-full-01", "confirmation-v1-supp-20261006-full-01"
SIGNS = {"qwen3.5-4b": ("+1",), "glm4-9b-0414": ("-1",), "gemma4-12b-it": ("+1", "-1"), "gpt-oss-20b": ("+1", "-1")}
RANDOM = tuple(f"random_s{s}" for s in range(5))


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_seeds(model, alpha: float) -> list[dict]:
    """Per-seed any-direction flip rate and parse rate of the matched-norm random directions."""
    out = []
    for op in RANDOM:
        rows, run = model.rows("random", op, alpha)
        stats = S.flip_stats(model.baseline("balanced", list(rows)), rows, alpha)
        out.append({"seed": op, "any_flip_itt": stats["any_flip_itt"], "parse_rate": stats["parse_rate"], "run": run})
    return out


def opposing_grid(model, condition: str, sign: float) -> list[dict]:
    out = []
    for alpha in model.alphas("evidence", f"dim_{condition}"):
        if alpha * sign <= 0:
            continue
        rows, run = model.rows("evidence", f"dim_{condition}", alpha)
        stats = S.flip_stats(model.baseline(condition, list(rows)), rows, alpha)
        out.append({"alpha": alpha, "on_class_n": stats["on_class_n"], "on_flip_itt": stats["on_flip_itt"],
                    "parse_rate": stats["parse_rate"], "run": run})
    return out


def build(artifacts: Path) -> None:
    tables = script("summarize_confirmation_paper_tables")
    confirmation = script("summarize_confirmation")
    out = {}
    for slug, _ in MODELS:
        model = tables.Model(artifacts, slug, SOURCE, SUPPLEMENT)
        cells = tables.build(model)
        run = artifacts / slug / "concept-cone-steering" / "runs" / SOURCE
        arm = {a: confirmation.load(run, a) for a in ("dim", "ops", "random", "jitter", "evidence", "anon")}
        arm["alpha0"] = json.loads((run / "alpha0" / "result.json").read_text())   # shared store, no completion flag
        cal = json.loads((run / "cal" / "calibration.json").read_text())
        evidence = arm["evidence"]["summary"]
        out[slug] = {
            "validation": cells["validation"],
            "random_seeds": {sign: {a: random_seeds(model, float(a)) for a in cells["validation"][sign]}
                             for sign in SIGNS[slug]},
            "opposing": {"+1": opposing_grid(model, "neg", 1.0), "-1": opposing_grid(model, "pos", -1.0)},
            "c7_contrasts": evidence["c7"]["contrasts"],
            "anon_baseline": arm["anon"]["summary"]["dim_balanced"]["baseline"],
            "c5": confirmation.c5(arm["alpha0"], arm["dim"], arm["random"], arm["jitter"], cal),
            # the pre-registered C4 count (summarize_confirmation.py: alpha0, dim, ops, evidence) and all generated rows
            "readout": confirmation.c4(arm["alpha0"], [arm[a] for a in ("dim", "ops", "evidence")]),
            "readout_all_arms": confirmation.c4(arm["alpha0"], [arm[a] for a in ("dim", "ops", "random", "evidence", "anon")]),
        }
    DATA.mkdir(exist_ok=True)
    (DATA / "validation.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {DATA / 'validation.json'}")


def ci(c: dict, dagger: bool = False) -> str:
    if c["status"] == "untestable":
        return "—"
    mark = "†" if dagger and c.get("operator_drift") else ""
    return f"{c['value']:.2f}{mark} [{c['lower']:.2f}, {c['upper']:.2f}]"


def tables() -> None:
    data = json.loads((DATA / "validation.json").read_text())
    print("Table 1. DIM validation by dose. Rand: largest share, over five random seeds, of companies whose decision "
          "changes in either direction. †: random rescaled to a re-estimated DIM.\n")
    print("| Model | α | DIM | Readout disagreement | Rand | Gain | Opposing | Anonymous |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for slug, name in MODELS:
        for sign in SIGNS[slug]:
            for alpha, r in sorted(data[slug]["validation"][sign].items(), key=lambda kv: abs(float(kv[0]))):
                print(f"| {name} | {float(alpha):+g} | {ci(r['dim'])} | {ci(r['ro'])} | {ci(r['rand'], True)} "
                      f"| {ci(r['delta'])} | {ci(r['still'])} | {ci(r['ae'])} |")

    print("\nTable 2. Random directions at |α| ≥ 8: parse rate of the seed that sets Rand, and seeds parsing ≥ 90%.\n")
    print("| Model | α | DIM parse | Rand | Parse of that seed | Seeds ≥ 0.90 |")
    print("|---|---:|---:|---:|---:|---:|")
    for slug, name in MODELS:
        for sign in SIGNS[slug]:
            for alpha, seeds in sorted(data[slug]["random_seeds"][sign].items(), key=lambda kv: abs(float(kv[0]))):
                if abs(float(alpha)) < 8:
                    continue
                top = max(seeds, key=lambda s: s["any_flip_itt"])
                dim_parse = data[slug]["validation"][sign][alpha]["dim"]["parse_rate"]
                print(f"| {name} | {float(alpha):+g} | {dim_parse:.2f} | {top['any_flip_itt']:.2f} "
                      f"| {top['parse_rate']:.2f} | {sum(s['parse_rate'] >= 0.9 for s in seeds)}/5 |")

    print("\nTable 3. Opposing evidence over the full calibrated grid (flip rate / parse rate). Sell→buy uses only "
          "N1, N2; buy→sell uses only P1, P2.\n")
    for slug, name in MODELS:
        for sign in SIGNS[slug]:
            label = "sell→buy" if sign == "+1" else "buy→sell"
            cells = [f"{p['alpha']:+g}: {p['on_flip_itt']:.2f} / {p['parse_rate']:.2f}"
                     for p in sorted(data[slug]["opposing"][sign], key=lambda p: abs(p["alpha"]))]
            print(f"- {name}, {label}: " + "; ".join(cells))

    print("\nTable 4. Pre-registered flip-dose contrasts (C7): companies whose unsteered decision is the source class "
          "under both conditions.\n")
    print("| Model | Contrast | Comparable | Higher dose | Equal | Lower | Blocked | Collapsed |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for slug, name in MODELS:
        for c in data[slug]["c7_contrasts"]:
            if not c.get("comparable_n"):
                continue
            print(f"| {name} | {c['adverse']} vs {c['reference']} ({c['direction']}) | {c['comparable_n']} "
                  f"| {c['higher_dose_under_adverse']} | {c['equal_dose']} | {c['lower_dose_under_adverse']} "
                  f"| {c['adverse_blocked']} | {c['adverse_collapsed']} |")

    print("\nTable 5. Anonymous identities: unsteered decisions on the balanced prompt (n = 10).\n")
    print("| Model | buy | sell |")
    print("|---|---:|---:|")
    for slug, name in MODELS:
        b = data[slug]["anon_baseline"]
        print(f"| {name} | {b['buy']} | {b['sell']} |")

    print("\nTable 6. Pre-registered C5 rule at ±α_50 and ±α_hi, and fixed-prefix readout disagreement over all "
          "parsed rows.\n")
    print("| Model | C5 points (α: Gain lower bound) | C5 holds | Readout disagreement, C4 arms | All generated rows |")
    print("|---|---|---|---:|---:|")
    for slug, name in MODELS:
        c5 = data[slug]["c5"]
        pts = "; ".join(f"{p['alpha']:+g}: {p['lower']:.3f}" for p in c5["points"]
                        if p.get("on_class_n") and p.get("lower") is not None)
        cols = [f"{r['fixed_prefix_divergent']}/{r['parsed']} ({r['fixed_prefix_divergent'] / r['parsed']:.3f})"
                for r in (data[slug]["readout"], data[slug]["readout_all_arms"])]
        print(f"| {name} | {pts} | {'yes' if c5['c5_holds'] else 'no'} | {cols[0]} | {cols[1]} |")
    for key, label in (("readout", "C4 arms (alpha0, dim, ops, evidence)"), ("readout_all_arms", "all generated rows")):
        d = sum(data[s][key]["fixed_prefix_divergent"] for s, _ in MODELS)
        p = sum(data[s][key]["parsed"] for s, _ in MODELS)
        print(f"\nAll models, {label}: {d}/{p} ({d / p:.3f}) parsed rows disagree with their fixed-prefix margin.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "tables"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    build(args.artifacts) if args.command == "build" else tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
