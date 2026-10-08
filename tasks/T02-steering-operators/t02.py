"""T02 steering operators: compact data from the tagged runs, and the report tables from that data (CPU only).

  build   read artifacts/ (confirmation-v1-freeze-20260925, confirmation-v1-20260925-full-01,
          confirmation-v1-supp-20261006-full-01) -> data/*.json
  tables  read data/*.json -> the Markdown tables in REPORT.md

Dose-grid cells are computed by scripts/summarize_confirmation_paper_tables.py, the summarizer behind the paper
tables; the pre-registered C8 verdict by scripts/summarize_confirmation.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from llm_bias.core.steering import summary as S
from llm_bias.core.steering.protocol import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
MODELS = (("qwen3.5-4b", "Qwen3.5-4B"), ("glm4-9b-0414", "GLM-4-9B"),
          ("gemma4-12b-it", "Gemma-4-12B"), ("gpt-oss-20b", "GPT-OSS-20B"))
SOURCE, SUPPLEMENT = "confirmation-v1-20260925-full-01", "confirmation-v1-supp-20261006-full-01"
FREEZE = "confirmation-v1-freeze-20260925"
LOW_PARSE = 0.9


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_status(model, arm: str, op: str, alpha: float) -> dict:
    rows, _ = model.rows(arm, op, alpha)
    source, goal = S.on_target(alpha)
    base = model.baseline("balanced", list(rows))
    n = len(rows)
    unparsed = [r for r in rows.values() if r["decision"] not in ("buy", "sell")]
    sources = [k for k in rows if base[k]["decision"] == source and rows[k]["decision"] in ("buy", "sell")]
    return {"parsed": (n - len(unparsed)) / n,
            "truncated": sum(r["unparsed_kind"] == "truncated" for r in unparsed) / n,
            "collapsed": sum(r["unparsed_kind"] == "collapsed" for r in unparsed) / n,
            "flip_among_parsed_k": sum(rows[k]["decision"] == goal for k in sources), "flip_among_parsed_n": len(sources)}


def build(artifacts: Path) -> None:
    tables = script("summarize_confirmation_paper_tables")
    confirmation = script("summarize_confirmation")
    out = {}
    for slug, _ in MODELS:
        model = tables.Model(artifacts, slug, SOURCE, SUPPLEMENT)
        cells = tables.build(model)
        run = artifacts / slug / "concept-cone-steering" / "runs" / SOURCE
        ops = json.loads((run / "ops" / "result.json").read_text())
        dim = json.loads((run / "dim" / "result.json").read_text())
        gates = json.loads((run / "gates" / "result.json").read_text())["checks"]
        freeze = json.loads((artifacts / slug / "concept-cone-steering" / "runs" / FREEZE / "manifest.json").read_text())

        low_parse = []
        for key, by_alpha in cells["dose_grid"].items():
            op = key[:-2]                                   # keys are "<op>+1" or "<op>-1"
            for alpha, c in by_alpha.items():
                if c.get("parse_rate") is not None and c["parse_rate"] < LOW_PARSE:
                    arm = "dim" if op == "dim" else "ops"
                    low_parse.append({"operator": op, "alpha": float(alpha), **parse_status(model, arm, op, float(alpha))})

        dim_full = [{"alpha": a, **{k: S.flip_stats(model.baseline("balanced", list(rows)), rows, a)[k]
                                     for k in ("on_class_n", "on_flip", "on_flip_itt", "parse_rate")}, "run": r}
                    for a in model.alphas("dim", "dim") for rows, r in [model.rows("dim", "dim", a)]]
        out[slug] = {
            "n_layers": MODEL_REGISTRY[slug].n_layers, "injection_layer": MODEL_REGISTRY[slug].peak,
            "suffix_tokens": freeze["families"]["balanced"]["K"],
            "steer_suffix_ids_sha256": freeze["steer_suffix_ids_sha256"],
            "baseline": cells["baseline"], "dose_grid": cells["dose_grid"], "low_parse": low_parse,
            "dim_full_grid": dim_full,
            "neuron": {k: ops["metadata"]["neuron"][k] for k in ("neuron", "cos", "second_cos", "n_candidates")},
            "cos_to_dim": {op: spec["dose"]["cos_to_dim_median"] for op, spec in
                           {"dim": dim["operators"]["dim"], **ops["operators"]}.items()},
            "c8": confirmation.c8(ops),
            "gates": {k: gates[k] for k in ("alpha0_repeat_identical", "zero_hook_identical", "last_layer_identical")},
            "structural_zero_ok": dim["summary"]["structural_zero_ok"],
        }
    DATA.mkdir(exist_ok=True)
    (DATA / "operators.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {DATA / 'operators.json'}")


OPERATORS = (("neuron", "Single neuron"), ("dim", "DIM"), ("cone2", "2D cone"), ("cone4", "4D cone"),
             ("cone4_projection", "4D cone (Proj)"), ("dim_orth_rand4", "4D cone (RandAxes)"))
SIGNS = {"qwen3.5-4b": ("+1",), "glm4-9b-0414": ("-1",), "gemma4-12b-it": ("+1", "-1"), "gpt-oss-20b": ("+1", "-1")}
DOSES = (0.25, 2.0, 4.0, 8.0, 16.0)


def fmt(c: dict) -> str:
    if c["status"] == "untestable":
        return "—"
    mark = "‡" if c.get("parse_rate") is not None and c["parse_rate"] < LOW_PARSE else ""
    return f"{c['value']:.2f}{mark} [{c['lower']:.2f}, {c['upper']:.2f}]"


def tables() -> None:
    data = json.loads((DATA / "operators.json").read_text())
    print("Table 1. Unsteered decisions on the balanced prompt (101 evaluation companies).\n")
    print("| Model | Injection layer | K | buy | sell |")
    print("|---|---:|---:|---:|---:|")
    for slug, name in MODELS:
        d = data[slug]
        print(f"| {name} | L{d['injection_layer']} | {d['suffix_tokens']} | {d['baseline']['buy']} | {d['baseline']['sell']} |")

    print("\nTable 2. Flip rate by operator and dose (ITT, 95% company bootstrap CI). ‡: parse rate below 90%.")
    for slug, name in MODELS:
        for sign in SIGNS[slug]:
            label = "sell→buy" if sign == "+1" else "buy→sell"
            doses = [f"{float(sign) * d:+g}" for d in DOSES]
            print(f"\n{name}, {label}\n")
            print("| Operator | " + " | ".join(doses) + " |")
            print("|---|" + "---:|" * len(doses))
            for op, op_name in OPERATORS:
                cells = data[slug]["dose_grid"][f"{op}{float(sign):+g}"]
                print(f"| {op_name} | " + " | ".join(fmt(cells[d]) for d in doses) + " |")

    print("\nTable 3. Output status of every ‡ cell (shares of 101 companies).\n")
    print("| Model | Operator | α | Parsed | Truncated | Collapsed | Flip among parsed |")
    print("|---|---|---:|---:|---:|---:|---:|")
    names = dict(OPERATORS)
    for slug, name in MODELS:
        for c in sorted(data[slug]["low_parse"], key=lambda c: (c["operator"], c["alpha"])):
            flips = (f"{c['flip_among_parsed_k']}/{c['flip_among_parsed_n']}" if c["flip_among_parsed_n"] else "—")
            print(f"| {name} | {names[c['operator']]} | {c['alpha']:+g} | {c['parsed']:.2f} | {c['truncated']:.2f} "
                  f"| {c['collapsed']:.2f} | {flips} |")

    print("\nTable 4. DIM over the full calibrated grid (flip rate / parse rate).\n")
    for slug, name in MODELS:
        cells = [f"{p['alpha']:+g}: {'—' if p['on_flip_itt'] is None else format(p['on_flip_itt'], '.2f')} / "
                 f"{p['parse_rate']:.2f}" for p in data[slug]["dim_full_grid"]]
        print(f"- {name}: " + "; ".join(cells))

    print("\nTable 5. Operator geometry and pre-registered C8 smoothness verdict.\n")
    print("| Model | Selected neuron of candidates (cos to mean_p d̂[p], runner-up) | Median over tokens of cos(B[p], d̂[p]): "
          "neuron / cone2 / cone4 / Proj / RandAxes | C8 holds (+ / −) |")
    print("|---|---|---|---|")
    for slug, name in MODELS:
        d = data[slug]
        cos = d["cos_to_dim"]
        c8 = d["c8"]["per_sign"]
        print(f"| {name} | {d['neuron']['neuron']} of {d['neuron']['n_candidates']} ({d['neuron']['cos']:.3f}, "
              f"{d['neuron']['second_cos']:.3f}) | {cos['neuron']:.3f} / {cos['cone2']:.3f} / {cos['cone4']:.3f} / "
              f"{cos['cone4_projection']:.3f} / {cos['dim_orth_rand4']:.3f} | "
              f"{'yes' if c8['sign_+'] else 'no'} / {'yes' if c8['sign_-'] else 'no'} |")

    print("\nGates (all models): " + ", ".join(
        f"{name}: {'pass' if all(data[s]['gates'].values()) and data[s]['structural_zero_ok'] else 'FAIL'}"
        for s, name in MODELS))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "tables"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    build(args.artifacts) if args.command == "build" else tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
