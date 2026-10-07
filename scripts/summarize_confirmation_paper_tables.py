"""Paper dose-grid and DIM-validation tables from confirmation-v1 plus its dose supplement (CPU only).

Every cell is recomputed from stored generated rows with the confirmation-v1 ITT definitions
(``summary.flip_stats``); the source run and the supplement run are only read. A cell whose rows are not
all present yet is reported as pending; a direction without source-class companies is "---". Each cell
carries a 95% percentile bootstrap CI over companies (``summary.bootstrap_ci``: 2000 draws, fixed seed);
Rand and Delta resample the same companies jointly across DIM and the five random directions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

from llm_bias.core.steering import summary as S

MODELS = (("qwen3.5-4b", "Qwen3.5-4B"), ("glm4-9b-0414", "GLM-4-9B"),
          ("gemma4-12b-it", "Gemma-4-12B"), ("gpt-oss-20b", "GPT-OSS-20B"))
DOSES = (0.25, 2.0, 4.0, 8.0, 16.0)
OPERATORS = (("ops", "neuron", "Single Neuron"), ("dim", "dim", "DIM (1D)"), ("ops", "cone2", "2D Cone"),
             ("ops", "cone4", "4D Cone"), ("ops", "cone4_projection", "4D Cone (Proj)"),
             ("ops", "dim_orth_rand4", "4D Cone (Rand)"))
RANDOM = tuple(f"random_s{s}" for s in range(5))
SIGNS = {1.0: ("sell", "buy", "Sell $\\to$ Buy\\\\$(\\alpha > 0)$"), -1.0: ("buy", "sell", "Buy $\\to$ Sell\\\\$(\\alpha < 0)$")}
PENDING, UNTESTABLE = "pending", "untestable"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--source-run-id", default="confirmation-v1-20260925-full-01")
    parser.add_argument("--supplement-run-id", default="confirmation-v1-supp-20261006-full-01")
    parser.add_argument("--out", required=True, help="JSON output with every cell and its provenance")
    return parser.parse_args(argv)


class Model:
    def __init__(self, root: Path, slug: str, source: str, supplement: str) -> None:
        runs = root / slug / "concept-cone-steering" / "runs"
        self.slug, self.runs = slug, {"source": runs / source, "supplement": runs / supplement}
        self._arms: dict[tuple[str, str], dict[str, Any] | None] = {}
        self.alpha0 = json.loads((self.runs["source"] / "alpha0" / "result.json").read_text())["rows"]

    def arm(self, run: str, arm: str) -> dict[str, Any] | None:
        if (run, arm) not in self._arms:
            path = self.runs[run] / arm / "result.json"
            self._arms[(run, arm)] = json.loads(path.read_text()) if path.exists() else None
        return self._arms[(run, arm)]

    def rows(self, arm: str, op: str, alpha: float) -> tuple[dict[str, Any] | None, str | None]:
        """All keys' rows of ``op`` at ``alpha`` and the run that holds them (None if not yet complete)."""
        for run in ("source", "supplement"):
            result = self.arm(run, arm)
            if result is None or op not in result["operators"]:
                continue
            spec = result["operators"][op]
            if alpha not in spec["grid"]:
                continue
            j = spec["grid"].index(alpha)
            stored = result["rows"].get(op, {})
            if any(key not in stored for key in spec["keys"]):
                return None, run
            return {key: stored[key][j] for key in spec["keys"]}, run
        return None, None

    def alphas(self, arm: str, op: str) -> list[float]:
        found = set()
        for run in ("source", "supplement"):
            result = self.arm(run, arm)
            if result is not None and op in result["operators"]:
                found |= set(result["operators"][op]["grid"])
        return sorted(found)

    def drift(self, arm: str) -> list[str]:
        result = self.arm("supplement", arm)
        return [d["label"] for d in (result or {}).get("operator_drift", [])]

    def baseline(self, condition: str, keys: Sequence[str]) -> dict[str, Any]:
        return {k: self.alpha0[condition][k] for k in keys}


def cell(value: float | None, status: str | None = None, run: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"value": value, "status": status or ("ok" if value is not None else PENDING), "run": run, **extra}


def with_ci(c: dict[str, Any], units: Sequence[Any], statistic: Callable[[Sequence[Any]], float | None]
            ) -> dict[str, Any]:
    ci = S.bootstrap_ci(list(units), statistic)
    if ci["point"] is None or abs(ci["point"] - c["value"]) > 1e-12:
        raise ValueError("bootstrap point estimate differs from the reported cell value")
    return {**c, "lower": ci["lower"], "upper": ci["upper"]}


def mean_or_none(xs: Sequence[float]) -> float | None:
    return fmean(xs) if xs else None


def on_flip(model: Model, arm: str, op: str, alpha: float, condition: str = "balanced") -> dict[str, Any]:
    rows, run = model.rows(arm, op, alpha)
    if rows is None:
        return cell(None, run=run)
    stats = S.flip_stats(model.baseline(condition, list(rows)), rows, alpha)
    if not stats["on_class_n"]:
        return cell(None, UNTESTABLE, run)
    source, goal = S.on_target(alpha)
    base = model.baseline(condition, list(rows))
    units = [float(rows[k]["decision"] == goal) for k in sorted(rows) if base[k]["decision"] == source]
    return with_ci(cell(stats["on_flip_itt"], run=run, k=stats["on_flip"], n=stats["on_class_n"],
                        parse_rate=stats["parse_rate"]), units, mean_or_none)


def readout_disagreement(model: Model, alpha: float) -> dict[str, Any]:
    rows, run = model.rows("dim", "dim", alpha)
    if rows is None:
        return cell(None, run=run)
    parsed = [r for r in rows.values() if r["decision"] in ("buy", "sell")]
    units = [float((r["margin"] > 0) != (r["decision"] == "buy")) for r in parsed]
    if not units:
        return cell(None, UNTESTABLE, run)
    return with_ci(cell(fmean(units), run=run, k=int(sum(units)), n=len(units)), units, mean_or_none)


def company_units(model: Model, alpha: float) -> tuple[list[dict[str, Any]] | None, set[str], str | None]:
    """Per-company DIM and random-direction outcomes at ``alpha`` for joint resampling."""
    source, goal = S.on_target(alpha)
    dim, dim_run = model.rows("dim", "dim", alpha)
    seeds, runs = {}, set()
    for op in RANDOM:
        rows, run = model.rows("random", op, alpha)
        if rows is None:
            return None, runs, run
        seeds[op], _ = rows, runs.add(run)
    base = model.baseline("balanced", list(seeds[RANDOM[0]]))
    units = []
    for k in sorted(base):
        b = base[k]["decision"]
        unit = {"source": b == source, "eligible": b in (source, goal),
                "seeds": [(b == source and seeds[op][k]["decision"] == goal) or
                          (b == goal and seeds[op][k]["decision"] == source) for op in RANDOM]}
        if dim is not None:
            unit["dim_flip"] = dim[k]["decision"] == goal
        units.append(unit)
    return units, runs, dim_run


def rand_stat(units: Sequence[dict[str, Any]]) -> float | None:
    eligible = sum(u["eligible"] for u in units)
    if not eligible:
        return None
    return max(sum(u["seeds"][i] for u in units) / eligible for i in range(len(RANDOM)))


def delta_stat(units: Sequence[dict[str, Any]]) -> float | None:
    sources = [u["dim_flip"] for u in units if u["source"]]
    rand = rand_stat(units)
    return None if not sources or rand is None else fmean(sources) - rand


def random_max(model: Model, alpha: float) -> dict[str, Any]:
    units, runs, run = company_units(model, alpha)
    if units is None:
        return cell(None, run=run)
    per_seed = {op: S.flip_stats(model.baseline("balanced", list(model.rows("random", op, alpha)[0])),
                                 model.rows("random", op, alpha)[0], alpha)["any_flip_itt"] for op in RANDOM}
    drift = model.drift("random") if "supplement" in runs else []
    return with_ci(cell(max(per_seed.values()), run="/".join(sorted(runs)), per_seed=per_seed, operator_drift=drift),
                   units, rand_stat)


def delta(model: Model, alpha: float, dim: dict[str, Any], rand: dict[str, Any]) -> dict[str, Any]:
    if dim["value"] is None or rand["value"] is None:
        return cell(None, UNTESTABLE if UNTESTABLE in (dim["status"], rand["status"]) else None)
    units, _, _ = company_units(model, alpha)
    return with_ci(cell(dim["value"] - rand["value"]), units, delta_stat)


def never_flip(model: Model, alpha: float, condition: str) -> dict[str, Any]:
    """Share of source-class companies with no on-target flip at any tested dose of this sign up to |alpha|."""
    op = f"dim_{condition}"
    source, goal = S.on_target(alpha)
    tested = [a for a in model.alphas("evidence", op) if a * alpha > 0 and abs(a) <= abs(alpha)]
    flipped: dict[str, bool] = {}
    for a in tested:
        rows, run = model.rows("evidence", op, a)
        if rows is None:
            return cell(None, run=run)
        for key, row in rows.items():
            flipped[key] = flipped.get(key, False) or row["decision"] == goal
    base = model.baseline(condition, list(flipped))
    sources = [k for k in flipped if base[k]["decision"] == source]
    if not sources:
        return cell(None, UNTESTABLE)
    units = [float(not flipped[k]) for k in sorted(sources)]
    return with_ci(cell(fmean(units), k=int(sum(units)), n=len(units), doses=tested), units, mean_or_none)


def build(model: Model) -> dict[str, Any]:
    base = model.alpha0["balanced"]
    targets = model.arm("source", "dim")["operators"]["dim"]["keys"]
    out: dict[str, Any] = {"baseline": {"buy": sum(base[k]["decision"] == "buy" for k in targets),
                                        "sell": sum(base[k]["decision"] == "sell" for k in targets),
                                        "n": len(targets)},
                           "dose_grid": {}, "validation": {}}
    for sign in SIGNS:
        for arm, op, _ in OPERATORS:
            out["dose_grid"][f"{op}{sign:+g}"] = {f"{sign * d:+g}": on_flip(model, arm, op, sign * d) for d in DOSES}
        adverse = "neg" if sign > 0 else "pos"
        rows = {}
        for d in DOSES:
            alpha = sign * d
            dim, rand = on_flip(model, "dim", "dim", alpha), random_max(model, alpha)
            rows[f"{alpha:+g}"] = {
                "dim": dim, "ro": readout_disagreement(model, alpha), "rand": rand,
                "delta": delta(model, alpha, dim, rand),
                "still": on_flip(model, "evidence", f"dim_{adverse}", alpha, adverse),
                "never": never_flip(model, alpha, adverse),
                "ae": on_flip(model, "anon", "dim_balanced", alpha)}
        out["validation"][f"{sign:+g}"] = rows
    return out


# Directions the paper tables report: Qwen and GLM have a single unsteered class.
TABLE_SIGNS = {"qwen3.5-4b": (1.0,), "glm4-9b-0414": (-1.0,), "gemma4-12b-it": (1.0, -1.0),
               "gpt-oss-20b": (1.0, -1.0)}
LOW_PARSE = 0.9


def fmt(c: dict[str, Any], pending: str) -> str:
    if c["status"] == UNTESTABLE:
        return "---"
    if c["value"] is None:
        return pending
    marks = ("\\dagger" if c.get("operator_drift") else "") + (
        "\\ddagger" if c.get("parse_rate") is not None and c["parse_rate"] < LOW_PARSE else "")
    text = num(c["value"]) + (f"$^{{{marks}}}$" if marks else "")
    if c.get("lower") is None:
        return text
    return f"\\cival{{{text}}}{{{num(c['lower'])}}}{{{num(c['upper'])}}}"


def num(value: float) -> str:
    return f"{0.0 if round(value, 2) == 0 else value:.2f}"      # no "-0.00"


CI_MACRO = ("% \\cival{value}{lower}{upper}: value with its 95% bootstrap CI stacked below. Inline instead:\n"
            "% \\renewcommand{\\cival}[3]{#1{\\tiny\\,[#2,\\,#3]}}; hide CIs: \\renewcommand{\\cival}[3]{#1}\n"
            "\\providecommand{\\cival}[3]{\\begin{tabular}[c]{@{}c@{}}#1\\\\[-3pt]{\\tiny[#2,\\,#3]}\\end{tabular}}\n")
CI_CAPTION = (r" Brackets: 95\% percentile bootstrap CI over companies (2{,}000 resamples); \emph{Rand} and "
              r"$\Delta$ resample companies jointly with DIM.")


def dose_grid_tex(models: dict[str, Any]) -> str:
    lines = [CI_MACRO + r"\begin{table*}[!t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lll ccccc c ccccc}", r"\toprule",
             r" & & & \multicolumn{11}{c}{\textbf{Dose Strength ($\alpha$) Flip Rates}} \\", r"\cmidrule(lr){4-14}",
             r"\textbf{Direction} & \textbf{Operator} & & " + " & ".join(
                 [f"\\textbf{{${-d:g}$}}" for d in reversed(DOSES)] + [r"\textbf{$0$}"] +
                 [f"\\textbf{{${d:g}$}}" for d in DOSES]) + r" \\", r"\midrule"]
    for slug, name in MODELS:
        data = models[slug]
        lines += ["", f"\\multicolumn{{14}}{{l}}{{\\textbf{{{name}}} \\textit{{(Unsteered Buy/Sell: "
                      f"{data['baseline']['buy']} / {data['baseline']['sell']})}}}} \\\\", r"\midrule"]
        for i, sign in enumerate(TABLE_SIGNS[slug]):
            if i:
                lines.append(r"\cmidrule(lr){1-14}")
            lines.append(f"\\multirow{{{len(OPERATORS)}}}{{*}}{{\\shortstack[l]{{{SIGNS[sign][2]}}}}}")
            for _, op, label in OPERATORS:
                cells = data["dose_grid"][f"{op}{sign:+g}"]
                neg = [fmt(cells[f"{-d:+g}"], "X.XX") if sign < 0 else "---" for d in reversed(DOSES)]
                pos = [fmt(cells[f"{d:+g}"], "X.XX") if sign > 0 else "---" for d in DOSES]
                lines.append(f" & {label:<15}& & " + " & ".join(neg + ["0.00"] + pos) + r" \\")
        lines.append(r"\midrule" if slug != MODELS[-1][0] else r"\bottomrule")
    lines += [r"\end{tabular}",
              r"\caption{Systematic evaluation of decision flip rates across the dose grid ($\alpha$) for Single "
              r"Neuron, 1D Stance Vector (DIM), and Multi-Axis Concept Cones, grouped by flipping direction. Baseline "
              r"unsteered Buy/Sell counts at $\alpha=0$ are listed beneath each model header. Negative doses "
              r"($\alpha < 0$) target Buy $\to$ Sell flips; positive doses ($\alpha > 0$) target Sell $\to$ Buy "
              r"flips. ``---'' indicates untestable directions or inapplicable dose signs. "
              r"$^\ddagger$: fewer than 90\% of steered outputs parse; unparsed outputs count as not flipped."
              + CI_CAPTION + "}",
              r"\label{tab:dose_grid_flip_rates}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def validation_tex(models: dict[str, Any]) -> str:
    lines = [CI_MACRO + r"\begin{table*}[!t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{l r c c cc cc c}", r"\toprule",
             r" & & & \textbf{RO} & \multicolumn{2}{c}{\textbf{RB}} & \multicolumn{2}{c}{\textbf{CE}} & \textbf{AE} \\",
             r"\cmidrule(lr){4-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-9}",
             r"\textbf{Model} & \textbf{$\alpha$} & \textbf{DIM}", r" & \textbf{Disagree} & \textbf{Rand} & \textbf{$\Delta$}",
             r" & \textbf{Still} & \textbf{Never} & \textbf{Flip} \\", r"\midrule"]
    for slug, name in MODELS:
        signs = TABLE_SIGNS[slug]
        lines += ["", f"\\multirow{{{6 * len(signs)}}}{{*}}{{{name}}}"]
        for i, sign in enumerate(signs):
            if i:
                lines.append(r"\cmidrule(lr){2-9}")
            label = "Sell $\\to$ Buy" if sign > 0 else "Buy $\\to$ Sell"
            lines += [f" & \\multicolumn{{8}}{{l}}{{\\textit{{{label}}}}} \\\\", r"\cmidrule(lr){2-9}"]
            for alpha, r in models[slug]["validation"][f"{sign:+g}"].items():
                vals = [fmt(r[k], r"\ph") for k in ("dim", "ro", "rand", "delta", "still", "never", "ae")]
                lines.append(f" & ${float(alpha):g}$ & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule" if slug != MODELS[-1][0] else r"\bottomrule")
    lines += [r"\end{tabular}",
              r"\caption{Validation of DIM steering across the dose grid. Positive doses target \texttt{sell}$\to$"
              r"\texttt{buy} and negative doses target \texttt{buy}$\to$\texttt{sell}; Qwen and GLM each have a single "
              r"unsteered class, so only one direction is testable. \textbf{DIM}: share of source-class companies "
              r"flipped to the target class on the balanced prompt. \textbf{RO} (Readout): share of parsed outputs whose "
              r"margin sign disagrees with the generated decision. \textbf{RB} (Random Baseline): \emph{Rand} is the "
              r"largest share, over five matched-norm random directions, of all evaluation companies whose decision "
              r"changes in either direction; $\Delta = \text{DIM} - \text{Rand}$. \textbf{CE} (Counter-Evidence): flip "
              r"rate when the prompt's evidence opposes the target decision (\emph{Still}), and share of source-class "
              r"companies with no flip at any tested dose of that sign up to $|\alpha|$ (\emph{Never}). \textbf{AE} "
              r"(Anonymized Entity): DIM flip rate on the ten anonymous identities. $^\dagger$: random directions "
              r"rescaled to a DIM direction re-estimated on different GPU hardware (supplement run)."
              + CI_CAPTION + "}",
              r"\label{tab:validation}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = {slug: build(Model(Path(args.artifacts), slug, args.source_run_id, args.supplement_run_id))
              for slug, _ in MODELS}
    out = Path(args.out)
    out.write_text(json.dumps({"source_run_id": args.source_run_id, "supplement_run_id": args.supplement_run_id,
                               "models": result}, indent=1, allow_nan=False))
    out.with_name("table_dose_grid.tex").write_text(dose_grid_tex(result))
    out.with_name("table_validation.tex").write_text(validation_tex(result))
    pending = sum(c["status"] == PENDING for m in result.values() for g in m["dose_grid"].values() for c in g.values())
    print(f"wrote {out} and two .tex tables; pending dose-grid cells (all directions): {pending}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
