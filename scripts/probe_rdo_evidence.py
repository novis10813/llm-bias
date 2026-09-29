"""rdo-cone-evidence-v1: do the gradient-trained directions still respond to the evidence in the prompt?

Protocol: docs/concept-cone-steering/rdo-cone-evidence-v1/proposal.md. Loads the learned unit directions
stored by probe_rdo_cone.py (SHA-256 verified), and evaluates them and DIM on the neg / mixed2 / zero
evidence prompts of the held-out companies with the same dose convention and complete-object parsing.

Output: artifacts/<slug>/concept-cone-steering/runs/<run-id>/result.json (compact rows and SHA digests only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_rdo_cone as base  # noqa: E402

from llm_bias.core.steering import directions as D  # noqa: E402
from llm_bias.core.steering import prompts as P  # noqa: E402
from llm_bias.core.steering import protocol as R  # noqa: E402
from llm_bias.core.steering.evaluate import validate_row  # noqa: E402

SCHEMA = "concept-cone-steering-rdo-cone-evidence-v1"
REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("neg", "mixed2", "zero")
NAMES = ("dim", "rdo1", "rco_b4")
DOSES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=".cache/models/<slug>")
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True, help="rdo-cone-v1-evidence-<date>-<phase>-NN")
    parser.add_argument("--directions-run", required=True, help="run directory holding directions.json")
    parser.add_argument("--ranking-json", default=base.DEFAULT_RANKING)
    parser.add_argument("--population-csv", default=R.POPULATION_CSV)
    parser.add_argument("--names", nargs="+", default=list(NAMES))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=P.CONDITIONS)
    parser.add_argument("--doses", nargs="+", type=float, default=list(DOSES))
    parser.add_argument("--smoke-tickers", nargs="+", default=["ABNB", "AEP"])
    parser.add_argument("--allow-dirty", action="store_true", help="smoke only: permit uncommitted code")
    args = parser.parse_args(argv)
    # Context reads the training hyper-parameters into its metadata; they are irrelevant here
    defaults = base.parse_args(["--model", args.model, "--phase", args.phase, "--run-id", args.run_id])
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


class EvidenceContext(base.Context):
    """The rdo-cone-v1 context with prompts of any evidence condition."""

    def fp_cond(self, ticker: str, condition: str) -> P.FormattedPrompt:
        cache = self.__dict__.setdefault("_fp_cond", {})
        if (ticker, condition) not in cache:
            prompt = P.render_decision_prompt(ticker, self.companies[ticker]["name"], condition)
            cache[(ticker, condition)] = P.format_decision_prompt(
                self.tokenizer, prompt, suffix_tokens=self.spec.suffix_tokens, key=f"{ticker}/{condition}")
        return cache[(ticker, condition)]


def load_directions(source: Path, names: Sequence[str]) -> tuple[dict[str, torch.Tensor], str]:
    path = source / "directions.json"
    stored = json.loads(path.read_text(encoding="utf-8"))["directions"]
    units: dict[str, torch.Tensor] = {}
    for name in names:
        if name == "dim":
            continue
        if name not in stored:
            raise ValueError(f"direction {name!r} is not in {path}")
        unit = torch.tensor(stored[name]["unit"], dtype=torch.float32)
        if D.tensor_sha256(unit) != stored[name]["sha256"]:
            raise ValueError(f"direction {name!r} does not match its stored SHA-256")
        units[name] = unit
    return units, R.sha256_bytes(path.read_bytes())


def run_eval(ctx: EvidenceContext, units: dict[str, torch.Tensor], d: torch.Tensor, dim_stats: dict[str, Any],
             median_h: float, source_sha: str) -> None:
    args = ctx.args
    path = ctx.root / "result.json"
    meta = {k: v for k, v in ctx.metadata.items() if k not in ("hyper", "train_tickers_n")}
    meta.update({"arm": "rdo_evidence", "schema": SCHEMA, "source_directions_sha256": source_sha,
                 "names": list(args.names), "conditions": list(args.conditions), "doses": list(args.doses),
                 "evidence_script_sha256": R.sha256_bytes((REPO / "scripts/probe_rdo_evidence.py").read_bytes())})
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["metadata"] != meta:
            raise ValueError("result metadata differs from this invocation; refusing to resume")
    else:
        result = {"metadata": meta, "dim": dim_stats, "baseline": {}, "dose_tables": {}, "rows": {}, "summary": {},
                  "complete": False}
    bases = {name: (d if name == "dim" else D.equal_norm(d, units[name])) for name in args.names}
    for name, shift in bases.items():
        result["dose_tables"][name] = D.dose_table(d, shift, median_h)
    for condition in args.conditions:
        baseline = result["baseline"].setdefault(condition, {})
        for ticker, row in baseline.items():
            validate_row(row, 0.0, f"baseline/{condition}/{ticker}", baseline=True)
        for ticker in ctx.targets:
            if ticker not in baseline:
                with torch.no_grad():
                    baseline[ticker] = ctx.evaluator.row(ctx.fp_cond(ticker, condition), 0.0, baseline=True,
                                                         label=f"alpha0/{condition}")
                base.atomic_write(path, result)
        for name, shift in bases.items():
            for alpha in args.doses:
                rows = result["rows"].setdefault(condition, {}).setdefault(name, {}).setdefault(base.akey(alpha), {})
                for ticker, row in rows.items():
                    validate_row(row, alpha, f"{condition}/{name}/{base.akey(alpha)}/{ticker}")
                todo = [t for t in ctx.targets if t not in rows]
                for ticker in todo:
                    with torch.no_grad():
                        rows[ticker] = ctx.evaluator.steered_row(ctx.fp_cond(ticker, condition), ctx.layer, shift,
                                                                 alpha, label=f"{condition}/{name}")
                if todo:
                    base.atomic_write(path, result)
    for condition in args.conditions:
        result["summary"][condition] = base.summarize(result["baseline"][condition], result["rows"][condition],
                                                      list(args.names))
    result["complete"] = True
    base.atomic_write(path, result)
    for condition in args.conditions:
        counts = {k: sum(r["decision"] == k for r in result["baseline"][condition].values()) for k in ("buy", "sell")}
        print(f"== {condition} (alpha0 {counts})", flush=True)
        base.print_table(result["summary"][condition], list(args.names))


def main(argv: Sequence[str] | None = None, *, model: Any = None, tokenizer: Any = None) -> int:
    args = parse_args(argv)
    ctx = EvidenceContext(args, model, tokenizer)
    units, source_sha = load_directions(Path(args.directions_run), args.names)
    d, dim_stats, median_h = ctx.dim()
    run_eval(ctx, units, d, dim_stats, median_h, source_sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
