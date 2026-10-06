"""Confirmation-v1 dose supplement: fixed extra alphas for operators of a completed confirmation run.

The source run (ranking, alpha0, cal and the registered operators) is read only. Operators are rebuilt with
the unchanged confirmation-v1 code, checked against the source definitions (direction statistics, cone /
neuron / random identities, dose tables), and generated only at the alphas registered in SUPPLEMENT, which
must not already be in the source grid. Rows go to a new run id with the same compact row schema; one
already-stored source row per operator is regenerated and compared as a replication record.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from llm_bias.core.steering import directions as D
from llm_bias.core.steering import prompts as P

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("probe_steering_confirmation",
                                               REPO / "scripts" / "probe_steering_confirmation.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

SOURCE_RUN = "confirmation-v1-20260925-full-01"
# Alphas missing from the source run for the paper's dose-grid and validation tables.
# "low": dim, ops, evidence (pos/neg) and anon (balanced); "random": the five matched-norm random seeds.
SUPPLEMENT: dict[str, dict[str, tuple[float, ...]]] = {
    "qwen3.5-4b": {"low": (0.25,), "random": (0.25, 2.0, 8.0, 16.0)},
    "glm4-9b-0414": {"low": (-0.25,), "random": (-16.0, -4.0, -2.0, -0.25)},
    "gemma4-12b-it": {"low": (), "random": (-16.0, -8.0, -4.0, -2.0, 2.0, 4.0, 8.0, 16.0)},
    "gpt-oss-20b": {"low": (-0.25, 0.25), "random": (-8.0, -2.0, -0.25, 0.25, 2.0, 8.0)},
}
ARMS = ("dim", "ops", "evidence", "anon", "random")
SOURCE_ARMS = ("ranking", "alpha0", "cal")
# Fields allowed to differ from the source base metadata; everything else must be identical.
ALLOWED_DRIFT = frozenset({"run_id", "phase", "targets", "model"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=".cache/models/<slug>")
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True, help="confirmation-v1-supp-<date>-<phase>-NN")
    parser.add_argument("--source-run-id", default=SOURCE_RUN)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--smoke-tickers", nargs="+", default=["ABNB", "AEP"])
    parser.add_argument("--max-rows", type=int, default=None, help="stop after N generated rows (resume test)")
    parser.add_argument("--population-csv", default=C.R.POPULATION_CSV)
    parser.add_argument("--allow-dirty", action="store_true", help="smoke only: permit uncommitted code")
    parser.add_argument("--allow-operator-drift", action="store_true",
                        help="record, instead of refusing, rebuilt operators that differ from the source")
    return parser.parse_args(argv)


def canonical(value: Any) -> Any:
    return json.loads(json.dumps(value))


class SupplementJob(C.Job):
    """Confirmation job that reads source-run arms and writes only new supplement arms."""

    def __init__(self, args: argparse.Namespace, *, model: Any = None, tokenizer: Any = None) -> None:
        if args.run_id == args.source_run_id:
            raise ValueError("supplement run id must differ from the source run id")
        super().__init__(args, model=model, tokenizer=tokenizer)
        self.source_root = self.root.parent / args.source_run_id
        stored = json.loads((self.source_root / "alpha0" / "result.json").read_text(encoding="utf-8"))["metadata"]
        source_meta = {k: v for k, v in stored.items() if k != "arm"}
        actual = canonical(self.base_metadata)
        drift = sorted(k for k in set(source_meta) | set(actual) if source_meta.get(k) != actual.get(k))
        if set(drift) - ALLOWED_DRIFT:
            raise ValueError(f"base metadata differs from source run beyond {sorted(ALLOWED_DRIFT)}: "
                             f"{sorted(set(drift) - ALLOWED_DRIFT)}")
        if not set(self.targets) <= set(source_meta["targets"]):
            raise ValueError("supplement targets must be source-run targets")
        self.identity = {k: actual[k] for k in drift}
        self.identity["supplement_of"] = {"run_id": args.source_run_id,
                                          "source_values": {k: source_meta.get(k) for k in drift}}
        # source arms validate against their own metadata; supplement arms override the drift fields
        self.base_metadata = source_meta
        self.drift_records: list[dict[str, Any]] = []

    def path(self, arm: str, name: str = "result.json") -> Path:
        return (self.source_root if arm in SOURCE_ARMS else self.root) / arm / name

    def alpha0(self, key: str, condition: str = "balanced") -> dict[str, Any]:
        rows = self.alpha0_store()["rows"].get(condition, {})
        if key not in rows:
            raise ValueError(f"source alpha0 lacks {condition}/{key}; a supplement never generates baselines")
        return rows[key]

    def source_arm(self, arm: str) -> dict[str, Any]:
        stored = json.loads((self.source_root / arm / "result.json").read_text(encoding="utf-8"))
        if not stored.get("complete"):
            raise ValueError(f"source arm {arm} is incomplete")
        return stored

    def check(self, label: str, source: Any, rebuilt: Any) -> None:
        if canonical(source) == canonical(rebuilt):
            return
        record = {"label": label, "source": canonical(source), "rebuilt": canonical(rebuilt)}
        if not self.args.allow_operator_drift:
            raise ValueError(f"rebuilt operator differs from the source run: {label}")
        print(f"WARNING: operator drift recorded for {label}", flush=True)
        self.drift_records.append(record)


def supplement_arm(job: SupplementJob, arm: str, grid: Sequence[float], operators: Mapping[str, tuple],
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``operators``: name -> (base, condition, keys, operator extra or None). Checks the rebuilt operator
    against the source, replicates one stored source row, then generates the supplement alphas."""
    source = job.source_arm(arm)
    drift_start = len(job.drift_records)
    gridarm = C.GridArm(job, arm, {**(dict(extra) if extra else {}), **job.identity,
                                   "supplement_grid": list(grid)})
    for k, v in (extra or {}).items():
        job.check(f"{arm}/metadata/{k}", source["metadata"].get(k), v)
    layer = job.spec.peak
    replication = gridarm.result.setdefault("replication", {})
    for name, (base, condition, keys, op_extra) in operators.items():
        spec = source["operators"][name]
        overlap = sorted(set(grid) & set(spec["grid"]))
        if overlap:
            raise ValueError(f"{arm}/{name}: supplement alphas {overlap} already in the source grid")
        if spec["layer"] != layer or spec["condition"] != condition or not set(keys) <= set(spec["keys"]):
            raise ValueError(f"{arm}/{name}: layer, condition or keys differ from the source operator")
        dose = D.dose_table(job.dim(layer)[0], base, job.median_h(layer))
        job.check(f"{arm}/{name}/dose", spec["dose"], dose)
        source_extra = {k: v for k, v in spec.items() if k not in ("layer", "grid", "condition", "keys", "dose")}
        job.check(f"{arm}/{name}/extra", source_extra, op_extra or {})
        if name not in replication:
            key, alpha = keys[0], spec["grid"][0]
            row = job.evaluator.steered_row(job.fp(key, condition), layer, base, alpha, label=f"{arm}/{name}/replicate")
            stored = source["rows"][name][key][0]
            replication[name] = {"key": key, "alpha": alpha,
                                 "text_identical": row["generated_text"] == stored["generated_text"],
                                 "decision_identical": row["decision"] == stored["decision"],
                                 "margin_abs_diff": abs(row["margin"] - stored["margin"])}
            gridarm.save()
        gridarm.operator(name, layer=layer, base=base, grid=grid, condition=condition, keys=keys,
                         extra=op_extra)
    summary = {name: C.operator_summary(job, gridarm, name, condition)
               for name, (_, condition, _, _) in operators.items()}
    drift = job.drift_records[drift_start:]
    if drift:
        gridarm.result["operator_drift"] = drift
    gridarm.finish(summary)
    return {"replication": replication, "drift": len(drift)}


def run_dim(job: SupplementJob, grid: Sequence[float]) -> dict[str, Any]:
    d, stats = job.dim()
    top, bottom = job.top_bottom()
    return supplement_arm(job, "dim", grid, {"dim": (d, "balanced", job.targets, None)},
                          {"top_10": top, "bottom_10": bottom, "direction": stats})


def run_ops(job: SupplementJob, grid: Sequence[float]) -> dict[str, Any]:
    # same construction as probe_steering_confirmation.arm_ops
    d, _ = job.dim()
    top, bottom = job.top_bottom()
    margins = {r["ticker"]: r["margin"] for r in job.ranking()}
    axes, cone_info = D.cone_axes(job.states(job.spec.peak, top), job.states(job.spec.peak, bottom), d,
                                  [margins[t] for t in top], [margins[t] for t in bottom])
    orth, orth_sha = D.orth_random_axes(d, C.ORTH_SEEDS)
    writes, labels = D.neuron_write_vectors(job.model_dir, job.spec, job.spec.peak)
    pick = D.select_neuron(writes, labels, D.unit_rows(d).mean(dim=0))
    neuron_info = {k: v for k, v in pick.items() if k not in ("unit", "cos_by_label")}
    neuron_info["target"] = "mean_p d_hat[p]"
    if job.slug == "qwen3.5-4b":
        layer, neuron = C.QWEN_DIAL
        d15, _ = job.dim(layer)
        w15, l15 = D.neuron_write_vectors(job.model_dir, job.spec, layer)
        pick15 = D.select_neuron(w15, l15, D.unit_rows(d15).mean(dim=0))
        cos = pick15["cos_by_label"]
        neuron_info["historical_L15"] = {"rule_selects": pick15["neuron"], "n8490_cos": float(cos[neuron]),
                                         "n8490_rank": int((cos > cos[neuron]).sum()) + 1,
                                         "selects_n8490": pick15["neuron"] == str(neuron)}
    del writes
    c2, c4 = D.cone_unit(d, axes, 2), D.cone_unit(d, axes, 4)
    bases = {
        "neuron": D.equal_norm(d, pick["unit"]),
        "cone2": D.equal_norm(d, c2),
        "cone4": D.equal_norm(d, c4),
        "cone4_projection": D.equal_projection(d, c4),
        "dim_orth_rand4": D.equal_norm(d, D.cone_unit(d, orth, 4)),
    }
    return supplement_arm(job, "ops", grid, {n: (b, "balanced", job.targets, None) for n, b in bases.items()},
                          {"cone": cone_info, "orth_random_sha256": orth_sha, "neuron": neuron_info})


def run_evidence(job: SupplementJob, grid: Sequence[float]) -> dict[str, Any]:
    d, _ = job.dim()
    return supplement_arm(job, "evidence", grid, {f"dim_{c}": (d, c, job.targets, None) for c in ("pos", "neg")})


def run_anon(job: SupplementJob, grid: Sequence[float]) -> dict[str, Any]:
    d, _ = job.dim()
    keys = [f"anon:{i}" for i in range(len(P.ANON_IDENTITIES))]
    return supplement_arm(job, "anon", grid, {"dim_balanced": (d, "balanced", keys, None)})


def run_random(job: SupplementJob, grid: Sequence[float]) -> dict[str, Any]:
    d, _ = job.dim()
    operators = {}
    for seed in C.RANDOM_SEEDS:
        u, sha = D.shared_random_direction(d.shape[-1], seed)
        operators[f"random_s{seed}"] = (D.equal_norm(d, u), "balanced", job.targets,
                                        {"seed": seed, "direction_sha256": sha})
    return supplement_arm(job, "random", grid, operators)


RUNNERS = {"dim": run_dim, "ops": run_ops, "evidence": run_evidence, "anon": run_anon, "random": run_random}


def main(argv: Sequence[str] | None = None, *, model: Any = None, tokenizer: Any = None) -> int:
    args = parse_args(argv)
    slug = Path(args.model).resolve().name
    plan = SUPPLEMENT[slug]
    arms = [a for a in ARMS if a in args.arms and plan["random" if a == "random" else "low"]]
    job = SupplementJob(args, model=model, tokenizer=tokenizer)
    job.root.mkdir(parents=True, exist_ok=True)
    invocation = {"arms": arms, "git": job.git, "runtime": C.R.runtime_versions(), "max_rows": args.max_rows,
                  "source_run_id": args.source_run_id, "plan": {k: list(v) for k, v in plan.items()}}
    with (job.root / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(invocation, ensure_ascii=False) + "\n")
    outcomes: dict[str, Any] = {}
    try:
        for arm in arms:
            print(f"==> supplement arm {arm} ({slug}, {args.phase})", flush=True)
            outcomes[arm] = RUNNERS[arm](job, plan["random" if arm == "random" else "low"])
            print(f"<== supplement arm {arm} complete: {outcomes[arm]}", flush=True)
    except C.MaxRowsReached:
        print("max rows reached; state saved for resume", flush=True)
        return 3
    C.atomic_write(job.root / f"job_summary-{'-'.join(arms)}.json",
                   {"identity": job.identity, "arms": arms, "outcomes": canonical(outcomes)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
