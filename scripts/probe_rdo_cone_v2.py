"""rdo-cone-v2: RDO/RCO trained with a validation split, lr schedule and a pre-registered convergence gate.

Protocol: docs/concept-cone-steering/rdo-cone-v2/proposal.md. The 402 construction companies are split into
train / validation; ``--phase tune`` trains and validates only (hyper-parameter grid), ``--phase full`` trains
with one seed and then evaluates on the held-out evaluation companies exactly as rdo-cone-v1 did.

Output: artifacts/<slug>/concept-cone-steering/runs/<run-id>/{training,directions}.json, train.log and, for
smoke/full, result.json. Only compact rows, learned unit directions and SHA-256 digests; no hidden states.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_rdo_cone as base  # noqa: E402

from llm_bias.core.inference.interventions import residual_interventions  # noqa: E402
from llm_bias.core.model import load_model  # noqa: E402
from llm_bias.core.steering import directions as D  # noqa: E402
from llm_bias.core.steering import prompts as P  # noqa: E402
from llm_bias.core.steering import protocol as R  # noqa: E402
from llm_bias.core.steering.evaluate import Evaluator, suffix_shift_transform, validate_row  # noqa: E402

SCHEMA = "concept-cone-steering-rdo-cone-v2"
REPO = Path(__file__).resolve().parents[1]
CODE_FILES = ("scripts/probe_rdo_cone_v2.py", *base.CODE_FILES)
VAL_SEED = 20260929
VAL_N = 80
CONVERGED_FRAC = 0.8
DOSES_LONG = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)   # rdo1, rco_centroid, dim
DOSES_BASIS = (1.0, 2.0, 4.0)                    # rco_b*
DOSES_SAMPLE = (1.0, 2.0)                        # rco_s*
DOSES_RAND = (1.0, 2.0, 4.0, 8.0)                # rand1


# ------------------------------------------------------------------------------------ pure helpers

def train_val_split(construction: Sequence[str], n_val: int, seed: int) -> tuple[list[str], list[str]]:
    """Validation companies drawn from the construction set; train keeps the construction order."""
    val = set(random.Random(seed).sample(sorted(construction), n_val))
    return [t for t in construction if t not in val], sorted(val)


def lr_factor(step: int, steps: int, warmup: int, floor: float) -> float:
    """Linear warmup over ``warmup`` steps, then cosine decay to ``floor``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def target_loss(steered: torch.Tensor, buy_id: int, sell_id: int, target: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``softplus(target − M)``; ``target=0`` is the rdo-cone-v1 loss."""
    margin = steered[buy_id] - steered[sell_id]
    return torch.nn.functional.softplus(target - margin), margin


def convergence(validation: dict[str, dict[str, float]], names: Sequence[str]) -> dict[str, Any]:
    """Pre-registered gate: every trained unit has validation frac(M > 0) >= 0.8 at the training dose."""
    fracs = {name: validation[name]["frac_positive"] for name in names}
    return {"rule": f"frac_positive >= {CONVERGED_FRAC} for {list(names)}", "frac_positive": fracs,
            "converged": all(v >= CONVERGED_FRAC for v in fracs.values())}


class Log:
    """Print to stdout and append to <run>/train.log."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, line: str) -> None:
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=".cache/models/<slug>")
    parser.add_argument("--phase", choices=("smoke", "tune", "full"), required=True)
    parser.add_argument("--run-id", required=True, help="rdo-cone-v2-<date>-<phase>-NN")
    parser.add_argument("--ranking-json", default=base.DEFAULT_RANKING, help="confirmation-v1 ranking arm result.json")
    parser.add_argument("--population-csv", default=R.POPULATION_CSV)
    parser.add_argument("--cone-dim", type=int, default=4)
    parser.add_argument("--cone-samples", type=int, default=8, help="evaluated cone samples")
    parser.add_argument("--cone-train-samples", type=int, default=2, help="cone samples per prompt per step")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--lr-floor", type=float, default=0.1, help="final lr as a fraction of --lr")
    parser.add_argument("--target-margin", type=float, default=0.0)
    parser.add_argument("--lambda-ret", type=float, default=1.0)
    parser.add_argument("--train-alpha", type=float, default=1.0, help="training dose in DIM-norm units")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--with-controls", action="store_true", help="also evaluate dim and rand1")
    parser.add_argument("--smoke-tickers", nargs="+", default=["ABNB", "AEP"])
    parser.add_argument("--smoke-train", type=int, default=8, help="smoke: number of training companies")
    parser.add_argument("--smoke-val", type=int, default=4, help="smoke: number of validation companies")
    parser.add_argument("--allow-dirty", action="store_true", help="smoke only: permit uncommitted code")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------ model context

class Context:
    """Model-bound state: prompts, train/validation/evaluation companies, evaluator, run log."""

    def __init__(self, args: argparse.Namespace, model: Any, tokenizer: Any) -> None:
        self.args = args
        self.smoke = args.phase == "smoke"
        self.model_dir = Path(args.model)
        self.slug = self.model_dir.resolve().name
        self.spec = R.model_spec(self.slug)
        if not args.run_id.startswith("rdo-cone-v2-") or f"-{args.phase}-" not in args.run_id:
            raise ValueError("run id must be rdo-cone-v2-<date>-<phase>-NN and match --phase")
        self.root = Path("artifacts") / self.slug / "concept-cone-steering" / "runs" / args.run_id
        self.git = R.git_provenance(REPO)
        if not self.smoke and self.git["code_dirty"]:
            raise ValueError(f"tune/full runs require committed code; dirty: {self.git['code_dirty_paths']}")
        if self.smoke and self.git["code_dirty"] and not args.allow_dirty:
            raise ValueError("smoke with uncommitted code requires --allow-dirty")
        self.companies, self.construction, self.evaluation = R.split_population(Path(args.population_csv), R.SPLIT_SEED)
        self.split_sha = R.split_sha256(self.construction, self.evaluation)
        stored = json.loads(Path(args.ranking_json).read_text(encoding="utf-8"))
        if not stored.get("complete") or stored["metadata"].get("split_sha256") != self.split_sha:
            raise ValueError("ranking is incomplete or was made on a different split")
        self.ranking = stored["rows"]
        self.ranking_sha = R.sha256_bytes(Path(args.ranking_json).read_bytes())
        if self.smoke and not set(args.smoke_tickers) <= set(self.evaluation):
            raise ValueError("smoke tickers must be held-out evaluation companies")
        train, val = train_val_split(self.construction, VAL_N, VAL_SEED)
        self.train_tickers = train[:args.smoke_train] if self.smoke else train
        self.val_tickers = val[:args.smoke_val] if self.smoke else val
        self.targets = list(args.smoke_tickers) if self.smoke else list(self.evaluation)
        self.log = Log(self.root / "train.log")
        if model is None:
            model, tokenizer, _ = load_model(str(self.model_dir), dtype="native" if self.spec.dtype == "native" else None)
        if model.n_layers != self.spec.n_layers:
            raise ValueError("model layer count differs from the registry")
        self.model, self.tokenizer = model, tokenizer
        self.hf = getattr(model, "_hf_model", model)
        self.hf.eval()
        for parameter in self.hf.parameters():
            parameter.requires_grad_(False)
        self.layer = self.spec.peak
        self.evaluator = Evaluator(model, tokenizer, max_new_tokens=R.MAX_NEW_TOKENS, printer=self.log)
        self.device = self.evaluator.device
        self._fp: dict[str, P.FormattedPrompt] = {}
        self._clean: dict[str, torch.Tensor] = {}
        self.metadata = {
            "schema": SCHEMA, "phase": args.phase, "run_id": args.run_id, **R.checkpoint_identity(self.model_dir),
            **R.template_provenance(tokenizer), "layer": self.layer, "split_seed": R.SPLIT_SEED,
            "split_sha256": self.split_sha, "ranking_sha256": self.ranking_sha,
            "val_seed": VAL_SEED, "train_tickers_n": len(self.train_tickers), "val_tickers": self.val_tickers,
            "hyper": {k: getattr(args, k) for k in (
                "cone_dim", "cone_samples", "cone_train_samples", "steps", "batch", "lr", "warmup", "lr_floor",
                "target_margin", "lambda_ret", "train_alpha", "seed", "val_every")},
            "code_sha256": R.file_sha256({name: REPO / name for name in CODE_FILES}),
            "primary_parse": "complete_object", "decision_prefix": R.DECISION_PREFIX,
        }

    def fp(self, ticker: str) -> P.FormattedPrompt:
        if ticker not in self._fp:
            prompt = P.render_decision_prompt(ticker, self.companies[ticker]["name"], "balanced")
            self._fp[ticker] = P.format_decision_prompt(
                self.tokenizer, prompt, suffix_tokens=self.spec.suffix_tokens, key=f"{ticker}/balanced")
        return self._fp[ticker]

    def dim(self) -> tuple[torch.Tensor, dict[str, Any], float]:
        """DIM anchor ``d[p]`` from construction Top/Bottom10 of the stored ranking, and median ||h||."""
        keep = set(self.construction)
        top, bottom = D.top_bottom([row for row in self.ranking if row["ticker"] in keep])
        states, _ = D.collect_suffix_states(self.model, [self.fp(t) for t in top + bottom], [self.layer])
        stacked = states[self.layer]
        d, stats = D.fit_dim_difference(stacked[:10], stacked[10:])
        return d.to(self.device), {**stats, "top": top, "bottom": bottom}, float(stacked.norm(dim=-1).median())

    def _forward(self, ids: torch.Tensor) -> torch.Tensor:
        try:
            out = self.hf(ids, logits_to_keep=1, use_cache=False)
        except TypeError:
            out = self.hf(ids)
        return out.logits[0, -1].float()

    def clean_logits(self, ticker: str) -> torch.Tensor:
        if ticker not in self._clean:
            ids = torch.tensor([list(self.fp(ticker).score_ids)], dtype=torch.long, device=self.device)
            with torch.no_grad():
                self._clean[ticker] = self._forward(ids)
        return self._clean[ticker]

    def steered_logits(self, fp: P.FormattedPrompt, shift: torch.Tensor) -> torch.Tensor:
        """Differentiable answer-prefix logits with ``shift`` ([K, d]) added at the steer suffix."""
        ids = torch.tensor([list(fp.score_ids)], dtype=torch.long, device=self.device)
        transform = suffix_shift_transform(fp.spans["steer_suffix"], shift)
        with residual_interventions(self.model, {self.layer: transform}):
            return self._forward(ids)


# ------------------------------------------------------------------------------------ training

def validate(ctx: Context, d: torch.Tensor, units: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Fixed-token margin and side-effect KL on the validation companies at the training dose."""
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, unit in units.items():
            margins, kls = [], []
            shift = ctx.args.train_alpha * D.equal_norm(d, unit)
            for ticker in ctx.val_tickers:
                fp = ctx.fp(ticker)
                steered = ctx.steered_logits(fp, shift)
                margins.append(float(steered[fp.buy_id] - steered[fp.sell_id]))
                kls.append(float(base.side_effect_kl(ctx.clean_logits(ticker), steered, (fp.buy_id, fp.sell_id))))
            out[name] = {"margin_median": median(margins), "margin_mean": fmean(margins),
                         "frac_positive": fmean(m > 0 for m in margins), "kl_mean": fmean(kls)}
    return out


def _val_units(basis: torch.Tensor, fixed: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    n = basis.shape[0]
    if n == 1:
        return {"rdo1": basis[0]}
    units = {f"rco_b{j + 1}": basis[j] for j in range(n)}
    units["rco_centroid"] = base.cone_unit(basis, torch.ones(n))
    units.update({f"rco_v{k + 1}": base.cone_unit(basis, s) for k, s in enumerate(fixed)})
    return units


def train_directions(ctx: Context, d: torch.Tensor, n: int, *, tag: str
                     ) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    """RDO (n=1) or RCO (n>1): Adam with warmup + cosine lr, Gram–Schmidt after every step, periodic validation."""
    args = ctx.args
    generator = torch.Generator().manual_seed(args.seed + (0 if n == 1 else 1))
    basis = torch.nn.Parameter(base.gram_schmidt(torch.randn(n, d.shape[-1], generator=generator)).to(ctx.device))
    optimizer = torch.optim.Adam([basis], lr=args.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, args.steps, args.warmup, args.lr_floor))
    rng = random.Random(args.seed + n)
    val_generator = torch.Generator().manual_seed(args.seed + 200)
    fixed = [base.sample_cone_coefficients(n, val_generator) for _ in range(2)] if n > 1 else []
    curve: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []

    def run_validation(step: int) -> None:
        with torch.no_grad():
            snapshot = basis.detach().float()
        record = validate(ctx, d, _val_units(snapshot, [s.to(snapshot) for s in fixed]))
        validation.append({"step": step, "units": record})
        ctx.log(f"[{tag}] val step {step} " + " ".join(
            f"{name}:M={v['margin_median']:+.3f}/pos={v['frac_positive']:.2f}/kl={v['kl_mean']:.4f}"
            for name, v in record.items()))

    run_validation(0)
    for step in range(args.steps):
        optimizer.zero_grad()
        chosen = rng.sample(ctx.train_tickers, min(args.batch, len(ctx.train_tickers)))
        acc: dict[str, list[float]] = {"loss_add": [], "margin": [], "kl": [], "basis_margin": []}
        for ticker in chosen:
            fp = ctx.fp(ticker)
            clean = ctx.clean_logits(ticker)
            if n == 1:
                units, sampled = [basis[0]], 1
            else:
                cone = [base.cone_unit(basis, base.sample_cone_coefficients(n, generator).to(ctx.device))
                        for _ in range(args.cone_train_samples)]
                units, sampled = cone + [basis[j] for j in range(n)], len(cone)
            weight = 1.0 / (len(units) * len(chosen))
            for index, unit in enumerate(units):
                steered = ctx.steered_logits(fp, args.train_alpha * D.equal_norm(d, unit))
                loss_add, margin = target_loss(steered, fp.buy_id, fp.sell_id, args.target_margin)
                kl = base.side_effect_kl(clean, steered, (fp.buy_id, fp.sell_id))
                (weight * (loss_add + args.lambda_ret * kl)).backward()
                if index < sampled:
                    acc["loss_add"].append(float(loss_add.detach()))
                    acc["margin"].append(float(margin.detach()))
                    acc["kl"].append(float(kl.detach()))
                else:
                    acc["basis_margin"].append(float(margin.detach()))
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        schedule.step()
        with torch.no_grad():
            basis.copy_(base.gram_schmidt(basis))
        record = {"step": step, "lr": lr, "loss_add": fmean(acc["loss_add"]), "margin": fmean(acc["margin"]),
                  "kl": fmean(acc["kl"]), "frac_margin_positive": fmean(m > 0 for m in acc["margin"])}
        if acc["basis_margin"]:
            record["basis_margin_mean"] = fmean(acc["basis_margin"])
            record["basis_margin_min"] = min(acc["basis_margin"])
        curve.append(record)
        if step % 10 == 0 or step == args.steps - 1:
            ctx.log(f"[{tag}] step {step} " + " ".join(f"{k}={v:.4g}" for k, v in record.items() if k != "step"))
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            run_validation(step + 1)
    return basis.detach().float().cpu(), curve, validation


def load_or_train(ctx: Context, d: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Learned unit directions and the convergence verdict; resumed from ``directions.json`` when present."""
    path = ctx.root / "directions.json"
    meta = {k: ctx.metadata[k] for k in ("schema", "run_id", "layer", "split_sha256", "ranking_sha256", "val_seed",
                                         "val_tickers", "hyper", "code_sha256", "train_tickers_n")}
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["metadata"] != meta:
            raise ValueError("directions.json metadata differs from this invocation; refusing to reuse")
        training = json.loads((ctx.root / "training.json").read_text(encoding="utf-8"))
        units = {name: torch.tensor(v["unit"], dtype=torch.float32) for name, v in stored["directions"].items()}
        return units, training["convergence"]
    args = ctx.args
    units: dict[str, torch.Tensor] = {}
    curves: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    rdo, curves["rdo1"], validation["rdo1"] = train_directions(ctx, d, 1, tag="rdo1")
    units["rdo1"] = rdo[0]
    basis, curves["rco"], validation["rco"] = train_directions(ctx, d, args.cone_dim, tag="rco")
    for j in range(args.cone_dim):
        units[f"rco_b{j + 1}"] = basis[j]
    units["rco_centroid"] = base.cone_unit(basis, torch.ones(args.cone_dim))
    generator = torch.Generator().manual_seed(args.seed + 100)
    for k in range(args.cone_samples):
        units[f"rco_s{k + 1}"] = base.cone_unit(basis, base.sample_cone_coefficients(args.cone_dim, generator))
    final = {**validation["rdo1"][-1]["units"], **validation["rco"][-1]["units"]}
    verdict = convergence(final, ["rdo1", *(f"rco_b{j + 1}" for j in range(args.cone_dim))])
    verdict["val_kl_mean"] = fmean(final[name]["kl_mean"] for name in verdict["frac_positive"])
    names = list(units)
    cos = {a: {b: float(units[a] @ units[b]) for b in names} for a in names}
    base.atomic_write(ctx.root / "training.json", {"metadata": meta, "curves": curves, "validation": validation,
                                                   "convergence": verdict, "cosine_matrix": cos})
    base.atomic_write(path, {"metadata": meta, "directions": {
        name: {"unit": [float(x) for x in u], "sha256": D.tensor_sha256(u)} for name, u in units.items()}})
    ctx.log(f"convergence: {json.dumps(verdict)}")
    return units, verdict


# ------------------------------------------------------------------------------------ evaluation

def evaluation_plan(units: dict[str, torch.Tensor], d: torch.Tensor, with_controls: bool
                    ) -> list[tuple[str, torch.Tensor, tuple[float, ...]]]:
    """(name, per-token base shift [K, d], doses) in fixed order."""
    plan: list[tuple[str, torch.Tensor, tuple[float, ...]]] = []
    if with_controls:
        random_unit, _ = D.shared_random_direction(d.shape[-1], 0)
        plan += [("dim", d, DOSES_LONG), ("rand1", D.equal_norm(d, random_unit), DOSES_RAND)]
    for name, unit in units.items():
        doses = (DOSES_LONG if name in ("rdo1", "rco_centroid")
                 else DOSES_BASIS if name.startswith("rco_b") else DOSES_SAMPLE)
        plan.append((name, D.equal_norm(d, unit), doses))
    return plan


def run_eval(ctx: Context, units: dict[str, torch.Tensor], verdict: dict[str, Any], d: torch.Tensor,
             dim_stats: dict[str, Any], median_h: float) -> None:
    path = ctx.root / "result.json"
    meta = {**ctx.metadata, "arm": "rdo_cone_v2_eval", "with_controls": ctx.args.with_controls}
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["metadata"] != meta:
            raise ValueError("result metadata differs from this invocation; refusing to resume")
    else:
        result = {"metadata": meta, "convergence": verdict, "dim": dim_stats, "baseline": {}, "dose_tables": {},
                  "rows": {}, "complete": False}
    for ticker, row in result["baseline"].items():
        validate_row(row, 0.0, f"baseline/{ticker}", baseline=True)
    for ticker in ctx.targets:
        if ticker not in result["baseline"]:
            with torch.no_grad():
                result["baseline"][ticker] = ctx.evaluator.row(ctx.fp(ticker), 0.0, baseline=True, label="alpha0")
            base.atomic_write(path, result)
    plan = evaluation_plan(units, d, ctx.args.with_controls)
    for name, shift, doses in plan:
        result["dose_tables"][name] = D.dose_table(d, shift, median_h)
        for alpha in doses:
            rows = result["rows"].setdefault(name, {}).setdefault(base.akey(alpha), {})
            for ticker, row in rows.items():
                validate_row(row, alpha, f"{name}/{base.akey(alpha)}/{ticker}")
            todo = [t for t in ctx.targets if t not in rows]
            for ticker in todo:
                with torch.no_grad():
                    rows[ticker] = ctx.evaluator.steered_row(ctx.fp(ticker), ctx.layer, shift, alpha, label=name)
            if todo:
                base.atomic_write(path, result)
    names = [name for name, _, _ in plan]
    result["summary"] = base.summarize(result["baseline"], result["rows"], names)
    result["complete"] = True
    base.atomic_write(path, result)
    print_table(ctx.log, result["summary"], names)


def print_table(log: Callable[[str], None], summary: dict[str, Any], names: Sequence[str]) -> None:
    log("direction          flip50  collapse  ITT flip / parse by dose")
    for name in names:
        entry = summary["per_direction"][name]
        cells = " ".join(f"{r['alpha']:g}:{r['on_flip_itt']:.2f}/{r['parse_rate']:.2f}"
                         if r["on_flip_itt"] is not None else f"{r['alpha']:g}:NA" for r in entry["per_alpha"])
        log(f"{name:18s} {entry['flip50_dose']!s:>6} {entry['collapse_dose']!s:>8}  {cells}")
    for alpha, cell in summary["cone_samples_best_of_n"].items():
        log(f"cone samples a={alpha}: mean={cell['mean_itt']:.2f} min={cell['min_itt']:.2f} "
            f"max={cell['max_itt']:.2f} best-of-{cell['samples']}={cell['best_of_n_itt']:.2f}")


def main(argv: Sequence[str] | None = None, *, model: Any = None, tokenizer: Any = None) -> int:
    args = parse_args(argv)
    if args.cone_dim < 2:
        raise ValueError("cone dimension must be at least 2")
    if args.steps < 1 or args.val_every < 1 or not 0 <= args.warmup < args.steps:
        raise ValueError("need steps >= 1, val_every >= 1 and 0 <= warmup < steps")
    ctx = Context(args, model, tokenizer)
    ctx.log(f"== {args.run_id} phase={args.phase} train={len(ctx.train_tickers)} val={len(ctx.val_tickers)} "
            f"hyper={json.dumps(ctx.metadata['hyper'])} commit={ctx.git['git_commit']}")
    d, dim_stats, median_h = ctx.dim()
    ctx.log(f"DIM anchor: top={dim_stats['top']} bottom={dim_stats['bottom']} median||d||={dim_stats['median']:.3f}")
    units, verdict = load_or_train(ctx, d)
    if args.phase != "tune":
        run_eval(ctx, units, verdict, d, dim_stats, median_h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
