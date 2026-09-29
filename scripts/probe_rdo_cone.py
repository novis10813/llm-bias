"""rdo-cone-v1: gradient-trained steering directions and cone (Wollschläger et al.) vs DIM, one model.

Protocol: docs/concept-cone-steering/rdo-cone-v1/proposal.md. Trains a single direction (RDO) and an
n-dimensional cone (RCO: projected gradient descent on the cone basis and on samples from the cone) on
the construction companies only, then evaluates them, DIM and a random control on the evaluation
companies with the confirmation-v1 prompt, dose convention and complete-object parsing.

Output: artifacts/<slug>/concept-cone-steering/runs/<run-id>/{training,directions,result}.json.
Only compact rows, learned unit directions and SHA-256 digests are written; no hidden states.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.model import load_model
from llm_bias.core.steering import directions as D
from llm_bias.core.steering import prompts as P
from llm_bias.core.steering import protocol as R
from llm_bias.core.steering import summary as S
from llm_bias.core.steering.evaluate import Evaluator, suffix_shift_transform, validate_row

SCHEMA = "concept-cone-steering-rdo-cone-v1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_RANKING = ("artifacts/qwen3.5-4b/concept-cone-steering/runs/"
                   "confirmation-v1-20260925-full-01/ranking/result.json")
CODE_FILES = ("scripts/probe_rdo_cone.py", "llm_bias/core/steering/protocol.py", "llm_bias/core/steering/prompts.py",
              "llm_bias/core/steering/directions.py", "llm_bias/core/steering/evaluate.py",
              "llm_bias/core/steering/summary.py", "llm_bias/core/decision_parsing.py",
              "llm_bias/core/decision_readout.py", "llm_bias/core/inference/generation.py",
              "llm_bias/core/inference/interventions.py")
DOSES_ONE = (1.0, 2.0, 4.0, 8.0)          # dim, rand1, rdo1
DOSES_CONE = (1.0, 2.0, 4.0)              # rco_* (basis, centroid, samples)
COLLAPSE_PARSE_RATE = 0.9
FLIP_HALF = 0.5


# ------------------------------------------------------------------------------------ pure math

def gram_schmidt(rows: torch.Tensor) -> torch.Tensor:
    """Orthonormalise the rows in order (fp32); raises on a degenerate row."""
    out: list[torch.Tensor] = []
    for row in rows.float():
        v = row.clone()
        for u in out:
            v = v - (v @ u) * u
        norm = v.norm()
        if not torch.isfinite(norm) or norm < 1e-8:
            raise ValueError("degenerate basis row in Gram-Schmidt")
        out.append(v / norm)
    return torch.stack(out)


def sample_cone_coefficients(n: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform unit vector on the positive orthant of the n-sphere (|Gaussian| normalised)."""
    s = torch.randn(n, generator=generator).abs()
    return s / s.norm()


def cone_unit(basis: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Unit direction of a non-negative combination of an orthonormal basis."""
    if bool((coefficients < 0).any()):
        raise ValueError("cone coefficients must be non-negative")
    r = coefficients.to(basis) @ basis
    return r / r.norm()


def side_effect_kl(clean: torch.Tensor, steered: torch.Tensor, exclude: Sequence[int]) -> torch.Tensor:
    """KL(clean || steered) of the next-token distribution after removing the decision tokens."""
    keep = torch.ones(clean.shape[-1], dtype=torch.bool, device=clean.device)
    keep[list(exclude)] = False
    log_clean = F.log_softmax(clean[keep], dim=-1)
    log_steered = F.log_softmax(steered[keep], dim=-1)
    return (log_clean.exp() * (log_clean - log_steered)).sum()


def addition_loss(steered: torch.Tensor, buy_id: int, sell_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``softplus(-M)`` (two-way cross-entropy towards buy) and the margin ``M``."""
    margin = steered[buy_id] - steered[sell_id]
    return F.softplus(-margin), margin


def first_dose(stats_by_alpha: dict[float, dict[str, Any]], key: str, predicate) -> float | None:
    for alpha in sorted(stats_by_alpha):
        value = stats_by_alpha[alpha][key]
        if value is not None and predicate(value):
            return alpha
    return None


# ------------------------------------------------------------------------------------ IO helpers

def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def akey(alpha: float) -> str:
    return f"{alpha:+g}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=".cache/models/<slug>")
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True, help="rdo-cone-v1-<date>-<phase>-NN")
    parser.add_argument("--ranking-json", default=DEFAULT_RANKING, help="confirmation-v1 ranking arm result.json")
    parser.add_argument("--population-csv", default=R.POPULATION_CSV)
    parser.add_argument("--cone-dim", type=int, default=4)
    parser.add_argument("--cone-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ret", type=float, default=1.0)
    parser.add_argument("--train-alpha", type=float, default=1.0, help="training dose in DIM-norm units")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--smoke-tickers", nargs="+", default=["ABNB", "AEP"])
    parser.add_argument("--smoke-train", type=int, default=8, help="smoke: number of construction companies")
    parser.add_argument("--allow-dirty", action="store_true", help="smoke only: permit uncommitted code")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------ training

class Context:
    """Model-bound state: prompts, DIM anchor, evaluator."""

    def __init__(self, args: argparse.Namespace, model: Any, tokenizer: Any) -> None:
        self.args = args
        self.smoke = args.phase == "smoke"
        self.model_dir = Path(args.model)
        self.slug = self.model_dir.resolve().name
        self.spec = R.model_spec(self.slug)
        if not args.run_id.startswith("rdo-cone-v1-") or f"-{args.phase}-" not in args.run_id:
            raise ValueError("run id must be rdo-cone-v1-<date>-<phase>-NN and match --phase")
        self.root = Path("artifacts") / self.slug / "concept-cone-steering" / "runs" / args.run_id
        self.git = R.git_provenance(REPO)
        if not self.smoke and self.git["code_dirty"]:
            raise ValueError(f"full runs require committed code; dirty: {self.git['code_dirty_paths']}")
        if self.smoke and self.git["code_dirty"] and not args.allow_dirty:
            raise ValueError("smoke with uncommitted code requires --allow-dirty")
        self.companies, self.construction, self.evaluation = R.split_population(Path(args.population_csv), R.SPLIT_SEED)
        self.split_sha = R.split_sha256(self.construction, self.evaluation)
        self.ranking_path = Path(args.ranking_json)
        stored = json.loads(self.ranking_path.read_text(encoding="utf-8"))
        if not stored.get("complete") or stored["metadata"].get("split_sha256") != self.split_sha:
            raise ValueError("ranking is incomplete or was made on a different split")
        self.ranking = stored["rows"]
        self.ranking_sha = R.sha256_bytes(self.ranking_path.read_bytes())
        if self.smoke and not set(args.smoke_tickers) <= set(self.evaluation):
            raise ValueError("smoke tickers must be held-out evaluation companies")
        self.targets = list(args.smoke_tickers) if self.smoke else list(self.evaluation)
        self.train_tickers = list(self.construction)[:args.smoke_train] if self.smoke else list(self.construction)
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
        self.evaluator = Evaluator(model, tokenizer, max_new_tokens=R.MAX_NEW_TOKENS,
                                   printer=lambda line: print(line, flush=True))
        self.device = self.evaluator.device
        self._fp: dict[str, P.FormattedPrompt] = {}
        self.metadata = {
            "schema": SCHEMA, "phase": args.phase, "run_id": args.run_id, **R.checkpoint_identity(self.model_dir),
            **R.template_provenance(tokenizer), "layer": self.layer, "split_seed": R.SPLIT_SEED,
            "split_sha256": self.split_sha, "ranking_sha256": self.ranking_sha, "targets": self.targets,
            "train_tickers_n": len(self.train_tickers),
            "hyper": {k: getattr(args, k) for k in ("cone_dim", "cone_samples", "steps", "batch", "lr",
                                                     "lambda_ret", "train_alpha", "seed")},
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

    def clean_logits(self, fp: P.FormattedPrompt) -> torch.Tensor:
        ids = torch.tensor([list(fp.score_ids)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self._forward(ids)

    def steered_logits(self, fp: P.FormattedPrompt, shift: torch.Tensor) -> torch.Tensor:
        """Differentiable answer-prefix logits with ``shift`` ([K, d]) added at the steer suffix."""
        ids = torch.tensor([list(fp.score_ids)], dtype=torch.long, device=self.device)
        transform = suffix_shift_transform(fp.spans["steer_suffix"], shift)
        with residual_interventions(self.model, {self.layer: transform}):
            return self._forward(ids)


def train_directions(ctx: Context, d: torch.Tensor, n: int, *, tag: str) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """RDO (n=1) or RCO (n>1): Adam on the basis, Gram-Schmidt after every step."""
    args = ctx.args
    generator = torch.Generator().manual_seed(args.seed + (0 if n == 1 else 1))
    d_model = d.shape[-1]
    basis = torch.nn.Parameter(gram_schmidt(torch.randn(n, d_model, generator=generator)).to(ctx.device))
    optimizer = torch.optim.Adam([basis], lr=args.lr)
    rng = random.Random(args.seed + n)
    curve: list[dict[str, Any]] = []
    for step in range(args.steps):
        optimizer.zero_grad()
        chosen = rng.sample(ctx.train_tickers, min(args.batch, len(ctx.train_tickers)))
        acc = {"loss_add": [], "margin": [], "kl": [], "basis_margin": []}
        for ticker in chosen:
            fp = ctx.fp(ticker)
            clean = ctx.clean_logits(fp)
            if n == 1:
                units, weights = [basis[0]], [1.0]
            else:
                s = sample_cone_coefficients(n, generator).to(ctx.device)
                units = [cone_unit(basis, s)] + [basis[j] for j in range(n)]
                weights = [1.0] + [1.0 / n] * n
            for index, (unit, weight) in enumerate(zip(units, weights)):
                shift = args.train_alpha * D.equal_norm(d, unit)
                steered = ctx.steered_logits(fp, shift)
                loss_add, margin = addition_loss(steered, fp.buy_id, fp.sell_id)
                kl = side_effect_kl(clean, steered, (fp.buy_id, fp.sell_id))
                ((weight / len(chosen)) * (loss_add + args.lambda_ret * kl)).backward()
                if index == 0:
                    acc["loss_add"].append(float(loss_add))
                    acc["margin"].append(float(margin))
                    acc["kl"].append(float(kl))
                else:
                    acc["basis_margin"].append(float(margin))
        optimizer.step()
        with torch.no_grad():
            basis.copy_(gram_schmidt(basis))
        record = {"step": step, "loss_add": fmean(acc["loss_add"]), "margin": fmean(acc["margin"]),
                  "kl": fmean(acc["kl"]), "frac_margin_positive": fmean(m > 0 for m in acc["margin"])}
        if acc["basis_margin"]:
            record["basis_margin_mean"] = fmean(acc["basis_margin"])
            record["basis_margin_min"] = min(acc["basis_margin"])
        curve.append(record)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"[{tag}] step {step} " + " ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "step"),
                  flush=True)
    return basis.detach().float().cpu(), curve


def load_or_train(ctx: Context, d: torch.Tensor) -> dict[str, torch.Tensor]:
    """Learned unit directions ``{name: [d_model]}``; resumed from ``directions.json`` when present."""
    path = ctx.root / "directions.json"
    meta = {k: ctx.metadata[k] for k in ("schema", "run_id", "layer", "split_sha256", "ranking_sha256", "hyper",
                                         "code_sha256", "train_tickers_n")}
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["metadata"] != meta:
            raise ValueError("directions.json metadata differs from this invocation; refusing to reuse")
        return {name: torch.tensor(v["unit"], dtype=torch.float32) for name, v in stored["directions"].items()}
    args = ctx.args
    units: dict[str, torch.Tensor] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    rdo, curves["rdo1"] = train_directions(ctx, d, 1, tag="rdo1")
    units["rdo1"] = rdo[0]
    basis, curves["rco"] = train_directions(ctx, d, args.cone_dim, tag="rco")
    for j in range(args.cone_dim):
        units[f"rco_b{j + 1}"] = basis[j]
    units["rco_centroid"] = cone_unit(basis, torch.ones(args.cone_dim))
    generator = torch.Generator().manual_seed(args.seed + 100)
    for k in range(args.cone_samples):
        units[f"rco_s{k + 1}"] = cone_unit(basis, sample_cone_coefficients(args.cone_dim, generator))
    names = list(units)
    cos = {a: {b: float(units[a] @ units[b]) for b in names} for a in names}
    atomic_write(ctx.root / "training.json", {"metadata": meta, "curves": curves, "cosine_matrix": cos})
    atomic_write(path, {"metadata": meta, "directions": {
        name: {"unit": [float(x) for x in u], "sha256": D.tensor_sha256(u)} for name, u in units.items()}})
    return units


# ------------------------------------------------------------------------------------ evaluation

def evaluation_plan(units: dict[str, torch.Tensor], d: torch.Tensor) -> list[tuple[str, torch.Tensor, tuple[float, ...]]]:
    """(name, per-token base shift [K, d], doses) in fixed order."""
    random_unit, _ = D.shared_random_direction(d.shape[-1], 0)
    plan = [("dim", d, DOSES_ONE), ("rand1", D.equal_norm(d, random_unit), DOSES_ONE)]
    plan += [(name, D.equal_norm(d, u), DOSES_ONE if name == "rdo1" else DOSES_CONE) for name, u in units.items()]
    return plan


def summarize(baseline: dict[str, Any], rows: dict[str, dict[str, dict[str, Any]]], names: Sequence[str]) -> dict[str, Any]:
    per_direction: dict[str, Any] = {}
    for name in names:
        by_alpha = {float(a): flip_row for a, flip_row in rows.get(name, {}).items()}
        stats = {alpha: S.flip_stats(baseline, by_alpha[alpha], alpha) for alpha in sorted(by_alpha) if by_alpha[alpha]}
        per_direction[name] = {
            "per_alpha": [stats[a] for a in sorted(stats)],
            "flip50_dose": first_dose(stats, "on_flip_itt", lambda v: v >= FLIP_HALF),
            "collapse_dose": first_dose(stats, "parse_rate", lambda v: v < COLLAPSE_PARSE_RATE),
        }
    samples = [n for n in names if n.startswith("rco_s")]
    best_of: dict[str, Any] = {}
    on_class = [t for t, row in baseline.items() if row["decision"] == "sell"]
    for alpha in DOSES_CONE:
        stat = [per_direction[n]["per_alpha"] for n in samples]
        rates = [next((r["on_flip_itt"] for r in per if r["alpha"] == alpha), None) for per in stat]
        rates = [r for r in rates if r is not None]
        union = [t for t in on_class if any(
            rows.get(n, {}).get(akey(alpha), {}).get(t, {}).get("decision") == "buy" for n in samples)]
        if rates:
            best_of[akey(alpha)] = {"samples": len(rates), "on_class_n": len(on_class), "best_of_n_flips": len(union),
                                    "best_of_n_itt": len(union) / len(on_class) if on_class else None,
                                    "mean_itt": fmean(rates), "min_itt": min(rates), "median_itt": median(rates),
                                    "max_itt": max(rates)}
    return {"per_direction": per_direction, "cone_samples_best_of_n": best_of}


def run_eval(ctx: Context, units: dict[str, torch.Tensor], d: torch.Tensor, dim_stats: dict[str, Any],
             median_h: float) -> None:
    path = ctx.root / "result.json"
    meta = {**ctx.metadata, "arm": "rdo_cone_eval"}
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["metadata"] != meta:
            raise ValueError("result metadata differs from this invocation; refusing to resume")
    else:
        result = {"metadata": meta, "dim": dim_stats, "baseline": {}, "dose_tables": {}, "rows": {}, "complete": False}
    for ticker, row in result["baseline"].items():
        validate_row(row, 0.0, f"baseline/{ticker}", baseline=True)
    for ticker in ctx.targets:
        if ticker not in result["baseline"]:
            with torch.no_grad():
                result["baseline"][ticker] = ctx.evaluator.row(ctx.fp(ticker), 0.0, baseline=True, label="alpha0")
            atomic_write(path, result)
    plan = evaluation_plan(units, d)
    for name, base, doses in plan:
        result["dose_tables"][name] = D.dose_table(d, base, median_h)
        for alpha in doses:
            unit_rows = result["rows"].setdefault(name, {}).setdefault(akey(alpha), {})
            for ticker, row in unit_rows.items():
                validate_row(row, alpha, f"{name}/{akey(alpha)}/{ticker}")
            todo = [t for t in ctx.targets if t not in unit_rows]
            for ticker in todo:
                with torch.no_grad():
                    unit_rows[ticker] = ctx.evaluator.steered_row(ctx.fp(ticker), ctx.layer, base, alpha, label=name)
            if todo:
                atomic_write(path, result)
    names = [name for name, _, _ in plan]
    result["summary"] = summarize(result["baseline"], result["rows"], names)
    result["complete"] = True
    atomic_write(path, result)
    print_table(result["summary"], names)


def print_table(summary: dict[str, Any], names: Sequence[str]) -> None:
    print("direction          flip50  collapse  ITT flip / parse by dose", flush=True)
    for name in names:
        entry = summary["per_direction"][name]
        cells = " ".join(f"{r['alpha']:g}:{r['on_flip_itt']:.2f}/{r['parse_rate']:.2f}"
                         if r["on_flip_itt"] is not None else f"{r['alpha']:g}:NA" for r in entry["per_alpha"])
        print(f"{name:18s} {entry['flip50_dose']!s:>6} {entry['collapse_dose']!s:>8}  {cells}", flush=True)
    for alpha, cell in summary["cone_samples_best_of_n"].items():
        print(f"cone samples a={alpha}: mean={cell['mean_itt']:.2f} min={cell['min_itt']:.2f} max={cell['max_itt']:.2f} "
              f"best-of-{cell['samples']}={cell['best_of_n_itt']:.2f}", flush=True)


def main(argv: Sequence[str] | None = None, *, model: Any = None, tokenizer: Any = None) -> int:
    args = parse_args(argv)
    if args.cone_dim < 2:
        raise ValueError("cone dimension must be at least 2")
    ctx = Context(args, model, tokenizer)
    d, dim_stats, median_h = ctx.dim()
    print(f"DIM anchor: top={dim_stats['top']} bottom={dim_stats['bottom']} median||d||={dim_stats['median']:.3f}",
          flush=True)
    units = load_or_train(ctx, d)
    run_eval(ctx, units, d, dim_stats, median_h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
