"""Confirmation-v1 multi-arm steering runner: one model load per job, arms run in a fixed order.

Protocol: docs/concept-cone-steering/confirmation-v1/proposal.md (and the operator-comparison-v2,
evidence-sensitivity-v1, c2-v3-steering-prompt, generalization-v1 proposals it references).

Every arm writes artifacts/<slug>/concept-cone-steering/runs/<run-id>/<arm>/result.json, saved
atomically after each (operator, prompt) unit and resumed fail-closed: metadata (including code and
calibration SHA-256) must match exactly and every stored row must re-derive from its generated text.
Only compact rows, dose statistics and SHA-256 digests are written; no hidden states.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable, Mapping, Sequence

import torch

from llm_bias.balanced_evidence_gap.intervention import make_span_transform, nearest_position_mapping
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.model import load_model
from llm_bias.core.steering import directions as D
from llm_bias.core.steering import prompts as P
from llm_bias.core.steering import protocol as R
from llm_bias.core.steering import summary as S
from llm_bias.core.steering.evaluate import Evaluator, fp32_margin, suffix_shift_transform, validate_row

SCHEMA = "concept-cone-steering-confirmation-v1"
REPO = Path(__file__).resolve().parents[1]
ARM_ORDER = ("gates", "ranking", "alpha0", "cal", "dim", "dim_layers", "random", "jitter", "ops", "shuffle",
             "evidence", "anon", "loso", "loso_construction", "split_seed", "c2v3", "c2v3_gen")
TIERS: dict[str, dict[str, tuple[str, ...]]] = {
    "qwen3.5-4b": {
        "tier1": ("gates", "ranking", "alpha0", "cal", "dim", "dim_layers", "random", "jitter", "ops", "evidence",
                  "anon", "c2v3", "c2v3_gen", "loso", "loso_construction", "split_seed"),
        "tier2": ("shuffle",)},
    "gemma4-12b-it": {
        "tier1": ("gates", "ranking", "alpha0", "cal", "dim", "random", "jitter", "ops", "evidence", "anon",
                  "c2v3", "c2v3_gen", "loso", "split_seed"),
        "tier2": ("shuffle", "loso_construction")},
    "glm4-9b-0414": {
        "tier1": ("gates", "ranking", "alpha0", "cal", "dim", "random", "jitter", "evidence", "anon", "c2v3",
                  "c2v3_gen", "loso", "loso_construction", "split_seed"),
        "tier2": ("ops",)},
    "gpt-oss-20b": {
        "tier1": ("gates", "ranking", "alpha0", "cal", "dim", "random", "jitter", "evidence", "anon", "c2v3",
                  "c2v3_gen", "loso", "split_seed"),
        "tier2": ("ops", "loso_construction")},
}
SPLIT_SEEDS = (20260923, 20260924, 20260925)
CAL_SEED, CAL_SIZE, CAL_SCAN = 20260926, 24, 120
RANDOM_SEEDS, JITTER_SEEDS, ORTH_SEEDS, SHUFFLE_SEEDS = (0, 1, 2, 3, 4), (10, 11, 12), (20, 21, 22), (100, 101, 102)
JITTER_FRACTION = 0.05
PLACEBO_SEED = 20260927
SMOKE_GRID = (-8.0, -2.0, 2.0, 8.0)
SMOKE_CAL_COMPANIES = 2
DIAG_COMPANIES = 10
R6_PAIRS = 20
R7_SPANS = ("steer_suffix", "entity")
QWEN_R2_LAYERS = (0, 14, 15, 17, 18)
QWEN_DIAL = (15, 8490)
CODE_FILES = (
    "scripts/probe_steering_confirmation.py", "llm_bias/core/steering/protocol.py",
    "llm_bias/core/steering/prompts.py", "llm_bias/core/steering/directions.py",
    "llm_bias/core/steering/evaluate.py", "llm_bias/core/steering/summary.py",
    "llm_bias/core/decision_parsing.py", "llm_bias/core/decision_readout.py",
    "llm_bias/core/inference/generation.py", "llm_bias/core/inference/interventions.py",
    "llm_bias/core/inference/forward.py", "llm_bias/balanced_evidence_gap/intervention.py",
)
V2_SMOKE = "dim-crossmodel-layer-sweep-v2-20260925-smoke-01"
V2_FULL = "dim-crossmodel-layer-sweep-v2-20260925-full-01"
QWEN_V1 = "dim-layer-sweep-v1-20260925-paper-01"


class MaxRowsReached(RuntimeError):
    """Deliberate interruption used to test resume equivalence."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=".cache/models/<slug>")
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True, help="confirmation-v1-<date>-<phase>-NN")
    parser.add_argument("--arms", nargs="+", default=["tier1"], help="arm names, 'tier1', 'tier2' or 'all'")
    parser.add_argument("--smoke-tickers", nargs="+", default=["ABNB", "AEP"])
    parser.add_argument("--max-rows", type=int, default=None, help="stop after N generated rows (resume test)")
    parser.add_argument("--population-csv", default=R.POPULATION_CSV)
    parser.add_argument("--allow-dirty", action="store_true", help="smoke only: permit uncommitted code")
    return parser.parse_args(argv)


def resolve_arms(slug: str, requested: Sequence[str]) -> list[str]:
    tiers = TIERS[slug]
    chosen: set[str] = set()
    for name in requested:
        if name == "all":
            chosen |= set(tiers["tier1"]) | set(tiers["tier2"])
        elif name in tiers:
            chosen |= set(tiers[name])
        elif name in ARM_ORDER:
            chosen.add(name)
        else:
            raise ValueError(f"unknown arm {name!r}")
    return [arm for arm in ARM_ORDER if arm in chosen]


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def akey(alpha: float) -> str:
    return f"{alpha:+g}"


class Job:
    """Model-bound state shared by all arms of one invocation."""

    def __init__(self, args: argparse.Namespace, *, model: Any = None, tokenizer: Any = None) -> None:
        self.args = args
        self.model_dir = Path(args.model)
        self.slug = self.model_dir.resolve().name
        self.spec = R.model_spec(self.slug)
        self.smoke = args.phase == "smoke"
        runs = Path("artifacts") / self.slug / "concept-cone-steering" / "runs"
        if not args.run_id.startswith(f"confirmation-v1-") or f"-{args.phase}-" not in args.run_id:
            raise ValueError("run id must be confirmation-v1-<date>-<phase>-NN and match --phase")
        self.root = runs / args.run_id
        self.git = R.git_provenance(REPO)
        if not self.smoke and self.git["code_dirty"]:
            raise ValueError(f"full runs require committed code; dirty: {self.git['code_dirty_paths']}")
        if self.smoke and self.git["code_dirty"] and not args.allow_dirty:
            raise ValueError("smoke with uncommitted code requires --allow-dirty")
        self.code_sha = R.file_sha256({name: REPO / name for name in CODE_FILES})
        self.companies, self.construction, self.evaluation = R.split_population(
            Path(args.population_csv), R.SPLIT_SEED)
        if self.smoke and (not set(args.smoke_tickers) <= set(self.evaluation)):
            raise ValueError("smoke tickers must be held-out evaluation companies")
        self.targets = list(args.smoke_tickers) if self.smoke else list(self.evaluation)
        if model is None:
            model, tokenizer, _ = load_model(str(self.model_dir), dtype="native" if self.spec.dtype == "native" else None)
        if model.n_layers != self.spec.n_layers:
            raise ValueError("model layer count differs from the registry")
        self.model, self.tokenizer = model, tokenizer
        self.rows_left = args.max_rows
        self.evaluator = Evaluator(model, tokenizer, max_new_tokens=R.MAX_NEW_TOKENS,
                                   printer=lambda line: print(line, flush=True))
        self._fp: dict[tuple[str, str], P.FormattedPrompt] = {}
        self._states: dict[tuple[int, str], torch.Tensor] = {}
        self._alpha0: dict[str, Any] | None = None
        self._ranking: list[dict[str, Any]] | None = None
        self._dim: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
        self._median_h: dict[int, float] = {}
        self._cal: dict[str, Any] | None = None
        k, _ = P.common_instruction_suffix(tokenizer, [
            P.render_decision_prompt(t, c["name"], cond) for cond in P.CONDITIONS
            for t, c in self.companies.items()] + [
            P.render_decision_prompt(t, n, cond) for cond in P.CONDITIONS for t, n in P.ANON_IDENTITIES])
        if k != self.spec.suffix_tokens:
            raise ValueError(f"common instruction suffix K={k} differs from registry {self.spec.suffix_tokens}")
        self.base_metadata = {
            "schema": SCHEMA, "phase": args.phase, "run_id": args.run_id, **R.checkpoint_identity(self.model_dir),
            "model_dtype": self.spec.dtype, "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
            **R.template_provenance(tokenizer), "split_seed": R.SPLIT_SEED,
            "split_sha256": R.split_sha256(self.construction, self.evaluation),
            "population_sha256": R.sha256_bytes(Path(args.population_csv).read_bytes()),
            "prompt_family_sha256": {c: P.prompt_family_sha256(self.companies, c) for c in P.CONDITIONS},
            "anon_identities": [list(x) for x in P.ANON_IDENTITIES], "suffix_tokens": k,
            "primary_layer": self.spec.peak, "c2_source": {k2: v for k2, v in R.c2_427_source(self.slug).items()
                                                           if k2 != "instruction_T"},
            "decision_prefix": R.DECISION_PREFIX, "max_new_tokens": R.MAX_NEW_TOKENS,
            # code identity is the per-file SHA; the commit is logged per invocation (docs-only commits
            # between tier1 and tier2 must not invalidate shared ranking / alpha-0 / CAL results)
            "primary_parse": "complete_object", "code_sha256": self.code_sha, "targets": self.targets,
        }

    # ---------------------------------------------------------------- prompts and states

    def fp(self, key: str, condition: str) -> P.FormattedPrompt:
        """``key`` is a ticker or ``anon:<index>``."""
        if (key, condition) not in self._fp:
            if key.startswith("anon:"):
                ticker, name = P.ANON_IDENTITIES[int(key.split(":")[1])]
            else:
                ticker, name = key, self.companies[key]["name"]
            prompt = P.render_decision_prompt(ticker, name, condition)
            self._fp[(key, condition)] = P.format_decision_prompt(
                self.tokenizer, prompt, suffix_tokens=self.spec.suffix_tokens, key=f"{key}/{condition}")
        return self._fp[(key, condition)]

    def states(self, layer: int, tickers: Sequence[str]) -> torch.Tensor:
        """Balanced clean post-block steer-suffix states ``[N, K, d]`` (cached on CPU per ticker)."""
        missing = [t for t in tickers if (layer, t) not in self._states]
        if missing:
            stacked, _ = D.collect_suffix_states(self.model, [self.fp(t, "balanced") for t in missing], [layer])
            for t, value in zip(missing, stacked[layer]):
                self._states[(layer, t)] = value.cpu()
        return torch.stack([self._states[(layer, t)] for t in tickers]).to(self.evaluator.device)

    def dim(self, layer: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        layer = self.spec.peak if layer is None else layer
        if layer not in self._dim:
            top, bottom = self.top_bottom()
            d, stats = D.fit_dim_difference(self.states(layer, top), self.states(layer, bottom))
            self._dim[layer] = (d, stats)
            everyone = self.states(layer, top + bottom)
            self._median_h[layer] = float(everyone.norm(dim=-1).median())
        return self._dim[layer]

    def median_h(self, layer: int | None = None) -> float:
        layer = self.spec.peak if layer is None else layer
        self.dim(layer)
        return self._median_h[layer]

    # ---------------------------------------------------------------- shared results

    def path(self, arm: str, name: str = "result.json") -> Path:
        return self.root / arm / name

    def count_row(self) -> None:
        if self.rows_left is not None:
            self.rows_left -= 1

    def check_budget(self) -> None:
        if self.rows_left is not None and self.rows_left <= 0:
            raise MaxRowsReached("max rows reached")

    def ranking(self) -> list[dict[str, Any]]:
        """Clean balanced fixed-prefix margins of all 503 companies, ascending (margin, ticker)."""
        if self._ranking is None:
            path = self.path("ranking")
            meta = {**self.base_metadata, "arm": "ranking"}
            if path.exists():
                stored = json.loads(path.read_text(encoding="utf-8"))
                if stored["metadata"] != meta or not stored.get("complete"):
                    raise ValueError("ranking arm metadata changed or incomplete; refusing to reuse")
                self._ranking = stored["rows"]
            else:
                rows = []
                for i, ticker in enumerate(sorted(self.companies)):
                    rows.append({"ticker": ticker, "margin": self.evaluator.fixed_prefix_margin(self.fp(ticker, "balanced"))})
                    if (i + 1) % 100 == 0:
                        print(f"ranked {i + 1}/503", flush=True)
                rows.sort(key=lambda r: (r["margin"], r["ticker"]))
                self._ranking = rows
                atomic_write(path, {"metadata": meta, "rows": rows, "complete": True})
        return self._ranking

    def construction_ranking(self, construction: Sequence[str] | None = None) -> list[dict[str, Any]]:
        keep = set(self.construction if construction is None else construction)
        return [row for row in self.ranking() if row["ticker"] in keep]

    def top_bottom(self) -> tuple[list[str], list[str]]:
        return D.top_bottom(self.construction_ranking())

    def alpha0_store(self) -> dict[str, Any]:
        if self._alpha0 is None:
            path = self.path("alpha0")
            meta = {**self.base_metadata, "arm": "alpha0"}
            if path.exists():
                stored = json.loads(path.read_text(encoding="utf-8"))
                if stored["metadata"] != meta:
                    raise ValueError("alpha0 store metadata changed; refusing to resume")
                for condition, rows in stored["rows"].items():
                    for key, row in rows.items():
                        validate_row(row, 0.0, f"alpha0/{condition}/{key}", baseline=True)
                self._alpha0 = stored
            else:
                self._alpha0 = {"metadata": meta, "rows": {}}
        return self._alpha0

    def alpha0(self, key: str, condition: str = "balanced") -> dict[str, Any]:
        """Shared alpha-0 row of one prompt/condition, generated once for every arm."""
        store = self.alpha0_store()
        rows = store["rows"].setdefault(condition, {})
        if key not in rows:
            self.check_budget()
            rows[key] = self.evaluator.row(self.fp(key, condition), 0.0, baseline=True, label="alpha0")
            self.count_row()
            atomic_write(self.path("alpha0"), store)
        return rows[key]

    def calibration(self) -> dict[str, Any]:
        if self._cal is None:
            path = self.path("cal", "calibration.json")
            if not path.exists():
                raise ValueError("calibration missing: run the cal arm first")
            cal = json.loads(path.read_text(encoding="utf-8"))
            if cal["metadata"]["code_sha256"] != self.code_sha:
                raise ValueError("calibration code SHA differs from current code (auto-abort)")
            if not cal.get("complete"):
                raise ValueError("calibration incomplete")
            self._cal = cal
        return self._cal

    def grid(self, kind: str) -> list[float]:
        """Nonzero alphas of the CAL-derived full/reduced/random grid (fixed test grid in smoke)."""
        if self.smoke:
            return list(SMOKE_GRID)
        rule = self.calibration()["rule"]
        if rule["abort"]:
            raise ValueError(f"CAL auto-abort: {rule['abort']}")
        values = {"full": rule["full_grid"], "reduced": rule["reduced_grid"], "random": rule["random_grid"]}[kind]
        return [a for a in values if a != 0]

    def alpha_hi(self) -> float:
        return 8.0 if self.smoke else float(self.calibration()["rule"]["alpha_hi"])

    def alpha_50(self) -> float:
        return 2.0 if self.smoke else float(self.calibration()["rule"]["alpha_50"])


class GridArm:
    """Rows of named operators over prompts and a nonzero-alpha grid, resumable per (operator, prompt)."""

    def __init__(self, job: Job, arm: str, extra_meta: Mapping[str, Any] | None = None) -> None:
        self.job, self.arm = job, arm
        self.path = job.path(arm)
        self.meta = {**job.base_metadata, "arm": arm, **(extra_meta or {})}
        if self.path.exists():
            stored = json.loads(self.path.read_text(encoding="utf-8"))
            if stored["metadata"] != self.meta:
                raise ValueError(f"{arm}: metadata differs from the stored run; refusing to resume")
            self.result = stored
            self.validate()
        else:
            self.result = {"metadata": self.meta, "operators": {}, "rows": {}, "complete": False}

    @classmethod
    def load(cls, job: Job, arm: str) -> "GridArm":
        """Read-only view of another arm's stored result (base metadata must match this job)."""
        path = job.path(arm)
        if not path.exists():
            raise ValueError(f"{arm} has not been run")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if any(stored["metadata"].get(k) != v for k, v in job.base_metadata.items()):
            raise ValueError(f"{arm}: stored base metadata differs from this job")
        view = cls.__new__(cls)
        view.job, view.arm, view.path, view.meta, view.result = job, arm, path, stored["metadata"], stored
        view.validate()
        return view

    def validate(self) -> None:
        for op, spec in self.result["operators"].items():
            if "grid" not in spec:        # custom (patching) operator
                for key, rows in self.result["rows"].get(op, {}).items():
                    for row in rows:
                        if "generated_text" in row:
                            validate_row({k: v for k, v in row.items() if k not in ("layer", "span")}, 0.0,
                                         f"{self.arm}/{op}/{key}")
                        elif any(isinstance(row.get(r), dict) and not math.isfinite(row[r]["patched"])
                                 for r in ("tf", "fp")):
                            raise ValueError(f"{self.arm}/{op}/{key}: nonfinite patched margin")
                continue
            grid = spec["grid"]
            for key, rows in self.result["rows"].get(op, {}).items():
                if len(rows) != len(grid):
                    raise ValueError(f"{self.arm}/{op}/{key}: stored grid length differs")
                for alpha, row in zip(grid, rows, strict=True):
                    validate_row(row, alpha, f"{self.arm}/{op}/{key}")

    def save(self) -> None:
        atomic_write(self.path, self.result)

    def operator(self, name: str, *, layer: int, base: torch.Tensor, grid: Sequence[float], condition: str,
                 keys: Sequence[str], extra: Mapping[str, Any] | None = None) -> None:
        # dose facts relative to the same-layer DIM and the same-layer median residual norm
        dose = D.dose_table(self.job.dim(layer)[0], base, self.job.median_h(layer))
        spec = {"layer": layer, "grid": list(grid), "condition": condition, "keys": list(keys), "dose": dose,
                **(dict(extra) if extra else {})}
        stored = self.result["operators"].get(name)
        if stored is not None and stored != json.loads(json.dumps(spec)):
            raise ValueError(f"{self.arm}/{name}: operator definition changed during resume")
        self.result["operators"][name] = json.loads(json.dumps(spec))
        rows = self.result["rows"].setdefault(name, {})
        for key in keys:
            if key in rows:
                continue
            fp = self.job.fp(key, condition)
            out = []
            for alpha in grid:
                self.job.check_budget()
                out.append(self.job.evaluator.steered_row(fp, layer, base, alpha, label=f"{self.arm}/{name}"))
                self.job.count_row()
            rows[key] = out
            self.save()

    def custom(self, name: str, spec: Mapping[str, Any], keys: Sequence[str],
               make_rows: Callable[[str], list[dict[str, Any]]]) -> None:
        """Operators whose rows are not a plain suffix shift (patching); same resume contract."""
        spec = json.loads(json.dumps(dict(spec)))
        stored = self.result["operators"].get(name)
        if stored is not None and stored != spec:
            raise ValueError(f"{self.arm}/{name}: operator definition changed during resume")
        self.result["operators"][name] = spec
        rows = self.result["rows"].setdefault(name, {})
        for key in keys:
            if key not in rows:
                self.job.check_budget()
                rows[key] = make_rows(key)
                self.job.count_row()
                self.save()

    def rows_by_alpha(self, name: str, key: str) -> dict[float, dict[str, Any]]:
        grid = self.result["operators"][name]["grid"]
        return dict(zip(grid, self.result["rows"][name][key], strict=True))

    def finish(self, summary: Mapping[str, Any]) -> None:
        self.validate()
        self.result["summary"] = summary
        self.result["complete"] = True
        self.save()


# ============================================================================ summaries


def operator_summary(job: Job, arm: GridArm, name: str, condition: str = "balanced") -> dict[str, Any]:
    spec = arm.result["operators"][name]
    keys = spec["keys"]
    baseline = {k: job.alpha0(k, condition) for k in keys}
    per_alpha = []
    for j, alpha in enumerate(spec["grid"]):
        steered = {k: arm.result["rows"][name][k][j] for k in keys}
        per_alpha.append(S.flip_stats(baseline, steered, alpha))
    return {"baseline": S.baseline_stats(baseline), "per_alpha": per_alpha}


# ============================================================================ arms


def arm_gates(job: Job) -> dict[str, Any]:
    """Determinism, hook and cross-run identity gates. Smoke: violations raise. Full: recorded."""
    ticker = "ABNB" if "ABNB" in job.evaluation else job.targets[0]
    first = job.alpha0(ticker)
    again = job.evaluator.row(job.fp(ticker, "balanced"), 0.0, baseline=True, label="gate-repeat")
    d, stats = job.dim()
    zero = job.evaluator.steered_row(job.fp(ticker, "balanced"), job.spec.peak, torch.zeros_like(d), 2.0,
                                     label="gate-zero-hook")
    last = job.evaluator.steered_row(job.fp(ticker, "balanced"), job.spec.last_layer, d, 8.0, label="gate-last-layer")
    checks: dict[str, Any] = {
        "alpha0_repeat_identical": again["generated_text"] == first["generated_text"] and again["margin"] == first["margin"],
        "zero_hook_identical": zero["generated_text"] == first["generated_text"] and abs(zero["margin"] - first["margin"]) <= 1e-6,
        "last_layer_identical": last["generated_text"] == first["generated_text"] and abs(last["margin"] - first["margin"]) <= 1e-6,
        "suffix_tokens": job.spec.suffix_tokens,
        "template_date": R.template_provenance(job.tokenizer).get("chat_template_date"),
        "formatted_ids_sha256": job.fp(ticker, "balanced").ids_sha256,
        "difference_sha256": stats["difference_sha256"],
    }
    runs = Path("artifacts") / job.slug / "concept-cone-steering" / "runs"
    v1_path = runs / QWEN_V1 / "tokenwise" / "result.json"
    v2_path = runs / V2_SMOKE / "tokenwise" / "result.json"
    checks["reference_text_identical"] = checks["reference_difference_identical"] = None
    if job.slug == "qwen3.5-4b" and v1_path.exists():
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        checks["reference"] = QWEN_V1
        checks["reference_text_identical"] = v1["targets"][ticker]["L16"]["rows"][0]["generated_text"] == first["generated_text"]
    elif v2_path.exists():
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        checks["reference"] = V2_SMOKE
        checks["reference_text_identical"] = v2["targets"][ticker]["baseline"]["generated_text"] == first["generated_text"]
        checks["reference_difference_identical"] = (
            v2["direction_diagnostics"][str(job.spec.peak)]["difference_sha256"] == stats["difference_sha256"])
    else:
        checks["reference"] = None
    failed = [k for k, v in checks.items() if v is False]
    checks["failed"] = failed
    atomic_write(job.path("gates"), {"metadata": {**job.base_metadata, "arm": "gates"}, "checks": checks,
                                     "rows": {"alpha0": first, "repeat": again, "zero_hook": zero, "last_layer": last},
                                     "complete": True})
    if failed and job.smoke:
        raise RuntimeError(f"smoke gates failed: {failed}")
    return checks


def arm_ranking(job: Job) -> dict[str, Any]:
    ranking = job.ranking()
    top, bottom = job.top_bottom()
    record: dict[str, Any] = {"top_10": top, "bottom_10": bottom}
    runs = Path("artifacts") / job.slug / "concept-cone-steering" / "runs"
    source = runs / (V2_FULL if job.slug != "qwen3.5-4b" else "c2-guided-paper-20260924") / (
        "tokenwise/result.json" if job.slug != "qwen3.5-4b" else "result.json")
    if source.exists():
        old = json.loads(source.read_text(encoding="utf-8"))
        old_rank = [r["ticker"] for r in old["ranking"]]
        record["reference"] = str(source)
        record["reference_top_bottom_identical"] = (old_rank[-10:] == top and old_rank[:10] == bottom)
        ours = {r["ticker"]: r["margin"] for r in ranking}
        record["reference_max_abs_margin_diff"] = max(abs(ours[r["ticker"]] - r["margin"]) for r in old["ranking"])
    atomic_write(job.path("ranking", "comparison.json"), record)
    return record


def arm_alpha0(job: Job) -> dict[str, Any]:
    rows = {t: job.alpha0(t) for t in job.targets}
    stats = S.baseline_stats(rows)
    stats["gate_parse_at_least_90"] = (stats["parse_rate"] or 0) >= 0.9
    stats["gate_policy"] = "recorded only (U11: results first; no automatic downgrade)"
    return stats


def cal_candidates(job: Job) -> list[str]:
    """Construction intersection over the three split seeds, minus this model's Top/Bottom10 under each."""
    pool = set(job.construction)
    excluded: set[str] = set()
    for seed in SPLIT_SEEDS:
        _, construction, _ = R.split_population(Path(job.args.population_csv), seed)
        pool &= set(construction)
        top, bottom = D.top_bottom(job.construction_ranking(construction))
        excluded |= set(top) | set(bottom)
    candidates = sorted(pool - excluded)
    random.Random(CAL_SEED).shuffle(candidates)
    return candidates


def arm_cal(job: Job) -> dict[str, Any]:
    path = job.path("cal", "calibration.json")
    meta = {**job.base_metadata, "arm": "cal", "ladder": list(S.CAL_LADDER), "cal_seed": CAL_SEED}
    cal = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "metadata": meta, "companies": None, "rows": {}, "fallback": {}, "complete": False}
    if cal["metadata"] != meta:
        raise ValueError("cal: metadata differs; refusing to resume")
    if cal.get("complete"):
        return cal["rule"]
    if cal["companies"] is None:
        candidates = cal_candidates(job)
        if job.smoke:
            chosen = candidates[:SMOKE_CAL_COMPANIES]
        else:
            by_class: dict[str, list[str]] = {"buy": [], "sell": [], "unparsed": []}
            scanned = []
            for ticker in candidates[:CAL_SCAN]:
                decision = job.alpha0(ticker)["decision"]
                by_class[decision].append(ticker)
                scanned.append(ticker)
                if len(by_class["buy"]) >= CAL_SIZE // 2 and len(by_class["sell"]) >= CAL_SIZE // 2:
                    break
            chosen = by_class["buy"][:CAL_SIZE // 2] + by_class["sell"][:CAL_SIZE // 2]
            spare = [t for t in by_class["buy"][CAL_SIZE // 2:] + by_class["sell"][CAL_SIZE // 2:]]
            chosen += spare[:CAL_SIZE - len(chosen)]
            cal["scan"] = {"scanned": len(scanned), "classes": {k: len(v) for k, v in by_class.items()}}
        cal["companies"] = sorted(chosen)
        atomic_write(path, cal)
    d, _ = job.dim()
    layer = job.spec.peak

    def ladder(ticker: str, condition: str, signs: Sequence[float]) -> dict[str, dict[str, Any]]:
        fp = job.fp(ticker, condition)
        out: dict[str, dict[str, Any]] = {}
        for sign in signs:
            misses = 0
            for magnitude in S.CAL_LADDER:
                job.check_budget()
                row = job.evaluator.steered_row(fp, layer, d, sign * magnitude, label=f"cal/{condition}")
                job.count_row()
                out[akey(sign * magnitude)] = row
                misses = misses + 1 if row["decision"] not in S.PARSED else 0
                if misses >= 2:      # adaptive stop: two consecutive unparsed on this sign
                    break
        return out

    for ticker in cal["companies"]:
        job.alpha0(ticker)
        if ticker not in cal["rows"]:
            cal["rows"][ticker] = ladder(ticker, "balanced", (1.0, -1.0))
            atomic_write(path, cal)
    baseline = {t: job.alpha0(t) for t in cal["companies"]}
    for name, sign, condition in (("sell->buy", 1.0, "neg"), ("buy->sell", -1.0, "pos")):
        source = S.on_target(sign)[0]
        if sum(b["decision"] == source for b in baseline.values()) >= 6 or job.smoke:
            continue
        fb = cal["fallback"].setdefault(name, {"condition": condition, "rows": {}})
        for ticker in cal["companies"]:
            job.alpha0(ticker, condition)
            if ticker not in fb["rows"]:
                fb["rows"][ticker] = ladder(ticker, condition, (sign,))
                atomic_write(path, cal)

    def as_float(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[float, Any]]:
        return {t: {float(a): r for a, r in by.items()} for t, by in rows.items()}

    fallback = {name: {"baseline": {t: job.alpha0(t, fb["condition"]) for t in cal["companies"]},
                       "rows": as_float(fb["rows"])} for name, fb in cal["fallback"].items()}
    rule = S.cal_rule(baseline, as_float(cal["rows"]), fallback)
    if job.smoke:
        rule["note"] = "smoke: code-path check only; eval arms use the fixed smoke grid"
    cal["rule"] = rule
    cal["complete"] = True
    atomic_write(path, cal)
    return rule


def arm_dim(job: Job) -> dict[str, Any]:
    d, stats = job.dim()
    top, bottom = job.top_bottom()
    arm = GridArm(job, "dim", {"top_10": top, "bottom_10": bottom, "direction": stats})
    arm.operator("dim", layer=job.spec.peak, base=d, grid=job.grid("full"), condition="balanced", keys=job.targets)
    diag_keys = job.targets[:DIAG_COMPANIES]
    hi = job.alpha_hi()
    arm.operator("dim_last_layer_diagnostic", layer=job.spec.last_layer, base=d, grid=[-hi, hi],
                 condition="balanced", keys=diag_keys, extra={"structural_zero": True})
    diag_ok = all(
        row["generated_text"] == job.alpha0(k)["generated_text"] and abs(row["margin"] - job.alpha0(k)["margin"]) <= 1e-6
        for k in diag_keys for row in arm.result["rows"]["dim_last_layer_diagnostic"][k])
    summary = {"dim": operator_summary(job, arm, "dim"), "structural_zero_ok": diag_ok}
    summary["r0_agreement"] = r0_agreement(job, arm)
    arm.finish(summary)
    if not diag_ok:
        print("WARNING: structural-zero diagnostic differs from alpha 0; do not interpret flips until explained",
              flush=True)
        if job.smoke:
            raise RuntimeError("structural-zero diagnostic failed")
    return summary


def r0_agreement(job: Job, arm: GridArm) -> dict[str, Any] | None:
    """Decision agreement with the V2 (R0) / Qwen V1 run at shared alphas and the primary layer."""
    runs = Path("artifacts") / job.slug / "concept-cone-steering" / "runs"
    if job.slug == "qwen3.5-4b":
        return None      # V1 used unit directions: alphas are not comparable
    path = runs / V2_FULL / "tokenwise" / "result.json"
    if not path.exists():
        return {"status": "R0 not available"}
    old = json.loads(path.read_text(encoding="utf-8"))
    if not old.get("complete"):
        return {"status": "R0 incomplete"}
    grid = arm.result["operators"]["dim"]["grid"]
    shared = [a for a in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0) if a in grid]
    same, total, base_same = 0, 0, 0
    for key in arm.result["rows"]["dim"]:
        base_same += old["targets"][key]["baseline"]["generated_text"] == job.alpha0(key)["generated_text"]
        rows = arm.rows_by_alpha("dim", key)
        old_rows = dict(zip([a for a in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)],
                            old["targets"][key][f"L{job.spec.peak}"]["rows"]))
        for alpha in shared:
            total += 1
            same += rows[alpha]["generated_text"] == old_rows[alpha]["generated_text"]
    return {"status": "compared", "shared_alphas": shared, "identical_text": same, "pairs": total,
            "alpha0_identical_text": base_same}


def arm_dim_layers(job: Job) -> dict[str, Any]:
    """Qwen R2: V2-convention layer sweep (raw difference, calibrated grid); L31 diagnostic only."""
    if job.slug != "qwen3.5-4b":
        raise ValueError("dim_layers is the Qwen R2 arm")
    arm = GridArm(job, "dim_layers")
    summary = {}
    for layer in QWEN_R2_LAYERS:
        d, stats = job.dim(layer)
        arm.operator(f"dim_L{layer}", layer=layer, base=d, grid=job.grid("full"), condition="balanced",
                     keys=job.targets, extra={"direction": stats})
        summary[f"dim_L{layer}"] = operator_summary(job, arm, f"dim_L{layer}")
    d, _ = job.dim()
    hi = job.alpha_hi()
    arm.operator("dim_L31_diagnostic", layer=31, base=d, grid=[-hi, hi], condition="balanced",
                 keys=job.targets[:DIAG_COMPANIES], extra={"structural_zero": True})
    arm.finish(summary)
    return {k: v["baseline"] for k, v in summary.items()}


def arm_random(job: Job) -> dict[str, Any]:
    d, _ = job.dim()
    arm = GridArm(job, "random")
    for seed in RANDOM_SEEDS:
        u, sha = D.shared_random_direction(d.shape[-1], seed)
        arm.operator(f"random_s{seed}", layer=job.spec.peak, base=D.equal_norm(d, u), grid=job.grid("random"),
                     condition="balanced", keys=job.targets, extra={"seed": seed, "direction_sha256": sha})
    dim_arm = GridArm.load(job, "dim")
    summary = {f"random_s{s}": operator_summary(job, arm, f"random_s{s}") for s in RANDOM_SEEDS}
    if dim_arm.result.get("complete"):
        baseline = {k: job.alpha0(k) for k in job.targets}
        summary["c5"] = []
        for alpha in job.grid("random"):
            if alpha not in dim_arm.result["operators"]["dim"]["grid"]:
                continue
            dim_rows = {k: dim_arm.rows_by_alpha("dim", k)[alpha] for k in job.targets}
            rand = {f"s{s}": {k: arm.rows_by_alpha(f"random_s{s}", k)[alpha] for k in job.targets} for s in RANDOM_SEEDS}
            summary["c5"].append(S.c5_contrast(baseline, dim_rows, rand, alpha))
    arm.finish(summary)
    return {k: v for k, v in summary.items() if k == "c5"}


def arm_jitter(job: Job) -> dict[str, Any]:
    d, _ = job.dim()
    arm = GridArm(job, "jitter", {"fraction_of_alpha_50": JITTER_FRACTION})
    dose = JITTER_FRACTION * job.alpha_50()
    for seed in JITTER_SEEDS:
        u, sha = D.shared_random_direction(d.shape[-1], seed)
        arm.operator(f"jitter_s{seed}", layer=job.spec.peak, base=D.equal_norm(d, u), grid=[-dose, dose],
                     condition="balanced", keys=job.targets, extra={"seed": seed, "direction_sha256": sha})
    summary = {f"jitter_s{s}": operator_summary(job, arm, f"jitter_s{s}") for s in JITTER_SEEDS}
    arm.finish(summary)
    return summary


def arm_ops(job: Job) -> dict[str, Any]:
    """Operator family at the primary layer: single-neuron write vector, DIM (=cone1), cone2/cone4, controls."""
    d, _ = job.dim()
    top, bottom = job.top_bottom()
    margins = {r["ticker"]: r["margin"] for r in job.ranking()}
    axes, cone_info = D.cone_axes(job.states(job.spec.peak, top), job.states(job.spec.peak, bottom), d,
                                  [margins[t] for t in top], [margins[t] for t in bottom])
    orth, orth_sha = D.orth_random_axes(d, ORTH_SEEDS)
    writes, labels = D.neuron_write_vectors(job.model_dir, job.spec, job.spec.peak)
    # target = token-mean of unit DIM directions: maximizes the mean per-token cosine of a shared
    # write vector (a raw token mean is dominated by the few high-norm suffix tokens)
    pick = D.select_neuron(writes, labels, D.unit_rows(d).mean(dim=0))
    neuron_info = {k: v for k, v in pick.items() if k not in ("unit", "cos_by_label")}
    neuron_info["target"] = "mean_p d_hat[p]"
    if job.slug == "qwen3.5-4b":
        layer, neuron = QWEN_DIAL
        d15, _ = job.dim(layer)
        w15, l15 = D.neuron_write_vectors(job.model_dir, job.spec, layer)
        pick15 = D.select_neuron(w15, l15, D.unit_rows(d15).mean(dim=0))
        cos = pick15["cos_by_label"]
        neuron_info["historical_L15"] = {"rule_selects": pick15["neuron"], "n8490_cos": float(cos[neuron]),
                                         "n8490_rank": int((cos > cos[neuron]).sum()) + 1,
                                         "selects_n8490": pick15["neuron"] == str(neuron)}
    del writes
    arm = GridArm(job, "ops", {"cone": cone_info, "orth_random_sha256": orth_sha, "neuron": neuron_info})
    grid = job.grid("full")
    c2, c4 = D.cone_unit(d, axes, 2), D.cone_unit(d, axes, 4)
    operators = {
        "neuron": D.equal_norm(d, pick["unit"]),
        "cone2": D.equal_norm(d, c2),
        "cone4": D.equal_norm(d, c4),
        "cone4_projection": D.equal_projection(d, c4),
        "dim_orth_rand4": D.equal_norm(d, D.cone_unit(d, orth, 4)),
    }
    for name, base in operators.items():
        arm.operator(name, layer=job.spec.peak, base=base, grid=grid, condition="balanced", keys=job.targets)
    summary = {name: operator_summary(job, arm, name) for name in operators}
    dim_arm = GridArm.load(job, "dim")
    if dim_arm.result.get("complete"):
        summary["c8"] = c8_summary(job, arm, dim_arm)
    arm.finish(summary)
    return {"neuron": neuron_info, "cone": cone_info}


def c8_summary(job: Job, ops: GridArm, dim_arm: GridArm) -> dict[str, Any]:
    """Smoothness on the projection-dose axis alpha_eff = alpha * cos(operator, DIM)."""
    out = {}
    sources = {"dim": (dim_arm, "dim"), **{n: (ops, n) for n in ("neuron", "cone2", "cone4", "cone4_projection",
                                                                   "dim_orth_rand4")}}
    for name, (arm, op) in sources.items():
        spec = arm.result["operators"][op]
        cos = spec["dose"]["projection_on_dim_median"] / spec["dose"]["inject_norm_median"] if spec["dose"]["inject_norm_median"] else 0.0
        scale = spec["dose"]["projection_on_dim_median"] / job.dim()[1]["median"]
        per_alpha = operator_summary(job, arm, op)["per_alpha"]
        out[name] = {"alpha_eff_per_alpha": scale, "cos_to_dim": cos}
        for sign in (1.0, -1.0):
            points = [(p["alpha"] * scale, p["mean_delta_margin"]) for p in per_alpha if p["alpha"] * sign > 0]
            out[name][f"sign_{'+' if sign > 0 else '-'}"] = S.smoothness(points)
    return out


def arm_shuffle(job: Job) -> dict[str, Any]:
    d, _ = job.dim()
    arm = GridArm(job, "shuffle")
    for seed in SHUFFLE_SEEDS:
        top, bottom = D.shuffled_groups(job.construction_ranking(), seed)
        null, stats = D.fit_dim_difference(job.states(job.spec.peak, top), job.states(job.spec.peak, bottom))
        arm.operator(f"shuffle_s{seed}", layer=job.spec.peak, base=D.equal_norm(d, D.unit_rows(null)),
                     grid=job.grid("reduced"), condition="balanced", keys=job.targets,
                     extra={"seed": seed, "top": top, "bottom": bottom, "direction": stats})
    summary = {f"shuffle_s{s}": operator_summary(job, arm, f"shuffle_s{s}") for s in SHUFFLE_SEEDS}
    arm.finish(summary)
    return summary


def arm_evidence(job: Job) -> dict[str, Any]:
    d, _ = job.dim()
    arm = GridArm(job, "evidence")
    plan = (("pos", "full"), ("neg", "full"), ("zero", "full"), ("mixed2", "reduced"))
    for condition, kind in plan:
        for key in job.targets:
            job.alpha0(key, condition)
        arm.operator(f"dim_{condition}", layer=job.spec.peak, base=d, grid=job.grid(kind), condition=condition,
                     keys=job.targets)
    summary = {f"dim_{c}": operator_summary(job, arm, f"dim_{c}", c) for c, _ in plan}
    dim_arm = GridArm.load(job, "dim")
    if dim_arm.result.get("complete"):
        summary["c7"] = c7_summary(job, arm, dim_arm)
    arm.finish(summary)
    return {k: v["baseline"] for k, v in summary.items() if k != "c7"}


def c7_summary(job: Job, evidence: GridArm, dim_arm: GridArm) -> dict[str, Any]:
    """Per-company flip doses per condition; contrasts only where both conditions share the alpha-0 class."""
    sources = {"balanced": (dim_arm, "dim"), **{c: (evidence, f"dim_{c}") for c in ("pos", "neg", "zero", "mixed2")}}
    doses: dict[str, dict[str, dict[str, Any]]] = {}
    for condition, (arm, op) in sources.items():
        doses[condition] = {}
        for key in job.targets:
            rows = arm.rows_by_alpha(op, key)
            base = job.alpha0(key, condition)
            doses[condition][key] = {"+": S.flip_dose(base, rows, 1.0), "-": S.flip_dose(base, rows, -1.0)}
    contrasts = []
    for adverse, reference, sign in (("neg", "balanced", "+"), ("neg", "mixed2", "+"),
                                      ("pos", "balanced", "-"), ("pos", "mixed2", "-")):
        pairs = [(doses[adverse][k][sign], doses[reference][k][sign]) for k in job.targets
                 if doses[adverse][k][sign].get("eligible") and doses[reference][k][sign].get("eligible")]
        contrasts.append({
            "adverse": adverse, "reference": reference, "direction": "sell->buy" if sign == "+" else "buy->sell",
            "comparable_n": len(pairs), "descriptive_only": len(pairs) < 10,
            "adverse_blocked": sum(a["censor"] == "blocked" for a, _ in pairs),
            "adverse_collapsed": sum(a["censor"] == "collapsed" for a, _ in pairs),
            "reference_blocked": sum(b["censor"] == "blocked" for _, b in pairs),
            "higher_dose_under_adverse": sum(a["dose"] is not None and b["dose"] is not None and a["dose"] > b["dose"]
                                             for a, b in pairs),
            "equal_dose": sum(a["dose"] is not None and a["dose"] == b["dose"] for a, b in pairs),
            "lower_dose_under_adverse": sum(a["dose"] is not None and b["dose"] is not None and a["dose"] < b["dose"]
                                            for a, b in pairs)})
    return {"doses": doses, "contrasts": contrasts}


def arm_anon(job: Job) -> dict[str, Any]:
    d, _ = job.dim()
    arm = GridArm(job, "anon")
    keys = [f"anon:{i}" for i in range(len(P.ANON_IDENTITIES))]
    for key in keys:
        text = job.fp(key, "balanced").formatted
        leaks = [t for t in job.companies if f"[{t}]" in text] + [
            c["name"] for c in job.companies.values() if c["name"] in text]
        if leaks:
            raise ValueError(f"anonymous prompt {key} contains real identities {leaks[:3]}")
    for condition in ("balanced", "pos", "neg", "zero"):
        for key in keys:
            job.alpha0(key, condition)
        arm.operator(f"dim_{condition}", layer=job.spec.peak, base=d, grid=job.grid("full"), condition=condition,
                     keys=keys)
    summary = {f"dim_{c}": operator_summary(job, arm, f"dim_{c}", c) for c in ("balanced", "pos", "neg", "zero")}
    summary["estimand"] = "identity level, n=10; compare to named per-alpha rates with identity bootstrap"
    arm.finish(summary)
    return {k: v["baseline"] for k, v in summary.items() if k != "estimand"}


def _fold_direction(job: Job, top: Sequence[str], bottom: Sequence[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    fold, stats = D.fit_dim_difference(job.states(job.spec.peak, top), job.states(job.spec.peak, bottom))
    d, _ = job.dim()
    stats["cos_to_main_median"] = float((D.unit_rows(fold) * D.unit_rows(d)).sum(-1).median())
    return fold, stats


def arm_loso(job: Job, *, construction_targets: bool = False) -> dict[str, Any]:
    """R9 (held-out eval) or R10 (construction companies): sector fold vs comparator vs placebo at G_red."""
    name = "loso_construction" if construction_targets else "loso"
    ranking = job.construction_ranking()
    pool = job.construction if construction_targets else job.targets
    targets = [t for t in pool if job.companies[t]["sector"] != R.UNSPECIFIED_SECTOR]
    if job.smoke and construction_targets:
        targets = targets[:2]
    sectors = sorted({job.companies[t]["sector"] for t in targets})
    arm = GridArm(job, name)
    grid = job.grid("reduced")
    for key in targets:
        job.alpha0(key)
    comparator = None
    for s_index, sector in enumerate(sectors):
        folds = D.loso_folds(ranking, job.companies, sector, PLACEBO_SEED + s_index)
        keys = [t for t in targets if job.companies[t]["sector"] == sector]
        variants = ("loso", "placebo") if not construction_targets else ("loso",)
        for variant in variants:
            fold, stats = _fold_direction(job, folds[variant]["top_10"], folds[variant]["bottom_10"])
            arm.operator(f"{variant}:{sector}", layer=job.spec.peak, base=fold, grid=grid, condition="balanced",
                         keys=keys, extra={"fold": folds[variant], "direction": stats})
        comparator = folds["comparator"]
    if comparator is not None:
        fold, stats = _fold_direction(job, comparator["top_10"], comparator["bottom_10"])
        in_fold = set(comparator["top_10"]) | set(comparator["bottom_10"])
        arm.operator("comparator", layer=job.spec.peak, base=fold, grid=grid, condition="balanced", keys=targets,
                     extra={"fold": comparator, "direction": stats,
                            "targets_in_comparator_construction": sorted(set(targets) & in_fold)})
    summary = loso_summary(job, arm, targets, sectors, construction_targets)
    arm.finish(summary)
    return {k: v for k, v in summary.items() if k != "per_sector"}


def loso_summary(job: Job, arm: GridArm, targets: Sequence[str], sectors: Sequence[str],
                 construction_targets: bool) -> dict[str, Any]:
    baseline = {k: job.alpha0(k) for k in targets}
    grid = arm.result["operators"]["comparator"]["grid"]
    excluded = set(arm.result["operators"]["comparator"].get("targets_in_comparator_construction", []))
    per_sector: dict[str, Any] = {}
    for sector in sectors:
        keys = [t for t in targets if job.companies[t]["sector"] == sector and t not in excluded]
        per_sector[sector] = {"n": len(keys), "per_alpha": []}
        for alpha in grid:
            entry = {"alpha": alpha}
            for variant in ("loso", "placebo", "comparator"):
                op = variant if variant == "comparator" else f"{variant}:{sector}"
                if op not in arm.result["operators"]:
                    continue
                steered = {k: arm.rows_by_alpha(op, k)[alpha] for k in keys}
                entry[variant] = S.flip_stats({k: baseline[k] for k in keys}, steered, alpha)
            per_sector[sector]["per_alpha"].append(entry)
    keys = [t for t in targets if t not in excluded]
    contrasts = []
    for alpha in grid:
        def stat(sample: Sequence[str], alpha: float = alpha) -> float | None:
            source, goal = S.on_target(alpha)
            klass = [t for t in sample if baseline[t]["decision"] == source]
            if not klass:
                return None
            loso = sum(arm.rows_by_alpha(f"loso:{job.companies[t]['sector']}", t)[alpha]["decision"] == goal for t in klass)
            comp = sum(arm.rows_by_alpha("comparator", t)[alpha]["decision"] == goal for t in klass)
            return (comp - loso) / len(klass)
        discord = sum(arm.rows_by_alpha(f"loso:{job.companies[t]['sector']}", t)[alpha]["decision"]
                      != arm.rows_by_alpha("comparator", t)[alpha]["decision"] for t in keys)
        contrasts.append({"alpha": alpha, "comparator_minus_loso_itt": S.bootstrap_ci(keys, stat),
                          "discordant": discord, "n": len(keys)})
    return {"per_sector": per_sector, "contrasts": contrasts, "excluded_in_comparator_construction": sorted(excluded),
            "construction_targets": construction_targets}


def arm_split_seed(job: Job) -> dict[str, Any]:
    arm = GridArm(job, "split_seed")
    grid = job.grid("reduced")
    cal_path = job.path("cal", "calibration.json")
    cal_companies = set(json.loads(cal_path.read_text(encoding="utf-8"))["companies"]) if cal_path.exists() else set()
    summary = {}
    for seed in SPLIT_SEEDS[1:]:
        _, construction, evaluation = R.split_population(Path(job.args.population_csv), seed)
        if cal_companies & set(evaluation):
            raise ValueError("CAL companies leak into a replication evaluation split")
        top, bottom = D.top_bottom(job.construction_ranking(construction))
        fold, stats = _fold_direction(job, top, bottom)
        keys = evaluation[:2] if job.smoke else evaluation
        for key in keys:
            job.alpha0(key)
        arm.operator(f"seed_{seed}", layer=job.spec.peak, base=fold, grid=grid, condition="balanced", keys=keys,
                     extra={"top_10": top, "bottom_10": bottom, "direction": stats,
                            "eval_overlap_with_main": len(set(evaluation) & set(job.evaluation))})
        summary[f"seed_{seed}"] = operator_summary(job, arm, f"seed_{seed}")
    arm.finish(summary)
    return {k: v["baseline"] for k, v in summary.items()}


# ---------------------------------------------------------------------------- C2 v3 (R6/R7)


def r6_pairs(job: Job) -> list[tuple[str, str]]:
    ranking = job.construction_ranking()
    k = 2 if job.smoke else R6_PAIRS
    top = [r["ticker"] for r in ranking[-k:]][::-1]       # highest margin first
    bottom = [r["ticker"] for r in ranking[:k]]           # lowest margin first
    pairs = []
    for high, low in zip(top, bottom):
        pairs += [(high, low), (low, high)]
    return pairs


def tf_spans(fp: P.FormattedPrompt, row: Mapping[str, Any]) -> dict[str, tuple[int, int]] | None:
    """Spans in the teacher-forced sequence: prompt spans plus the generated answer prefix."""
    if row.get("realized_status") != "ok":
        return None
    spans = {k: v for k, v in fp.spans.items() if k != "answer_prefix"}
    spans["answer_prefix"] = (len(fp.ids), len(fp.ids) + int(row["realized_step"]))
    return spans


def span_mapping(span: str, source: tuple[int, int], target: tuple[int, int]) -> dict[int, int]:
    """BEG Phase 2 contract: offset identity for equal-length spans, nearest-normalized otherwise."""
    if source[1] <= source[0] or target[1] <= target[0]:
        return {}
    if source[1] - source[0] == target[1] - target[0] and span != "entity":
        return {target[0] + i: source[0] + i for i in range(target[1] - target[0])}
    return nearest_position_mapping(source, target)


def arm_c2v3(job: Job) -> dict[str, Any]:
    """R6: cross-company span x layer patching, margin only; primary teacher-forced clean-path readout."""
    pairs = r6_pairs(job)
    companies = sorted({t for pair in pairs for t in pair})
    base = {t: job.alpha0(t) for t in companies}
    fps = {t: job.fp(t, "balanced") for t in companies}
    layers = list(range(job.spec.n_layers))
    ev = job.evaluator
    clean_tf = {t: ev.teacher_forced_margin(fps[t], base[t]) for t in companies}
    clean_fp = {t: ev.fixed_prefix_margin(fps[t]) for t in companies}
    arm = GridArm(job, "c2v3", {"pairs": [list(p) for p in pairs], "layers": layers, "spans": list(P.SPAN_NAMES)})

    def tf_ids(t: str) -> list[int]:
        return list(fps[t].ids) + list(base[t]["new_ids"][:base[t]["realized_step"]])

    def run_direction(key: str) -> list[dict[str, Any]]:
        src, tgt = key.split("->")
        records = []
        readouts = {"fp": (list(fps[src].score_ids), list(fps[tgt].score_ids), fps[src].spans, fps[tgt].spans,
                           clean_fp[src], clean_fp[tgt], (fps[tgt].buy_id, fps[tgt].sell_id))}
        if clean_tf[src] is not None and clean_tf[tgt] is not None:
            readouts["tf"] = (tf_ids(src), tf_ids(tgt), tf_spans(fps[src], base[src]), tf_spans(fps[tgt], base[tgt]),
                              clean_tf[src], clean_tf[tgt], tuple(base[tgt]["realized_token_ids"]))
        captured = {name: record_residuals(job.model, ev._ids(r[0]), layers) for name, r in readouts.items()}
        for layer in layers:
            for span in P.SPAN_NAMES:
                entry: dict[str, Any] = {"layer": layer, "span": span}
                for name, (s_ids, t_ids, s_spans, t_spans, m_src, m_tgt, pair) in readouts.items():
                    mapping = span_mapping(span, s_spans[span], t_spans[span])
                    if not mapping:
                        entry[name] = None
                        continue
                    transform = make_span_transform(captured[name][layer], mapping)
                    with residual_interventions(job.model, {layer: transform}):
                        res = record_residuals(job.model, ev._ids(t_ids), [ev.final_layer])[ev.final_layer]
                    patched = fp32_margin(job.model, res[:, -1, :], int(pair[0]), int(pair[1]))
                    gap = m_src - m_tgt
                    entry[name] = {"patched": patched, "T": (patched - m_tgt) / gap if abs(gap) >= 1e-6 else None}
                records.append(entry)
        del captured
        return records

    keys = [f"{s}->{t}" for s, t in pairs]
    arm.custom("patch", {"clean_tf": clean_tf, "clean_fp": clean_fp}, keys, run_direction)
    self_check = self_patch_check(job, fps[companies[0]], base[companies[0]])
    summary = c2v3_summary(arm, keys, layers)
    summary["self_patch_max_abs"] = self_check
    arm.finish(summary)
    if job.smoke and self_check > 1e-6:
        raise RuntimeError(f"self-patch is not a no-op: {self_check}")
    return {k: summary[k] for k in ("band", "peak", "self_patch_max_abs")}


def self_patch_check(job: Job, fp: P.FormattedPrompt, row: Mapping[str, Any]) -> float:
    ev = job.evaluator
    ids = list(fp.score_ids)
    layers = list(range(job.spec.n_layers))
    captured = record_residuals(job.model, ev._ids(ids), layers)
    clean = ev.fixed_prefix_margin(fp)
    worst = 0.0
    for layer in layers:
        for span in P.SPAN_NAMES:
            mapping = span_mapping(span, fp.spans[span], fp.spans[span])
            if not mapping:
                continue
            with residual_interventions(job.model, {layer: make_span_transform(captured[layer], mapping)}):
                res = record_residuals(job.model, ev._ids(ids), [ev.final_layer])[ev.final_layer]
            worst = max(worst, abs(fp32_margin(job.model, res[:, -1, :], fp.buy_id, fp.sell_id) - clean))
    return worst


def c2v3_summary(arm: GridArm, keys: Sequence[str], layers: Sequence[int]) -> dict[str, Any]:
    curves: dict[str, dict[str, dict[str, Any]]] = {}
    for readout in ("tf", "fp"):
        curves[readout] = {}
        for span in P.SPAN_NAMES:
            curves[readout][span] = {}
            for layer in layers:
                values = [e[readout]["T"] for key in keys for e in arm.result["rows"]["patch"][key]
                          if e["layer"] == layer and e["span"] == span and e.get(readout) and e[readout]["T"] is not None]
                ci = S.bootstrap_ci(values, lambda xs: fmean(xs) if xs else None, samples=2000) if len(values) >= 4 else None
                curves[readout][span][str(layer)] = {"mean_T": fmean(values) if values else None, "n": len(values),
                                                     "ci": ci}
    steer = {int(l): v["mean_T"] for l, v in curves["tf"]["steer_suffix"].items() if v["mean_T"] is not None}
    if not steer:
        steer = {int(l): v["mean_T"] for l, v in curves["fp"]["steer_suffix"].items() if v["mean_T"] is not None}
    peak = max(steer, key=steer.get) if steer else None
    band = sorted(l for l, v in steer.items() if peak is not None and v >= 0.7 * steer[peak])
    return {"curves": curves, "peak": peak, "band": band,
            "band_rule": "steer_suffix teacher-forced curve, T >= 0.7 x peak (fixed-prefix if TF unavailable)"}


def arm_c2v3_gen(job: Job) -> dict[str, Any]:
    """R7: patch-under-generation at the R6 band peak +-2, steer_suffix and entity spans."""
    stored = json.loads(job.path("c2v3").read_text(encoding="utf-8"))
    if not stored.get("complete"):
        raise ValueError("c2v3_gen requires a complete c2v3 arm")
    peak = stored["summary"]["peak"]
    layers = sorted({l for l in (peak - 2, peak, peak + 2) if 0 <= l < job.spec.n_layers})
    pairs = r6_pairs(job)
    companies = sorted({t for pair in pairs for t in pair})
    base = {t: job.alpha0(t) for t in companies}
    fps = {t: job.fp(t, "balanced") for t in companies}
    arm = GridArm(job, "c2v3_gen", {"layers": layers, "spans": list(R7_SPANS), "peak": peak})
    ev = job.evaluator

    def run(key: str) -> list[dict[str, Any]]:
        src, tgt = key.split("->")
        captured = record_residuals(job.model, ev._ids(list(fps[src].ids)), layers)
        out = []
        for layer in layers:
            for span in R7_SPANS:
                mapping = span_mapping(span, fps[src].spans[span], fps[tgt].spans[span])
                row = ev.row(fps[tgt], 0.0, {layer: _prefill_only(make_span_transform(captured[layer], mapping))},
                             label=f"c2v3_gen/L{layer}/{span}")
                out.append({"layer": layer, "span": span, **row})
        return out

    keys = [f"{s}->{t}" for s, t in pairs]
    arm.custom("patch_generation", {"layers": layers}, keys, run)
    self_keys = companies[:2] if job.smoke else companies[:10]

    def run_self(key: str) -> list[dict[str, Any]]:
        captured = record_residuals(job.model, ev._ids(list(fps[key].ids)), [peak])
        mapping = span_mapping("steer_suffix", fps[key].spans["steer_suffix"], fps[key].spans["steer_suffix"])
        row = ev.row(fps[key], 0.0, {peak: _prefill_only(make_span_transform(captured[peak], mapping))},
                     label="c2v3_gen/self")
        return [{"layer": peak, "span": "steer_suffix", **row}]

    arm.custom("self_patch", {"layer": peak}, self_keys, run_self)
    self_ok = all(arm.result["rows"]["self_patch"][k][0]["generated_text"] == base[k]["generated_text"] for k in self_keys)
    summary = {"self_patch_identical": self_ok, "per_cell": []}
    for layer in layers:
        for span in R7_SPANS:
            moved = eligible = parsed = 0
            shifts = []
            for key in keys:
                src, tgt = key.split("->")
                row = next(e for e in arm.result["rows"]["patch_generation"][key] if e["layer"] == layer and e["span"] == span)
                parsed += row["decision"] in S.PARSED
                # realized-path margin moved toward the source's clean realized margin (sign-aligned)
                m_src, m_tgt = base[src]["realized_margin"], base[tgt]["realized_margin"]
                if None not in (row["realized_margin"], m_src, m_tgt) and m_src != m_tgt:
                    shifts.append((row["realized_margin"] - m_tgt) * (1 if m_src > m_tgt else -1))
                if base[src]["decision"] in S.PARSED and base[tgt]["decision"] in S.PARSED and base[src]["decision"] != base[tgt]["decision"]:
                    eligible += 1
                    moved += row["decision"] == base[src]["decision"]
            summary["per_cell"].append({"layer": layer, "span": span, "toward_source_flips": moved,
                                        "eligible_pairs": eligible, "parsed": parsed, "n": len(keys),
                                        "mean_realized_shift_toward_source": fmean(shifts) if shifts else None,
                                        "realized_shift_n": len(shifts)})
    arm.finish(summary)
    if job.smoke and not self_ok:
        raise RuntimeError("self-patch generation differs from alpha 0")
    return summary


def _prefill_only(transform: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    def wrapped(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor
        return transform(tensor)
    return wrapped


ARMS: dict[str, Callable[[Job], Any]] = {
    "gates": arm_gates, "ranking": arm_ranking, "alpha0": arm_alpha0, "cal": arm_cal, "dim": arm_dim,
    "dim_layers": arm_dim_layers, "random": arm_random, "jitter": arm_jitter, "ops": arm_ops, "shuffle": arm_shuffle,
    "evidence": arm_evidence, "anon": arm_anon, "loso": arm_loso,
    "loso_construction": lambda job: arm_loso(job, construction_targets=True),
    "split_seed": arm_split_seed, "c2v3": arm_c2v3, "c2v3_gen": arm_c2v3_gen,
}


def main(argv: Sequence[str] | None = None, *, model: Any = None, tokenizer: Any = None) -> int:
    args = parse_args(argv)
    slug = Path(args.model).resolve().name
    arms = resolve_arms(slug, args.arms)
    job = Job(args, model=model, tokenizer=tokenizer)
    job.root.mkdir(parents=True, exist_ok=True)
    invocation = {"arms": arms, "git": job.git, "runtime": R.runtime_versions(), "max_rows": args.max_rows}
    log = job.root / "invocations.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(invocation, ensure_ascii=False) + "\n")
    outcomes: dict[str, Any] = {}
    try:
        for arm in arms:
            print(f"==> arm {arm} ({slug}, {args.phase})", flush=True)
            outcomes[arm] = ARMS[arm](job)
            print(f"<== arm {arm} complete", flush=True)
    except MaxRowsReached:
        print("max rows reached; state saved for resume", flush=True)
        return 3
    atomic_write(job.root / "job_summary.json", {"metadata": job.base_metadata, "arms": arms,
                                                 "outcomes": json.loads(json.dumps(outcomes, default=str))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
