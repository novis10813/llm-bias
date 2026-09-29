"""evidence-scan-v1: which company x evidence-combination prompts make the model answer buy?

Protocol: docs/concept-cone-steering/evidence-scan-v1/proposal.md. The evidence comes from the per-company pool of
eight items (four for a price increase, four for a decrease) built by ``scripts/build_evidence_pool.py``; the
prompt skeleton is the frozen balanced prompt, unchanged. Every company is scanned with its 30 random two-plus-two
combinations plus the frozen four-sentence reference, each with the answer options in both orders.

``--phase screen`` measures the fixed-prefix margin log p(buy) - log p(sell) on the construction companies;
``--phase confirm`` repeats it on the held-out evaluation companies and adds real greedy generation on the canonical
rendering; ``--phase analyze`` (CPU only) summarizes. Only compact rows and provenance are written; no hidden states.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_rdo_cone as base  # noqa: E402
import probe_rdo_cone_v2 as v2  # noqa: E402

from llm_bias.balanced_evidence_gap.template import (  # noqa: E402
    EVIDENCE_MARKER, EVIDENCE_N1, EVIDENCE_N2, EVIDENCE_P1, EVIDENCE_P2)
from llm_bias.core.model import load_model  # noqa: E402
from llm_bias.core.steering import prompts as P  # noqa: E402
from llm_bias.core.steering import protocol as R  # noqa: E402
from llm_bias.core.steering.evaluate import Evaluator  # noqa: E402

SCHEMA = "concept-cone-steering-evidence-scan-v1"
REPO = Path(__file__).resolve().parents[1]
CODE_FILES = ("scripts/probe_evidence_scan.py", "scripts/build_evidence_pool.py", "llm_bias/core/steering/prompts.py",
              "llm_bias/core/steering/protocol.py", "llm_bias/core/steering/evaluate.py",
              "llm_bias/balanced_evidence_gap/template.py")
POOL_SCHEMA = "evidence-pool-v1"
SUFFIX_TOKENS = 16          # instruction-length sanity check only; nothing is injected in this scan
REF = "ref"                 # the frozen four-sentence balanced evidence (P1, P2, N1, N2)
REF_POLARITY = "++--"
N_COMBOS = 30
UNITS = (REF, *(f"a{k:02d}" for k in range(N_COMBOS)))
RENDERINGS: tuple[bool, ...] = (False, True)   # options reversed?; False = '"buy" or "sell"' is canonical
CANON = False
PATTERNS = ("++--", "+-+-", "+--+", "-++-", "-+-+", "--++")


# ------------------------------------------------------------------------------------ pure helpers

def render_scan_prompt(ticker: str, name: str, items: Sequence[str], reverse: bool) -> str:
    """The frozen balanced-prompt skeleton with the given evidence bullets."""
    options = '"sell" or "buy"' if reverse else '"buy" or "sell"'
    evidence = "\n".join(f"- {item}" for item in items)
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"Stock Ticker: [{ticker}]\n\nStock Name: [{name}]\n\n"
        f"{EVIDENCE_MARKER}\n\n{evidence}\n\n"
        "—\n\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        f'"decision": {options}\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


def load_pool(path: Path) -> tuple[dict[str, Any], str]:
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    if payload.get("schema") != POOL_SCHEMA:
        raise ValueError(f"{path} is not an {POOL_SCHEMA} file")
    for ticker, company in payload["companies"].items():
        pol = [item["polarity"] for item in company["items"]]
        if len(pol) != 8 or pol.count("+") != 4 or len(company["attribute_combos"]) != N_COMBOS:
            raise ValueError(f"{ticker}: malformed evidence pool entry")
        if any(sum(pol[k] == "+" for k in combo) != 2 or len(set(combo)) != 4 for combo in company["attribute_combos"]):
            raise ValueError(f"{ticker}: an attribute combination is not two positive plus two negative items")
    return payload, R.sha256_bytes(payload_bytes)


def unit_evidence(company: dict[str, Any], unit: str) -> tuple[tuple[str, ...], str]:
    """(evidence items in prompt order, polarity string such as '+-+-') for one unit of a company."""
    if unit == REF:
        return (EVIDENCE_P1, EVIDENCE_P2, EVIDENCE_N1, EVIDENCE_N2), REF_POLARITY
    combo = company["attribute_combos"][int(unit[1:])]
    items = company["items"]
    return tuple(items[k]["text"] for k in combo), "".join(items[k]["polarity"] for k in combo)


def rkey(reverse: bool) -> str:
    return f"r{int(reverse)}"


def row_key(unit: str, ticker: str, reverse: bool) -> str:
    return f"{unit}|{ticker}|{rkey(reverse)}"


def parse_shard(text: str) -> tuple[int, int]:
    index, count = (int(x) for x in text.split("/"))
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"bad shard {text!r}; expected i/n with 0 <= i < n")
    return index, count


def shard_items(values: Sequence[str], shard: tuple[int, int]) -> list[str]:
    index, count = shard
    return [v for i, v in enumerate(values) if i % count == index]


def buy_rate(margins: Sequence[float]) -> float:
    return sum(1 for m in margins if m > 0) / len(margins)


def _dummies(labels: Sequence[Any]) -> np.ndarray:
    levels = sorted(set(labels), key=str)
    index = {level: i for i, level in enumerate(levels)}
    out = np.zeros((len(labels), len(levels)))
    out[np.arange(len(labels)), [index[x] for x in labels]] = 1.0
    return out


def r_squared(y: np.ndarray, design: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return 1.0 - float(((y - design @ coef) ** 2).sum()) / total


def position_effects(y: np.ndarray, tickers: Sequence[str], polarity: Sequence[str]) -> dict[str, float]:
    """Mean margin change when the item at a position is positive, with a company fixed effect."""
    cols = np.array([[1.0 if p[k] == "+" else 0.0 for k in range(4)] for p in polarity])
    d = _dummies(tickers)
    design = np.hstack([d, cols])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {f"position_{k + 1}": float(coef[d.shape[1] + k]) for k in range(4)}


# ------------------------------------------------------------------------------------ analysis

def load_runs(slug: str, run_ids: Sequence[str], pool_sha: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for run_id in run_ids:
        path = Path("artifacts") / slug / "concept-cone-steering" / "runs" / run_id / "result.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not stored.get("complete"):
            raise ValueError(f"{run_id} is incomplete")
        if stored["metadata"]["pool_sha256"] != pool_sha:
            raise ValueError(f"{run_id} used a different evidence pool")
        clash = set(rows) & set(stored["rows"])
        if clash:
            raise ValueError(f"{run_id} repeats {len(clash)} rows already loaded")
        rows.update(stored["rows"])
    return rows


def _collect(rows: dict[str, Any], pool: dict[str, Any], tickers: Sequence[str], reverse: bool
             ) -> tuple[list[str], list[str], np.ndarray]:
    """Attribute-unit cells of one rendering: (ticker per cell, polarity per cell, margin per cell)."""
    names, pols, margins = [], [], []
    for t in tickers:
        for unit in UNITS[1:]:
            names.append(t)
            pols.append(unit_evidence(pool[t], unit)[1])
            margins.append(rows[row_key(unit, t, reverse)]["margin"])
    return names, pols, np.array(margins)


def _group_rate(values: np.ndarray, groups: Sequence[str]) -> dict[str, dict[str, float]]:
    out = {}
    for g in sorted(set(groups)):
        v = values[[x == g for x in groups]]
        out[g] = {"n": int(len(v)), "buy_rate": float((v > 0).mean()), "mean_margin": float(v.mean())}
    return out


def screen_summary(rows: dict[str, Any], pool: dict[str, Any], tickers: Sequence[str]) -> dict[str, Any]:
    names, pols, canon = _collect(rows, pool, tickers, CANON)
    _, _, rev = _collect(rows, pool, tickers, not CANON)
    per_ticker = canon.reshape(len(tickers), N_COMBOS)
    fractions = (per_ticker > 0).mean(axis=1)
    order = np.argsort(-fractions, kind="stable")
    listing = [{"ticker": tickers[j], "name": pool[tickers[j]]["name_hint"], "sector": pool[tickers[j]]["sector"],
                "buy_fraction": float(fractions[j])} for j in order]
    edges = [0.0, 1e-9, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - 1e-9, 1.0 + 1e-9]
    hist = np.histogram(fractions, bins=edges)[0].tolist()
    sectors: dict[str, list[float]] = {}
    for j, t in enumerate(tickers):
        sectors.setdefault(pool[t]["sector"], []).append(float(fractions[j]))
    ref_canon = [rows[row_key(REF, t, CANON)]["margin"] for t in tickers]
    ref_rev = [rows[row_key(REF, t, not CANON)]["margin"] for t in tickers]
    design_ticker = _dummies(names)
    design_pattern = _dummies(pols)
    return {
        "n_companies": len(tickers), "n_cells": int(len(canon)),
        "attribute_canonical": {"buy_rate": float((canon > 0).mean()), "mean_margin": float(canon.mean()),
                                "median_margin": float(np.median(canon))},
        "attribute_reversed_options": {"buy_rate": float((rev > 0).mean()), "mean_margin": float(rev.mean())},
        "options_order_shift_mean_margin": float((canon - rev).mean()),
        "by_pattern_canonical": _group_rate(canon, pols),
        "by_first_polarity_canonical": _group_rate(canon, [p[0] for p in pols]),
        "by_last_polarity_canonical": _group_rate(canon, [p[-1] for p in pols]),
        "position_effects_canonical": position_effects(canon, names, pols),
        "variance_explained_canonical": {"company": r_squared(canon, design_ticker),
                                         "pattern": r_squared(canon, design_pattern),
                                         "company_plus_pattern": r_squared(canon, np.hstack([design_ticker, design_pattern]))},
        "company_buy_fraction": {"bins": edges[:-1], "histogram": hist,
                                 "all_sell": int((fractions == 0).sum()), "all_buy": int((fractions == 1).sum()),
                                 "mixed": int(((fractions > 0) & (fractions < 1)).sum()),
                                 "most_buy": listing[:15], "least_buy": listing[-15:]},
        "sector_mean_buy_fraction": {s: fmean(v) for s, v in sorted(sectors.items())},
        "reference_frozen": {"canonical_buy_rate": buy_rate(ref_canon), "reversed_buy_rate": buy_rate(ref_rev),
                             "canonical_mean_margin": fmean(ref_canon)},
    }


def confirm_summary(rows: dict[str, Any], pool: dict[str, Any], tickers: Sequence[str],
                    screen: dict[str, Any] | None) -> dict[str, Any]:
    cells = []
    for t in tickers:
        for unit in UNITS:
            row = rows[row_key(unit, t, CANON)]
            polarity = unit_evidence(pool[t], unit)[1]
            cells.append({"unit": unit, "polarity": polarity, **row})

    def stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        parsed = [c for c in subset if c["decision"] in ("buy", "sell")]
        return {"n": len(subset), "parse_rate": len(parsed) / len(subset),
                "generated_buy_rate": fmean(c["decision"] == "buy" for c in parsed) if parsed else None,
                "margin_buy_rate": buy_rate([c["margin"] for c in subset]),
                "margin_sign_agrees_with_generation": (fmean((c["margin"] > 0) == (c["decision"] == "buy")
                                                             for c in parsed) if parsed else None)}

    attribute = [c for c in cells if c["unit"] != REF]
    out = {"n_companies": len(tickers), "attribute": stats(attribute),
           "by_pattern": {p: stats([c for c in attribute if c["polarity"] == p]) for p in PATTERNS},
           "reference_frozen": stats([c for c in cells if c["unit"] == REF])}
    reversed_rate = buy_rate([rows[row_key(u, t, not CANON)]["margin"] for t in tickers for u in UNITS[1:]])
    out["attribute_reversed_options_margin_buy_rate"] = reversed_rate
    fractions = {t: fmean(rows[row_key(u, t, CANON)]["decision"] == "buy" for u in UNITS[1:]) for t in tickers}
    out["company_generated_buy_fraction"] = {"all_sell": sum(v == 0 for v in fractions.values()),
                                             "all_buy": sum(v == 1 for v in fractions.values()),
                                             "mixed": sum(0 < v < 1 for v in fractions.values())}
    if screen:
        out["construction_attribute_canonical_buy_rate"] = screen["attribute_canonical"]["buy_rate"]
    return out


# ------------------------------------------------------------------------------------ model-bound scan

class Scan:
    def __init__(self, args: argparse.Namespace, model: Any, tokenizer: Any) -> None:
        self.args = args
        self.phase = args.phase
        self.model_dir = Path(args.model)
        self.slug = self.model_dir.resolve().name
        self.spec = R.model_spec(self.slug)
        if not args.run_id.startswith("evidence-scan-v1-") or f"-{args.phase}-" not in args.run_id:
            raise ValueError("run id must be evidence-scan-v1-<date>-<phase>-NN and match --phase")
        self.root = Path("artifacts") / self.slug / "concept-cone-steering" / "runs" / args.run_id
        self.git = R.git_provenance(REPO)
        smoke = self.phase == "smoke"
        if not smoke and self.git["code_dirty"]:
            raise ValueError(f"screen/confirm runs require committed code; dirty: {self.git['code_dirty_paths']}")
        if smoke and self.git["code_dirty"] and not args.allow_dirty:
            raise ValueError("smoke with uncommitted code requires --allow-dirty")
        self.companies, self.construction, self.evaluation = R.split_population(Path(args.population_csv), R.SPLIT_SEED)
        self.split_sha = R.split_sha256(self.construction, self.evaluation)
        payload, self.pool_sha = load_pool(Path(args.pool))
        self.pool = payload["companies"]
        if not set(self.pool) <= set(self.companies):
            raise ValueError("the evidence pool names companies outside the population")
        self.units = UNITS
        if smoke:
            if not set(args.smoke_tickers) <= set(self.evaluation) & set(self.pool):
                raise ValueError("smoke tickers must be held-out evaluation companies with an evidence pool")
            self.tickers, self.generate = list(args.smoke_tickers), True
            self.units = UNITS[:1 + args.smoke_units]
        else:
            source = self.construction if self.phase == "screen" else self.evaluation
            pooled = [t for t in source if t in self.pool]
            self.tickers = shard_items(pooled, parse_shard(args.shard))
            self.generate = self.phase == "confirm"
        self.log = v2.Log(self.root / "scan.log")
        if model is None:
            model, tokenizer, _ = load_model(str(self.model_dir), dtype="native" if self.spec.dtype == "native" else None)
        if model.n_layers != self.spec.n_layers:
            raise ValueError("model layer count differs from the registry")
        self.model, self.tokenizer = model, tokenizer
        hf = getattr(model, "_hf_model", model)
        hf.eval()
        for parameter in hf.parameters():
            parameter.requires_grad_(False)
        self.evaluator = Evaluator(model, tokenizer, max_new_tokens=R.MAX_NEW_TOKENS, printer=self.log)
        self.metadata = {
            "schema": SCHEMA, "phase": self.phase, "run_id": args.run_id, **R.checkpoint_identity(self.model_dir),
            **R.template_provenance(tokenizer), "split_seed": R.SPLIT_SEED, "split_sha256": self.split_sha,
            "pool_sha256": self.pool_sha, "shard": args.shard, "tickers": self.tickers, "units": list(self.units),
            "renderings": [rkey(r) for r in RENDERINGS], "generate_canonical": self.generate,
            "code_sha256": R.file_sha256({name: REPO / name for name in CODE_FILES}),
            "decision_prefix": R.DECISION_PREFIX, "primary_parse": "complete_object",
            "git_commit": self.git["git_commit"],
        }

    def fp(self, unit: str, ticker: str, reverse: bool) -> P.FormattedPrompt:
        items, _ = unit_evidence(self.pool[ticker], unit)
        prompt = render_scan_prompt(ticker, self.companies[ticker]["name"], items, reverse)
        return P.format_decision_prompt(self.tokenizer, prompt, suffix_tokens=SUFFIX_TOKENS,
                                        key=row_key(unit, ticker, reverse))

    def run(self) -> None:
        path = self.root / "result.json"
        rows: dict[str, Any] = {}
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored["metadata"] != self.metadata:
                raise ValueError("existing result.json has different metadata; use a new run id")
            rows = stored["rows"]
        total = len(self.tickers) * len(self.units) * len(RENDERINGS)
        self.log(f"== {self.args.run_id} phase={self.phase} tickers={len(self.tickers)} units={len(self.units)} "
                 f"rows={total} done={len(rows)} commit={self.git['git_commit']}")

        def save(complete: bool) -> None:
            base.atomic_write(path, {"metadata": self.metadata, "rows": rows, "complete": complete})

        started, fresh = time.time(), 0
        for ticker in self.tickers:
            for unit in self.units:
                for reverse in RENDERINGS:
                    key = row_key(unit, ticker, reverse)
                    if key in rows:
                        continue
                    fp = self.fp(unit, ticker, reverse)
                    if self.generate and reverse == CANON:
                        row = self.evaluator.row(fp, 0.0, None, label=key)
                        rows[key] = {k: row[k] for k in ("margin", "generated_text", "decision", "format", "finish",
                                                         "n_new_tokens")}
                    else:
                        rows[key] = {"margin": self.evaluator.fixed_prefix_margin(fp)}
                    fresh += 1
                    if fresh % 500 == 0:
                        save(False)
                        rate = fresh / (time.time() - started)
                        self.log(f"progress {len(rows)}/{total} ({rate:.1f} rows/s, eta {(total - len(rows)) / rate / 60:.0f} min)")
        save(True)
        self.log(f"done {len(rows)} rows")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=".cache/models/<slug>")
    parser.add_argument("--phase", choices=("smoke", "screen", "confirm", "analyze"), required=True)
    parser.add_argument("--run-id", required=True, help="evidence-scan-v1-<date>-<phase>-NN")
    parser.add_argument("--pool", required=True, help="evidence_pool.json from scripts/build_evidence_pool.py")
    parser.add_argument("--population-csv", default=R.POPULATION_CSV)
    parser.add_argument("--shard", default="0/1", help="i/n: run the companies whose index mod n is i")
    parser.add_argument("--screen-runs", nargs="*", default=[], help="analyze: screen run ids to merge")
    parser.add_argument("--confirm-runs", nargs="*", default=[], help="analyze: confirm run ids to merge")
    parser.add_argument("--smoke-tickers", nargs="+", default=["ABNB", "AEP"])
    parser.add_argument("--smoke-units", type=int, default=3, help="smoke: attribute combinations per company")
    parser.add_argument("--allow-dirty", action="store_true", help="smoke only: permit uncommitted code")
    return parser.parse_args(argv)


def analyze(args: argparse.Namespace) -> None:
    slug = Path(args.model).resolve().name
    if not args.run_id.startswith("evidence-scan-v1-") or "-analyze-" not in args.run_id:
        raise ValueError("run id must be evidence-scan-v1-<date>-analyze-NN")
    root = Path("artifacts") / slug / "concept-cone-steering" / "runs" / args.run_id
    log = v2.Log(root / "scan.log")
    companies, construction, evaluation = R.split_population(Path(args.population_csv), R.SPLIT_SEED)
    payload, pool_sha = load_pool(Path(args.pool))
    pool = payload["companies"]
    summary: dict[str, Any] = {"schema": SCHEMA, "run_id": args.run_id, "pool_sha256": pool_sha,
                               "screen_runs": args.screen_runs, "confirm_runs": args.confirm_runs,
                               "git_commit": R.git_provenance(REPO)["git_commit"]}
    screen = None
    if args.screen_runs:
        tickers = [t for t in construction if t in pool]
        screen = screen_summary(load_runs(slug, args.screen_runs, pool_sha), pool, tickers)
        summary["screen"] = screen
        log(f"screen: {screen['n_companies']} companies, canonical buy rate {screen['attribute_canonical']['buy_rate']:.3f}, "
            f"reversed {screen['attribute_reversed_options']['buy_rate']:.3f}, ref {screen['reference_frozen']}")
        log(f"company buy fractions {screen['company_buy_fraction']['all_sell']} all-sell / "
            f"{screen['company_buy_fraction']['mixed']} mixed / {screen['company_buy_fraction']['all_buy']} all-buy; "
            f"variance explained {screen['variance_explained_canonical']}; positions {screen['position_effects_canonical']}")
        for pattern, v in screen["by_pattern_canonical"].items():
            log(f"  pattern {pattern}: buy_rate={v['buy_rate']:.3f} mean M={v['mean_margin']:+.2f} n={v['n']}")
    if args.confirm_runs:
        tickers = [t for t in evaluation if t in pool]
        summary["confirm"] = confirm_summary(load_runs(slug, args.confirm_runs, pool_sha), pool, tickers, screen)
        c = summary["confirm"]
        log(f"confirm: {c['n_companies']} companies; attribute {c['attribute']}; ref {c['reference_frozen']}")
    base.atomic_write(root / "summary.json", summary)


def main(argv: Sequence[str] | None = None, *, model: Any = None, tokenizer: Any = None) -> int:
    args = parse_args(argv)
    if args.phase == "analyze":
        analyze(args)
        return 0
    Scan(args, model, tokenizer).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
