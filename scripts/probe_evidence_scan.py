"""evidence-scan-v1: which company x evidence-combination prompts make the model answer buy?

Protocol: docs/concept-cone-steering/evidence-scan-v1/proposal.md. The evidence comes from the per-company pool of
eight items (four for a price increase, four for a decrease) built by ``scripts/build_evidence_pool.py``; the
prompt skeleton is the frozen balanced prompt, unchanged. Every company is scanned with its 30 random two-plus-two
combinations plus the frozen four-sentence reference. The canonical rendering (options "buy" or "sell") gets a real
greedy generation with complete-object parsing; the reversed rendering gets the fixed-prefix margin only.

``--phase screen`` runs the construction companies, ``--phase confirm`` the held-out evaluation companies (same
procedure), ``--phase analyze`` (CPU only) summarizes. Only compact rows and provenance are written.
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
    """Marginal margin change when the item at one position is positive, with a company fixed effect.

    Each position is regressed on its own: with two positive items per prompt the four position indicators sum to a
    constant, so a joint regression is not identified.
    """
    d = _dummies(tickers)
    out = {}
    for k in range(4):
        column = np.array([[1.0 if p[k] == "+" else 0.0] for p in polarity])
        coef, *_ = np.linalg.lstsq(np.hstack([d, column]), y, rcond=None)
        out[f"position_{k + 1}"] = float(coef[-1])
    return out


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


def _cells(rows: dict[str, Any], pool: dict[str, Any], tickers: Sequence[str]) -> list[dict[str, Any]]:
    """Canonical-rendering attribute cells with their company, polarity pattern and generation record."""
    return [{"ticker": t, "polarity": unit_evidence(pool[t], unit)[1], **rows[row_key(unit, t, CANON)]}
            for t in tickers for unit in UNITS[1:]]


def _is_parsed(cell: dict[str, Any]) -> bool:
    return cell["decision"] in ("buy", "sell")


def describe(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates over one group of canonical cells; buy rates are over parsed generations."""
    parsed = [c for c in cells if _is_parsed(c)]
    paths: dict[str, int] = {}
    for c in cells:
        paths[c["path_class"]] = paths.get(c["path_class"], 0) + 1
    return {"n": len(cells), "parse_rate": len(parsed) / len(cells),
            "generated_buy_rate": fmean(c["decision"] == "buy" for c in parsed) if parsed else None,
            "fixed_prefix_margin_buy_rate": buy_rate([c["margin"] for c in cells]),
            "margin_sign_agrees_with_generation": (fmean((c["margin"] > 0) == (c["decision"] == "buy")
                                                         for c in parsed) if parsed else None),
            "mean_realized_margin": fmean(c["realized_margin"] for c in cells if c["realized_margin"] is not None)
            if any(c["realized_margin"] is not None for c in cells) else None,
            "path_class_share": {k: v / len(cells) for k, v in sorted(paths.items())}}


def _by(cells: list[dict[str, Any]], key) -> dict[str, dict[str, Any]]:
    groups = sorted({key(c) for c in cells})
    return {g: describe([c for c in cells if key(c) == g]) for g in groups}


def scan_summary(rows: dict[str, Any], pool: dict[str, Any], tickers: Sequence[str]) -> dict[str, Any]:
    """Everything reported for one group of companies (construction for screen, evaluation for confirm)."""
    cells = _cells(rows, pool, tickers)
    parsed = [c for c in cells if _is_parsed(c)]
    fractions: dict[str, float] = {}
    for t in tickers:
        mine = [c for c in parsed if c["ticker"] == t]
        if mine:
            fractions[t] = fmean(c["decision"] == "buy" for c in mine)
    listing = sorted(({"ticker": t, "name": pool[t]["name_hint"], "sector": pool[t]["sector"], "buy_fraction": f}
                      for t, f in fractions.items()), key=lambda x: (-x["buy_fraction"], x["ticker"]))
    values = np.array(list(fractions.values()))
    edges = [0.0, 1e-9, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - 1e-9, 1.0 + 1e-9]
    sectors: dict[str, list[float]] = {}
    for t, f in fractions.items():
        sectors.setdefault(pool[t]["sector"], []).append(f)
    buy = np.array([1.0 if c["decision"] == "buy" else 0.0 for c in parsed])
    names = [c["ticker"] for c in parsed]
    pols = [c["polarity"] for c in parsed]
    with_margin = [c for c in parsed if c["realized_margin"] is not None]
    y = np.array([c["realized_margin"] for c in with_margin])
    m_names, m_pols = [c["ticker"] for c in with_margin], [c["polarity"] for c in with_margin]
    fixed_c = np.array([rows[row_key(u, t, CANON)]["margin"] for t in tickers for u in UNITS[1:]])
    fixed_r = np.array([rows[row_key(u, t, not CANON)]["margin"] for t in tickers for u in UNITS[1:]])
    ref_cells = [{"ticker": t, "polarity": REF_POLARITY, **rows[row_key(REF, t, CANON)]} for t in tickers]
    return {
        "n_companies": len(tickers), "attribute": describe(cells),
        "by_pattern": _by(cells, lambda c: c["polarity"]),
        "by_first_polarity": _by(cells, lambda c: c["polarity"][0]),
        "by_last_polarity": _by(cells, lambda c: c["polarity"][-1]),
        "options_order": {"fixed_prefix_margin_buy_rate_canonical": float((fixed_c > 0).mean()),
                          "fixed_prefix_margin_buy_rate_reversed": float((fixed_r > 0).mean()),
                          "mean_margin_shift_canonical_minus_reversed": float((fixed_c - fixed_r).mean())},
        "variance_explained_realized_margin": {
            "company": r_squared(y, _dummies(m_names)), "pattern": r_squared(y, _dummies(m_pols)),
            "company_plus_pattern": r_squared(y, np.hstack([_dummies(m_names), _dummies(m_pols)]))},
        "variance_explained_buy_indicator": {
            "company": r_squared(buy, _dummies(names)), "pattern": r_squared(buy, _dummies(pols)),
            "company_plus_pattern": r_squared(buy, np.hstack([_dummies(names), _dummies(pols)]))},
        "position_effects_realized_margin": position_effects(y, m_names, m_pols),
        "position_effects_buy_indicator": position_effects(buy, names, pols),
        "company_buy_fraction": {"bins": edges[:-1], "histogram": np.histogram(values, bins=edges)[0].tolist(),
                                 "all_sell": int((values == 0).sum()), "all_buy": int((values == 1).sum()),
                                 "mixed": int(((values > 0) & (values < 1)).sum()),
                                 "most_buy": listing[:15], "least_buy": listing[-15:]},
        "sector_mean_buy_fraction": {s: fmean(v) for s, v in sorted(sectors.items())},
        "reference_frozen": describe(ref_cells),
        "reference_frozen_reversed_fixed_prefix_margin_buy_rate": buy_rate(
            [rows[row_key(REF, t, not CANON)]["margin"] for t in tickers]),
    }


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
            self.tickers = list(args.smoke_tickers)
            self.units = UNITS[:1 + args.smoke_units]
        else:
            source = self.construction if self.phase == "screen" else self.evaluation
            pooled = [t for t in source if t in self.pool]
            self.tickers = shard_items(pooled, parse_shard(args.shard))
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
            "renderings": [rkey(r) for r in RENDERINGS], "generate_canonical": True,
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
                    if reverse == CANON:
                        row = self.evaluator.row(fp, 0.0, None, label=key)
                        rows[key] = {k: row[k] for k in ("margin", "generated_text", "decision", "format", "finish",
                                                         "n_new_tokens", "path_class", "realized_margin",
                                                         "realized_status")}
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
    for name, run_ids, source in (("screen", args.screen_runs, construction), ("confirm", args.confirm_runs, evaluation)):
        if not run_ids:
            continue
        tickers = [t for t in source if t in pool]
        summary[name] = scan_summary(load_runs(slug, run_ids, pool_sha), pool, tickers)
        s = summary[name]
        log(f"{name}: {s['n_companies']} companies; attribute {s['attribute']}")
        log(f"  company buy fractions: {s['company_buy_fraction']['all_sell']} all-sell / "
            f"{s['company_buy_fraction']['mixed']} mixed / {s['company_buy_fraction']['all_buy']} all-buy")
        log(f"  variance explained (realized margin) {s['variance_explained_realized_margin']}; "
            f"positions {s['position_effects_realized_margin']}")
        log(f"  options order {s['options_order']}; ref {s['reference_frozen']}")
        for pattern, v in s["by_pattern"].items():
            log(f"  pattern {pattern}: buy_rate={v['generated_buy_rate']:.3f} n={v['n']} parse={v['parse_rate']:.3f}")
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
