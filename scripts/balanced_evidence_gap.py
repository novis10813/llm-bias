"""Balanced Evidence Gap — Phase 1 behavioral confirmation operator.

Runs a 3-stage pipeline (prepare → forward → analyze) that measures
whether identical balanced financial evidence paired with different company
names produces a measurable decision margin gap on Qwen3.5-4B.

Usage
-----
Smoke preflight (no output files written):
    python scripts/balanced_evidence_gap.py --smoke \\
        --model .cache/models/qwen3.5-4b

Formal run:
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\
    uv run --no-sync python scripts/balanced_evidence_gap.py \\
        --model .cache/models/qwen3.5-4b \\
        --run-id balanced-gap-gpu-bf16-01

See docs/balanced-evidence-gap/details/proposal-phase1.md for full protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time
import uuid

# ── constants ────────────────────────────────────────────────────────────────

INPUT_PATH = pathlib.Path("data/baseline/investment-dial/exploratory-v1.json")
ARTIFACTS_ROOT = pathlib.Path("artifacts")
MODEL_SLUG = "qwen3.5-4b"
SCHEMA_VERSION = "balanced-evidence-gap-v1"

# 16 target companies from test split, 4 per sector
TARGET_TICKERS: dict[str, list[str]] = {
    "Information Technology": ["AMAT", "GLW", "HPE", "IT"],
    "Financials":             ["AXP", "BLK", "C",   "GS"],
    "Health Care":            ["ABT", "BDX", "DHR", "SYK"],
    "Industrials":            ["CSX", "DE",  "HON", "NSC"],
}
ALL_TICKERS: list[str] = [t for ts in TARGET_TICKERS.values() for t in ts]
SECTOR_OF: dict[str, str] = {t: s for s, ts in TARGET_TICKERS.items() for t in ts}

POSITIVE_COUNT = 2   # balanced: 2 positive + 2 negative evidence items
REPEATS = 4          # evidence re-samples per company
SEED = 42
PREFIX = '{"decision": "'

ANON_TICKER = "TICKER"
ANON_NAME = "Company X"


# ── prompt construction ───────────────────────────────────────────────────────

def build_prompt(ticker: str, name: str, evidence: list[str], reverse: bool) -> str:
    options = '"sell" or "buy"' if reverse else '"buy" or "sell"'
    items = "\n".join(f"- {item}" for item in evidence)
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"Stock Ticker: [{ticker}]\n\nStock Name: [{name}]\n\n"
        "— Evidence —\n\n"
        f"{items}\n\n"
        "—\n\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        f'"decision": {options}\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


def stage_prepare(companies: dict, smoke: bool) -> list[dict]:
    """Build all named + anonymous prompts. Returns list of row dicts."""
    rows: list[dict] = []

    for ticker in (ALL_TICKERS[:1] if smoke else ALL_TICKERS):
        company = companies[ticker]
        name = company["name"]
        sector = SECTOR_OF[ticker]
        pairs = company["evidence_pairs"]

        rng = random.Random(f"{SEED}:{ticker}")
        for repeat in range(1 if smoke else REPEATS):
            sampled = rng.sample(pairs, 4)
            order = list(range(4))
            rng.shuffle(order)

            pos_items = [sampled[i]["positive"] for i in range(POSITIVE_COUNT)]
            neg_items = [sampled[i]["negative"] for i in range(4 - POSITIVE_COUNT)]
            items_ordered = [pos_items + neg_items][0]
            evidence = [items_ordered[i] for i in order]

            for reverse in ([False] if smoke else [False, True]):
                base_id = f"{ticker}:{repeat}:{int(reverse)}"
                # named
                rows.append({
                    "id": f"named:{base_id}",
                    "condition": "named",
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "repeat": repeat,
                    "reverse": reverse,
                    "positive_count": POSITIVE_COUNT,
                    "prompt": build_prompt(ticker, name, evidence, reverse),
                })
                # anonymous
                rows.append({
                    "id": f"anon:{base_id}",
                    "condition": "anonymous",
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "repeat": repeat,
                    "reverse": reverse,
                    "positive_count": POSITIVE_COUNT,
                    "prompt": build_prompt(ANON_TICKER, ANON_NAME, evidence, reverse),
                })

    return rows


# ── forward inference ─────────────────────────────────────────────────────────

def stage_forward(rows: list[dict], model_path: str) -> list[dict]:
    """Run forward inference; returns rows augmented with margin + decision."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # resolve buy/sell token IDs (must be single tokens after JSON prefix)
    def _get_token_id(word: str) -> int:
        full = PREFIX + word
        ids = tok.encode(full, add_special_tokens=False)
        prefix_ids = tok.encode(PREFIX, add_special_tokens=False)
        tail = ids[len(prefix_ids):]
        if len(tail) != 1:
            raise RuntimeError(f"'{word}' tokenises to {tail} (not single token) after prefix")
        return tail[0]

    buy_id = _get_token_id("buy")
    sell_id = _get_token_id("sell")
    if buy_id == sell_id:
        raise RuntimeError("buy and sell map to same token")

    results = []
    for i, row in enumerate(rows):
        # Apply chat template
        messages = [{"role": "user", "content": row["prompt"]}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        decision_text = text + PREFIX
        input_ids_dec = tok.encode(decision_text, add_special_tokens=False)
        inputs = torch.tensor([input_ids_dec]).to(model.device)

        with torch.no_grad():
            out = model(inputs)
        logits = out.logits[0, -1].float()

        buy_logit = logits[buy_id].item()
        sell_logit = logits[sell_id].item()
        margin = buy_logit - sell_logit
        decision = "buy" if margin > 0 else "sell"

        results.append(row | {
            "buy_logit": buy_logit,
            "sell_logit": sell_logit,
            "margin": margin,
            "decision": decision,
            "buy_id": buy_id,
            "sell_id": sell_id,
        })

        if (i + 1) % 20 == 0:
            print(f"  forward {i+1}/{len(rows)} — {row['ticker']} {row['condition']} "
                  f"repeat={row['repeat']} margin={margin:.3f} [{decision}]", flush=True)

    return results


# ── analysis ──────────────────────────────────────────────────────────────────

def stage_analyze(results: list[dict]) -> dict:
    """Compute per-company margins, named-vs-anon gap, sector summary."""
    import statistics

    named = {r["id"].split(":")[1]: r for r in results if r["condition"] == "named"}
    anon  = {r["id"].split(":")[1]: r for r in results if r["condition"] == "anonymous"}

    per_company: dict[str, dict] = {}
    gaps: list[float] = []

    for ticker in ALL_TICKERS:
        n_rows = [r for r in results if r["condition"] == "named" and r["ticker"] == ticker]
        a_rows = [r for r in results if r["condition"] == "anonymous" and r["ticker"] == ticker]
        if not n_rows:
            continue

        n_margins = [r["margin"] for r in n_rows]
        a_margins = [r["margin"] for r in a_rows]
        pair_gaps = [n["margin"] - a["margin"]
                     for n, a in zip(
                         sorted(n_rows, key=lambda r: (r["repeat"], r["reverse"])),
                         sorted(a_rows, key=lambda r: (r["repeat"], r["reverse"]))
                     )]
        gaps.extend(pair_gaps)

        per_company[ticker] = {
            "sector": SECTOR_OF[ticker],
            "named_margin_mean": statistics.mean(n_margins),
            "named_margin_median": statistics.median(n_margins),
            "anon_margin_mean": statistics.mean(a_margins) if a_margins else None,
            "gap_mean": statistics.mean(pair_gaps) if pair_gaps else None,
            "named_decisions": [r["decision"] for r in n_rows],
            "anon_decisions": [r["decision"] for r in a_rows],
            "decision_flips": sum(
                1 for n, a in zip(
                    sorted(n_rows, key=lambda r: (r["repeat"], r["reverse"])),
                    sorted(a_rows, key=lambda r: (r["repeat"], r["reverse"]))
                ) if n["decision"] != a["decision"]
            ),
        }

    # cross-company spread of named median margins
    named_medians = [v["named_margin_median"] for v in per_company.values()]
    spread_iqr = (
        sorted(named_medians)[int(0.75 * len(named_medians))]
        - sorted(named_medians)[int(0.25 * len(named_medians))]
    ) if len(named_medians) >= 4 else None

    # sector summaries
    sector_summary: dict[str, dict] = {}
    for sector, tickers in TARGET_TICKERS.items():
        medians = [per_company[t]["named_margin_median"]
                   for t in tickers if t in per_company]
        sector_summary[sector] = {
            "named_margin_mean": statistics.mean(medians) if medians else None,
        }

    # named-vs-anon gap bootstrap CI (simple percentile)
    gap_ci = None
    if len(gaps) >= 4:
        boot = [statistics.mean(random.choices(gaps, k=len(gaps))) for _ in range(2000)]
        boot.sort()
        gap_ci = {"lower": boot[50], "upper": boot[1950]}

    total_flips = sum(v["decision_flips"] for v in per_company.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "n_companies": len(per_company),
        "n_prompts": len(results),
        "cross_company_named_margin_iqr": spread_iqr,
        "global_gap_mean": statistics.mean(gaps) if gaps else None,
        "global_gap_ci_95": gap_ci,
        "total_named_vs_anon_decision_flips": total_flips,
        "per_company": per_company,
        "sector_summary": sector_summary,
        "descriptive_pass_criteria": {
            "cross_company_iqr_gt_0p5": (spread_iqr or 0) > 0.5,
            "gap_ci_excludes_zero": (
                gap_ci is not None
                and (gap_ci["lower"] > 0 or gap_ci["upper"] < 0)
            ),
            "ge_2_decision_flips": total_flips >= 2,
        },
    }


# ── manifest ──────────────────────────────────────────────────────────────────

def write_manifest(run_root: pathlib.Path, run_id: str, model_path: str,
                   started_at: str, elapsed_s: float) -> None:
    import hashlib

    def sha256(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    artifacts = {}
    for p in sorted(run_root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            artifacts[str(p.relative_to(run_root))] = sha256(p)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_path": model_path,
        "started_at": started_at,
        "elapsed_seconds": round(elapsed_s, 1),
        "status": "complete",
        "artifacts": artifacts,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint (e.g. .cache/models/qwen3.5-4b)")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier (default: auto-generated timestamp)")
    parser.add_argument("--smoke", action="store_true",
                        help="Preflight: 1 company, 1 repeat, no output files written")
    args = parser.parse_args()

    # validate input
    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: input not found: {INPUT_PATH}")
    raw = json.loads(INPUT_PATH.read_text())
    companies = {c["ticker"]: c for c in raw["companies"]}
    missing = [t for t in ALL_TICKERS if t not in companies]
    if missing:
        sys.exit(f"ERROR: tickers missing from input: {missing}")
    wrong_split = [t for t in ALL_TICKERS if companies[t]["split"] != "test"]
    if wrong_split:
        sys.exit(f"ERROR: tickers not in test split: {wrong_split}")

    run_id = args.run_id or f"balanced-gap-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    # ── prepare ──
    print(f"[prepare] building prompts (smoke={args.smoke}) …")
    rows = stage_prepare(companies, smoke=args.smoke)
    print(f"[prepare] {len(rows)} prompts ready")

    if args.smoke:
        print("[smoke] running forward on 2 prompts …")
        results = stage_forward(rows[:2], args.model)
        for r in results:
            print(f"  {r['id']:30s}  margin={r['margin']:+.3f}  [{r['decision']}]")
        print("[smoke] PASS — no output files written")
        return

    # ── set up run directory ──
    run_root = ARTIFACTS_ROOT / MODEL_SLUG / "balanced-evidence-gap" / "runs" / run_id
    prepare_dir = run_root / "prepare"
    forward_dir = run_root / "forward"
    analyze_dir = run_root / "analyze"
    for d in (prepare_dir, forward_dir, analyze_dir):
        d.mkdir(parents=True, exist_ok=True)

    # write prompts
    prompts_path = prepare_dir / "prompts.jsonl"
    with prompts_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare] wrote {prompts_path}")

    # ── forward ──
    print(f"[forward] running inference on {len(rows)} prompts …")
    results = stage_forward(rows, args.model)
    results_path = forward_dir / "results.jsonl"
    with results_path.open("w") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "prompt"}  # omit raw prompt text
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[forward] wrote {results_path}")

    # ── analyze ──
    print("[analyze] computing summary …")
    summary = stage_analyze(results)
    summary_path = analyze_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[analyze] wrote {summary_path}")

    # print quick verdict
    criteria = summary["descriptive_pass_criteria"]
    print("\n── Phase 1 descriptive criteria ──")
    print(f"  cross-company IQR > 0.5 nats : {criteria['cross_company_iqr_gt_0p5']}"
          f"  (IQR = {summary['cross_company_named_margin_iqr']:.3f})")
    print(f"  gap CI excludes zero         : {criteria['gap_ci_excludes_zero']}"
          f"  (CI = {summary['global_gap_ci_95']})")
    print(f"  ≥2 named/anon decision flips : {criteria['ge_2_decision_flips']}"
          f"  (flips = {summary['total_named_vs_anon_decision_flips']})")

    # ── manifest ──
    elapsed = time.time() - t0
    write_manifest(run_root, run_id, args.model, started_at, elapsed)
    print(f"\n[done] run_id={run_id}  elapsed={elapsed:.0f}s")
    print(f"       root: {run_root}")


if __name__ == "__main__":
    main()
