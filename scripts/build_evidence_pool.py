"""Derive the per-company evidence pool from the converted baseline trial plan (evidence-scan-v1 input).

Source: ``data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv`` (converted from the external baseline
project; that project's generator is not in this repo). Each company has a pool of eight evidence items (four
that argue for a price increase, four for a decrease), and the ``attribute`` condition lists random two-plus-two
combinations of the pool in random order. The polarity of the pool is recovered from those combinations: exactly
one split of the eight items into 4 + 4 makes every ``attribute`` prompt contain two of each, and the side is
signed by counting price-increase versus price-decrease wording.

Output: ``evidence_pool.json`` with, per company, the eight items (with polarity) and the ordered attribute
combinations as indices into the items. Fails closed if the split is not unique up to swapping the two sides.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

SCHEMA = "evidence-pool-v1"
PREFIX = "prompt_with_context_attribute_"
PCT = r"\d+(?:\.\d+)?%"
UP_WORDS = (r"(?:increase|rise|rises|gain|gains|upside|rally|appreciation|higher|rebound|jump|uplift|upward|"
            r"appreciate|climb|advance|growth|outperform|re-rating|expansion)")
DOWN_WORDS = (r"(?:decrease|decline|drop|fall|falls|downside|correction|lower|reduction|pullback|downward|"
              r"depreciation|slide|erosion|contraction|derating|de-rating|underperform|loss)")
UP = re.compile(rf"(?:{PCT}\s+(?:price\s+|stock\s+)?{UP_WORDS}|{UP_WORDS}\s+(?:of|by|in)\s+"
                rf"(?:approximately\s+|about\s+|roughly\s+)?{PCT})", re.I)
DOWN = re.compile(rf"(?:{PCT}\s+(?:price\s+|stock\s+)?{DOWN_WORDS}|{DOWN_WORDS}\s+(?:of|by|in)\s+"
                  rf"(?:approximately\s+|about\s+|roughly\s+)?{PCT})", re.I)
MIN_SIGN_MARGIN = 2


def evidence_items(prompt: str) -> list[str]:
    return re.findall(r"^\d+\. (.+)$", prompt, re.M)


def wording_score(text: str) -> int:
    return len(UP.findall(text)) - len(DOWN.findall(text))


def build_company(row: dict[str, str]) -> dict:
    sets = [evidence_items(row[c]) for c in row if c.startswith(PREFIX) and row[c]]
    pool = sorted({item for s in sets for item in s})
    if len(pool) != 8 or any(len(s) != 4 for s in sets):
        raise ValueError(f"{row['ticker']}: expected an 8-item pool and 4-item attribute prompts")
    splits = [set(pool[i] for i in a) for a in itertools.combinations(range(8), 4)
              if all(sum(item in {pool[i] for i in a} for item in s) == 2 for s in sets)]
    if len(splits) != 2 or splits[0] & splits[1]:
        raise ValueError(f"{row['ticker']}: polarity split is not unique ({len(splits)} candidates)")
    side = splits[0]
    margin = sum(wording_score(i) for i in side) - sum(wording_score(i) for i in set(pool) - side)
    if abs(margin) < MIN_SIGN_MARGIN:
        raise ValueError(f"{row['ticker']}: wording margin {margin} too small to sign the split")
    positive = side if margin > 0 else set(pool) - side
    items = [{"text": text, "polarity": "+" if text in positive else "-"} for text in pool]
    index = {item["text"]: k for k, item in enumerate(items)}
    return {"name_hint": row["name"], "sector": row["sector"], "items": items,
            "attribute_combos": [[index[i] for i in s] for s in sets], "sign_margin": abs(margin)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    csv.field_size_limit(sys.maxsize)
    source = Path(args.source)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    companies = {row["ticker"]: build_company(row) for row in rows}
    if len(companies) != len(rows):
        raise ValueError("duplicate tickers in the source")
    payload = {"schema": SCHEMA, "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "n_companies": len(companies), "companies": companies}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    weak = sorted(t for t, c in companies.items() if c["sign_margin"] < 3)
    print(f"{len(companies)} companies; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}; "
          f"weakly signed (margin < 3, checked by hand): {weak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
