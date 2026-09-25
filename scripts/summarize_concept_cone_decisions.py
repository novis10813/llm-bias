#!/usr/bin/env python3
"""Summarize observed alpha-0-to-alpha generated-decision transitions.

Run from the repository root:
    uv run python scripts/summarize_concept_cone_decisions.py --input <run>/result.json
    uv run python scripts/summarize_concept_cone_decisions.py --input <run>/result.json --supplementary <run>/decision_reparse.json

Writes new decision_transitions_{strict,supplementary}.{json,md} beside result.json.
The latter is post-hoc and must not replace the strict result. No model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DECISIONS = {"buy", "sell"}
TRANSITIONS = ("buy_to_buy", "buy_to_sell", "sell_to_buy", "sell_to_sell")


def summarize(path: Path, supplementary: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    original = json.loads(raw)
    metadata = original["metadata"]
    if not original.get("complete") or metadata.get("mode") != "evaluation":
        raise ValueError("requires a complete evaluation result")
    tickers = metadata["target_tickers"]
    alphas = metadata["alphas"]
    if (not tickers or len(set(tickers)) != len(tickers) or set(original["targets"]) != set(tickers)
            or not alphas or alphas[0] != 0 or len(set(alphas)) != len(alphas)):
        raise ValueError("invalid target or alpha grid")
    digest = hashlib.sha256(raw).hexdigest()
    mode = "strict" if supplementary is None else "supplementary"
    other = None
    other_digest = None
    if supplementary is not None:
        other_raw = supplementary.read_bytes()
        other_digest = hashlib.sha256(other_raw).hexdigest()
        other = json.loads(other_raw)
        if (other.get("schema") != "concept-cone-decision-reparse-v1"
                or other.get("source_sha256") != digest
                or other.get("tickers") != tickers or other.get("alphas") != alphas
                or other.get("model_slug") != metadata["model_slug"]
                or other.get("layer") != metadata["layer"] or set(other["targets"]) != set(tickers)):
            raise ValueError("supplementary decision source does not match original run")

    by_ticker: dict[str, list[str | None]] = {}
    for ticker in tickers:
        rows = original["targets"][ticker]["cone_centroid"]
        if len(rows) != len(alphas) or [r["alpha"] for r in rows] != alphas:
            raise ValueError(f"incomplete original alpha grid for {ticker}")
        if other is None:
            if any(not isinstance(r["parse_ok"], bool) for r in rows):
                raise ValueError(f"strict parse status missing for {ticker}")
            decisions = [r["decision"] if r["parse_ok"] else None for r in rows]
        else:
            derived_rows = other["targets"][ticker]
            if len(derived_rows) != len(alphas) or [r["alpha"] for r in derived_rows] != alphas:
                raise ValueError(f"incomplete supplementary alpha grid for {ticker}")
            decisions = [r["decision"] for r in derived_rows]
        if any(d is not None and d not in DECISIONS for d in decisions):
            raise ValueError(f"invalid generated decision for {ticker}")
        by_ticker[ticker] = decisions

    baseline_buy = sum(by_ticker[t][0] == "buy" for t in tickers)
    baseline_sell = sum(by_ticker[t][0] == "sell" for t in tickers)
    summary = []
    for j, alpha in enumerate(alphas):
        counts = dict.fromkeys(TRANSITIONS, 0)
        current_parsed = 0
        for ticker in tickers:
            base, current = by_ticker[ticker][0], by_ticker[ticker][j]
            current_parsed += current is not None
            if base is not None and current is not None:
                counts[f"{base}_to_{current}"] += 1
        paired_buy = counts["buy_to_buy"] + counts["buy_to_sell"]
        paired_sell = counts["sell_to_sell"] + counts["sell_to_buy"]
        paired = paired_buy + paired_sell
        flipped = counts["buy_to_sell"] + counts["sell_to_buy"]
        summary.append({"alpha": alpha, "n": len(tickers), "alpha_parsed": current_parsed,
                        "baseline_buy": baseline_buy, "baseline_sell": baseline_sell,
                        "paired": paired, "paired_baseline_buy": paired_buy,
                        "paired_baseline_sell": paired_sell, "transitions": counts,
                        "flipped": flipped,
                        "flip_rate": flipped / paired if paired else None,
                        "buy_to_sell_rate": counts["buy_to_sell"] / paired_buy if paired_buy else None,
                        "sell_to_buy_rate": counts["sell_to_buy"] / paired_sell if paired_sell else None})

    result = {"schema": "concept-cone-decision-transitions-v1", "decision_source": mode,
              "interpretation": "paired generated alpha-0 vs alpha decision; never classify by margin or C2 T",
              "source": str(path), "source_sha256": digest,
              "supplementary_source": str(supplementary) if supplementary else None,
              "supplementary_sha256": other_digest,
              "model_slug": metadata["model_slug"], "layer": metadata["layer"],
              "alphas": alphas, "summary": summary}
    label = "post-hoc supplementary" if other is not None else "original strict JSON"
    headers = " | ".join(f"alpha={alpha:g}" for alpha in alphas)
    lines = [f"# {metadata['model_slug']} generated-decision transitions ({label})", "",
             "Classes are generated buy/sell at alpha 0, not margin sign or C2 T.",
             "Only companies with parseable decisions at both alpha 0 and this alpha enter paired denominators.",
             "A missing baseline class makes its directional rate undefined, not 0%.",
             f"Source: `{path}` (SHA-256 `{digest}`)."]
    if supplementary is not None:
        lines.append(f"Supplementary parser: `{supplementary}` (SHA-256 `{other_digest}`).")
    lines.extend(["", f"| count / rate | {headers} |", "|---|" + "---|" * len(alphas)])
    def add(label: str, values: list[str]) -> None:
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    for title, key in (("alpha parsed", "alpha_parsed"), ("baseline buy", "baseline_buy"),
                       ("baseline sell", "baseline_sell"), ("valid pairs", "paired"),
                       ("paired baseline buy", "paired_baseline_buy"),
                       ("paired baseline sell", "paired_baseline_sell")):
        add(title, [f"{s[key]}/{s['n']}" for s in summary])
    for title, key in (("buy→buy", "buy_to_buy"), ("buy→sell", "buy_to_sell"),
                       ("sell→buy", "sell_to_buy"), ("sell→sell", "sell_to_sell")):
        add(title, [str(s["transitions"][key]) for s in summary])
    add("total flips / valid pairs", [f"{s['flipped']}/{s['paired']}" if s["paired"] else "— (0 pairs)" for s in summary])
    add("buy→sell / paired baseline buy", [
        f"{s['transitions']['buy_to_sell']}/{s['paired_baseline_buy']}" if s["paired_baseline_buy"] else "— (0 buy)"
        for s in summary])
    add("sell→buy / paired baseline sell", [
        f"{s['transitions']['sell_to_buy']}/{s['paired_baseline_sell']}" if s["paired_baseline_sell"] else "— (0 sell)"
        for s in summary])
    return result, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path, help="Matching decision_reparse.json; post-hoc")
    args = parser.parse_args()
    mode = "supplementary" if args.supplementary else "strict"
    output = args.input.with_name(f"decision_transitions_{mode}.json")
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("transition output exists; original and previous analyses are never overwritten")
    result, table = summarize(args.input, args.supplementary)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(table, encoding="utf-8")
    print(f"Wrote {output} and {markdown}")
    for row in result["summary"]:
        c = row["transitions"]
        buy_rate = f"{c['buy_to_sell']}/{row['paired_baseline_buy']}" if row['paired_baseline_buy'] else "undefined (0 buy)"
        sell_rate = f"{c['sell_to_buy']}/{row['paired_baseline_sell']}" if row['paired_baseline_sell'] else "undefined (0 sell)"
        total_rate = f"{row['flipped']}/{row['paired']}" if row['paired'] else "undefined (0 pairs)"
        print(f"  alpha={row['alpha']:g}: buy→sell {buy_rate}, sell→buy {sell_rate}; total {total_rate}")


if __name__ == "__main__":
    main()
