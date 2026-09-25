#!/usr/bin/env python3
"""Supplementary, complete-object decision parsing for persisted concept-cone runs.

Run from the repository root:
    uv run python scripts/reparse_concept_cone_decisions.py --input artifacts/<model>/concept-cone-steering/runs/<run>/result.json

Writes decision_reparse.{json,md} beside the input; never edits the original run.
This post-hoc analysis does NOT replace the original strict JSON parse rate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from llm_bias.core.decision_parsing import FORMATS, parse_complete_decision


def reparse_run(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    original = json.loads(raw)
    metadata = original["metadata"]
    if not original.get("complete") or metadata.get("mode") != "evaluation":
        raise ValueError("requires a complete evaluation result")
    alphas = metadata["alphas"]
    tickers = metadata["target_tickers"]
    if not alphas or not tickers or len(set(tickers)) != len(tickers) or set(original["targets"]) != set(tickers):
        raise ValueError("invalid alpha or target grid")
    if alphas[0] != 0 or len(set(alphas)) != len(alphas):
        raise ValueError("alpha 0 baseline required without duplicates")

    parsed: dict[str, list[dict[str, Any]]] = {}
    format_counts: Counter[str] = Counter()
    for ticker in tickers:
        rows = original["targets"][ticker]["cone_centroid"]
        if len(rows) != len(alphas) or [row["alpha"] for row in rows] != alphas:
            raise ValueError(f"incomplete alpha grid for {ticker}")
        parsed[ticker] = []
        for row in rows:
            decision, kind = parse_complete_decision(row["generated_text"])
            strict = row["parse_ok"]
            if not isinstance(strict, bool) or (strict and row["decision"] != decision):
                raise ValueError(f"strict decision mismatch for {ticker}, alpha {row['alpha']}")
            format_counts[kind] += 1
            parsed[ticker].append({"alpha": row["alpha"], "decision": decision,
                                   "format": kind, "strict_parse_ok": strict})

    summary = []
    for j, alpha in enumerate(alphas):
        rows = [parsed[t][j] for t in tickers]
        base = [parsed[t][0] for t in tickers]
        pairs = [(b, r) for b, r in zip(base, rows, strict=True)
                 if b["decision"] is not None and r["decision"] is not None]
        n = len(tickers)
        flips = sum(b["decision"] != r["decision"] for b, r in pairs)
        summary.append({"alpha": alpha, "n": n,
                        "strict_parsed": sum(row["strict_parse_ok"] for row in rows),
                        "reparsed": sum(row["decision"] is not None for row in rows),
                        "valid_flip_pairs": len(pairs), "flipped": flips,
                        "flip_rate": flips / len(pairs) if pairs else None})

    result = {"schema": "concept-cone-decision-reparse-v1",
              "analysis_status": "post-hoc supplementary; original strict parse unchanged",
              "source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
              "allowed_formats": list(FORMATS), "format_counts": dict(sorted(format_counts.items())),
              "model_slug": metadata["model_slug"], "layer": metadata["layer"],
              "alphas": alphas, "tickers": tickers, "summary": summary, "targets": parsed}

    headers = " | ".join(f"alpha={a:g}" for a in alphas)
    lines = [f"# {metadata['model_slug']} — generated decision reparse (supplementary)", "",
             "Original strict JSON results remain unchanged. Cells are fixed-prefix margin (complete-object decision).",
             f"Source: `{path}` (SHA-256 `{result['source_sha256']}`).", "",
             f"| measure | {headers} |", "|---|" + "---|" * len(alphas)]
    for label, key in (("strict JSON parsed", "strict_parsed"), ("reparsed", "reparsed"),
                       ("paired denominators", "valid_flip_pairs"), ("flips", "flipped")):
        lines.append("| " + label + " | " + " | ".join(f"{s[key]}/{s['n']}" for s in summary) + " |")
    lines.extend(["", f"| ticker | {headers} |", "|---|" + "---|" * len(alphas)])
    for ticker in tickers:
        original_rows = original["targets"][ticker]["cone_centroid"]
        lines.append("| " + ticker + " | " + " | ".join(
            f"{original_row['margin']:+.3f} ({row['decision'] or 'unparsed'})"
            for original_row, row in zip(original_rows, parsed[ticker], strict=True)) + " |")
    return result, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Complete result.json from concept-cone evaluation")
    args = parser.parse_args()
    json_out = args.input.with_name("decision_reparse.json")
    md_out = json_out.with_suffix(".md")
    if args.input == json_out or json_out.exists() or md_out.exists():
        raise FileExistsError("analysis output already exists; original artifacts are never overwritten")
    result, table = reparse_run(args.input)
    json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(table, encoding="utf-8")
    print(f"Reparsed {len(result['tickers'])} targets: {json_out}, {md_out}")
    for row in result["summary"]:
        print(f"  alpha={row['alpha']:g}: strict {row['strict_parsed']}/{row['n']}, "
              f"reparsed {row['reparsed']}/{row['n']}, flips {row['flipped']}/{row['valid_flip_pairs']}")


if __name__ == "__main__":
    main()
