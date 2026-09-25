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

FORMATS = ("bare_json", "fenced_json", "thought_bare_json", "thought_fenced_json")


def parse_complete_decision(text: str) -> tuple[str | None, str]:
    """Accept a whole two-key JSON object behind at most exact observed wrappers."""
    if not isinstance(text, str):
        return None, "invalid"
    body = text.strip()
    thought = body.startswith("thought\n")
    if thought:
        body = body[len("thought\n"):]
    fenced = body.startswith("```json\n")
    if fenced:
        if not body.endswith("\n```"):
            return None, "invalid"
        body = body[len("```json\n"):-len("\n```")]
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None, "invalid"
    if (not isinstance(obj, dict) or set(obj) != {"decision", "reason"}
            or obj["decision"] not in ("buy", "sell")
            or not isinstance(obj["reason"], str) or not obj["reason"].strip()):
        return None, "invalid"
    kind = ("thought_" if thought else "") + ("fenced_json" if fenced else "bare_json")
    return obj["decision"], kind


def reparse_dim_run(path: Path, raw: bytes, original: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Supplementary full-object decisions for a complete crossmodel DIM layer grid."""
    metadata = original["metadata"]
    layers = metadata["layers"]
    alphas = metadata["alphas"]
    tickers = metadata["target_tickers"]
    if (not original.get("complete") or metadata.get("mode") != "evaluation"
            or metadata.get("schema") not in ("dim-tokenwise-crossmodel-v1", "dim-single-all-crossmodel-v1")
            or not isinstance(layers, list) or not layers or len(layers) != len(set(layers))
            or not isinstance(tickers, list) or len(tickers) != 101 or len(set(tickers)) != 101
            or not isinstance(alphas, list) or alphas != [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            or set(original.get("targets", {})) != set(tickers)):
        raise ValueError("requires a complete crossmodel DIM 101-company layer grid")
    parsed: dict[str, dict[str, Any]] = {}
    format_counts: Counter[str] = Counter()
    for ticker in tickers:
        by_layer = original["targets"][ticker]
        if set(by_layer) != {f"L{layer}" for layer in layers}:
            raise ValueError(f"incomplete DIM layer grid for {ticker}")
        parsed[ticker] = {}
        for layer in layers:
            key = f"L{layer}"
            rows = by_layer[key]["rows"]
            if len(rows) != len(alphas) or [row["alpha"] for row in rows] != alphas:
                raise ValueError(f"incomplete alpha grid for {ticker}/{key}")
            derived = []
            for row in rows:
                text = row["generated_text"]
                try:
                    strict_obj = json.loads(text)
                    strict_decision = strict_obj.get("decision") if isinstance(strict_obj, dict) else None
                except (ValueError, TypeError):
                    strict_decision = None
                if strict_decision not in ("buy", "sell"):
                    strict_decision = "unparsed"
                if (row.get("parse_ok") is not (strict_decision != "unparsed")
                        or row.get("decision") != strict_decision):
                    raise ValueError(f"strict decision mismatch for {ticker}/{key}/{row['alpha']}")
                decision, kind = parse_complete_decision(text)
                format_counts[kind] += 1
                derived.append({"alpha": row["alpha"], "decision": decision,
                                "format": kind, "strict_parse_ok": row["parse_ok"]})
            parsed[ticker][key] = {"rows": derived}

    summaries = {}
    for layer in layers:
        name = f"L{layer}"
        baseline = [parsed[t][name]["rows"][0] for t in tickers]
        base_buy = sum(b["decision"] == "buy" for b in baseline)
        base_sell = sum(b["decision"] == "sell" for b in baseline)
        summaries[str(layer)] = []
        for j, alpha in enumerate(alphas):
            rows = [parsed[t][name]["rows"][j] for t in tickers]
            pairs = [(b["decision"], r["decision"]) for b, r in zip(baseline, rows, strict=True)
                     if b["decision"] in ("buy", "sell") and r["decision"] in ("buy", "sell")]
            buy_n = sum(b == "buy" for b, _ in pairs)
            sell_n = sum(b == "sell" for b, _ in pairs)
            buy_to_sell = sum(b == "buy" and r == "sell" for b, r in pairs)
            sell_to_buy = sum(b == "sell" and r == "buy" for b, r in pairs)
            summaries[str(layer)].append({
                "alpha": alpha, "n": len(tickers), "strict_parsed": sum(r["strict_parse_ok"] for r in rows),
                "reparsed": sum(r["decision"] is not None for r in rows),
                "baseline_buy_n": base_buy, "baseline_sell_n": base_sell,
                "valid_flip_pairs": len(pairs), "buy_valid_pairs": buy_n, "sell_valid_pairs": sell_n,
                "buy_to_sell": buy_to_sell, "sell_to_buy": sell_to_buy,
                "buy_to_sell_rate": buy_to_sell / buy_n if buy_n else None,
                "sell_to_buy_rate": sell_to_buy / sell_n if sell_n else None,
                "flipped": buy_to_sell + sell_to_buy,
                "flip_rate": (buy_to_sell + sell_to_buy) / len(pairs) if pairs else None,
            })
    result = {"schema": "dim-crossmodel-decision-reparse-v1",
              "analysis_status": "supplementary complete-object parse; strict primary result unchanged",
              "source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
              "allowed_formats": list(FORMATS), "format_counts": dict(sorted(format_counts.items())),
              "model_slug": metadata["model_slug"], "dim_arm": metadata["dim_arm"],
              "layers": layers, "alphas": alphas, "tickers": tickers,
              "summary": summaries, "targets": parsed}
    headers = " | ".join(f"alpha={a:g}" for a in alphas)
    lines = [f"# {metadata['model_slug']} {metadata['dim_arm']} — DIM generated decision reparse (supplementary)", "",
             "Complete two-key objects only; strict full-JSON decisions and margins remain in the original result.",
             f"Source: `{path}` (SHA-256 `{result['source_sha256']}`).", "",
             f"| layer / measure | {headers} |", "|---|" + "---|" * len(alphas)]
    for layer in layers:
        summary = summaries[str(layer)]
        for label, value in (("strict parsed", "strict_parsed"), ("complete-object parsed", "reparsed"),
                             ("valid paired", "valid_flip_pairs"), ("sell→buy", "sell_to_buy"),
                             ("buy→sell", "buy_to_sell")):
            lines.append(f"| L{layer} {label} | " + " | ".join(
                f"{s[value]}/{s['n']}" if value in ("strict_parsed", "reparsed") else
                f"{s[value]}/{s['sell_valid_pairs'] if value == 'sell_to_buy' else s['buy_valid_pairs']}"
                if value in ("sell_to_buy", "buy_to_sell") and
                (s['sell_valid_pairs'] if value == 'sell_to_buy' else s['buy_valid_pairs']) else
                "—" if value in ("sell_to_buy", "buy_to_sell") else str(s[value]) for s in summary) + " |")
    return result, "\n".join(lines) + "\n"


def reparse_run(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    original = json.loads(raw)
    metadata = original["metadata"]
    if metadata.get("schema") in ("dim-tokenwise-crossmodel-v1", "dim-single-all-crossmodel-v1"):
        return reparse_dim_run(path, raw, original)
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
