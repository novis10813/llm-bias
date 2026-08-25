#!/usr/bin/env python
"""Render a Markdown report from jspace TF-IDF keyword tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "summary.json").read_text())
    sector_rows = [
        json.loads(line)
        for line in (args.run_dir / "sector_keywords.jsonl").open()
    ]
    company_rows = [
        json.loads(line)
        for line in (args.run_dir / "company_keywords.jsonl").open()
    ]
    params = summary["params"]
    band = ", ".join(str(l) for l in params["band_layers"])

    def is_fragment(token: str) -> bool:
        return token.startswith(("-", "/")) or "/" in token

    by_sector: dict[str, list[dict]] = defaultdict(list)
    for row in sector_rows:
        by_sector[row["document"]].append(row)

    lines: list[str] = []
    lines.append("# Workspace-band token analysis (TF-IDF)")
    lines.append("")
    lines.append(
        f"Source: `{params['weighting']}`. Band layers **{band}**; motor contrast "
        f"L{params['motor_layer']}. {summary['records_seen']} records, "
        f"{summary['unique_companies']} companies, {summary['sectors']} sectors."
    )
    lines.append("")
    lines.append("> Descriptive readout statistics only — the Jacobian lens is a "
                 "transported-representation readout, not an attention map or causal claim.")
    lines.append("")

    # ------------------------------------------------------------- scope 1
    lines.append("## Scope 1: industry-distinctive tokens")
    lines.append("")
    lines.append("| sector | top keywords (TF-IDF) | high GWS-specificity (>3x band/motor) |")
    lines.append("|---|---|---|")
    for sector in sorted(by_sector):
        rows = by_sector[sector]
        core = [r for r in rows if not is_fragment(r["token"])][: args.top_n]
        specific = [r for r in rows if r["gws_specificity"] > 3 and not is_fragment(r["token"])][
            : args.top_n
        ]

        def fmt(rows_: list[dict], key: str) -> str:
            return ", ".join(f"{r['token']} ({r[key]:.4g})" for r in rows_) or "—"

        lines.append(f"| **{sector}** | {fmt(core, 'tfidf')} | {fmt(specific, 'gws_specificity')} |")
    lines.append("")

    # ------------------------------------------------------------- scope 2
    by_company: dict[str, list[dict]] = defaultdict(list)
    for row in company_rows:
        by_company[row["document"]].append(row)
    by_sector_companies: dict[str, list[str]] = defaultdict(list)
    for ticker in sorted(by_company):
        sector = by_company[ticker][0]["sector"]
        by_sector_companies[sector].append(ticker)

    lines.append("## Scope 2: company-distinctive tokens within industry")
    lines.append("")
    lines.append(f"Tokens ranked by log-lift vs industry share; only tokens appearing "
                 f"in >= {params.get('min_record_df', 3)} of the company's records.")
    lines.append("")
    for sector in sorted(by_sector_companies):
        lines.append(f"### {sector}")
        lines.append("")
        for ticker in by_sector_companies[sector]:
            rows = by_company[ticker][: args.top_n]
            tokens = ", ".join(
                f"{r['token']} ({r['industry_lift_log']:.1f})" for r in rows if not is_fragment(r["token"])
            )
            name = rows[0]["name"] if rows else ""
            lines.append(f"- **{ticker}** ({name}): {tokens or '—'}")
        lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
