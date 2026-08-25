#!/usr/bin/env python
"""TF-IDF analysis of workspace-band lens readout tokens.

Reads a baseline-trial ``lens_readout.jsonl`` (lens-forward stage) and scores
which vocabulary tokens are distinctive per sector (scope 1: all industries vs
one industry) and per company within its industry (scope 2).

Token weight = readout probability, pooled over all top-k entries of the
workspace-band layers (default 14-26); layer 31 (actual model logits) is
aggregated separately as a motor/output contrast. Records are equally
weighted: each record contributes its own token-weight distribution, so
generation length does not dominate.

Scoring:
  scope 1  TF    = token weight share within the sector document
           IDF   = ln(n_sectors / df(token)) over sector documents
           score = TF * IDF; also band/motor share ratio (GWS specificity)
  scope 2  lift  = (share_in_company + eps) / (share_in_industry + eps)
           with the same band/motor contrast

Only compact aggregate tables are written; no activations or raw readouts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata


def normalize_token(text: str) -> str | None:
    """Normalise a readout token for counting; None when it should be skipped."""
    token = text.strip()
    if not token:
        return None
    token = unicodedata.normalize("NFKC", token).lower()
    if not re.search(r"[a-z]", token):
        return None  # punctuation-only, digits-only, symbols
    return token


def accumulate_layers(
    positions: list[dict],
    *,
    band_layers: set[int],
    motor_layer: int,
    counter: Counter,
    motor_counter: Counter,
) -> tuple[float, float]:
    """Fold one record's readout into token counters; returns (band, motor) mass."""
    band_mass = 0.0
    motor_mass = 0.0
    for position in positions:
        for layer in position["layers"]:
            index = layer["layer"]
            if index in band_layers:
                target, is_band = counter, True
            elif index == motor_layer:
                target, is_band = motor_counter, False
            else:
                continue
            for entry in layer["top_tokens"]:
                token = normalize_token(entry["token"])
                if token is None:
                    continue
                probability = entry["probability"]
                target[token] += probability
                if is_band:
                    band_mass += probability
                else:
                    motor_mass += probability
    return band_mass, motor_mass


def tfidf_scores(
    doc_weights: dict[str, Counter], *, min_df: int
) -> dict[str, list[dict]]:
    """Per-document keyword tables scored by TF * ln(N / df)."""
    n_docs = len(doc_weights)
    doc_totals = {key: sum(counter.values()) for key, counter in doc_weights.items()}
    doc_freq: Counter = Counter()
    for counter in doc_weights.values():
        doc_freq.update(counter.keys())

    tables: dict[str, list[dict]] = {}
    for key, counter in doc_weights.items():
        total = doc_totals[key] or 1.0
        rows = []
        for token, weight in counter.items():
            df = doc_freq[token]
            if df < min_df:
                continue
            idf = math.log(n_docs / df)
            rows.append(
                {
                    "token": token,
                    "weight": round(weight, 6),
                    "share": round(weight / total, 8),
                    "df": df,
                    "idf": round(idf, 6),
                    "tfidf": round((weight / total) * idf, 8),
                }
            )
        rows.sort(key=lambda row: row["tfidf"], reverse=True)
        tables[key] = rows
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", required=True, type=Path)
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="trial-plan CSV providing ticker -> sector/name mapping")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--band-layers", default="14,15,16,17,18,19,20,21,22,23,24,25,26")
    parser.add_argument("--motor-layer", type=int, default=31)
    parser.add_argument("--top-n", type=int, default=50,
                        help="keywords kept per document in the report tables")
    parser.add_argument("--min-df", type=int, default=3,
                        help="minimum document frequency for scope-1 scoring")
    parser.add_argument("--min-record-df", type=int, default=3,
                        help="minimum number of the company's own records a token "
                        "must appear in for scope-2 ranking")
    args = parser.parse_args()

    band_layers = {int(x) for x in args.band_layers.split(",") if x.strip()}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_meta: dict[str, dict] = {}
    with open(args.input_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ticker_meta[row["ticker"]] = {
                "sector": row["sector"],
                "name": row.get("name", ""),
                "marketcap": row.get("marketcap", ""),
            }

    company_band: dict[str, Counter] = {}
    company_motor: dict[str, Counter] = {}
    # token -> set of record ordinals it appeared in, per company
    company_token_records: dict[str, dict[str, set[int]]] = {}
    records_seen = 0
    with open(args.readout, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ticker = row.get("ticker") or "UNKNOWN"
            # per-record counters, then equal-weight normalisation before
            # merging, so generation length does not dominate
            record_band: Counter = Counter()
            record_motor: Counter = Counter()
            band_mass, motor_mass = accumulate_layers(
                row["positions"],
                band_layers=band_layers,
                motor_layer=args.motor_layer,
                counter=record_band,
                motor_counter=record_motor,
            )
            if band_mass <= 0:
                continue
            band_counter = company_band.setdefault(ticker, Counter())
            motor_counter = company_motor.setdefault(ticker, Counter())
            for token, value in record_band.items():
                band_counter[token] += value / band_mass
            if motor_mass > 0:
                for token, value in record_motor.items():
                    motor_counter[token] += value / motor_mass
            token_records = company_token_records.setdefault(ticker, {})
            for token in record_band:
                token_records.setdefault(token, set()).add(records_seen)
            records_seen += 1
            if records_seen % 2000 == 0:
                print(f"processed {records_seen} records")

    # ------------------------------------------------- scope 1: by sector
    sector_band: dict[str, Counter] = {}
    sector_motor: dict[str, Counter] = {}
    for ticker, counter in company_band.items():
        sector = ticker_meta.get(ticker, {}).get("sector") or "UNKNOWN"
        sector_band.setdefault(sector, Counter()).update(counter)
        sector_motor.setdefault(sector, Counter()).update(company_motor[ticker])

    sector_tables = tfidf_scores(sector_band, min_df=args.min_df)
    sector_total_band = {k: sum(c.values()) for k, c in sector_band.items()}
    sector_rows = []
    for sector, table in sorted(sector_tables.items()):
        band_total = sector_total_band.get(sector, 1.0)
        motor_total = sum(sector_motor.get(sector, Counter()).values()) or 1.0
        for entry in table[: args.top_n]:
            token = entry["token"]
            band_share = sector_band[sector][token] / band_total
            motor_share = sector_motor.get(sector, Counter())[token] / motor_total
            sector_rows.append(
                {
                    "scope": "sector",
                    "document": sector,
                    **entry,
                    "motor_share": round(motor_share, 8),
                    "gws_specificity": round(band_share / (motor_share + 1e-9), 4),
                }
            )
    write_jsonl(output_dir / "sector_keywords.jsonl", sector_rows, overwrite=True)

    # ------------------------------------- scope 2: company vs its industry
    company_rows = []
    for ticker in sorted(company_band):
        meta = ticker_meta.get(ticker, {})
        sector = meta.get("sector") or "UNKNOWN"
        company_counter = company_band[ticker]
        company_total = sum(company_counter.values())
        industry_counter = sector_band.get(sector)
        if not company_counter or not industry_counter:
            continue
        industry_total = sum(industry_counter.values())
        eps = 1e-12
        token_records = company_token_records.get(ticker, {})
        lifts = []
        for token, weight in company_counter.items():
            if len(token_records.get(token, ())) < args.min_record_df:
                continue
            company_share = weight / company_total
            industry_share = (industry_counter[token] / industry_total) if token in industry_counter else 0.0
            lifts.append(
                (
                    math.log((company_share + eps) / (industry_share + eps)),
                    company_share,
                    token,
                )
            )
        lifts.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for score, share, token in lifts[: args.top_n]:
            company_rows.append(
                {
                    "scope": "company",
                    "document": ticker,
                    "name": meta.get("name", ""),
                    "sector": sector,
                    "token": token,
                    "company_share": round(share, 8),
                    "record_df": len(token_records.get(token, ())),
                    "industry_lift_log": round(score, 6),
                }
            )
    write_jsonl(output_dir / "company_keywords.jsonl", company_rows, overwrite=True)

    summary = {
        "artifact_type": "jspace_tfidf_analysis",
        "schema_version": 1,
        "records_seen": records_seen,
        "unique_companies": len(company_band),
        "sectors": len(sector_band),
        "params": {
            "band_layers": sorted(band_layers),
            "motor_layer": args.motor_layer,
            "top_n_per_document": args.top_n,
            "min_df_scope1": args.min_df,
            "weighting": "readout probability, pooled over band layers, equal record weight",
            "normalisation": "NFKC, lowercase, strip; skip tokens without [a-z]",
        },
    }
    write_json(output_dir / "summary.json", summary, overwrite=True)
    write_metadata(
        output_dir / "summary.json.metadata.json",
        {"readout_path": str(args.readout), "input_csv": str(args.input_csv)},
        overwrite=True,
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
