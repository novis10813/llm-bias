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
from typing import Callable

import numpy as np

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


class ConceptNormalizer:
    """Conservative raw-token to readable-concept mapping.

    Prompt vocabulary terms are retained as observed domain language. Other
    alphabetic tokens must pass an English word-frequency threshold. Simplemma
    consolidates inflectional variants before any counts are accumulated.
    """

    def __init__(self, prompt_vocab: set[str], *, min_zipf: float = 2.0):
        from simplemma import lemmatize
        from wordfreq import zipf_frequency

        self.prompt_vocab = prompt_vocab
        self.min_zipf = min_zipf
        self._lemmatize = lemmatize
        self._zipf_frequency = zipf_frequency
        self.mapping: dict[str, tuple[str | None, str]] = {}

    def __call__(self, token: str) -> str | None:
        cached = self.mapping.get(token)
        if cached is not None:
            return cached[0]
        if re.fullmatch(r"<[^>]+>", token) or re.fullmatch(r"<\|[^|]+\|>", token):
            result = (None, "filtered_special_token")
        else:
            surface = re.sub(r"^[^a-z]+|[^a-z]+$", "", token)
            if not re.fullmatch(r"[a-z]+", surface):
                result = (None, "filtered_non_alphabetic")
            else:
                concept = self._lemmatize(surface, lang="en").lower()
                if len(concept) < 2:
                    result = (None, "filtered_short")
                elif (
                    surface in self.prompt_vocab
                    or concept in self.prompt_vocab
                    or self._zipf_frequency(concept, "en") >= self.min_zipf
                ):
                    if concept != surface:
                        reason = "lemmatized"
                    elif surface != token:
                        reason = "punctuation_stripped"
                    elif surface in self.prompt_vocab:
                        reason = "prompt_vocabulary"
                    else:
                        reason = "english_lexicon"
                    result = (concept, reason)
                else:
                    result = (None, "filtered_low_frequency")
        self.mapping[token] = result
        return result[0]


def accumulate_layers(
    positions: list[dict],
    *,
    band_layers: set[int],
    motor_layer: int,
    counter: Counter,
    motor_counter: Counter,
    token_transform: Callable[[str], str | None] | None = None,
    raw_token_weights: Counter | None = None,
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
                if raw_token_weights is not None:
                    raw_token_weights[token] += probability
                if token_transform is not None:
                    token = token_transform(token)
                    if token is None:
                        continue
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


def log_odds_table(
    focus: Counter,
    background: Counter,
    *,
    global_counter: Counter,
    alpha: float = 1.0,
    min_count: float = 0.0,
):
    """Monroe et al. (2017) log-odds ratio with informative Dirichlet prior.

    Returns rows sorted by |z| desc: token, delta (log-odds focus vs
    background), se, z. ``global_counter`` supplies the prior token
    distribution pi; ``alpha`` is the total pseudo-count mass.
    """
    global_total = sum(global_counter.values()) or 1.0
    focus_total = sum(focus.values())
    background_total = sum(background.values())
    vocab = set(focus) | set(global_counter)
    rows = []
    for token in vocab:
        pi = global_counter.get(token, 0.0) / global_total
        y1 = focus.get(token, 0.0)
        y2 = background.get(token, 0.0)
        if y1 < min_count:
            continue
        delta = (
            math.log(y1 + alpha * pi)
            - math.log(focus_total + alpha - y1 + alpha * (1 - pi))
            - math.log(y2 + alpha * pi)
            + math.log(background_total + alpha - y2 + alpha * (1 - pi))
        )
        variance = 1.0 / (y1 + alpha * pi) + 1.0 / (y2 + alpha * pi)
        se = math.sqrt(variance)
        rows.append(
            {
                "token": token,
                "delta_logodds": round(delta, 6),
                "se": round(se, 6),
                "logodds_z": round(delta / se, 4),
            }
        )
    rows.sort(key=lambda row: abs(row["logodds_z"]), reverse=True)
    return rows


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
    parser.add_argument("--logodds-alpha", type=float, default=1.0,
                        help="total Dirichlet pseudo-count mass for the "
                        "informative-prior log-odds scoring")
    parser.add_argument("--company-matrix-npz", type=Path, default=None,
                        help="optional path for a compact company x top-vocab "
                        "weight matrix (npz) for downstream visualisation")
    parser.add_argument("--matrix-vocab", type=int, default=5000,
                        help="vocabulary size for --company-matrix-npz")
    parser.add_argument("--concept-normalize", action="store_true",
                        help="lemmatize and conservatively filter token pieces "
                        "before accumulating any analysis counts")
    parser.add_argument("--min-english-zipf", type=float, default=2.0,
                        help="wordfreq threshold used by --concept-normalize")
    args = parser.parse_args()

    band_layers = {int(x) for x in args.band_layers.split(",") if x.strip()}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_meta: dict[str, dict] = {}
    sector_prompt_tokens: dict[str, Counter] = {}
    with open(args.input_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ticker_meta[row["ticker"]] = {
                "sector": row["sector"],
                "name": row.get("name", ""),
                "marketcap": row.get("marketcap", ""),
            }
            # prompt-echo baseline: token distribution of the raw prompt text
            prompt_counter = sector_prompt_tokens.setdefault(row["sector"], Counter())
            for column, text in row.items():
                if not column.startswith("prompt_with_context") or not text:
                    continue
                for word in re.findall(r"[A-Za-z]+", text):
                    token = normalize_token(word)
                    if token is not None:
                        prompt_counter[token] += 1

    concept_normalizer: ConceptNormalizer | None = None
    if args.concept_normalize:
        prompt_vocab = {
            token
            for counter in sector_prompt_tokens.values()
            for token in counter
        }
        concept_normalizer = ConceptNormalizer(
            prompt_vocab, min_zipf=args.min_english_zipf
        )
        normalized_prompt_tokens: dict[str, Counter] = {}
        for sector, counter in sector_prompt_tokens.items():
            normalized = Counter()
            for token, count in counter.items():
                concept = concept_normalizer(token)
                if concept is not None:
                    normalized[concept] += count
            normalized_prompt_tokens[sector] = normalized
        sector_prompt_tokens = normalized_prompt_tokens

    company_band: dict[str, Counter] = {}
    company_motor: dict[str, Counter] = {}
    # token -> set of record ordinals it appeared in, per company
    company_token_records: dict[str, dict[str, set[int]]] = {}
    raw_token_weights: Counter = Counter()
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
                token_transform=concept_normalizer,
                raw_token_weights=raw_token_weights if concept_normalizer else None,
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
    global_counter: Counter = Counter()
    for counter in sector_band.values():
        global_counter.update(counter)
    sector_rows = []
    sector_logodds_rows = []
    for sector, table in sorted(sector_tables.items()):
        band_total = sector_total_band.get(sector, 1.0)
        motor_total = sum(sector_motor.get(sector, Counter()).values()) or 1.0
        background = Counter(global_counter)
        background.subtract(sector_band[sector])
        logodds_by_token = {
            row["token"]: row
            for row in log_odds_table(
                sector_band[sector], background, global_counter=global_counter,
                alpha=args.logodds_alpha,
            )
        }
        sector_logodds_rows.extend(
            {
                "scope": "sector",
                "document": sector,
                **row,
            }
            for row in logodds_by_token.values()
        )
        prompt_counter = sector_prompt_tokens.get(sector, Counter())
        prompt_total = sum(prompt_counter.values()) or 1.0
        for entry in table[: args.top_n]:
            token = entry["token"]
            band_share = sector_band[sector][token] / band_total
            motor_share = sector_motor.get(sector, Counter())[token] / motor_total
            prompt_share = prompt_counter.get(token, 0) / prompt_total
            lod = logodds_by_token.get(token, {})
            sector_rows.append(
                {
                    "scope": "sector",
                    "document": sector,
                    **entry,
                    "motor_share": round(motor_share, 8),
                    "gws_specificity": round(band_share / (motor_share + 1e-9), 4),
                    "prompt_share": round(prompt_share, 8),
                    "echo_lift": round(band_share / prompt_share, 4) if prompt_share > 0 else None,
                    "delta_logodds": lod.get("delta_logodds"),
                    "logodds_z": lod.get("logodds_z"),
                }
            )
    write_jsonl(output_dir / "sector_keywords.jsonl", sector_rows, overwrite=True)
    write_jsonl(output_dir / "sector_logodds.jsonl", sector_logodds_rows, overwrite=True)

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
        industry_excl = Counter(industry_counter)
        industry_excl.subtract(company_counter)
        industry_excl_total = sum(industry_excl.values()) or 1.0
        logodds_by_token = {
            row["token"]: row
            for row in log_odds_table(
                company_counter, industry_excl, global_counter=global_counter,
                alpha=args.logodds_alpha,
            )
        }
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
        for score, share, token in lifts[: args.top_n * 2]:
            lod = logodds_by_token.get(token, {})
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
                    "delta_logodds": lod.get("delta_logodds"),
                    "logodds_z": lod.get("logodds_z"),
                }
            )
    write_jsonl(output_dir / "company_keywords.jsonl", company_rows, overwrite=True)

    if args.company_matrix_npz is not None:
        global_total = Counter()
        for counter in company_band.values():
            global_total.update(counter)
        top_tokens = [
            token
            for token, _ in global_total.most_common(args.matrix_vocab)
        ]
        token_index = {token: i for i, token in enumerate(top_tokens)}
        tickers = sorted(company_band)
        matrix = np.zeros((len(tickers), len(top_tokens)), dtype=np.float32)
        for row_i, ticker in enumerate(tickers):
            for token, weight in company_band[ticker].items():
                col = token_index.get(token)
                if col is not None:
                    matrix[row_i, col] = weight
        # row-normalise so each company is a distribution
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix /= row_sums
        np.savez_compressed(
            args.company_matrix_npz,
            matrix=matrix,
            tickers=np.array(tickers),
            tokens=np.array(top_tokens),
            sectors=np.array([ticker_meta.get(t, {}).get("sector", "UNKNOWN") for t in tickers]),
        )
    # z-ranked company tables: primary scope-2 ranking by Monroe log-odds
    for row in company_rows:
        row["_z"] = abs(row.get("logodds_z") or 0.0)
    company_rows.sort(key=lambda r: (r["document"], -r["_z"]))
    for row in company_rows:
        row.pop("_z")
    write_jsonl(output_dir / "company_logodds.jsonl", company_rows, overwrite=True)

    if concept_normalizer is not None:
        mapping_rows = []
        for raw_token, (concept, reason) in concept_normalizer.mapping.items():
            mapping_rows.append(
                {
                    "raw_token": raw_token,
                    "concept": concept,
                    "reason": reason,
                    "raw_probability_weight": round(raw_token_weights.get(raw_token, 0.0), 6),
                }
            )
        mapping_rows.sort(
            key=lambda row: row["raw_probability_weight"], reverse=True
        )
        write_jsonl(output_dir / "token_mapping.jsonl", mapping_rows, overwrite=True)

    summary = {
        "artifact_type": "jspace_tfidf_analysis",
        "schema_version": 2 if args.concept_normalize else 1,
        "records_seen": records_seen,
        "unique_companies": len(company_band),
        "sectors": len(sector_band),
        "params": {
            "band_layers": sorted(band_layers),
            "motor_layer": args.motor_layer,
            "top_n_per_document": args.top_n,
            "min_df_scope1": args.min_df,
            "weighting": "readout probability, pooled over band layers, equal record weight",
            "normalisation": (
                "NFKC, lowercase, strip, English lemmatization; retain prompt "
                f"vocabulary or wordfreq zipf >= {args.min_english_zipf}"
                if args.concept_normalize
                else "NFKC, lowercase, strip; skip tokens without [a-z]"
            ),
            "concept_normalize": args.concept_normalize,
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
