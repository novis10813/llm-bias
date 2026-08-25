#!/usr/bin/env python
"""Compare raw-token and normalized-concept J-space keyword rankings."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from llm_bias.core.artifacts.io import write_json


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def grouped(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(path):
        result[row["document"]].append(row)
    return result


def mapped_top(
    rows: list[dict],
    *,
    score: str,
    mapping: dict[str, str | None],
    top_k: int,
    positive_only: bool = False,
) -> list[str]:
    ranked = sorted(rows, key=lambda row: row[score], reverse=True)
    concepts = []
    seen = set()
    for row in ranked:
        if positive_only and row[score] <= 0:
            continue
        concept = mapping.get(row["token"])
        if concept is None or concept in seen:
            continue
        seen.add(concept)
        concepts.append(concept)
        if len(concepts) == top_k:
            break
    return concepts


def direct_top(
    rows: list[dict], *, score: str, top_k: int, positive_only: bool = False
) -> list[str]:
    ranked = sorted(rows, key=lambda row: row[score], reverse=True)
    return [
        row["token"]
        for row in ranked
        if not positive_only or row[score] > 0
    ][:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", required=True, type=Path)
    parser.add_argument("--normalized-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    mapping = {
        row["raw_token"]: row["concept"]
        for row in load_jsonl(args.normalized_run / "token_mapping.jsonl")
    }
    raw_tfidf = grouped(args.raw_run / "sector_keywords.jsonl")
    norm_tfidf = grouped(args.normalized_run / "sector_keywords.jsonl")
    raw_lod = grouped(args.raw_run / "sector_logodds.jsonl")
    norm_lod = grouped(args.normalized_run / "sector_logodds.jsonl")

    rows = []
    for sector in sorted(norm_tfidf):
        raw_tf = mapped_top(
            raw_tfidf[sector], score="tfidf", mapping=mapping, top_k=args.top_k
        )
        norm_tf = direct_top(
            norm_tfidf[sector], score="tfidf", top_k=args.top_k
        )
        raw_z = mapped_top(
            raw_lod[sector], score="logodds_z", mapping=mapping,
            top_k=args.top_k, positive_only=True,
        )
        norm_z = direct_top(
            norm_lod[sector], score="logodds_z", top_k=args.top_k,
            positive_only=True,
        )
        rows.append(
            {
                "sector": sector,
                "tfidf_overlap": len(set(raw_tf) & set(norm_tf)),
                "logodds_overlap": len(set(raw_z) & set(norm_z)),
                "raw_tfidf_concepts": raw_tf,
                "normalized_tfidf_concepts": norm_tf,
                "raw_logodds_concepts": raw_z,
                "normalized_logodds_concepts": norm_z,
            }
        )
    result = {
        "top_k": args.top_k,
        "comparison": "raw rankings mapped to concepts before overlap",
        "mean_tfidf_overlap": sum(r["tfidf_overlap"] for r in rows) / len(rows),
        "mean_logodds_overlap": sum(r["logodds_overlap"] for r in rows) / len(rows),
        "sectors": rows,
    }
    write_json(args.output, result, overwrite=True)
    print(
        f"TF-IDF mean overlap {result['mean_tfidf_overlap']:.2f}/{args.top_k}; "
        f"log-odds {result['mean_logodds_overlap']:.2f}/{args.top_k}"
    )


if __name__ == "__main__":
    main()
