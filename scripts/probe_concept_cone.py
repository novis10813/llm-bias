"""Probe: Multi-dimensional Concept Cone steering via Token-wise Sector-Demeaned Contrastive SVD.

Protocol / Theory: Wollschläger et al. (2025) Concept Cones applied to 200-company held-out cohort.
Implements:
1. Sector Demeaning across 100 instruction tokens (removes industry semantics).
2. Slice-wise Contrastive SVD on Top-20 vs Bottom-20 extreme stance pairs.
3. Sign alignment to construct an orthonormal basis for the Buy Polyhedral Cone.
4. Multi-company evaluation of individual basis rays vs Cone Centroid.

Outputs compact derived summaries and generation decisions. No raw activations are persisted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from statistics import median
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.adapter import InjectedModelAdapter
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.generation import GenerationConfig, generate_tokens
from llm_bias.core.inference.interventions import pre_residual_interventions, record_block_states, residual_interventions
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids, token_span
from llm_bias.entity_to_dial.dial_probe import answer_token_ids, margin_from_log_probs
from llm_bias.entity_to_dial.heldout_transfer import HELDOUT_LAYER, _render_frozen_prompt
from llm_bias.entity_to_dial.spans import instruction_char_span
from llm_bias.entity_to_dial.template import EVIDENCE_CLOSE, EVIDENCE_MARKER, NAME_LINE_PREFIX, TICKER_LINE_PREFIX

MACRO_SCENARIOS = {
    "macro_mixed": [
        "The Federal Reserve signaled a potential pause in interest rate hikes as labor markets show early cooling.",
        "Core inflation remains persistent at 4.8%, driven by housing and services costs, while goods prices contract.",
        "Corporate earnings revisions have diverged sharply: defensive dividend sectors gained 4% while cyclical growth fell 8%.",
    ],
    "macro_hawkish": [
        "The Federal Reserve raised interest rates by 50 bps, pushing benchmark borrowing costs to a 15-year high.",
        "Headline CPI inflation remains elevated at 6.2%, significantly eroding real household disposable income.",
        "Corporate bond yield spreads widened 80 bps as capital markets tightened liquidity and credit availability.",
    ],
    "macro_dovish": [
        "The Federal Reserve announced a 50 bps interest rate cut, signaling an accommodative monetary easing cycle.",
        "Headline CPI cooled to 2.1%, restoring real wage growth and supporting robust retail consumption.",
        "Benchmark equity indices reached record highs as corporate credit spreads compressed to historical lows.",
    ],
}


def render_custom_prompt(ticker: str, name: str, evidence_bullets: Sequence[str]) -> str:
    """Render a custom prompt with specific evidence bullets or zero evidence."""
    if evidence_bullets:
        evidence_text = "\n".join(f"- {b}" for b in evidence_bullets)
        evidence_block = f"{EVIDENCE_MARKER}\n\n{evidence_text}\n\n{EVIDENCE_CLOSE[2:]}"
    else:
        evidence_block = ""
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"{TICKER_LINE_PREFIX}{ticker}]\n\n{NAME_LINE_PREFIX}{name}]\n\n"
        f"{evidence_block}"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        '"decision": "buy" or "sell"\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Token-wise Sector-Demeaned Contrastive SVD Concept Cone")
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b", help="Model checkpoint path")
    parser.add_argument(
        "--heldout-run",
        default="artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/entity-to-dial-heldout-transfer-v1-01",
        help="Path to heldout transfer run to obtain cohort clean rankings",
    )
    parser.add_argument("--k-contrast-pairs", type=int, default=20, help="Number of Top vs Bottom extreme pairs")
    parser.add_argument("--k-cone-dim", type=int, default=4, help="Dimensionality of the Concept Cone")
    parser.add_argument(
        "--evidence-mode",
        choices=["balanced", "zero_evidence", "macro_mixed", "macro_hawkish", "macro_dovish"],
        default="balanced",
        help="Evidence context for evaluation",
    )
    parser.add_argument(
        "--target-tickers",
        nargs="+",
        default=["MO", "FOXA", "CNC"],
        help="Tickers to evaluate",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 2.0, 3.0, 4.0, 5.0],
        help="Alpha multiplier sweep",
    )
    parser.add_argument(
        "--eval-individual-rays",
        action="store_true",
        default=False,
        help="Also evaluate each basis ray individually on the first target ticker",
    )
    parser.add_argument(
        "--leave-out-sector",
        default=None,
        help="Exclude this sector from the contrast construction set (leave-one-sector-out)",
    )
    parser.add_argument(
        "--inject-layer",
        type=int,
        default=None,
        help="Layer for the residual intervention (default: heldout layer 15)",
    )
    parser.add_argument(
        "--measure-stance-cosine",
        action="store_true",
        default=False,
        help="Compute per-token cosine between cone axis b1 and the Top/Bottom difference-in-means direction",
    )
    parser.add_argument(
        "--centroid-dims",
        nargs="+",
        type=int,
        default=None,
        help="Evaluate centroids of the first j axes (dimension ablation; each must be <= k-cone-dim)",
    )
    parser.add_argument(
        "--persist-directions",
        default=None,
        help="Directory to persist the cone basis as a compact derived operator with provenance",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Extract only (no evaluation); used for cosine measurement / persistence runs",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write compact derived results")
    parser.add_argument("--cohort-mode", choices=["legacy", "sp500_v1", "sp500_paper", "sp500_dim_paper", "sp500_dim_crossmodel"], default="legacy",
                        help="sp500_v1: old 16-company C2 pilot; sp500_paper: 427-company-selected cone; sp500_dim_paper: Qwen DIM; sp500_dim_crossmodel: Gemma/GLM/GPT DIM")
    parser.add_argument("--dim-arm", choices=["tokenwise", "single_all"], default="tokenwise",
                        help="DIM paper arm; ignored by other cohort modes")
    parser.add_argument("--dim-layers", nargs="+", type=int, default=list(DIM_PAPER_LAYERS),
                        help="Qwen default: L0,L14-L18,L31; for crossmodel explicitly pass the frozen model-specific layer subset")
    parser.add_argument("--model-dtype", choices=["bf16", "native"], default="bf16")
    parser.add_argument("--population-csv", default="data/sp500_constituents_2020_2025.csv")
    parser.add_argument("--split-seed", type=int, default=20260923)
    parser.add_argument("--smoke-tickers", nargs="+", default=None,
                        help="Evaluate these tickers instead of the 101 held-out companies (non-confirmatory)")
    return parser.parse_args()


DECISION_PREFIX = '{"decision": "'
# The old runs must remain reproducible, but must not be relabeled as 427-company-selected.
C2_LAYERS = {"qwen3.5-4b": 15, "gemma4-12b-it": 25, "glm4-9b-0414": 19, "gpt-oss-20b": 12}
PAPER_C2_LAYERS = {"qwen3.5-4b": 16, "gemma4-12b-it": 27, "glm4-9b-0414": 19, "gpt-oss-20b": 14}


def c2_427_layer_source(slug: str) -> dict[str, Any]:
    """Bind paper selection to the persisted 427-company instruction peak."""
    if slug not in PAPER_C2_LAYERS:
        raise ValueError(f"no paper C2 layer for {slug}")
    source = Path("artifacts") / slug / "balanced-evidence-gap-phase2" / "runs" / "phase2b-v2-427-01" / "analyze" / "summary.json"
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    data = json.loads(payload)
    curve = data["curves"]["instruction"]
    peak = int(max(curve, key=lambda k: curve[k]["mean_normalized_transfer"]))
    if peak != PAPER_C2_LAYERS[slug] or curve[str(peak)]["n_directions"] != 854:
        raise ValueError(f"427-company C2 source does not select L{PAPER_C2_LAYERS[slug]} for {slug}")
    return {"path": str(source), "sha256": digest, "instruction_peak": peak,
            "n_directions": 854}


DIM_PAPER_LAYERS = (0, 14, 15, 16, 17, 18, 31)
# Freeze source bytes, model identity and preselected C2 layer sets before new DIM evaluations.
DIM_CROSSMODEL_CONFIG = {
    "gemma4-12b-it": {
        "n_layers": 48, "peak": 27, "layers": (0, 26, 27, 28, 29, 30, 31, 32, 47),
        "dtype": "bf16", "suffix_tokens": 100, "source_run": "c2-guided-paper-20260924",
        "source_schema": "concept-cone-sp500-paper-v1",
        "source_sha256": "13c2b4a6f1538a7ab84183e86c7e8ba4fe63decff5a898e5d3251a73eb09c7e7",
        "c2_sha256": "295170b9c09ca453ce7b285c37a0f128d99cf68fc1a20970f390f084fdd54635",
    },
    "glm4-9b-0414": {
        "n_layers": 40, "peak": 19, "layers": (0, 17, 18, 19, 20, 21, 39),
        "dtype": "bf16", "suffix_tokens": 98, "source_run": "crossmodel-v1-eval-20260924",
        "source_schema": "concept-cone-sp500-v1",
        "source_sha256": "025a1b4a76f0a564b8b593a19a6ed2805278e2a23d5799a9eaa50ce574a926be",
        "c2_sha256": "d0b7e41ac21adc1cb858fbdc6a75e34338109cab84f3630bcd6f839119b480a6",
    },
    "gpt-oss-20b": {
        "n_layers": 24, "peak": 14, "layers": (0, 12, 13, 14, 15, 16, 23),
        "dtype": "native", "suffix_tokens": 99, "source_run": "c2-guided-paper-20260924",
        "source_schema": "concept-cone-sp500-paper-v1",
        "source_sha256": "6987e4197e683965018ac5f92347daafe841bc80b7ff41e00fa376d7db7ae393",
        "c2_sha256": "f8ccb7c24d25048d358a32562ba8fffd799384a482b1e17152d7f0bef543a506",
    },
}
DIM_PAPER_SOURCE_FIELDS = (
    "schema", "mode", "model", "model_slug", "model_config_sha256",
    "tokenizer_config_sha256", "tokenizer", "checkpoint_files", "population_sha256",
    "split_seed", "split_sha256", "construction_tickers", "evaluation_tickers",
    "prompt_family_sha256", "prompt_renderer", "decision_prefix", "layer", "k_pairs",
    "k_cone", "alphas", "dtype", "max_new_tokens", "c2_layer_source",
)


def validate_dim_c2_curve(curve: Mapping[str, Mapping[str, Any]]) -> dict[int, float]:
    """Freeze layer candidates from the independent 427-company C2 result."""
    if set(curve) != {str(i) for i in range(32)}:
        raise ValueError("C2 instruction curve must cover all 32 layers")
    transfers: dict[int, float] = {}
    for i in range(32):
        row = curve[str(i)]
        value = row.get("mean_normalized_transfer")
        if row.get("n_directions") != 854:
            raise ValueError("C2 curve must contain 854 directions per layer")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("nonfinite C2 instruction transfer")
        transfers[i] = float(value)
    high = {i for i, value in transfers.items() if value >= 0.25}
    if high != {14, 15, 16, 17, 18} or max(transfers, key=transfers.get) != 16:
        raise ValueError("C2 candidate layers no longer match frozen instruction T rule")
    return {i: transfers[i] for i in DIM_PAPER_LAYERS}


def validate_dim_crossmodel_curve(
    curve: Mapping[str, Mapping[str, Any]], spec: Mapping[str, Any],
) -> dict[int, float]:
    """Verify the fixed C2 peak, relative-threshold set and end-layer comparators."""
    n_layers = spec["n_layers"]
    if set(curve) != {str(i) for i in range(n_layers)}:
        raise ValueError("crossmodel C2 instruction curve does not cover all layers")
    transfers: dict[int, float] = {}
    for layer in range(n_layers):
        row = curve[str(layer)]
        value = row.get("mean_normalized_transfer")
        if row.get("n_directions") != 854:
            raise ValueError("crossmodel C2 curve must contain 854 directions per layer")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("nonfinite crossmodel C2 instruction transfer")
        transfers[layer] = float(value)
    peak = spec["peak"]
    if max(transfers, key=transfers.get) != peak or transfers[peak] <= 0:
        raise ValueError("crossmodel C2 peak differs from frozen layer")
    high = {layer for layer, value in transfers.items() if value >= 0.70 * transfers[peak]}
    if len(high) < 5:
        high |= {layer for layer in range(peak - 2, peak + 3) if 0 <= layer < n_layers}
    selected = high | {0, n_layers - 1}
    if selected != set(spec["layers"]):
        raise ValueError("crossmodel C2 candidate layers changed")
    return {layer: transfers[layer] for layer in spec["layers"]}


def validate_dim_paper_ranking(
    source: Mapping[str, Any], expected: Mapping[str, Any],
    construction: Sequence[str], evaluation: Sequence[str],
    *, source_spec: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Accept only the complete model-local cone ranking of the same disjoint cohort."""
    spec = source_spec or {"suffix_tokens": 100, "source_schema": "concept-cone-sp500-paper-v1",
                           "slug": "qwen3.5-4b", "peak": 16, "dtype": "bf16"}
    fields = tuple(field for field in DIM_PAPER_SOURCE_FIELDS
                   if field != "c2_layer_source" or spec["slug"] != "glm4-9b-0414")
    metadata = source.get("metadata")
    if not isinstance(metadata, dict) or any(
        field not in expected or field not in metadata or metadata[field] != expected[field]
        for field in fields
    ) or (spec["slug"] == "glm4-9b-0414" and "c2_layer_source" in metadata):
        raise ValueError("paper cone source metadata mismatch")
    if (source.get("complete") is not True or source.get("effective_tokens") != spec["suffix_tokens"]
            or metadata["schema"] != spec["source_schema"]
            or metadata["mode"] != "evaluation" or metadata["model_slug"] != spec["slug"]
            or metadata["layer"] != spec["peak"] or metadata["k_pairs"] != 20 or metadata["k_cone"] != 4
            or metadata["split_seed"] != 20260923 or metadata["dtype"] != spec["dtype"]
            or metadata["alphas"] != [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            or metadata["decision_prefix"] != DECISION_PREFIX or metadata["max_new_tokens"] != 192):
        raise ValueError("paper cone source protocol mismatch")
    if (len(construction) != 402 or len(evaluation) != 101
            or set(construction) & set(evaluation)
            or metadata["construction_tickers"] != list(construction)
            or metadata["evaluation_tickers"] != list(evaluation)):
        raise ValueError("paper cone construction/evaluation split mismatch")
    ranking = source.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != 402 or any(
        not isinstance(row, dict) or not isinstance(row.get("ticker"), str)
        or isinstance(row.get("margin"), bool) or not isinstance(row.get("margin"), (int, float))
        or not math.isfinite(row["margin"]) for row in ranking
    ):
        raise ValueError("paper cone ranking has nonfinite or invalid rows")
    ranked = [row["ticker"] for row in ranking]
    if set(ranked) != set(construction) or len(set(ranked)) != 402 or set(ranked) & set(evaluation):
        raise ValueError("paper cone ranking differs from construction cohort")
    if ranking != sorted(ranking, key=lambda row: (row["margin"], row["ticker"])):
        raise ValueError("paper cone ranking order changed")
    if source.get("top_20") != ranked[-20:] or source.get("bottom_20") != ranked[:20]:
        raise ValueError("paper cone top/bottom provenance differs from ranking")
    return ranked[-10:], ranked[:10]


class DegenerateDimDirection(ValueError):
    """Retain only a scalar norm for a fail-closed direction-fit diagnostic."""

    def __init__(self, norm: float, layer: int | None = None) -> None:
        super().__init__(f"degenerate DIM mean difference at L{layer}: norm={norm:g}")
        self.norm = norm
        self.layer = layer


def fit_dim_direction(top: torch.Tensor, bottom: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute model-local 1D mean difference in fp32, without storing raw states."""
    if (top.shape != bottom.shape or top.ndim not in (2, 3) or top.shape[0] != 10
            or (top.ndim == 3 and top.shape[1] < 16) or top.shape[-1] < 2):
        raise ValueError("DIM state shape must be [10,d] or [10,K>=16,d]")
    if not torch.isfinite(top).all() or not torch.isfinite(bottom).all():
        raise ValueError("DIM states contain nonfinite values")
    difference = top.float().mean(dim=0) - bottom.float().mean(dim=0)
    norms = difference.norm(dim=-1)
    if not torch.isfinite(norms).all():
        raise ValueError("DIM direction has nonfinite norm")
    if torch.any(norms <= 1e-8):
        raise DegenerateDimDirection(float(norms.min()))
    unit = difference / norms.unsqueeze(-1)
    if not torch.isfinite(unit).all() or torch.any((unit.norm(dim=-1) - 1).abs() > 1e-5):
        raise ValueError("DIM unit direction failed normalization")
    flat = norms.reshape(-1)
    return unit, {"min": float(flat.min()), "median": float(flat.median()), "max": float(flat.max())}


def split_population(path: Path, seed: int) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    """Freeze a disjoint 402/101 split from the 2024 S&P 500 population."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("year") == "2024" and r.get("index_name") == "S&P 500"]
    if len(rows) != 503 or len({r["ticker"] for r in rows}) != 503:
        raise ValueError(f"expected 503 unique 2024 S&P 500 tickers, got {len(rows)}")
    if any(not r.get("company_name") or not r.get("gics_sector") for r in rows):
        raise ValueError("missing company name or sector")
    companies = {r["ticker"]: {"ticker": r["ticker"], "name": r["company_name"],
                              "sector": r["gics_sector"]} for r in rows}
    shuffled = sorted(companies)
    random.Random(seed).shuffle(shuffled)
    construction, evaluation = sorted(shuffled[:402]), sorted(shuffled[402:])
    assert len(construction) == 402 and len(evaluation) == 101 and not set(construction) & set(evaluation)
    return companies, construction, evaluation


def prepare_instruction_suffix(tokenizer: Any, companies: Mapping[str, Mapping[str, str]]) -> int:
    """Check identical ending token IDs inside the instruction for *all* companies."""
    common: list[int] | None = None
    for ticker, company in companies.items():
        prompt = _render_frozen_prompt(ticker, company["name"], order=0, reverse=False)
        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        begin = fmt.find(prompt)
        if begin < 0:
            raise ValueError(f"chat template omitted prompt for {ticker}")
        start, end = instruction_char_span(prompt)
        span = token_span(tokenizer, fmt, begin + start, begin + end, add_special_tokens=True)
        if span is None:
            raise ValueError(f"missing instruction span for {ticker}")
        suffix = input_ids(tokenizer, fmt)[span[0]:span[1]]
        if common is None:
            common = suffix
        else:
            n = 0
            for left, right in zip(reversed(common), reversed(suffix)):
                if left != right:
                    break
                n += 1
            common = common[-n:] if n else []
        if len(common) < 16:
            raise ValueError(f"no usable common instruction suffix at {ticker}")
    if not common:
        raise ValueError("empty instruction suffix")
    return len(common)


def rank_construction(model: Any, tokenizer: Any, companies: Mapping[str, Mapping[str, str]],
                      tickers: Sequence[str]) -> list[dict[str, Any]]:
    """Model-local clean fixed-prefix buy-vs-sell ranking, no evaluation leakage."""
    rows: list[dict[str, Any]] = []
    for index, ticker in enumerate(tickers):
        c = companies[ticker]
        prompt = _render_frozen_prompt(ticker, c["name"], order=0, reverse=False)
        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        scoring = fmt + DECISION_PREFIX
        buy_id, sell_id = answer_token_ids(tokenizer, scoring)
        ids = torch.tensor([input_ids(tokenizer, scoring)], dtype=torch.long, device=model.input_device)
        res = record_residuals(model, ids, [model.n_layers - 1])[model.n_layers - 1]
        margin = margin_from_log_probs(fp32_next_token_log_probs(model, res[:, -1, :]), buy_id, sell_id)
        rows.append({"ticker": ticker, "margin": margin})
        if (index + 1) % 50 == 0:
            print(f"Ranked {index + 1}/{len(tickers)} companies", flush=True)
    return sorted(rows, key=lambda r: (r["margin"], r["ticker"]))


def alpha_summary(results: Mapping[str, Any], tickers: Sequence[str], alphas: Sequence[float]) -> list[dict[str, Any]]:
    """Counts use all evaluated tickers; unparsed decisions are not called flips."""
    summary = []
    for j, alpha in enumerate(alphas):
        rows = [results["targets"][t]["cone_centroid"][j] for t in tickers]
        clean = [results["targets"][t]["cone_centroid"][0] for t in tickers]
        if any(r["alpha"] != alpha for r in rows):
            raise ValueError(f"incomplete alpha grid for {alpha}")
        parsed = sum(r["decision"] in ("buy", "sell") for r in rows)
        valid_pairs = [(b, r) for b, r in zip(clean, rows, strict=True)
                       if b["decision"] in ("buy", "sell") and r["decision"] in ("buy", "sell")]
        flipped = sum(b["decision"] != r["decision"] for b, r in valid_pairs)
        summary.append({"alpha": alpha, "n": len(tickers), "parsed": parsed,
                        "parse_rate": parsed / len(tickers), "valid_flip_pairs": len(valid_pairs),
                        "flipped": flipped, "flip_rate": flipped / len(valid_pairs) if valid_pairs else None})
    return summary


def alpha_table(results: Mapping[str, Any], tickers: Sequence[str], alphas: Sequence[float]) -> str:
    """Human-readable per-company margin and greedy decision; alpha is the column key."""
    names = [f"alpha={a:g}" for a in alphas]
    lines = ["| ticker | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for ticker in tickers:
        rows = results["targets"][ticker]["cone_centroid"]
        if [r["alpha"] for r in rows] != list(alphas):
            raise ValueError(f"incomplete alpha grid for {ticker}")
        lines.append("| " + ticker + " | " + " | ".join(
            f"{r['margin']:+.3f} ({r['decision']})" for r in rows) + " |")
    return "\n".join(lines) + "\n"


def validate_dim_resume(
    result: Mapping[str, Any], metadata: Mapping[str, Any], tickers: Sequence[str],
    layers: Sequence[int], alphas: Sequence[float],
) -> None:
    """Fail closed on changed provenance or partially corrupted ticker/layer records."""
    if result.get("metadata") != metadata or result.get("complete") not in (True, False):
        raise ValueError("DIM resume metadata or completion flag mismatch")
    targets = result.get("targets")
    if not isinstance(targets, dict) or not set(targets).issubset(set(tickers)):
        raise ValueError("DIM resume target cohort differs")
    valid_layers = {f"L{i}" for i in layers}
    for ticker, by_layer in targets.items():
        if not isinstance(by_layer, dict) or not set(by_layer).issubset(valid_layers):
            raise ValueError(f"DIM resume layer mismatch for {ticker}")
        for name, entry in by_layer.items():
            rows = entry.get("rows") if isinstance(entry, dict) else None
            if not isinstance(rows, list) or len(rows) != len(alphas) or any(
                not isinstance(row, dict) or row.get("alpha") != alpha
                for alpha, row in zip(alphas, rows, strict=True)
            ):
                raise ValueError(f"DIM resume alpha grid mismatch for {ticker}/{name}")
            for row in rows:
                value = row.get("margin")
                text = row.get("generated_text")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"DIM resume nonfinite margin for {ticker}/{name}")
                if not isinstance(text, str) or row.get("decision") not in ("buy", "sell", "unparsed"):
                    raise ValueError(f"DIM resume invalid generated text/decision for {ticker}/{name}")
                try:
                    parsed = json.loads(text)
                    decision = parsed.get("decision") if isinstance(parsed, dict) else None
                except (ValueError, TypeError):
                    decision = None
                if decision not in ("buy", "sell"):
                    decision = "unparsed"
                if row.get("decision") != decision or row.get("parse_ok") is not (decision != "unparsed"):
                    raise ValueError(f"DIM resume strict JSON mismatch for {ticker}/{name}")
    if result["complete"] and (set(targets) != set(tickers) or any(
        set(targets[t]) != valid_layers for t in tickers
    )):
        raise ValueError("DIM resume marked complete but layer/ticker matrix is incomplete")


def validate_dim_baseline_consistency(
    targets: Mapping[str, Any], tickers: Sequence[str], layers: Sequence[int],
) -> None:
    """The unhooked alpha-zero score and generation must agree across sweep layers."""
    if not layers:
        raise ValueError("DIM baseline requires candidate layers")
    for ticker in tickers:
        baseline = targets[ticker][f"L{layers[0]}"]["rows"][0]
        for layer in layers[1:]:
            other = targets[ticker][f"L{layer}"]["rows"][0]
            if (abs(baseline["margin"] - other["margin"]) > 1e-3
                    or baseline["generated_text"] != other["generated_text"]):
                raise ValueError(f"crossmodel DIM alpha-zero baseline differs across layers for {ticker}")


def dim_layer_summary(
    targets: Mapping[str, Any], tickers: Sequence[str], layer: int, alphas: Sequence[float],
) -> list[dict[str, Any]]:
    """Paired strict generated transitions and distinct fixed-answer margin shifts."""
    if not tickers or not alphas or alphas[0] != 0:
        raise ValueError("DIM requires a nonempty alpha-zero baseline")
    all_rows: list[list[Mapping[str, Any]]] = []
    for ticker in tickers:
        rows = targets[ticker][f"L{layer}"]["rows"]
        if len(rows) != len(alphas) or any(r["alpha"] != a for r, a in zip(rows, alphas, strict=True)):
            raise ValueError("DIM alpha grid incomplete")
        if any(not math.isfinite(r["margin"]) for r in rows):
            raise ValueError("DIM nonfinite margin")
        all_rows.append(rows)
    baseline = [rows[0] for rows in all_rows]
    n = len(tickers)
    base_buy = sum(b["decision"] == "buy" for b in baseline)
    base_sell = sum(b["decision"] == "sell" for b in baseline)
    summary: list[dict[str, Any]] = []
    for j, alpha in enumerate(alphas):
        current = [rows[j] for rows in all_rows]
        valid = [(b["decision"], r["decision"]) for b, r in zip(baseline, current, strict=True)
                 if b["decision"] in ("buy", "sell") and r["decision"] in ("buy", "sell")]
        buy_valid = sum(b == "buy" for b, _ in valid)
        sell_valid = sum(b == "sell" for b, _ in valid)
        buy_to_sell = sum(b == "buy" and r == "sell" for b, r in valid)
        sell_to_buy = sum(b == "sell" and r == "buy" for b, r in valid)
        parsed = sum(r["decision"] in ("buy", "sell") for r in current)
        margins = [r["margin"] for r in current]
        record = {"alpha": alpha, "n": n, "strict_parsed": parsed, "parse_rate": parsed / n,
                  "median_margin": median(margins),
                  "mean_delta_margin": sum(r["margin"] - b["margin"] for b, r in zip(baseline, current, strict=True)) / n,
                  "baseline_buy_n": base_buy, "baseline_sell_n": base_sell,
                  "buy_valid_pairs": buy_valid, "sell_valid_pairs": sell_valid,
                  "buy_to_sell": buy_to_sell, "sell_to_buy": sell_to_buy,
                  "buy_to_sell_rate": buy_to_sell / buy_valid if buy_valid else None,
                  "sell_to_buy_rate": sell_to_buy / sell_valid if sell_valid else None,
                  "valid_flip_pairs": len(valid), "flipped": buy_to_sell + sell_to_buy,
                  "flip_rate": (buy_to_sell + sell_to_buy) / len(valid) if valid else None}
        summary.append(record)
    return summary


def dim_alpha_table(summaries: Mapping[int, Sequence[Mapping[str, Any]]], alphas: Sequence[float]) -> str:
    """One metric per table, with alpha as columns and layer as rows."""
    sections: list[str] = []
    def make_table(label: str, render: Any) -> None:
        lines = [f"### {label}", "| layer | " + " | ".join(f"alpha={a:g}" for a in alphas) + " |",
                 "|---|" + "---|" * len(alphas)]
        for layer, rows in sorted(summaries.items()):
            if [r["alpha"] for r in rows] != list(alphas):
                raise ValueError(f"DIM alpha grid incomplete at L{layer}")
            lines.append(f"| L{layer} | " + " | ".join(render(r) for r in rows) + " |")
        sections.append("\n".join(lines))
    make_table("Median fixed-prefix margin (nats)", lambda r: f"{r['median_margin']:+.3f}")
    make_table("Mean paired margin shift ΔM (nats)", lambda r: f"{r['mean_delta_margin']:+.3f}")
    make_table("Strict full-JSON decisions / n", lambda r: f"{r['strict_parsed']}/{r['n']}")
    make_table("Sell→buy / valid alpha-zero sell pairs", lambda r:
               f"{r['sell_to_buy']}/{r['sell_valid_pairs']}" if r["sell_valid_pairs"] else "—")
    make_table("Buy→sell / valid alpha-zero buy pairs", lambda r:
               f"{r['buy_to_sell']}/{r['buy_valid_pairs']}" if r["buy_valid_pairs"] else "—")
    make_table("Any decision flip / all valid pairs", lambda r:
               f"{r['flipped']}/{r['valid_flip_pairs']}" if r["valid_flip_pairs"] else "—")
    return "\n\n".join(sections) + "\n"


def extract_concept_cone_basis(
    model: Any,
    tokenizer: Any,
    heldout_root: Path,
    k_pairs: int = 20,
    k_cone: int = 4,
    exclude_sector: str | None = None,
    *,
    ranked: Sequence[str] | None = None,
    layer: int = HELDOUT_LAYER,
    suffix_length: int = 100,
    companies: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, dict[str, Any]],
    dict[str, torch.Tensor],
    list[str],
    list[str],
]:
    """Extract slice-wise sector-demeaned contrastive SVD basis [100, 2560, k_cone]."""
    if ranked is None:
        cohort_manifest = json.loads((heldout_root / "prepare/cohort_manifest.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (heldout_root / "forward/records.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        clean_records = [r for r in records if r["arm"] == "clean"]
        ticker_margins: dict[str, float] = {}
        company_by_ticker = {c["ticker"]: c for c in cohort_manifest["companies"]}
        for r in clean_records:
            ticker_margins[r["target_ticker"]] = r["target_margin"]
            ticker_margins[r["source_ticker"]] = r["source_margin"]
        ranked_tickers = sorted(ticker_margins.items(), key=lambda x: x[1])
    else:
        if companies is None or any(t not in companies for t in ranked):
            raise ValueError("construction ranking requires all company records")
        company_by_ticker = companies
        ranked_tickers = [(ticker, float(i)) for i, ticker in enumerate(ranked)]
    if exclude_sector:
        ranked_tickers = [
            (t, m) for t, m in ranked_tickers if company_by_ticker[t]["sector"] != exclude_sector
        ]
        print(f"Leave-one-sector-out: excluded sector '{exclude_sector}', {len(ranked_tickers)} companies remain")

    if len(ranked_tickers) < 2 * k_pairs:
        raise ValueError("not enough construction companies for contrast pairs")
    top_tickers = [t for t, _ in ranked_tickers[-k_pairs:]]
    bottom_tickers = [t for t, _ in ranked_tickers[:k_pairs]]

    def get_company_span_state(ticker: str) -> tuple[torch.Tensor, str]:
        c = company_by_ticker[ticker]
        prompt = _render_frozen_prompt(ticker, c["name"], order=0, reverse=False)
        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        c_start, c_end = instruction_char_span(prompt)
        b_start = fmt.find(prompt)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        if span is None or span[1] - span[0] < suffix_length:
            raise ValueError(f"instruction span too short for {ticker}")
        with torch.no_grad():
            res = record_residuals(model, ids, [layer])[layer]
        return res[0, span[1] - suffix_length:span[1], :].float(), c["sector"]

    print(f"Extracting L{layer} instruction-span states for Top {k_pairs} and Bottom {k_pairs} companies...")
    states: dict[str, torch.Tensor] = {}
    sector_map: dict[str, str] = {}
    for t in top_tickers + bottom_tickers:
        s, sec = get_company_span_state(t)
        states[t] = s
        sector_map[t] = sec

    # Sector Demeaning
    sector_groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for t, sec in sector_map.items():
        sector_groups[sec].append(states[t])
    sector_means = {sec: torch.stack(group, dim=0).mean(dim=0) for sec, group in sector_groups.items()}

    deltas: list[torch.Tensor] = []
    for i in range(k_pairs):
        t_top = top_tickers[k_pairs - 1 - i]
        t_bot = bottom_tickers[i]
        demeaned_top = states[t_top] - sector_means[sector_map[t_top]]
        demeaned_bot = states[t_bot] - sector_means[sector_map[t_bot]]
        deltas.append(demeaned_top - demeaned_bot)

    delta_tensor = torch.stack(deltas, dim=0)  # [k_pairs, 100, 2560]

    # Slice-wise SVD; token count and hidden size depend on the model.
    if delta_tensor.shape[1] != suffix_length:
        raise ValueError("inconsistent instruction suffix length")
    cone_bases = torch.zeros(suffix_length, delta_tensor.shape[-1], k_cone, device=model.input_device)
    for p in range(suffix_length):
        D_p = delta_tensor[:, p, :]
        U_p, S_p, Vh_p = torch.linalg.svd(D_p, full_matrices=False)
        Vk_p = Vh_p[:k_cone, :].T
        mean_p = D_p.mean(dim=0)
        for j in range(k_cone):
            cos_j = torch.dot(Vk_p[:, j], mean_p)
            sign_j = 1.0 if cos_j >= 0 else -1.0
            cone_bases[p, :, j] = (sign_j * Vk_p[:, j]).to(model.input_device)

    return cone_bases, company_by_ticker, states, top_tickers, bottom_tickers


def compute_stance_cosine(
    states: dict[str, torch.Tensor],
    top_tickers: list[str],
    bottom_tickers: list[str],
    cone_bases: torch.Tensor,
    k: int = 10,
) -> dict[str, Any]:
    """Per-token cosine between cone axis b1 and the non-demeaned Top/Bottom difference-in-means direction."""
    top_k = top_tickers[-k:]
    bot_k = bottom_tickers[:k]
    v_dim = torch.stack([states[t] for t in top_k], dim=0).mean(dim=0) - torch.stack(
        [states[t] for t in bot_k], dim=0
    ).mean(dim=0)
    b1 = cone_bases[..., 0]
    cos = (b1 * v_dim).sum(dim=-1) / (b1.norm(dim=-1) * v_dim.norm(dim=-1)).clamp_min(1e-8)
    cos_list = [float(x) for x in cos.cpu().tolist()]
    return {
        "k_reference": k,
        "top_reference": top_k,
        "bottom_reference": bot_k,
        "cos_mean": sum(cos_list) / len(cos_list),
        "cos_median": sorted(cos_list)[len(cos_list) // 2],
        "cos_min": min(cos_list),
        "cos_max": max(cos_list),
        "cos_per_token": cos_list,
    }


def persist_cone_basis(
    cone_bases: torch.Tensor,
    centroid: torch.Tensor,
    out_dir: str,
    meta: dict[str, Any],
) -> Path:
    """Persist cone basis + centroid (compact derived steering operator) with provenance. No raw activations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor_path = out / "cone.pt"
    payload = {
        "cone_bases": cone_bases.detach().to(torch.bfloat16).cpu(),
        "centroid": centroid.detach().to(torch.bfloat16).cpu(),
    }
    torch.save(payload, tensor_path)
    prov = {
        "kind": "concept_cone_basis",
        "shape": list(cone_bases.shape),
        "dtype_stored": "bfloat16",
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": meta.get("git_commit"),
        "script": "scripts/probe_concept_cone.py",
        "model": meta.get("model"),
        "heldout_run": meta.get("heldout_run"),
        "k_contrast_pairs": meta.get("k_contrast_pairs"),
        "k_cone": meta.get("k_cone"),
        "leave_out_sector": meta.get("leave_out_sector"),
        "torch_version": torch.__version__,
        "stance_cosine": meta.get("stance_cosine"),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    return tensor_path


def run_cone_evaluation(
    model: Any,
    tokenizer: Any,
    cone_bases: torch.Tensor,
    company_by_ticker: dict[str, dict[str, Any]],
    target_tickers: Sequence[str],
    alphas: Sequence[float],
    evidence_mode: str = "balanced",
    eval_individual_rays: bool = False,
    inject_layer: int | None = None,
    centroid_dims: Sequence[int] | None = None,
    fixed_prefix: bool = False,
    existing_targets: Mapping[str, Any] | None = None,
    on_target: Any = None,
    max_new_tokens: int = 48,
    save_full_text: bool = False,
    operator_label: str | None = None,
    pre_block_all: bool = False,
) -> dict[str, Any]:
    target_gen = InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)
    final_layer = model.n_layers - 1
    k_cone = cone_bases.shape[-1]
    inject_layer = inject_layer if inject_layer is not None else HELDOUT_LAYER

    # Cone centroid ray(s): full centroid by default, or first-j centroids for the dimension ablation
    rays: dict[str, torch.Tensor] = {}
    if centroid_dims:
        for j in centroid_dims:
            if not 1 <= j <= k_cone:
                raise ValueError(f"centroid dim {j} out of range [1, {k_cone}]")
            rays[f"centroid_dim{j}"] = cone_bases[..., :j].sum(dim=-1) / (float(j) ** 0.5)
    else:
        rays["cone_centroid"] = cone_bases.sum(dim=-1) / (float(k_cone) ** 0.5)

    def ray_label(name: str) -> str:
        if name == "cone_centroid":
            return operator_label or f"{k_cone}D Concept Cone Centroid"
        return f"Cone Centroid first-{name.rsplit('dim', 1)[1]} axes"

    results: dict[str, Any] = {
        "k_cone": k_cone,
        "inject_layer": inject_layer,
        "evidence_mode": evidence_mode,
        "targets": dict(existing_targets or {}),
    }

    def eval_ticker_with_ray(
        ticker: str,
        ray_tensor: torch.Tensor,
        ray_label: str,
    ) -> list[dict[str, Any]]:
        c = company_by_ticker.get(ticker, {"name": ticker})
        if evidence_mode == "balanced":
            prompt = _render_frozen_prompt(ticker, c["name"], order=0, reverse=False)
        elif evidence_mode == "zero_evidence":
            prompt = render_custom_prompt(ticker, c["name"], [])
        else:
            bullets = MACRO_SCENARIOS[evidence_mode]
            prompt = render_custom_prompt(ticker, c["name"], bullets)

        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        inst_marker = "Your final response must be a single, valid JSON object."
        c_start = prompt.find(inst_marker)
        c_end = len(prompt)
        b_start = fmt.find(prompt)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        buy_id, sell_id = answer_token_ids(tokenizer, fmt + DECISION_PREFIX)
        if span is None:
            raise ValueError(f"instruction span missing for {ticker}")
        if fixed_prefix:
            if pre_block_all and ray_tensor.ndim != 1:
                raise ValueError("single_all intervention requires one [d_model] direction")
            length = 1 if pre_block_all else ray_tensor.shape[0]
            if span[1] - span[0] < length:
                raise ValueError(f"instruction span too short for {ticker}")
            start, end = span[1] - length, span[1]
            score_ids = torch.tensor([input_ids(tokenizer, fmt + DECISION_PREFIX)],
                                     device=model.input_device, dtype=torch.long)
            if score_ids.shape[1] <= ids.shape[1] or not torch.equal(score_ids[0, start:end], ids[0, start:end]):
                raise ValueError(f"scoring prefix changed instruction tokenization for {ticker}")
        else:
            start, end = span[0], span[1]
            assert (end - start) == 100, f"instruction span length {end - start} != 100"

        rows = []
        print(f"\n========================================================")
        print(f"Subject: {ticker} ({c['name']}) | Ray: {ray_label}")
        print(f"========================================================")

        for a in alphas:
            shift = (a * ray_tensor).to(device=model.input_device)
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                if pre_block_all:
                    # One layer-local vector is added at *all* positions, including cached decode.
                    return tensor + shift.reshape(1, 1, -1).to(dtype=tensor.dtype)
                if tensor.ndim == 3 and tensor.shape[1] == 1: return tensor
                out = tensor.clone()
                out[:, start:end, :] = out[:, start:end, :] + shift.unsqueeze(0)
                return out

            interventions = {inject_layer: transform} if a != 0.0 else None
            if interventions:
                context = pre_residual_interventions if pre_block_all else residual_interventions
                with context(model, interventions):
                    res = record_residuals(model, score_ids if fixed_prefix else ids, [final_layer])[final_layer]
                    seq = generate_tokens(target_gen, ids, gen_config)
            else:
                with torch.no_grad():
                    res = record_residuals(model, score_ids if fixed_prefix else ids, [final_layer])[final_layer]
                    seq = generate_tokens(target_gen, ids, gen_config)

            margin = margin_from_log_probs(fp32_next_token_log_probs(model, res[:, -1, :]), buy_id, sell_id)
            gen_text = tokenizer.decode(seq[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
            match = re.search(r'"decision"\s*:\s*"\s*(buy|sell)\s*"', gen_text, flags=re.IGNORECASE)
            dec = match.group(1).lower() if match else "unparsed"
            if fixed_prefix:
                try:
                    parsed = json.loads(gen_text)
                    dec = parsed["decision"] if isinstance(parsed, dict) and parsed.get("decision") in ("buy", "sell") else "unparsed"
                except (ValueError, TypeError, KeyError):
                    dec = "unparsed"

            print(f"  alpha={a:4.1f} | margin={margin:+.3f} | dec={dec:4s} | {repr(gen_text[:60])}", flush=True)
            row = {"alpha": a, "margin": margin, "decision": dec,
                   "generated_text": gen_text if save_full_text or not fixed_prefix else gen_text[:1024]}
            if fixed_prefix:
                row["parse_ok"] = dec != "unparsed"
            rows.append(row)
        return rows

    # Evaluate cone centroid ray(s) across all target tickers
    for ticker in target_tickers:
        if ticker in results["targets"]:
            continue
        results["targets"][ticker] = {
            ray_name: eval_ticker_with_ray(ticker, ray, ray_label(ray_name)) for ray_name, ray in rays.items()
        }
        if on_target is not None:
            on_target(results)

    # Optionally evaluate individual rays on first ticker
    if eval_individual_rays and target_tickers:
        first = target_tickers[0]
        results["targets"][first]["individual_rays"] = {}
        for j in range(k_cone):
            ray_j = cone_bases[:, :, j]
            results["targets"][first]["individual_rays"][f"b_{j+1}"] = eval_ticker_with_ray(
                first, ray_j, f"Basis Ray b_{j+1}"
            )

    return results


def extract_dim_layer_directions(
    model: Any, tokenizer: Any, companies: Mapping[str, Mapping[str, str]],
    top: Sequence[str], bottom: Sequence[str], layers: Sequence[int], suffix_length: int,
    *, arm: str = "tokenwise",
) -> dict[int, tuple[torch.Tensor, dict[str, float]]]:
    """Reduce 20 clean construction prompts to each layer's own DIM ray."""
    if arm not in ("tokenwise", "single_all"):
        raise ValueError("unknown DIM arm")
    if len(top) != 10 or len(bottom) != 10 or set(top) & set(bottom) or suffix_length < 16:
        raise ValueError("DIM construction groups or instruction suffix invalid")
    samples: dict[int, dict[str, list[torch.Tensor]]] = {layer: {"top": [], "bottom": []} for layer in layers}
    for group, tickers in (("top", top), ("bottom", bottom)):
        for ticker in tickers:
            company = companies[ticker]
            prompt = _render_frozen_prompt(ticker, company["name"], order=0, reverse=False)
            fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
            begin = fmt.find(prompt)
            c_start, c_end = instruction_char_span(prompt)
            if begin < 0:
                raise ValueError(f"chat template omitted prompt for {ticker}")
            span = token_span(tokenizer, fmt, begin + c_start, begin + c_end, add_special_tokens=True)
            ids = torch.tensor([input_ids(tokenizer, fmt)], dtype=torch.long, device=model.input_device)
            if span is None or span[1] - span[0] < suffix_length or span[1] > ids.shape[1]:
                raise ValueError(f"DIM construction instruction span invalid for {ticker}")
            with torch.no_grad():
                states = (record_block_states(model, ids, layers) if arm == "single_all"
                          else record_residuals(model, ids, layers))
            for layer in layers:
                selected = (states[layer]["pre"][0, span[1] - 1, :]
                            if arm == "single_all" else states[layer][0, span[1] - suffix_length:span[1], :])
                samples[layer][group].append(selected.float())
            del states
    directions = {}
    for layer in layers:
        try:
            directions[layer] = fit_dim_direction(torch.stack(samples[layer]["top"]),
                                                  torch.stack(samples[layer]["bottom"]))
        except DegenerateDimDirection as exc:
            raise DegenerateDimDirection(exc.norm, layer) from exc
    return directions


def run_dim_paper(args: argparse.Namespace, *, crossmodel: bool = False) -> None:
    """DIM layer sweep with versioned Qwen and three-model entrypoints."""
    slug = Path(args.model).resolve().name
    spec = DIM_CROSSMODEL_CONFIG.get(slug) if crossmodel else None
    if crossmodel and spec is None:
        raise ValueError("crossmodel DIM requires Gemma4, GLM4 or GPT-OSS")
    layers = list(args.dim_layers)
    allowed_layers = spec["layers"] if spec else DIM_PAPER_LAYERS
    alphas = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if not crossmodel and (slug != "qwen3.5-4b" or args.model_dtype != "bf16"):
        raise ValueError("DIM V1 requires Qwen3.5-4B with bf16 model dtype")
    if spec and args.model_dtype != spec["dtype"]:
        raise ValueError(f"crossmodel DIM requires {spec['dtype']} model dtype for {slug}")
    if (not layers or len(layers) != len(set(layers)) or layers != sorted(layers)
            or not set(layers).issubset(allowed_layers)):
        raise ValueError(f"DIM candidate layers must be an ordered subset of {allowed_layers}")
    if args.alphas != alphas or args.split_seed != 20260923 or args.evidence_mode != "balanced":
        raise ValueError("DIM paper alpha grid, split seed and balanced evidence are frozen")
    if (args.inject_layer is not None or args.eval_individual_rays or args.centroid_dims
            or args.skip_eval or args.leave_out_sector or args.persist_directions):
        raise ValueError("DIM paper mode rejects legacy cone controls")
    if not args.output_json:
        raise ValueError("DIM paper mode requires --output-json")
    out = Path(args.output_json)
    base = Path("artifacts") / slug / "concept-cone-steering" / "runs"
    run_name = out.parent.parent.name
    prefix = "dim-crossmodel-layer-sweep-v1-" if crossmodel else "dim-layer-sweep-v1-"
    if (out.name != "result.json" or len(out.parts) < 4 or out.parent.name != args.dim_arm
            or not run_name.startswith(prefix) or run_name == prefix[:-1]
            or not out.resolve().is_relative_to(base.resolve())):
        raise ValueError(f"DIM output must be artifacts/<model>/concept-cone-steering/runs/{prefix}<id>/<arm>/result.json")
    companies, construction, evaluation = split_population(Path(args.population_csv), args.split_seed)
    smoke = args.smoke_tickers is not None
    tickers = list(args.smoke_tickers) if smoke else evaluation
    if not tickers or len(set(tickers)) != len(tickers) or not set(tickers).issubset(evaluation):
        raise ValueError("DIM smoke/evaluation targets must be distinct members of the 101 held-out companies")
    c2_info = c2_427_layer_source(slug)
    c2_bytes = Path(c2_info["path"]).read_bytes()
    if spec and (hashlib.sha256(c2_bytes).hexdigest() != spec["c2_sha256"]
                 or c2_info["sha256"] != spec["c2_sha256"]):
        raise ValueError("crossmodel C2 source SHA differs from frozen protocol")
    c2_data = json.loads(c2_bytes)
    transfer = (validate_dim_crossmodel_curve(c2_data["curves"]["instruction"], spec) if spec
                else validate_dim_c2_curve(c2_data["curves"]["instruction"]))
    paper_path = base / (spec["source_run"] if spec else "c2-guided-paper-20260924") / "result.json"
    paper_bytes = paper_path.read_bytes()
    if spec and hashlib.sha256(paper_bytes).hexdigest() != spec["source_sha256"]:
        raise ValueError("crossmodel ranking source SHA differs from frozen protocol")
    paper = json.loads(paper_bytes)
    cfg = Path(args.model) / "config.json"
    tokenizer_cfg = Path(args.model) / "tokenizer_config.json"
    weights = sorted(Path(args.model).glob("*.safetensors"))
    if not weights:
        raise ValueError("DIM model checkpoint weights missing")
    prompts = json.dumps([
        (ticker, _render_frozen_prompt(ticker, companies[ticker]["name"], order=0, reverse=False))
        for ticker in sorted(companies)
    ], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    split = json.dumps({"construction": construction, "evaluation": evaluation}, sort_keys=True).encode()
    model, tokenizer, _ = load_model(args.model, dtype="native" if spec and spec["dtype"] == "native" else None)
    if model.n_layers != (spec["n_layers"] if spec else 32) or max(layers) >= model.n_layers:
        raise ValueError("DIM model layer count differs from C2 source")
    expected = {
        "schema": spec["source_schema"] if spec else "concept-cone-sp500-paper-v1",
        "mode": "evaluation", "model": str(Path(args.model).resolve()),
        "model_slug": slug, "model_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(tokenizer_cfg.read_bytes()).hexdigest(),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
        "checkpoint_files": [{"name": path.name, "bytes": path.stat().st_size} for path in weights],
        "population_sha256": hashlib.sha256(Path(args.population_csv).read_bytes()).hexdigest(),
        "split_seed": args.split_seed, "split_sha256": hashlib.sha256(split).hexdigest(),
        "construction_tickers": construction, "evaluation_tickers": evaluation,
        "prompt_family_sha256": hashlib.sha256(prompts).hexdigest(),
        "prompt_renderer": "entity_to_dial.heldout_transfer._render_frozen_prompt(order=0,reverse=False)",
        "decision_prefix": DECISION_PREFIX, "layer": spec["peak"] if spec else 16, "k_pairs": 20, "k_cone": 4,
        "alphas": alphas, "dtype": spec["dtype"] if spec else "bf16", "max_new_tokens": 192,
    }
    if slug != "glm4-9b-0414" or not crossmodel:
        expected["c2_layer_source"] = c2_info
    top, bottom = validate_dim_paper_ranking(
        paper, expected, construction, evaluation,
        source_spec={**spec, "slug": slug} if spec else None)
    if set(tickers) & (set(top) | set(bottom)):
        raise ValueError("DIM evaluation overlaps direction construction")
    suffix_length = prepare_instruction_suffix(tokenizer, companies)
    if suffix_length != (spec["suffix_tokens"] if spec else 100):
        raise ValueError("DIM instruction suffix differs from source cone")
    metadata = {"schema": (f"dim-{'tokenwise' if args.dim_arm == 'tokenwise' else 'single-all'}-crossmodel-v1" if crossmodel
                           else "dim-tokenwise-paper-v1" if args.dim_arm == "tokenwise" else "dim-single-all-paper-v1"),
                "dim_arm": args.dim_arm, "mode": "smoke" if smoke else "evaluation", "model": expected["model"],
                "model_slug": slug, "layers": layers, "alphas": alphas, "construction_tickers": construction,
                "evaluation_tickers": evaluation, "target_tickers": tickers, "top_10": top, "bottom_10": bottom,
                "source_paper_result": str(paper_path), "source_paper_sha256": hashlib.sha256(paper_bytes).hexdigest(),
                "c2_layer_source": c2_info, "c2_instruction_T": {str(l): transfer[l] for l in layers},
                "split_sha256": expected["split_sha256"], "population_sha256": expected["population_sha256"],
                "prompt_family_sha256": expected["prompt_family_sha256"],
                "model_config_sha256": expected["model_config_sha256"],
                "tokenizer_config_sha256": expected["tokenizer_config_sha256"],
                "checkpoint_files": expected["checkpoint_files"], "tokenizer": expected["tokenizer"],
                "decision_prefix": DECISION_PREFIX, "max_new_tokens": 192, "suffix_tokens": suffix_length}
    if spec:
        metadata.update({"model_dtype": spec["dtype"], "protocol": "crossmodel-dim-layer-sweep-v1",
                         "frozen_source_sha256": spec["source_sha256"], "frozen_c2_sha256": spec["c2_sha256"],
                         "source_cone_schema": spec["source_schema"]})
    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
        validate_dim_resume(result, metadata, tickers, layers, alphas)
        if result["complete"]:
            print(f"Already complete: {out}")
            return
    else:
        result = {"metadata": metadata, "direction_diagnostics": {}, "targets": {}, "complete": False}
    try:
        directions = extract_dim_layer_directions(model, tokenizer, companies, top, bottom, layers,
                                                  suffix_length, arm=args.dim_arm)
    except DegenerateDimDirection as exc:
        if crossmodel and args.dim_arm == "single_all" and exc.layer == 0:
            diagnostic = {"schema": "dim-crossmodel-fit-diagnostic-v1", "model_slug": slug,
                          "dim_arm": args.dim_arm, "layer": 0, "difference_norm": exc.norm,
                          "threshold": 1e-8, "status": "fail_closed", "source_paper_result": str(paper_path),
                          "source_paper_sha256": hashlib.sha256(paper_bytes).hexdigest(),
                          "c2_layer_source": c2_info, "suffix_tokens": suffix_length,
                          "top_10": top, "bottom_10": bottom}
            diag_path = out.parent / "l0_fit_diagnostic.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            if diag_path.exists():
                if json.loads(diag_path.read_text(encoding="utf-8")) != diagnostic:
                    raise ValueError(f"DIM L0 diagnostic already exists with different provenance: {diag_path}") from exc
            else:
                diag_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
            print(f"L0 single_all failed closed; compact diagnostic: {diag_path}", flush=True)
        raise
    diagnostics = {str(layer): {**norms, "unit_sha256": hashlib.sha256(
        ray.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
                   for layer, (ray, norms) in directions.items()}
    if result.get("direction_diagnostics") not in ({}, diagnostics):
        raise ValueError("DIM directions changed during resume")
    result["direction_diagnostics"] = diagnostics
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(out)

    save()
    for layer in layers:
        ray = directions[layer][0].unsqueeze(-1)
        for ticker in tickers:
            existing = result["targets"].get(ticker, {})
            if f"L{layer}" in existing:
                continue
            evaluated = run_cone_evaluation(
                model, tokenizer, ray, companies, [ticker], alphas, inject_layer=layer,
                evidence_mode="balanced", fixed_prefix=True, max_new_tokens=192,
                save_full_text=True, operator_label=f"L{layer} {args.dim_arm} DIM",
                pre_block_all=args.dim_arm == "single_all")
            result["targets"].setdefault(ticker, {})[f"L{layer}"] = {
                "rows": evaluated["targets"][ticker]["cone_centroid"]}
            save()
    validate_dim_resume(result, metadata, tickers, layers, alphas)
    if crossmodel:
        validate_dim_baseline_consistency(result["targets"], tickers, layers)
    result["summary"] = {str(layer): dim_layer_summary(result["targets"], tickers, layer, alphas)
                         for layer in layers}
    result["complete"] = True
    validate_dim_resume(result, metadata, tickers, layers, alphas)
    save()
    method = ("Pre-block one-vector addition at all prompt and decode tokens"
              if args.dim_arm == "single_all" else f"Post-block token-wise addition at the {suffix_length}-token instruction suffix")
    report = (f"# {slug if crossmodel else 'Qwen3.5-4B'} {args.dim_arm} DIM layer sweep ({'smoke' if smoke else '101-company evaluation'})\n\n"
              f"{method}. Descriptive only: no random-direction control or independent confirmation; "
              "identical alpha is not an equal whole-sequence injection dose across arms.\n\n"
              + dim_alpha_table({layer: result["summary"][str(layer)] for layer in layers}, alphas))
    out.with_suffix(".md").write_text(report, encoding="utf-8")
    print(f"DIM complete: {out}")


def run_sp500_v1(args: argparse.Namespace, *, paper: bool = False) -> None:
    """Independent model-local ranking, fit, and disjoint evaluation."""
    slug = Path(args.model).resolve().name
    layers = PAPER_C2_LAYERS if paper else C2_LAYERS
    if slug not in layers:
        raise ValueError(f"C2 layer unknown for {slug}; supported: {sorted(layers)}")
    layer = layers[slug]
    source = c2_427_layer_source(slug) if paper else None
    if args.inject_layer is not None and args.inject_layer != layer:
        raise ValueError(f"C2 {slug} must use L{layer} for fit and injection")
    if slug == "gpt-oss-20b" and args.model_dtype != "native":
        raise ValueError("GPT-OSS needs --model-dtype native to preserve packed weights")
    if args.k_contrast_pairs != 20 or args.k_cone_dim != 4 or args.evidence_mode != "balanced":
        raise ValueError("sp500_v1 freezes 20 contrast pairs, 4D cone, and balanced evidence")
    if args.eval_individual_rays or args.centroid_dims or args.skip_eval or args.leave_out_sector or args.persist_directions:
        raise ValueError("sp500_v1 does not support legacy ray/skip/sector/persistence options")
    alphas = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if list(args.alphas) != alphas:
        raise ValueError(f"sp500_v1 alpha grid is frozen to {alphas}")
    if not args.output_json:
        raise ValueError("sp500_v1 requires --output-json for compact provenance and restart")
    out = Path(args.output_json)
    if out.suffix != ".json":
        raise ValueError("--output-json must have .json suffix")
    if "artifacts" not in out.parts or slug not in out.parts or "concept-cone-steering" not in out.parts:
        raise ValueError("output must be under artifacts/<model>/concept-cone-steering/")
    if paper and not any(part.startswith("c2-guided-paper-") for part in out.parts):
        raise ValueError("paper output requires a c2-guided-paper-<run> directory")
    population = Path(args.population_csv)
    companies, construction, evaluation = split_population(population, args.split_seed)
    smoke = args.smoke_tickers is not None
    if smoke:
        if len(set(args.smoke_tickers)) != len(args.smoke_tickers) or any(t not in companies for t in args.smoke_tickers):
            raise ValueError("invalid smoke tickers")
        targets = list(args.smoke_tickers)
        # Even smoke targets must be excluded from construction.
        construction = [t for t in construction if t not in targets]
    else:
        targets = evaluation
    split_payload = json.dumps({"construction": construction, "evaluation": evaluation}, sort_keys=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cfg = Path(args.model) / "config.json"
    prompt_bytes = json.dumps([
        (ticker, _render_frozen_prompt(ticker, companies[ticker]["name"], order=0, reverse=False))
        for ticker in sorted(companies)
    ], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    weights = sorted(Path(args.model).glob("*.safetensors"))
    if not weights:
        raise ValueError(f"no local checkpoint weights for {slug}")
    metadata = {
        "schema": "concept-cone-sp500-paper-v1" if paper else "concept-cone-sp500-v1",
        "mode": "smoke" if smoke else "evaluation",
        "prompt_family_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "checkpoint_files": [{"name": p.name, "bytes": p.stat().st_size} for p in weights],
        "model": str(Path(args.model).resolve()), "model_slug": slug,
        "model_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256((Path(args.model) / "tokenizer_config.json").read_bytes()).hexdigest(),
        "population": str(population.resolve()),
        "population_sha256": hashlib.sha256(population.read_bytes()).hexdigest(),
        "split_seed": args.split_seed, "split_sha256": hashlib.sha256(split_payload.encode()).hexdigest(),
        "construction_tickers": construction, "evaluation_tickers": evaluation,
        "target_tickers": targets, "layer": layer, "k_pairs": 20, "k_cone": 4,
        "alphas": alphas, "dtype": args.model_dtype, "git_commit": commit,
        "torch_version": torch.__version__,
        "prompt_renderer": "entity_to_dial.heldout_transfer._render_frozen_prompt(order=0,reverse=False)",
        "decision_prefix": DECISION_PREFIX, "max_new_tokens": 192,
    }
    if paper:
        metadata["c2_layer_source"] = source
    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
        existing_meta = dict(result.get("metadata") or {})
        existing_meta.pop("tokenizer", None)
        if existing_meta != metadata:
            raise ValueError(f"cannot resume: metadata differs from {out}")
        if result.get("complete"):
            print(f"Already complete: {out}")
            return
    else:
        result = {"metadata": metadata, "ranking": None, "effective_tokens": None,
                  "targets": {}, "complete": False}
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)

    model, tokenizer, _ = load_model(args.model, dtype="native" if args.model_dtype == "native" else None)
    if not 0 <= layer < model.n_layers:
        raise ValueError(f"C2 layer L{layer} invalid for model with {model.n_layers} layers")
    length = prepare_instruction_suffix(tokenizer, companies)
    if result["effective_tokens"] not in (None, length):
        raise ValueError("instruction token alignment changed since previous run")
    result["effective_tokens"] = length
    metadata["tokenizer"] = str(getattr(tokenizer, "name_or_path", "unknown"))
    if result["metadata"].get("tokenizer", metadata["tokenizer"]) != metadata["tokenizer"]:
        raise ValueError("tokenizer identity changed during run")
    result["metadata"] = metadata
    if result["ranking"] is None:
        result["ranking"] = rank_construction(model, tokenizer, companies, construction)
        save()
    ranking = result["ranking"]
    if [r["ticker"] for r in ranking] != [r["ticker"] for r in sorted(ranking, key=lambda r: (r["margin"], r["ticker"]))]:
        raise ValueError("ranking order corrupt")
    if {r["ticker"] for r in ranking} != set(construction):
        raise ValueError("ranking population differs from construction set")
    ranked_tickers = [r["ticker"] for r in ranking]
    cone, _, _, top, bottom = extract_concept_cone_basis(
        model, tokenizer, Path(args.heldout_run), k_pairs=20, k_cone=4,
        ranked=ranked_tickers, companies=companies, layer=layer, suffix_length=length,
    )
    if set(targets) & (set(top) | set(bottom)):
        raise ValueError("evaluation target overlaps cone construction")
    result["top_20"] = top
    result["bottom_20"] = bottom
    norms = (cone.sum(dim=-1) / 2.0).norm(dim=-1)
    result["centroid_norms"] = [float(x) for x in norms.cpu().tolist()]
    save()

    def checkpoint(evaluated: Mapping[str, Any]) -> None:
        result["targets"] = evaluated["targets"]
        save()

    evaluated = run_cone_evaluation(model, tokenizer, cone, companies, targets, alphas,
                                     evidence_mode="balanced", inject_layer=layer,
                                     fixed_prefix=True, existing_targets=result["targets"], on_target=checkpoint,
                                     max_new_tokens=192)
    result["targets"] = evaluated["targets"]
    if set(result["targets"]) != set(targets):
        raise ValueError("evaluation incomplete")
    result["summary"] = alpha_summary(result, targets, alphas)
    result["complete"] = True
    save()
    out.with_suffix(".md").write_text(
        f"# {slug} Concept Cone (" + ("smoke" if smoke else "101-company evaluation") + ")\n\n"
        + "Cell: fixed-prefix margin (greedy JSON decision). Raw alpha is model-local.\n\n"
        + alpha_table(result, targets, alphas), encoding="utf-8")
    print(f"Completed {len(targets)} tickers; table: {out.with_suffix('.md')}")


def main() -> None:
    args = parse_args()
    if args.cohort_mode in ("sp500_dim_paper", "sp500_dim_crossmodel"):
        run_dim_paper(args, crossmodel=args.cohort_mode == "sp500_dim_crossmodel")
        return
    if args.cohort_mode in ("sp500_v1", "sp500_paper"):
        run_sp500_v1(args, paper=args.cohort_mode == "sp500_paper")
        return
    print(f"Loading model from {args.model}...")
    model, tokenizer, _ = load_model(args.model, dtype=None)

    heldout_root = Path(args.heldout_run)
    cone_bases, company_by_ticker, states, top_tickers, bottom_tickers = extract_concept_cone_basis(
        model,
        tokenizer,
        heldout_root,
        k_pairs=args.k_contrast_pairs,
        k_cone=args.k_cone_dim,
        exclude_sector=args.leave_out_sector,
    )

    results: dict[str, Any] = {}

    if args.measure_stance_cosine:
        cosine = compute_stance_cosine(states, top_tickers, bottom_tickers, cone_bases)
        results["stance_cosine"] = cosine
        print(
            f"\nStance cosine b1 vs v_DIM (k={cosine['k_reference']}): "
            f"mean={cosine['cos_mean']:.4f} median={cosine['cos_median']:.4f} "
            f"min={cosine['cos_min']:.4f} max={cosine['cos_max']:.4f}"
        )

    if args.persist_directions:
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            git_commit = None
        w_full = cone_bases.sum(dim=-1) / (float(args.k_cone_dim) ** 0.5)
        persist_cone_basis(
            cone_bases,
            w_full,
            args.persist_directions,
            {
                "git_commit": git_commit,
                "model": args.model,
                "heldout_run": args.heldout_run,
                "k_contrast_pairs": args.k_contrast_pairs,
                "k_cone": args.k_cone_dim,
                "leave_out_sector": args.leave_out_sector,
                "stance_cosine": results.get("stance_cosine"),
            },
        )
        print(f"Persisted cone basis to {args.persist_directions}")

    target_tickers = args.target_tickers
    if target_tickers == ["all"]:
        manifest = json.loads((heldout_root / "prepare/cohort_manifest.json").read_text(encoding="utf-8"))
        target_tickers = [c["ticker"] for c in manifest["companies"]]
        print(f"Expanding --target-tickers all to {len(target_tickers)} cohort tickers")

    if args.skip_eval:
        results.update({"k_cone": args.k_cone_dim, "evidence_mode": args.evidence_mode, "targets": {}})
        print("Skipping evaluation (--skip-eval)")
    else:
        results.update(
            run_cone_evaluation(
                model,
                tokenizer,
                cone_bases,
                company_by_ticker,
                target_tickers,
                args.alphas,
                evidence_mode=args.evidence_mode,
                eval_individual_rays=args.eval_individual_rays,
                inject_layer=args.inject_layer,
                centroid_dims=args.centroid_dims,
            )
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
