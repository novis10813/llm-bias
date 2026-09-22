"""Entity-to-Dial held-out transfer V1 workflow.

This module owns the V1 deterministic cohort/pairing preparation, compact
fixed-margin transfer analysis, and `prepare → forward → analyze → finalize`
workflow. Raw model states remain transient and are never serialized.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

import csv

from llm_bias.core.artifacts.io import write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import record_block_states, residual_interventions
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids, token_span

from .analysis import toward_source_delta
from .block_patch import nearest_position_mapping
from .dial_probe import answer_token_ids, margin_from_log_probs, scoring_ids
from .joint_patch import load_pca_basis, make_full_transform, make_projected_transplant
from .spans import instruction_char_span
from .template import EVIDENCE_CLOSE, EVIDENCE_MARKER, NAME_LINE_PREFIX, TICKER_LINE_PREFIX

_EVIDENCE_ORDERS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Q3 revenue grew 14% year over year, exceeding analyst consensus by 3%.",
        "Free cash flow reached a record quarterly high, up 22% year over year.",
        "Gross margin contracted 300 basis points due to rising input costs.",
        "Full-year revenue guidance was revised downward by 6%.",
    ),
    (
        "Full-year revenue guidance was revised downward by 6%.",
        "Gross margin contracted 300 basis points due to rising input costs.",
        "Free cash flow reached a record quarterly high, up 22% year over year.",
        "Q3 revenue grew 14% year over year, exceeding analyst consensus by 3%.",
    ),
)


def _render_frozen_prompt(ticker: str, name: str, *, order: int, reverse: bool) -> str:
    """Render the existing shared-evidence Entity-to-Dial prompt family.

    The package's current template module exposes the frozen markers but not a
    renderer; keeping this compatibility renderer here avoids importing the
    unrelated balanced-evidence experiment package.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}")
    options = '"sell" or "buy"' if reverse else '"buy" or "sell"'
    evidence = "\n".join(f"- {item}" for item in _EVIDENCE_ORDERS[order])
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"{TICKER_LINE_PREFIX}{ticker}]\n\n{NAME_LINE_PREFIX}{name}]\n\n"
        f"{EVIDENCE_MARKER}\n\n{evidence}\n\n"
        f"{EVIDENCE_CLOSE[2:]}"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        f'"decision": {options}\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )

HELDOUT_TRANSFER_SCHEMA = "entity-to-dial-heldout-transfer-v1"
HELDOUT_PROTOCOL = "docs/entity-to-dial/details/proposal-heldout-transfer-v1.md"
HELDOUT_MODEL_SLUG = "qwen3.5-4b"
HELDOUT_COHORT_SIZE = 200
HELDOUT_SELECTION_SEED = 20260930
HELDOUT_RANDOM_BASIS_COUNT = 4
HELDOUT_EVAL_VARIANTS = ((True, 0), (False, 1), (True, 1))
HELDOUT_SELECTION_VARIANT = (False, 0)
HELDOUT_LAYER = 15
HELDOUT_BASIS_K = 8
HELDOUT_RATIO_MIN_FULL_DM = 0.2
HELDOUT_RECOVERY_TARGET = 0.8
HELDOUT_BOOTSTRAP_SAMPLES = 10_000
HELDOUT_BOOTSTRAP_SEED = 20260930
HELDOUT_NOOP_TOLERANCE = 1e-12

CONSTRUCTION_TICKERS: tuple[str, ...] = (
    "AMAT", "GLW", "HPE", "IT", "AXP", "BLK", "C", "GS",
    "ABT", "BDX", "DHR", "SYK", "CSX", "DE", "HON", "NSC",
)

M6_EXCLUSION_MANIFEST_SCHEMA = "selective-intervention-m6-external-v1"
_REQUIRED_M6_FIELDS = ("ticker", "name", "sector")
_REQUIRED_PROMPT_FIELDS = (
    "prompt_id", "ticker", "name", "sector", "reverse", "order", "role",
    "formatted", "instruction_span", "formatted_sha256", "chat_template_sha256",
    "tokenizer_name_or_path", "token_count",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_population_contract(path: str | Path) -> list[dict[str, str]]:
    """Local copy of the canonical 503-row CSV contract.

    The existing population helper belongs to another experiment package; this
    package deliberately keeps the contract local rather than importing it.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: row.get(key, "").strip() for key in ("ticker", "company_name", "gics_sector")}
            for row in csv.DictReader(handle)
            if row.get("index_name") == "S&P 500" and row.get("year") == "2024"
        ]
    if len(rows) != 503:
        raise ValueError(f"expected 503 S&P 500 2024 rows, got {len(rows)}")
    tickers = [row["ticker"] for row in rows]
    if len(set(tickers)) != len(tickers):
        raise ValueError("population tickers must be unique")
    if any(not row["company_name"] for row in rows):
        raise ValueError("population company names must be non-empty")
    if any(not row["ticker"] or not row["gics_sector"] for row in rows):
        raise ValueError("population ticker and sector fields must be non-empty")
    return sorted(rows, key=lambda row: row["ticker"])


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _manifest_payload_and_hash(manifest: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], str]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, _sha256_file(path)
    payload = dict(manifest)
    return payload, _sha256_bytes(_canonical_json(payload))


def _company_fields(row: Mapping[str, Any]) -> dict[str, str]:
    ticker = str(row.get("ticker", "")).strip()
    name = str(row.get("name", row.get("company_name", ""))).strip()
    sector = str(row.get("sector", row.get("gics_sector", ""))).strip()
    if not ticker or not name or not sector:
        raise ValueError(f"company row requires non-empty ticker, name, sector: {row!r}")
    return {"ticker": ticker, "name": name, "sector": sector}


def validate_m6_exclusion_manifest(
    manifest: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate and normalize the twelve-company M6 exclusion manifest.

    Only the local schema contract is checked.  In particular, a historical
    population hash recorded by M6 is provenance, not a hash to compare with
    the held-out population CSV.
    """
    payload, digest = _manifest_payload_and_hash(manifest)
    if payload.get("schema_version") != M6_EXCLUSION_MANIFEST_SCHEMA:
        raise ValueError("M6 manifest schema_version is invalid")
    companies = payload.get("companies")
    if not isinstance(companies, list) or len(companies) != 12:
        raise ValueError("M6 manifest must contain exactly 12 company objects")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in companies:
        if not isinstance(item, Mapping):
            raise ValueError("M6 company entries must be objects")
        values = _company_fields(item)
        if values["ticker"] in seen:
            raise ValueError(f"duplicate M6 ticker: {values['ticker']}")
        seen.add(values["ticker"])
        normalized.append(values)
    return {
        "schema_version": payload.get("schema_version"),
        "companies": normalized,
        "m6_manifest_sha256": digest,
        "m6_tickers": sorted(seen),
    }


def _load_e01_basis(e01: str | Path | Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(e01, Mapping):
        payload = dict(e01)
        manifest = payload.get("manifest", payload)
        summary = payload.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("E-01 summary is unavailable")
        vectors = summary.get("pca_basis_vectors")
        values = summary.get("pca_singular_values")
        if not isinstance(vectors, list) or not isinstance(values, list):
            raise ValueError("E-01 summary is invalid")
        # Keep validation identical to the persisted loader without creating a
        # temporary artifact.  The loader is the canonical shape contract.
        basis_rows = torch.tensor(vectors, dtype=torch.float64)
        singular = torch.tensor(values, dtype=torch.float64)
        if basis_rows.ndim != 2 or basis_rows.shape[0] != 16:
            raise ValueError("E-01 basis must contain 16 rows")
        if singular.ndim != 1 or singular.shape[0] != 16:
            raise ValueError("E-01 singular values must contain 16 values")
        if not torch.isfinite(basis_rows).all() or not torch.isfinite(singular).all():
            raise ValueError("E-01 basis and singular values must be finite")
        if bool((singular <= 0).any()) or bool((singular[1:] > singular[:-1] + 1e-9).any()):
            raise ValueError("E-01 singular values must be positive and decreasing")
        if not torch.allclose(basis_rows @ basis_rows.T, torch.eye(16, dtype=torch.float64), atol=1e-6):
            raise ValueError("E-01 basis rows must be orthonormal")
        if manifest.get("status") != "complete":
            raise ValueError("E-01 manifest must be complete")
        return basis_rows.T.contiguous().float()[:, :HELDOUT_BASIS_K], {
            "manifest_status": "complete",
            "summary_schema": "entity-to-dial-e-01",
        }

    root = Path(e01)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if not manifest_path.is_file():
        raise FileNotFoundError(f"E-01 manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("E-01 manifest must be complete")
    summary_path = root / "analyze" / "summary.json" if root.is_dir() else root
    if not summary_path.is_file():
        raise FileNotFoundError(f"E-01 summary not found: {summary_path}")
    basis, _singular = load_pca_basis(summary_path, expected_k=16)
    return basis[:, :HELDOUT_BASIS_K].contiguous().cpu(), {
        "manifest_sha256": _sha256_file(manifest_path),
        "summary_sha256": _sha256_file(summary_path),
        "manifest_status": manifest["status"],
    }


def verify_heldout_inputs(
    e01_run: str | Path | Mapping[str, Any],
    m6_manifest: Mapping[str, Any] | str | Path,
    population_path: str | Path,
) -> dict[str, Any]:
    """Validate frozen inputs and return compact provenance plus CPU ``V8``."""
    population = _load_population_contract(population_path)
    population_sha256 = _sha256_file(population_path)
    m6 = validate_m6_exclusion_manifest(m6_manifest)
    overlap = set(CONSTRUCTION_TICKERS) & set(m6["m6_tickers"])
    if overlap:
        raise ValueError(f"construction and M6 exclusions overlap: {sorted(overlap)}")
    basis, e01_provenance = _load_e01_basis(e01_run)
    return {
        "schema_version": HELDOUT_TRANSFER_SCHEMA,
        "protocol": HELDOUT_PROTOCOL,
        "population": {
            "path": str(population_path),
            "sha256": population_sha256,
            "n_rows": len(population),
        },
        "exclusions": {
            "construction_tickers": list(CONSTRUCTION_TICKERS),
            "m6_manifest_sha256": m6["m6_manifest_sha256"],
            "m6_tickers": m6["m6_tickers"],
        },
        "e01": e01_provenance,
        "v8": basis,
    }


def prepare_heldout_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    m6_manifest: Mapping[str, Any] | str | Path,
    population_path: str | Path,
    selection_seed: int = HELDOUT_SELECTION_SEED,
    cohort_size: int = HELDOUT_COHORT_SIZE,
) -> dict[str, Any]:
    """Build the deterministic proportional cohort manifest.

    ``cohort_size`` is intentionally a helper-level injection for deterministic
    fake-model tests.  The workflow's default remains the frozen 200-company
    cohort.
    """
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")
    canonical = _load_population_contract(population_path)
    supplied = [_company_fields(row) for row in rows]
    if {r["ticker"] for r in supplied} != {r["ticker"] for r in canonical}:
        raise ValueError("supplied rows do not match the canonical population")
    by_ticker = {r["ticker"]: r for r in supplied}
    canonical = [by_ticker[r["ticker"]] for r in canonical]
    m6 = validate_m6_exclusion_manifest(m6_manifest)
    m6_tickers = set(m6["m6_tickers"])
    overlap = set(CONSTRUCTION_TICKERS) & m6_tickers
    if overlap:
        raise ValueError(f"construction and M6 exclusions overlap: {sorted(overlap)}")
    excluded = set(CONSTRUCTION_TICKERS) | m6_tickers
    eligible = [r for r in canonical if r["ticker"] not in excluded]
    if len(eligible) < cohort_size:
        raise ValueError(f"eligible population has {len(eligible)} rows, need {cohort_size}")
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in eligible:
        buckets.setdefault(row["sector"], []).append(row)
    sectors = sorted(buckets)
    raw = {s: cohort_size * len(buckets[s]) / len(eligible) for s in sectors}
    quotas = {s: math.floor(raw[s]) for s in sectors}
    remainder = cohort_size - sum(quotas.values())
    for sector in sorted(sectors, key=lambda s: (-(raw[s] - quotas[s]), s))[:remainder]:
        quotas[sector] += 1
    selected: list[dict[str, str]] = []
    for index, sector in enumerate(sectors):
        bucket = list(buckets[sector])
        local_seed = int.from_bytes(
            hashlib.sha256(f"{selection_seed}:{index}:{sector}".encode()).digest()[:8], "big"
        )
        random.Random(local_seed).shuffle(bucket)
        selected.extend(bucket[:quotas[sector]])
    selected.sort(key=lambda row: row["ticker"])
    if len(selected) != cohort_size or len({r["ticker"] for r in selected}) != cohort_size:
        raise AssertionError(f"cohort selection did not produce {cohort_size} unique companies")
    return {
        "schema_version": HELDOUT_TRANSFER_SCHEMA,
        "selection_seed": int(selection_seed),
        "population": {
            "path": str(population_path),
            "sha256": _sha256_file(population_path),
            "n_rows": len(canonical),
        },
        "cohort_size": int(cohort_size),
        "exclusions": {
            "construction_tickers": list(CONSTRUCTION_TICKERS),
            "m6_manifest_sha256": m6["m6_manifest_sha256"],
            "m6_tickers": m6["m6_tickers"],
            "eligible_ticker_count": len(eligible),
        },
        "sector_counts": {
            sector: {"eligible": len(buckets[sector]), "quota": quotas[sector]}
            for sector in sectors
        },
        "companies": selected,
        "selected_ticker_sha256": _sha256_bytes("\n".join(r["ticker"] for r in selected).encode()),
        "raw_runtime_payloads": False,
    }


def prepare_heldout_prompt_records(cohort_manifest: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    """Render exactly four frozen prompt records per cohort company."""
    companies = cohort_manifest.get("companies")
    if not isinstance(companies, list):
        raise ValueError("cohort manifest companies must be a list")
    records: list[dict[str, Any]] = []
    for company in companies:
        values = _company_fields(company)
        for reverse, order in (HELDOUT_SELECTION_VARIANT, *HELDOUT_EVAL_VARIANTS):
            prompt = _render_frozen_prompt(values["ticker"], values["name"], order=order, reverse=reverse)
            formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
            if not formatted:
                raise ValueError("formatted prompt body must be non-empty")
            body_start = formatted.find(prompt)
            if body_start < 0:
                raise ValueError("formatted prompt body does not contain rendered prompt")
            char_start, char_end = instruction_char_span(prompt)
            instruction = token_span(
                tokenizer, formatted, body_start + char_start, body_start + char_end,
                add_special_tokens=True,
            )
            if instruction is None or instruction[1] <= instruction[0]:
                raise ValueError("instruction span is empty")
            count = len(input_ids(tokenizer, formatted, add_special_tokens=True))
            if count <= 0:
                raise ValueError("token_count must be positive")
            template = getattr(tokenizer, "chat_template", None)
            template_value = template if isinstance(template, str) else repr(template)
            records.append({
                "prompt_id": f"{values['ticker']}:rev{int(reverse)}:ord{order}",
                **values,
                "reverse": bool(reverse),
                "order": int(order),
                "role": "selection" if (not reverse and order == 0) else "evaluation",
                "formatted": formatted,
                "instruction_span": [int(instruction[0]), int(instruction[1])],
                "formatted_sha256": _sha256_bytes(formatted.encode()),
                "chat_template_sha256": _sha256_bytes(template_value.encode()),
                "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
                "token_count": count,
            })
    validate_prompt_roles(records, cohort_manifest)
    return records


def validate_prompt_roles(records: Sequence[Mapping[str, Any]], cohort_manifest: Mapping[str, Any] | None = None) -> None:
    """Validate the four-record split and reject malformed prompt metadata."""
    expected = {(False, 0): "selection", (True, 0): "evaluation", (False, 1): "evaluation", (True, 1): "evaluation"}
    by_ticker: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        missing = [key for key in _REQUIRED_PROMPT_FIELDS if key not in record]
        if missing:
            raise ValueError(f"prompt record missing fields: {missing}")
        key = (bool(record["reverse"]), int(record["order"]))
        if key not in expected or record["role"] != expected[key]:
            raise ValueError("invalid selection/evaluation prompt role")
        span = record["instruction_span"]
        if not isinstance(span, (list, tuple)) or len(span) != 2 or int(span[1]) <= int(span[0]):
            raise ValueError("instruction span must be non-empty")
        if not record["formatted"] or int(record["token_count"]) <= 0:
            raise ValueError("formatted body and token_count must be non-empty/positive")
        by_ticker.setdefault(str(record["ticker"]), []).append(record)
    if cohort_manifest is not None:
        expected_tickers = {str(row["ticker"]) for row in cohort_manifest.get("companies", [])}
        if set(by_ticker) != expected_tickers:
            raise ValueError("prompt records do not match cohort tickers")
    for ticker, values in by_ticker.items():
        keys = [(bool(r["reverse"]), int(r["order"])) for r in values]
        if len(values) != 4 or len(set(keys)) != 4 or set(keys) != set(expected):
            raise ValueError(f"ticker {ticker} must have exactly one of each four prompt variants")


def _selection_rows(selection_records: Sequence[Mapping[str, Any]], cohort_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the graph's selection-only input, without requiring eval rows."""
    companies = cohort_manifest.get("companies")
    if not isinstance(companies, list):
        raise ValueError("cohort manifest companies must be a list")
    rows = [dict(r) for r in selection_records]
    if len(rows) != len(companies):
        raise ValueError("graph input must contain exactly one selection record per company")
    expected_tickers = {str(row["ticker"]) for row in companies}
    if any(r.get("role") != "selection" or r.get("reverse") is not False or r.get("order") != 0 for r in rows):
        raise ValueError("graphs require selection-role records only")
    if {str(r.get("ticker")) for r in rows} != expected_tickers:
        raise ValueError("selection records do not match cohort tickers")
    if any(not isinstance(r.get("margin"), (int, float)) or not math.isfinite(float(r["margin"])) for r in rows):
        raise ValueError("selection margins must be finite")
    if len({r["ticker"] for r in rows}) != len(rows):
        raise ValueError("selection tickers must be unique")
    return rows


def _graph_base(stratum: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": HELDOUT_TRANSFER_SCHEMA,
        "stratum": stratum,
        "selection_margin_provenance": [
            {"ticker": r["ticker"], "sector": r["sector"], "margin": float(r["margin"])}
            for r in sorted(rows, key=lambda r: r["ticker"])
        ],
        "pairs": [],
        "unmatched_tickers": [],
        "raw_runtime_payloads": False,
    }


def build_sector_within_graph(selection_records: Sequence[Mapping[str, Any]], cohort_manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = _selection_rows(selection_records, cohort_manifest)
    graph = _graph_base("sector_within", rows)
    for sector in sorted({r["sector"] for r in rows}):
        bucket = sorted((r for r in rows if r["sector"] == sector), key=lambda r: (float(r["margin"]), r["ticker"]))
        for j in range(len(bucket) // 2):
            low, high = bucket[j], bucket[-1 - j]
            graph["pairs"].append({
                "pair_id": f"within:{sector.lower().replace(' ', '-')}:{high['ticker']}:{low['ticker']}",
                "high_ticker": high["ticker"], "low_ticker": low["ticker"],
                "high_margin": float(high["margin"]), "low_margin": float(low["margin"]),
                "high_sector": sector, "low_sector": sector, "stratum": "sector_within",
            })
        if len(bucket) % 2:
            graph["unmatched_tickers"].append(bucket[len(bucket) // 2]["ticker"])
    graph["pairs"].sort(key=lambda p: p["pair_id"])
    used = [t for pair in graph["pairs"] for t in (pair["high_ticker"], pair["low_ticker"])]
    accounted = used + list(graph["unmatched_tickers"])
    expected = [str(row["ticker"]) for row in rows]
    if len(accounted) != len(set(accounted)) or set(accounted) != set(expected):
        raise ValueError("sector-within graph must account for each ticker exactly once")
    return graph


def build_cross_sector_graph(
    selection_records: Sequence[Mapping[str, Any]],
    cohort_manifest: Mapping[str, Any],
    *,
    cohort_size: int | None = None,
) -> dict[str, Any]:
    rows = _selection_rows(selection_records, cohort_manifest)
    expected_size = (
        int(cohort_manifest.get("cohort_size", HELDOUT_COHORT_SIZE))
        if cohort_size is None
        else int(cohort_size)
    )
    if expected_size <= 0 or expected_size % 2 or len(rows) != expected_size:
        raise ValueError(f"cross-sector graph requires exactly {expected_size} selection records")
    half = expected_size // 2
    ordered = sorted(rows, key=lambda r: (float(r["margin"]), r["ticker"]))
    targets = ordered[:half]
    sources = sorted(ordered[half:], key=lambda r: (-float(r["margin"]), r["ticker"]))
    target_order = sorted(targets, key=lambda r: (float(r["margin"]), r["ticker"]))
    match: dict[str, Mapping[str, Any]] = {}
    def visit(target: Mapping[str, Any], seen: set[str]) -> bool:
        for source in sources:
            st = str(source["ticker"])
            if st in seen or source["sector"] == target["sector"]:
                continue
            seen.add(st)
            prior = match.get(st)
            if prior is None or visit(prior, seen):
                match[st] = target
                return True
        return False
    if any(not visit(target, set()) for target in target_order) or len(match) != half:
        raise ValueError(f"cross-sector graph does not have a perfect {half}-edge matching")
    graph = _graph_base("cross_sector", rows)
    for source_ticker, target in match.items():
        source = next(r for r in sources if r["ticker"] == source_ticker)
        high, low = source, target
        graph["pairs"].append({
            "pair_id": f"cross:{high['ticker']}:{low['ticker']}",
            "high_ticker": high["ticker"], "low_ticker": low["ticker"],
            "high_margin": float(high["margin"]), "low_margin": float(low["margin"]),
            "high_sector": high["sector"], "low_sector": low["sector"], "stratum": "cross_sector",
        })
    graph["pairs"].sort(key=lambda p: p["pair_id"])
    used = [t for pair in graph["pairs"] for t in (pair["high_ticker"], pair["low_ticker"])]
    if len(used) != expected_size or len(set(used)) != expected_size:
        raise ValueError("cross-sector graph must use every ticker exactly once")
    if any(p["high_sector"] == p["low_sector"] for p in graph["pairs"]):
        raise ValueError("cross-sector graph contains a same-sector edge")
    return graph


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return _sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def random_orthonormal_bases(
    width: int,
    *,
    count: int = HELDOUT_RANDOM_BASIS_COUNT,
    k: int = HELDOUT_BASIS_K,
    seed: int = HELDOUT_SELECTION_SEED,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Create reproducible CPU float32 bases with canonical QR signs."""
    if width <= 0 or count <= 0 or k <= 0:
        raise ValueError("width, count, and k must be positive")
    if k > width:
        raise ValueError("k must not exceed width")
    bases: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for index in range(count):
        current_seed = int(seed) + index
        generator = torch.Generator(device="cpu").manual_seed(current_seed)
        matrix = torch.randn((width, k), dtype=torch.float64, generator=generator)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        for column in range(k):
            values = q[:, column].abs()
            maximum = float(values.max())
            first = int(torch.nonzero(values == maximum, as_tuple=False)[0].item())
            if q[first, column] < 0:
                q[:, column].neg_()
        basis = q.to(dtype=torch.float32).contiguous()
        gram_error = float((basis.double().T @ basis.double() - torch.eye(k, dtype=torch.float64)).abs().max())
        bases.append(basis)
        metadata.append({
            "index": index, "seed": current_seed, "shape": [width, k],
            "orthonormality_max_error": gram_error,
            "row_gram_orthonormality_max_error": gram_error,
            "sha256": _tensor_sha256(basis),
        })
    return bases, metadata


_EVALUATION_ARMS: tuple[str, ...] = (
    "clean", "full", "v8", "random_8d_0", "random_8d_1", "random_8d_2", "random_8d_3",
)
_FORBIDDEN_COMPACT_KEYS = frozenset(
    {"activation", "residual", "hidden", "delta", "cache", "basis", "input_ids", "logits"}
)


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _variant_key(reverse: Any, order: Any) -> tuple[bool, int]:
    if not isinstance(reverse, bool) or not isinstance(order, int) or isinstance(order, bool):
        raise ValueError("evaluation variant must contain boolean reverse and integer order")
    return reverse, order


def _variant_label(variant: tuple[bool, int]) -> str:
    return f"reverse={int(variant[0])},order={variant[1]}"


def _graph_by_stratum(graphs: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(graphs, Mapping):
        values = list(graphs.values()) if "pairs" not in graphs else [graphs]
    elif isinstance(graphs, Sequence) and not isinstance(graphs, (str, bytes)):
        values = list(graphs)
    else:
        raise ValueError("graphs must be a graph mapping or sequence of graph mappings")
    result: dict[str, Mapping[str, Any]] = {}
    for graph in values:
        if not isinstance(graph, Mapping) or graph.get("stratum") not in ("sector_within", "cross_sector"):
            raise ValueError("graphs must contain sector_within and cross_sector graph objects")
        stratum = str(graph["stratum"])
        if stratum in result:
            raise ValueError(f"duplicate graph stratum: {stratum}")
        pairs = graph.get("pairs")
        if not isinstance(pairs, list):
            raise ValueError(f"{stratum} graph pairs must be a list")
        result[stratum] = graph
    if set(result) != {"sector_within", "cross_sector"}:
        raise ValueError("graphs must contain exactly sector_within and cross_sector")
    return result


def _graph_pairs(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    pairs = graph.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("graph pairs must be a list")
    provenance = graph.get("selection_margin_provenance")
    if not isinstance(provenance, list):
        raise ValueError("graph is missing selection margin provenance")
    margin_by_ticker: dict[str, tuple[str, float]] = {}
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("selection margin provenance rows must be objects")
        ticker = row.get("ticker")
        sector = row.get("sector")
        if not isinstance(ticker, str) or not ticker or not isinstance(sector, str) or not sector:
            raise ValueError("selection margin provenance is incomplete")
        if ticker in margin_by_ticker:
            raise ValueError("selection margin provenance tickers must be unique")
        margin_by_ticker[ticker] = (sector, _finite_number(row.get("margin"), "selection margin"))
        if any(field in row for field in ("arm", "direction", "patched_margin", "source_margin", "target_margin")):
            raise ValueError("graph provenance must contain selection fields only")
    result: dict[str, Mapping[str, Any]] = {}
    used: list[str] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("graph pair must be an object")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in result:
            raise ValueError("graph pair IDs must be non-empty and unique")
        required = ("high_ticker", "low_ticker", "high_sector", "low_sector")
        if any(not isinstance(pair.get(field), str) or not pair[field] for field in required):
            raise ValueError(f"graph pair {pair_id!r} has incomplete endpoint metadata")
        if pair.get("stratum") != graph.get("stratum"):
            raise ValueError(f"graph pair {pair_id!r} has the wrong stratum")
        if pair["high_ticker"] == pair["low_ticker"]:
            raise ValueError(f"graph pair {pair_id!r} has identical endpoints")
        for endpoint, sector_key, margin_key in (
            ("high_ticker", "high_sector", "high_margin"),
            ("low_ticker", "low_sector", "low_margin"),
        ):
            ticker = str(pair[endpoint])
            if ticker not in margin_by_ticker:
                raise ValueError(f"graph pair {pair_id!r} references an unknown ticker")
            provenance_sector, provenance_margin = margin_by_ticker[ticker]
            if pair[sector_key] != provenance_sector:
                raise ValueError(f"graph pair {pair_id!r} disagrees with selection provenance")
            if abs(_finite_number(pair.get(margin_key), margin_key) - provenance_margin) > 1e-12:
                raise ValueError(f"graph pair {pair_id!r} disagrees with selection margin provenance")
            used.append(ticker)
        if any(field in pair for field in ("arm", "direction", "patched_margin", "source_margin", "target_margin")):
            raise ValueError("pair graph must not contain evaluation fields")
        result[pair_id] = pair
    unmatched = graph.get("unmatched_tickers", [])
    if not isinstance(unmatched, list) or any(not isinstance(ticker, str) for ticker in unmatched):
        raise ValueError("graph unmatched_tickers must be a string list")
    accounted = used + unmatched
    expected = set(margin_by_ticker)
    if len(accounted) != len(set(accounted)) or set(accounted) != expected:
        raise ValueError("graph must account for each selected ticker exactly once")
    if graph.get("stratum") == "cross_sector":
        if unmatched or len(used) != len(expected):
            raise ValueError("cross-sector graph must use every selected ticker exactly once")
        if any(pair["high_sector"] == pair["low_sector"] for pair in result.values()):
            raise ValueError("cross-sector graph contains a same-sector edge")
    if graph.get("raw_runtime_payloads") is not False:
        raise ValueError("graph must set raw_runtime_payloads=false")
    return result


def _edge_direction(pair: Mapping[str, Any], direction: str) -> tuple[str, str, str, str]:
    if direction == "high_to_low":
        return (str(pair["high_ticker"]), str(pair["low_ticker"]),
                str(pair["high_sector"]), str(pair["low_sector"]))
    if direction == "low_to_high":
        return (str(pair["low_ticker"]), str(pair["high_ticker"]),
                str(pair["low_sector"]), str(pair["high_sector"]))
    raise ValueError(f"invalid transfer direction: {direction!r}")


def _compact_key_guard(value: Any, *, path: str = "summary") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in _FORBIDDEN_COMPACT_KEYS:
                raise ValueError(f"forbidden raw-payload key at {path}.{key}")
            _compact_key_guard(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _compact_key_guard(child, path=f"{path}[{index}]")


def validate_compact_analysis(summary: Mapping[str, Any]) -> None:
    """Fail closed if an analysis summary contains raw runtime payloads."""
    if not isinstance(summary, Mapping):
        raise ValueError("compact analysis summary must be an object")
    _compact_key_guard(summary)
    if summary.get("raw_runtime_payloads") is not False:
        raise ValueError("compact analysis must set raw_runtime_payloads=false")


# Descriptive aliases make the compact-schema check easy to discover without
# introducing a second schema implementation.
validate_compact_summary = validate_compact_analysis
validate_heldout_transfer_summary = validate_compact_analysis


def _percentile_ci(values: Sequence[float], *, samples: int, seed: int) -> list[float]:
    del seed
    if not values:
        return [None, None]  # type: ignore[list-item]
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    sorted_values = sorted(float(value) for value in values)
    return [sorted_values[int(0.025 * samples)], sorted_values[int(0.975 * samples) - 1]]


def _bootstrap_metric(
    bundles: Sequence[Mapping[str, Mapping[str, Any]]],
    selector: str | None,
    metric: str,
    *,
    samples: int,
    seed: int,
    random_index: int | None = None,
) -> list[float]:
    """Bootstrap one metric while drawing complete high/low pair bundles."""
    if not bundles:
        return []
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = [bundles[rng.randrange(len(bundles))] for _ in bundles]
        candidates = [bundle.get(selector) for bundle in selected] if selector else [
            edge for bundle in selected for edge in bundle.values()
        ]
        edge_values = []
        for edge in candidates:
            if not isinstance(edge, Mapping) or not edge.get("eligible"):
                continue
            value = edge.get(metric)
            if random_index is not None:
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise ValueError("random ratio values must be sequences")
                value = value[random_index]
            if value is not None:
                edge_values.append(float(value))
        if not edge_values:
            raise ValueError("bootstrap bundle has no eligible metric values")
        values.append(float(statistics.median(edge_values)))
    return values


def _metric_summary(
    edges: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    selector: str | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    excluded_count: int,
) -> dict[str, Any]:
    relevant_bundles = [
        bundle for bundle in bundles
        if (
            isinstance(bundle.get(selector), Mapping) and bundle[selector].get("eligible")
            if selector
            else any(isinstance(edge, Mapping) and edge.get("eligible") for edge in bundle.values())
        )
    ]
    selected_edges = [
        bundle[selector] for bundle in relevant_bundles
        if selector and bundle.get(selector, {}).get("eligible")
    ] if selector else [
        edge for bundle in relevant_bundles for edge in bundle.values()
        if edge.get("eligible")
    ]
    if not selected_edges:
        return {
            "status": "not_evaluable",
            "eligible_edge_count": 0,
            "excluded_edge_count": excluded_count,
            "eligible_bundle_count": 0,
            "v8_median_ratio": None,
            "random_median_ratios": [None] * 4,
            "paired_advantage_median": None,
            "full_reference_median_abs_delta_m": None,
            "bootstrap_ci_95": {"v8_median_ratio": [None, None],
                                 "random_median_ratios": [[None, None] for _ in range(4)],
                                 "paired_advantage_median": [None, None],
                                 "full_reference_median_abs_delta_m": [None, None]},
        }

    v8_values = [float(edge["ratio_v8"]) for edge in selected_edges]
    random_values = [[float(edge["ratio_random"][index]) for edge in selected_edges] for index in range(4)]
    advantage_values = [float(edge["paired_advantage"]) for edge in selected_edges]
    full_values = [abs(float(edge["median_full_delta_m"])) for edge in selected_edges]
    v8_median = statistics.median(v8_values)
    random_medians = [statistics.median(values) for values in random_values]
    advantage_median = statistics.median(advantage_values)
    full_median = statistics.median(full_values)
    v8_boot = _bootstrap_metric(relevant_bundles, selector, "ratio_v8", samples=bootstrap_samples, seed=bootstrap_seed)
    random_boot = [
        _bootstrap_metric(
            relevant_bundles, selector, "ratio_random", samples=bootstrap_samples,
            seed=bootstrap_seed + index + 1, random_index=index,
        )
        for index in range(4)
    ]
    advantage_boot = _bootstrap_metric(relevant_bundles, selector, "paired_advantage", samples=bootstrap_samples, seed=bootstrap_seed + 5)
    full_boot = _bootstrap_metric(relevant_bundles, selector, "full_reference_magnitude", samples=bootstrap_samples, seed=bootstrap_seed + 6)
    v8_ci = _percentile_ci(v8_boot, samples=bootstrap_samples, seed=bootstrap_seed) if v8_boot else [None, None]
    random_ci = [_percentile_ci(values, samples=bootstrap_samples, seed=bootstrap_seed)
                 if values else [None, None] for values in random_boot]
    advantage_ci = _percentile_ci(advantage_boot, samples=bootstrap_samples, seed=bootstrap_seed) if advantage_boot else [None, None]
    full_ci = _percentile_ci(full_boot, samples=bootstrap_samples, seed=bootstrap_seed) if full_boot else [None, None]
    if v8_median < HELDOUT_RECOVERY_TARGET:
        status = "not_supported"
    elif v8_ci[0] is not None and v8_ci[0] >= HELDOUT_RECOVERY_TARGET and advantage_median > 0 and advantage_ci[0] is not None and advantage_ci[0] > 0:
        status = "supported"
    else:
        status = "unresolved"
    return {
        "status": status,
        "eligible_edge_count": len(selected_edges),
        "excluded_edge_count": excluded_count,
        "eligible_bundle_count": len(relevant_bundles),
        "v8_median_ratio": v8_median,
        "random_median_ratios": random_medians,
        "paired_advantage_median": advantage_median,
        "full_reference_median_abs_delta_m": full_median,
        "bootstrap_ci_95": {
            "v8_median_ratio": v8_ci,
            "random_median_ratios": random_ci,
            "paired_advantage_median": advantage_ci,
            "full_reference_median_abs_delta_m": full_ci,
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def analyze_heldout_transfer(
    records: Sequence[Mapping[str, Any]],
    graphs: Any,
    *,
    bootstrap_samples: int = HELDOUT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = HELDOUT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Reduce compact margin records into per-edge and paired-bundle results.

    The only values used for transfer shifts are the compact source, target and
    patched margins.  Eligibility is decided before any non-reference arm is
    used, and bootstrap draws are complete undirected pair bundles.
    """
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    graph_map = _graph_by_stratum(graphs)
    pair_maps = {stratum: _graph_pairs(graph) for stratum, graph in graph_map.items()}
    expected_variants = set(HELDOUT_EVAL_VARIANTS)
    cells: dict[tuple[str, str, str, bool, int, bool, int], dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("analysis records must be objects")
        stratum = record.get("stratum")
        pair_id = record.get("pair_id")
        direction = record.get("direction")
        if stratum not in pair_maps or pair_id not in pair_maps[stratum]:
            raise ValueError("record does not refer to a frozen graph pair")
        if record.get("role") == "selection" or (record.get("reverse"), record.get("order")) == HELDOUT_SELECTION_VARIANT:
            raise ValueError("selection records cannot enter evaluation analysis")
        variant = _variant_key(record.get("reverse"), record.get("order"))
        if variant not in expected_variants:
            raise ValueError("record contains an unknown evaluation variant")
        if direction not in ("high_to_low", "low_to_high"):
            raise ValueError("record has an invalid direction")
        source, target, source_sector, target_sector = _edge_direction(pair_maps[stratum][pair_id], direction)
        if (record.get("source_ticker"), record.get("target_ticker")) != (source, target):
            raise ValueError("record endpoints do not match its graph pair and direction")
        if (record.get("source_sector"), record.get("target_sector")) != (source_sector, target_sector):
            raise ValueError("record sectors do not match its graph pair and direction")
        arm = record.get("arm")
        if arm not in _EVALUATION_ARMS:
            raise ValueError(f"unknown evaluation arm: {arm!r}")
        cell_key = (stratum, str(pair_id), str(direction), variant[0], variant[1], bool(variant[0]), int(variant[1]))
        arm_map = cells.setdefault(cell_key, {})
        if arm in arm_map:
            raise ValueError("duplicate arm within an evaluation cell")
        for field in ("source_margin", "target_margin", "patched_margin"):
            _finite_number(record.get(field), field)
        arm_map[str(arm)] = record

    grouped: dict[tuple[str, str, str], dict[tuple[bool, int], dict[str, Mapping[str, Any]]]] = {}
    for (stratum, pair_id, direction, reverse, order, _reverse, _order), arm_map in cells.items():
        grouped.setdefault((stratum, pair_id, direction), {})[(reverse, order)] = arm_map
    expected_cells = {
        (stratum, pair_id, direction)
        for stratum, pairs in pair_maps.items()
        for pair_id in pairs
        for direction in ("high_to_low", "low_to_high")
    }
    if set(grouped) != expected_cells:
        missing = sorted(expected_cells - set(grouped))
        extra = sorted(set(grouped) - expected_cells)
        raise ValueError(f"evaluation records do not cover frozen graph cells; missing={missing[:3]}, extra={extra[:3]}")
    if not grouped:
        raise ValueError("no evaluation records")

    edge_rows: list[dict[str, Any]] = []
    for (stratum, pair_id, direction), variants in sorted(grouped.items()):
        if set(variants) != expected_variants:
            raise ValueError("each transfer must contain all three evaluation variants")
        for variant, arm_map in variants.items():
            if set(arm_map) != set(_EVALUATION_ARMS):
                missing = sorted(set(_EVALUATION_ARMS) - set(arm_map))
                extra = sorted(set(arm_map) - set(_EVALUATION_ARMS))
                raise ValueError(f"evaluation cell must contain exactly seven arms; missing={missing}, extra={extra}")
        pair = pair_maps[stratum][pair_id]
        source, target, source_sector, target_sector = _edge_direction(pair, direction)
        by_arm: dict[str, list[float]] = {arm: [] for arm in _EVALUATION_ARMS}
        variant_deltas: dict[str, dict[str, float]] = {}
        for variant in sorted(expected_variants):
            arm_map = variants[variant]
            label = _variant_label(variant)
            variant_deltas[label] = {}
            for arm in _EVALUATION_ARMS:
                record = arm_map[arm]
                shift = toward_source_delta(
                    _finite_number(record["patched_margin"], "patched_margin"),
                    _finite_number(record["source_margin"], "source_margin"),
                    _finite_number(record["target_margin"], "target_margin"),
                )
                if not math.isfinite(shift):
                    raise ValueError("toward-source shift must be finite")
                by_arm[arm].append(shift)
                variant_deltas[label][arm] = shift
        medians = {arm: statistics.median(values) for arm, values in by_arm.items()}
        full_delta = medians["full"]
        eligible = abs(full_delta) >= HELDOUT_RATIO_MIN_FULL_DM
        edge: dict[str, Any] = {
            "stratum": stratum, "pair_id": pair_id, "direction": direction,
            "source_ticker": source, "target_ticker": target,
            "source_sector": source_sector, "target_sector": target_sector,
            "variant_deltas": variant_deltas,
            "median_full_delta_m": full_delta,
            "median_v8_delta_m": medians["v8"],
            "median_random_delta_m": [medians[f"random_8d_{index}"] for index in range(4)],
            "eligible": eligible,
            "raw_runtime_payloads": False,
        }
        if not eligible:
            edge.update({"exclusion_reason": "full_reference_below_threshold",
                         "full_reference_magnitude": abs(full_delta),
                         "ratio_v8": None, "ratio_random": [None] * 4,
                         "paired_advantage": None})
        else:
            ratio_v8 = medians["v8"] / full_delta
            ratio_random = [medians[f"random_8d_{index}"] / full_delta for index in range(4)]
            edge.update({"ratio_v8": ratio_v8, "ratio_random": ratio_random,
                         "paired_advantage": ratio_v8 - statistics.median(ratio_random),
                         "full_reference_magnitude": abs(full_delta)})
        edge_rows.append(edge)

    # Eligibility is per directed transfer.  A bootstrap draw retains the
    # complete pair bundle whenever either direction is eligible; each metric
    # filters that bundle to its own eligible edge(s).
    grouped_edges: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for edge in edge_rows:
        grouped_edges.setdefault((edge["stratum"], edge["pair_id"]), {})[edge["direction"]] = edge
    strata: dict[str, Any] = {}
    for stratum in ("sector_within", "cross_sector"):
        stratum_edges = [edge for edge in edge_rows if edge["stratum"] == stratum]
        excluded_count = sum(not edge["eligible"] for edge in stratum_edges)
        bundles = [values for (edge_stratum, _), values in sorted(grouped_edges.items())
                   if edge_stratum == stratum and set(values) == {"high_to_low", "low_to_high"}
                   and any(edge["eligible"] for edge in values.values())]
        all_metric = _metric_summary(stratum_edges, bundles, selector=None,
                                     bootstrap_samples=bootstrap_samples,
                                     bootstrap_seed=bootstrap_seed,
                                     excluded_count=excluded_count)
        direction_metrics = {
            direction: _metric_summary(stratum_edges, bundles, selector=direction,
                                       bootstrap_samples=bootstrap_samples,
                                       bootstrap_seed=bootstrap_seed + (1 if direction == "low_to_high" else 0),
                                       excluded_count=sum(1 for edge in stratum_edges
                                                          if edge["direction"] == direction and not edge["eligible"]))
            for direction in ("high_to_low", "low_to_high")
        }
        # Breakdowns use the directed source/target sector labels.  Their
        # bundles remain paired; a bundle can therefore contribute to two
        # directional sector cells but never crosses strata.
        breakdown: dict[str, Any] = {}
        labels = sorted({
            (edge["source_sector"], edge["target_sector"])
            for edge in stratum_edges
        })
        for source_sector, target_sector in labels:
            matching = [edge for edge in stratum_edges
                        if edge["source_sector"] == source_sector and edge["target_sector"] == target_sector]
            matching_ids = {(edge["stratum"], edge["pair_id"]) for edge in matching}
            matching_bundles = [values for key, values in grouped_edges.items()
                                if key in matching_ids and set(values) == {"high_to_low", "low_to_high"}
                                and any(edge["eligible"] for edge in values.values())]
            matching_directions = {edge["direction"] for edge in matching}
            breakdown_selector = next(iter(matching_directions)) if len(matching_directions) == 1 else None
            breakdown[f"{source_sector}->{target_sector}"] = _metric_summary(
                matching, matching_bundles, selector=breakdown_selector,
                bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
                excluded_count=sum(not edge["eligible"] for edge in matching),
            )
        strata[stratum] = {
            "status": all_metric["status"],
            "all_directions": all_metric,
            "high_to_low": direction_metrics["high_to_low"],
            "low_to_high": direction_metrics["low_to_high"],
            "sector_breakdown": breakdown,
            "eligible_edge_count": sum(edge["eligible"] for edge in stratum_edges),
            "excluded_edge_count": excluded_count,
            "eligible_bundle_count": len(bundles),
        }

    summary = {
        "schema_version": HELDOUT_TRANSFER_SCHEMA,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "full_reference_threshold": HELDOUT_RATIO_MIN_FULL_DM,
        "recovery_target": HELDOUT_RECOVERY_TARGET,
        "strata": strata,
        "edges": edge_rows,
        "raw_runtime_payloads": False,
    }
    validate_compact_analysis(summary)
    return summary


def _write_compact_json(path: Path, value: Any) -> None:
    """Write a compact artifact without widening core's raw-state policy.

    The held-out protocol deliberately names scalar derived fields such as
    ``toward_source_delta_m`` and ``delta_m``.  They are not runtime tensors,
    so this local writer preserves the protocol schema while still rejecting
    forbidden payload values before publication.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _compact_key_guard(value)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_compact_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            _compact_key_guard(row)
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return len(rows)


def _register_output(run: ArtifactRun, path: Path, *, artifact_type: str, stage: str, count: int | None = None) -> None:
    run.manifest.register_artifact(path, artifact_type=artifact_type, stage=stage, role="output", record_count=count)


def _prompt_map(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, bool, int], Mapping[str, Any]]:
    validate_prompt_roles(records)
    return {(str(row["ticker"]), bool(row["reverse"]), int(row["order"])): row for row in records}


def _margin_from_residual(model: Any, residual: torch.Tensor, tokenizer: Any, formatted: str) -> float:
    buy_id, sell_id = answer_token_ids(tokenizer, formatted + '{"decision": "')
    return margin_from_log_probs(fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id)


def _input_tensor(model: Any, tokenizer: Any, formatted: str) -> torch.Tensor:
    device = getattr(model, "input_device", "cpu")
    return torch.tensor([scoring_ids(tokenizer, formatted)], dtype=torch.long, device=device)


def _counted_transform(transform: Any, fires: dict[str, int]) -> Any:
    def wrapped(tensor: torch.Tensor) -> torch.Tensor:
        fires["hook_fires"] += 1
        if fires["hook_fires"] > 1:
            raise RuntimeError("held-out transfer hook fired more than once")
        return transform(tensor)
    return wrapped


def _projected_transplant_for_mapping(
    source_post: torch.Tensor,
    target_post: torch.Tensor,
    basis: torch.Tensor,
    mapping: Mapping[int, int],
) -> Any:
    """Build the low-level projected transplant with target-position rows."""
    positions = sorted(int(position) for position in mapping)
    delta = torch.stack(
        [
            source_post[0, int(mapping[position])].float() - target_post[0, position].float()
            for position in positions
        ],
        dim=0,
    )
    return make_projected_transplant(delta, basis, positions)


def _patched_margin(
    model: Any,
    tokenizer: Any,
    target: Mapping[str, Any],
    transform: Any,
    *,
    final_layer: int,
) -> tuple[float, int]:
    tensor = _input_tensor(model, tokenizer, str(target["formatted"]))
    fires = {"hook_fires": 0}
    with residual_interventions(model, {HELDOUT_LAYER: _counted_transform(transform, fires)}):
        residual = record_residuals(model, tensor, [final_layer])[final_layer]
    if fires["hook_fires"] != 1:
        raise RuntimeError(f"held-out transfer hook fired {fires['hook_fires']} times")
    margin = _margin_from_residual(model, residual, tokenizer, str(target["formatted"]))
    if not math.isfinite(margin):
        raise ValueError("patched margin is non-finite")
    return margin, fires["hook_fires"]


def _forward_heldout(
    run: ArtifactRun,
    model_path: str,
    prompts: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    v8: torch.Tensor,
    random_bases: Sequence[torch.Tensor],
    *,
    cohort_size: int,
) -> None:
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("forward") as stage:
        model, tokenizer, _ = load_model(model_path, dtype=None)
        final_layer = int(model.n_layers) - 1
        if HELDOUT_LAYER >= len(model.layers):
            raise ValueError(f"held-out layer {HELDOUT_LAYER} is unavailable")
        prompt_by_key = _prompt_map(prompts)
        selection_rows: list[dict[str, Any]] = []
        for company in cohort["companies"]:
            ticker = str(company["ticker"])
            row = prompt_by_key[(ticker, False, 0)]
            tensor = _input_tensor(model, tokenizer, str(row["formatted"]))
            states = record_block_states(model, tensor, [HELDOUT_LAYER, final_layer])
            margin = _margin_from_residual(model, states[final_layer]["post"], tokenizer, str(row["formatted"]))
            if not math.isfinite(margin):
                raise ValueError(f"selection margin is non-finite for {ticker}")
            selection_rows.append({
                "ticker": ticker, "name": row["name"], "sector": row["sector"],
                "reverse": False, "order": 0, "margin": margin,
                "role": "selection", "raw_runtime_payloads": False,
            })
        selection_path = out_dir / "selection.jsonl"
        selection_count = _write_compact_jsonl(selection_path, selection_rows)
        _register_output(run, selection_path, artifact_type="entity_to_dial_heldout_selection", stage="forward", count=selection_count)

        selection_for_graph = [dict(row) for row in selection_rows]
        within = build_sector_within_graph(selection_for_graph, cohort)
        cross = build_cross_sector_graph(selection_for_graph, cohort, cohort_size=cohort_size)
        graphs = {"sector_within": within, "cross_sector": cross}
        # Validate the just-frozen graphs before any evaluation-arm forward.
        _graph_by_stratum(graphs)
        graph_path = out_dir / "pair_graphs.json"
        _write_compact_json(graph_path, graphs)
        _register_output(run, graph_path, artifact_type="entity_to_dial_heldout_pair_graphs", stage="forward")

        records: list[dict[str, Any]] = []
        noops: list[dict[str, Any]] = []
        noops_seen: set[tuple[str, bool, int]] = set()
        bases = [v8, *random_bases]
        arm_names = ["v8", *[f"random_8d_{i}" for i in range(len(random_bases))]]
        for graph in (within, cross):
            stratum = str(graph["stratum"])
            for pair in graph["pairs"]:
                for direction in ("high_to_low", "low_to_high"):
                    source_ticker, target_ticker, source_sector, target_sector = _edge_direction(pair, direction)
                    for reverse, order in HELDOUT_EVAL_VARIANTS:
                        source = prompt_by_key[(source_ticker, reverse, order)]
                        target = prompt_by_key[(target_ticker, reverse, order)]
                        source_tensor = _input_tensor(model, tokenizer, str(source["formatted"]))
                        target_tensor = _input_tensor(model, tokenizer, str(target["formatted"]))
                        source_state = record_block_states(model, source_tensor, [HELDOUT_LAYER, final_layer])
                        source_post = source_state[HELDOUT_LAYER]["post"]
                        source_clean = _margin_from_residual(model, source_state[final_layer]["post"], tokenizer, str(source["formatted"]))
                        target_state = record_block_states(model, target_tensor, [HELDOUT_LAYER, final_layer])
                        target_post = target_state[HELDOUT_LAYER]["post"]
                        target_clean = _margin_from_residual(model, target_state[final_layer]["post"], tokenizer, str(target["formatted"]))
                        source_margin = source_clean
                        target_margin = target_clean
                        if not all(math.isfinite(x) for x in (source_margin, target_margin)):
                            raise ValueError("non-finite clean transfer margin")
                        mapping = nearest_position_mapping(tuple(source["instruction_span"]), tuple(target["instruction_span"]))
                        if not mapping:
                            raise ValueError("empty instruction-span mapping")
                        clean_row = {
                            "stratum": stratum, "pair_id": pair["pair_id"], "direction": direction,
                            "source_ticker": source_ticker, "target_ticker": target_ticker,
                            "source_sector": source_sector, "target_sector": target_sector,
                            "reverse": bool(reverse), "order": int(order), "arm": "clean",
                            "source_margin": source_margin, "target_margin": target_margin,
                            "patched_margin": target_clean,
                            "toward_source_delta_m": 0.0,
                            "hook_fires": 0, "raw_runtime_payloads": False,
                        }
                        records.append(clean_row)
                        full_margin, full_fires = _patched_margin(
                            model, tokenizer, target, make_full_transform(source_post, mapping=mapping), final_layer=final_layer
                        )
                        patched_values = [("full", full_margin, full_fires)]
                        for arm, basis in zip(arm_names, bases, strict=True):
                            margin, fires = _patched_margin(
                                model, tokenizer, target,
                                _projected_transplant_for_mapping(source_post, target_post, basis, mapping),
                                final_layer=final_layer,
                            )
                            patched_values.append((arm, margin, fires))
                        for arm, margin, fires in patched_values:
                            records.append({
                                "stratum": stratum, "pair_id": pair["pair_id"], "direction": direction,
                                "source_ticker": source_ticker, "target_ticker": target_ticker,
                                "source_sector": source_sector, "target_sector": target_sector,
                                "reverse": bool(reverse), "order": int(order), "arm": arm,
                                "source_margin": source_margin, "target_margin": target_clean,
                                "patched_margin": margin,
                                "toward_source_delta_m": toward_source_delta(margin, source_margin, target_clean),
                                "hook_fires": fires, "raw_runtime_payloads": False,
                            })
                        # Self-source checks are deliberately separate from pair records.
        # Self-source checks cover every cohort company, including odd-sector
        # within-graph leftovers.  They are not graph evaluation records.
        for company in cohort["companies"]:
            ticker = str(company["ticker"])
            for reverse, order in HELDOUT_EVAL_VARIANTS:
                noop_key = (ticker, bool(reverse), int(order))
                if noop_key in noops_seen:
                    continue
                noops_seen.add(noop_key)
                target = prompt_by_key[noop_key]
                target_tensor = _input_tensor(model, tokenizer, str(target["formatted"]))
                target_state = record_block_states(model, target_tensor, [HELDOUT_LAYER, final_layer])
                target_post = target_state[HELDOUT_LAYER]["post"]
                target_clean = _margin_from_residual(model, target_state[final_layer]["post"], tokenizer, str(target["formatted"]))
                identity = {int(pos): int(pos) for pos in range(int(target["instruction_span"][0]), int(target["instruction_span"][1]))}
                self_full, self_full_fires = _patched_margin(model, tokenizer, target, make_full_transform(target_post, mapping=identity), final_layer=final_layer)
                self_v8, self_v8_fires = _patched_margin(
                    model,
                    tokenizer,
                    target,
                    _projected_transplant_for_mapping(target_post, target_post, v8, identity),
                    final_layer=final_layer,
                )
                for arm, margin, fires in (("full_self_noop", self_full, self_full_fires), ("v8_self_noop", self_v8, self_v8_fires)):
                    delta = margin - target_clean
                    if not math.isfinite(delta) or abs(delta) > HELDOUT_NOOP_TOLERANCE:
                        raise ValueError(f"self-source no-op violated for {ticker}/{reverse}/{order}/{arm}: {delta:.3e}")
                    noops.append({"ticker": ticker, "reverse": bool(reverse), "order": int(order), "arm": arm,
                                  "clean_margin": target_clean, "patched_margin": margin, "delta_m": delta,
                                  "pass": True, "raw_runtime_payloads": False})
        expected_noops = cohort_size * len(HELDOUT_EVAL_VARIANTS) * 2
        if len(noops) != expected_noops:
            raise AssertionError(f"expected {expected_noops} self-source no-op records, got {len(noops)}")
        records_path = out_dir / "records.jsonl"
        noop_path = out_dir / "noop_records.jsonl"
        record_count = _write_compact_jsonl(records_path, records)
        noop_count = _write_compact_jsonl(noop_path, noops)
        _register_output(run, records_path, artifact_type="entity_to_dial_heldout_records", stage="forward", count=record_count)
        _register_output(run, noop_path, artifact_type="entity_to_dial_heldout_noop_records", stage="forward", count=noop_count)
        write_metadata(out_dir / "metadata.json", {"schema_version": HELDOUT_TRANSFER_SCHEMA, "artifact_type": "entity_to_dial_heldout_forward", "n_records": record_count, "n_noop_records": noop_count, "arms": ["clean", "full", "v8", "random_8d_0", "random_8d_1", "random_8d_2", "random_8d_3"], "raw_runtime_payloads": False}, overwrite=True)
        _register_output(run, out_dir / "metadata.json", artifact_type="entity_to_dial_heldout_forward_metadata", stage="forward")
        stage.count(record_count)


def _run_heldout_transfer(
    *,
    model_path: str | Path | None = None,
    model: str | Path | None = None,
    run_id: str,
    phase_e_run: str | Path,
    m6_manifest: Mapping[str, Any] | str | Path,
    population_csv: str | Path | None = None,
    population_path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
    cohort_size: int = HELDOUT_COHORT_SIZE,
    bootstrap_samples: int = HELDOUT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = HELDOUT_BOOTSTRAP_SEED,
) -> Path:
    """Internal workflow runner; ``cohort_size`` exists only for fake-model tests."""
    resolved_model = model_path if model_path is not None else model
    if resolved_model is None:
        raise TypeError("model_path or model is required")
    population = population_csv if population_csv is not None else population_path
    if population is None:
        raise TypeError("population_csv or population_path is required")
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")
    # All immutable provenance checks happen before the model is loaded.
    provenance = verify_heldout_inputs(phase_e_run, m6_manifest, population)
    tokenizer = load_tokenizer(str(resolved_model))
    rows = _load_population_contract(population)
    cohort = prepare_heldout_cohort(rows, m6_manifest=m6_manifest, population_path=population, cohort_size=cohort_size)
    prompts = prepare_heldout_prompt_records(cohort, tokenizer)
    random_width = int(provenance["v8"].shape[0])
    random_bases, random_metadata = random_orthonormal_bases(random_width)
    prepare_provenance = {key: value for key, value in provenance.items() if key != "v8"}
    prepare_provenance["random_bases"] = random_metadata
    prepare_provenance["model"] = str(resolved_model)
    prepare_provenance["raw_runtime_payloads"] = False
    run = ArtifactRun.create(str(resolved_model), "entity-to-dial-heldout-transfer", run_id, artifact_root=artifact_root)
    try:
        prepare_dir = run.run_directory / "prepare"
        with run.stage("prepare") as stage:
            cohort_path = prepare_dir / "cohort_manifest.json"
            prompt_path = prepare_dir / "prompts.jsonl"
            _write_compact_json(cohort_path, cohort)
            prompt_count = _write_compact_jsonl(prompt_path, prompts)
            _write_compact_json(prepare_dir / "provenance.json", prepare_provenance)
            _register_output(run, cohort_path, artifact_type="entity_to_dial_heldout_cohort", stage="prepare")
            _register_output(run, prompt_path, artifact_type="entity_to_dial_heldout_prompts", stage="prepare", count=prompt_count)
            _register_output(run, prepare_dir / "provenance.json", artifact_type="entity_to_dial_heldout_provenance", stage="prepare")
            write_metadata(prepare_dir / "metadata.json", {"schema_version": HELDOUT_TRANSFER_SCHEMA, "artifact_type": "entity_to_dial_heldout_prepare", "n_companies": cohort_size, "n_prompts": prompt_count, "raw_runtime_payloads": False}, overwrite=True)
            _register_output(run, prepare_dir / "metadata.json", artifact_type="entity_to_dial_heldout_prepare_metadata", stage="prepare")
            stage.count(prompt_count)
        _forward_heldout(run, str(resolved_model), prompts, cohort, provenance["v8"], random_bases, cohort_size=cohort_size)
        graphs = json.loads((run.run_directory / "forward" / "pair_graphs.json").read_text(encoding="utf-8"))
        records = []
        with (run.run_directory / "forward" / "records.jsonl").open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        summary = analyze_heldout_transfer(records, graphs, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
        with run.stage("analyze") as stage:
            analyze_dir = run.run_directory / "analyze"
            summary_path = analyze_dir / "summary.json"
            _write_compact_json(summary_path, summary)
            _register_output(run, summary_path, artifact_type="entity_to_dial_heldout_analysis", stage="analyze")
            write_metadata(analyze_dir / "metadata.json", {"schema_version": HELDOUT_TRANSFER_SCHEMA, "artifact_type": "entity_to_dial_heldout_analysis_metadata", "status": {key: value["status"] for key, value in summary["strata"].items()}, "raw_runtime_payloads": False}, overwrite=True)
            _register_output(run, analyze_dir / "metadata.json", artifact_type="entity_to_dial_heldout_analysis_metadata", stage="analyze")
            stage.count(len(summary["edges"]))
        run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


def run_heldout_transfer(
    *,
    model_path: str | Path | None = None,
    model: str | Path | None = None,
    run_id: str,
    phase_e_run: str | Path,
    m6_manifest: Mapping[str, Any] | str | Path,
    population_csv: str | Path | None = None,
    population_path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
    bootstrap_samples: int = HELDOUT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = HELDOUT_BOOTSTRAP_SEED,
    smoke: bool = False,
) -> Path:
    """Run formal V1 or its one-pair smoke through the sole public entry.

    The formal branch intentionally fixes cohort size at 200. ``smoke=True``
    bypasses selection, formal pair graphs and analysis; it validates only one
    ticker-sorted pair and must not be treated as a formal result.
    """
    resolved_model = model_path if model_path is not None else model
    population = population_csv if population_csv is not None else population_path
    if resolved_model is None or population is None:
        raise TypeError("model_path/model and population_csv/population_path are required")
    if smoke:
        return _run_heldout_transfer_smoke(
            model_path=resolved_model,
            run_id=run_id,
            phase_e_run=phase_e_run,
            m6_manifest=m6_manifest,
            population_csv=population,
            artifact_root=artifact_root,
        )
    return _run_heldout_transfer(
        model_path=resolved_model,
        run_id=run_id,
        phase_e_run=phase_e_run,
        m6_manifest=m6_manifest,
        population_csv=population,
        artifact_root=artifact_root,
        cohort_size=HELDOUT_COHORT_SIZE,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _run_heldout_transfer_smoke(
    *,
    model_path: str | Path,
    run_id: str,
    phase_e_run: str | Path,
    m6_manifest: Mapping[str, Any] | str | Path,
    population_csv: str | Path,
    artifact_root: str | Path = "artifacts",
) -> Path:
    """Run the V1 one-pair smoke without selection, graphs, or analysis."""
    provenance = verify_heldout_inputs(phase_e_run, m6_manifest, population_csv)
    tokenizer = load_tokenizer(str(model_path))
    cohort = prepare_heldout_cohort(
        _load_population_contract(population_csv),
        m6_manifest=m6_manifest,
        population_path=population_csv,
    )
    source_company, target_company = sorted(cohort["companies"], key=lambda row: row["ticker"])[:2]
    smoke_cohort = {
        **cohort,
        "cohort_size": 2,
        "companies": [source_company, target_company],
        "smoke_source_selected_ticker_sha256": cohort["selected_ticker_sha256"],
        "smoke": True,
    }
    prompts = [
        row for row in prepare_heldout_prompt_records(smoke_cohort, tokenizer)
        if (bool(row["reverse"]), int(row["order"])) == (True, 0)
    ]
    if len(prompts) != 2:
        raise AssertionError("smoke must prepare exactly one evaluation variant for two companies")
    random_basis, random_metadata = random_orthonormal_bases(int(provenance["v8"].shape[0]), count=1)
    run = ArtifactRun.create(str(model_path), "entity-to-dial-heldout-transfer", run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            prepare_dir = run.run_directory / "prepare"
            _write_compact_json(prepare_dir / "cohort_manifest.json", smoke_cohort)
            prompt_count = _write_compact_jsonl(prepare_dir / "prompts.jsonl", prompts)
            _write_compact_json(
                prepare_dir / "provenance.json",
                {
                    **{key: value for key, value in provenance.items() if key != "v8"},
                    "random_bases": random_metadata,
                    "smoke": True,
                    "raw_runtime_payloads": False,
                },
            )
            for path, artifact_type, count in (
                (prepare_dir / "cohort_manifest.json", "entity_to_dial_heldout_smoke_cohort", None),
                (prepare_dir / "prompts.jsonl", "entity_to_dial_heldout_smoke_prompts", prompt_count),
                (prepare_dir / "provenance.json", "entity_to_dial_heldout_smoke_provenance", None),
            ):
                _register_output(run, path, artifact_type=artifact_type, stage="prepare", count=count)
            stage.count(prompt_count)
        with run.stage("forward") as stage:
            model, runtime_tokenizer, _ = load_model(str(model_path), dtype=None)
            if HELDOUT_LAYER >= len(model.layers):
                raise ValueError(f"held-out layer {HELDOUT_LAYER} is unavailable")
            final_layer = int(model.n_layers) - 1
            source = next(row for row in prompts if row["ticker"] == source_company["ticker"])
            target = next(row for row in prompts if row["ticker"] == target_company["ticker"])
            source_state = record_block_states(
                model, _input_tensor(model, runtime_tokenizer, str(source["formatted"])), [HELDOUT_LAYER, final_layer]
            )
            target_state = record_block_states(
                model, _input_tensor(model, runtime_tokenizer, str(target["formatted"])), [HELDOUT_LAYER, final_layer]
            )
            source_margin = _margin_from_residual(model, source_state[final_layer]["post"], runtime_tokenizer, str(source["formatted"]))
            target_margin = _margin_from_residual(model, target_state[final_layer]["post"], runtime_tokenizer, str(target["formatted"]))
            if not all(math.isfinite(value) for value in (source_margin, target_margin)):
                raise ValueError("smoke clean margins must be finite")
            mapping = nearest_position_mapping(tuple(source["instruction_span"]), tuple(target["instruction_span"]))
            if not mapping:
                raise ValueError("smoke instruction-span mapping is empty")
            source_post = source_state[HELDOUT_LAYER]["post"]
            target_post = target_state[HELDOUT_LAYER]["post"]
            full, full_fires = _patched_margin(model, runtime_tokenizer, target, make_full_transform(source_post, mapping=mapping), final_layer=final_layer)
            v8, v8_fires = _patched_margin(model, runtime_tokenizer, target, _projected_transplant_for_mapping(source_post, target_post, provenance["v8"], mapping), final_layer=final_layer)
            random, random_fires = _patched_margin(model, runtime_tokenizer, target, _projected_transplant_for_mapping(source_post, target_post, random_basis[0], mapping), final_layer=final_layer)
            identity = {position: position for position in range(int(target["instruction_span"][0]), int(target["instruction_span"][1]))}
            full_noop, full_noop_fires = _patched_margin(model, runtime_tokenizer, target, make_full_transform(target_post, mapping=identity), final_layer=final_layer)
            v8_noop, v8_noop_fires = _patched_margin(model, runtime_tokenizer, target, _projected_transplant_for_mapping(target_post, target_post, provenance["v8"], identity), final_layer=final_layer)
            for arm, margin, fires in (("full_self_noop", full_noop, full_noop_fires), ("v8_self_noop", v8_noop, v8_noop_fires)):
                if abs(margin - target_margin) > HELDOUT_NOOP_TOLERANCE:
                    raise ValueError(f"smoke self-source no-op violated for {arm}")
                if fires != 1:
                    raise RuntimeError(f"smoke {arm} hook fired {fires} times")
            smoke = {
                "schema_version": HELDOUT_TRANSFER_SCHEMA,
                "smoke": True,
                "source_ticker": source_company["ticker"],
                "target_ticker": target_company["ticker"],
                "variant": {"reverse": True, "order": 0},
                "margins": {"clean": target_margin, "full": full, "v8": v8, "random_8d_0": random},
                "hook_fires": {"full": full_fires, "v8": v8_fires, "random_8d_0": random_fires, "full_self_noop": full_noop_fires, "v8_self_noop": v8_noop_fires},
                "noops": {"full_self_noop_delta_m": full_noop - target_margin, "v8_self_noop_delta_m": v8_noop - target_margin},
                "raw_runtime_payloads": False,
            }
            smoke_path = run.run_directory / "forward" / "smoke.json"
            _write_compact_json(smoke_path, smoke)
            _register_output(run, smoke_path, artifact_type="entity_to_dial_heldout_smoke", stage="forward")
            stage.count(1)
        run.finalize(required_stages={"prepare", "forward"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


# Useful explicit aliases for callers/tests that name the operations by role.
prepare_prompt_records = prepare_heldout_prompt_records
validate_prompt_records = validate_prompt_roles
