"""Frozen, tokenizer-only input preparation for entity-cell experiments."""
from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import sha256_bytes, sha256_file, sha256_json, stable_record_id
from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.prompt_input.encoding import input_ids, token_span

SPLITS = ("discovery", "calibration", "test")
TECHNOLOGY = "Technology"
HEADER_VARIANT_COUNT = 12
LOCALIZATION_VARIANT_IDS = tuple(range(8))
HELD_VARIANT_IDS = tuple(range(8, 12))
BASELINE_RECORD_COUNT = 399
FINANCIAL_PROMPT_COLUMNS = (
    "prompt_with_context_attribute_0",
    "prompt_with_context_attribute_1",
    "prompt_with_context_attribute_2",
)
BASELINE_SCHEMA_VERSION = 1
PREPARATION_SCHEMA_VERSION = 1
BASELINE_EXACT_IDENTITY = "paper-appendix-a-v1"

# These strings are the frozen V1 prefix family. The canonical two-line header and
# every character after it stay byte-for-byte unchanged.
HEADER_VARIANT_SPECS = (
    (0, "localization", "Use the company identity below when reading the evidence."),
    (1, "localization", "Read the following company profile before reviewing the evidence."),
    (2, "localization", "The financial case concerns the company identified below."),
    (3, "localization", "Identify the company in this report before making a decision."),
    (4, "localization", "Keep the named company in view while interpreting the report."),
    (5, "localization", "First inspect the company header, then inspect the evidence."),
    (6, "localization", "Use the named issuer as the subject of the financial report."),
    (7, "localization", "The report below names the company whose evidence you must assess."),
    (8, "held", "Start with the issuer information shown in the report header."),
    (9, "held", "Read the issuer header and then evaluate the financial evidence."),
    (10, "held", "This report provides an identified company and supporting evidence."),
    (11, "held", "Before answering, note the company named in the report."),
)

_TICKER_RE = re.compile(r"^Stock Ticker: \[(?P<value>[^\]\r\n]+)\]$", re.MULTILINE)
_NAME_RE = re.compile(r"^Stock Name: \[(?P<value>[^\]\r\n]+)\]$", re.MULTILINE)
_HEADER_RE = re.compile(
    r"^Stock Ticker: \[[^\]\r\n]+\]\nStock Name: \[[^\]\r\n]+\]",
    re.MULTILINE,
)
_EVIDENCE_START = "--- Evidence ---"
_EVIDENCE_END = "\n---\nRespond"


def _match_one(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"prompt must contain exactly one canonical {label} line")
    return matches[0]


def parse_header(prompt: str) -> tuple[str, str, tuple[int, int]]:
    """Return ticker, name, and the character range of the two-line header."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    ticker = _match_one(_TICKER_RE, prompt, "ticker")
    name = _match_one(_NAME_RE, prompt, "name")
    header = _HEADER_RE.search(prompt)
    if header is None or header.start() != ticker.start() or header.end() != name.end():
        raise ValueError("prompt must contain adjacent canonical ticker/name header lines")
    return ticker.group("value"), name.group("value"), (header.start(), header.end())


def _render_prefix(prompt: str, prefix: str) -> str:
    ticker, name, header_range = parse_header(prompt)
    header = prompt[header_range[0] : header_range[1]]
    rendered = prefix + "\n" + header + prompt[header_range[1] :]
    rendered_ticker, rendered_name, rendered_range = parse_header(rendered)
    if (rendered_ticker, rendered_name) != (ticker, name):
        raise ValueError("header variant changed canonical identity")
    if rendered[rendered_range[1] :] != prompt[header_range[1] :]:
        raise ValueError("header variant changed text after the header")
    return rendered


def render_header_variants(prompt: str, *, ticker: str | None = None, name: str | None = None) -> list[dict[str, Any]]:
    """Render and validate the frozen eight-plus-four header-prefix variants."""
    source_ticker, source_name, _ = parse_header(prompt)
    if ticker is not None and ticker != source_ticker:
        raise ValueError("prompt header ticker does not match supplied ticker")
    if name is not None and name != source_name:
        raise ValueError("prompt header name does not match supplied name")
    rows = []
    for variant_id, family, prefix in HEADER_VARIANT_SPECS:
        rendered = _render_prefix(prompt, prefix)
        rows.append(
            {
                "variant_id": variant_id,
                "variant_family": family,
                "prefix": prefix,
                "ticker": source_ticker,
                "name": source_name,
                "prompt": rendered,
            }
        )
    if len(rows) != HEADER_VARIANT_COUNT:
        raise AssertionError("frozen header variant contract is incomplete")
    return rows


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def load_split_manifest(path: str | Path, input_path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Load the schema-version-1 ticker split and bind it to the source CSV."""
    manifest_path, source_path = Path(path), Path(input_path)
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("split manifest must be a schema-version-1 object")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("split manifest must contain non-empty assignments")
    if payload.get("input_sha256") != sha256_file(source_path):
        raise ValueError("split manifest is not bound to the requested input CSV")
    normalized = {str(ticker): str(split) for ticker, split in assignments.items()}
    if any(split not in SPLITS for split in normalized.values()):
        raise ValueError("split manifest contains an unknown split")
    if len(normalized) != len(assignments):
        raise ValueError("split manifest contains duplicate ticker identities")
    return normalized, payload


def select_split_rows(
    input_path: str | Path,
    split_manifest: str | Path,
    *,
    split: str = "discovery",
    sector: str = TECHNOLOGY,
) -> list[dict[str, Any]]:
    """Select whole ticker identities from one manifest split without leakage."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    assignments, _ = load_split_manifest(split_manifest, input_path)
    rows: list[dict[str, Any]] = []
    with Path(input_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "name", "sector", *FINANCIAL_PROMPT_COLUMNS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"input CSV is missing required columns: {missing}")
        seen: dict[str, str] = {}
        selected_tickers: set[str] = set()
        for source_row_index, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip()
            name = (row.get("name") or "").strip()
            if row.get("sector") != sector or assignments.get(ticker) != split:
                continue
            if not ticker or not name:
                raise ValueError("selected rows must contain non-empty ticker and name")
            previous_name = seen.setdefault(ticker, name)
            if previous_name != name:
                raise ValueError(f"ticker {ticker!r} has conflicting names")
            if ticker in selected_tickers:
                raise ValueError(f"ticker {ticker!r} occurs more than once in the selected input")
            selected_tickers.add(ticker)
            prompts = {column: row.get(column) or "" for column in FINANCIAL_PROMPT_COLUMNS}
            if any(not value for value in prompts.values()):
                missing = [column for column, value in prompts.items() if not value]
                raise ValueError(f"ticker {ticker!r} has empty financial prompt columns: {missing}")
            rows.append(
                {
                    "source_row_index": source_row_index,
                    "ticker": ticker,
                    "name": name,
                    "sector": row.get("sector"),
                    "date": row.get("Date") or row.get("date") or "",
                    "marketcap": row.get("marketcap") or "",
                    "prompts": prompts,
                }
            )
    if not rows:
        raise ValueError(f"no {sector!r} rows match split {split!r}")
    return rows


def _baseline_rows(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid baseline JSONL at line {line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError("baseline JSONL rows must be objects")
                rows.append(value)
        return rows
    return [dict(row) for row in source]


def validate_baseline_contract(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    identity: str | None,
    expected_count: int = BASELINE_RECORD_COUNT,
) -> list[dict[str, Any]]:
    """Validate an explicit versioned generic-baseline JSONL contract.

    ``paper-appendix-a-v1`` identifies the exact paper list. Any replacement must
    use an explicit ``adapted:...`` identity; the preparation code never creates
    generic prompts when the file is absent.
    """
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("baseline identity is required; use paper-appendix-a-v1 or adapted:<version>")
    identity = identity.strip()
    if identity == BASELINE_EXACT_IDENTITY:
        raise ValueError(
            "the exact paper Appendix A prompt list is not present; use an explicit adapted:<version> identity"
        )
    if not identity.startswith("adapted:") or identity == "adapted:":
        raise ValueError("baseline identity must be an explicit adapted:<version> identity")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    rows = _baseline_rows(source)
    if len(rows) != expected_count:
        raise ValueError(f"baseline contract requires exactly {expected_count} records, got {len(rows)}")
    result = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        prompt_id = row.get("prompt_id")
        prompt = row.get("prompt")
        digest = row.get("prompt_sha256", row.get("sha256"))
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"baseline row {index} is missing prompt_id")
        if prompt_id in seen:
            raise ValueError(f"baseline prompt_id is duplicated: {prompt_id}")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"baseline row {index} is missing prompt")
        expected_digest = sha256_bytes(prompt.encode("utf-8"))
        if digest != expected_digest:
            raise ValueError(f"baseline prompt hash mismatch for {prompt_id}")
        seen.add(prompt_id)
        result.append(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "artifact_type": "entity_cell_generic_baseline_prompt",
                "baseline_identity": identity,
                "prompt_id": prompt_id,
                "prompt_sha256": expected_digest,
                "prompt": prompt,
            }
        )
    return result


def _token_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    values = encoded if isinstance(encoded, Mapping) else vars(encoded)
    ids = input_ids(tokenizer, text, add_special_tokens=True)
    offsets = values.get("offset_mapping")
    specials = values.get("special_tokens_mask")
    if offsets is None:
        raise TypeError("tokenizer must provide input_ids and offset_mapping for entity preparation")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if hasattr(specials, "tolist"):
        specials = specials.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if offsets and isinstance(offsets[0], (list, tuple)) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if specials and isinstance(specials[0], (list, tuple)):
        specials = specials[0]
    if specials is None:
        specials = [False] * len(offsets)
    return [int(value) for value in ids], [tuple(map(int, pair)) for pair in offsets], [bool(value) for value in specials]


def _formatted_prompt(tokenizer: Any, raw_prompt: str) -> tuple[str, int]:
    # Importing the shared formatter here keeps this module's span logic on the
    # canonical path and allows tests to use small tokenizer doubles.
    from llm_bias.core.prompt_input.encoding import format_prompt

    formatted = format_prompt(tokenizer, raw_prompt, use_chat_template=True, enable_thinking=False)
    offset = formatted.find(raw_prompt)
    if offset < 0:
        raise ValueError("formatted chat prompt does not contain the raw prompt")
    return formatted, offset


def _positions_for_range(
    offsets: list[tuple[int, int]],
    specials: list[bool],
    start: int,
    end: int,
    *,
    limit: int,
    contained: bool = False,
) -> set[int]:
    return {
        index
        for index, ((token_start, token_end), special) in enumerate(zip(offsets, specials, strict=True))
        if index < limit
        and not special
        and token_end > token_start
        and (
            token_start >= start and token_end <= end
            if contained
            else token_end > start and token_start < end
        )
    }


def _ranges(positions: set[int]) -> list[list[int]]:
    if not positions:
        return []
    ordered = sorted(positions)
    result: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            result.append([start, previous + 1])
            start = value
        previous = value
    result.append([start, previous + 1])
    return result


def _source_groups(tokenizer: Any, raw_prompt: str, formatted: str, raw_offset: int) -> dict[str, Any]:
    ids, offsets, specials = _token_offsets(tokenizer, formatted)
    content_positions = [
        index for index, ((start, end), special) in enumerate(zip(offsets, specials, strict=True))
        if not special and end > start
    ]
    if len(content_positions) < 2:
        raise ValueError("formatted prompt has no final query position")
    final_query_position = content_positions[-1]
    ticker, name, header_range = parse_header(raw_prompt)
    ticker_match = _match_one(_TICKER_RE, raw_prompt, "ticker")
    name_match = _match_one(_NAME_RE, raw_prompt, "name")
    ticker_start, ticker_end = ticker_match.start("value"), ticker_match.end("value")
    name_start, name_end = name_match.start("value"), name_match.end("value")
    evidence_marker = raw_prompt.find(_EVIDENCE_START)
    evidence_content_start = evidence_marker + len(_EVIDENCE_START) if evidence_marker >= 0 else -1
    evidence_content_end = raw_prompt.find(_EVIDENCE_END, evidence_content_start) if evidence_marker >= 0 else -1
    if evidence_marker < 0 or evidence_content_end < 0:
        raise ValueError("prompt must contain the frozen evidence markers")
    if evidence_marker <= header_range[1] or name_end > evidence_marker:
        raise ValueError("company header must precede the evidence section")
    absolute = lambda value: raw_offset + value
    identity = _positions_for_range(
        offsets, specials, absolute(ticker_start), absolute(ticker_end),
        limit=final_query_position,
    )
    identity |= _positions_for_range(
        offsets, specials, absolute(name_start), absolute(name_end),
        limit=final_query_position,
    )
    evidence = _positions_for_range(offsets, specials, absolute(evidence_content_start), absolute(evidence_content_end), limit=final_query_position)
    instruction = _positions_for_range(offsets, specials, absolute(evidence_content_end), len(formatted), limit=final_query_position)
    assigned: dict[str, set[int]] = {
        "identity_header": identity,
        "evidence": evidence - identity,
        "instruction_context": instruction - identity - evidence,
    }
    used = set().union(*assigned.values())
    source_positions = {
        index for index in range(final_query_position)
        if not specials[index] and offsets[index][1] > offsets[index][0]
    }
    assigned["other_prefix"] = source_positions - used
    covered = set().union(*assigned.values())
    if covered != source_positions or any(len(left & right) for i, left in enumerate(assigned.values()) for right in list(assigned.values())[i + 1 :]):
        raise ValueError("E2 source groups must be disjoint and cover all source positions")
    shared_name_span = token_span(
        tokenizer, formatted, absolute(name_start), absolute(name_end), add_special_tokens=True
    )
    name_positions = _positions_for_range(
        offsets, specials, absolute(name_start), absolute(name_end),
        limit=final_query_position, contained=True,
    )
    if shared_name_span is None:
        name_positions = set()
    if not name_positions:
        raise ValueError("company-name content does not map to a non-empty token span")
    name_span = (min(name_positions), max(name_positions) + 1)
    return {
        "final_query_position": final_query_position,
        "company_name_content_token_span": {
            "char_start": absolute(name_start),
            "char_end": absolute(name_end),
            "token_start": name_span[0],
            "token_end": name_span[1],
            "final_content_token": name_span[1] - 1,
            "final_content_token_id": ids[name_span[1] - 1],
        },
        "source_groups": {name: {"ranges": _ranges(values), "token_count": len(values)} for name, values in assigned.items()},
        "token_count": len(ids),
        "input_ids": ids,
    }


def _prepare_header_rows(tokenizer: Any, source_rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    rows = []
    for source in source_rows:
        base_prompt = source["prompts"][FINANCIAL_PROMPT_COLUMNS[0]]
        variants = render_header_variants(base_prompt, ticker=source["ticker"], name=source["name"])
        for variant in variants:
            formatted, raw_offset = _formatted_prompt(tokenizer, variant["prompt"])
            groups = _source_groups(tokenizer, variant["prompt"], formatted, raw_offset)
            rows.append(
                {
                    "schema_version": PREPARATION_SCHEMA_VERSION,
                    "artifact_type": "entity_cell_header_variant",
                    "variant_id": stable_record_id(source["ticker"], variant["variant_id"], split),
                    "ticker": source["ticker"],
                    "name": source["name"],
                    "sector": source["sector"],
                    "split": split,
                    "variant_number": variant["variant_id"],
                    "variant_family": variant["variant_family"],
                    "prompt_column": FINANCIAL_PROMPT_COLUMNS[0],
                    "source_row_index": source["source_row_index"],
                    "source_date": source["date"],
                    "source_prompt_sha256": sha256_bytes(base_prompt.encode("utf-8")),
                    "prompt_sha256": sha256_bytes(variant["prompt"].encode("utf-8")),
                    "prompt": variant["prompt"],
                    "formatted_prompt_sha256": sha256_bytes(formatted.encode("utf-8")),
                    "company_name_content_token_span": groups["company_name_content_token_span"],
                    "final_company_name_content_token": groups["company_name_content_token_span"]["final_content_token"],
                    "final_company_name_content_token_id": groups["company_name_content_token_span"]["final_content_token_id"],
                    "final_query_position": groups["final_query_position"],
                    "token_count": groups["token_count"],
                    "input_ids": groups["input_ids"],
                }
            )
    return rows


def _prepare_financial_rows(tokenizer: Any, source_rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    rows = []
    for source in source_rows:
        for column in FINANCIAL_PROMPT_COLUMNS:
            raw_prompt = source["prompts"][column]
            ticker, name, _ = parse_header(raw_prompt)
            if (ticker, name) != (source["ticker"], source["name"]):
                raise ValueError(f"{column} header does not match CSV identity for {source['ticker']}")
            formatted, raw_offset = _formatted_prompt(tokenizer, raw_prompt)
            groups = _source_groups(tokenizer, raw_prompt, formatted, raw_offset)
            rows.append(
                {
                    "schema_version": PREPARATION_SCHEMA_VERSION,
                    "artifact_type": "entity_cell_financial_prompt",
                    "prompt_id": stable_record_id(source["ticker"], column, split),
                    "ticker": source["ticker"],
                    "name": source["name"],
                    "sector": source["sector"],
                    "split": split,
                    "prompt_column": column,
                    "source_row_index": source["source_row_index"],
                    "source_date": source["date"],
                    "marketcap": source["marketcap"],
                    "source_prompt_sha256": sha256_bytes(raw_prompt.encode("utf-8")),
                    "formatted_prompt_sha256": sha256_bytes(formatted.encode("utf-8")),
                    "prompt": raw_prompt,
                    "token_count": groups["token_count"],
                    "final_query_position": groups["final_query_position"],
                    "company_name_content_token_span": groups["company_name_content_token_span"],
                    "final_company_name_content_token": groups["company_name_content_token_span"]["final_content_token"],
                    "final_company_name_content_token_id": groups["company_name_content_token_span"]["final_content_token_id"],
                    "source_groups": groups["source_groups"],
                    "input_ids": groups["input_ids"],
                }
            )
    return rows


def prepare_inputs(
    *,
    tokenizer: Any,
    input_path: str | Path,
    split_manifest: str | Path,
    baseline_source: str | Path | Iterable[Mapping[str, Any]],
    baseline_identity: str | None,
    split: str = "discovery",
    sector: str = TECHNOLOGY,
    baseline_expected_count: int = BASELINE_RECORD_COUNT,
) -> dict[str, Any]:
    """Prepare compact E1/E2 input records without loading a model."""
    source_rows = select_split_rows(input_path, split_manifest, split=split, sector=sector)
    baseline = validate_baseline_contract(
        baseline_source, identity=baseline_identity, expected_count=baseline_expected_count
    )
    header_variants = _prepare_header_rows(tokenizer, source_rows, split=split)
    financial_prompts = _prepare_financial_rows(tokenizer, source_rows, split=split)
    if len(header_variants) != len(source_rows) * HEADER_VARIANT_COUNT:
        raise AssertionError("each selected ticker must have exactly twelve header variants")
    if len(financial_prompts) != len(source_rows) * len(FINANCIAL_PROMPT_COLUMNS):
        raise AssertionError("each selected ticker must have exactly three financial prompts")
    config = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "experiment": "entity-cell-localization",
        "version": "v1",
        "split": split,
        "sector": sector,
        "header_variant_count": HEADER_VARIANT_COUNT,
        "localization_variant_ids": list(LOCALIZATION_VARIANT_IDS),
        "held_variant_ids": list(HELD_VARIANT_IDS),
        "financial_prompt_columns": list(FINANCIAL_PROMPT_COLUMNS),
        "baseline_expected_count": baseline_expected_count,
        "baseline_identity": baseline_identity,
    }
    return {
        "header_variants": header_variants,
        "generic_baseline": baseline,
        "financial_prompts": financial_prompts,
        "config": config,
        "ticker_count": len(source_rows),
    }


def validate_prepared_inputs(prepared: Mapping[str, Any]) -> None:
    """Fail closed on preparation cardinality, families, groups, and raw payloads."""
    headers = list(prepared.get("header_variants", ()))
    financial = list(prepared.get("financial_prompts", ()))
    baseline = list(prepared.get("generic_baseline", ()))
    config = prepared.get("config")
    if not isinstance(config, Mapping) or config.get("header_variant_count") != HEADER_VARIANT_COUNT:
        raise ValueError("prepared inputs have no frozen twelve-variant config")
    by_ticker: dict[str, list[Mapping[str, Any]]] = {}
    for row in headers:
        ticker = str(row.get("ticker"))
        by_ticker.setdefault(ticker, []).append(row)
        variant_number = row.get("variant_number")
        if variant_number not in range(HEADER_VARIANT_COUNT):
            raise ValueError("header variant number is outside the frozen range")
        if parse_header(str(row.get("prompt")))[:2] != (ticker, str(row.get("name"))):
            raise ValueError("header variant does not preserve the canonical identity header")
    for rows in by_ticker.values():
        if len(rows) != HEADER_VARIANT_COUNT:
            raise ValueError("each ticker must have exactly twelve header variants")
        if {row["variant_number"] for row in rows} != set(range(HEADER_VARIANT_COUNT)):
            raise ValueError("each ticker must contain each frozen header variant exactly once")
        families = {row["variant_number"]: row.get("variant_family") for row in rows}
        if any(families[index] != "localization" for index in LOCALIZATION_VARIANT_IDS):
            raise ValueError("localization variant family is invalid")
        if any(families[index] != "held" for index in HELD_VARIANT_IDS):
            raise ValueError("held variant family is invalid")
    if len(financial) != len(by_ticker) * len(FINANCIAL_PROMPT_COLUMNS):
        raise ValueError("financial prompt count does not match the three frozen columns")
    if len(baseline) != int(config.get("baseline_expected_count", BASELINE_RECORD_COUNT)):
        raise ValueError("generic baseline count does not match config")
    financial_ids = {(row.get("ticker"), row.get("prompt_column")) for row in financial}
    expected_columns = set(FINANCIAL_PROMPT_COLUMNS)
    if any(
        {row.get("prompt_column") for row in financial if row.get("ticker") == ticker} != expected_columns
        for ticker in by_ticker
    ) or len(financial_ids) != len(financial):
        raise ValueError("each ticker must contain each frozen financial prompt column exactly once")
    for row in financial:
        groups = row.get("source_groups")
        if not isinstance(groups, Mapping):
            raise ValueError("financial prompt is missing source groups")
        if set(groups) != {"identity_header", "evidence", "instruction_context", "other_prefix"}:
            raise ValueError("financial prompt must contain the four frozen E2 source groups")
        seen: set[int] = set()
        for group in groups.values():
            for start, end in group.get("ranges", ()):
                if start < 0 or end <= start or seen.intersection(range(start, end)):
                    raise ValueError("source groups overlap or contain invalid ranges")
                seen.update(range(start, end))
        expected = set(range(int(row["final_query_position"])))
        if seen != expected:
            raise ValueError("source groups do not cover all positions before final query position")
    forbidden = {"activation", "activations", "residual", "hidden_state", "gradient", "jacobian", "attention"}
    def walk(value: Any, key: str = "") -> None:
        if any(part in key.lower().replace("-", "_").split("_") for part in forbidden):
            raise ValueError(f"raw runtime payload is not allowed: {key}")
        if hasattr(value, "detach") or value.__class__.__module__.startswith("numpy"):
            raise ValueError("prepared inputs must not contain tensors or arrays")
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child, key)
    walk(prepared)


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "class": tokenizer.__class__.__qualname__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "identity_sha256": sha256_json(
            {
                "name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
                "class": tokenizer.__class__.__qualname__,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
            }
        ),
    }


def prepare_artifacts(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    baseline_source: str | Path,
    baseline_identity: str | None,
    tokenizer: Any,
    model: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    dataset: str = "entity-cell-localization",
    split: str = "discovery",
    sector: str = TECHNOLOGY,
    baseline_expected_count: int = BASELINE_RECORD_COUNT,
) -> Path:
    """Write prepared JSONL and metadata through the shared lifecycle utilities."""
    input_path, split_manifest, baseline_source = map(Path, (input_path, split_manifest, baseline_source))
    prepared = prepare_inputs(
        tokenizer=tokenizer,
        input_path=input_path,
        split_manifest=split_manifest,
        baseline_source=baseline_source,
        baseline_identity=baseline_identity,
        split=split,
        sector=sector,
        baseline_expected_count=baseline_expected_count,
    )
    validate_prepared_inputs(prepared)
    run = ArtifactRun.create(model, dataset, run_id, artifact_root=artifact_root)
    prepare_dir = run.run_directory / "prepare"
    paths = {
        "header_variants": prepare_dir / "header_variants.jsonl",
        "generic_baseline": prepare_dir / "generic_baseline.jsonl",
        "financial_prompts": prepare_dir / "financial_prompts.jsonl",
        "metadata": prepare_dir / "metadata.json",
        "config": prepare_dir / "config.json",
    }
    try:
        for path, artifact_type in (
            (input_path, "entity_cell_source_csv"),
            (split_manifest, "entity_cell_split_manifest"),
            (baseline_source, "entity_cell_generic_baseline_input"),
        ):
            run.manifest.register_artifact(path, artifact_type=artifact_type, stage="prepare", role="input")
        with run.stage("prepare") as stage:
            counts = {
                "header_variants": write_jsonl(paths["header_variants"], prepared["header_variants"]),
                "generic_baseline": write_jsonl(paths["generic_baseline"], prepared["generic_baseline"]),
                "financial_prompts": write_jsonl(paths["financial_prompts"], prepared["financial_prompts"]),
            }
            write_json(paths["config"], prepared["config"])
            metadata = {
                "schema_version": PREPARATION_SCHEMA_VERSION,
                "artifact_type": "entity_cell_prepare_metadata",
                "experiment": "entity-cell-localization",
                "version": "v1",
                "model": model,
                "tokenizer": _tokenizer_identity(tokenizer),
                "tokenizer_sha256": _tokenizer_identity(tokenizer)["identity_sha256"],
                "split": split,
                "sector": sector,
                "input": str(input_path),
                "input_sha256": sha256_file(input_path),
                "split_manifest": str(split_manifest),
                "split_manifest_sha256": sha256_file(split_manifest),
                "baseline_input": str(baseline_source),
                "baseline_input_sha256": sha256_file(baseline_source),
                "baseline_identity": baseline_identity,
                "config_sha256": sha256_json(prepared["config"]),
                "record_counts": counts,
                "ticker_count": prepared["ticker_count"],
                "raw_runtime_payloads": False,
            }
            metadata["prepared_artifact_sha256"] = {
                key: sha256_file(paths[key]) for key in ("header_variants", "generic_baseline", "financial_prompts")
            }
            write_metadata(paths["metadata"], metadata)
            stage.count(sum(counts.values()))
        for key, artifact_type in (
            ("header_variants", "entity_cell_header_variant"),
            ("generic_baseline", "entity_cell_generic_baseline_prompt"),
            ("financial_prompts", "entity_cell_financial_prompt"),
            ("metadata", "entity_cell_prepare_metadata"),
            ("config", "entity_cell_prepare_config"),
        ):
            run.manifest.register_artifact(
                paths[key], artifact_type=artifact_type, stage="prepare", role="output",
                record_count=counts.get(key),
            )
        run.manifest.save()
        run.finalize(required_stages={"prepare"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = [
    "BASELINE_EXACT_IDENTITY",
    "BASELINE_RECORD_COUNT",
    "FINANCIAL_PROMPT_COLUMNS",
    "HEADER_VARIANT_COUNT",
    "HEADER_VARIANT_SPECS",
    "HELD_VARIANT_IDS",
    "LOCALIZATION_VARIANT_IDS",
    "load_split_manifest",
    "parse_header",
    "prepare_artifacts",
    "prepare_inputs",
    "render_header_variants",
    "select_split_rows",
    "validate_baseline_contract",
    "validate_prepared_inputs",
]
