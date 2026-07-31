"""Stream extracted 8-K JSON files into an auditable staging dataset."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable

from llm_bias.edgar_preparation.taxonomy import (
    SUPPORTING_ONLY_CODES,
    TAXONOMY_VERSION,
    item_metadata,
)

DATASET_SCHEMA_VERSION = "edgar-8k-clean-v1"
DEFAULT_INPUT = "../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/8-K"
DEFAULT_OUTPUT = "artifacts/edgar_8k/cleaned"
MIN_ANALYSIS_CHARS = 80
MIN_ALPHA_TOKENS = 12
MAX_REMOVABLE_BLOCK_CHARS = 2_500

_ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
_ITEM_KEY_RE = re.compile(r"^item_(\d+(?:\.\d+)?)$")
_NOT_APPLICABLE_RE = re.compile(
    r"^\s*(?:not\s+applicable|none|n\s*/?\s*a)\s*[\.;]?\s*$", re.IGNORECASE
)
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_EXHIBIT_RE = re.compile(r"\bexhibit(?:s)?\s+\d", re.IGNORECASE)
_NUMERIC_FACT_RE = re.compile(
    r"(?:[$€£]\s*\(?\d|\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s+"
    r"(?:thousand|million|billion)\b)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*%|\bpercent\b)", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"(?:[$€£]\s*\(?\d|\bUSD\b)", re.IGNORECASE)
_PRESS_RELEASE_RE = re.compile(r"\bpress release\b", re.IGNORECASE)
_COMPARISON_RE = re.compile(
    r"\b(?:increase[ds]?|decrease[ds]?|grew|declined|higher|lower|"
    r"compared with|versus|year[- ]over[- ]year)\b",
    re.IGNORECASE,
)

_BOILERPLATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "securities_act_boilerplate",
        re.compile(
            r"\bThe information (?:contained|furnished|included|in this).{0,600}?"
            r"shall not be deemed\s+[\"“]?filed[\"”]?.{0,1200}?"
            r"(?:\.(?=\s+[A-Z])|\.$|$)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "incorporation_by_reference",
        re.compile(
            r"\b(?:The information|It|Such information).{0,800}?"
            r"(?:shall not|will not|is not).{0,400}?"
            r"incorporat(?:ed|ion) by reference.{0,800}?"
            r"(?:\.(?=\s+[A-Z])|\.$|$)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "exhibit_reference",
        re.compile(
            r"\b(?:A copy|Copies) of .{0,800}?(?:is|are) "
            r"(?:attached|furnished|filed|included).{0,500}?"
            r"\bExhibit(?:s)?\b.{0,300}?\.",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

_LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "lp",
    "ltd",
    "limited",
    "nv",
    "plc",
}
_GENERIC_SHORT_NAMES = {"bank", "company", "corporation", "group", "holdings"}


class PreparationError(RuntimeError):
    """Raised when a source or staged dataset violates the preparation contract."""


@dataclass(frozen=True)
class RemovedBlock:
    kind: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "start": self.start, "end": self.end}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize transport artifacts without changing case, numbers, or meaning."""
    value = unicodedata.normalize("NFKC", html.unescape(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(
        character
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
        else " "
        for character in value
    )
    value = re.sub(r"[^\S\n]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _heading_span(text: str, item_code: str) -> RemovedBlock | None:
    escaped = re.escape(item_code)
    match = re.match(
        rf"^\s*ITEM\s+{escaped}\b[.\s:;\-–—]*(?:[^\n]{{0,180}})?(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return RemovedBlock("item_heading", match.start(), match.end())


def find_removed_blocks(text: str, item_code: str) -> list[RemovedBlock]:
    """Locate non-overlapping removable spans in normalized text."""
    candidates: list[RemovedBlock] = []
    heading = _heading_span(text, item_code)
    if heading:
        candidates.append(heading)
    for kind, pattern in _BOILERPLATE_PATTERNS:
        candidates.extend(
            RemovedBlock(kind, match.start(), match.end())
            for match in pattern.finditer(text)
        )
    accepted: list[RemovedBlock] = []
    ordered = sorted(candidates, key=lambda row: (row.start, -(row.end - row.start)))
    for candidate in ordered:
        if candidate.end - candidate.start > MAX_REMOVABLE_BLOCK_CHARS:
            continue
        if any(candidate.start < row.end and candidate.end > row.start for row in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda row: row.start)


def analysis_text(text: str, removed: Iterable[RemovedBlock]) -> str:
    """Remove recorded spans and normalize the remaining whitespace."""
    parts: list[str] = []
    cursor = 0
    for block in removed:
        parts.append(text[cursor : block.start])
        parts.append(" ")
        cursor = block.end
    parts.append(text[cursor:])
    return normalize_text("".join(parts))


def _company_aliases(company: str) -> list[tuple[str, str]]:
    legal_name = normalize_text(re.sub(r"\s*/[^/]+/\s*", " ", company)).strip(" ,.")
    aliases: list[tuple[str, str]] = []
    if legal_name:
        aliases.append((legal_name, "legal_name"))
    terms = re.findall(r"[A-Za-z0-9]+", legal_name)
    while terms and terms[-1].lower() in _LEGAL_SUFFIXES:
        terms.pop()
    short_name = " ".join(terms)
    if (
        short_name
        and short_name.casefold() != legal_name.casefold()
        and len(re.sub(r"[^A-Za-z]", "", short_name)) >= 4
        and short_name.casefold() not in _GENERIC_SHORT_NAMES
    ):
        aliases.append((short_name, "short_name"))
    return aliases


def _alias_pattern(alias: str) -> re.Pattern[str] | None:
    terms = re.findall(r"[A-Za-z0-9]+", alias)
    if not terms:
        return None
    body = r"[^A-Za-z0-9]+".join(re.escape(term) for term in terms)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def find_entity_mentions(text: str, company: str) -> list[dict[str, Any]]:
    """Find auditable registrant aliases; this is deliberately not general NER."""
    candidates: list[dict[str, Any]] = []
    for alias, kind in _company_aliases(company):
        pattern = _alias_pattern(alias)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            candidates.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "kind": kind,
                }
            )
    generic = re.compile(
        r"\b(?:the\s+Company|the\s+Registrant|Registrant)\b", re.IGNORECASE
    )
    for match in generic.finditer(text):
        candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "kind": "generic_reference",
            }
        )
    accepted: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates, key=lambda row: (row["start"], -(row["end"] - row["start"]))
    ):
        if any(
            candidate["start"] < row["end"] and candidate["end"] > row["start"]
            for row in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda row: row["start"])


def _item_sort_key(item_code: str) -> tuple[int, int]:
    major, _, minor = item_code.partition(".")
    return int(major), int(minor or 0)


def _validate_date(value: Any, field: str, source: Path, *, required: bool) -> str | None:
    if value in (None, "") and not required:
        return None
    if not isinstance(value, str):
        raise PreparationError(f"{source.name}: {field} must be an ISO date string")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise PreparationError(f"{source.name}: invalid {field} {value!r}") from error
    return value


def _required_string(data: dict[str, Any], field: str, source: Path) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{source.name}: missing non-empty string field {field!r}")
    return value.strip()


def _section_record(
    *,
    filing_id: str,
    item_code: str,
    source_text: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_text(source_text)
    removed = find_removed_blocks(normalized, item_code)
    analyzed = analysis_text(normalized, removed)
    title, family, item_schema = item_metadata(item_code)
    alpha_tokens = len(_ALPHA_TOKEN_RE.findall(analyzed))
    nonspace_chars = len(re.sub(r"\s", "", analyzed))
    reasons: list[str] = []
    if _NOT_APPLICABLE_RE.fullmatch(analyzed):
        status = "not_applicable"
        reasons.append("not_applicable")
    elif item_code in SUPPORTING_ONLY_CODES:
        status = "supporting_only"
        reasons.append("supporting_material_item")
    elif nonspace_chars < MIN_ANALYSIS_CHARS or alpha_tokens < MIN_ALPHA_TOKENS:
        status = "insufficient_content"
        reasons.append("analysis_text_below_threshold")
    else:
        status = "candidate"
    has_numeric_fact = bool(_NUMERIC_FACT_RE.search(analyzed))
    has_exhibit_reference = bool(_EXHIBIT_RE.search(normalized))
    announcement_only = bool(
        _PRESS_RELEASE_RE.search(analyzed)
        and has_exhibit_reference
        and not has_numeric_fact
        and not _COMPARISON_RE.search(analyzed)
    )
    if announcement_only:
        reasons.append("announcement_only")
    section_id = f"{filing_id}:item-{item_code}"
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "section_id": section_id,
        "filing_id": filing_id,
        "cik": str(data["cik"]),
        "company": str(data["company"]),
        "filing_date": str(data["filing_date"]),
        "item_code": item_code,
        "item_schema": item_schema,
        "item_title": title,
        "event_family": family,
        "taxonomy_version": TAXONOMY_VERSION,
        "candidate_status": status,
        "rejection_reasons": reasons,
        "normalized_text": normalized,
        "analysis_text": analyzed,
        "removed_blocks": [block.to_dict() for block in removed],
        "entity_mentions": find_entity_mentions(normalized, str(data["company"])),
        "entity_annotation_scope": "registrant_aliases_only",
        "normalized_char_count": len(normalized),
        "analysis_char_count": len(analyzed),
        "analysis_nonspace_char_count": nonspace_chars,
        "analysis_alpha_token_count": alpha_tokens,
        "has_numeric_fact": has_numeric_fact,
        "has_currency": bool(_CURRENCY_RE.search(analyzed)),
        "has_percent": bool(_PERCENT_RE.search(analyzed)),
        "has_exhibit_reference": has_exhibit_reference,
        "announcement_only": announcement_only,
    }


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"


def _file_record(
    source: Path, content: bytes
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        data = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{source.name}: invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise PreparationError(f"{source.name}: JSON root must be an object")
    cik = _required_string(data, "cik", source)
    company = _required_string(data, "company", source)
    filing_type = _required_string(data, "filing_type", source)
    if filing_type != "8-K":
        raise PreparationError(f"{source.name}: expected filing_type '8-K', got {filing_type!r}")
    filing_date = _validate_date(data.get("filing_date"), "filing_date", source, required=True)
    period = _validate_date(
        data.get("period_of_report"), "period_of_report", source, required=False
    )
    accession_match = _ACCESSION_RE.search(source.name)
    if not accession_match:
        raise PreparationError(f"{source.name}: cannot parse SEC accession")
    accession = accession_match.group(1)
    filing_id = f"8k:{accession}"
    item_values: list[tuple[str, str]] = []
    all_item_codes: list[str] = []
    for key, value in data.items():
        match = _ITEM_KEY_RE.fullmatch(key)
        if not match:
            continue
        item_code = match.group(1)
        all_item_codes.append(item_code)
        if not isinstance(value, str):
            raise PreparationError(f"{source.name}: {key} must be a string")
        if value.strip():
            item_values.append((item_code, value))
    all_item_codes.sort(key=_item_sort_key)
    item_values.sort(key=lambda row: _item_sort_key(row[0]))
    sections = [
        _section_record(
            filing_id=filing_id,
            item_code=item_code,
            source_text=text,
            data=data,
        )
        for item_code, text in item_values
    ]
    item_schemas = {row["item_schema"] for row in sections}
    flags: list[str] = []
    if not sections:
        flags.append("no_nonempty_sections")
    if item_schemas == {"legacy"}:
        flags.append("legacy_item_schema")
    elif len(item_schemas) > 1:
        flags.append("mixed_item_schema")
    record = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "filing_id": filing_id,
        "accession": accession,
        "source_file": source.name,
        "source_sha256": _sha256_bytes(content),
        "cik": cik,
        "company": company,
        "filing_type": filing_type,
        "filing_date": filing_date,
        "period_of_report": period,
        "sic": str(data.get("sic") or ""),
        "state_of_inc": str(data.get("state_of_inc") or ""),
        "state_location": str(data.get("state_location") or ""),
        "fiscal_year_end": str(data.get("fiscal_year_end") or ""),
        "filing_html_index": str(data.get("filing_html_index") or ""),
        "htm_filing_link": str(data.get("htm_filing_link") or ""),
        "complete_text_filing_link": str(data.get("complete_text_filing_link") or ""),
        "all_item_codes": all_item_codes,
        "nonempty_item_codes": [row[0] for row in item_values],
        "section_count": len(sections),
        "candidate_count": sum(
            row["candidate_status"] == "candidate" for row in sections
        ),
        "quality_flags": flags,
    }
    return record, sections


def _increment_nested(counter: dict[str, Counter[str]], key: str, value: str) -> None:
    counter[key][value] += 1


def clean_filings(
    input_dir: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Clean extracted 8-K files and atomically publish JSONL artifacts."""
    source_root = Path(input_dir).resolve()
    destination = Path(output_dir)
    if not source_root.is_dir():
        raise PreparationError(f"input directory does not exist: {source_root}")
    if destination.exists():
        raise PreparationError(f"output directory already exists: {destination}")
    if max_files is not None and max_files < 1:
        raise PreparationError("max_files must be positive")
    files = sorted(source_root.glob("*.json"), key=lambda path: path.name)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise PreparationError(f"no JSON files found under {source_root}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    filings_path = temporary / "filings.jsonl"
    sections_path = temporary / "sections.jsonl"
    source_fingerprint = hashlib.sha256()
    years: Counter[str] = Counter()
    companies: set[str] = set()
    ciks: set[str] = set()
    aggregate: dict[str, Counter[str]] = defaultdict(Counter)
    section_total = 0
    candidate_total = 0
    started = datetime.now(UTC)

    try:
        with filings_path.open("w", encoding="utf-8") as filings_handle, sections_path.open(
            "w", encoding="utf-8"
        ) as sections_handle:
            for index, source in enumerate(files, start=1):
                content = source.read_bytes()
                filing, sections = _file_record(source, content)
                filings_handle.write(_json_line(filing))
                for section in sections:
                    sections_handle.write(_json_line(section))
                    section_total += 1
                    candidate_total += section["candidate_status"] == "candidate"
                    _increment_nested(aggregate, "item_code", section["item_code"])
                    _increment_nested(aggregate, "event_family", section["event_family"])
                    _increment_nested(
                        aggregate, "candidate_status", section["candidate_status"]
                    )
                    for reason in section["rejection_reasons"]:
                        _increment_nested(aggregate, "reason", reason)
                years[filing["filing_date"][:4]] += 1
                companies.add(filing["company"])
                ciks.add(filing["cik"])
                source_fingerprint.update(source.name.encode("utf-8"))
                source_fingerprint.update(b"\0")
                source_fingerprint.update(filing["source_sha256"].encode("ascii"))
                source_fingerprint.update(b"\n")
                if index % 10_000 == 0 or index == len(files):
                    print(f"Processed {index}/{len(files)} filings", flush=True)

        quality_report = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "filing_count": len(files),
            "section_count": section_total,
            "candidate_count": candidate_total,
            "unique_cik_count": len(ciks),
            "unique_company_string_count": len(companies),
            "filings_by_year": dict(sorted(years.items())),
            "sections_by_item_code": dict(sorted(aggregate["item_code"].items())),
            "sections_by_event_family": dict(
                sorted(aggregate["event_family"].items())
            ),
            "sections_by_candidate_status": dict(
                sorted(aggregate["candidate_status"].items())
            ),
            "quality_reasons": dict(sorted(aggregate["reason"].items())),
        }
        quality_path = temporary / "quality_report.json"
        quality_path.write_text(
            json.dumps(quality_report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        completed = datetime.now(UTC)
        outputs = {
            name: {
                "sha256": _sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in ("filings.jsonl", "sections.jsonl", "quality_report.json")
        }
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "cleaner_version": "1",
            "created_at": completed.isoformat(),
            "duration_seconds": (completed - started).total_seconds(),
            "input_root": str(source_root),
            "input_glob": "*.json",
            "max_files": max_files,
            "source_fingerprint_sha256": source_fingerprint.hexdigest(),
            "counts": {
                "filings": len(files),
                "sections": section_total,
                "candidates": candidate_total,
            },
            "outputs": outputs,
            "thresholds": {
                "minimum_analysis_nonspace_chars": MIN_ANALYSIS_CHARS,
                "minimum_analysis_alpha_tokens": MIN_ALPHA_TOKENS,
            },
            "entity_annotation_scope": "registrant_aliases_only",
            "source_limitations": [
                "The upstream extracted filings may have removed numerical tables.",
                "Most exhibits and complete-submission documents are not present in this source.",
                "Missing extracted text must not be interpreted as a missing company disclosure.",
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PreparationError(f"{path.name}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise PreparationError(
                    f"{path.name}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise PreparationError(
                    f"{path.name}:{line_number}: row must be an object"
                )
            yield line_number, row


def validate_dataset(input_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Validate staged hashes, schemas, IDs, references, and text offsets."""
    root = Path(input_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PreparationError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise PreparationError("unsupported manifest schema_version")
    for filename, metadata in manifest.get("outputs", {}).items():
        path = root / filename
        if not path.is_file():
            raise PreparationError(f"missing output: {path}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise PreparationError(f"hash mismatch: {path}")

    filing_ids: set[str] = set()
    filing_count = 0
    for line_number, row in _read_jsonl(root / "filings.jsonl"):
        if row.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise PreparationError(
                f"filings.jsonl:{line_number}: unsupported schema_version"
            )
        filing_id = row.get("filing_id")
        if not isinstance(filing_id, str) or filing_id in filing_ids:
            raise PreparationError(
                f"filings.jsonl:{line_number}: missing or duplicate filing_id"
            )
        filing_ids.add(filing_id)
        filing_count += 1

    section_ids: set[str] = set()
    section_count = 0
    candidate_count = 0
    valid_statuses = {
        "candidate",
        "not_applicable",
        "supporting_only",
        "insufficient_content",
    }
    for line_number, row in _read_jsonl(root / "sections.jsonl"):
        prefix = f"sections.jsonl:{line_number}"
        if row.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise PreparationError(f"{prefix}: unsupported schema_version")
        section_id = row.get("section_id")
        if not isinstance(section_id, str) or section_id in section_ids:
            raise PreparationError(f"{prefix}: missing or duplicate section_id")
        section_ids.add(section_id)
        if row.get("filing_id") not in filing_ids:
            raise PreparationError(f"{prefix}: unknown filing_id")
        normalized = row.get("normalized_text")
        if not isinstance(normalized, str):
            raise PreparationError(f"{prefix}: normalized_text must be a string")
        previous_end = 0
        for block in row.get("removed_blocks", []):
            start, end = block.get("start"), block.get("end")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < previous_end
                or end <= start
                or end > len(normalized)
            ):
                raise PreparationError(f"{prefix}: invalid or overlapping removed span")
            previous_end = end
        previous_end = 0
        for mention in row.get("entity_mentions", []):
            start, end = mention.get("start"), mention.get("end")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < previous_end
                or end <= start
                or end > len(normalized)
                or normalized[start:end] != mention.get("text")
            ):
                raise PreparationError(f"{prefix}: invalid entity mention span")
            previous_end = end
        status = row.get("candidate_status")
        reasons = row.get("rejection_reasons")
        if status not in valid_statuses or not isinstance(reasons, list):
            raise PreparationError(f"{prefix}: invalid candidate status or reasons")
        if status == "candidate":
            candidate_count += 1
        elif not reasons:
            raise PreparationError(f"{prefix}: non-candidate must have a reason")
        section_count += 1

    expected = manifest.get("counts", {})
    actual = {
        "filings": filing_count,
        "sections": section_count,
        "candidates": candidate_count,
    }
    if actual != expected:
        raise PreparationError(f"manifest count mismatch: expected {expected}, got {actual}")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "valid": True,
        "counts": actual,
    }
