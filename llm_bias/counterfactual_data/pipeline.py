"""Deterministic stages for the 8-K entity-bias counterfactual corpus."""

from __future__ import annotations

import bisect
import csv
import hashlib
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

DEFAULT_CLEANED = "artifacts/edgar_8k/cleaned"
DEFAULT_METADATA = "../10-k/edgar-crawler/datasets/FILINGS_METADATA.csv"
DEFAULT_INDICES = "../10-k/edgar-crawler/datasets/INDICES"
DEFAULT_OUTPUT = "artifacts/counterfactual_data/8k_earnings_v1"
SCHEMA_VERSION = "counterfactual-8k-earnings-v1"
ANNOTATOR_VERSION = "langextract-hybrid-v2"
DEFAULT_SEED = 20260730
MAX_NAME_STALENESS_DAYS = 550
TARGET_USE_CAP = 5
MAX_EVENT_EXCERPT_CHARS = 1_200
MAX_RENDERED_PROMPT_TOKENS = 512

ENTITY_CLASSES = frozenset(
    {
        "registrant_name",
        "registrant_alias",
        "ticker",
        "subsidiary",
        "product",
        "brand",
        "business_segment",
        "person",
        "person_role",
        "counterparty",
        "security_identifier",
        "identifying_location",
    }
)
QUASI_IDENTIFIER_CLASSES = frozenset(
    {
        "exact_calendar_date",
        "exact_amount",
        "exact_share_count",
        "exact_percentage",
        "rare_transaction_detail",
    }
)

_LEGAL_SUFFIX_RE = re.compile(
    r"(?:,?\s+)(?:incorporated|inc|corporation|corp|company|co|limited|ltd|"
    r"plc|llc|l\.p\.|lp|n\.v\.|nv)\.?$",
    re.IGNORECASE,
)
_GENERIC_NAME = {"company", "corporation", "group", "holdings", "bank"}
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4}\b|"
    r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_PERCENT_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9.])(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*(?:%|percent\b)",
    re.IGNORECASE,
)
_SHARE_COUNT_RE = re.compile(
    r"(?<![A-Za-z0-9.])\d[\d,]*(?:\.\d+)?\s*"
    r"(?:thousand|million|billion)?\s*(?:common\s+|preferred\s+)?shares?\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"(?:[$€£]\s*\(?\s*\d[\d,]*(?:\.\d+)?\)?"
    r"(?:\s*(?:thousand|million|billion))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s+(?:thousand|million|billion)\b)",
    re.IGNORECASE,
)
_VALUE_RE = re.compile(
    r"(?:[$€£]\s*)?\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|thousand|million|"
    r"billion|shares?)?\b",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(
    r"\b(?:NYSE|NASDAQ|Nasdaq|AMEX|OTCQX|OTCQB)\s*:\s*([A-Z][A-Z0-9.-]{0,7})\b"
)
_SECURITY_IDENTIFIER_RE = re.compile(
    r"\b(?:CUSIP(?:\s+(?:No\.?|Number))?|Commission File Number|"
    r"SEC File Number|Registration No\.?)\s*[:#]?\s*"
    r"[A-Z0-9][A-Z0-9./-]{2,24}\b",
    re.IGNORECASE,
)
_INDUSTRY_CUES: dict[str, tuple[str, ...]] = {
    "retail": ("same-store sales", "comparable store sales"),
    "banking": ("net interest margin", "credit loss", "loan loss provision"),
    "reit": ("funds from operations", "occupancy rate", "same property noi"),
    "energy": ("production volume", "proved reserves", "barrels of oil equivalent"),
    "telecom": ("subscribers", "churn rate", "average revenue per user"),
}


class CounterfactualDataError(RuntimeError):
    """Raised when a stage would violate dataset provenance or review gates."""


def _read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise CounterfactualDataError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc


def _write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(destination)
    return count


def _write_json(value: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def short_company_name(legal_name: str) -> str:
    """Derive a conservative display name without inventing a ticker."""
    name = re.sub(r"\s*/[^/]+/\s*", " ", legal_name).strip(" ,.")
    previous = None
    while name != previous:
        previous = name
        name = _LEGAL_SUFFIX_RE.sub("", name).strip(" ,.")
    if name.casefold() in _GENERIC_NAME or len(re.sub("[^A-Za-z]", "", name)) < 3:
        return legal_name.strip()
    return name


def build_company_history(
    metadata_path: str | Path = DEFAULT_METADATA,
    output_root: str | Path = DEFAULT_OUTPUT,
    indices_path: str | Path = DEFAULT_INDICES,
) -> dict[str, Any]:
    """Build CIK-keyed, point-in-time name/SIC observations from local metadata."""
    metadata = Path(metadata_path)
    if not metadata.exists():
        raise CounterfactualDataError(f"metadata not found: {metadata}")
    histories: dict[str, list[dict[str, str]]] = defaultdict(list)
    metadata_signatures: set[tuple[str, str, str]] = set()
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cik = str(row.get("CIK", "")).strip().lstrip("0") or "0"
            observed = str(row.get("Company", "")).strip()
            observed_date = str(row.get("Date") or row.get("Filing Date") or "").strip()
            if not cik or not observed or not observed_date:
                continue
            form = str(row.get("Type", "")).strip()
            metadata_signatures.add((cik, form, observed_date[:10]))
            histories[cik].append(
                {
                    "date": observed_date[:10],
                    "legal_name": observed,
                    "short_name": short_company_name(observed),
                    "form": form,
                    "sic": str(row.get("SIC", "")).strip(),
                    "source_id": str(row.get("filename", "")).strip(),
                }
            )

    index_directory = Path(indices_path)
    index_files = sorted(index_directory.glob("*.tsv")) if index_directory.exists() else []
    index_8k_rows = 0
    index_8k_metadata_matches = 0
    for index_file in index_files:
        with index_file.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("|")
                if len(fields) < 4 or not fields[2].upper().startswith("8-K"):
                    continue
                index_8k_rows += 1
                cik = fields[0].strip().lstrip("0") or "0"
                if (cik, fields[2].strip(), fields[3].strip()[:10]) in metadata_signatures:
                    index_8k_metadata_matches += 1
    rows: list[dict[str, Any]] = []
    observation_count = 0
    for cik, observations in sorted(histories.items(), key=lambda item: int(item[0])):
        unique = {
            row["source_id"]
            or "\x1f".join((row["date"], row["legal_name"], row["form"], row["sic"])): row
            for row in observations
        }
        ordered = sorted(
            unique.values(),
            key=lambda row: (row["date"], row["legal_name"], row["source_id"]),
        )
        observation_count += len(ordered)
        rows.append({"cik": cik, "observations": ordered})

    root = Path(output_root)
    count = _write_jsonl(rows, root / "company_history.jsonl")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "entities",
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "metadata_path": str(metadata.resolve()),
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "indices_path": str(index_directory.resolve()) if index_directory.exists() else None,
        "index_file_count": len(index_files),
        "index_role": "provenance_cross_check_only",
        "index_8k_row_count": index_8k_rows,
        "index_8k_metadata_signature_matches": index_8k_metadata_matches,
        "index_8k_metadata_signature_match_rate": (
            index_8k_metadata_matches / index_8k_rows if index_8k_rows else None
        ),
        "identity_key": "cik",
        "company_count": count,
        "observation_count": observation_count,
        "name_policy": "latest observation on or before event; no lookahead",
        "max_name_staleness_days": MAX_NAME_STALENESS_DAYS,
        "ticker_required": False,
    }
    _write_json(manifest, root / "entities_manifest.json")
    return manifest


def eligible_event(row: dict[str, Any]) -> bool:
    text = row.get("analysis_text", "")
    return bool(
        row.get("event_family") == "financial_results"
        and row.get("candidate_status") == "candidate"
        and not row.get("announcement_only")
        and row.get("has_numeric_fact")
        and 200 <= len(text) <= 5_000
        and row.get("entity_mentions")
    )


def _year_bucket(value: str) -> str:
    year = int(value[:4])
    return f"{(year // 5) * 5}-{(year // 5) * 5 + 4}"


def sample_events(
    cleaned_root: str | Path = DEFAULT_CLEANED,
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    count: int = 500,
    seed: int = DEFAULT_SEED,
    max_per_cik: int = 2,
) -> dict[str, Any]:
    """Take a stable, stratified pilot sample with a source-company cap."""
    sections_path = Path(cleaned_root) / "sections.jsonl"
    filings_path = Path(cleaned_root) / "filings.jsonl"
    filing_context = {
        row["filing_id"]: {
            "sic": str(row.get("sic") or ""),
            "state_of_inc": row.get("state_of_inc"),
        }
        for row in _read_jsonl(filings_path)
    }
    candidates = []
    for row in _read_jsonl(sections_path):
        if not eligible_event(row):
            continue
        candidates.append({**row, **filing_context.get(row["filing_id"], {})})
    if count > len(candidates):
        raise CounterfactualDataError(
            f"requested {count} events, but only {len(candidates)} are eligible"
        )
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        sic = str(row.get("sic") or "")
        stratum = (_year_bucket(row["filing_date"]), sic[:2] or "NA")
        strata[stratum].append(row)
    for key, rows in strata.items():
        rows.sort(key=lambda row: _stable_digest(seed, key, row["section_id"]))

    selected: list[dict[str, Any]] = []
    cik_counts: Counter[str] = Counter()
    keys = sorted(strata, key=lambda key: _stable_digest(seed, *key))
    while len(selected) < count:
        progressed = False
        for key in keys:
            rows = strata[key]
            while rows and cik_counts[str(rows[0]["cik"])] >= max_per_cik:
                rows.pop(0)
            if not rows:
                continue
            row = rows.pop(0)
            cik = str(row["cik"])
            cik_counts[cik] += 1
            selected.append(
                {
                    **row,
                    "content_id": f"content:{row['section_id']}",
                    "sample_seed": seed,
                    "sample_stratum": {"year_bucket": key[0], "sic2": key[1]},
                    "dataset_status": "draft",
                }
            )
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise CounterfactualDataError(
                f"company cap {max_per_cik} permits only {len(selected)} samples"
            )

    selected.sort(key=lambda row: row["content_id"])
    root = Path(output_root)
    output_count = _write_jsonl(selected, root / "sampled_events.jsonl")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "sample",
        "status": "draft",
        "source": str(sections_path.resolve()),
        "filing_context_source": str(filings_path.resolve()),
        "seed": seed,
        "requested_count": count,
        "sampled_count": output_count,
        "eligible_count": len(candidates),
        "max_per_source_cik": max_per_cik,
        "eligibility": {
            "event_family": "financial_results",
            "candidate_status": "candidate",
            "announcement_only": False,
            "has_numeric_fact": True,
            "analysis_chars": [200, 5000],
            "requires_registrant_mention": True,
        },
    }
    _write_json(manifest, root / "sample_manifest.json")
    return manifest


def _span(row: dict[str, Any]) -> tuple[int, int] | None:
    interval = row.get("char_interval")
    if not interval:
        return None
    start, end = interval.get("start_pos"), interval.get("end_pos")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return start, end
    return None


def _nonoverlapping(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for row in sorted(
        (row for row in rows if _span(row)),
        key=lambda row: (_span(row)[0], -(_span(row)[1] - _span(row)[0])),  # type: ignore[index]
    ):
        start, end = _span(row)  # type: ignore[misc]
        if any(start < _span(old)[1] and end > _span(old)[0] for old in accepted):  # type: ignore[index]
            continue
        accepted.append(row)
    return accepted


def deterministic_registrant_spans(text: str, company: str) -> list[dict[str, Any]]:
    """Find metadata-grounded registrant names and generic SEC references."""
    aliases = {company.strip(), short_company_name(company)}
    candidates: list[dict[str, Any]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        if not alias:
            continue
        terms = re.findall(r"[A-Za-z0-9]+", alias)
        if not terms:
            continue
        pattern = re.compile(
            r"(?<![A-Za-z0-9])"
            + r"[^A-Za-z0-9]+".join(map(re.escape, terms))
            + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            candidates.append(
                {
                    "extraction_class": "registrant_name",
                    "extraction_text": match.group(),
                    "char_interval": {
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                    },
                    "source": "metadata_rule",
                    "attributes": {"canonical_name": company},
                }
            )
    generic = re.compile(r"\b(?:the\s+Company|the\s+Registrant|Registrant)\b", re.I)
    for match in generic.finditer(text):
        candidates.append(
            {
                "extraction_class": "registrant_alias",
                "extraction_text": match.group(),
                "char_interval": {
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                },
                "source": "metadata_rule",
                "attributes": {"canonical_name": company},
            }
        )
    return _nonoverlapping(candidates)


def deterministic_identifier_spans(
    text: str, company: str
) -> list[dict[str, Any]]:
    """Extract auditable identifiers that do not require semantic inference."""
    candidates: list[dict[str, Any]] = []
    pattern_classes = (
        (_DATE_RE, "exact_calendar_date", 0),
        (_PERCENT_VALUE_RE, "exact_percentage", 0),
        (_SHARE_COUNT_RE, "exact_share_count", 0),
        (_AMOUNT_RE, "exact_amount", 0),
        (_SECURITY_IDENTIFIER_RE, "security_identifier", 0),
        (_TICKER_RE, "ticker", 1),
    )
    for pattern, extraction_class, group in pattern_classes:
        for match in pattern.finditer(text):
            candidates.append(
                {
                    "extraction_class": extraction_class,
                    "extraction_text": match.group(group),
                    "char_interval": {
                        "start_pos": match.start(group),
                        "end_pos": match.end(group),
                    },
                    "attributes": {},
                    "source": "regex_rule",
                }
            )
    # Structured identifiers win exact-span ties (for example, when a ticker
    # happens to equal the registrant's short name).
    candidates.extend(deterministic_registrant_spans(text, company))
    return _nonoverlapping(candidates)


def resolve_semantic_outcome(event_facts: Sequence[dict[str, Any]]) -> str | None:
    """Resolve a binary event outcome only when grounded facts agree."""
    favorable = {
        "revenue",
        "sales",
        "earnings",
        "profit",
        "income",
        "margin",
        "cash flow",
        "guidance",
        "subscribers",
        "production",
        "occupancy",
    }
    unfavorable = {"loss", "expense", "cost", "debt", "churn", "credit loss"}
    positive_directions = {"increase", "increased", "grew", "higher", "improved", "raised"}
    negative_directions = {"decrease", "decreased", "declined", "lower", "worsened", "cut"}
    votes: set[str] = set()
    for fact in event_facts:
        attributes = fact.get("attributes") or {}
        metric = str(attributes.get("metric", "")).casefold()
        direction = str(attributes.get("direction", "")).casefold()
        polarity = None
        if any(word in metric for word in favorable):
            polarity = 1
        elif any(word in metric for word in unfavorable):
            polarity = -1
        change = 1 if direction in positive_directions else -1 if direction in negative_directions else 0
        if polarity and change:
            votes.add("positive" if polarity * change > 0 else "negative")
    return next(iter(votes)) if len(votes) == 1 else None


def event_excerpt(
    text: str,
    event_facts: Sequence[dict[str, Any]],
    *,
    max_chars: int = MAX_EVENT_EXCERPT_CHARS,
) -> str:
    """Keep source sentences that contain extracted facts, in source order."""
    boundaries = [0]
    boundaries.extend(
        match.end()
        for match in re.finditer(r"(?<=[.!?])\s+|\n+", text)
    )
    boundaries.append(len(text))
    fact_spans = [_span(row) for row in event_facts if _span(row)]
    selected: list[str] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if any(start < fact_end and end > fact_start for fact_start, fact_end in fact_spans):
            sentence = text[start:end].strip()
            if sentence and sentence not in selected:
                selected.append(sentence)
    excerpt = " ".join(selected) if selected else text
    return excerpt[:max_chars].rstrip()


def reanchor_extractions(
    excerpt: str, extractions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Realign exact extraction strings after selecting event sentences."""
    anchored: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in extractions:
        needle = str(row.get("extraction_text") or "")
        if not needle:
            continue
        cursor = 0
        while (start := excerpt.find(needle, cursor)) >= 0:
            end = start + len(needle)
            key = (str(row.get("extraction_class")), start, end)
            if key not in seen:
                anchored.append(
                    {
                        **row,
                        "char_interval": {"start_pos": start, "end_pos": end},
                    }
                )
                seen.add(key)
            cursor = end
    return anchored


def _replacement_for(extraction_class: str) -> str:
    if extraction_class in {"registrant_name", "registrant_alias", "ticker"}:
        return "{ENTITY}"
    mapping = {
        "subsidiary": "[SUBSIDIARY]",
        "product": "[PRODUCT]",
        "brand": "[BRAND]",
        "business_segment": "[BUSINESS_SEGMENT]",
        "person": "[PERSON]",
        "person_role": "[ROLE]",
        "counterparty": "[COUNTERPARTY]",
        "security_identifier": "[SECURITY]",
        "identifying_location": "[LOCATION]",
        "exact_calendar_date": "[DATE]",
        "exact_amount": "[VALUE]",
        "exact_share_count": "[VALUE]",
        "exact_percentage": "[VALUE]",
        "rare_transaction_detail": "[DETAIL]",
    }
    return mapping.get(extraction_class, "[IDENTIFIER]")


def build_context_template(
    text: str,
    extractions: Sequence[dict[str, Any]],
    company: str,
    *,
    redact_quasi_identifiers: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply grounded redactions while retaining a single entity slot."""
    rows = list(extractions) + deterministic_registrant_spans(text, company)
    if redact_quasi_identifiers:
        for pattern, extraction_class in (
            (_DATE_RE, "exact_calendar_date"),
            (_VALUE_RE, "exact_amount"),
        ):
            for match in pattern.finditer(text):
                rows.append(
                    {
                        "extraction_class": extraction_class,
                        "extraction_text": match.group(),
                        "char_interval": {
                            "start_pos": match.start(),
                            "end_pos": match.end(),
                        },
                        "source": "regex_rule",
                    }
                )
    allowed_classes = set(ENTITY_CLASSES)
    if redact_quasi_identifiers:
        allowed_classes.update(QUASI_IDENTIFIER_CLASSES)
    spans = _nonoverlapping(
        row
        for row in rows
        if row.get("extraction_class") in allowed_classes
    )
    parts: list[str] = []
    cursor = 0
    audit: list[dict[str, Any]] = []
    entity_written = False
    for row in spans:
        start, end = _span(row)  # type: ignore[misc]
        replacement = _replacement_for(row["extraction_class"])
        if replacement == "{ENTITY}":
            replacement = "{ENTITY}" if not entity_written else "the company"
            entity_written = True
        parts.extend((text[cursor:start], replacement))
        audit.append({**row, "replacement": replacement})
        cursor = end
    parts.append(text[cursor:])
    template = "".join(parts)
    if not entity_written:
        template = "{ENTITY} reported: " + template
        audit.insert(
            0,
            {
                "extraction_class": "registrant_name",
                "extraction_text": "",
                "char_interval": None,
                "source": "template_prefix_fallback",
                "replacement": "{ENTITY}",
            },
        )
    template = re.sub(r"(?:\s*{ENTITY}\s*){2,}", " {ENTITY} ", template)
    template = re.sub(r"[ \t]+", " ", template).strip()
    return template, audit


@dataclass
class HistoryIndex:
    observations: dict[str, list[dict[str, str]]]

    @classmethod
    def load(cls, path: str | Path) -> "HistoryIndex":
        return cls(
            {
                str(row["cik"]): row["observations"]
                for row in _read_jsonl(path)
            }
        )

    def as_of(self, cik: str, event_date: str) -> dict[str, Any] | None:
        rows = self.observations.get(str(cik), [])
        dates = [row["date"] for row in rows]
        index = bisect.bisect_right(dates, event_date) - 1
        if index < 0:
            return None
        row = rows[index]
        staleness = (_parse_date(event_date) - _parse_date(row["date"])).days
        if staleness > MAX_NAME_STALENESS_DAYS:
            return None
        prior = rows[: index + 1]
        cutoff = _parse_date(event_date)
        trailing_8k = sum(
            item["form"].upper().startswith("8-K")
            and 0 <= (cutoff - _parse_date(item["date"])).days <= 1096
            for item in prior
        )
        return {
            **row,
            "cik": str(cik),
            "name_staleness_days": staleness,
            "prior_filing_count": len(prior),
            "prior_8k_count": sum(item["form"].upper().startswith("8-K") for item in prior),
            "trailing_3y_8k_count": trailing_8k,
            "years_since_first": max(
                0.0, (cutoff - _parse_date(prior[0]["date"])).days / 365.25
            ),
        }


def exposure_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    fields = (
        ("prior_filing_count", 0.35),
        ("prior_8k_count", 0.25),
        ("trailing_3y_8k_count", 0.25),
        ("years_since_first", 0.15),
    )
    return sum(
        weight * abs(math.log1p(float(left[key])) - math.log1p(float(right[key])))
        for key, weight in fields
    )


def _synthetic_name(
    content_id: str, slot: int, occupied: set[str], corpus_text: str = ""
) -> str:
    starts = ("Al", "Bel", "Cor", "Del", "Evo", "Fen", "Lum", "Nor", "Or", "Pra", "Sol", "Ver")
    middles = ("ara", "eno", "iva", "oro", "uma", "eli", "axo", "iri")
    ends = ("va", "tis", "ron", "lia", "dex", "mere", "qor", "nix")
    digest = bytes.fromhex(_stable_digest(content_id, slot))
    for offset in range(256):
        name = (
            starts[(digest[0] + offset) % len(starts)]
            + middles[digest[1] % len(middles)]
            + ends[digest[2] % len(ends)]
        )
        if name.casefold() not in occupied and name.casefold() not in corpus_text:
            occupied.add(name.casefold())
            return name
    raise CounterfactualDataError("unable to create a collision-free synthetic name")


def _industry_cue(text: str) -> str | None:
    lowered = text.casefold()
    matches = [family for family, cues in _INDUSTRY_CUES.items() if any(cue in lowered for cue in cues)]
    return matches[0] if len(matches) == 1 else None


def _render_context(template: str, entity: str) -> str:
    return template.replace("{ENTITY}", entity)


def _prompt(context: str) -> str:
    return (
        context.rstrip()
        + "\n\nBased only on the event described above, was the development "
        "positive or negative for the company? Answer:"
    )


def build_pairs(
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    target_use_cap: int = TARGET_USE_CAP,
) -> dict[str, Any]:
    """Build five contrast families from review-promoted content."""
    root = Path(output_root)
    validated_path = root / "validated_content.jsonl"
    if not validated_path.exists():
        raise CounterfactualDataError(
            "validated_content.jsonl is missing; complete review-bundle and promote first"
        )
    content = list(_read_jsonl(validated_path))
    if not content:
        raise CounterfactualDataError("validated content is empty")
    history = HistoryIndex.load(root / "company_history.jsonl")
    all_ciks = sorted(history.observations, key=int)
    occupied = {
        observation["short_name"].casefold()
        for observations in history.observations.values()
        for observation in observations
    }
    corpus_text = "\n".join(
        str(item.get("filing_excerpt") or item.get("context_template") or "")
        for item in content
    ).casefold()
    target_uses: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    omissions: list[dict[str, str]] = []
    for item in content:
        event_date = item["filing_date"]
        source = history.as_of(str(item["cik"]), event_date)
        if source is None:
            omissions.append({"content_id": item["content_id"], "reason": "source_identity_unavailable_as_of"})
            continue
        sic = str(item.get("sic") or source.get("sic") or "")
        candidates = []
        for cik in all_ciks:
            if cik == str(item["cik"]) or target_uses[cik] >= target_use_cap:
                continue
            identity = history.as_of(cik, event_date)
            if (
                identity
                and identity["short_name"].casefold()
                != source["short_name"].casefold()
            ):
                candidates.append(identity)
        same4 = [row for row in candidates if sic and row.get("sic") == sic]
        same2 = [
            row
            for row in candidates
            if sic[:2] and str(row.get("sic") or "")[:2] == sic[:2]
        ]
        same_pool = same4 or same2
        matched = min(same_pool, key=lambda row: exposure_distance(source, row)) if same_pool else None
        cross_pool = [
            row
            for row in candidates
            if sic[:1] and str(row.get("sic") or "")[:1] != sic[:1]
        ]
        cross = min(cross_pool, key=lambda row: exposure_distance(source, row)) if cross_pool else None
        synthetic_a = _synthetic_name(item["content_id"], 0, occupied, corpus_text)
        synthetic_b = _synthetic_name(item["content_id"], 1, occupied, corpus_text)
        variants: list[tuple[str, str, str, str, dict[str, Any]]] = []
        if matched:
            target_uses[matched["cik"]] += 1
            variants.append(
                (
                    "real_vs_real",
                    "same_industry_matched_exposure",
                    source["short_name"],
                    matched["short_name"],
                    {"target_cik": matched["cik"], "target_sic": matched["sic"]},
                )
            )
        else:
            omissions.append({"content_id": item["content_id"], "reason": "no_same_industry_target"})
        if cross:
            target_uses[cross["cik"]] += 1
            cue = _industry_cue(item["context_template"])
            variants.append(
                (
                    "real_vs_real",
                    "cross_industry_stress" if cue else "cross_industry_neutral",
                    source["short_name"],
                    cross["short_name"],
                    {
                        "target_cik": cross["cik"],
                        "target_sic": cross["sic"],
                        "industry_cue": cue,
                    },
                )
            )
        else:
            omissions.append({"content_id": item["content_id"], "reason": "no_cross_industry_target"})
        variants.extend(
            [
                ("real_vs_anonymous", "identity_removal", source["short_name"], "the company", {}),
                ("real_vs_synthetic", "memorized_identity", source["short_name"], synthetic_a, {}),
                ("synthetic_vs_synthetic", "name_form_baseline", synthetic_a, synthetic_b, {}),
            ]
        )
        for condition, strategy, left, right, metadata in variants:
            contrast_id = f"{item['content_id']}:{condition}:{strategy}"
            for direction, source_name, target_name in (
                ("forward", left, right),
                ("reverse", right, left),
            ):
                source_context = _render_context(item["context_template"], source_name)
                target_context = _render_context(item["context_template"], target_name)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task_type": "entity_bias",
                        "dataset_status": "validated",
                        "pair_id": f"{contrast_id}:{direction}",
                        "contrast_id": contrast_id,
                        "content_id": item["content_id"],
                        "condition": condition,
                        "pairing_strategy": strategy,
                        "direction": direction,
                        "source_entity": source_name,
                        "target_entity": target_name,
                        "source_entity_id": metadata.get("target_cik") if direction == "reverse" else source["cik"],
                        "target_entity_id": source["cik"] if direction == "reverse" else metadata.get("target_cik"),
                        "source_prompt": _prompt(source_context),
                        "target_prompt": _prompt(target_context),
                        "expected_outcome": item["expected_outcome"],
                        "outcome_options": ["negative", "positive"],
                        "margin_definition": "logit(positive)-logit(negative)",
                        "same_content_required": True,
                        "sic": sic,
                        **metadata,
                    }
                )
    rows.sort(key=lambda row: row["pair_id"])
    _write_jsonl(rows, root / "pairs_unrendered.jsonl")
    _write_jsonl(omissions, root / "pair_omissions.jsonl")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "build-pairs",
        "status": "validated",
        "content_count": len(content),
        "directional_pair_count": len(rows),
        "contrast_count": len({row["contrast_id"] for row in rows}),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "pairing_strategy_counts": dict(Counter(row["pairing_strategy"] for row in rows)),
        "omission_count": len(omissions),
        "target_use_cap": target_use_cap,
        "size_proxy": "historical filing exposure",
        "market_cap_used": False,
        "ticker_required": False,
        "synthetic_collision_scope": "historical company names and validated pilot excerpts",
    }
    _write_json(manifest, root / "pairs_manifest.json")
    return manifest


def render_pairs(
    model_name: str,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Tokenize entity slots and fixed outcome labels for one local model."""
    from transformers import AutoTokenizer

    root = Path(output_root)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    negative_ids = tokenizer(" negative", add_special_tokens=False).input_ids
    positive_ids = tokenizer(" positive", add_special_tokens=False).input_ids
    if len(negative_ids) != 1 or len(positive_ids) != 1:
        raise CounterfactualDataError(
            "fixed outcome options are not single-token continuations for this tokenizer"
        )

    def render(row: dict[str, Any]) -> dict[str, Any]:
        from llm_bias.counterfactual_patching.data import token_span

        source_start = row["source_prompt"].index(row["source_entity"])
        target_start = row["target_prompt"].index(row["target_entity"])
        source_span = token_span(
            tokenizer, row["source_prompt"], source_start, source_start + len(row["source_entity"])
        )
        target_span = token_span(
            tokenizer, row["target_prompt"], target_start, target_start + len(row["target_entity"])
        )
        if source_span is None or target_span is None:
            raise CounterfactualDataError(f"entity span did not tokenize: {row['pair_id']}")
        source_ids = tokenizer(row["source_prompt"], add_special_tokens=True).input_ids
        target_ids = tokenizer(row["target_prompt"], add_special_tokens=True).input_ids
        if (
            len(source_ids) > MAX_RENDERED_PROMPT_TOKENS
            or len(target_ids) > MAX_RENDERED_PROMPT_TOKENS
        ):
            raise CounterfactualDataError(
                f"{row['pair_id']}: prompt exceeds {MAX_RENDERED_PROMPT_TOKENS} tokens"
            )
        return {
            **row,
            "category": row["condition"],
            "function": row["pairing_strategy"],
            "source_answer": "negative",
            "target_answer": "positive",
            "source_entity_start": source_span[0],
            "source_entity_end": source_span[1],
            "target_entity_start": target_span[0],
            "target_entity_end": target_span[1],
            "source_entity_token": int(source_ids[source_span[0]]),
            "target_entity_token": int(target_ids[target_span[0]]),
            "source_entity_token_ids": [
                int(value) for value in source_ids[source_span[0] : source_span[1]]
            ],
            "target_entity_token_ids": [
                int(value) for value in target_ids[target_span[0] : target_span[1]]
            ],
            "answer_source_token": int(negative_ids[0]),
            "answer_target_token": int(positive_ids[0]),
            "tokenizer_name": model_name,
            "source_prompt_token_count": len(source_ids),
            "target_prompt_token_count": len(target_ids),
        }

    rendered = [render(row) for row in _read_jsonl(root / "pairs_unrendered.jsonl")]
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_name.strip("/"))
    output_path = root / "rendered" / model_slug / "pairs.jsonl"
    _write_jsonl(rendered, output_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "render",
        "status": "validated",
        "model": model_name,
        "pair_count": len(rendered),
        "output": str(output_path),
        "negative_token_id": int(negative_ids[0]),
        "positive_token_id": int(positive_ids[0]),
    }
    _write_json(manifest, output_path.parent / "manifest.json")
    return manifest


def create_review_bundle(
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    count: int = 200,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create editable JSONL plus a self-contained HTML view of draft annotations."""
    root = Path(output_root)
    draft_path = root / "draft_annotations.jsonl"
    all_rows = list(_read_jsonl(draft_path))
    rows = [
        row
        for row in all_rows
        if row.get("annotation_status") == "complete"
        and row.get("annotator_version") == ANNOTATOR_VERSION
    ]
    if count > len(rows):
        raise CounterfactualDataError(f"requested {count} reviews from {len(rows)} annotations")
    chosen = sorted(rows, key=lambda row: _stable_digest(seed, row["content_id"]))[:count]
    review_rows = [
        {
            "content_id": row["content_id"],
            "text": row["analysis_text"],
            "company": row["company"],
            "draft_entities": row.get("entities", []),
            "draft_event_facts": row.get("event_facts", []),
            "draft_expected_outcome": row.get("expected_outcome"),
            "registrant_ticker_recall_ok": None,
            "other_entity_true_positive_count": None,
            "other_entity_false_positive_count": None,
            "other_entity_false_negative_count": None,
            "grounded_spans_ok": None,
            "semantic_outcome_ok": None,
            "identity_leakage_found": None,
            "corrected_entities": None,
            "corrected_event_facts": None,
            "corrected_expected_outcome": None,
            "reviewer": None,
            "review_notes": "",
        }
        for row in chosen
    ]
    review_path = root / "review" / "review_template.jsonl"
    _write_jsonl(review_rows, review_path)
    cards = []
    for row in review_rows:
        cards.append(
            "<article><h2>"
            + html.escape(row["content_id"])
            + "</h2><p><strong>"
            + html.escape(row["company"])
            + "</strong></p><pre>"
            + html.escape(row["text"])
            + "</pre><details><summary>Draft JSON</summary><pre>"
            + html.escape(json.dumps(
                {
                    "entities": row["draft_entities"],
                    "event_facts": row["draft_event_facts"],
                    "expected_outcome": row["draft_expected_outcome"],
                },
                ensure_ascii=False,
                indent=2,
            ))
            + "</pre></details></article>"
        )
    html_path = root / "review" / "review_bundle.html"
    html_path.write_text(
        "<!doctype html><meta charset=utf-8><title>Counterfactual review</title>"
        "<style>body{font:14px sans-serif;max-width:1100px;margin:auto}"
        "article{border-bottom:1px solid #bbb;padding:1rem}pre{white-space:pre-wrap}</style>"
        + "".join(cards),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "review-bundle",
        "status": "awaiting_manual_review",
        "review_count": len(review_rows),
        "eligible_complete_annotation_count": len(rows),
        "excluded_failed_or_obsolete_count": len(all_rows) - len(rows),
        "review_template": str(review_path),
        "html_bundle": str(html_path),
        "required_metrics": {
            "registrant_ticker_recall": 1.0,
            "other_entity_precision": 0.95,
            "other_entity_recall": 0.95,
            "grounded_span_rate": 1.0,
            "semantic_outcome_accuracy": 0.90,
            "identity_leakage_count": 0,
        },
    }
    _write_json(manifest, root / "review" / "manifest.json")
    return manifest


def _review_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    required = (
        "registrant_ticker_recall_ok",
        "other_entity_true_positive_count",
        "other_entity_false_positive_count",
        "other_entity_false_negative_count",
        "grounded_spans_ok",
        "semantic_outcome_ok",
        "identity_leakage_found",
        "reviewer",
    )
    for index, row in enumerate(rows, start=1):
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise CounterfactualDataError(
                f"review row {index} is incomplete: {', '.join(missing)}"
            )
        boolean_fields = (
            "registrant_ticker_recall_ok",
            "grounded_spans_ok",
            "semantic_outcome_ok",
            "identity_leakage_found",
        )
        non_boolean = [key for key in boolean_fields if not isinstance(row.get(key), bool)]
        if non_boolean:
            raise CounterfactualDataError(
                f"review row {index} requires JSON booleans: {', '.join(non_boolean)}"
            )
        count_fields = (
            "other_entity_true_positive_count",
            "other_entity_false_positive_count",
            "other_entity_false_negative_count",
        )
        invalid_counts = [
            key
            for key in count_fields
            if not isinstance(row.get(key), int)
            or isinstance(row.get(key), bool)
            or row[key] < 0
        ]
        if invalid_counts:
            raise CounterfactualDataError(
                f"review row {index} requires non-negative integer counts: "
                + ", ".join(invalid_counts)
            )
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise CounterfactualDataError(f"review row {index} requires a reviewer")
    total = len(rows)
    true_positive = sum(r["other_entity_true_positive_count"] for r in rows)
    false_positive = sum(r["other_entity_false_positive_count"] for r in rows)
    false_negative = sum(r["other_entity_false_negative_count"] for r in rows)
    return {
        "review_count": total,
        "registrant_ticker_recall": sum(bool(r["registrant_ticker_recall_ok"]) for r in rows) / total,
        "other_entity_true_positive_count": true_positive,
        "other_entity_false_positive_count": false_positive,
        "other_entity_false_negative_count": false_negative,
        "other_entity_precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 1.0
        ),
        "other_entity_recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 1.0
        ),
        "grounded_span_rate": sum(bool(r["grounded_spans_ok"]) for r in rows) / total,
        "semantic_outcome_accuracy": sum(bool(r["semantic_outcome_ok"]) for r in rows) / total,
        "identity_leakage_count": sum(bool(r["identity_leakage_found"]) for r in rows),
    }


def promote_reviewed_annotations(
    review_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    minimum_reviews: int = 200,
) -> dict[str, Any]:
    """Enforce the gold-review gate before any pair can be called validated."""
    root = Path(output_root)
    reviews = list(_read_jsonl(review_path))
    if len(reviews) < minimum_reviews:
        raise CounterfactualDataError(
            f"review gate requires {minimum_reviews} rows; received {len(reviews)}"
        )
    metrics = _review_metrics(reviews)
    failures = []
    thresholds = {
        "registrant_ticker_recall": 1.0,
        "other_entity_precision": 0.95,
        "other_entity_recall": 0.95,
        "grounded_span_rate": 1.0,
        "semantic_outcome_accuracy": 0.90,
    }
    for key, threshold in thresholds.items():
        if float(metrics[key]) < threshold:
            failures.append(f"{key}={metrics[key]:.3f} < {threshold:.3f}")
    if metrics["identity_leakage_count"]:
        failures.append(f"identity_leakage_count={metrics['identity_leakage_count']} != 0")
    if failures:
        raise CounterfactualDataError("review gate failed: " + "; ".join(failures))

    review_by_id = {row["content_id"]: row for row in reviews}
    promoted = []
    rejected = []
    for draft in _read_jsonl(root / "draft_annotations.jsonl"):
        review = review_by_id.get(draft["content_id"])
        if review:
            if review.get("corrected_entities") is not None:
                draft["entities"] = review["corrected_entities"]
            if review.get("corrected_event_facts") is not None:
                draft["event_facts"] = review["corrected_event_facts"]
            if review.get("corrected_expected_outcome") is not None:
                draft["expected_outcome"] = review["corrected_expected_outcome"]
        if (
            draft.get("annotator_version") != ANNOTATOR_VERSION
            or draft.get("annotation_status") != "complete"
        ):
            rejected.append(
                {
                    "content_id": draft["content_id"],
                    "reason": "annotation_not_complete_or_current",
                }
            )
            continue
        if draft.get("expected_outcome") not in {"positive", "negative"}:
            rejected.append({"content_id": draft["content_id"], "reason": "ambiguous_semantic_outcome"})
            continue
        excerpt = event_excerpt(draft["analysis_text"], draft.get("event_facts", []))
        excerpt_entities = reanchor_extractions(excerpt, draft.get("entities", []))
        template, redactions = build_context_template(
            excerpt, excerpt_entities, draft["company"]
        )
        specificity_template, specificity_redactions = build_context_template(
            excerpt,
            excerpt_entities,
            draft["company"],
            redact_quasi_identifiers=False,
        )
        aliases = {draft["company"].casefold(), short_company_name(draft["company"]).casefold()}
        aliases.update(
            str(entity.get("extraction_text", "")).casefold()
            for entity in draft.get("entities", [])
            if entity.get("extraction_class")
            in {"registrant_name", "registrant_alias", "ticker"}
            and str(entity.get("extraction_text", "")).casefold()
            not in {"the company", "the registrant", "registrant"}
        )
        leakage = [alias for alias in aliases if alias and alias in template.casefold()]
        if leakage:
            rejected.append(
                {"content_id": draft["content_id"], "reason": "registrant_identity_leakage", "matches": leakage}
            )
            continue
        promoted.append(
            {
                **draft,
                "dataset_status": "validated",
                "filing_excerpt": excerpt,
                "context_template": template,
                "template_redactions": redactions,
                "specificity_template": specificity_template,
                "specificity_redactions": specificity_redactions,
                "specificity_flag": True,
                "review_gate_metrics": metrics,
            }
        )
    _write_jsonl(promoted, root / "validated_content.jsonl")
    _write_jsonl(rejected, root / "promotion_rejections.jsonl")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "promote",
        "status": "validated",
        "review_metrics": metrics,
        "promoted_count": len(promoted),
        "rejected_count": len(rejected),
    }
    _write_json(manifest, root / "promotion_manifest.json")
    return manifest


def validate_outputs(output_root: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Validate invariants without requiring a tokenizer or annotation server."""
    root = Path(output_root)
    errors: list[str] = []
    checks: dict[str, Any] = {}
    if (root / "sampled_events.jsonl").exists():
        sample = list(_read_jsonl(root / "sampled_events.jsonl"))
        checks["sample_count"] = len(sample)
        if len({row["content_id"] for row in sample}) != len(sample):
            errors.append("duplicate content_id in sample")
        if any(not eligible_event(row) for row in sample):
            errors.append("sample contains ineligible event")
    if (root / "draft_annotations.jsonl").exists():
        annotations = list(_read_jsonl(root / "draft_annotations.jsonl"))
        checks["draft_annotation_count"] = len(annotations)
        checks["draft_failed_count"] = sum(
            row.get("annotation_status") == "failed" for row in annotations
        )
        if len({row.get("content_id") for row in annotations}) != len(annotations):
            errors.append("duplicate content_id in draft annotations")
        for row in annotations:
            content_id = row.get("content_id", "<missing>")
            if row.get("annotator_version") != ANNOTATOR_VERSION:
                errors.append(f"{content_id}: obsolete or missing annotator_version")
            if row.get("dataset_status") != "draft":
                errors.append(f"{content_id}: annotation must remain draft")
            status = row.get("annotation_status")
            if status == "failed":
                if not row.get("annotation_error"):
                    errors.append(f"{content_id}: failed annotation lacks error")
                continue
            if status != "complete":
                errors.append(f"{content_id}: invalid annotation_status")
                continue
            text = str(row.get("analysis_text") or "")
            for extraction in row.get("entities", []) + row.get("event_facts", []):
                span = _span(extraction)
                if (
                    span is None
                    or text[span[0] : span[1]]
                    != extraction.get("extraction_text")
                ):
                    errors.append(f"{content_id}: non-exact grounded extraction")
                    break
    if (root / "validated_content.jsonl").exists():
        content = list(_read_jsonl(root / "validated_content.jsonl"))
        checks["validated_content_count"] = len(content)
        for row in content:
            if row.get("expected_outcome") not in {"positive", "negative"}:
                errors.append(f"{row.get('content_id')}: invalid expected_outcome")
            if row.get("context_template", "").count("{ENTITY}") != 1:
                errors.append(f"{row.get('content_id')}: template must have one entity slot")
    if (root / "pairs_unrendered.jsonl").exists():
        pairs = list(_read_jsonl(root / "pairs_unrendered.jsonl"))
        checks["directional_pair_count"] = len(pairs)
        for row in pairs:
            if row["expected_outcome"] not in {"positive", "negative"}:
                errors.append(f"{row['pair_id']}: invalid outcome")
            source_without = row["source_prompt"].replace(row["source_entity"], "{ENTITY}", 1)
            target_without = row["target_prompt"].replace(row["target_entity"], "{ENTITY}", 1)
            if source_without != target_without:
                errors.append(f"{row['pair_id']}: content differs beyond entity")
            if row["margin_definition"] != "logit(positive)-logit(negative)":
                errors.append(f"{row['pair_id']}: wrong margin")
    checks["error_count"] = len(errors)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validate",
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors[:100],
    }
    _write_json(report, root / "validation_report.json")
    if errors:
        raise CounterfactualDataError(f"validation failed with {len(errors)} errors")
    return report
