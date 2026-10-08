"""Shared, externally reviewed evidence contracts for new stance experiments.

These validators audit structure, exact text bytes, and declared provenance. They
neither acquire sources nor approve financial truth, strength, numeric claims, or
source quality. In particular, a supplied source hash is provenance, not proof of
quality; numerical/source-quality reviews must come from an external review.
Historical four-slot pools are not accepted or converted.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, fields

from llm_bias.core.artifact_paths import sha256_bytes
from llm_bias.core.population import PopulationMember, validate_members

_CONDITIONS = ("++", "+-", "-+", "--")
_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class EvidenceItem:
    """One independently identified, reviewed observation; all fields are strings.

    ``text`` is preserved verbatim and ``content_sha256`` hashes its exact UTF-8
    bytes. ``source`` identifies the external source, and ``source_sha256`` is a
    supplied digest of reviewed source bytes, never derived from text or a URL.
    ``review_id`` identifies that review. ``strength`` is a nonblank label whose
    meaning is documented by the supplied label/review, not a numeric score or a
    built-in research scale. ``numerical_review`` records the external audit of
    numbers/units/bases (or an explicit no-numbers review), and
    ``source_quality_review`` records the external source-quality assessment.
    The validator checks their presence, not whether those assessments are true.
    """

    item_id: str
    text: str
    polarity: str
    source: str
    source_sha256: str
    content_sha256: str
    review_id: str
    strength: str
    numerical_review: str
    source_quality_review: str


@dataclass(frozen=True)
class EvidencePair:
    """Two distinct item IDs for one ticker × trial × literal condition row.

    ``condition`` is ++, +-, -+, or -- and specifies the two slot polarities.
    ``validate_evidence_pair`` audits one row; ``validate_evidence_pairs`` requires
    the full declared member/trial/condition product, exact sharing across
    tickers, and consistent content/metadata for each item ID.
    """

    ticker: str
    trial_id: str
    condition: str
    evidence1: EvidenceItem
    evidence2: EvidenceItem


@dataclass(frozen=True)
class EvidenceOrderContrast:
    """Explicit within-trial slot-order contrast applied to every member.

    ``condition1`` and ``condition2`` must be distinct reversed conditions, so
    mixed +-/-+ rows can be declared as an order-only contrast. Both rows must
    contain the exact same full item records in reversed slot order. Merely
    opposite polarities, or equal text under different item IDs, do not suffice.
    Undeclared conditions/trials and duplicate/reversed declarations are errors.
    """

    trial_id: str
    condition1: str
    condition2: str


def _require_nonblank(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")


def validate_evidence_item(item: EvidenceItem) -> EvidenceItem:
    """Validate metadata and exact UTF-8 content SHA-256; return the unchanged item.

    Source bytes are not consumed here: the supplied source digest is required
    and syntax-checked, but no reviewed bytes or quality approval are invented.
    """
    if not isinstance(item, EvidenceItem):
        raise ValueError("evidence slots must contain EvidenceItem records")
    for field in fields(item):
        _require_nonblank(getattr(item, field.name), field.name)
    if item.polarity not in ("+", "-"):
        raise ValueError("polarity must be '+' or '-'")
    for name in ("source_sha256", "content_sha256"):
        if _HASH.fullmatch(getattr(item, name)) is None:
            raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    if item.content_sha256 != sha256_bytes(item.text.encode("utf-8")):
        raise ValueError("content_sha256 does not match exact UTF-8 text")
    return item


def validate_evidence_pair(pair: EvidencePair) -> EvidencePair:
    """Validate one generation row without claiming complete population coverage."""
    if not isinstance(pair, EvidencePair):
        raise ValueError("pairs must contain EvidencePair records, not legacy pools")
    for name in ("ticker", "trial_id", "condition"):
        _require_nonblank(getattr(pair, name), name)
    if pair.condition not in _CONDITIONS:
        raise ValueError(f"condition must be one of {_CONDITIONS}")
    first = validate_evidence_item(pair.evidence1)
    second = validate_evidence_item(pair.evidence2)
    if first.item_id == second.item_id:
        raise ValueError("evidence slots must have independent item IDs")
    if first.polarity + second.polarity != pair.condition:
        raise ValueError("evidence slot polarity must match condition")
    return pair


def validate_evidence_pairs(
    members: Iterable[PopulationMember],
    trial_ids: Iterable[str],
    pairs: Iterable[EvidencePair],
    *,
    order_contrasts: Iterable[EvidenceOrderContrast] = (),
) -> tuple[EvidencePair, ...]:
    """Return ticker/trial/condition-sorted, complete, shared validated pairs.

    Declared trials are nonempty unique nonblank strings. Exactly one row is
    required for every member × trial × four conditions. For each trial/condition
    the two full item records (IDs, bytes, hashes and reviews) must be identical
    across tickers. An item ID must denote the same full record everywhere.
    Order contrasts are explicit, not inferred from polarity or evidence text.
    """
    validated_members = validate_members(members)
    if isinstance(trial_ids, (str, bytes)):
        raise ValueError("trial_ids must be an iterable of trial strings, not one string")
    trials = tuple(trial_ids)
    if not trials:
        raise ValueError("trial_ids must be nonempty")
    for trial in trials:
        _require_nonblank(trial, "trial_id")
    if len(set(trials)) != len(trials):
        raise ValueError("duplicate declared trial_id")
    tickers = {member.ticker for member in validated_members}
    expected = {(ticker, trial, condition)
                for ticker in tickers for trial in trials for condition in _CONDITIONS}
    rows: dict[tuple[str, str, str], EvidencePair] = {}
    items: dict[str, EvidenceItem] = {}
    shared: dict[tuple[str, str], tuple[EvidenceItem, EvidenceItem]] = {}
    for candidate in pairs:
        pair = validate_evidence_pair(candidate)
        key = (pair.ticker, pair.trial_id, pair.condition)
        if key not in expected:
            raise ValueError(f"unknown evidence row key: {key!r}")
        if key in rows:
            raise ValueError(f"duplicate evidence row key: {key!r}")
        rows[key] = pair
        slots = (pair.evidence1, pair.evidence2)
        for item in slots:
            if item.item_id in items and items[item.item_id] != item:
                raise ValueError(f"item_id {item.item_id!r} has inconsistent content or metadata")
            items[item.item_id] = item
        shared_key = (pair.trial_id, pair.condition)
        if shared_key in shared and shared[shared_key] != slots:
            raise ValueError("evidence bytes and metadata must be shared across tickers")
        shared[shared_key] = slots
    if set(rows) != expected:
        raise ValueError(f"evidence coverage is incomplete: {len(expected - set(rows))} missing rows")

    seen_contrasts: set[tuple[str, tuple[str, str]]] = set()
    for contrast in order_contrasts:
        if not isinstance(contrast, EvidenceOrderContrast):
            raise ValueError("order contrasts must contain EvidenceOrderContrast records")
        for field in fields(contrast):
            _require_nonblank(getattr(contrast, field.name), f"order contrast {field.name}")
        a, b = contrast.condition1, contrast.condition2
        if (contrast.trial_id not in trials or a not in _CONDITIONS or b not in _CONDITIONS
                or a == b or a[::-1] != b):
            raise ValueError("order contrast must name a declared trial and distinct reversed conditions")
        contrast_key = (contrast.trial_id, tuple(sorted((a, b))))
        if contrast_key in seen_contrasts:
            raise ValueError("duplicate order contrast declaration")
        seen_contrasts.add(contrast_key)
        first = shared[(contrast.trial_id, a)]
        second = shared[(contrast.trial_id, b)]
        if first != second[::-1]:
            raise ValueError("order contrast requires swapped identical item IDs, content and metadata")
    return tuple(rows[key] for key in sorted(rows))
