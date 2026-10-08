"""Frozen-source population and issuer-isolated roles for new stance experiments.

The fixture validators do not authorize alternative research populations. Issuer
IDs must be supplied explicitly; neither company names nor tickers imply them.
Historical population and split helpers remain independent of this module.
"""
from __future__ import annotations

import csv
import io
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_bias.core.artifact_paths import sha256_bytes, sha256_json

_SOURCE_SHA256 = "27c454d250ce513fda2016b43e7009ccbe0f5381e6c7648d3efe0d418c501a38"
_EXPECTED_COUNT = 503
_ROLE_ORDER = ("fit", "validation", "calibration", "evaluation")


@dataclass(frozen=True)
class PopulationMember:
    """One retained ticker/share class with externally supplied issuer metadata."""

    ticker: str
    name: str
    sector: str
    issuer_id: str


@dataclass(frozen=True)
class PopulationSnapshot:
    """Ticker-sorted membership and canonical source/membership provenance."""

    members: tuple[PopulationMember, ...]
    source_sha256: str
    membership_sha256: str


@dataclass(frozen=True)
class RoleAssignments:
    """Validated issuer-disjoint roles, their counts and canonical identity."""

    roles: dict[str, list[str]]
    assignments: dict[str, str]
    ticker_counts: dict[str, int]
    issuer_counts: dict[str, int]
    assignment_hash: str


def _require_nonblank(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")


def validate_population_rows(rows: Iterable[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    """Validate and sort exactly 503 selected 2024 S&P 500 CSV rows.

    This pure helper returns copies of the selected rows and preserves their
    fields verbatim, including Unspecified sectors. It has no count override or
    source-digest option and is not a research loader.
    """
    selected = [dict(row) for row in rows
                if row.get("year") == "2024" and row.get("index_name") == "S&P 500"]
    if len(selected) != _EXPECTED_COUNT:
        raise ValueError(f"expected {_EXPECTED_COUNT} population rows, got {len(selected)}")
    seen: set[str] = set()
    for row in selected:
        for field in ("ticker", "company_name", "gics_sector"):
            _require_nonblank(row.get(field), field)
        ticker = row["ticker"]
        if ticker in seen:
            raise ValueError(f"duplicate population ticker: {ticker!r}")
        seen.add(ticker)
    return tuple(sorted(selected, key=lambda row: row["ticker"]))


def validate_members(members: Iterable[PopulationMember]) -> tuple[PopulationMember, ...]:
    """Validate nonempty unique members, allowing tiny non-research fixtures."""
    validated = tuple(members)
    if not validated:
        raise ValueError("members must be nonempty")
    seen: set[str] = set()
    for member in validated:
        if not isinstance(member, PopulationMember):
            raise ValueError("members must contain PopulationMember records")
        for field in ("ticker", "name", "sector", "issuer_id"):
            _require_nonblank(getattr(member, field), field)
        if member.ticker in seen:
            raise ValueError(f"duplicate member ticker: {member.ticker!r}")
        seen.add(member.ticker)
    return tuple(sorted(validated, key=lambda member: member.ticker))


def load_population(
    csv_path: str | Path, issuer_by_ticker: Mapping[str, str],
) -> PopulationSnapshot:
    """Load only the exact frozen CSV bytes with a complete issuer mapping.

    The digest is checked before decoding or parsing. Mapping keys must equal
    the selected tickers exactly; multiple tickers may share one issuer ID.
    """
    payload = Path(csv_path).read_bytes()
    source_sha256 = sha256_bytes(payload)
    if source_sha256 != _SOURCE_SHA256:
        raise ValueError(f"population source SHA-256 differs from the frozen digest: {source_sha256}")
    rows = validate_population_rows(csv.DictReader(io.StringIO(payload.decode("utf-8"), newline="")))
    tickers = {row["ticker"] for row in rows}
    if set(issuer_by_ticker) != tickers:
        raise ValueError("issuer mapping keys must exactly equal population membership")
    members = validate_members(PopulationMember(
        ticker=row["ticker"], name=row["company_name"], sector=row["gics_sector"],
        issuer_id=issuer_by_ticker[row["ticker"]],
    ) for row in rows)
    return PopulationSnapshot(
        members=members, source_sha256=source_sha256,
        membership_sha256=sha256_json([asdict(member) for member in members]),
    )


def validate_roles(
    members: Iterable[PopulationMember], roles: Mapping[str, list[str]],
) -> RoleAssignments:
    """Validate exact coverage by four nonempty roles, with no issuer crossing."""
    validated = validate_members(members)
    if set(roles) != set(_ROLE_ORDER):
        raise ValueError(f"roles must be exactly {_ROLE_ORDER}")
    issuer_by_ticker = {member.ticker: member.issuer_id for member in validated}
    assignments: dict[str, str] = {}
    issuer_roles: dict[str, str] = {}
    sorted_roles: dict[str, list[str]] = {}
    for role in _ROLE_ORDER:
        tickers = roles[role]
        if not isinstance(tickers, list) or not tickers:
            raise ValueError(f"role {role!r} must be a nonempty ticker list")
        for ticker in tickers:
            _require_nonblank(ticker, "ticker")
            if ticker not in issuer_by_ticker:
                raise ValueError(f"unknown role ticker: {ticker!r}")
            if ticker in assignments:
                raise ValueError(f"duplicate role ticker: {ticker!r}")
            issuer = issuer_by_ticker[ticker]
            if issuer in issuer_roles and issuer_roles[issuer] != role:
                raise ValueError(f"issuer {issuer!r} crosses roles")
            assignments[ticker] = role
            issuer_roles[issuer] = role
        sorted_roles[role] = sorted(tickers)
    if set(assignments) != set(issuer_by_ticker):
        raise ValueError("roles must cover every population ticker")
    return RoleAssignments(
        roles=sorted_roles,
        assignments={member.ticker: assignments[member.ticker] for member in validated},
        ticker_counts={role: len(sorted_roles[role]) for role in _ROLE_ORDER},
        issuer_counts={role: sum(assigned == role for assigned in issuer_roles.values())
                       for role in _ROLE_ORDER},
        assignment_hash=sha256_json({"roles": sorted_roles}),
    )


def assign_roles(
    members: Iterable[PopulationMember], seed: int, issuer_counts: Mapping[str, int],
) -> RoleAssignments:
    """Shuffle sorted issuers with a local RNG and slice in fixed role order.

    Counts are issuer counts, not ticker counts. Every role must receive at
    least one issuer, and every share class follows its issuer into that role.
    """
    validated = validate_members(members)
    if set(issuer_counts) != set(_ROLE_ORDER):
        raise ValueError(f"issuer counts must specify exactly {_ROLE_ORDER}")
    for role in _ROLE_ORDER:
        count = issuer_counts[role]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"issuer count for {role!r} must be a positive integer")
    issuers = sorted({member.issuer_id for member in validated})
    if sum(issuer_counts.values()) != len(issuers):
        raise ValueError("issuer counts must sum to the population issuer count")
    random.Random(seed).shuffle(issuers)
    issuer_roles: dict[str, str] = {}
    offset = 0
    for role in _ROLE_ORDER:
        end = offset + issuer_counts[role]
        issuer_roles.update((issuer, role) for issuer in issuers[offset:end])
        offset = end
    roles: dict[str, list[str]] = {role: [] for role in _ROLE_ORDER}
    for member in validated:
        roles[issuer_roles[member.issuer_id]].append(member.ticker)
    return validate_roles(validated, roles)


def require_role(
    tickers: Iterable[str], assignments: Mapping[str, str], allowed_roles: Iterable[str],
) -> None:
    """Guard explicitly permitted inputs against foreign, duplicate or leaked tickers.

    Callers must allow only fit for fitting, validation for selection,
    calibration for dose calibration, and evaluation for final outcomes.
    """
    allowed = set(allowed_roles)
    if not allowed or not allowed <= set(_ROLE_ORDER):
        raise ValueError("allowed roles must be nonempty known roles")
    seen: set[str] = set()
    for ticker in tickers:
        _require_nonblank(ticker, "ticker")
        if ticker in seen:
            raise ValueError(f"duplicate requested ticker: {ticker!r}")
        seen.add(ticker)
        if ticker not in assignments:
            raise ValueError(f"unknown requested ticker: {ticker!r}")
        if assignments[ticker] not in allowed:
            raise ValueError(f"ticker {ticker!r} is not in an allowed role")
