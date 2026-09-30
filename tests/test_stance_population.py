"""Fixture contracts for the new population and issuer-isolated role helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from llm_bias.core import population as P

SOURCE_SHA256 = "27c454d250ce513fda2016b43e7009ccbe0f5381e6c7648d3efe0d418c501a38"
ROLE_ORDER = ("fit", "validation", "calibration", "evaluation")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "data/sp500_constituents_2020_2025.csv"


def _canonical_hash(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


@pytest.fixture
def population_rows():
    rows = [{
        "index_name": "S&P 500", "year": "2024", "ticker": f"T{i:03}",
        "company_name": f"Société {i}", "gics_sector": "Unspecified" if i == 0 else "Energy",
    } for i in range(503)]
    # These rows must not enter the selected population, even with missing fields.
    rows.extend([
        {"index_name": "S&P 500", "year": "2023"},
        {"index_name": "Other index", "year": "2024"},
    ])
    return rows


@pytest.fixture
def members():
    # Six tickers, five issuers: AA/AAB are distinct share classes of issuer-a.
    return (
        P.PopulationMember("AA", "Société A (Class A)", "Unspecified", "issuer-a"),
        P.PopulationMember("AAB", "Société A (Class B)", "Unspecified", "issuer-a"),
        P.PopulationMember("BB", "Company B", "Energy", "issuer-b"),
        P.PopulationMember("CC", "Company C", "Utilities", "issuer-c"),
        P.PopulationMember("DD", "Company D", "Industrials", "issuer-d"),
        P.PopulationMember("EE", "Company E", "Financials", "issuer-e"),
    )


@pytest.fixture
def roles():
    return {
        "fit": ["AA", "AAB", "BB"], "validation": ["CC"],
        "calibration": ["DD"], "evaluation": ["EE"],
    }


@pytest.fixture
def real_source():
    if not SOURCE_PATH.is_file():
        pytest.skip("untracked frozen population CSV is unavailable")
    return SOURCE_PATH


@pytest.fixture
def test_issuer_mapping(real_source):
    # Test-only unique IDs derived from tickers; NOT an approved issuer mapping.
    with real_source.open(newline="", encoding="utf-8") as handle:
        return {
            row["ticker"]: f"fixture-issuer-{row['ticker']}"
            for row in csv.DictReader(handle)
            if row["index_name"] == "S&P 500" and row["year"] == "2024"
        }


def test_source_digest_is_checked_before_decode_or_parse(tmp_path, monkeypatch):
    source = tmp_path / "malformed.csv"
    source.write_bytes(b"\xffnot valid UTF-8 or CSV")

    def forbidden_parse(*args, **kwargs):
        pytest.fail("CSV parsing occurred before rejecting the source digest")

    monkeypatch.setattr(P.csv, "DictReader", forbidden_parse)
    with pytest.raises(ValueError, match="SHA-256"):
        P.load_population(source, {})


def test_full_count_fixture_does_not_bypass_research_digest(population_rows, tmp_path):
    source = tmp_path / "synthetic.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "index_name", "year", "ticker", "company_name", "gics_sector",
        ))
        writer.writeheader()
        writer.writerows(population_rows[:503])
    P.validate_population_rows(population_rows)
    mapping = {row["ticker"]: row["ticker"] for row in population_rows[:503]}
    with pytest.raises(ValueError, match="SHA-256"):
        P.load_population(source, mapping)


def test_population_selection_is_sorted_deterministic_and_preserves_fields(population_rows):
    original = [dict(row) for row in population_rows]
    shuffled = list(population_rows)
    random.Random(31).shuffle(shuffled)
    selected = P.validate_population_rows(population_rows)
    assert selected == P.validate_population_rows(shuffled)
    assert isinstance(selected, tuple)
    assert len(selected) == 503
    assert [row["ticker"] for row in selected] == [f"T{i:03}" for i in range(503)]
    assert selected[0]["company_name"] == "Société 0"
    assert selected[0]["gics_sector"] == "Unspecified"
    assert population_rows == original


@pytest.mark.parametrize("count", [0, 6, 502, 504])
def test_population_rows_require_exact_full_count(population_rows, count):
    rows = population_rows[:min(count, 503)]
    if count == 504:
        rows.append({**population_rows[0], "ticker": "EXTRA"})
    with pytest.raises(ValueError, match="503"):
        P.validate_population_rows(rows)


def test_duplicate_population_ticker_is_rejected(population_rows):
    population_rows[1]["ticker"] = population_rows[0]["ticker"]
    with pytest.raises(ValueError, match="duplicate"):
        P.validate_population_rows(population_rows)


@pytest.mark.parametrize("field", ["ticker", "company_name", "gics_sector"])
@pytest.mark.parametrize("bad_value", [None, "", " \t", 42])
def test_population_rows_reject_missing_or_blank_fields(population_rows, field, bad_value):
    population_rows[0][field] = bad_value
    with pytest.raises(ValueError, match=field):
        P.validate_population_rows(population_rows)


@pytest.mark.parametrize("field", ["ticker", "company_name", "gics_sector"])
def test_population_rows_reject_absent_fields(population_rows, field):
    del population_rows[0][field]
    with pytest.raises(ValueError, match=field):
        P.validate_population_rows(population_rows)


def test_real_source_snapshot_has_exact_canonical_membership_hash(real_source, test_issuer_mapping):
    snapshot = P.load_population(real_source, test_issuer_mapping)
    assert isinstance(snapshot, P.PopulationSnapshot)
    assert isinstance(snapshot.members, tuple)
    assert len(snapshot.members) == 503
    assert snapshot.source_sha256 == SOURCE_SHA256
    assert snapshot.members == tuple(sorted(snapshot.members, key=lambda member: member.ticker))
    assert sum(member.sector == "Unspecified" for member in snapshot.members) == 29
    tickers = {member.ticker for member in snapshot.members}
    assert {"GOOG", "GOOGL", "FOX", "FOXA", "NWS", "NWSA"} <= tickers
    assert snapshot.membership_sha256 == _canonical_hash([asdict(member) for member in snapshot.members])
    assert snapshot == P.load_population(str(real_source), dict(reversed(list(test_issuer_mapping.items()))))
    changed = {**test_issuer_mapping, "GOOG": "different-fixture-issuer"}
    assert P.load_population(real_source, changed).membership_sha256 != snapshot.membership_sha256


@pytest.mark.parametrize("change", ["missing", "foreign", "substituted"])
def test_loader_requires_exact_issuer_mapping_keys(real_source, test_issuer_mapping, change):
    mapping = dict(test_issuer_mapping)
    if change in {"missing", "substituted"}:
        mapping.pop("GOOG")
    if change in {"foreign", "substituted"}:
        mapping["FOREIGN"] = "fixture-foreign"
    with pytest.raises(ValueError, match="issuer mapping"):
        P.load_population(real_source, mapping)


@pytest.mark.parametrize("bad_value", [None, "", " \t", 42])
def test_loader_rejects_missing_or_blank_issuer_ids(real_source, test_issuer_mapping, bad_value):
    test_issuer_mapping["GOOG"] = bad_value
    with pytest.raises(ValueError, match="issuer_id"):
        P.load_population(real_source, test_issuer_mapping)


def test_loader_preserves_two_tickers_with_one_supplied_issuer(real_source, test_issuer_mapping):
    test_issuer_mapping["GOOG"] = test_issuer_mapping["GOOGL"] = "fixture-shared-issuer"
    snapshot = P.load_population(real_source, test_issuer_mapping)
    shared = [member.ticker for member in snapshot.members if member.issuer_id == "fixture-shared-issuer"]
    assert shared == ["GOOG", "GOOGL"]
    assert len(snapshot.members) == 503


def test_tiny_member_validator_preserves_share_classes_and_sorts(members):
    assert P.validate_members(reversed(members)) == members
    assert P.validate_members(members[:1]) == members[:1]


@pytest.mark.parametrize("field", ["ticker", "name", "sector", "issuer_id"])
@pytest.mark.parametrize("bad_value", [None, "", " \t", 42])
def test_member_validator_rejects_missing_metadata(members, field, bad_value):
    invalid = (replace(members[0], **{field: bad_value}), *members[1:])
    with pytest.raises(ValueError, match=field):
        P.validate_members(invalid)


def test_member_validator_rejects_empty_and_duplicate_tickers(members):
    with pytest.raises(ValueError, match="nonempty"):
        P.validate_members(())
    with pytest.raises(ValueError, match="duplicate"):
        P.validate_members((*members, members[0]))


def test_validate_roles_returns_sorted_assignments_counts_and_exact_hash(members, roles):
    inputs = {role: list(reversed(tickers)) for role, tickers in reversed(list(roles.items()))}
    result = P.validate_roles(reversed(members), inputs)
    assert isinstance(result, P.RoleAssignments)
    assert result.roles == roles
    assert tuple(result.roles) == ROLE_ORDER
    assert result.assignments == {
        "AA": "fit", "AAB": "fit", "BB": "fit", "CC": "validation",
        "DD": "calibration", "EE": "evaluation",
    }
    assert result.ticker_counts == dict(zip(ROLE_ORDER, (3, 1, 1, 1)))
    assert result.issuer_counts == dict(zip(ROLE_ORDER, (2, 1, 1, 1)))
    assert result.assignment_hash == _canonical_hash({"roles": roles})
    assert result == P.validate_roles(members, roles)
    result.roles["fit"].append("FOREIGN")
    assert inputs["fit"] == ["BB", "AAB", "AA"]  # Result owns its sorted lists.


def test_assign_roles_shuffles_issuers_and_uses_fixed_slice_order(members):
    counts = dict(zip(ROLE_ORDER, (2, 1, 1, 1)))
    result = P.assign_roles(members, seed=17, issuer_counts=dict(reversed(list(counts.items()))))
    # Hand-computed from seed 17's issuer permutation: a, c, b, d, e.
    assert result.roles == {
        "fit": ["AA", "AAB", "CC"], "validation": ["BB"],
        "calibration": ["DD"], "evaluation": ["EE"],
    }
    assert result.ticker_counts == dict(zip(ROLE_ORDER, (3, 1, 1, 1)))
    assert result.issuer_counts == counts
    assert result == P.assign_roles(reversed(members), seed=17, issuer_counts=counts)
    assert result == P.validate_roles(members, result.roles)
    assert result.assignment_hash == _canonical_hash({"roles": result.roles})
    assert P.assign_roles(members, seed=2, issuer_counts=counts).assignment_hash != result.assignment_hash


@pytest.mark.parametrize("seed", [2, 7, 17, 31])
def test_assign_roles_keeps_share_classes_together_without_global_rng_effects(members, seed):
    state = random.getstate()
    result = P.assign_roles(members, seed, dict(zip(ROLE_ORDER, (1, 1, 1, 2))))
    assert random.getstate() == state
    assert result.assignments["AA"] == result.assignments["AAB"]
    assert set(result.assignments) == {member.ticker for member in members}
    assert all(result.ticker_counts.values())


@pytest.mark.parametrize("counts", [
    {"fit": -1, "validation": 2, "calibration": 2, "evaluation": 2},
    {"fit": 2.0, "validation": 1, "calibration": 1, "evaluation": 1},
    {"fit": True, "validation": 1, "calibration": 1, "evaluation": 2},
    {"fit": "2", "validation": 1, "calibration": 1, "evaluation": 1},
    {"fit": 0, "validation": 1, "calibration": 1, "evaluation": 3},
    {"fit": 1, "validation": 1, "calibration": 1, "evaluation": 1},
    {"fit": 3, "validation": 1, "calibration": 1, "evaluation": 1},
    {"fit": 2, "validation": 1, "calibration": 1},
    {"fit": 2, "validation": 1, "calibration": 1, "evaluation": 1, "other": 1},
])
def test_assign_roles_rejects_invalid_explicit_issuer_counts(members, counts):
    with pytest.raises(ValueError):
        P.assign_roles(members, 17, counts)


@pytest.mark.parametrize("change", [
    "missing_role", "extra_role", "empty_role", "missing_ticker", "foreign_ticker",
    "duplicate_within", "duplicate_across", "issuer_crossing", "not_a_list",
])
def test_validate_roles_rejects_incomplete_or_leaking_assignments(members, roles, change):
    if change == "missing_role":
        del roles["evaluation"]
    elif change == "extra_role":
        roles["other"] = ["AA"]
    elif change == "empty_role":
        roles["fit"].extend(roles["evaluation"])
        roles["evaluation"] = []
    elif change == "missing_ticker":
        roles["fit"].remove("BB")
    elif change == "foreign_ticker":
        roles["fit"].append("FOREIGN")
    elif change == "duplicate_within":
        roles["fit"].append("AA")
    elif change == "duplicate_across":
        roles["evaluation"].append("AA")
    elif change == "issuer_crossing":
        roles["fit"].remove("AAB")
        roles["evaluation"].append("AAB")
    elif change == "not_a_list":
        roles["evaluation"] = "EE"
    with pytest.raises(ValueError):
        P.validate_roles(members, roles)


@pytest.mark.parametrize("operation", ["assign", "validate"])
@pytest.mark.parametrize("invalid", ["empty", "duplicate", "missing_issuer"])
def test_role_helpers_validate_members_first(members, roles, operation, invalid):
    if invalid == "empty":
        members = ()
    elif invalid == "duplicate":
        members = (*members, members[0])
    else:
        members = (replace(members[0], issuer_id=" "), *members[1:])
    with pytest.raises(ValueError):
        if operation == "assign":
            P.assign_roles(members, 17, dict(zip(ROLE_ORDER, (2, 1, 1, 1))))
        else:
            P.validate_roles(members, roles)


@pytest.mark.parametrize("role", ROLE_ORDER)
def test_role_guards_allow_only_the_declared_consumption_role(members, roles, role):
    result = P.validate_roles(members, roles)
    assert P.require_role(roles[role], result.assignments, {role}) is None
    for other_role in set(ROLE_ORDER) - {role}:
        with pytest.raises(ValueError, match="allowed"):
            P.require_role(roles[other_role], result.assignments, {role})


@pytest.mark.parametrize("tickers", [["AA", "AA"], ["FOREIGN"], ["AA", "FOREIGN"]])
def test_role_guards_reject_duplicates_and_unknown_tickers(members, roles, tickers):
    result = P.validate_roles(members, roles)
    with pytest.raises(ValueError):
        P.require_role(tickers, result.assignments, {"fit"})


def test_role_guards_allow_explicit_role_union_and_reject_unknown_roles(members, roles):
    result = P.validate_roles(members, roles)
    P.require_role(["AA", "CC"], result.assignments, {"fit", "validation"})
    with pytest.raises(ValueError, match="role"):
        P.require_role(["AA"], result.assignments, {"unknown"})
