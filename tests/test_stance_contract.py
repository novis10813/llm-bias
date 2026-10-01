"""Frozen execution accounting with synthetic identities and no model loading."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from decimal import Decimal

import pytest

from llm_bias.core import experiment_contract as C


IDENTITY_FIELDS = (
    "protocol_sha256", "population_sha256", "issuer_sha256", "roles_sha256",
    "evidence_sha256", "schema_sha256", "template_sha256", "model_sha256",
    "code_sha256", "backend_sha256", "generation_policy_sha256",
    "operator_sha256", "parent_sha256",
)
KEY_FIELDS = ("stage", "ticker", "condition", "trial_id", "arm", "dose")
GATES = ("schema_supported", "no_op")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@pytest.fixture
def identity():
    return C.PlanIdentity(**{name: hashlib.sha256(name.encode()).hexdigest()
                             for name in IDENTITY_FIELDS})


@pytest.fixture
def keys():
    return tuple(C.RowKey("evaluation", ticker, condition, "trial-1", arm, dose)
                 for ticker in ("AAA", "BÉB") for condition in ("+-", "-+")
                 for arm, dose in (("baseline", "0"), ("steering", "0.5")))


@pytest.fixture
def plan(identity, keys):
    return C.ExperimentPlan(identity, keys, GATES)


def _outcome(key, **changes):
    record = dict(ticker=key.ticker, issuer_id=f"fixture-{key.ticker}",
                  condition=key.condition, trial_id=key.trial_id, decision="buy",
                  decision_complete=True, schema_complete=True, reason_valid=True,
                  finish_reason="eos", failure_type=None)
    return C.GenerationOutcome(**(record | changes))


def _rows(plan):
    return tuple(C.ExecutionRow(key, _outcome(key)) for key in plan.keys)


def _shards(plan, num_shards=3, *, gates=None, rows=None):
    rows = _rows(plan) if rows is None else rows
    gates = {name: True for name in plan.gate_names} if gates is None else gates
    return tuple(C.ExecutionShard(
        plan.plan_hash, i, num_shards,
        tuple(row for row in rows if C.shard_index_for_key(row.key, num_shards) == i), gates,
    ) for i in range(num_shards))


def test_exact_named_contract_fields():
    assert tuple(field.name for field in fields(C.PlanIdentity)) == IDENTITY_FIELDS
    assert tuple(field.name for field in fields(C.RowKey)) == KEY_FIELDS
    assert tuple(field.name for field in fields(C.GenerationOutcome)) == (
        "ticker", "issuer_id", "condition", "trial_id", "decision", "decision_complete",
        "schema_complete", "reason_valid", "finish_reason", "failure_type",
    )
    assert tuple(field.name for field in fields(C.ExecutionRow)) == ("key", "outcome")
    assert tuple(field.name for field in fields(C.ExecutionShard)) == (
        "plan_hash", "shard_index", "num_shards", "rows", "gates",
    )


@pytest.mark.parametrize(("value", "expected"), [
    (0, "0"), (1, "1"), (-1, "-1"), (0.5, "0.5"),
    (" 01.5000 ", "1.5"), ("1e-7", "0.0000001"), ("1E+3", "1000"),
    (Decimal("12.3400"), "12.34"), (Decimal("0.0000"), "0"),
    (Decimal("123456789012345678901234567890.123400"),
     "123456789012345678901234567890.1234"),
])
def test_canonical_dose_normalization(value, expected):
    assert C.canonical_dose_str(value) == expected


@pytest.mark.parametrize("value", [
    True, False, None, [], {}, "", "garbage", float("nan"), float("inf"),
    float("-inf"), Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"),
    "NaN", "Infinity", "-0", "-0.000", -0.0, Decimal("-0E+5"),
])
def test_canonical_dose_rejects_invalid_and_negative_zero(value):
    with pytest.raises(ValueError):
        C.canonical_dose_str(value)


@pytest.mark.parametrize("field", KEY_FIELDS)
@pytest.mark.parametrize("bad", [None, "", " \t", 1, True])
def test_row_keys_require_nonblank_strings(keys, field, bad):
    with pytest.raises(ValueError):
        replace(keys[0], **{field: bad})


@pytest.mark.parametrize("dose", ["0.0", "0.50", "01", "1e-3", "+1", " 1 ", "-0"])
def test_row_key_dose_must_already_be_canonical(keys, dose):
    with pytest.raises(ValueError, match="dose"):
        replace(keys[0], dose=dose)


@pytest.mark.parametrize("condition", ["positive", "+", "++ ", "--+"])
def test_only_literal_conditions(keys, condition):
    with pytest.raises(ValueError, match="condition"):
        replace(keys[0], condition=condition)


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
@pytest.mark.parametrize("bad", [None, "", "a" * 63, "A" * 64, "g" * 64, 1])
def test_every_identity_hash_is_strict(identity, field, bad):
    with pytest.raises(ValueError, match=field):
        replace(identity, **{field: bad})


@pytest.mark.parametrize("field", IDENTITY_FIELDS[:-2])
def test_not_applicable_is_restricted_to_operator_parent(identity, field):
    with pytest.raises(ValueError, match=field):
        replace(identity, **{field: "not_applicable"})


@pytest.mark.parametrize("field", IDENTITY_FIELDS[-2:])
def test_not_applicable_is_baseline_only(identity, keys, field):
    baseline_identity = replace(identity, **{field: "not_applicable"})
    C.ExperimentPlan(baseline_identity, [key for key in keys if key.arm == "baseline"], GATES)
    with pytest.raises(ValueError, match="baseline"):
        C.ExperimentPlan(baseline_identity, keys, GATES)


def test_plan_hash_exact_payload_and_roundtrip(plan):
    expected = {"identity": plan.identity.to_dict(),
                "keys": [key.to_dict() for key in sorted(plan.keys)],
                "gate_names": sorted(GATES)}
    assert plan.plan_hash == _hash(expected)
    assert plan.to_dict() == expected | {"plan_hash": _hash(expected)}
    assert C.ExperimentPlan.from_json(plan.to_json()).to_json() == plan.to_json()
    reordered = C.ExperimentPlan(plan.identity, reversed(plan.keys), reversed(GATES))
    assert reordered.to_json() == plan.to_json()
    assert C.ExperimentPlan(plan.identity, plan.keys, [*GATES, "cache"]).plan_hash != plan.plan_hash


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_each_identity_hash_changes_plan_and_refuses_resume(plan, field):
    changed = C.ExperimentPlan(replace(plan.identity, **{field: "f" * 64}), plan.keys, GATES)
    assert changed.plan_hash != plan.plan_hash
    with pytest.raises(ValueError, match="resume"):
        C.validate_resume(plan, changed)


def test_resume_checks_keys_gates_and_is_order_independent(plan):
    C.validate_resume(plan, C.ExperimentPlan(plan.identity, reversed(plan.keys), reversed(GATES)))
    for changed in (C.ExperimentPlan(plan.identity, plan.keys[:-1], GATES),
                    C.ExperimentPlan(plan.identity, plan.keys, ["different_gate"])):
        with pytest.raises(ValueError, match="resume"):
            C.validate_resume(plan, changed)


@pytest.mark.parametrize(("field", "value"), tuple(zip(
    KEY_FIELDS, ("localization", "CCC", "++", "trial-2", "control", "-1"),
)))
def test_resume_checks_every_key_dimension(plan, field, value):
    keys = [replace(plan.keys[0], **{field: value}), *plan.keys[1:]]
    current = C.ExperimentPlan(plan.identity, keys, GATES)
    assert current.plan_hash != plan.plan_hash
    with pytest.raises(ValueError, match="resume"):
        C.validate_resume(plan, current)


def test_plan_defensively_copies_and_freezes_records(identity, keys):
    mutable_keys, gates = list(keys), list(GATES)
    plan = C.ExperimentPlan(identity, mutable_keys, gates)
    before = plan.to_json()
    assert plan.identity is not identity
    assert all(planned is not supplied for planned in plan.keys for supplied in keys)
    mutable_keys.clear()
    gates.append("forged_gate")
    exported = plan.to_dict()
    exported["identity"]["code_sha256"] = "f" * 64
    exported["keys"][0]["ticker"] = "forged"
    exported["gate_names"].append("forged")
    assert plan.to_json() == before
    for record, field, value in ((plan, "keys", ()), (plan.identity, "code_sha256", "f" * 64),
                                 (plan.keys[0], "ticker", "forged")):
        assert not hasattr(record, "__dict__")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(record, field, value)
    # Even bypassing frozen setters on the originally supplied records cannot alter the plan.
    object.__setattr__(identity, "code_sha256", "f" * 64)
    object.__setattr__(keys[0], "ticker", "forged")
    assert plan.to_json() == before
    with pytest.raises(ValueError):
        C.ExperimentPlan.from_dict(exported)


@pytest.mark.parametrize("bad_keys", [[], ["not a key"], "not key records"])
def test_plan_requires_nonempty_typed_keys(identity, bad_keys):
    with pytest.raises(ValueError):
        C.ExperimentPlan(identity, bad_keys, GATES)


def test_plan_rejects_duplicate_keys(identity, keys):
    with pytest.raises(ValueError, match="duplicate"):
        C.ExperimentPlan(identity, [*keys, keys[0]], GATES)


@pytest.mark.parametrize("gates", [[], ["same", "same"], [""], [" \t"], [True], "one_gate"])
def test_plan_requires_nonempty_unique_named_gates(identity, keys, gates):
    with pytest.raises(ValueError):
        C.ExperimentPlan(identity, keys, gates)


def test_full_key_sha256_sharding_exact_disjoint_union(plan):
    subsets = [plan.keys_for_shard(i, 5) for i in range(5)]
    assert set().union(*map(set, subsets)) == set(plan.keys)
    assert sum(map(len, subsets)) == len(plan.keys)
    for key in plan.keys:
        expected = int(_hash(key.to_dict()), 16) % 5
        assert C.shard_index_for_key(key, 5) == expected
        assert key in subsets[expected]
    # Every dimension is hashed, not merely ticker or a truncated digest.
    sample = plan.keys[0]
    for field, value in zip(KEY_FIELDS, ("localization", "CCC", "++", "trial-2", "control", "-1")):
        key = replace(sample, **{field: value})
        assert C.shard_index_for_key(key, 104729) == int(_hash(key.to_dict()), 16) % 104729


def test_sharding_is_independent_of_process_hash_seed(plan):
    script = ("import json; from llm_bias.core.experiment_contract import RowKey, shard_index_for_key; "
              f"k=RowKey.from_json({plan.keys[0].to_json()!r}); print(shard_index_for_key(k, 104729))")
    values = [subprocess.check_output([sys.executable, "-c", script], text=True,
                                      env=os.environ | {"PYTHONHASHSEED": str(seed)}).strip()
              for seed in (1, 19)]
    assert values == [str(C.shard_index_for_key(plan.keys[0], 104729))] * 2


@pytest.mark.parametrize("num_shards", [0, -1, True, 1.0, "2"])
def test_shard_count_requires_positive_integer(plan, num_shards):
    with pytest.raises(ValueError):
        C.shard_index_for_key(plan.keys[0], num_shards)
    with pytest.raises(ValueError):
        plan.keys_for_shard(0, num_shards)


@pytest.mark.parametrize("index", [-1, 3, True, 0.0, "0"])
def test_shard_index_requires_integer_in_range(plan, index):
    with pytest.raises(ValueError):
        plan.keys_for_shard(index, 3)


def test_primary_valid_outcome_and_tensor_free_row_roundtrip(plan):
    row = _rows(plan)[0]
    assert row.outcome.primary_valid
    assert C.validate_outcome(row.outcome) == row.outcome
    assert C.ExecutionRow.from_json(row.to_json()) == row
    assert set(row.to_dict()["outcome"]) == {field.name for field in fields(C.GenerationOutcome)}


@pytest.mark.parametrize("finish", ["eos", "schema_complete"])
def test_primary_valid_legal_finishes(plan, finish):
    assert _outcome(plan.keys[0], finish_reason=finish, decision="sell").primary_valid


FAILURES = (
    ("truncated", "token_budget", False, False),
    ("timeout", "timeout", False, False),
    ("exception", "exception", False, False),
    ("no_legal_token", "no_legal_token", False, False),
    ("invalid_json", "eos", False, False),
    ("invalid_schema", "schema_complete", False, False),
    ("invalid_reason", "schema_complete", True, False),
    ("unsupported_channel", "unsupported", False, False),
)


@pytest.mark.parametrize(("failure", "finish", "schema", "reason"), FAILURES)
@pytest.mark.parametrize("decision_complete", [False, True])
def test_failures_clear_primary_decision_and_may_retain_diagnostic_completion(
    plan, failure, finish, schema, reason, decision_complete,
):
    if schema and not decision_complete:
        with pytest.raises(ValueError):
            _outcome(plan.keys[0], decision=None, failure_type=failure, finish_reason=finish,
                     decision_complete=decision_complete, schema_complete=schema, reason_valid=reason)
        return
    outcome = _outcome(plan.keys[0], decision=None, failure_type=failure, finish_reason=finish,
                       decision_complete=decision_complete, schema_complete=schema, reason_valid=reason)
    assert not outcome.primary_valid
    assert outcome.decision_complete is decision_complete
    assert C.GenerationOutcome.from_json(outcome.to_json()) == outcome
    with pytest.raises(ValueError, match="decision"):
        replace(outcome, decision="buy")


@pytest.mark.parametrize("field", ["decision_complete", "schema_complete", "reason_valid"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_outcome_boolean_fields_are_strict(plan, field, value):
    with pytest.raises(ValueError, match=field):
        _outcome(plan.keys[0], **{field: value})


@pytest.mark.parametrize("field", ["ticker", "issuer_id", "trial_id", "condition"])
@pytest.mark.parametrize("value", [None, "", " \t", 1])
def test_outcome_identity_fields_are_nonblank_strings(plan, field, value):
    with pytest.raises(ValueError):
        _outcome(plan.keys[0], **{field: value})


@pytest.mark.parametrize("changes", [
    {"decision": "hold"}, {"decision": 1}, {"decision": None},
    {"decision_complete": False}, {"schema_complete": False}, {"reason_valid": False},
    {"finish_reason": "token_budget"}, {"finish_reason": "timeout"},
    {"finish_reason": "stopped"}, {"failure_type": "parse_error"}, {"condition": "mixed"},
    {"failure_type": "exception", "finish_reason": "exception"},
    {"decision": None, "failure_type": "truncated", "finish_reason": "token_budget"},
    {"decision": None, "failure_type": "invalid_json", "finish_reason": "eos"},
    {"decision": None, "failure_type": "invalid_schema", "finish_reason": "eos"},
    {"decision": None, "failure_type": "invalid_reason", "finish_reason": "eos"},
    {"decision": None, "decision_complete": False, "schema_complete": True,
     "reason_valid": False, "failure_type": "invalid_reason"},
    {"decision": None, "decision_complete": True, "schema_complete": False,
     "reason_valid": True, "failure_type": "invalid_schema"},
])
def test_outcome_rejects_contradictory_statuses(plan, changes):
    with pytest.raises(ValueError):
        _outcome(plan.keys[0], **changes)


@pytest.mark.parametrize(("failure", "finish", "schema", "reason"), FAILURES)
def test_failure_finish_pairings_are_enforced(plan, failure, finish, schema, reason):
    bad_finish = "timeout" if finish != "timeout" else "exception"
    with pytest.raises(ValueError):
        _outcome(plan.keys[0], decision=None, failure_type=failure, finish_reason=bad_finish,
                 schema_complete=schema, reason_valid=reason)


@pytest.mark.parametrize("field", ["ticker", "condition", "trial_id"])
def test_execution_row_requires_matching_outcome_identity(plan, field):
    row = _rows(plan)[0]
    value = "--" if field == "condition" else "other"
    with pytest.raises(ValueError, match="match"):
        C.ExecutionRow(row.key, replace(row.outcome, **{field: value}))


def test_execution_row_copies_input_records(plan):
    key = plan.keys[0]
    outcome = _outcome(key)
    row = C.ExecutionRow(key, outcome)
    original = row.to_json()
    object.__setattr__(outcome, "decision", "hold")
    object.__setattr__(key, "ticker", "forged")
    assert row.to_json() == original
    with pytest.raises(ValueError):
        C.ExecutionRow(key, outcome)


def test_validate_rows_rejects_duplicates_foreign_and_requires_complete(plan):
    rows = _rows(plan)
    assert C.validate_rows(plan, reversed(rows), require_complete=True) == rows
    assert C.validate_rows(plan, rows[:1]) == rows[:1]
    with pytest.raises(ValueError, match="duplicate"):
        C.validate_rows(plan, [rows[0], rows[0]])
    key = replace(rows[0].key, stage="foreign-stage")
    with pytest.raises(ValueError, match="foreign|unexpected"):
        C.validate_rows(plan, [C.ExecutionRow(key, _outcome(key))])
    with pytest.raises(ValueError, match="missing|incomplete"):
        C.validate_rows(plan, rows[:-1], require_complete=True)


def test_progress_distinguishes_absent_rows_and_executed_failures(plan):
    gates = {name: True for name in GATES}
    empty = C.progress(plan, [], gates)
    assert empty == dict(planned=8, executed=0, missing=8, complete=False, eligible=False,
                         gates=dict(sorted(gates.items())))
    failure = _outcome(plan.keys[0], decision=None, decision_complete=True,
                       schema_complete=False, reason_valid=False,
                       finish_reason="token_budget", failure_type="truncated")
    rows = [C.ExecutionRow(plan.keys[0], failure), *_rows(plan)[1:]]
    result = C.progress(plan, rows, gates)
    # An executed failure contributes to coverage; gates, not valid-decision counts, determine eligibility.
    assert result == empty | dict(executed=8, missing=0, complete=True, eligible=True)
    assert set(result) == {"planned", "executed", "missing", "complete", "eligible", "gates"}
    partial = C.progress(plan, rows[:-1], gates)
    assert (partial["executed"], partial["missing"], partial["complete"], partial["eligible"]) == (
        7, 1, False, False,
    )


@pytest.mark.parametrize(("failure", "finish", "schema", "reason"), FAILURES)
def test_every_executed_failure_can_complete_full_merge(plan, failure, finish, schema, reason):
    rows = list(_rows(plan))
    rows[0] = C.ExecutionRow(plan.keys[0], _outcome(
        plan.keys[0], decision=None, decision_complete=True, schema_complete=schema,
        reason_valid=reason, failure_type=failure, finish_reason=finish,
    ))
    merged, state = C.merge_shards(plan, _shards(plan, rows=rows))
    assert len(merged) == state["planned"] == state["executed"] == 8
    assert state["missing"] == 0 and state["complete"] and state["eligible"]
    assert merged[0].outcome.decision is None and not merged[0].outcome.primary_valid
    assert merged[0].outcome.failure_type == failure


def test_progress_requires_every_gate_for_eligibility(plan):
    rows = _rows(plan)
    for gates in (None, {}, {"no_op": True}, {"no_op": True, "schema_supported": False}):
        state = C.progress(plan, rows, gates)
        assert state["complete"] and not state["eligible"]
        assert set(state["gates"]) == set(GATES)
    assert C.progress(plan, rows)["gates"] == {name: None for name in sorted(GATES)}
    for gates in ({"unknown": True}, {"no_op": 1}, {"no_op": "passed"}, {"no_op": None}):
        with pytest.raises(ValueError):
            C.progress(plan, rows, gates)


@pytest.mark.parametrize(("rows_complete", "gates_pass", "field", "claimed_value"), [
    (False, True, "complete", True), (False, True, "eligible", True),
    (True, False, "eligible", True), (True, True, "complete", False),
    (True, True, "eligible", False), (True, True, "complete", 1),
    (True, True, "eligible", "true"), (True, True, "executed", 7),
])
def test_forged_progress_flags_counts_and_types_rejected(
    plan, rows_complete, gates_pass, field, claimed_value,
):
    rows = _rows(plan) if rows_complete else _rows(plan)[:-1]
    gates = {name: gates_pass for name in GATES}
    claim = C.progress(plan, rows, gates) | {field: claimed_value}
    with pytest.raises(ValueError):
        C.load_progress(plan, rows, gates, _json(claim))


def test_progress_json_import_is_strict_and_recomputed(plan):
    rows, gates = _rows(plan), {name: True for name in GATES}
    state = C.progress(plan, rows, gates)
    assert C.load_progress(plan, rows, gates, _json(state)) == state
    for record in (state | {"efficacy": 1}, state | {"gates": {"no_op": True}},
                   state | {"planned": True}, state | {"gates": {name: 1 for name in GATES}}):
        with pytest.raises(ValueError):
            C.load_progress(plan, rows, gates, _json(record))


def test_shard_union_merge_and_roundtrip(plan):
    shards = _shards(plan, 3)
    merged_rows, state = C.merge_shards(plan, reversed(shards))
    assert merged_rows == _rows(plan)
    assert state == C.progress(plan, merged_rows, {name: True for name in GATES})
    assert state["complete"] and state["eligible"]
    for shard in shards:
        assert C.validate_shard(plan, shard, num_shards=3) == shard
        loaded = C.ExecutionShard.from_json(shard.to_json(), plan=plan, num_shards=3)
        assert loaded.to_json() == shard.to_json()


def test_shard_defensive_row_gate_copies_and_immutable_gates(plan):
    rows, gates = list(_rows(plan)), {name: True for name in GATES}
    shard = C.ExecutionShard(plan.plan_hash, 0, 1, rows, gates)
    original = shard.to_json()
    rows.clear()
    gates["no_op"] = False
    exported = shard.to_dict()
    exported["gates"]["no_op"] = False
    exported["rows"].clear()
    assert shard.to_json() == original
    with pytest.raises(TypeError):
        shard.gates["no_op"] = False
    assert C.ExecutionShard.from_dict(exported, plan=plan).gates["no_op"] is False
    assert C.ExecutionShard.from_dict(exported, plan=plan).rows == ()


def test_shard_enforces_expected_plan_hash_count_and_keys(plan):
    shard = _shards(plan, 1)[0]
    with pytest.raises(ValueError, match="hash|identity"):
        C.validate_shard(plan, replace(shard, plan_hash="f" * 64))
    with pytest.raises(ValueError, match="count|num_shards"):
        C.validate_shard(plan, shard, num_shards=2)
    foreign_key = replace(plan.keys[0], trial_id="not-planned")
    foreign = C.ExecutionRow(foreign_key, _outcome(foreign_key))
    with pytest.raises(ValueError, match="foreign|unexpected"):
        C.validate_shard(plan, C.ExecutionShard(plan.plan_hash, 0, 1, (foreign,), shard.gates))


def test_wrong_shard_assignment_rejected(plan):
    row = _rows(plan)[0]
    right_index = C.shard_index_for_key(row.key, 3)
    with pytest.raises(ValueError, match="shard"):
        C.ExecutionShard(plan.plan_hash, (right_index + 1) % 3, 3,
                         (row,), {name: True for name in GATES})


@pytest.mark.parametrize("gates", [{"no_op": True}, {"no_op": True, "schema_supported": True, "extra": True}])
def test_shard_gate_names_exactly_match_plan(plan, gates):
    with pytest.raises(ValueError, match="gate"):
        C.validate_shard(plan, C.ExecutionShard(plan.plan_hash, 0, 1, (), gates))


@pytest.mark.parametrize("gates", [{"no_op": 1}, {"no_op": None}, {"no_op": "true"}, {"": True}])
def test_shard_gate_statuses_are_strict(plan, gates):
    with pytest.raises(ValueError, match="gate"):
        C.ExecutionShard(plan.plan_hash, 0, 1, (), gates)


@pytest.mark.parametrize(("index", "count"), [(-1, 1), (1, 1), (True, 1), (0, True), (0, 0), (0, 1.0)])
def test_shard_constructor_validates_index_and_count(plan, index, count):
    with pytest.raises(ValueError):
        C.ExecutionShard(plan.plan_hash, index, count, (), {name: True for name in GATES})


def test_duplicate_and_overlapping_shards_are_rejected_even_when_empty(plan):
    shards = _shards(plan, 20)
    empty = next(shard for shard in shards if not shard.rows)
    with pytest.raises(ValueError, match="duplicate|overlap"):
        C.merge_shards(plan, [*shards, empty])
    populated = next(shard for shard in shards if shard.rows)
    with pytest.raises(ValueError, match="duplicate|overlap"):
        C.merge_shards(plan, [*shards, populated])
    with pytest.raises(ValueError, match="duplicate|overlap"):
        C.ExecutionShard(plan.plan_hash, populated.shard_index, 20,
                         (*populated.rows, populated.rows[0]), populated.gates)


def test_missing_empty_shard_cannot_claim_complete_even_with_full_row_coverage(plan):
    shards = _shards(plan, 20)
    empty = next(shard for shard in shards if not shard.rows)
    partial = [shard for shard in shards if shard.shard_index != empty.shard_index]
    with pytest.raises(ValueError, match="shard|incomplete"):
        C.merge_shards(plan, partial)
    rows, state = C.merge_shards(plan, partial, require_complete=False)
    assert len(rows) == 8 and state["missing"] == 0
    assert not state["complete"] and not state["eligible"]
    assert state["gates"] == {name: None for name in sorted(GATES)}


def test_all_shards_with_missing_rows_are_progress_only(plan):
    shards = _shards(plan, rows=_rows(plan)[:-1])
    with pytest.raises(ValueError, match="missing|incomplete"):
        C.merge_shards(plan, shards)
    rows, state = C.merge_shards(plan, shards, require_complete=False)
    assert len(rows) == 7 and state["missing"] == 1
    assert not state["complete"] and not state["eligible"]


def test_failed_gate_is_preserved_in_full_merge(plan):
    shards = list(_shards(plan))
    shards[0] = replace(shards[0], gates={"no_op": False, "schema_supported": True})
    rows, state = C.merge_shards(plan, shards)
    assert len(rows) == 8 and state["complete"] and not state["eligible"]
    assert state["gates"] == {"no_op": False, "schema_supported": True}


def test_merge_rejects_mixed_counts_and_identities_before_summary(plan, monkeypatch):
    def forbidden_progress(*args, **kwargs):
        pytest.fail("summary ran before validating incompatible shards")

    monkeypatch.setattr(C, "progress", forbidden_progress)
    with pytest.raises(ValueError):
        C.merge_shards(plan, [_shards(plan, 1)[0], _shards(plan, 2)[0]])
    with pytest.raises(ValueError):
        C.merge_shards(plan, [replace(_shards(plan, 1)[0], plan_hash="f" * 64)])


def test_missing_shard_cannot_mask_a_known_failed_gate(plan):
    shards = list(_shards(plan, 20))
    empty = next(shard for shard in shards if not shard.rows)
    retained = [shard for shard in shards if shard.shard_index != empty.shard_index]
    retained[0] = replace(retained[0], gates={"no_op": False, "schema_supported": True})
    rows, state = C.merge_shards(plan, retained, require_complete=False)
    assert len(rows) == 8 and not state["complete"] and not state["eligible"]
    assert state["gates"] == {"no_op": False, "schema_supported": None}


def test_empty_merge_progress_and_full_refusal(plan):
    with pytest.raises(ValueError):
        C.merge_shards(plan, [])
    rows, state = C.merge_shards(plan, [], require_complete=False)
    assert rows == () and state == C.progress(plan, [])


@pytest.mark.parametrize("category", [
    "activations", "raw_activations", "activation", "residuals", "raw_residuals", "residual",
    "gradients", "raw_gradients", "gradient", "KV", "kv_cache", "past_key_values",
    "raw_tensors", "raw_tensor", "tensor", "tensors", "hidden_states", "logits",
    "margin", "buy_sell_margin", "scores", "score_ids", "buy_id", "sell_id",
])
@pytest.mark.parametrize("location", ["row", "outcome", "nested"])
def test_prohibited_raw_payload_and_scoring_keys_rejected_recursively(plan, category, location):
    record = _rows(plan)[0].to_dict()
    target = record if location == "row" else record["outcome"]
    if location == "nested":
        target["unexpected"] = [{"deep": {category: []}}]
    else:
        target[category] = []
    with pytest.raises(ValueError, match="prohibited"):
        C.ExecutionRow.from_json(_json(record))


@pytest.mark.parametrize("record_type", ["key", "identity", "plan", "outcome", "row", "shard"])
def test_every_record_rejects_unknown_and_missing_fields(plan, record_type):
    records = {"key": plan.keys[0], "identity": plan.identity, "plan": plan,
               "outcome": _outcome(plan.keys[0]), "row": _rows(plan)[0], "shard": _shards(plan, 1)[0]}
    record = records[record_type]
    kwargs = {"plan": plan} if record_type == "shard" else {}
    exported = record.to_dict()
    for invalid in (exported | {"extra": "unknown"},
                    {name: value for name, value in exported.items() if name != next(iter(exported))}):
        with pytest.raises(ValueError):
            type(record).from_dict(invalid, **kwargs)


@pytest.mark.parametrize("record_type", ["key", "identity", "plan", "outcome", "row", "shard"])
def test_every_record_json_loader_rejects_duplicate_keys_and_nonfinite_constants(plan, record_type):
    records = {"key": plan.keys[0], "identity": plan.identity, "plan": plan,
               "outcome": _outcome(plan.keys[0]), "row": _rows(plan)[0], "shard": _shards(plan, 1)[0]}
    record = records[record_type]
    kwargs = {"plan": plan} if record_type == "shard" else {}
    for text in ('{"extra":NaN}', '{"extra":Infinity}', '{"extra":-Infinity}',
                 '{"duplicate":1,"duplicate":2}', '[]', 'null', record.to_json() + ' {}'):
        with pytest.raises(ValueError):
            type(record).from_json(text, **kwargs)


def test_nested_duplicate_and_extra_keys_rejected(plan):
    row = _rows(plan)[0]
    duplicate = row.to_json().replace('"decision":"buy"', '"decision":"sell","decision":"buy"')
    with pytest.raises(ValueError, match="duplicate"):
        C.ExecutionRow.from_json(duplicate)
    for field in ("generated_text", "generated_token_ids", "reason", "provenance"):
        record = row.to_dict()
        record["outcome"][field] = "not supported by this typed version"
        with pytest.raises(ValueError):
            C.ExecutionRow.from_dict(record)


def test_plan_import_revalidates_claimed_hash_and_mutated_keys_gates(plan):
    for change in ("hash", "identity", "keys", "gates"):
        record = plan.to_dict()
        if change == "hash":
            record["plan_hash"] = "f" * 64
        elif change == "identity":
            record["identity"]["schema_sha256"] = "f" * 64
        elif change == "keys":
            record["keys"].pop()
        else:
            record["gate_names"].append("new_gate")
        with pytest.raises(ValueError, match="hash"):
            C.ExperimentPlan.from_dict(record)


def test_ingestion_revalidates_typed_records_even_if_frozen_setters_were_bypassed(plan):
    row = _rows(plan)[0]
    object.__setattr__(row.outcome, "decision", "hold")
    with pytest.raises(ValueError, match="decision"):
        C.validate_rows(plan, [row])
    identity = replace(plan.identity)
    object.__setattr__(identity, "backend_sha256", "forged")
    with pytest.raises(ValueError, match="backend_sha256"):
        C.ExperimentPlan(identity, plan.keys, GATES)
    key = replace(plan.keys[0])
    object.__setattr__(key, "dose", "0.50")
    with pytest.raises(ValueError, match="dose"):
        C.ExperimentPlan(plan.identity, [key], GATES)


def test_non_json_mutable_exports_are_rejected(plan):
    record = _rows(plan)[0].to_dict()
    record["outcome"]["extra"] = object()
    with pytest.raises(ValueError):
        C.ExecutionRow.from_dict(record)


@pytest.mark.parametrize("name", ["scores", "Raw_Activations", "raw-residuals", "KV cache"])
def test_reserved_gate_names_rejected_before_export(plan, name):
    with pytest.raises(ValueError, match="reserved gate"):
        C.ExperimentPlan(plan.identity, plan.keys, (name,))
    with pytest.raises(ValueError, match="reserved gate"):
        C.ExecutionShard(plan.plan_hash, 0, 1, _rows(plan), {name: True})
    with pytest.raises(ValueError, match="reserved gate"):
        C.progress(plan, [], {name: True})


def test_claimed_progress_import_rejects_bad_json(plan):
    for payload in ('{"complete":true,"complete":false}', '{"complete":NaN}', '[]', 'null'):
        with pytest.raises(ValueError):
            C.load_progress(plan, [], {}, payload)
