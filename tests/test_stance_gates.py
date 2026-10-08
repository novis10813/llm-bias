"""Exact full-output no-op eligibility, without models or generated-data persistence."""
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json

import numpy as np
import pytest

from llm_bias.core import experiment_contract as C
from llm_bias.core.stance_gates import (
    GATE_NAME, GateGenerationInput, NoOpGateResult, evaluate_noop_gate,
)

CONFIG_HASH = "a1" * 32
CHECK_KEYS = {
    "valid_config_hash", "baseline_nonempty", "baseline_primary_valid",
    "repeat_token_match", "zero_token_match", "self_token_match",
    "decision_match", "text_match", "schema_match", "all_primary_valid",
}
PREFIXES = ("baseline", "repeat", "zero", "self")
DIAGNOSTIC_KEYS = {
    f"{prefix}_{suffix}" for prefix in PREFIXES for suffix in
    ("token_count", "token_sha256", "text_sha256", "decision", "failure_type")
}
FAILURES = (
    "exception", "invalid_json", "invalid_reason", "invalid_schema",
    "no_legal_token", "timeout", "truncated", "unsupported_channel",
    "unsupported_tokenizer",
)


def generation(**changes):
    return replace(GateGenerationInput(
        token_ids=(11, 22, 33), text='{"decision":"buy","reason":"理由"}',
        decision="buy", schema_complete=True, reason_valid=True,
        decision_complete=True, failure_type=None,
    ), **changes)


def evaluate(*arms, **kwargs):
    return evaluate_noop_gate(*arms, config_hash=CONFIG_HASH, **kwargs)


def test_exact_success_and_canonical_compact_diagnostics():
    baseline = generation()
    result = evaluate(*(generation() for _ in PREFIXES))
    assert isinstance(result, NoOpGateResult)
    assert GATE_NAME == "no_op"
    assert result.passed is True
    assert result.config_hash == CONFIG_HASH
    assert set(result.checks) == CHECK_KEYS
    assert all(type(value) is bool and value for value in result.checks.values())
    assert set(result.diagnostics) == DIAGNOSTIC_KEYS
    expected_tokens = hashlib.sha256(b"[11,22,33]").hexdigest()
    expected_text = hashlib.sha256(baseline.text.encode("utf-8")).hexdigest()
    for prefix in PREFIXES:
        assert result.diagnostics[f"{prefix}_token_count"] == 3
        assert type(result.diagnostics[f"{prefix}_token_count"]) is int
        assert result.diagnostics[f"{prefix}_token_sha256"] == expected_tokens
        assert result.diagnostics[f"{prefix}_text_sha256"] == expected_text
        assert result.diagnostics[f"{prefix}_decision"] == "buy"
        assert result.diagnostics[f"{prefix}_failure_type"] is None
    payload = result.to_dict()
    assert set(payload) == {"passed", "config_hash", "checks", "diagnostics"}
    assert result.to_json() == json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False)
    assert json.loads(result.to_json()) == payload
    assert result.as_gate_dict() == {"no_op": True}
    assert baseline.text not in result.to_json()
    assert "token_ids" not in result.to_json()
    assert all(type(value) in (int, str, type(None)) for value in result.diagnostics.values())


def test_inputs_freeze_and_defensively_copy_iterables():
    tokens = [0, 11, 22]
    item = generation(token_ids=tokens)
    tokens.append(33)
    assert item.token_ids == (0, 11, 22)
    assert generation(token_ids=iter([0, 11])).token_ids == (0, 11)
    assert not hasattr(item, "__dict__")
    with pytest.raises(FrozenInstanceError):
        item.text = "changed"
    with pytest.raises(FrozenInstanceError):
        item.token_ids = ()


@pytest.mark.parametrize("tokens", [
    "123", b"123", bytearray(b"123"), {1: 2}, None, 123,
    [True], [False], [-1], [1.0], ["1"], [None], [[1]], [np.int64(1)],
])
def test_rejects_malformed_tokens(tokens):
    with pytest.raises((ValueError, TypeError)):
        generation(token_ids=tokens)


@pytest.mark.parametrize("token", [2**63, 10**5000], ids=["int64_overflow", "json_digit_overflow"])
def test_rejects_tokens_outside_hf_signed_int64(token):
    with pytest.raises(ValueError, match="int64"):
        generation(token_ids=(token,))


def test_largest_signed_int64_token_is_hashable():
    item = generation(token_ids=(0, 2**63 - 1))
    result = evaluate(item, item, item, item)
    assert result.passed is True
    expected = hashlib.sha256(b"[0,9223372036854775807]").hexdigest()
    assert result.diagnostics["baseline_token_sha256"] == expected


@pytest.mark.parametrize("text", [None, b"text", 1, "\ud800", "\udfff"])
def test_rejects_non_utf8_text(text):
    with pytest.raises((ValueError, TypeError)):
        generation(text=text)


@pytest.mark.parametrize("decision", ["BUY", "hold", "", True, 1, [], {}])
def test_rejects_unknown_decisions(decision):
    with pytest.raises((ValueError, TypeError)):
        generation(decision=decision)


@pytest.mark.parametrize("flag", ["schema_complete", "reason_valid", "decision_complete"])
@pytest.mark.parametrize("value", [0, 1, None, "true", np.bool_(True)])
def test_flags_are_genuine_booleans(flag, value):
    with pytest.raises((ValueError, TypeError)):
        generation(**{flag: value})


@pytest.mark.parametrize("failure", ["unsupported", "failed", "", True, 1, [], {}])
def test_rejects_noncanonical_failures(failure):
    with pytest.raises((ValueError, TypeError)):
        generation(failure_type=failure)


@pytest.mark.parametrize("failure", FAILURES)
def test_all_execution_failures_are_inputs_but_never_pass(failure):
    failed = generation(failure_type=failure)
    assert failed.primary_valid is False
    result = evaluate(failed, failed, failed, failed)
    assert result.passed is False
    assert result.checks["baseline_nonempty"] is True
    assert result.checks["baseline_primary_valid"] is False
    assert result.checks["all_primary_valid"] is False
    assert all(result.checks[key] for key in CHECK_KEYS - {
        "baseline_primary_valid", "all_primary_valid",
    })
    assert set(result.diagnostics) == DIAGNOSTIC_KEYS
    for prefix in PREFIXES:
        assert result.diagnostics[f"{prefix}_failure_type"] == failure


@pytest.mark.parametrize("change", [
    {"token_ids": ()}, {"decision": None}, {"schema_complete": False},
    {"reason_valid": False}, {"decision_complete": False},
    {"failure_type": "exception"},
])
def test_primary_valid_requires_every_success_component(change):
    assert generation().primary_valid is True
    assert generation(decision="sell").primary_valid is True
    assert generation(**change).primary_valid is False


def test_empty_repeated_failure_preserves_full_comparisons():
    baseline = generation(token_ids=(), text="", decision=None, schema_complete=False,
                          decision_complete=False, reason_valid=False, failure_type="exception")
    repeated = replace(baseline, text="different failure text", failure_type="timeout")
    zero = replace(baseline, token_ids=(0,), decision="sell", schema_complete=True)
    result = evaluate(baseline, repeated, zero, baseline)
    assert result.checks == {
        "valid_config_hash": True, "baseline_nonempty": False,
        "baseline_primary_valid": False, "repeat_token_match": True,
        "zero_token_match": False, "self_token_match": True,
        "decision_match": False, "text_match": False, "schema_match": False,
        "all_primary_valid": False,
    }
    assert result.passed is False
    assert set(result.diagnostics) == DIAGNOSTIC_KEYS
    assert result.diagnostics["baseline_token_count"] == 0
    assert result.diagnostics["baseline_token_sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert result.diagnostics["baseline_text_sha256"] == hashlib.sha256(b"").hexdigest()
    assert result.diagnostics["repeat_failure_type"] == "timeout"


@pytest.mark.parametrize("arm,check", [(1, "repeat_token_match"), (2, "zero_token_match"),
                                        (3, "self_token_match")])
@pytest.mark.parametrize("tokens", [(11, 22, 34), (11, 22), (11, 22, 33, 44), (33, 22, 11)])
def test_complete_token_identity_in_each_control(arm, check, tokens):
    arms = [generation() for _ in PREFIXES]
    arms[arm] = generation(token_ids=tokens)  # Same parsed decision/reason is insufficient.
    result = evaluate(*arms)
    assert result.passed is False
    assert result.checks[check] is False
    assert all(result.checks[key] for key in CHECK_KEYS - {check})
    assert result.diagnostics[f"{PREFIXES[arm]}_token_sha256"] != result.diagnostics["baseline_token_sha256"]


@pytest.mark.parametrize("arm", [1, 2, 3])
@pytest.mark.parametrize("change,check", [
    ({"text": "different reason"}, "text_match"),
    ({"decision": "sell"}, "decision_match"),
    ({"schema_complete": False}, "schema_match"),
    ({"reason_valid": False}, "schema_match"),
    ({"decision_complete": False}, "schema_match"),
    ({"failure_type": "invalid_reason"}, "all_primary_valid"),
])
def test_identical_tokens_cannot_hide_contradictory_text_decision_or_flags(arm, change, check):
    arms = [generation() for _ in PREFIXES]
    arms[arm] = generation(**change)
    result = evaluate(*arms)
    assert result.passed is False
    assert result.checks[check] is False
    assert all(result.checks[key] for key in
               ("repeat_token_match", "zero_token_match", "self_token_match"))
    assert result.passed == all(result.checks.values())


def test_text_is_exact_utf8_not_normalized_or_reparsed():
    composed = generation(text="é")
    decomposed = generation(text="e\u0301")
    assert evaluate(composed, composed, composed, composed).passed
    result = evaluate(composed, decomposed, composed, composed)
    assert result.checks["text_match"] is False
    assert result.diagnostics["baseline_text_sha256"] != result.diagnostics["repeat_text_sha256"]
    # Caller binds structured flags: this pure gate deliberately does not parse JSON.
    assert evaluate(*(generation(text="") for _ in PREFIXES)).passed


@pytest.mark.parametrize("config_hash", [None, 1, b"a" * 64, "A" * 64, "g" * 64,
                                         "a" * 63, "a" * 65, "a" * 64 + "\n"])
def test_bad_config_hash_raises_instead_of_returning_gate(config_hash):
    with pytest.raises((ValueError, TypeError)):
        evaluate_noop_gate(*(generation() for _ in PREFIXES), config_hash=config_hash)


@pytest.mark.parametrize("arm", range(4))
@pytest.mark.parametrize("invalid", [None, {}, (), "generation", object()])
def test_requires_actual_input_instances(arm, invalid):
    arms = [generation() for _ in PREFIXES]
    arms[arm] = invalid
    with pytest.raises((ValueError, TypeError)):
        evaluate(*arms)


@pytest.mark.parametrize("bypass", ["full", "stage", "precision", "hardware", "smoke_only"])
def test_no_bypass_arguments(bypass):
    with pytest.raises(TypeError):
        evaluate(*(generation() for _ in PREFIXES), **{bypass: True})


def test_result_and_all_exports_are_immutable_or_defensive():
    result = evaluate(*(generation() for _ in PREFIXES))
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.passed = False
    with pytest.raises(FrozenInstanceError):
        result.config_hash = "b" * 64
    original = result.to_json()
    result.checks["text_match"] = False
    result.diagnostics["baseline_decision"] = "sell"
    payload = result.to_dict()
    payload["checks"].clear()
    payload["diagnostics"]["baseline_token_count"] = 999
    payload["passed"] = False
    result.as_gate_dict()["no_op"] = False
    assert result.to_json() == original
    checks, diagnostics = result.checks, result.diagnostics
    copied = NoOpGateResult(result.passed, result.config_hash, checks, diagnostics)
    checks.clear()
    diagnostics.clear()
    assert copied.to_json() == original
    assert result.to_dict()["checks"] is not result.to_dict()["checks"]
    assert result.to_dict()["diagnostics"] is not result.diagnostics


@pytest.mark.parametrize("stage", ["baseline", "localization", "method", "comparison"])
@pytest.mark.parametrize("failure", [None, "unsupported_tokenizer"])
def test_false_noop_blocks_existing_progress_and_full_merge_at_every_stage(stage, failure):
    identity = C.PlanIdentity(**{field.name: CONFIG_HASH for field in fields(C.PlanIdentity)})
    key = C.RowKey(stage, "ACME", "++", "trial-1", "baseline", "0")
    plan = C.ExperimentPlan(identity, (key,), (GATE_NAME,))
    outcome = C.GenerationOutcome(
        "ACME", "issuer-1", "++", "trial-1", "buy" if failure is None else None,
        True, True, True, "eos" if failure is None else "unsupported", failure,
    )
    row = C.ExecutionRow(key, outcome)
    good = generation()
    control = (generation(token_ids=(11, 22, 99)) if failure is None
               else generation(decision=None, failure_type=failure))
    gate = evaluate(good, control, good, good)
    state = C.progress(plan, (row,), gate.as_gate_dict())
    assert state["complete"] is True
    assert state["eligible"] is False
    assert state["gates"] == {"no_op": False}
    shard = C.ExecutionShard(plan.plan_hash, 0, 1, (row,), gate.as_gate_dict())
    merged_rows, full_state = C.merge_shards(plan, (shard,), require_complete=True)
    assert merged_rows == (row,)
    assert (full_state["planned"], full_state["executed"], full_state["missing"]) == (1, 1, 0)
    assert merged_rows[0].outcome.failure_type == failure
    assert set(gate.checks) == CHECK_KEYS and set(gate.diagnostics) == DIAGNOSTIC_KEYS
    assert gate.diagnostics["repeat_failure_type"] == failure
    assert full_state == state
    with pytest.raises(ValueError, match="contradicts"):
        C.load_progress(plan, (row,), gate.as_gate_dict(), json.dumps(state | {"eligible": True}))
    passing_gate = evaluate(good, good, good, good)
    assert C.progress(plan, (row,), passing_gate.as_gate_dict())["eligible"] is True


@pytest.mark.parametrize("failed_arm", range(4))
def test_one_unsupported_tokenizer_arm_with_identical_complete_diagnostics_fails(failed_arm):
    arms = [generation() for _ in PREFIXES]
    arms[failed_arm] = generation(failure_type="unsupported_tokenizer")
    result = evaluate(*arms)
    assert result.passed is False
    assert set(result.checks) == CHECK_KEYS and len(result.checks) == 10
    assert result.checks["all_primary_valid"] is False
    assert result.checks["baseline_primary_valid"] is (failed_arm != 0)
    assert all(result.checks[key] for key in CHECK_KEYS - {"all_primary_valid", "baseline_primary_valid"})
    assert set(result.diagnostics) == DIAGNOSTIC_KEYS and len(result.diagnostics) == 20
    for prefix in PREFIXES:
        assert result.diagnostics[f"{prefix}_failure_type"] == (
            "unsupported_tokenizer" if prefix == PREFIXES[failed_arm] else None
        )
        assert result.diagnostics[f"{prefix}_decision"] == "buy"
        assert result.diagnostics[f"{prefix}_token_count"] == 3
        assert result.diagnostics[f"{prefix}_token_sha256"] == result.diagnostics["baseline_token_sha256"]
        assert result.diagnostics[f"{prefix}_text_sha256"] == result.diagnostics["baseline_text_sha256"]
    copied = NoOpGateResult(**json.loads(result.to_json()))
    assert copied.to_json() == result.to_json()
    assert copied.as_gate_dict() == {"no_op": False}


@pytest.mark.parametrize("changes", [
    {"token_ids": (True,)}, {"token_ids": (2**63,)}, {"text": "\ud800"},
    {"schema_complete": 1}, {"reason_valid": None}, {"decision_complete": "true"},
])
def test_unsupported_tokenizer_does_not_bypass_caller_validation(changes):
    with pytest.raises(ValueError):
        generation(failure_type="unsupported_tokenizer", **changes)
