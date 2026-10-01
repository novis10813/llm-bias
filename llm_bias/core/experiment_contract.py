"""Frozen plans and tensor-free execution accounting for new stance experiments.

These pure contracts consume explicit provenance hashes; they do not approve
protocols, inputs, models or research gates. An executed failed generation is a
row, whereas an unexecuted generation is absent. Coverage and declared gates,
not observed decisions, determine execution eligibility. No efficacy is computed.

JSON imports accept only the named fields of this typed version. Generated text,
token provenance and scoring fields are deliberately not extensible payloads.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json

_HASH = re.compile(r"[0-9a-f]{64}")
_CONDITIONS = ("++", "+-", "-+", "--")
_FINISHES = ("eos", "schema_complete", "token_budget", "timeout", "exception",
             "no_legal_token", "unsupported")
_FAILURE_FINISH = {
    "truncated": "token_budget", "timeout": "timeout", "exception": "exception",
    "no_legal_token": "no_legal_token", "unsupported_channel": "unsupported",
    "unsupported_tokenizer": "unsupported",
}
_FAILURES = (*_FAILURE_FINISH, "invalid_json", "invalid_schema", "invalid_reason")
_PROHIBITED = {
    "activation", "activations", "rawactivation", "rawactivations",
    "residual", "residuals", "rawresidual", "rawresiduals",
    "gradient", "gradients", "rawgradient", "rawgradients",
    "kv", "kvcache", "rawkv", "rawkvcache", "pastkeyvalues",
    "tensor", "tensors", "rawtensor", "rawtensors", "hiddenstates", "logits",
    "margin", "margins", "buysellmargin", "score", "scores", "scoreids",
    "buyid", "sellid",
}


def _nonblank(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")


def _gate_name(value: Any) -> None:
    _nonblank(value, "gate name")
    if re.sub(r"[^a-z0-9]", "", value.lower()) in _PROHIBITED:
        raise ValueError(f"reserved gate name: {value!r}")


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")


def _bool(value: Any, label: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{label} must be strictly bool")


def _sequence(value: Iterable[Any], label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be an iterable of records, not a string or mapping")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable") from exc


def _audit_payload(value: Any) -> None:
    """Reject forbidden keys at any depth, even beneath an unknown outer field."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized in _PROHIBITED:
                raise ValueError(f"prohibited artifact/scoring payload key: {key!r}")
            _audit_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _audit_payload(item)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("payload must contain only JSON values, never raw tensors")


def _json_copy(value: Any) -> Any:
    try:
        _audit_payload(value)
        return json.loads(canonical_json_bytes(value))
    except (TypeError, RecursionError, OverflowError) as exc:
        raise ValueError("payload must be finite, tensor-free JSON") from exc


def _strict_json(text: str) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    if not isinstance(text, str):
        raise ValueError("JSON input must be text")
    try:
        value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=nonfinite)
        return _json_copy(value)
    except (TypeError, RecursionError) as exc:
        raise ValueError("invalid JSON input") from exc


def _record_payload(value: Any, names: Iterable[str], label: str) -> dict[str, Any]:
    names = tuple(names)
    value = _json_copy(value)
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{label} has missing or unknown fields; expected {names}")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _export(value: Any) -> Any:
    if isinstance(value, _JSONRecord):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _export(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_export(item) for item in value]
    return value


class _JSONRecord:
    """Canonical defensive exports and strict imports for the named records."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _export(getattr(self, field.name)) for field in fields(self)}

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value):
        record = _record_payload(value, (field.name for field in fields(cls)), cls.__name__)
        return cls(**record)

    @classmethod
    def from_json(cls, text: str, **kwargs):
        return cls.from_dict(_strict_json(text), **kwargs)


def _copy_record(value: Any, cls):
    if not isinstance(value, cls):
        raise ValueError(f"expected a {cls.__name__} record")
    return cls(**{field.name: getattr(value, field.name) for field in fields(cls)})


def canonical_dose_str(value: str | int | float | Decimal) -> str:
    """Normalize Decimal(str(value)) exactly, without exponent output or rounding.

    Negative zero is rejected rather than silently losing its sign. Positive
    zero normalizes to '0'. RowKey requires this already-canonical spelling.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("dose must be a finite str/int/float/Decimal, never bool")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("dose must be a finite decimal") from exc
    if not number.is_finite() or (number.is_zero() and number.is_signed()):
        raise ValueError("dose must be finite and cannot be negative zero")
    result = format(number, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


@dataclass(frozen=True, slots=True, order=True)
class RowKey(_JSONRecord):
    """Full planned unit; dose is an already-canonical finite decimal string."""

    stage: str
    ticker: str
    condition: str
    trial_id: str
    arm: str
    dose: str

    def __post_init__(self):
        for field in fields(self):
            _nonblank(getattr(self, field.name), field.name)
        if self.condition not in _CONDITIONS:
            raise ValueError(f"condition must be one of {_CONDITIONS}")
        if canonical_dose_str(self.dose) != self.dose:
            raise ValueError("dose must already be a canonical decimal string")


@dataclass(frozen=True, slots=True)
class PlanIdentity(_JSONRecord):
    """Thirteen explicit identities, without inferred protocol or input defaults.

    Only operator/parent can be 'not_applicable', and ExperimentPlan restricts
    those sentinels to plans containing exclusively arm='baseline' keys.
    """

    protocol_sha256: str
    population_sha256: str
    issuer_sha256: str
    roles_sha256: str
    evidence_sha256: str
    schema_sha256: str
    template_sha256: str
    model_sha256: str
    code_sha256: str
    backend_sha256: str
    generation_policy_sha256: str
    operator_sha256: str
    parent_sha256: str

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ("operator_sha256", "parent_sha256") and value == "not_applicable":
                continue
            _hash(value, field.name)


@dataclass(frozen=True, slots=True)
class ExperimentPlan(_JSONRecord):
    """Copied, immutable sorted keys/gates and identity, frozen before execution.

    Hash payload is exactly identity, sorted key dictionaries, sorted gate names.
    Exports add the derived plan_hash; imports verify it, never trust a claim.
    """

    identity: PlanIdentity
    keys: tuple[RowKey, ...]
    gate_names: tuple[str, ...]

    def __post_init__(self):
        identity = _copy_record(self.identity, PlanIdentity)
        keys = tuple(_copy_record(key, RowKey) for key in _sequence(self.keys, "keys"))
        if not keys:
            raise ValueError("planned keys must be nonempty")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate planned row key")
        if ("not_applicable" in (identity.operator_sha256, identity.parent_sha256)
                and any(key.arm != "baseline" for key in keys)):
            raise ValueError("not_applicable operator/parent identity is baseline-only")
        gates = _sequence(self.gate_names, "gate_names")
        if not gates:
            raise ValueError("gate_names must be nonempty")
        for gate in gates:
            _gate_name(gate)
        if len(set(gates)) != len(gates):
            raise ValueError("duplicate gate name")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "keys", tuple(sorted(keys)))
        object.__setattr__(self, "gate_names", tuple(sorted(gates)))

    @property
    def plan_hash(self) -> str:
        return sha256_json(super(ExperimentPlan, self).to_dict())

    def to_dict(self) -> dict[str, Any]:
        return super(ExperimentPlan, self).to_dict() | {"plan_hash": self.plan_hash}

    @classmethod
    def from_dict(cls, value):
        record = _record_payload(value, ("identity", "keys", "gate_names", "plan_hash"), cls.__name__)
        _hash(record["plan_hash"], "plan_hash")
        plan = cls(
            PlanIdentity.from_dict(record["identity"]),
            tuple(RowKey.from_dict(key) for key in _array(record["keys"], "keys")),
            _array(record["gate_names"], "gate_names"),
        )
        if record["plan_hash"] != plan.plan_hash:
            raise ValueError("claimed plan_hash differs from canonical plan hash")
        return plan

    def keys_for_shard(self, shard_index: int, num_shards: int) -> tuple[RowKey, ...]:
        _shard_numbers(shard_index, num_shards)
        plan = _copy_record(self, ExperimentPlan)
        return tuple(key for key in plan.keys if shard_index_for_key(key, num_shards) == shard_index)


def _shard_numbers(index: int, count: int) -> None:
    if type(count) is not int or count < 1:
        raise ValueError("num_shards must be a positive integer")
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("shard_index must be an integer in [0, num_shards)")


def shard_index_for_key(key: RowKey, num_shards: int) -> int:
    """Full SHA-256 integer of canonical full-key JSON modulo num_shards."""
    _shard_numbers(0, num_shards)
    validated = _copy_record(key, RowKey)
    return int(sha256_json(validated.to_dict()), 16) % num_shards


@dataclass(frozen=True, slots=True)
class GenerationOutcome(_JSONRecord):
    """Tensor-free generated decision status, not a parser or decision repair.

    Failures always have a null primary decision. decision_complete may remain
    true diagnostically on failed complete-decision cases; schema_complete may
    be true with an invalid whitespace reason. This version stores no reason
    text, generated text/tokens, arbitrary provenance or scoring payload.
    """

    ticker: str
    issuer_id: str
    condition: str
    trial_id: str
    decision: str | None
    decision_complete: bool
    schema_complete: bool
    reason_valid: bool
    finish_reason: str
    failure_type: str | None

    def __post_init__(self):
        for name in ("ticker", "issuer_id", "condition", "trial_id"):
            _nonblank(getattr(self, name), name)
        if self.condition not in _CONDITIONS:
            raise ValueError(f"condition must be one of {_CONDITIONS}")
        for name in ("decision_complete", "schema_complete", "reason_valid"):
            _bool(getattr(self, name), name)
        if self.decision is not None and self.decision not in ("buy", "sell"):
            raise ValueError("decision must be buy, sell or null")
        if self.finish_reason not in _FINISHES:
            raise ValueError("unknown finish_reason")
        if self.failure_type is not None and self.failure_type not in _FAILURES:
            raise ValueError("unknown failure_type")
        if self.schema_complete and not self.decision_complete:
            raise ValueError("schema_complete requires decision_complete")
        if self.reason_valid and not self.schema_complete:
            raise ValueError("reason_valid requires schema_complete")
        if self.decision is not None and (not self.decision_complete or self.failure_type is not None):
            raise ValueError("a primary decision requires decision_complete and no failure")
        if self.failure_type is None:
            if not self.primary_valid:
                raise ValueError("null failure_type requires a primary-valid complete outcome")
            return
        if self.decision is not None:
            raise ValueError("every failure must clear primary decision to null")
        expected = _FAILURE_FINISH.get(self.failure_type)
        if expected is not None and self.finish_reason != expected:
            raise ValueError("failure_type does not match finish_reason")
        if expected is None and self.finish_reason not in ("eos", "schema_complete"):
            raise ValueError("invalid payload failure requires an eos/schema_complete finish")
        if self.failure_type in ("truncated", "invalid_json", "invalid_schema") and self.schema_complete:
            raise ValueError(f"{self.failure_type} requires schema_complete false")
        if self.failure_type == "invalid_reason" and self.reason_valid:
            raise ValueError("invalid_reason requires reason_valid false")

    @property
    def primary_valid(self) -> bool:
        return (self.decision in ("buy", "sell") and self.decision_complete
                and self.schema_complete and self.reason_valid and self.failure_type is None
                and self.finish_reason in ("eos", "schema_complete"))


def validate_outcome(outcome: GenerationOutcome) -> GenerationOutcome:
    """Revalidate a typed outcome without salvaging any failed primary decision."""
    return _copy_record(outcome, GenerationOutcome)


@dataclass(frozen=True, slots=True)
class ExecutionRow(_JSONRecord):
    """One executed planned key, including a completed failed generation."""

    key: RowKey
    outcome: GenerationOutcome

    def __post_init__(self):
        key = _copy_record(self.key, RowKey)
        outcome = validate_outcome(self.outcome)
        if (key.ticker, key.condition, key.trial_id) != (
            outcome.ticker, outcome.condition, outcome.trial_id,
        ):
            raise ValueError("outcome identity must match row key ticker/condition/trial")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "outcome", outcome)

    @classmethod
    def from_dict(cls, value):
        record = _record_payload(value, ("key", "outcome"), cls.__name__)
        return cls(RowKey.from_dict(record["key"]), GenerationOutcome.from_dict(record["outcome"]))


def validate_rows(
    plan: ExperimentPlan, rows: Iterable[ExecutionRow], *, require_complete: bool = False,
) -> tuple[ExecutionRow, ...]:
    """Revalidate copied rows, rejecting foreign/duplicate keys before coverage."""
    _bool(require_complete, "require_complete")
    plan = _copy_record(plan, ExperimentPlan)
    expected = set(plan.keys)
    validated: dict[RowKey, ExecutionRow] = {}
    for candidate in _sequence(rows, "rows"):
        row = _copy_record(candidate, ExecutionRow)
        if row.key not in expected:
            raise ValueError("foreign/unexpected execution row key")
        if row.key in validated:
            raise ValueError("duplicate/overlapping execution row key")
        validated[row.key] = row
    if require_complete and set(validated) != expected:
        raise ValueError(f"incomplete execution: {len(expected - set(validated))} missing rows")
    return tuple(validated[key] for key in sorted(validated))


def _gates(plan: ExperimentPlan, gates: Mapping[str, bool] | None, *, exact: bool) -> dict[str, bool | None]:
    if gates is None:
        if exact:
            raise ValueError("shard gates must exactly match plan gate_names")
        gates = {}
    if not isinstance(gates, Mapping):
        raise ValueError("gates must be a mapping")
    for gate, passed in gates.items():
        _gate_name(gate)
        _bool(passed, f"gate {gate!r}")
    if not set(gates) <= set(plan.gate_names) or (exact and set(gates) != set(plan.gate_names)):
        raise ValueError("gate names must match declared plan gate_names")
    return {gate: gates.get(gate) for gate in plan.gate_names}


def progress(
    plan: ExperimentPlan, rows: Iterable[ExecutionRow], gates: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Computed accounting only; missing gate states are null, never passed.

    Complete means exact row coverage. Eligibility additionally requires every
    declared gate true. Completed failures count as executed, not missing.
    """
    plan = _copy_record(plan, ExperimentPlan)
    validated = validate_rows(plan, rows)
    states = _gates(plan, gates, exact=False)
    complete = len(validated) == len(plan.keys)
    return {
        "planned": len(plan.keys), "executed": len(validated),
        "missing": len(plan.keys) - len(validated), "complete": complete,
        "eligible": complete and all(value is True for value in states.values()),
        "gates": states,
    }


def load_progress(
    plan: ExperimentPlan, rows: Iterable[ExecutionRow], gates: Mapping[str, bool] | None, text: str,
) -> dict[str, Any]:
    """Import row-level progress only if every field equals recomputed accounting.

    Comparing canonical JSON also rejects bool/int substitutions. No claimed
    complete or eligible flag can override missing rows or unpassed gates.
    """
    computed = progress(plan, rows, gates)
    claimed = _record_payload(_strict_json(text), computed, "progress")
    if canonical_json_bytes(claimed) != canonical_json_bytes(computed):
        raise ValueError("claimed progress/complete/eligible contradicts computed state")
    return computed


@dataclass(frozen=True, slots=True)
class ExecutionShard(_JSONRecord):
    """Registered shard with copied rows and immutable copied boolean gates.

    Constructor checks its index/count, unique keys and deterministic ownership.
    validate_shard or the plan-bound JSON loader additionally checks the plan
    hash, planned membership, exact gate names and an optional expected count.
    """

    plan_hash: str
    shard_index: int
    num_shards: int
    rows: tuple[ExecutionRow, ...]
    gates: dict[str, bool]

    def __post_init__(self):
        _hash(self.plan_hash, "plan_hash")
        _shard_numbers(self.shard_index, self.num_shards)
        rows = tuple(_copy_record(row, ExecutionRow) for row in _sequence(self.rows, "rows"))
        seen = set()
        for row in rows:
            if row.key in seen:
                raise ValueError("duplicate/overlapping row keys within shard")
            seen.add(row.key)
            if shard_index_for_key(row.key, self.num_shards) != self.shard_index:
                raise ValueError("row key belongs to a different deterministic shard")
        if not isinstance(self.gates, Mapping):
            raise ValueError("gates must be a mapping")
        gates = {}
        for name, passed in self.gates.items():
            _gate_name(name)
            _bool(passed, f"gate {name!r}")
            gates[name] = passed
        object.__setattr__(self, "rows", tuple(sorted(rows, key=lambda row: row.key)))
        object.__setattr__(self, "gates", MappingProxyType(dict(sorted(gates.items()))))

    @classmethod
    def from_dict(cls, value, *, plan: ExperimentPlan, num_shards: int | None = None):
        record = _record_payload(value, ("plan_hash", "shard_index", "num_shards", "rows", "gates"), cls.__name__)
        shard = cls(
            record["plan_hash"], record["shard_index"], record["num_shards"],
            tuple(ExecutionRow.from_dict(row) for row in _array(record["rows"], "rows")),
            record["gates"],
        )
        return validate_shard(plan, shard, num_shards=num_shards)


def validate_shard(
    plan: ExperimentPlan, shard: ExecutionShard, *, num_shards: int | None = None,
) -> ExecutionShard:
    """Check identity and registration before interpreting row coverage or gates."""
    plan = _copy_record(plan, ExperimentPlan)
    shard = _copy_record(shard, ExecutionShard)
    if shard.plan_hash != plan.plan_hash:
        raise ValueError("shard plan hash/identity differs from expected plan")
    if num_shards is not None:
        _shard_numbers(0, num_shards)
        if shard.num_shards != num_shards:
            raise ValueError("shard num_shards differs from expected shard count")
    validate_rows(plan, shard.rows)
    _gates(plan, shard.gates, exact=True)
    return shard


def validate_resume(saved_plan: ExperimentPlan, current_plan: ExperimentPlan) -> None:
    """Require exact full identities, planned keys and gates, not just a run ID."""
    saved = _copy_record(saved_plan, ExperimentPlan)
    current = _copy_record(current_plan, ExperimentPlan)
    if saved.to_json() != current.to_json():
        raise ValueError("resume requires identical canonical identities, keys and gate_names")


def merge_shards(
    plan: ExperimentPlan, shards: Iterable[ExecutionShard], *, require_complete: bool = True,
) -> tuple[tuple[ExecutionRow, ...], dict[str, Any]]:
    """Return validated sorted rows and computed coverage/gates, never efficacy.

    Full mode requires all registered indices and exact planned row coverage,
    including empty shards. A failed gate does not erase executed rows or make
    coverage incomplete; it makes eligibility false. Partial mode never treats
    unregistered/missing shards as passed gates or completed execution.
    """
    _bool(require_complete, "require_complete")
    plan = _copy_record(plan, ExperimentPlan)
    candidates = _sequence(shards, "shards")
    if not candidates:
        if require_complete:
            raise ValueError("incomplete merge: no registered shards")
        return (), progress(plan, ())
    first = _copy_record(candidates[0], ExecutionShard)
    num_shards = first.num_shards
    by_index: dict[int, ExecutionShard] = {}
    all_rows = []
    for candidate in candidates:
        shard = validate_shard(plan, candidate, num_shards=num_shards)
        if shard.shard_index in by_index:
            raise ValueError("duplicate shard index/overlapping shard")
        by_index[shard.shard_index] = shard
        all_rows.extend(shard.rows)
    registered = len(by_index) == num_shards
    rows = validate_rows(plan, all_rows, require_complete=require_complete)
    if require_complete and not registered:
        raise ValueError("incomplete merge: missing registered shard indices")
    states = {}
    for gate in plan.gate_names:
        if any(shard.gates[gate] is False for shard in by_index.values()):
            states[gate] = False
        elif registered:
            states[gate] = True
    state = progress(plan, rows, states)
    if not registered:
        state["complete"] = False
        state["eligible"] = False
    return rows, state
