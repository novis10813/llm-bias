"""Pure, fail-closed full-generation no-op identity gate.

Inputs hold complete output IDs transiently. Exports contain only scalar status,
counts and canonical hashes; structured validity flags must be bound to verified
generation results by the caller. This gate neither parses text nor runs models.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json

GATE_NAME = "no_op"
_HASH = re.compile(r"[0-9a-f]{64}")
# Public GenerationOutcome failure vocabulary; no inference-module dependency.
_FAILURES = (
    "exception", "invalid_json", "invalid_reason", "invalid_schema",
    "no_legal_token", "timeout", "truncated", "unsupported_channel",
    "unsupported_tokenizer",
)
_PREFIXES = ("baseline", "repeat", "zero", "self")
_CHECK_KEYS = frozenset((
    "valid_config_hash", "baseline_nonempty", "baseline_primary_valid",
    "repeat_token_match", "zero_token_match", "self_token_match",
    "decision_match", "text_match", "schema_match", "all_primary_valid",
))
_DIAGNOSTIC_KEYS = frozenset(
    f"{prefix}_{suffix}" for prefix in _PREFIXES for suffix in
    ("token_count", "token_sha256", "text_sha256", "decision", "failure_type")
)


def _require_hash(value: str, label: str) -> None:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")


def _require_decision(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or value not in ("buy", "sell")):
        raise ValueError("decision must be buy, sell or None")


def _require_failure(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or value not in _FAILURES):
        raise ValueError("failure_type must be a canonical execution failure or None")


@dataclass(frozen=True, slots=True)
class GateGenerationInput:
    """Immutable complete output and independently verified structured status.

    Failed/incomplete states are legal inputs, but can never pass the gate. No
    successful-outcome invariants are imposed on their diagnostic flags.
    """

    token_ids: tuple[int, ...]
    text: str
    decision: str | None
    schema_complete: bool
    reason_valid: bool
    decision_complete: bool
    failure_type: str | None

    def __post_init__(self):
        if isinstance(self.token_ids, (str, bytes, bytearray, Mapping)):
            raise ValueError("token_ids must be an iterable of Python integer IDs")
        try:
            tokens = tuple(self.token_ids)
        except TypeError as exc:
            raise ValueError("token_ids must be an iterable of Python integer IDs") from exc
        if any(type(token) is not int or not 0 <= token <= 2**63 - 1 for token in tokens):
            raise ValueError("token IDs must be genuine nonbool nonnegative signed-int64 Python integers")
        object.__setattr__(self, "token_ids", tokens)
        if not isinstance(self.text, str):
            raise ValueError("text must be a UTF-8-encodable string")
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("text must be a UTF-8-encodable string") from exc
        _require_decision(self.decision)
        _require_failure(self.failure_type)
        for name in ("schema_complete", "reason_valid", "decision_complete"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be strictly bool")

    @property
    def primary_valid(self) -> bool:
        return (bool(self.token_ids) and self.decision in ("buy", "sell")
                and self.schema_complete and self.reason_valid and self.decision_complete
                and self.failure_type is None)


@dataclass(frozen=True, slots=True, init=False)
class NoOpGateResult:
    """Immutable compact result with fresh, defensive dictionary exports."""

    passed: bool
    config_hash: str
    _checks: tuple[tuple[str, bool], ...]
    _diagnostics: tuple[tuple[str, int | str | None], ...]

    def __init__(self, passed: bool, config_hash: str, checks: dict[str, bool],
                 diagnostics: dict[str, int | str | None]):
        _require_hash(config_hash, "config_hash")
        if not isinstance(checks, Mapping) or set(checks) != _CHECK_KEYS:
            raise ValueError("checks must contain exactly the ten no-op check keys")
        checks = dict(checks)
        if any(type(value) is not bool for value in checks.values()):
            raise ValueError("check values must be strictly bool")
        if type(passed) is not bool or passed != all(checks.values()):
            raise ValueError("passed must equal all(checks.values())")
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != _DIAGNOSTIC_KEYS:
            raise ValueError("diagnostics must contain exactly the twenty compact keys")
        diagnostics = dict(diagnostics)
        for prefix in _PREFIXES:
            count = diagnostics[f"{prefix}_token_count"]
            if type(count) is not int or count < 0:
                raise ValueError("token_count must be a nonnegative Python integer")
            for suffix in ("token_sha256", "text_sha256"):
                _require_hash(diagnostics[f"{prefix}_{suffix}"], suffix)
            _require_decision(diagnostics[f"{prefix}_decision"])
            _require_failure(diagnostics[f"{prefix}_failure_type"])
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "config_hash", config_hash)
        object.__setattr__(self, "_checks", tuple(sorted(checks.items())))
        object.__setattr__(self, "_diagnostics", tuple(sorted(diagnostics.items())))

    @property
    def checks(self) -> dict[str, bool]:
        return dict(self._checks)

    @property
    def diagnostics(self) -> dict[str, int | str | None]:
        return dict(self._diagnostics)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "config_hash": self.config_hash,
                "checks": self.checks, "diagnostics": self.diagnostics}

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    def as_gate_dict(self) -> dict[str, bool]:
        return {GATE_NAME: self.passed}


def evaluate_noop_gate(
    baseline: GateGenerationInput,
    repeated_baseline: GateGenerationInput,
    zero_addition: GateGenerationInput,
    self_replacement: GateGenerationInput,
    *, config_hash: str,
) -> NoOpGateResult:
    """Compare all four complete generations, including failed/empty outputs.

    Bad invocations raise; well-formed failed generations return a full failed
    result. Zero addition must come from a valid nonzero direction with dose 0,
    not from an invalid all-zero vector. There are no stage/hardware bypasses.
    """
    arms = (baseline, repeated_baseline, zero_addition, self_replacement)
    if any(not isinstance(arm, GateGenerationInput) for arm in arms):
        raise TypeError("all four generations must be GateGenerationInput instances")
    _require_hash(config_hash, "config_hash")
    controls = arms[1:]
    baseline_schema = (baseline.decision_complete, baseline.schema_complete, baseline.reason_valid)
    checks = {
        "valid_config_hash": True,
        "baseline_nonempty": bool(baseline.token_ids),
        "baseline_primary_valid": baseline.primary_valid,
        "repeat_token_match": repeated_baseline.token_ids == baseline.token_ids,
        "zero_token_match": zero_addition.token_ids == baseline.token_ids,
        "self_token_match": self_replacement.token_ids == baseline.token_ids,
        "decision_match": all(arm.decision == baseline.decision for arm in controls),
        "text_match": all(arm.text == baseline.text for arm in controls),
        "schema_match": all(
            (arm.decision_complete, arm.schema_complete, arm.reason_valid) == baseline_schema
            for arm in controls
        ),
        "all_primary_valid": all(arm.primary_valid for arm in arms),
    }
    diagnostics = {}
    for prefix, arm in zip(_PREFIXES, arms):
        diagnostics.update({
            f"{prefix}_token_count": len(arm.token_ids),
            f"{prefix}_token_sha256": sha256_json(list(arm.token_ids)),
            f"{prefix}_text_sha256": sha256_bytes(arm.text.encode("utf-8")),
            f"{prefix}_decision": arm.decision,
            f"{prefix}_failure_type": arm.failure_type,
        })
    return NoOpGateResult(all(checks.values()), config_hash, checks, diagnostics)
