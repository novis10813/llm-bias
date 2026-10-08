"""Lossless baseline adapters; validation does not certify model provenance."""
from __future__ import annotations

import json
import math

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.experiment_contract import ExecutionRow, ExperimentPlan, GenerationOutcome, RowKey
from llm_bias.core.inference.structured_output import StructuredGenerationResult
from llm_bias.core.stance_baseline_inputs import BaselineInputs
from llm_bias.core.stance_baseline_plan import build_baseline_plan
from llm_bias.core.stance_gates import GateGenerationInput


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate provenance key')
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError('nonfinite provenance constant')


def _validate_generation_result(result: StructuredGenerationResult) -> GenerationOutcome:
    """Check structural integrity only, without decoding or reparsing payloads."""
    if not isinstance(result, StructuredGenerationResult):
        raise ValueError('expected StructuredGenerationResult')
    try:
        tokens = result.generated_token_ids
        if (type(tokens) is not tuple
                or any(type(t) is not int or not 0 <= t <= 2**63 - 1 for t in tokens)
                or result.generated_token_sha256 != sha256_json(list(tokens))):
            raise ValueError('invalid continuation IDs or hash')
        for name in ('generated_text', 'json_payload', 'reason', 'error_message', 'decode_error'):
            value = getattr(result, name)
            if value is None and name in ('reason', 'error_message', 'decode_error'):
                continue
            if not isinstance(value, str):
                raise ValueError(f'{name} must be a UTF8 string')
            value.encode('utf-8', errors='strict')
        elapsed = result.elapsed_seconds
        if (type(elapsed) not in (int, float) or elapsed < 0 or not math.isfinite(elapsed)):
            raise ValueError('elapsed_seconds must be finite and nonnegative')
        if type(result._provenance_bytes) is not bytes:
            raise ValueError('provenance storage must be bytes')
        provenance = json.loads(result._provenance_bytes.decode('utf-8', errors='strict'),
                                object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        # allow_nan=False also rejects overflowed finite-looking JSON literals.
        if (not isinstance(provenance, dict)
                or canonical_json_bytes(provenance) != result._provenance_bytes):
            raise ValueError('provenance must be a canonical strict JSON object')
        outcome = GenerationOutcome(
            'synthetic', 'synthetic', '++', 'synthetic', result.decision,
            result.decision_complete, result.schema_complete, result.reason_valid,
            result.finish_reason, result.failure_type,
        )
        if result.failure_type is None:
            if (result.reason is None or not result.reason.strip() or result.error_message is not None
                    or result.decode_error is not None or not tokens or not result.generated_text
                    or not result.json_payload):
                raise ValueError('success requires complete nonempty output and reason without errors')
        elif result.reason is not None:
            raise ValueError('failure must have null primary reason')
        return outcome
    except (TypeError, AttributeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError('malformed generation result') from exc


def baseline_execution_row(
    plan: ExperimentPlan, inputs: BaselineInputs, key: RowKey, result: StructuredGenerationResult,
) -> ExecutionRow:
    """Bind the complete approved plan before converting an executed outcome."""
    if not isinstance(plan, ExperimentPlan) or not isinstance(key, RowKey):
        raise ValueError('expected ExperimentPlan and RowKey')
    try:
        rebuilt = build_baseline_plan(inputs, plan.identity)
        if rebuilt.to_json() != plan.to_json():
            raise ValueError('supplied plan differs from complete approved baseline plan')
        if (key.stage != 'baseline' or key.arm != 'baseline' or key.dose != '0'
                or key not in rebuilt.keys):
            raise ValueError('foreign baseline key')
        status = _validate_generation_result(result)
        issuer_id = next(member.issuer_id for member in inputs.members if member.ticker == key.ticker)
        outcome = GenerationOutcome(**(status.to_dict() | {
            'ticker': key.ticker, 'issuer_id': issuer_id,
            'condition': key.condition, 'trial_id': key.trial_id,
        }))
        return ExecutionRow(key, outcome)
    except (TypeError, AttributeError, UnicodeError, OverflowError, RecursionError, StopIteration) as exc:
        raise ValueError('malformed baseline invocation') from exc


def baseline_gate_input(result: StructuredGenerationResult) -> GateGenerationInput:
    """Preserve the entire continuation, including Harmony analysis and headers."""
    _validate_generation_result(result)
    return GateGenerationInput(
        result.generated_token_ids, result.generated_text, result.decision,
        result.schema_complete, result.reason_valid, result.decision_complete, result.failure_type,
    )
