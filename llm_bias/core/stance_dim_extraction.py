"""Fit-only generated-label DIM math. Captures remain transient caller inputs.

The caller supplies an approved full baseline plan and authenticated generated
outcomes. This module validates their structural bindings, not a live model or
source files. It never chooses a condition, site, scale, or validation winner.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from .artifact_paths import canonical_json_bytes, sha256_json
from .experiment_contract import ExecutionRow, ExperimentPlan, RowKey, validate_outcome
from .population import validate_members, validate_roles

CONDITION_PRIORITY = ('+-', '-+', '++', '--')
MIN_ISSUERS_PER_CLASS = 20
_SPANS = ('entity', 'evidence1', 'evidence2', 'instruction')


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected lowercase SHA-256')


def _array(value, shape):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('expected a real finite numeric array') from exc
    if array.shape != shape or array.dtype.kind not in 'fiu':
        raise ValueError(f'expected real numeric shape {shape}, not bool/complex/object')
    with np.errstate(over='ignore', invalid='ignore'):
        array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError('array must be finite in float64')
    return array


def _mean(vectors, d_model):
    # Divide before summing to avoid overflow of a finite arithmetic mean.
    n = len(vectors)
    try:
        mean = tuple(math.fsum(v[j] / n for v in vectors) for j in range(d_model))
    except OverflowError as exc:
        raise ValueError('nonfinite derived mean') from exc
    if not all(math.isfinite(v) for v in mean):
        raise ValueError('nonfinite derived mean')
    return mean


def pool_original_span(
    residuals, positions, *, original_prompt_length: int, d_model: int,
) -> tuple[float, ...]:
    """Memory-only mean of declared absolute original-prompt token positions.

    Requires precisely [original_prompt_length,d_model], with unique in-bounds
    positions. No generated prefix, batch dimension, implicit slicing or export.
    The returned pooled residual is transient input, not an artifact record.
    """
    _positive_int(original_prompt_length, 'original_prompt_length')
    _positive_int(d_model, 'd_model')
    array = _array(residuals, (original_prompt_length, d_model))
    if isinstance(positions, (str, bytes, Mapping)):
        raise ValueError('positions must be a nonempty integer sequence')
    try:
        positions = tuple(positions)
    except TypeError as exc:
        raise ValueError('positions must be iterable') from exc
    if (not positions or any(type(p) is not int or not 0 <= p < original_prompt_length
                             for p in positions) or len(set(positions)) != len(positions)):
        raise ValueError('positions must be unique original-prompt indices')
    return _mean([array[p] for p in sorted(positions)], d_model)


@dataclass(frozen=True, slots=True)
class DIMExtraction:
    """Immutable canonical bytes of derived vectors, compact coverage and bindings."""

    _payload_bytes: bytes = field(repr=False)

    def to_dict(self):
        # Each export is independent. No arrays or caller-owned mappings survive.
        return json.loads(self._payload_bytes)

    @property
    def extraction_sha256(self):
        return self.to_dict()['extraction_sha256']


def extract_dim_candidates(
    *, plan: ExperimentPlan, members, roles, rows, pooled_residuals: Mapping,
    layer: int, span: str, d_model: int, parent_sha256: str,
    candidate_panel_sha256: str, capture_policy_sha256: str,
) -> DIMExtraction:
    """Return four separate buy-minus-sell candidates for one declared layer/span.

    Only fit baseline rows and fit captures are accepted. Missing planned rows,
    primary-invalid generations and absent captures remain explicit coverage.
    No margin, scale, teacher-text, method-result or condition-selection input.
    Full-population approval and capture authenticity remain caller obligations.
    """
    if not isinstance(plan, ExperimentPlan):
        raise ValueError('expected ExperimentPlan')
    plan = ExperimentPlan.from_dict(plan.to_dict())
    members = validate_members(members)
    assignments = validate_roles(members, roles)
    if type(layer) is not int or layer < 0 or span not in _SPANS:
        raise ValueError('expected a declared nonnegative layer and original prompt span')
    _positive_int(d_model, 'd_model')
    for digest in (parent_sha256, candidate_panel_sha256, capture_policy_sha256):
        _digest(digest)
    issuers = {m.ticker: m.issuer_id for m in members}
    if any((k.stage, k.arm, k.dose) != ('baseline', 'baseline', '0')
           or k.ticker not in issuers for k in plan.keys):
        raise ValueError('plan must be a full-membership baseline plan')
    if {(k.ticker, k.condition) for k in plan.keys} != {
            (ticker, condition) for ticker in issuers for condition in CONDITION_PRIORITY}:
        raise ValueError('plan must account for every member in every condition')
    fit_keys = {k for k in plan.keys if assignments.assignments[k.ticker] == 'fit'}
    outcomes = {}
    for row in rows:
        if not isinstance(row, ExecutionRow) or row.key not in fit_keys:
            raise ValueError('generated teachers must be planned fit baseline rows only')
        if row.key in outcomes:
            raise ValueError('duplicate generated teacher')
        outcome = validate_outcome(row.outcome)
        expected = (row.key.ticker, issuers[row.key.ticker], row.key.condition, row.key.trial_id)
        if (outcome.ticker, outcome.issuer_id, outcome.condition, outcome.trial_id) != expected:
            raise ValueError('teacher issuer/key binding differs')
        outcomes[row.key] = outcome
    if not isinstance(pooled_residuals, Mapping):
        raise ValueError('pooled_residuals must be a transient mapping')
    vectors = {}
    for key, value in pooled_residuals.items():
        if type(key) is not RowKey or key not in fit_keys:
            raise ValueError('capture must belong to a planned fit baseline row')
        if key not in outcomes:
            raise ValueError('capture without an executed generated teacher')
        # Validate even invalid-label captures. Never silently hide malformed math.
        vectors[key] = _array(value, (d_model,))

    candidates = []
    for condition in CONDITION_PRIORITY:
        planned = sorted(k for k in fit_keys if k.condition == condition)
        generated = {label: [] for label in ('buy', 'sell')}
        companies = {label: defaultdict(list) for label in generated}
        failures = Counter()
        missing = unknown = missing_capture = 0
        for key in planned:
            outcome = outcomes.get(key)
            if outcome is None:
                missing += 1
            elif not outcome.primary_valid:
                unknown += 1
                failures[outcome.failure_type] += 1
            else:
                generated[outcome.decision].append(key)
                if key not in vectors:
                    missing_capture += 1
                else:
                    companies[outcome.decision][key.ticker].append(vectors[key])
        issuer_vectors = {label: defaultdict(list) for label in generated}
        for label in generated:
            for ticker in sorted(companies[label]):
                issuer_vectors[label][issuers[ticker]].append(
                    _mean(companies[label][ticker], d_model))
        class_means = {}
        for label in generated:
            if issuer_vectors[label]:
                class_means[label] = _mean([
                    _mean(issuer_vectors[label][issuer], d_model)
                    for issuer in sorted(issuer_vectors[label])], d_model)
        counts = {label: len(issuer_vectors[label]) for label in generated}
        reasons = [f'insufficient_{label}_issuers' for label in generated
                   if counts[label] < MIN_ISSUERS_PER_CLASS]
        direction = norm = None
        if not reasons:
            delta = tuple(b - s for b, s in zip(class_means['buy'], class_means['sell']))
            norm = math.hypot(*delta)
            if not all(math.isfinite(v) for v in delta) or not math.isfinite(norm):
                raise ValueError('nonfinite derived direction/norm')
            if norm == 0:
                reasons.append('zero_direction')
                norm = None
            else:
                direction = delta
        generated_issuers = {
            label: {issuers[k.ticker] for k in generated[label]} for label in generated}
        contributing_rows = sum(len(vs) for group in companies.values() for vs in group.values())
        candidates.append(dict(
            condition=condition, status='untestable' if reasons else 'sufficient',
            reasons=reasons, direction=direction, direction_norm=norm,
            coverage=dict(
                coverage_complete=(missing == 0 and missing_capture == 0),
                planned_rows=len(planned), planned_companies=len({k.ticker for k in planned}),
                planned_issuers=len({issuers[k.ticker] for k in planned}),
                executed_rows=len(planned) - missing, missing_rows=missing,
                unknown_rows=unknown, failures=dict(sorted(failures.items())),
                missing_capture_rows=missing_capture, contributing_rows=contributing_rows,
                generated_rows={label: len(generated[label]) for label in generated},
                generated_companies={label: len({k.ticker for k in generated[label]})
                                     for label in generated},
                generated_issuers={label: len(generated_issuers[label]) for label in generated},
                contributing_companies={label: len(companies[label]) for label in generated},
                contributing_issuers=counts,
                generated_class_issuer_overlap=len(generated_issuers['buy'] & generated_issuers['sell']),
                contributing_class_issuer_overlap=len(set(issuer_vectors['buy']) & set(issuer_vectors['sell'])),
            )))
    payload = dict(schema_version=1, kind='stance_dim_extraction_v1',
        provenance=dict(
            plan_hash=plan.plan_hash, plan_identity=plan.identity.to_dict(),
            parent_sha256=parent_sha256, candidate_panel_sha256=candidate_panel_sha256,
            capture_policy_sha256=capture_policy_sha256,
            membership_sha256=sha256_json([asdict(m) for m in members]),
            roles_sha256=assignments.assignment_hash,
            fit_keys_sha256=sha256_json([k.to_dict() for k in sorted(fit_keys)]),
            teacher_rows_sha256=sha256_json([
                {'key': k.to_dict(), 'outcome': outcomes[k].to_dict()} for k in sorted(outcomes)]),
            capture_keys_sha256=sha256_json([k.to_dict() for k in sorted(vectors)]),
            layer=layer, span=span, d_model=d_model,
            pooling='original_prompt_span_mean_then_company_then_issuer_then_class',
            sign='buy_minus_sell', normalization='none',
            label_source='primary_valid_generated_fit_baseline',
            min_distinct_issuers_per_class=MIN_ISSUERS_PER_CLASS,
            condition_priority=CONDITION_PRIORITY,
        ), candidates=candidates)
    payload['extraction_sha256'] = sha256_json(payload)
    return DIMExtraction(canonical_json_bytes(payload))
