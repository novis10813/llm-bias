"""Analytical fit-only DIM contracts, without a checkpoint or raw exports."""
from dataclasses import replace

import numpy as np
import pytest

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.experiment_contract import (
    ExecutionRow, ExperimentPlan, GenerationOutcome, PlanIdentity, RowKey,
)
from llm_bias.core.population import PopulationMember
from llm_bias.core.stance_dim_extraction import extract_dim_candidates, pool_original_span


@pytest.fixture
def case():
    members = [PopulationMember(f'{label}{i}', 'Company', 'Sector', f'{label}{i}')
               for label in ('B', 'S') for i in range(20)]
    members += [PopulationMember(r, 'Company', 'Sector', r)
                for r in ('V', 'C', 'E')]
    roles = dict(fit=[m.ticker for m in members[:-3]], validation=['V'],
                 calibration=['C'], evaluation=['E'])
    keys = tuple(RowKey('baseline', m.ticker, c, 'trial', 'baseline', '0')
                 for m in members for c in ('++', '+-', '-+', '--'))
    identity = PlanIdentity(*(['a' * 64] * 11), 'not_applicable', 'not_applicable')
    plan = ExperimentPlan(identity, keys, ('noop',))
    rows, vectors = [], {}
    for key in keys:
        if key.ticker not in roles['fit']:
            continue
        decision = 'buy' if key.ticker.startswith('B') else 'sell'
        rows.append(ExecutionRow(key, GenerationOutcome(
            key.ticker, key.ticker, key.condition, key.trial_id, decision,
            True, True, True, 'schema_complete', None)))
        vectors[key] = np.array([4., 3.]) if decision == 'buy' else np.array([1., 1.])
    return dict(plan=plan, members=members, roles=roles, rows=rows,
                pooled_residuals=vectors, layer=2, span='entity', d_model=2,
                parent_sha256='b' * 64, candidate_panel_sha256='c' * 64,
                capture_policy_sha256='d' * 64)


def run(case, **changes):
    return extract_dim_candidates(**(case | changes))


def candidate(result, condition='+-'):
    return next(c for c in result.to_dict()['candidates'] if c['condition'] == condition)


def test_hand_calculation_and_no_raw_export(case):
    result = run(case)
    assert [c['condition'] for c in result.to_dict()['candidates']] == ['+-', '-+', '++', '--']
    c = candidate(result)
    assert c['status'] == 'sufficient'
    assert c['direction'] == [3., 2.]
    assert c['direction_norm'] == pytest.approx(np.sqrt(13))
    assert c['coverage']['planned_rows'] == 40
    assert c['coverage']['contributing_issuers'] == {'buy': 20, 'sell': 20}
    data = result.to_dict()
    digest = data.pop('extraction_sha256')
    assert sha256_json(data) == digest
    assert 'pooled_residuals' not in str(data)
    assert 'preferred_condition' not in data


def test_span_mean_and_scale(case):
    assert pool_original_span(np.array([[100., 100.], [2., 4.], [4., 8.]]),
                              (1, 2), original_prompt_length=3, d_model=2) == (3., 6.)
    scaled = {k: v * 10 for k, v in case['pooled_residuals'].items()}
    assert candidate(run(case, pooled_residuals=scaled))['direction'] == [30., 20.]


@pytest.mark.parametrize('bad', [np.array([[1., 2.]]), np.array([np.nan, 1.]),
                                np.array([np.inf, 1.]), np.array([1.]),
                                np.array([True, False]), np.array([1+2j, 1])])
def test_invalid_vectors_reject(case, bad):
    vectors = dict(case['pooled_residuals']); vectors[next(iter(vectors))] = bad
    with pytest.raises(ValueError): run(case, pooled_residuals=vectors)


@pytest.mark.parametrize('positions,length', [((), 3), ((1, 1), 3), ((3,), 3),
                                               ((-1,), 3), ((True,), 3), ((1,), 2)])
def test_bad_original_span(positions, length):
    with pytest.raises(ValueError):
        pool_original_span(np.ones((3, 2)), positions, original_prompt_length=length, d_model=2)


def test_company_first_trials_and_share_class_invariance(case):
    # B0 has two trials: its company mean is [6, 5], not two votes.
    k = next(k for k in case['pooled_residuals'] if k.ticker == 'B0' and k.condition == '+-')
    extra = replace(k, trial_id='second')
    plan = replace(case['plan'], keys=case['plan'].keys + (extra,))
    row = next(r for r in case['rows'] if r.key == k)
    rows = case['rows'] + [ExecutionRow(extra, replace(row.outcome, trial_id='second'))]
    vectors = case['pooled_residuals'] | {extra: np.array([8., 7.])}
    result = run(case, plan=plan, rows=rows, pooled_residuals=vectors)
    assert candidate(result)['direction'] == pytest.approx([3.1, 2.1])
    # Duplicate B1 as another share class. Same issuer receives one vote.
    member = PopulationMember('B1A', 'Company', 'Sector', 'B1')
    newkeys = tuple(replace(key, ticker='B1A') for key in case['plan'].keys if key.ticker == 'B1')
    newrows = [ExecutionRow(replace(r.key, ticker='B1A'), replace(r.outcome, ticker='B1A'))
               for r in case['rows'] if r.key.ticker == 'B1']
    more = {key: np.array([4., 3.]) for key in newkeys}
    dup = run(case, members=case['members'] + [member],
              roles=case['roles'] | {'fit': case['roles']['fit'] + ['B1A']},
              plan=replace(case['plan'], keys=case['plan'].keys + newkeys),
              rows=case['rows'] + newrows, pooled_residuals=case['pooled_residuals'] | more)
    assert candidate(dup)['direction'] == candidate(run(case))['direction']
    assert candidate(dup)['coverage']['contributing_issuers']['buy'] == 20


def test_missing_failure_and_capture_coverage(case):
    k = next(k for k in case['pooled_residuals'] if k.condition == '+-')
    rows = [r for r in case['rows'] if r.key != k]
    vectors = {key: v for key, v in case['pooled_residuals'].items() if key != k}
    c = candidate(run(case, rows=rows, pooled_residuals=vectors))
    assert c['status'] == 'untestable'
    assert c['coverage']['missing_rows'] == 1
    assert c['coverage']['planned_rows'] == 40
    old = next(r for r in case['rows'] if r.key == k)
    failed = ExecutionRow(k, replace(old.outcome, decision=None, decision_complete=False,
        schema_complete=False, reason_valid=False, finish_reason='timeout', failure_type='timeout'))
    c = candidate(run(case, rows=rows + [failed], pooled_residuals=vectors))
    assert c['coverage']['unknown_rows'] == 1
    assert c['coverage']['failures'] == {'timeout': 1}
    c = candidate(run(case, pooled_residuals=vectors))
    assert c['coverage']['missing_capture_rows'] == 1
    assert c['coverage']['generated_rows']['buy'] == 20
    assert c['coverage']['contributing_issuers']['buy'] == 19


def test_no_classes_or_cross_condition_pooling(case):
    rows = [ExecutionRow(r.key, replace(r.outcome, decision='sell'))
            if r.key.condition == '+-' else r for r in case['rows']]
    result = run(case, rows=rows)
    c = candidate(result)
    assert c['status'] == 'untestable' and c['direction'] is None
    assert c['coverage']['generated_issuers']['buy'] == 0
    assert candidate(result, '-+')['status'] == 'sufficient'
    assert 'insufficient_buy_issuers' in c['reasons']


def test_zero_and_overflow_direction(case):
    vectors = {k: np.ones(2) for k in case['pooled_residuals']}
    assert candidate(run(case, pooled_residuals=vectors))['reasons'] == ['zero_direction']
    vectors = {k: np.array([1e308, 1e308]) * (1 if k.ticker.startswith('B') else -1)
               for k in vectors}
    with pytest.raises(ValueError): run(case, pooled_residuals=vectors)


@pytest.mark.parametrize('role', ['validation', 'calibration', 'evaluation'])
def test_teacher_role_leak_rejected(case, role):
    ticker = case['roles'][role][0]
    key = next(k for k in case['plan'].keys if k.ticker == ticker)
    row = ExecutionRow(key, GenerationOutcome(ticker, ticker, key.condition, key.trial_id,
        'buy', True, True, True, 'schema_complete', None))
    with pytest.raises(ValueError): run(case, rows=case['rows'] + [row])
    with pytest.raises(ValueError):
        run(case, pooled_residuals=case['pooled_residuals'] | {key: np.ones(2)})


def test_bindings_duplicates_immutable_provenance(case):
    result = run(case)
    old = result.to_dict()
    case['roles']['fit'].clear()
    case['pooled_residuals'][next(iter(case['pooled_residuals']))][:] = 99
    exported = result.to_dict(); exported['provenance']['layer'] = 99
    assert result.to_dict() == old
    with pytest.raises(Exception): result._payload_bytes = b'{}'


@pytest.mark.parametrize('bad', ['duplicate', 'issuer', 'foreign', 'hash', 'scale', 'shape'])
def test_bad_contract(case, bad):
    if bad == 'duplicate': case['rows'].append(case['rows'][0])
    elif bad == 'issuer':
        r = case['rows'][0]
        case['rows'][0] = ExecutionRow(r.key, replace(r.outcome, issuer_id='wrong'))
    elif bad == 'foreign':
        k = next(iter(case['pooled_residuals']))
        case['pooled_residuals'][replace(k, trial_id='foreign')] = np.ones(2)
    elif bad == 'hash': case['parent_sha256'] = 'bad'
    elif bad == 'scale': case['scale'] = 2
    else: case['d_model'] = True
    with pytest.raises((ValueError, TypeError)): run(case)


def test_empty_observations_are_honest(case):
    result = run(case, rows=[], pooled_residuals={})
    for c in result.to_dict()['candidates']:
        assert c['status'] == 'untestable'
        assert c['coverage']['missing_rows'] == 40
        assert c['coverage']['unknown_rows'] == 0
        assert c['coverage']['coverage_complete'] is False


def test_order_independence_and_teacher_binding(case):
    first = run(case)
    reordered = run(case, rows=case['rows'][::-1], members=case['members'][::-1],
                    pooled_residuals=dict(reversed(list(case['pooled_residuals'].items()))))
    assert first.to_dict() == reordered.to_dict()
    rows = [ExecutionRow(r.key, replace(r.outcome, decision='sell'))
            if r.key.ticker == 'B0' else r for r in case['rows']]
    assert run(case, rows=rows).extraction_sha256 != first.extraction_sha256


def test_nonuniform_share_classes_have_equal_issuer_weight(case):
    member = PopulationMember('B0A', 'Company', 'Sector', 'B0')
    keys = tuple(replace(k, ticker='B0A') for k in case['plan'].keys if k.ticker == 'B0')
    rows = [ExecutionRow(replace(r.key, ticker='B0A'), replace(r.outcome, ticker='B0A'))
            for r in case['rows'] if r.key.ticker == 'B0']
    result = run(case, members=case['members'] + [member],
        roles=case['roles'] | {'fit': case['roles']['fit'] + ['B0A']},
        plan=replace(case['plan'], keys=case['plan'].keys + keys),
        rows=case['rows'] + rows,
        pooled_residuals=case['pooled_residuals'] | {k: np.array([8., 7.]) for k in keys})
    # B0 issuer averages [4,3] and [8,7] to [6,5]. Twenty issuer votes.
    assert candidate(result)['direction'] == pytest.approx([3.1, 2.1])


def test_full_member_plan_and_issuer_role_crossing_reject(case):
    plan = replace(case['plan'], keys=tuple(k for k in case['plan'].keys if k.ticker != 'B0'))
    with pytest.raises(ValueError): run(case, plan=plan)
    members = [replace(m, issuer_id='B0') if m.ticker == 'V' else m for m in case['members']]
    with pytest.raises(ValueError): run(case, members=members)
