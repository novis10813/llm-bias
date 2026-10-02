"""Full-population pair planning; labels classify but never choose donors."""
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.experiment_contract import ExecutionRow
from llm_bias.core.stance_baseline_parent import CompletedBaseline, load_completed_baseline
from llm_bias.core.stance_localization_pairs import (
    LocalizationPair, LocalizationPairTable, build_localization_pairs, pairing_policy,
)
from test_stance_baseline_parent import parent_bundle, _records_for, _summary_for
from test_stance_baseline_adapter import bundle


@pytest.fixture(scope='module')
def fixture(parent_bundle):
    inputs = parent_bundle['inputs']
    _, rows, _ = _records_for(parent_bundle, {})
    parent = CompletedBaseline(
        parent_bundle['plan'], tuple(rows.values()), 'a' * 64,
        canonical_json_bytes(parent_bundle['metadata']) + b'\n',
        canonical_json_bytes(_summary_for(parent_bundle['plan'], rows)) + b'\n',
        b'{}', MappingProxyType({}),
    )
    return inputs, parent


def with_rows(parent, rows):
    return replace(parent, rows=tuple(rows), _summary_bytes=canonical_json_bytes(
        _summary_for(parent.plan, {r.key: r for r in rows})) + b'\n')


def test_full_table(fixture):
    inputs, parent = fixture
    table = build_localization_pairs(inputs, parent)
    assert len(table.pairs) == 4024
    assert {p.target_key for p in table.pairs} == set(parent.plan.keys)
    counts = {role: sum(p.role == role for p in table.pairs)
              for role in ('fit', 'validation', 'calibration', 'evaluation')}
    assert counts == dict(fit=2416, validation=600, calibration=208, evaluation=800)
    for p in table.pairs:
        assert inputs.roles['assignments'][p.target_key.ticker] == p.role
        assert inputs.roles['assignments'][p.donor_key.ticker] == p.role
        if p.family == 'cross_company':
            assert inputs.issuer_by_ticker[p.target_key.ticker] != inputs.issuer_by_ticker[p.donor_key.ticker]
            assert p.target_key.condition == p.donor_key.condition
        else:
            assert p.target_key.ticker == p.donor_key.ticker
            assert p.target_key.condition != p.donor_key.condition
    assert table == build_localization_pairs(inputs, replace(parent, rows=parent.rows[::-1]))


def test_donor_choices_ignore_labels_and_preserve_all_rows(fixture):
    inputs, parent = fixture
    old = build_localization_pairs(inputs, parent)
    rows = [ExecutionRow(r.key, replace(r.outcome, decision='buy')) for r in parent.rows]
    same = build_localization_pairs(inputs, with_rows(parent, rows))
    assert all(p.clean_relation == 'same' for p in same.pairs)
    assert [(p.target_key, p.donor_key) for p in old.pairs] == [(p.target_key, p.donor_key) for p in same.pairs]
    assert len(same.pairs) == 4024
    rows[0] = ExecutionRow(rows[0].key, replace(rows[0].outcome,
        decision=None, decision_complete=False, schema_complete=False, reason_valid=False,
        finish_reason='timeout', failure_type='timeout'))
    invalid = build_localization_pairs(inputs, with_rows(parent, rows))
    assert len(invalid.pairs) == 4024
    assert any(p.clean_relation == 'invalid_parent' for p in invalid.pairs)


def test_ring_known_issuer_and_lexicographic_donor(fixture):
    inputs, parent = fixture
    table = build_localization_pairs(inputs, parent)
    target = next(m for m in inputs.members if m.ticker == 'GOOG')
    role = inputs.roles['assignments'][target.ticker]
    groups = {}
    for m in inputs.members:
        if inputs.roles['assignments'][m.ticker] == role:
            groups.setdefault(m.issuer_id, []).append(m.ticker)
    ring = sorted(groups, key=lambda issuer: (sha256_json(
        {'seed': 20261002, 'role': role, 'issuer_id': issuer}), issuer))
    donor = min(groups[ring[(ring.index(target.issuer_id) + 1) % len(ring)]])
    selected = [p for p in table.pairs if p.family == 'cross_company'
                and p.target_key.ticker in ('GOOG', 'GOOGL')]
    assert len(selected) == 8
    assert {p.donor_key.ticker for p in selected} == {donor}
    assert all(p.donor_key.ticker not in ('GOOG', 'GOOGL') for p in selected)


def test_evidence_involution_and_hashes(fixture):
    inputs, parent = fixture
    table = build_localization_pairs(inputs, parent)
    lookup = {(p.target_key, p.family): p for p in table.pairs}
    for p in table.pairs:
        data = p.to_dict(); digest = data.pop('pair_sha256')
        assert sha256_json(data) == digest
        if p.family == 'cross_evidence':
            assert lookup[p.donor_key, p.family].donor_key == p.target_key
            assert p.contrast == ('polarity_content' if p.target_key.condition in ('++', '--') else 'evidence_order')
    data = table.to_dict(); digest = data.pop('table_sha256')
    assert sha256_json(data) == digest
    assert table.policy_sha256 == sha256_json(pairing_policy())
    changed = build_localization_pairs(inputs, replace(parent, parent_sha256='b' * 64))
    assert changed.policy_sha256 == table.policy_sha256
    assert changed.table_sha256 != table.table_sha256
    exported = table.to_dict(); exported['pairs'][0]['role'] = 'broken'
    assert table.to_dict()['pairs'][0]['role'] != 'broken'
    policy = pairing_policy(); policy['seed'] = 0
    assert pairing_policy()['seed'] == 20261002


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'issuer', 'key', 'summary', 'hash', 'plan'])
def test_bad_parent_rejected(fixture, change):
    inputs, parent = fixture
    rows = list(parent.rows)
    if change == 'missing': parent = with_rows(parent, rows[:-1])
    elif change == 'duplicate': parent = replace(parent, rows=tuple(rows[:-1] + [rows[0]]))
    elif change == 'issuer':
        rows[0] = ExecutionRow(rows[0].key, replace(rows[0].outcome, issuer_id='foreign'))
        parent = with_rows(parent, rows)
    elif change == 'key':
        rows[0] = ExecutionRow(replace(rows[0].key, arm='foreign'), rows[0].outcome)
        parent = replace(parent, rows=tuple(rows))
    elif change == 'summary': parent = replace(parent, _summary_bytes=b'{"research_eligible":true}\n')
    elif change == 'hash': parent = replace(parent, parent_sha256='bad')
    else: parent = replace(parent, plan=replace(parent.plan, keys=parent.plan.keys[:-1]))
    with pytest.raises(ValueError): build_localization_pairs(inputs, parent)


def test_plan_rebuilt_once(fixture, monkeypatch):
    import llm_bias.core.stance_localization_pairs as module
    original = module.build_baseline_plan; calls = []
    def counted(*args):
        calls.append(1); return original(*args)
    monkeypatch.setattr(module, 'build_baseline_plan', counted)
    build_localization_pairs(*fixture)
    assert len(calls) == 1


@pytest.mark.parametrize('field,value', [('role', 'other'), ('family', 'other'),
    ('contrast', 'other'), ('clean_relation', 'other'), ('pair_sha256', 'a' * 64)])
def test_bad_pair(fixture, field, value):
    pair = build_localization_pairs(*fixture).pairs[0]
    with pytest.raises(ValueError): replace(pair, **{field: value})


def test_bad_table(fixture):
    table = build_localization_pairs(*fixture)
    for updates in ({'pairs': table.pairs[::-1]}, {'pairs': (table.pairs[0],) * 2},
                    {'table_sha256': 'a' * 64}, {'policy_sha256': 'bad'}, {'pairs': list(table.pairs)}):
        with pytest.raises(ValueError): replace(table, **updates)
    with pytest.raises(ValueError): build_localization_pairs(None, fixture[1])


def test_actual_glm_optional():
    from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
    root = Path('artifacts/glm4-9b-0414/concept-cone-steering/runs/diagnostic-baseline-resumed-stance-rb260930-18')
    if not root.exists(): pytest.skip('ignored actual GLM artifact not present')
    inputs = load_baseline_inputs('data/concept-cone-steering/rebuild-v1/compiled')
    parent = load_completed_baseline(root, inputs=inputs)
    table = build_localization_pairs(inputs, parent)
    assert len(table.pairs) == 4024
    assert sum(p.clean_relation == 'opposite' for p in table.pairs) == 1772
    assert sum(p.clean_relation == 'same' for p in table.pairs) == 2252
