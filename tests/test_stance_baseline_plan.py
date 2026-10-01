"""Full approved baseline planning, without model execution or certification."""
import copy
import shutil
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.experiment_contract import ExperimentPlan, PlanIdentity, RowKey, progress, shard_index_for_key
from llm_bias.core.stance_baseline_inputs import BaselineInputs, load_baseline_inputs
from llm_bias.core.stance_baseline_plan import build_baseline_plan

PACK = Path(__file__).resolve().parents[1] / 'data/concept-cone-steering/rebuild-v1/compiled'


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp('approved-plan-inputs')
    for path in PACK.iterdir():
        shutil.copyfile(path, directory / path.name)
    return load_baseline_inputs(directory)


def synthetic_identity(inputs):
    values = {f.name: sha256_json('synthetic unverified ' + f.name) for f in fields(PlanIdentity)}
    values.update(population_sha256=inputs.membership_sha256, issuer_sha256=inputs.issuer_mapping_sha256,
                  roles_sha256=inputs.roles_sha256, evidence_sha256=inputs.evidence_sha256,
                  operator_sha256='not_applicable', parent_sha256='not_applicable')
    return PlanIdentity(**values)


def test_full_plan(inputs):
    identity = synthetic_identity(inputs)
    plan = build_baseline_plan(inputs, identity)
    expected = {RowKey('baseline', p.ticker, p.condition, p.trial_id, 'baseline', '0') for p in inputs.pairs}
    assert len(plan.keys) == 2012 and set(plan.keys) == expected
    assert {k.ticker for k in plan.keys} == set(inputs.roles['assignments'])
    assert {inputs.roles['assignments'][k.ticker] for k in plan.keys} == {'fit', 'validation', 'calibration', 'evaluation'}
    assert plan.gate_names == ('input_integrity', 'model_binding', 'no_op', 'protocol_frozen')
    assert plan.identity == identity and plan.identity is not identity
    assert ExperimentPlan.from_json(plan.to_json()) == plan
    state = progress(plan, ())
    assert not state['eligible'] and all(v is None for v in state['gates'].values())
    shards = [plan.keys_for_shard(i, 7) for i in range(7)]
    assert sum(map(len, shards)) == 2012
    assert set().union(*map(set, shards)) == expected
    assert shards == [plan.keys_for_shard(i, 7) for i in range(7)]
    # More shards than keys guarantees empty registered shards without a costly all-shard scan.
    occupied = {shard_index_for_key(k, 2013) for k in plan.keys}
    assert plan.keys_for_shard(next(i for i in range(2013) if i not in occupied), 2013) == ()


@pytest.mark.parametrize('field', ['population_sha256', 'issuer_sha256', 'roles_sha256', 'evidence_sha256',
                                   'operator_sha256', 'parent_sha256'])
def test_identity_mismatch(inputs, field):
    with pytest.raises(ValueError):
        build_baseline_plan(inputs, replace(synthetic_identity(inputs), **{field: '0' * 64}))


def test_forged_identity_revalidated(inputs):
    identity = synthetic_identity(inputs)
    object.__setattr__(identity, 'model_sha256', 'not-a-hash')
    with pytest.raises(ValueError):
        build_baseline_plan(inputs, identity)
    with pytest.raises(ValueError):
        build_baseline_plan(inputs, object())


@pytest.mark.parametrize('change', ['empty', 'truncated', 'duplicate', 'foreign', 'pair-list', 'member-list',
    'member-empty', 'member-duplicate', 'member-dict', 'pair-dict', 'item-dict', 'item-list-field',
    'member-nonstring', 'pair-nonstring', 'roles', 'manifest', 'roles-storage', 'manifest-storage',
    'membership_sha256', 'issuer_mapping_sha256', 'roles_sha256', 'evidence_sha256', 'manifest_sha256'])
def test_direct_forges_rejected(inputs, change):
    changes = {}
    if change == 'empty': changes['pairs'] = ()
    elif change == 'truncated': changes['pairs'] = inputs.pairs[:-1]
    elif change == 'duplicate': changes['pairs'] = inputs.pairs[:-1] + (inputs.pairs[0],)
    elif change == 'foreign': changes['pairs'] = (replace(inputs.pairs[0], ticker='FOREIGN'),) + inputs.pairs[1:]
    elif change == 'pair-list': changes['pairs'] = list(inputs.pairs)
    elif change == 'member-list': changes['members'] = list(inputs.members)
    elif change == 'member-empty': changes['members'] = ()
    elif change == 'member-duplicate': changes['members'] = inputs.members[:-1] + (inputs.members[0],)
    elif change == 'member-dict': changes['members'] = (asdict(inputs.members[0]),) + inputs.members[1:]
    elif change == 'pair-dict': changes['pairs'] = (asdict(inputs.pairs[0]),) + inputs.pairs[1:]
    elif change in ('item-dict', 'item-list-field', 'pair-nonstring'):
        p = inputs.pairs[0]
        p = replace(p, evidence1=asdict(p.evidence1)) if change == 'item-dict' else (
            replace(p, evidence1=replace(p.evidence1, text=['mutable'])) if change == 'item-list-field'
            else replace(p, ticker=1))
        changes['pairs'] = (p,) + inputs.pairs[1:]
    elif change == 'member-nonstring': changes['members'] = (replace(inputs.members[0], name=1),) + inputs.members[1:]
    elif change == 'roles':
        roles = inputs.roles
        roles['assignments'][inputs.members[0].ticker] = 'evaluation'
        changes['_roles_bytes'] = canonical_json_bytes(roles) + b'\n'
    elif change == 'manifest': changes['_manifest_bytes'] = b'{}\n'
    elif change == 'roles-storage': changes['_roles_bytes'] = bytearray(inputs._roles_bytes)
    elif change == 'manifest-storage': changes['_manifest_bytes'] = b'\xff'
    else: changes[change] = '0' * 64
    with pytest.raises(ValueError):
        build_baseline_plan(replace(inputs, **changes), synthetic_identity(inputs))


def test_full_count_self_consistent_forgery(inputs):
    pairs = tuple(replace(p, evidence1=replace(p.evidence1, text='Forged evidence',
                  content_sha256=sha256_bytes(b'Forged evidence'))) for p in inputs.pairs)
    evidence_hash = sha256_bytes(b''.join(canonical_json_bytes(asdict(p)) + b'\n' for p in pairs))
    forged = replace(inputs, pairs=pairs, evidence_sha256=evidence_hash)
    assert len(forged.pairs) == 2012
    with pytest.raises(ValueError):
        build_baseline_plan(forged, synthetic_identity(forged))
    manifest = forged.manifest
    manifest['output_hashes']['evidence_pairs.jsonl'] = evidence_hash
    raw = canonical_json_bytes(manifest) + b'\n'
    forged = replace(forged, _manifest_bytes=raw, manifest_sha256=sha256_bytes(raw))
    with pytest.raises(ValueError):
        build_baseline_plan(forged, synthetic_identity(forged))


def test_exact_defensive_copy(inputs):
    copied = BaselineInputs(**{f.name: copy.deepcopy(getattr(inputs, f.name)) for f in fields(BaselineInputs)})
    assert build_baseline_plan(copied, synthetic_identity(copied)) == build_baseline_plan(inputs, synthetic_identity(inputs))
    with pytest.raises(ValueError):
        build_baseline_plan(object(), synthetic_identity(inputs))


def test_import_safety():
    script = '''
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'jlens'} or fullname.startswith('llm_bias.core.inference'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from llm_bias.core.stance_baseline_plan import build_baseline_plan
assert not any(n.split('.')[0] in {'torch', 'transformers', 'jlens'} for n in sys.modules)
'''
    subprocess.run([sys.executable, '-c', script], check=True)
