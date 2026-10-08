"""Pinned production pack and independent semantic boundary tests (no models)."""
import copy
import json
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from llm_bias.core import stance_baseline_inputs as loader
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.population import PopulationMember, assign_roles

PACK = Path(__file__).resolve().parents[1] / 'data/concept-cone-steering/rebuild-v1/compiled'


@pytest.fixture
def pack(tmp_path):
    for path in PACK.iterdir():
        shutil.copyfile(path, tmp_path / path.name)
    return tmp_path


def documents():
    return (json.loads((PACK / 'population.json').read_bytes()),
            json.loads((PACK / 'roles.json').read_bytes()),
            [json.loads(line) for line in (PACK / 'evidence_pairs.jsonl').read_bytes().splitlines()],
            json.loads((PACK / 'inputs_manifest.json').read_bytes()))


def test_valid_and_immutable(pack):
    bundle = loader.load_baseline_inputs(pack)
    assert len(bundle.members) == 503
    assert len(set(bundle.issuer_by_ticker.values())) == 500
    assert len(bundle.pairs) == 2012
    assert sum(m.sector == 'Unspecified' for m in bundle.members) == 29
    assert bundle.manifest['eligibility']['research_eligible'] is False
    assert bundle.roles['ticker_counts'] == dict(fit=302, validation=75, calibration=26, evaluation=100)
    assert bundle.roles['assignments']['NWS'] == bundle.roles['assignments']['NWSA'] == 'calibration'
    for name, attr in [('membership_hash', 'membership_sha256'), ('roles_hash', 'roles_sha256'),
                       ('issuer_mapping_hash', 'issuer_mapping_sha256')]:
        assert getattr(bundle, attr) == bundle.manifest['semantic_hashes'][name]
    assert bundle.evidence_sha256 == sha256_bytes((pack / 'evidence_pairs.jsonl').read_bytes())
    assert bundle.manifest_sha256 == loader._MANIFEST_SHA256
    for pair in bundle.pairs:
        assert bundle.pair_for(pair.ticker, pair.condition, pair.trial_id) == pair
    for args in [('FOREIGN', '++', 'factset-20241115-a'), ('A', 'bad', 'factset-20241115-a'), ('A', '++', 'bad')]:
        with pytest.raises(ValueError):
            bundle.pair_for(*args)
    bundle.roles['roles']['fit'].clear()
    bundle.manifest['counts'].clear()
    bundle.issuer_by_ticker.clear()
    assert len(bundle.roles['roles']['fit']) == 302
    assert bundle.manifest['counts']['population_count'] == 503
    assert len(bundle.issuer_by_ticker) == 503
    for obj, field, value in [(bundle, 'members', ()), (bundle.members[0], 'name', 'x'),
                              (bundle.pairs[0], 'ticker', 'x'), (bundle.pairs[0].evidence1, 'text', 'x')]:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(obj, field, value)


@pytest.mark.parametrize('name', ['population.json', 'roles.json', 'evidence_pairs.jsonl', 'inputs_manifest.json'])
@pytest.mark.parametrize('change', ['tamper', 'missing', 'symlink', 'directory'])
def test_files_rejected(pack, name, change):
    path = pack / name
    if change == 'tamper':
        path.write_bytes(path.read_bytes() + b' ')
    else:
        path.unlink()
        if change == 'symlink':
            path.symlink_to(PACK / name)
        elif change == 'directory':
            path.mkdir()
    with pytest.raises((ValueError, OSError)):
        loader.load_baseline_inputs(pack)


def test_extra_and_directory_link(pack, tmp_path):
    (pack / 'extra').write_text('x')
    with pytest.raises(ValueError):
        loader.load_baseline_inputs(pack)
    (pack / 'extra').unlink()
    link = tmp_path.parent / (tmp_path.name + '-link')
    link.symlink_to(pack, target_is_directory=True)
    with pytest.raises(ValueError):
        loader.load_baseline_inputs(link)


@pytest.mark.parametrize('raw', [b'\xff\n', b'{"a":1,"a":2}\n', b'{"a":NaN}\n',
                                  b'{"b":1,"a":2}\n', b'{}', b'{}\n\n'])
def test_strict_parser(raw):
    with pytest.raises(ValueError):
        loader._parse_json(raw)


@pytest.mark.parametrize('change', ['members', 'sector', 'issuer', 'roles', 'metadata', 'swap', 'coverage', 'extra'])
def test_semantic_rejections(change):
    population, roles, pairs, manifest = documents()
    if change == 'members':
        population['members'].pop()
    elif change == 'sector':
        population['members'][0]['sector'] = 'Unspecified'
    elif change == 'issuer':
        population['members'][0]['issuer_id'] = 'foreign'
    elif change == 'roles':
        roles['ticker_counts']['fit'] = 301
    elif change == 'metadata':
        for pair in pairs:
            pair['evidence1']['review_id'] = 'foreign'
    elif change == 'swap':
        for pair in pairs:
            if pair['condition'] == '-+':
                pair['evidence2'] = copy.deepcopy(next(p['evidence2'] for p in pairs if p['condition'] == '++'))
    elif change == 'coverage':
        pairs.pop()
    else:
        pairs[0]['evidence1']['extra'] = 'x'
    with pytest.raises(ValueError):
        loader._validate_documents(population, roles, pairs, manifest)


@pytest.mark.parametrize('ticker', ['A', 'FOX', 'GOOG', 'NWS'])
def test_exact_issuer_identity_even_with_consistent_hashes_and_roles(ticker):
    population, roles, pairs, manifest = documents()
    old = next(m['issuer_id'] for m in population['members'] if m['ticker'] == ticker)
    for member in population['members']:
        if member['issuer_id'] == old:
            member['issuer_id'] = old + '!'
    population['membership_sha256'] = sha256_json(population['members'])
    manifest['semantic_hashes']['membership_hash'] = population['membership_sha256']
    manifest['semantic_hashes']['issuer_mapping_hash'] = sha256_json(
        {m['ticker']: m['issuer_id'] for m in population['members']})
    roles = asdict(assign_roles(
        [PopulationMember(**m) for m in population['members']], 20260930,
        dict(fit=300, validation=75, calibration=25, evaluation=100)))
    manifest['semantic_hashes']['roles_hash'] = roles['assignment_hash']
    assert roles['ticker_counts'] == dict(fit=302, validation=75, calibration=26, evaluation=100)
    with pytest.raises(ValueError, match='exact registered issuer mapping'):
        loader._validate_documents(population, roles, pairs, manifest)


def test_import_safety():
    script = '''
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'jlens'} or fullname == 'scripts.compile_stance_inputs':
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
load_baseline_inputs(sys.argv[1])
assert not any(n.split('.')[0] in {'torch', 'transformers', 'jlens'} for n in sys.modules)
'''
    subprocess.run([sys.executable, '-c', script, str(PACK)], check=True)


def test_full_synthetic_semantics_are_not_production_authorization(pack):
    # Replace all display names without narrowing the full cohort or changing issuers.
    population, roles, pairs, manifest = documents()
    for index, member in enumerate(population['members']):
        member['name'] = f'Synthetic company {index}'
    digest = sha256_json(population['members'])
    population['membership_sha256'] = digest
    manifest['semantic_hashes']['membership_hash'] = digest
    members, validated = loader._validate_documents(population, roles, pairs, manifest)
    assert len(members) == 503 and len(validated) == 2012
    payload = canonical_json_bytes(population) + b'\n'
    (pack / 'population.json').write_bytes(payload)
    manifest['output_hashes']['population.json'] = sha256_bytes(payload)
    (pack / 'inputs_manifest.json').write_bytes(canonical_json_bytes(manifest) + b'\n')
    with pytest.raises(ValueError, match='manifest raw SHA-256 mismatch'):
        loader.load_baseline_inputs(pack)


@pytest.mark.parametrize('name,raw', [
    ('population.json', b'\xff\n'),
    ('roles.json', b'{"roles":{},"roles":{}}\n'),
    ('evidence_pairs.jsonl', b'\n'),
    ('population.json', b'{"z":1,"a":2}\n'),
])
def test_malformed_production_bytes_hit_pin(pack, name, raw):
    (pack / name).write_bytes(raw)
    with pytest.raises(ValueError, match='raw SHA-256 mismatch'):
        loader.load_baseline_inputs(pack)


@pytest.mark.parametrize('location', ['population', 'member', 'roles', 'role-map', 'role-counts',
                                      'manifest', 'counts', 'semantic', 'outputs', 'code',
                                      'eligibility', 'provenance', 'pair', 'item'])
def test_exact_nested_field_sets(location):
    population, roles, pairs, manifest = documents()
    targets = {
        'population': population, 'member': population['members'][0], 'roles': roles,
        'role-map': roles['roles'], 'role-counts': roles['ticker_counts'],
        'manifest': manifest, 'counts': manifest['counts'], 'semantic': manifest['semantic_hashes'],
        'outputs': manifest['output_hashes'], 'code': manifest['code_sha256'],
        'eligibility': manifest['eligibility'], 'provenance': manifest['provenance'],
        'pair': pairs[0], 'item': pairs[0]['evidence1'],
    }
    targets[location]['unexpected'] = 'x'
    with pytest.raises(ValueError):
        loader._validate_documents(population, roles, pairs, manifest)
