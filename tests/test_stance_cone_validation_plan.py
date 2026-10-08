"""Complete-role CPU fixtures. No checkpoint, outcomes, or CUDA."""
from dataclasses import replace
import json

import pytest
import torch

from test_stance_baseline_adapter import bundle
from llm_bias.core import stance_cone_validation_plan as compiler
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b'\n')


@pytest.fixture
def saved(tmp_path, bundle, monkeypatch):
    inputs, plan = bundle
    root = tmp_path / 'training'
    audit = dict(kind='stance_cone_training_grid_acceptance_v1', status='passed',
        training_completed=True, grid_count=6, completed_steps=1812, authenticated_teachers=906,
        fit_tickers_per_run=302, accepted_operator=False, research_eligible=False,
        generated_validation_completed=False, teacher_content_freshly_authenticated=True,
        training_artifact_files_unchanged=True, training_backend_and_model_bindings_verified=True,
        training_source_inventory_and_hashes_verified=True, parent_sha256='a' * 64,
        compiler_provenance_discrepancies=[dict(path='compiler', recorded_git_head='old',
                                             matching_later_revision='new')],
        validation_scope='saved training audit only, not live weights', runs=[])
    teacher = dict(manifest_sha256='b' * 64, manifest=dict(accepted=True, planned_count=906,
        parent_sha256=audit['parent_sha256'], inputs_manifest_sha256=inputs.manifest_sha256,
        roles_sha256=inputs.roles_sha256, plan_hash=plan.plan_hash,
        source_sha256='c' * 64, review_sha256='d' * 64))
    for k, seed, name in compiler.GRID:
        basis = torch.zeros(k, 4096)
        basis[:, :k] = torch.eye(k)
        common = dict(dimension=k, seed=seed, accepted_operator=False, research_eligible=False)
        config = common | dict(declared_grid=dict(dimensions=[2, 4], seeds=list(compiler.SEEDS)),
            diagnostic_one_step=False, planned_steps=302,
            ticker_order=sorted(t for t, r in inputs.roles['assignments'].items() if r == 'fit'),
            layer=19, scope='original_instruction_post_block', dose=32,
            inputs_manifest_sha256=inputs.manifest_sha256, roles_sha256=inputs.roles_sha256,
            parent_sha256=audit['parent_sha256'], plan_hash=plan.plan_hash,
            full_plan_identity=plan.identity.to_dict(), teacher_binding=teacher,
            runtime=dict(backend=dict(actual_embedding_dtype='torch.bfloat16', actual_head_dtype='torch.bfloat16')))
        digest = sha256_json(config)
        operator = common | dict(config_sha256=digest, training_completed=True,
            layer=19, scope='original_instruction_post_block', dose=32, basis=basis.tolist())
        summary = common | dict(config_sha256=digest, training_completed=True, status='training_completed',
            completed_steps=302, attempted_steps=302, diagnostic_one_step=False)
        for filename, value in [('config.json', config), ('operator.json', operator), ('summary.json', summary)]:
            publish(root / name / filename, value)
        audit['runs'].append(common | dict(run=name, config_sha256=digest))
    audit_path = tmp_path / 'audit.json'
    def repin():
        audit['file_sha256'] = {str(p.relative_to(root)): sha256_bytes(p.read_bytes())
                               for p in root.rglob('*.json')}
        publish(audit_path, audit)
        monkeypatch.setattr(compiler, 'AUDIT_SHA256', sha256_bytes(audit_path.read_bytes()))
    repin()
    return inputs, plan, root, audit_path, audit, repin


def compile_saved(saved):
    return compiler.compile_cone_validation_plan(*saved[:4])


def test_full_panel_count_normalization_seed_and_no_leak(saved, tmp_path):
    torch.manual_seed(45)
    rng = torch.random.get_rng_state().clone()
    plan = compile_saved(saved)
    assert torch.equal(rng, torch.random.get_rng_state())
    value = plan.to_dict()
    assert value['coverage']['planned_cells'] == len(value['cells']) == 129600
    assert value['coverage']['directions'] == 48
    assert compiler.DOSES == (-64, -32, -8, -2, 0, 2, 8, 32, 64)
    assert {c['dose'] for c in value['cells']} == set(compiler.DOSES)
    assert len({(c['cone'], c['family'], c['dose'], c['row']['ticker'], c['row']['condition'])
                for c in value['cells']}) == 129600
    inputs = saved[0]
    assert all(inputs.roles['assignments'][c['row']['ticker']] == 'validation' for c in value['cells'])
    assert {c['row']['condition'] for c in value['cells']} == {'++', '--', '+-', '-+'}
    generator = torch.Generator(device='cpu').manual_seed(20261006)
    for k, seed, name in compiler.GRID:
        ds = [d for d in value['panel']['directions'] if d['cone'] == name]
        assert len(ds) == k + 5
        for d in ds:
            direction = torch.tensor(d['direction'], dtype=torch.float64)
            coeff = torch.tensor(d['coefficients'], dtype=torch.float64)
            assert direction.norm().item() == pytest.approx(1)
            assert coeff.sum().item() == pytest.approx(1)
            assert (coeff >= 0).all()
            expected = torch.zeros(4096, dtype=torch.float64)
            expected[:k] = coeff / coeff.norm()
            assert torch.allclose(expected, direction)
        for d in ds[-4:]:
            c = torch.rand(k, generator=generator) + .01
            c = (c / c.sum()).double()
            c /= c.sum()
            assert d['coefficients'] == pytest.approx(c.tolist())
            assert min(d['coefficients']) > 0
    assert plan.plan_id == compile_saved(saved).plan_id
    assert value['panel']['compiler_provenance_discrepancies'] == saved[4]['compiler_provenance_discrepancies']
    value['cells'].clear()
    assert len(plan.to_dict()['cells']) == 129600
    output = tmp_path / 'published'
    compiler.write_cone_validation_plan(output, plan)
    manifest = json.loads((output / 'manifest.json').read_bytes())
    assert manifest['plan_id'] == plan.plan_id
    for filename, digest in manifest['file_sha256'].items():
        assert sha256_bytes((output / filename).read_bytes()) == digest
    with pytest.raises(FileExistsError):
        compiler.write_cone_validation_plan(output, plan)


@pytest.mark.parametrize('mutation', ['missing_run', 'raw_hash', 'config_hash', 'parent', 'token_policy',
    'input_hash', 'roles', 'accepted', 'grid', 'zero_basis', 'nan_basis', 'wrong_width',
    'nonunit', 'nonbf16', 'cancelling'])
def test_reject_training_or_binding_mutations(saved, mutation):
    inputs, plan, root, audit_path, audit, repin = saved
    path = root / compiler.GRID[0][2]
    if mutation == 'missing_run':
        import shutil
        shutil.rmtree(path)
    elif mutation == 'raw_hash':
        (path / 'summary.json').write_bytes(b'{}\n')
    else:
        filename = 'operator.json' if mutation in ('config_hash', 'accepted', 'zero_basis',
            'nan_basis', 'wrong_width', 'nonunit', 'nonbf16', 'cancelling') else 'config.json'
        record = json.loads((path / filename).read_bytes())
        if mutation == 'config_hash': record['config_sha256'] = '0' * 64
        elif mutation == 'parent': record['parent_sha256'] = '0' * 64
        elif mutation == 'token_policy': record['full_plan_identity']['generation_policy_sha256'] = '0' * 64
        elif mutation == 'input_hash': record['inputs_manifest_sha256'] = '0' * 64
        elif mutation == 'roles': record['ticker_order'][0] = next(t for t, r in inputs.roles['assignments'].items() if r == 'evaluation')
        elif mutation == 'accepted': record['accepted_operator'] = True
        elif mutation == 'grid': record['declared_grid']['seeds'] = [20261003]
        elif mutation == 'zero_basis': record['basis'][0] = [0.] * 4096
        elif mutation == 'nan_basis': record['basis'][0][0] = float('nan')
        elif mutation == 'wrong_width': record['basis'][0].pop()
        elif mutation == 'nonunit': record['basis'][0][0] = 2.
        elif mutation == 'nonbf16': record['basis'][0][0] = 1.00001
        elif mutation == 'cancelling': record['basis'][1] = [-v for v in record['basis'][0]]
        if mutation == 'nan_basis':
            (path / filename).write_text(json.dumps(record))
        else:
            publish(path / filename, record)
        if filename == 'config.json':
            digest = sha256_json(record)
            for other in ('operator.json', 'summary.json'):
                bound = json.loads((path / other).read_bytes())
                bound['config_sha256'] = digest
                publish(path / other, bound)
            audit['runs'][0]['config_sha256'] = digest
        # Rehash fixtures to challenge semantic checks, never reauthorize production audit.
        repin()
    with pytest.raises((ValueError, RuntimeError)):
        compile_saved(saved)


@pytest.mark.parametrize('coeff', [[0, 0], [-1, 2], [float('nan'), 1], [float('inf'), 1], [1]])
def test_bad_coefficients(coeff):
    with pytest.raises(ValueError):
        compiler.unit_direction([[1, 0], [0, 1]], coeff)


def test_zero_direction_and_nonfinite_basis():
    with pytest.raises(ValueError):
        compiler.unit_direction([[1, 0], [-1, 0]], [1, 1])
    with pytest.raises(ValueError):
        compiler.unit_direction([[float('inf'), 0], [0, 1]], [1, 1])


def test_reject_unapproved_or_partial_inputs(saved):
    with pytest.raises(ValueError):
        compiler.compile_cone_validation_plan(replace(saved[0], pairs=saved[0].pairs[:-1]), *saved[1:4])
    with pytest.raises(ValueError):
        compiler.compile_cone_validation_plan(replace(saved[0], roles_sha256='0' * 64), *saved[1:4])


def test_audit_pin_mismatch(saved):
    saved[3].write_bytes(saved[3].read_bytes() + b' ')
    with pytest.raises(ValueError, match='audit SHA'):
        compile_saved(saved)
