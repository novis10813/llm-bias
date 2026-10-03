"""Full-role compiler fixtures and real structured fake-HF generations, no CUDA."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_stance_baseline_adapter import bundle
from test_stance_cone_validation_plan import saved, compile_saved
from test_stance_noop_execution import setup, Block
from scripts import run_stance_cone_validation as runner
from llm_bias.core.artifact_paths import sha256_json, canonical_json_bytes
from llm_bias.core.inference.structured_output import generate_structured
from llm_bias.core.inference.harmony_generation import generate_harmony_structured
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop
from scripts.recover_stance_baseline_truncations import read_file, publish, recovery_store

HASH = 'a' * 64


@pytest.fixture
def micro(tmp_path, bundle, setup):
    inputs, full = bundle
    model, tokenizer, ids, cap, kwargs = setup
    root = model.hf_model
    root.layers = torch.nn.ModuleList([Block() for _ in range(20)])
    model.layers = root.layers
    embedding = torch.nn.Embedding(512, 4).double()
    root.get_input_embeddings = lambda: embedding
    root.eval()
    policy = kwargs['policy']
    driver = generate_harmony_structured if policy.channel_policy == 'harmony_no_tools' else generate_structured
    keys = [k for k in full.keys if inputs.roles['assignments'][k.ticker] == 'validation'][:2]
    prompt = SimpleNamespace(inference_token_ids=tuple(ids[0].tolist()),
                             instruction_span=SimpleNamespace(token_start=0, token_end=1))
    def clean(key):
        return driver(model, tokenizer, ids, cap, policy=policy)
    baseline = clean(keys[0])
    parent = SimpleNamespace(generation_for=lambda key: baseline)
    directions = [dict(cone='K2-seed', family='axis', direction=[1., 0., 0., 0.])]
    cells = [dict(cone='K2-seed', family='axis', dose=dose, row=k.to_dict(),
                  arm='positive_cone' if dose > 0 else 'negative_cone_financial_adaptation' if dose < 0 else 'zero_baseline')
             for dose in (-64, -32, -8, -2, 0, 2, 8, 32, 64) for k in keys]
    plan = dict(panel=dict(directions=directions), cells=cells)
    calls = []
    def gate(key):
        return execute_prompt_noop(model, tokenizer, ids, cap, policy=policy,
            layer=19, hook_site='post', zero_vector=torch.tensor([1., 0., 0., 0.]),
            prompt_positions=[0], config_hash=HASH)
    def intervene(key, direction, dose):
        calls.append((key, direction, dose))
        return runner.addition(model, tokenizer, prompt, cap, policy,
                               torch.tensor([1., 0., 0., 0.]), dose)
    records = tmp_path / 'records'
    records.mkdir()
    def run(**changes):
        arguments = dict(records=records, plan=plan, parent=parent, inputs=inputs,
            config_hash=HASH, index=0, count=1, gate=gate, clean=clean, intervene=intervene)
        return runner.execute(**(arguments | changes))
    return SimpleNamespace(run=run, records=records, plan=plan, root=root,
                           parent=parent, calls=calls, baseline=baseline, keys=keys)


def test_full_fixed_public_grid_and_shards(saved):
    plan = compile_saved(saved).to_dict()
    assert len(plan['cells']) == 129600
    assert sum(c['dose'] == 0 for c in plan['cells']) == 14400
    shards = [runner.assigned_cells(plan, i, 7) for i in range(7)]
    assert sum(map(len, shards)) == 129600
    assert len(set().union(*(set(s) for s in shards))) == 129600
    for shard in shards:
        groups = {}
        for c in shard.values():
            groups.setdefault((c['cone'], c['family']), []).append(c)
            assert saved[0].roles['assignments'][c['row']['ticker']] == 'validation'
        assert all(len(cells) == 2700 for cells in groups.values())
        assert all(len({sha256_json(c['row']) for c in cells}) == 300 for cells in groups.values())
    with pytest.raises(ValueError):
        runner.assigned_cells(plan, 0, 49)


def test_genuine_completion_zero_references_scope_and_resume(micro):
    report = micro.run()
    assert report['complete_shard'] and not report['complete_global']
    assert report['executed_logical'] == 18 and report['missing'] == 0
    assert report['independent_zero_generations'] == 2
    assert {dose for _, _, dose in micro.calls} == {-64, -32, -8, -2, 0, 2, 8, 32, 64}
    cells = runner.assigned_cells(micro.plan, 0, 1)
    for name, cell in cells.items():
        item = runner.unwrap(read_file(micro.records / name), HASH)
        if cell['dose'] == 0:
            assert not item['source_generation']['independent_replicate']
            assert item['source_generation']['zero_name'].startswith('zero-')
        else:
            diag = item['actual']['diagnostics']
            assert diag['layer'] == 19 and diag['scope'] == 'prompt_only'
            assert diag['selected_token_opportunities'] > 0
            assert diag['changed_token_count'] > 0
    calls = micro.root.calls
    assert micro.run() == report
    assert micro.root.calls == calls
    raw = '\n'.join(p.read_text() for p in micro.records.iterdir())
    assert 'tensor(' not in raw and 'residuals' not in raw and 'past_key_values' not in raw
    assert not micro.root._forward_pre_hooks and not micro.root._forward_hooks


def test_no1_halts_before_effects(micro):
    micro.root.fail = (micro.root.calls + 1, 0)
    report = micro.run()
    assert not report['complete_shard'] and report['missing'] == 18
    assert not micro.calls
    assert (micro.records / 'gate.json').exists()


def test_parent_drift_retains_actual_clean_and_no_effects(micro):
    different = replace(micro.baseline, generated_text=micro.baseline.generated_text + ' drift')
    report = micro.run(clean=lambda key: different)
    assert report['halt_reason'] == 'clean_failure_or_parent_drift'
    assert report['executed_logical'] == 0 and not micro.calls
    rows = [read_file(p)['payload'] for p in micro.records.glob('clean-*.json')]
    assert any(r['generation']['generated_text'].endswith(' drift') for r in rows)


def test_failed_generation_is_actual_row_in_itt(micro):
    original = micro.root.generate
    def generate(prompt, **kwargs):
        # Gate, clean and zero complete, then fail every nonzero call.
        if len(micro.calls) > 1:
            micro.root.fail = (micro.root.calls + 1, 0)
        return original(prompt, **kwargs)
    micro.root.generate = generate
    report = micro.run()
    assert report['complete_shard']
    assert sum(g['counts'].get('failure', 0) for g in report['groups']) == 16
    assert sum(g['counts']['planned'] for g in report['groups']) == 18
    for name, cell in runner.assigned_cells(micro.plan, 0, 1).items():
        if cell['dose']:
            result = read_file(micro.records / name)['payload']['actual']['generation']
            assert result['failure_type'] == 'exception'
            assert 'generated_token_ids' in result and 'generated_text' in result


def test_foreign_hash_and_zero_reference_rejected(micro):
    micro.run()
    cell = next(c for c in micro.plan['cells'] if c['dose'] == 0)
    path = micro.records / (sha256_json(cell) + '.json')
    item = read_file(path)
    item['payload']['source_generation']['zero_record_sha256'] = 'b' * 64
    item['payload_sha256'] = sha256_json(item['payload'])
    path.write_bytes(canonical_json_bytes(item) + b'\n')
    with pytest.raises(ValueError, match='zero source'):
        micro.run()
    with pytest.raises(ValueError):
        runner.unwrap(item, 'c' * 64)


def test_legacy_and_out_of_plan_rejected(micro):
    publish(micro.records, 'legacy.json', dict(generation=micro.baseline.to_dict()))
    with pytest.raises(ValueError, match='foreign/legacy'):
        micro.run()


def test_registration_write_once(tmp_path):
    directory = tmp_path / 'store'
    registration = dict(source='a', plan='b')
    with recovery_store(directory, registration) as records:
        publish(records, 'record.json', dict(value=1))
        with pytest.raises(FileExistsError):
            publish(records, 'record.json', dict(value=2))
    with recovery_store(directory, registration):
        pass
    with pytest.raises(ValueError, match='registration'):
        with recovery_store(directory, dict(source='changed', plan='b')):
            pass


def test_cli_has_no_selection_controls():
    flags = {a.dest for a in runner.parser()._actions}
    assert flags == {'help', 'inputs', 'parent', 'model', 'training_dir', 'training_audit',
                     'output_dir', 'shard_index', 'num_shards'}


@pytest.mark.parametrize('use_cache', [False, True])
def test_actual_addition_signed_doses_and_cache(setup, use_cache):
    model, tokenizer, ids, cap, kwargs = setup
    root = model.hf_model
    root.layers = torch.nn.ModuleList([Block() for _ in range(20)])
    model.layers = root.layers
    embedding = torch.nn.Embedding(512, 4).double()
    root.get_input_embeddings = lambda: embedding
    policy = replace(kwargs['policy'], use_cache=use_cache)
    prompt = SimpleNamespace(inference_token_ids=tuple(ids[0].tolist()),
                             instruction_span=SimpleNamespace(token_start=0, token_end=1))
    for dose in (-64, 0, 64):
        actual, diag = runner.addition(model, tokenizer, prompt, cap, policy, torch.ones(4) / 2, dose)
        assert actual.failure_type is None
        assert diag['changed_token_count'] == (0 if dose == 0 else diag['selected_token_opportunities'])
        assert diag['delta_l2_max'] == abs(dose)
    assert not root._forward_hooks and not root._forward_pre_hooks
