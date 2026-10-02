from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts import run_stance_neuron_discovery as runner
from scripts.recover_stance_baseline_truncations import recovery_store, publish, read_file
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.experiment_contract import RowKey
from llm_bias.core.stance_neuron_panel import build_neuron_panel, NeuronDiscoveryPlan, NeuronRole
from test_stance_localization_execution import setup, clean


@pytest.fixture
def grid(setup):
    if setup['route'] != 'plain':
        pytest.skip('GLM plain JSON only')
    root = setup['model'].hf_model
    # Real native down-projection executed by each original fake block.
    for block in root.layers:
        block.mlp = torch.nn.Module()
        block.mlp.down_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
        with torch.no_grad():
            block.mlp.down_proj.weight.copy_(torch.eye(4))
        block.forward = lambda x, block=block: block.mlp.down_proj(x + 1) + 2
    panel = build_neuron_panel(setup['model'])
    roles = tuple(NeuronRole(t, t, r) for t, r in zip('ABCD',
        ('fit', 'validation', 'calibration', 'evaluation')))
    plan = NeuronDiscoveryPlan(panel, roles, *('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64))
    key = RowKey('baseline', 'A', '+-', 'trial', 'baseline', '0')
    parent = SimpleNamespace(generation_for=lambda k: setup['expected_target'])
    desc = runner.descriptor(plan, (key,), 0, 2, {})
    attempts = {'gate': 0, 'cell': 0}
    def generate(arm=None):
        return runner.execute_native(setup['model'], setup['tokenizer'],
            torch.tensor([setup['target_prompt'].inference_token_ids]), setup['capability'],
            setup['policy'], panel, arm)
    def gate(key, layer):
        attempts['gate'] += 1
        return runner.execute_gate(generate, panel, layer)
    def cell(key, arm):
        attempts['cell'] += 1
        return generate(arm)
    return SimpleNamespace(setup=setup, plan=plan, keys=(key,), parent=parent,
        desc=desc, gate=gate, cell=cell, generate=generate, attempts=attempts)


def execute(root, grid, gate=None, cell=None):
    with recovery_store(root, {'descriptor': grid.desc}) as records:
        return runner.run_grid(records, grid.plan, grid.keys, grid.desc, grid.parent,
                               gate or grid.gate, cell or grid.cell)


def test_closed_cli_and_partition():
    required = ['--model', 'm', '--inputs', 'i', '--parent', 'p', '--output-dir', 'o']
    for option in ('--target-tickers', '--layer', '--dose', '--phase', '--max-new-tokens'):
        with pytest.raises(SystemExit):
            runner.parser().parse_args(required + [option, '1'])
    parts = [runner.shard_layers(40, i, 7) for i in range(7)]
    assert sorted(x for p in parts for x in p) == list(range(40))


def test_cells_scope_role_and_signed_budget(grid):
    cells = list(runner.cells(grid.plan, grid.keys, grid.desc))
    assert len(cells) == 4 * 2
    assert {key['dose'] for _, key, _, _ in cells} == {-2, 2}
    assert all(key['scope'] == 'prompt_and_decode' and key['condition'] == '+-'
               and key['ticker'] == 'A' and key['layer'] == 0 for _, key, _, _ in cells)
    assert all(name == sha256_json(key) + '.json' for name, key, _, _ in cells)
    assert grid.desc['plan']['roles'] == grid.plan.to_dict()['roles']


def test_full_native_gate_resume_and_accounting(tmp_path, grid):
    report = execute(tmp_path / 'run', grid)
    assert report['complete_shard'] and not report['complete_global']
    assert report['research_eligible'] is False
    assert report['executed'] == report['planned'] == 8 and report['missing'] == 0
    assert len(report['groups']) == 8
    assert all(g['planned'] == 1 and g['baseline_buy'] == 1 for g in report['groups'])
    before = grid.attempts.copy()
    assert execute(tmp_path / 'run', grid) == report
    assert grid.attempts == before
    clean(grid.setup['model'].hf_model)


def test_native_edits_both_prompt_and_decode_and_cleanup(grid):
    root = grid.setup['model'].hf_model
    observed = []
    projection = root.layers[0].mlp.down_proj
    handle = projection.register_forward_hook(lambda m, args, output: observed.append(args[0].clone()))
    arm = runner.arm_for(grid.plan.panel, 0, grid.plan.panel.coordinates[0][0], 2)
    try:
        grid.generate(arm)
    finally:
        handle.remove()
    assert len(observed) > 1 and observed[0].shape[1] > 1
    assert observed[1].shape[1] == 1 if grid.setup['policy'].use_cache else observed[1].shape[1] > 1
    neuron = arm.neuron
    ids = grid.setup['target_prompt'].inference_token_ids
    assert observed[0][0, 0, neuron] == ids[0] + 3
    # Native input is token ID +1, with +2 at the selected coordinate on decode too.
    for values in observed:
        other = (neuron + 1) % 4
        assert torch.all(values[..., neuron] - values[..., other] == 2)
    clean(root)


def test_failed_executed_cell_no_retry(tmp_path, grid):
    def failing(key, arm):
        root = grid.setup['model'].hf_model
        root.fail = (root.calls + 1, 0)
        return grid.cell(key, arm)
    report = execute(tmp_path / 'run', grid, cell=failing)
    assert report['failure'] == 8
    assert all(g['sell_to_buy'] == 0 and g['buy_denominator'] == 1 for g in report['groups'])
    before = grid.attempts.copy()
    assert execute(tmp_path / 'run', grid) == report
    assert grid.attempts == before


def test_gate_drift_retained_abort_no_retry(tmp_path, grid):
    def bad(key, layer):
        result = grid.gate(key, layer)
        return result[:2] + (replace(result[2], generated_text='drift'),)
    with pytest.raises(RuntimeError, match='gate'):
        execute(tmp_path / 'run', grid, gate=bad)
    assert grid.attempts['cell'] == 0
    before = grid.attempts.copy()
    with pytest.raises(RuntimeError, match='gate'):
        execute(tmp_path / 'run', grid)
    assert grid.attempts == before
    assert not (tmp_path / 'run' / 'summary.json').exists()


@pytest.mark.parametrize('mutation', ['hash', 'parent', 'provenance', 'foreign', 'symlink', 'gate'])
def test_resume_rejects_before_work(tmp_path, grid, mutation):
    root = tmp_path / 'run'
    execute(root, grid)
    records = root / 'records'
    path = next(p for p in records.iterdir() if not p.name.startswith('gate_'))
    item = read_file(path)
    if mutation == 'foreign':
        publish(records, '.pending-foreign.tmp', {})
    elif mutation == 'symlink':
        raw = path.read_bytes(); path.unlink()
        target = tmp_path / 'outside'; target.write_bytes(raw); path.symlink_to(target)
    elif mutation == 'gate':
        next(records.glob('gate_*.json')).unlink()
    else:
        if mutation == 'hash':
            item['payload']['key']['neuron'] = 999
        elif mutation == 'parent':
            item['payload']['expected']['generated_text'] = 'tampered'
            item['payload_sha256'] = sha256_json(item['payload'])
        else:
            item['payload']['generation']['provenance']['grammar_sha256'] = 'f' * 64
            item['payload_sha256'] = sha256_json(item['payload'])
        path.write_bytes(canonical_json_bytes(item) + b'\n')
    before = grid.attempts.copy()
    with pytest.raises((ValueError, OSError)):
        execute(root, grid)
    assert grid.attempts == before


def test_interruption_resume_and_summary_missing_rejected(tmp_path, grid):
    root = tmp_path / 'run'
    def interrupted(key, arm):
        if grid.attempts['cell'] == 2:
            raise KeyboardInterrupt()
        return grid.cell(key, arm)
    with pytest.raises(KeyboardInterrupt):
        execute(root, grid, cell=interrupted)
    assert not (root / 'summary.json').exists()
    assert len(list((root / 'records').iterdir())) == 3
    report = execute(root, grid)
    assert grid.attempts['cell'] == 8
    publish(root, 'summary.json', report)
    next(p for p in (root / 'records').iterdir() if not p.name.startswith('gate_')).unlink()
    before = grid.attempts.copy()
    with pytest.raises(ValueError, match='incomplete'):
        execute(root, grid)
    assert grid.attempts == before


def test_full40_independent16_exact302_plan():
    root = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(
        down_proj=torch.nn.Linear(16, 2))) for _ in range(40)])
    panel = build_neuron_panel(root)
    roles = tuple(NeuronRole(f'F{i:03}', f'F{i:03}', 'fit') for i in range(302)) + tuple(
        NeuronRole(t, t, role) for t, role in zip('VCE', ('validation', 'calibration', 'evaluation')))
    plan = NeuronDiscoveryPlan(panel, roles, *('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64))
    keys = tuple(RowKey('baseline', r.ticker, '+-', 'trial', 'baseline', '0')
                 for r in roles if r.role == 'fit')
    total = 0
    for index in range(7):
        desc = runner.descriptor(plan, keys, index, 7, {})
        assert desc['planned_global'] == 386560
        counts = {}
        for _, key, target, arm in runner.cells(plan, keys, desc):
            counts.setdefault((arm.layer, arm.neuron, arm.delta), set()).add(target.ticker)
            total += 1
        assert all(tickers == set(plan.role_ids('discovery')) for tickers in counts.values())
    assert total == 386560
    with pytest.raises(ValueError, match='exact'):
        runner.descriptor(plan, keys[:-1], 0, 1, {})
    with pytest.raises(ValueError, match='exact'):
        runner.descriptor(plan, keys + (replace(keys[0], condition='++'),), 0, 1, {})


def test_actual_flip_directional_and_zero_denominator(tmp_path, grid):
    root = grid.setup['model'].hf_model
    root.sensitive = 0
    root.sentinel = grid.setup['target_prompt'].inference_token_ids[0] + 6 + 2
    root.flip_target = tuple(grid.setup['tokenizer'].encode(
        '{"decision":"sell","reason":"native changed"}', add_special_tokens=False))
    report = execute(tmp_path / 'run', grid)
    # Root observes residual coordinate0. Editing that native coordinate at +2
    # genuinely changes greedy decision. Other native coordinates do not.
    rows = [g for g in report['groups'] if g['neuron'] == 0]
    positive = next(g for g in rows if g['dose'] == 2)
    assert positive['buy_to_sell'] == 1 and positive['any_flip'] == 1
    assert positive['buy_to_sell_rate'] == 1 and positive['sell_to_buy_rate'] is None
    assert sum(g['any_flip'] for g in report['groups']) == 1


def test_changed_registration_rejects(tmp_path, grid):
    root = tmp_path / 'run'
    execute(root, grid)
    grid.desc['bindings']['different'] = True
    before = grid.attempts.copy()
    with pytest.raises(ValueError, match='registration'):
        execute(root, grid)
    assert grid.attempts == before


def test_native_zero_gate_really_forwards_hook(grid, monkeypatch):
    from contextlib import contextmanager
    original = runner.scoped_mlp_addition
    seen = []
    @contextmanager
    def observe(*args, **kwargs):
        seen.append((kwargs['layer'], kwargs['neuron'], kwargs['delta'], kwargs['scope']))
        with original(*args, **kwargs) as metadata:
            yield metadata
    monkeypatch.setattr(runner, 'scoped_mlp_addition', observe)
    baseline, repeat, zero = grid.gate(grid.keys[0], 0)
    assert runner._same_full_output(baseline, repeat) and runner._same_full_output(baseline, zero)
    assert seen == [(0, grid.plan.panel.coordinates[0][0], 0, 'prompt_and_decode')]
    clean(grid.setup['model'].hf_model)
