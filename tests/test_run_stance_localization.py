"""LC4 internal fake grids only. Public CLI has no reduced-cohort override."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from test_stance_localization_execution import setup
from scripts import run_stance_localization as runner
from scripts.recover_stance_baseline_truncations import recovery_store, publish, read_file
from llm_bias.core.artifact_paths import sha256_json, canonical_json_bytes
from llm_bias.core.experiment_contract import RowKey
from llm_bias.core.stance_localization_pairs import LocalizationPair
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop
from llm_bias.core.inference.stance_localization_execution import execute_prompt_replacement


@pytest.fixture
def grid(setup):
    if setup['route'] != 'plain':
        pytest.skip('first GPU runner explicitly rejects Harmony')
    donor = RowKey('baseline', 'AAA', '++', 'trial', 'baseline', '0')
    target = RowKey('baseline', 'BBB', '++', 'trial', 'baseline', '0')
    payload = dict(target_key=target.to_dict(), donor_key=donor.to_dict(), role='fit',
                   family='cross_company', contrast='entity_context', clean_relation='same')
    pair = LocalizationPair(target, donor, 'fit', 'cross_company', 'entity_context', 'same',
                            sha256_json(payload))
    table = SimpleNamespace(pairs=(pair,), parent_sha256='a' * 64,
                            table_sha256='b' * 64, inputs_manifest_sha256='c' * 64)
    desc = runner.descriptor(table, 'alignment_sensitivity', 2, 0, 2, {})
    outputs = {donor: setup['expected_donor'], target: setup['expected_target']}
    parent = SimpleNamespace(generation_for=outputs.__getitem__)
    prompts = {donor: setup['donor_prompt'], target: setup['target_prompt']}
    attempts = {'gate': 0, 'cell': 0}
    def gate(key, layer, span, config_hash):
        attempts['gate'] += 1
        prompt = prompts[key]
        record = getattr(prompt, span + '_span')
        return execute_prompt_noop(setup['model'], setup['tokenizer'],
            torch.tensor([prompt.inference_token_ids]), setup['capability'],
            policy=setup['policy'], layer=layer, hook_site='post',
            zero_vector=torch.ones(4), prompt_positions=list(range(record.token_start, record.token_end)),
            config_hash=config_hash)
    def cell(pair, key, mapping):
        attempts['cell'] += 1
        return execute_prompt_replacement(setup['model'], setup['tokenizer'],
            prompts[pair.donor_key], prompts[pair.target_key], setup['capability'],
            policy=setup['policy'], layer=key['layer'], hook_site='post', alignment=mapping,
            expected_donor=parent.generation_for(pair.donor_key),
            expected_target=parent.generation_for(pair.target_key))
    return SimpleNamespace(setup=setup, table=table, desc=desc, pair=pair,
        parent=parent, prompts=prompts, gate=gate, cell=cell, attempts=attempts)


def execute(root, grid, gate=None, cell=None):
    with recovery_store(root, {'descriptor': grid.desc}) as records:
        return runner.run_grid(records, grid.table, grid.desc, grid.prompts, grid.parent,
                               gate or grid.gate, cell or grid.cell)


def test_closed_cli_and_stages():
    required = ['--model', 'm', '--inputs', 'i', '--parent', 'p', '--output-dir', 'o']
    args = runner.parser().parse_args(required)
    assert args.phase == 'primary' and args.num_shards == 1
    for option in ('--target-tickers', '--layer', '--span', '--max-new-tokens', '--mod'):
        with pytest.raises(SystemExit):
            runner.parser().parse_args(required + [option, '1'])
    assert len(runner.STAGES['primary']) == 4
    assert len(runner.STAGES['position']) == 16
    assert runner.STAGES['alignment_sensitivity'] == (('entity', 'full'),)


def test_all_layers_exact_partition():
    for count in (1, 2, 40):
        for shards in (1, count):
            parts = [runner.shard_layers(count, i, shards) for i in range(shards)]
            assert sorted(x for part in parts for x in part) == list(range(count))
            assert sum(map(len, parts)) == count
    for values in ((0, 0, 1), (2, 0, 3), (2, -1, 1), (2, 1, 1), (True, 0, 1)):
        with pytest.raises(ValueError):
            runner.shard_layers(*values)


def test_descriptor_full_table_roles_and_filename(grid):
    assert grid.desc['full_table_roles'] == ['fit', 'validation', 'calibration', 'evaluation']
    assert grid.desc['execution_roles'] == ['fit', 'validation']
    name, key, pair = next(runner.cells(grid.table, grid.desc))
    assert name == sha256_json(key) + '.json'
    assert key == dict(pair_sha256=pair.pair_sha256, layer=0, span='entity', selector='full')
    other = grid.table.pairs[0]
    # Execution selection never changes the descriptor's full parent/table identity.
    table = SimpleNamespace(**(vars(grid.table) | {'pairs': (other, replace(other, role='evaluation',
        pair_sha256=sha256_json(other._payload() | {'role': 'evaluation'})))}))
    assert len(tuple(runner.cells(table, grid.desc))) == 1


def test_grid_resume_full_outputs_no_retry(tmp_path, grid):
    report = execute(tmp_path / 'shard', grid)
    assert report['complete_shard'] and not report['complete_global']
    assert report['research_eligible'] is False
    assert report['counts'] == dict(planned=1, executed=1, invalid_parent=0, halted=0, failure=0)
    before = grid.attempts.copy()
    assert execute(tmp_path / 'shard', grid) == report
    assert grid.attempts == before
    records = tmp_path / 'shard' / 'records'
    cell = next(read_file(p) for p in records.iterdir() if not p.name.startswith('gate_'))
    assert canonical_json_bytes(cell['execution']['expected_target']) == canonical_json_bytes(
        grid.parent.generation_for(grid.pair.target_key).to_dict())
    assert cell['execution']['alignment']['policy'] == 'tail_overlap'
    assert cell['execution']['diagnostics']['hook_site'] == 'post'


@pytest.mark.parametrize('mutation', ['mapping', 'snapshot', 'diagnostics', 'provenance', 'match', 'status', 'gate'])
def test_resume_rejects_corruption(tmp_path, grid, mutation):
    root = tmp_path / 'shard'
    execute(root, grid)
    records = root / 'records'
    path = next(p for p in records.iterdir() if p.name.startswith('gate_') == (mutation == 'gate'))
    item = read_file(path)
    if mutation == 'gate':
        item['gate']['checks']['text_match'] = False
    elif mutation == 'mapping':
        item['execution']['alignment']['source_indices'] = [999]
    elif mutation == 'snapshot':
        item['execution']['expected_target']['elapsed_seconds'] += 1
    elif mutation == 'diagnostics':
        item['execution']['diagnostics']['layer'] = 1
    elif mutation == 'provenance':
        item['execution']['intervention']['provenance']['grammar_sha256'] = 'd' * 64
    elif mutation == 'match':
        item['execution']['match_checks']['donor'] = False
    else:
        item['status'] = 'executed_but_ignored'
    path.write_bytes(canonical_json_bytes(item) + b'\n')
    before = grid.attempts.copy()
    with pytest.raises((ValueError, TypeError)):
        execute(root, grid)
    assert grid.attempts == before


@pytest.mark.parametrize('kind', ['foreign', 'staging', 'symlink', 'missing_gate'])
def test_rejects_foreign_layout(tmp_path, grid, kind):
    root = tmp_path / 'shard'
    execute(root, grid)
    records = root / 'records'
    if kind == 'foreign':
        publish(records, 'foreign.json', {})
    elif kind == 'staging':
        publish(records, '.pending-interrupted.tmp', {})
    elif kind == 'symlink':
        path = next(p for p in records.iterdir() if not p.name.startswith('gate_'))
        data = path.read_bytes()
        path.unlink()
        destination = tmp_path / 'foreign'
        destination.write_bytes(data)
        path.symlink_to(destination)
    else:
        next(p for p in records.iterdir() if p.name.startswith('gate_')).unlink()
    with pytest.raises((ValueError, OSError)):
        execute(root, grid)


def test_parent_drift_is_persisted_halt_no_retry(tmp_path, grid):
    def drifting(pair, key, mapping):
        # LC2 donor full-output equality fails before target/intervention.
        grid.setup['model'].hf_model.target = tuple(grid.setup['tokenizer'].encode(
            '{"decision":"sell","reason":"drift"}', add_special_tokens=False))
        return grid.cell(pair, key, mapping)
    root = tmp_path / 'shard'
    with pytest.raises(RuntimeError, match='halted cell'):
        execute(root, grid, cell=drifting)
    assert not (root / 'summary.json').exists()
    records = root / 'records'
    item = next(read_file(p) for p in records.iterdir() if not p.name.startswith('gate_'))
    assert item['status'] == 'donor_mismatch'
    before = grid.attempts.copy()
    with pytest.raises(RuntimeError, match='recorded halted'):
        execute(root, grid)
    assert grid.attempts == before


def test_failed_intervention_fixed_denominator_no_retry(tmp_path, grid):
    def failing(pair, key, mapping):
        root = grid.setup['model'].hf_model
        root.fail = (root.calls + 3, 0)
        return grid.cell(pair, key, mapping)
    report = execute(tmp_path / 'shard', grid, cell=failing)
    assert report['counts']['failure'] == 1 and report['counts']['executed'] == 1
    assert report['groups'][0]['flip'] == 0
    before = grid.attempts.copy()
    assert execute(tmp_path / 'shard', grid) == report
    assert grid.attempts == before


def test_gate_drift_abort_before_cells(tmp_path, grid):
    def drift(key, layer, span, config_hash):
        root = grid.setup['model'].hf_model
        root.target = tuple(grid.setup['tokenizer'].encode(
            '{"decision":"sell","reason":"drift"}', add_special_tokens=False))
        return grid.gate(key, layer, span, config_hash)
    root = tmp_path / 'shard'
    with pytest.raises(RuntimeError, match='parent drift'):
        execute(root, grid, gate=drift)
    assert grid.attempts['cell'] == 0
    assert list((root / 'records').glob('gate_*.json'))
    with pytest.raises(RuntimeError):
        execute(root, grid)
    assert grid.attempts['gate'] == 1


def test_invalid_parent_has_no_fabricated_cell_outputs(tmp_path, grid):
    pair = grid.pair
    payload = pair._payload() | {'clean_relation': 'invalid_parent'}
    grid.table.pairs = (replace(pair, clean_relation='invalid_parent', pair_sha256=sha256_json(payload)),)
    report = execute(tmp_path / 'shard', grid)
    assert report['counts']['invalid_parent'] == 1
    assert report['counts']['executed'] == 0 and grid.attempts['cell'] == 0
    assert report['groups'][0]['eligible'] == 0


def test_registration_changed_identity_and_incomplete_init(tmp_path, grid):
    root = tmp_path / 'shard'
    execute(root, grid)
    grid.desc['bindings']['changed'] = True
    with pytest.raises(ValueError, match='registration'):
        execute(root, grid)
    incomplete = tmp_path / 'incomplete'
    incomplete.mkdir()
    with pytest.raises(OSError):
        execute(incomplete, grid)


def test_complete_summary_forbids_missing_cell_reexecution(tmp_path, grid):
    root = tmp_path / 'shard'
    report = execute(root, grid)
    publish(root, 'summary.json', report)
    next(p for p in (root / 'records').iterdir() if not p.name.startswith('gate_')).unlink()
    before = grid.attempts.copy()
    with pytest.raises(ValueError, match='summary with incomplete'):
        execute(root, grid)
    assert grid.attempts == before


def test_opposite_clean_failed_output_is_no_flip(tmp_path, grid):
    pair = grid.pair
    payload = pair._payload() | {'clean_relation': 'opposite'}
    grid.table.pairs = (replace(pair, clean_relation='opposite', pair_sha256=sha256_json(payload)),)
    def fail(pair, key, mapping):
        root = grid.setup['model'].hf_model
        root.fail = (root.calls + 3, 0)
        return grid.cell(pair, key, mapping)
    report = execute(tmp_path / 'shard', grid, cell=fail)
    assert report['groups'][0]['eligible'] == 1
    assert report['groups'][0]['flip'] == 0
    assert report['counts']['failure'] == 1
