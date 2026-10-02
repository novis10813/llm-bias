"""LC5 production scope and actual CPU greedy/grouped equivalence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import fcntl
import os

import pytest

from test_stance_localization_execution import setup, clean
from test_run_stance_localization import grid
from test_stance_localization_grouped import without_elapsed
from scripts import run_stance_localization as old
from scripts import run_stance_localization_grouped as runner
from scripts.recover_stance_baseline_truncations import recovery_store, publish, read_file
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json, sha256_bytes
from llm_bias.core.inference.stance_localization_grouped import execute_grouped_prompt_replacement


@pytest.fixture
def primary(grid):
    grid.desc = runner.descriptor(grid.table, 'primary', 2, 0, 1, {})
    grid.groups = []
    def group(pair, coordinates):
        grid.groups.append(coordinates)
        s = grid.setup
        return execute_grouped_prompt_replacement(s['model'], s['tokenizer'],
            grid.prompts[pair.donor_key], grid.prompts[pair.target_key], s['capability'],
            policy=s['policy'], cells=coordinates,
            expected_donor=grid.parent.generation_for(pair.donor_key),
            expected_target=grid.parent.generation_for(pair.target_key))
    grid.group = group
    return grid


def execute(root, g, *, group=None, gate=None, snapshot=None):
    desc = g.desc if snapshot is None else g.desc | {'prior_source': snapshot['source']}
    registration = dict(descriptor=desc, descriptor_sha256=sha256_json(desc))
    with recovery_store(root, registration) as records:
        return runner.run_grid(records, g.table, desc, g.prompts, g.parent,
                               gate or g.gate, group or g.group, snapshot=snapshot)


def test_closed_cli_and_full_grid():
    required = ['--model', 'm', '--inputs', 'i', '--parent', 'p', '--output-dir', 'o']
    assert runner.parser().parse_args(required).prior_run is None
    for option in (['--phase', 'position'], ['--layer', '0'], ['--span', 'entity'],
                   ['--target-tickers', 'AAA'], ['--max-new-tokens', '1']):
        with pytest.raises(SystemExit):
            runner.parser().parse_args(required + option)
    pairs = tuple(SimpleNamespace(role='fit', pair_sha256=str(i)) for i in range(3016))
    table = SimpleNamespace(pairs=pairs, parent_sha256='a', table_sha256='b',
                            inputs_manifest_sha256='c')
    full = runner.descriptor(table, 'primary', 40, 0, 1, {})
    assert sum(1 for _ in runner.cells(table, full)) == 482560
    all_keys = {name for name, _, _ in old.cells(table, full)}
    shards = [runner.descriptor(table, 'primary', 40, i, 4, {}) for i in range(4)]
    parts = [{name for name, _, _ in runner.cells(table, d)} for d in shards]
    assert all(len(p) == 120640 for p in parts)
    assert set.union(*parts) == all_keys
    assert sum(map(len, parts)) == len(all_keys)
    assert full['logical_grid_sha256'] == sha256_json(runner.logical_descriptor(
        old.descriptor(table, 'primary', 40, 0, 1, {'different_runtime': True})))


def test_full_records_equal_lc4_and_calls_resume(tmp_path, primary):
    g = primary
    old_desc = old.descriptor(g.table, 'primary', 2, 0, 1, {})
    before = g.setup['model'].hf_model.calls
    with recovery_store(tmp_path / 'old', {'descriptor': old_desc}) as records:
        reference = old.run_grid(records, g.table, old_desc, g.prompts, g.parent, g.gate, g.cell)
    old_calls = g.setup['model'].hf_model.calls - before
    before = g.setup['model'].hf_model.calls
    actual = execute(tmp_path / 'new', g)
    new_calls = g.setup['model'].hf_model.calls - before
    assert old_calls - new_calls == 2 * 8 - 2
    assert len(g.groups) == 1 and len(g.groups[0]) == 8
    assert actual['counts'] == reference['counts']
    assert actual['groups'] == reference['groups']
    assert actual['origins'] == {'current': 8, 'prior': 0}
    assert actual['complete_global'] and not actual['research_eligible']
    for path in (tmp_path / 'new' / 'records').iterdir():
        if not path.name.startswith('gate_'):
            a = read_file(path); b = read_file(tmp_path / 'old' / 'records' / path.name)
            a.pop('config_hash'); b.pop('config_hash')
            assert without_elapsed(a) == without_elapsed(b)
    publish(tmp_path / 'new', 'summary.json', actual)
    before = g.setup['model'].hf_model.calls
    assert execute(tmp_path / 'new', g) == actual
    assert g.setup['model'].hf_model.calls == before
    clean(g.setup['model'].hf_model)


def test_gate_precedes_all_group_effects(tmp_path, primary):
    g = primary
    def drift(*args):
        g.setup['model'].hf_model.target = tuple(g.setup['tokenizer'].encode(
            '{"decision":"sell","reason":"drift"}', add_special_tokens=False))
        return g.gate(*args)
    with pytest.raises(RuntimeError):
        execute(tmp_path / 'new', g, gate=drift)
    assert not g.groups
    before = g.setup['model'].hf_model.calls
    with pytest.raises(RuntimeError):
        execute(tmp_path / 'new', g)
    assert g.setup['model'].hf_model.calls == before


def test_group_halt_closed_no_fake_cells_and_no_retry(tmp_path, primary):
    g = primary
    def drift(pair, cells):
        g.setup['model'].hf_model.target = tuple(g.setup['tokenizer'].encode(
            '{"decision":"sell","reason":"drift"}', add_special_tokens=False))
        return g.group(pair, cells)
    root = tmp_path / 'new'
    with pytest.raises(RuntimeError, match='halted group'):
        execute(root, g, group=drift)
    records = list((root / 'records').iterdir())
    assert len(records) == 9
    halt = read_file(next(p for p in records if p.name.startswith('halt_')))
    assert halt['status'] == 'donor_mismatch' and len(halt['keys']) == 8
    assert 'execution' not in halt and not (root / 'summary.json').exists()
    before = g.setup['model'].hf_model.calls
    with pytest.raises(RuntimeError, match='recorded halted group'):
        execute(root, g)
    assert g.setup['model'].hf_model.calls == before
    clean(g.setup['model'].hf_model)


def test_failure_retained_and_resume(tmp_path, primary):
    g = primary
    def fail(pair, cells):
        g.setup['model'].hf_model.fail = (g.setup['model'].hf_model.calls + 3, 0)
        return g.group(pair, cells)
    report = execute(tmp_path / 'new', g, group=fail)
    assert report['counts']['executed'] == 8 and report['counts']['failure'] == 1
    assert execute(tmp_path / 'new', g) == report
    assert len(g.groups) == 1


@pytest.mark.parametrize('kind', ['mapping', 'provenance', 'status', 'staging', 'foreign',
                                 'missing_gate', 'missing_cell_summary', 'symlink'])
def test_resume_rejects_corruption_before_effects(tmp_path, primary, kind):
    root = tmp_path / 'new'
    report = execute(root, primary)
    records = root / 'records'
    path = next(p for p in records.iterdir() if not p.name.startswith('gate_'))
    if kind in ('mapping', 'provenance', 'status'):
        item = read_file(path)
        if kind == 'mapping':
            item['execution']['alignment']['source_indices'] = [999]
        elif kind == 'provenance':
            item['execution']['intervention']['provenance']['grammar_sha256'] = 'f' * 64
        else:
            item['status'] = 'foreign'
        path.write_bytes(canonical_json_bytes(item) + b'\n')
    elif kind in ('staging', 'foreign'):
        publish(records, '.pending-crash.tmp' if kind == 'staging' else 'foreign.json', {})
    elif kind == 'missing_gate':
        next(records.glob('gate_*.json')).unlink()
    elif kind == 'missing_cell_summary':
        publish(root, 'summary.json', report)
        path.unlink()
    else:
        destination = tmp_path / 'external'
        destination.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(destination)
    before = primary.setup['model'].hf_model.calls
    with pytest.raises((ValueError, TypeError, OSError)):
        execute(root, primary)
    assert primary.setup['model'].hf_model.calls == before


def test_partial_publish_resume_only_missing_cells(tmp_path, primary, monkeypatch):
    g = primary
    real = runner.publish
    count = 0
    def interrupted(directory, name, item):
        nonlocal count
        if not name.startswith('gate_'):
            count += 1
            if count == 4:
                raise RuntimeError('crash before publish')
        return real(directory, name, item)
    monkeypatch.setattr(runner, 'publish', interrupted)
    with pytest.raises(RuntimeError, match='crash'):
        execute(tmp_path / 'new', g)
    assert len(g.groups[0]) == 8
    monkeypatch.setattr(runner, 'publish', real)
    report = execute(tmp_path / 'new', g)
    assert len(g.groups[1]) == 5
    assert report['counts']['executed'] == 8
    clean(g.setup['model'].hf_model)


def test_postpublish_crash_no_retry_and_staging_rejected(tmp_path, primary, monkeypatch):
    real = runner.publish
    def interrupted(directory, name, item):
        real(directory, name, item)
        if not name.startswith('gate_'):
            raise RuntimeError('postpublish')
    monkeypatch.setattr(runner, 'publish', interrupted)
    with pytest.raises(RuntimeError):
        execute(tmp_path / 'new', primary)
    monkeypatch.setattr(runner, 'publish', real)
    report = execute(tmp_path / 'new', primary)
    assert len(primary.groups[1]) == 7 and report['counts']['executed'] == 8


def bindings():
    return dict(template={}, generation_policy={}, grammar={}, runtime=dict(model={}, backend={},
                code=dict(source_sha256=dict(runner.approved_lc4_sources()))))


def make_prior(root, g, *, partial=False, failed=False):
    g.desc = runner.descriptor(g.table, 'primary', 2, 0, 1, bindings())
    desc = old.descriptor(g.table, 'primary', 2, 0, 1, bindings())
    def cell(pair, key, mapping):
        if failed:
            g.setup['model'].hf_model.fail = (g.setup['model'].hf_model.calls + 3, 0)
        return g.cell(pair, key, mapping)
    with recovery_store(root, dict(descriptor=desc, descriptor_sha256=sha256_json(desc))) as records:
        report = old.run_grid(records, g.table, desc, g.prompts, g.parent, g.gate, cell)
    if partial:
        cell_paths = sorted(p for p in (root / 'records').iterdir() if not p.name.startswith('gate_'))
        for path in cell_paths[3:]:
            path.unlink()
    else:
        publish(root, 'summary.json', report)
    return desc


@pytest.mark.parametrize('partial', [False, True])
def test_prior_readonly_provenance_full_and_partial(tmp_path, primary, partial):
    g = primary
    prior = tmp_path / 'prior'; new = tmp_path / 'new'
    old_desc = make_prior(prior, g, partial=partial, failed=True)
    before_bytes = {str(p.relative_to(prior)): p.read_bytes() for p in prior.rglob('*') if p.is_file()}
    with runner.prior_snapshot(prior, g.table, g.desc, g.prompts, g.parent) as snapshot:
        report = execute(new, g, snapshot=snapshot)
        assert report['origins'] == {'current': 5 if partial else 0, 'prior': 3 if partial else 8}
        assert report['counts']['failure'] == (3 if partial else 8)
        assert len(g.groups) == (1 if partial else 0)
        for name, raw in snapshot['records'].items():
            item = read_file(new / 'records' / name)
            assert item['original_record'] == raw
            assert raw['config_hash'] == sha256_json(old_desc)
            assert item['config_hash'] == report['config_hash'] != raw['config_hash']
        before = g.setup['model'].hf_model.calls
        assert execute(new, g, snapshot=snapshot) == report
        assert g.setup['model'].hf_model.calls == before
    assert before_bytes == {str(p.relative_to(prior)): p.read_bytes() for p in prior.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind', ['active', 'mapping', 'logical', 'policy', 'backend',
                                 'staging', 'missing_gate', 'halt', 'registration', 'missing_source'])
def test_prior_rejections(tmp_path, primary, kind):
    g = primary
    root = tmp_path / 'prior'
    make_prior(root, g, partial=True)
    fd = None
    if kind == 'active':
        fd = os.open(root / 'store.lock', os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif kind in ('logical', 'policy', 'backend', 'registration', 'missing_source'):
        path = root / 'registration.json'; item = read_file(path)
        if kind == 'missing_source':
            item['descriptor']['bindings']['runtime']['code']['source_sha256'].pop(
                'llm_bias/core/inference/stance_localization_execution.py')
        elif kind == 'logical':
            item['descriptor']['layers'] = [1]
        elif kind == 'policy':
            item['descriptor']['bindings']['generation_policy'] = {'different': True}
        elif kind == 'backend':
            item['descriptor']['bindings']['runtime']['backend'] = {'different': True}
        else:
            item['descriptor_sha256'] = 'x'
        if kind != 'registration':
            item['descriptor_sha256'] = sha256_json(item['descriptor'])
        path.write_bytes(canonical_json_bytes(item) + b'\n')
    elif kind == 'missing_gate':
        for path in (root / 'records').glob('gate_*.json'):
            path.unlink()
    elif kind == 'mapping':
        path = next(p for p in (root / 'records').iterdir() if not p.name.startswith('gate_'))
        item = read_file(path); item['execution']['alignment']['source_indices'] = [999]
        path.write_bytes(canonical_json_bytes(item) + b'\n')
    else:
        publish(root / 'records', '.pending-crash.tmp' if kind == 'staging' else 'halt_foreign.json', {})
    try:
        with pytest.raises((ValueError, OSError, TypeError)):
            with runner.prior_snapshot(root, g.table, g.desc, g.prompts, g.parent):
                pytest.fail('invalid prior accepted')
    finally:
        if fd is not None:
            os.close(fd)


def test_invalid_parent_no_fabrication(tmp_path, primary):
    pair = primary.pair
    payload = pair._payload() | {'clean_relation': 'invalid_parent'}
    primary.table.pairs = (replace(pair, clean_relation='invalid_parent',
                                  pair_sha256=sha256_json(payload)),)
    report = execute(tmp_path / 'new', primary)
    assert not primary.groups and report['counts']['invalid_parent'] == 8
    assert report['counts']['executed'] == 0


def test_two_pairs_no_cross_pair_cache_and_exact_scope(tmp_path, primary):
    g = primary
    pair = g.pair
    payload = pair._payload() | {'role': 'validation'}
    other = replace(pair, role='validation', pair_sha256=sha256_json(payload))
    # Internal tiny table exercises two roles without resampling either pair.
    g.table.pairs = (pair, other)
    report = execute(tmp_path / 'new', g)
    assert len(g.groups) == 2 and all(len(c) == 8 for c in g.groups)
    assert report['counts']['planned'] == report['counts']['executed'] == 16
    assert {group['role'] for group in report['groups']} == {'fit', 'validation'}
    clean(g.setup['model'].hf_model)


def test_source_payload_corruption_and_changed_registration(tmp_path, primary):
    g = primary
    prior = tmp_path / 'prior'; new = tmp_path / 'new'
    make_prior(prior, g)
    with runner.prior_snapshot(prior, g.table, g.desc, g.prompts, g.parent) as snapshot:
        execute(new, g, snapshot=snapshot)
        path = next(p for p in (new / 'records').iterdir() if not p.name.startswith('gate_'))
        item = read_file(path)
        item['original_record']['execution']['intervention']['elapsed_seconds'] += 1
        path.write_bytes(canonical_json_bytes(item) + b'\n')
        before = g.setup['model'].hf_model.calls
        with pytest.raises(ValueError, match='prior payload differs'):
            execute(new, g, snapshot=snapshot)
        assert g.setup['model'].hf_model.calls == before
    g.desc['bindings']['changed'] = True
    with pytest.raises(ValueError, match='registration'):
        execute(new, g)


def test_helper_exception_cleanup_no_summary(tmp_path, primary):
    def broken(pair, coordinates):
        raise ValueError('caller failure')
    root = tmp_path / 'new'
    with pytest.raises(ValueError, match='caller failure'):
        execute(root, primary, group=broken)
    assert not (root / 'summary.json').exists()
    assert len(list((root / 'records').iterdir())) == 8
    clean(primary.setup['model'].hf_model)
    report = execute(root, primary)
    assert report['counts']['executed'] == 8


def test_incomplete_initializer_and_summary_publication(tmp_path, primary):
    incomplete = tmp_path / 'incomplete'; incomplete.mkdir()
    with pytest.raises(OSError):
        execute(incomplete, primary)
    root = tmp_path / 'new'
    args = SimpleNamespace(output_dir=root)
    assert runner.execute_run(args, primary.desc, primary.table, primary.prompts,
        primary.parent, primary.gate, primary.group, None) == 0
    assert read_file(root / 'summary.json')['complete_global']
    before = primary.setup['model'].hf_model.calls
    assert runner.execute_run(args, primary.desc, primary.table, primary.prompts,
        primary.parent, primary.gate, primary.group, None) == 0
    assert primary.setup['model'].hf_model.calls == before
