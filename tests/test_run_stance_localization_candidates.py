"""Panel-only CPU contract and genuine grouped fake generation, never GPU evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_run_stance_localization import grid
from test_run_stance_localization_grouped import primary
from test_run_stance_localization_plain_crossmodel import parent_bindings
from test_stance_localization_execution import setup, clean
from scripts import run_stance_localization_candidates as runner
from scripts import run_stance_localization_grouped as grouped
from scripts.recover_stance_baseline_truncations import read_file
from llm_bias.core.artifact_paths import sha256_bytes
from llm_bias.core.stance_localization_candidate_panel import build_localization_candidate_panel


def full_table():
    roles = ['fit'] * 2416 + ['validation'] * 600 + ['calibration'] * 208 + ['evaluation'] * 800
    return SimpleNamespace(pairs=tuple(SimpleNamespace(role=r, pair_sha256=str(i)) for i, r in enumerate(roles)),
        parent_sha256='a', table_sha256='b', inputs_manifest_sha256='c')


def candidate(g):
    panel = build_localization_candidate_panel(model_slug='qwen3.5-4b', actual_layer_count=32)
    # A real panel-order shard with only L0, executable on the two-layer fake model.
    g.desc = runner.descriptor(g.table, panel, 0, 6, {'issuer_by_ticker': {'BBB': 'issuer'}})
    return g


def execute(root, g, gate=None, group=None):
    return runner.execute_run(SimpleNamespace(output_dir=root), g.desc, g.table, g.prompts, g.parent,
                              gate or g.gate, group or g.group)


def test_closed_cli():
    base = ['--model', 'm', '--inputs', 'i', '--parent', 'p', '--output-dir', 'o', '--model-slug', 'glm4-9b-0414']
    assert runner.parser().parse_args(base).phase == 'primary'
    for extra in (['--layers', '0'], ['--prior-run', 'v1'], ['--span', 'entity'], ['--model-slug', 'GLM'],
                  ['--phase', 'position'], ['--max-new-tokens', '1'], ['--target-tickers', 'AAA']):
        with pytest.raises(SystemExit):
            runner.parser().parse_args(base + extra)


@pytest.mark.parametrize('slug,count,total', [('qwen3.5-4b', 32, 72384), ('glm4-9b-0414', 40, 72384),
    ('gemma4-12b-it', 48, 84448), ('gpt-oss-20b', 24, 96512)])
def test_fixed_full_cells_and_shard_union(slug, count, total):
    table = full_table()
    panel = build_localization_candidate_panel(model_slug=slug, actual_layer_count=count)
    desc = runner.descriptor(table, panel, 0, 1, {})
    keys = {name for name, _, _ in runner.cells(table, desc)}
    assert len(keys) == desc['global_planned_cells'] == total
    assert desc['layers'] == list(panel.layers)
    assert desc['execution_roles'] == ['fit', 'validation']
    assert desc['full_table_roles'] == ['fit', 'validation', 'calibration', 'evaluation']
    assert desc['arms'] == [[span, 'full'] for span in runner.SPANS]
    parts = [runner.descriptor(table, panel, i, 3, {}) for i in range(3)]
    assert parts[0]['layers'] == list(panel.layers[::3])
    sets = [{name for name, _, _ in runner.cells(table, d)} for d in parts]
    assert set.union(*sets) == keys and sum(map(len, sets)) == len(keys)
    assert all(p.role in runner.ROLES for _, _, p in runner.cells(table, desc))
    for index, shards in ((-1, 1), (1, 1), (0, len(panel.layers) + 1), (True, 1)):
        with pytest.raises(ValueError):
            runner.shard_panel(panel.layers, index, shards)


@pytest.mark.parametrize('slug,family,count', [('qwen3.5-4b', 'qwen3_5_text', 32),
    ('gemma4-12b-it', 'gemma4_text', 48), ('glm4-9b-0414', 'glm4', 40)])
def test_native_config_and_authenticated_inventory(tmp_path, slug, family, count):
    config = dict(model_type=family, num_hidden_layers=count, torch_dtype='bfloat16')
    (tmp_path / 'config.json').write_text(json.dumps(config))
    identity = runner.config_identity(tmp_path, slug)
    fake = SimpleNamespace(hf_model=SimpleNamespace(config=SimpleNamespace(**config)),
                           layers=[object() for _ in range(count)])
    assert runner.authenticate_layers(fake, identity) == count
    fake.layers = [object()] * count
    with pytest.raises(ValueError, match='distinct'):
        runner.authenticate_layers(fake, identity)
    fake.layers = [object() for _ in range(count - 1)]
    with pytest.raises(ValueError, match='wrapper'):
        runner.authenticate_layers(fake, identity)
    config['num_hidden_layers'] = count - 1
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runner.config_identity(tmp_path, slug)


def test_plain_policies_and_full_roles():
    bindings = parent_bindings()
    for slug in ('qwen3.5-4b', 'gemma4-12b-it'):
        runner.validate_parent(full_table(), bindings, slug)
    with pytest.raises(ValueError, match='requested BF16'):
        runner.validate_parent(full_table(), bindings, 'glm4-9b-0414')
    bindings['backend']['requested_dtype'] = 'bfloat16'
    runner.validate_parent(full_table(), bindings, 'glm4-9b-0414')
    for field, value in (('max_new_tokens', 4096), ('timeout_seconds', 1200), ('use_cache', False),
                         ('channel_policy', 'harmony_no_tools')):
        bad = deepcopy(bindings)
        bad['generation_policy']['policy'][field] = value
        with pytest.raises(ValueError):
            runner.validate_parent(full_table(), bad, 'glm4-9b-0414')
    table = full_table()
    table.pairs = table.pairs[:-1]
    with pytest.raises(ValueError, match='full'):
        runner.validate_parent(table, bindings, 'glm4-9b-0414')


def test_gpt_explicitly_unsupported_before_io_or_loader(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('GPT unsupported route must not load or create artifacts')
    monkeypatch.setattr(runner, 'load_baseline_inputs', forbidden)
    monkeypatch.setattr(runner, 'load_model', forbidden)
    args = SimpleNamespace(phase='primary', model_slug='gpt-oss-20b', output_dir=tmp_path / 'new')
    with pytest.raises(NotImplementedError, match='strict merged Harmony replay'):
        runner.run(args)
    assert not args.output_dir.exists()
    with pytest.raises(NotImplementedError):
        runner.config_identity(tmp_path, 'gpt-oss-20b')


def test_sources_and_full_model_path_trust(tmp_path, monkeypatch):
    metadata = dict(model=dict(resolved_path=str(tmp_path), metadata_file_sha256={'config.json': 'a'}),
        backend={n: n for n in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
                               'cudnn', 'deterministic_algorithms', 'python')},
        code=dict(source_sha256={'llm_bias/core/model.py': 'retained'}))
    monkeypatch.setattr(runner.plain, 'runtime_metadata', lambda p: deepcopy(metadata))
    result = runner.bind_runtime(tmp_path, metadata)
    assert result['code']['source_sha256']['llm_bias/core/model.py'] == 'retained'
    for name in ('run_stance_localization_candidates.py', 'run_stance_localization_grouped.py',
                 'run_stance_localization_plain_crossmodel.py', 'run_stance_localization.py'):
        assert result['code']['source_sha256']['scripts/' + name] == sha256_bytes((runner.ROOT / 'scripts' / name).read_bytes())
    bad = deepcopy(metadata)
    bad['model']['resolved_path'] = str(tmp_path / 'same-glm-name')
    with pytest.raises(ValueError, match='checkpoint'):
        runner.bind_runtime(tmp_path, bad)


def test_real_group_core_resume_no_raw_and_source_na(tmp_path, primary):
    g = candidate(primary)
    root = tmp_path / 'new'
    execute(root, g)
    report = read_file(root / 'summary.json')
    assert report['counts']['planned'] == report['counts']['executed'] == 4
    assert report['gates'] == 4 and len(g.groups) == 1 and len(g.groups[0]) == 4
    assert not report['candidate_panel_complete'] and not report['research_eligible']
    assert all(row['itt'] is None for row in report['directional_groups'])
    assert len(report['directional_groups']) == 8
    calls = g.setup['model'].hf_model.calls
    execute(root, g)
    assert g.setup['model'].hf_model.calls == calls
    for path in (root / 'records').iterdir():
        text = path.read_text()
        assert 'residuals' not in text and 'tensor(' not in text and 'past_key_values' not in text
    clean(g.setup['model'].hf_model)
    changed = deepcopy(g.desc)
    changed['bindings']['changed'] = True
    with pytest.raises(ValueError):
        runner.execute_run(SimpleNamespace(output_dir=root), changed, g.table, g.prompts, g.parent, g.gate, g.group)


def test_noop_parent_drift_blocks_before_effects_and_resume(tmp_path, primary):
    g = candidate(primary)
    def drift(*args):
        g.setup['model'].hf_model.target = tuple(g.setup['tokenizer'].encode(
            '{"decision":"sell","reason":"drift"}', add_special_tokens=False))
        return g.gate(*args)
    with pytest.raises(RuntimeError):
        execute(tmp_path / 'new', g, gate=drift)
    assert not g.groups and not (tmp_path / 'new' / 'summary.json').exists()
    calls = g.setup['model'].hf_model.calls
    with pytest.raises(RuntimeError):
        execute(tmp_path / 'new', g)
    assert g.setup['model'].hf_model.calls == calls
    clean(g.setup['model'].hf_model)


def test_clean_abort_retains_halt_not_fabricated_rows(tmp_path, primary):
    g = candidate(primary)
    def drift(pair, coordinates):
        g.setup['model'].hf_model.target = tuple(g.setup['tokenizer'].encode(
            '{"decision":"sell","reason":"drift"}', add_special_tokens=False))
        return g.group(pair, coordinates)
    with pytest.raises(RuntimeError, match='halted group'):
        execute(tmp_path / 'new', g, group=drift)
    files = list((tmp_path / 'new' / 'records').iterdir())
    assert len(files) == 5 and sum(p.name.startswith('halt_') for p in files) == 1
    calls = g.setup['model'].hf_model.calls
    with pytest.raises(RuntimeError, match='halted group'):
        execute(tmp_path / 'new', g)
    assert g.setup['model'].hf_model.calls == calls
    clean(g.setup['model'].hf_model)


def test_actual_failed_intervention_itt_retained(tmp_path, primary):
    g = candidate(primary)
    def fail(pair, coordinates):
        g.setup['model'].hf_model.fail = (g.setup['model'].hf_model.calls + 3, 0)
        return g.group(pair, coordinates)
    execute(tmp_path / 'new', g, group=fail)
    report = read_file(tmp_path / 'new' / 'summary.json')
    assert report['counts']['failure'] == 1 and report['counts']['executed'] == 4
    calls = g.setup['model'].hf_model.calls
    execute(tmp_path / 'new', g)
    assert g.setup['model'].hf_model.calls == calls
    clean(g.setup['model'].hf_model)


@pytest.mark.parametrize('slug,family,count,mode', [('glm4-9b-0414', 'glm4', 40, 'bfloat16'),
    ('qwen3.5-4b', 'qwen3_5_text', 32, 'native'), ('qwen3.5-4b', 'qwen3_5_text', 32, 'bfloat16'),
    ('gemma4-12b-it', 'gemma4_text', 48, 'bfloat16')])
def test_fake_loader_exact_mode_inventory_before_panel_store(tmp_path, monkeypatch, slug, family, count, mode):
    (tmp_path / 'config.json').write_text(json.dumps(dict(model_type=family,
        num_hidden_layers=count, torch_dtype='bfloat16')))
    identity = runner.config_identity(tmp_path, slug)
    bindings = parent_bindings()
    bindings['backend']['requested_dtype'] = mode
    parent = SimpleNamespace(metadata={'bindings': bindings})
    monkeypatch.setattr(runner, 'load_baseline_inputs', lambda p: object())
    monkeypatch.setattr(runner, 'load_completed_baseline', lambda p, inputs: parent)
    monkeypatch.setattr(runner, 'build_localization_pairs', lambda i, p: full_table())
    monkeypatch.setattr(runner, 'bind_runtime', lambda p, b: dict(model=dict(
        metadata_file_sha256={'config.json': identity['config_sha256']})))
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: True)
    hf = SimpleNamespace(eval=lambda: None, config=SimpleNamespace(model_type=family, num_hidden_layers=count))
    fake = SimpleNamespace(hf_model=hf, layers=[object() for _ in range(count - 1)])
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return fake, None, 'cuda:0'
    monkeypatch.setattr(runner, 'load_model', load)
    monkeypatch.setattr(runner, 'generation_adapter', lambda m: m)
    args = SimpleNamespace(phase='primary', model_slug=slug, inputs='i', parent='p', model=tmp_path,
                           output_dir=tmp_path / 'new', shard_index=0, num_shards=1)
    with pytest.raises(ValueError, match='wrapper'):
        runner.run(args)
    assert calls == [(str(tmp_path.resolve()), dict(device_map=None,
        dtype='native' if mode == 'native' else runner.torch.bfloat16))]
    assert not args.output_dir.exists()
    calls.clear()
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA required'):
        runner.run(args)
    assert not calls and not args.output_dir.exists()


def test_descriptor_tamper_and_v1_root_rejected(tmp_path, primary):
    g = candidate(primary)
    for field, value in (('layers', [0, 1]), ('actual_layer_count', 40), ('panel_sha256', 'a'),
                         ('global_planned_cells', 1), ('execution_roles', ['evaluation'])):
        bad = deepcopy(g.desc)
        bad[field] = value
        with pytest.raises(ValueError):
            runner._plan(g.table, bad)
    grouped.execute_run(SimpleNamespace(output_dir=tmp_path / 'old'),
        grouped.descriptor(g.table, 'primary', 2, 0, 1, {}), g.table, g.prompts, g.parent, g.gate, g.group, None)
    with pytest.raises(ValueError):
        execute(tmp_path / 'old', g)
    clean(g.setup['model'].hf_model)


def test_company_first_and_issuer_balanced_failure_denominators():
    plan, saved = {}, {}
    for i, (ticker, flip, failed) in enumerate([('AAA', True, False), ('AAA', False, True),
                                              ('BBB', True, False), ('CCC', False, False)]):
        key = dict(span='entity', layer=0)
        pair = SimpleNamespace(role='fit', family='cross_company', contrast='entity_context',
            clean_relation='opposite', target_key=SimpleNamespace(ticker=ticker))
        plan[str(i)] = (key, pair)
        saved[str(i)] = SimpleNamespace(expected_donor=SimpleNamespace(decision='buy'),
            intervention=SimpleNamespace(failure_type='runtime' if failed else None,
                decision='buy' if flip else 'sell'))
    rows = runner.company_issuer_summary(plan, saved, {'AAA': 'one', 'BBB': 'one', 'CCC': 'two'})
    buy, sell = rows
    assert buy['company_first_itt'] == .5 and buy['issuer_balanced_itt'] == .375
    assert buy['companies'] == 3 and buy['issuers'] == 2
    assert sell['company_first_itt'] is None and sell['issuers'] == 0
    directions = runner.directional_summary(plan, saved)
    assert directions[0]['eligible'] == 4 and directions[0]['failure'] == 1 and directions[0]['itt'] == .5
    assert directions[1]['itt'] is None


def test_missing_planned_record_never_certifies_complete(tmp_path, primary):
    g = candidate(primary)
    root = tmp_path / 'new'
    execute(root, g)
    path = next(p for p in (root / 'records').iterdir() if not p.name.startswith('gate_'))
    path.unlink()
    calls = g.setup['model'].hf_model.calls
    with pytest.raises(ValueError, match='incomplete'):
        execute(root, g)
    assert g.setup['model'].hf_model.calls == calls
    clean(g.setup['model'].hf_model)
