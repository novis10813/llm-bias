"""CPU-only adaptation checks. No checkpoint or GPU claim."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from test_run_stance_localization_grouped import primary
from test_run_stance_localization import grid
from test_stance_localization_execution import setup, clean
from scripts import run_stance_localization_plain_crossmodel as runner
from scripts import run_stance_localization_grouped as grouped
from scripts.recover_stance_baseline_truncations import read_file
from llm_bias.core.artifact_paths import sha256_bytes


def write_config(root, family='qwen3_5_text', count=32, **extra):
    text = dict(model_type=family, num_hidden_layers=count, dtype='bfloat16') | extra
    (root / 'config.json').write_text(json.dumps(dict(text_config=text)))
    return runner.config_identity(root)


def test_config_native_wrapper_depth(tmp_path):
    for family, count in [('qwen3_5_text', 32), ('gemma4_text', 48), ('gemma4_text', 36)]:
        identity = write_config(tmp_path, family, count)
        native = SimpleNamespace(get_text_config=lambda: SimpleNamespace(
            model_type=family, num_hidden_layers=count))
        fake = SimpleNamespace(hf_model=SimpleNamespace(config=native),
                               layers=[object() for _ in range(count)])
        assert runner.authenticate_layers(fake, identity) == count
        assert identity['config_sha256'] == sha256_bytes((tmp_path / 'config.json').read_bytes())
        fake.layers.pop()
        with pytest.raises(ValueError, match='wrapper'):
            runner.authenticate_layers(fake, identity)
        fake.layers = [object()] * count
        with pytest.raises(ValueError, match='distinct'):
            runner.authenticate_layers(fake, identity)
        fake.layers = [object() for _ in range(count)]
        native.get_text_config = lambda: SimpleNamespace(model_type=family, num_hidden_layers=count + 1)
        with pytest.raises(ValueError, match='depth'):
            runner.authenticate_layers(fake, identity)


@pytest.mark.parametrize('family,count,extra', [
    ('glm', 40, {}), ('qwen3_5_text', 40, {}), ('gemma4_text', True, {}),
    ('gemma4_text', 0, {}), ('gemma4_text', 48, {'dtype': 'float32'})])
def test_config_rejects_unapproved_routes(tmp_path, family, count, extra):
    with pytest.raises(ValueError):
        write_config(tmp_path, family, count, **extra)


def parent_bindings():
    return dict(generation_policy=dict(policy=dict(max_new_tokens=512, timeout_seconds=180,
        use_cache=True, channel_policy='plain_json', pad_token_id=0)),
        backend=dict(requested_dtype='native', head_dtype='torch.bfloat16',
                     embedding_dtype='torch.bfloat16'))


def full_table():
    return SimpleNamespace(pairs=tuple(SimpleNamespace(role='fit' if i < 3016 else 'evaluation',
        pair_sha256=str(i)) for i in range(4024)), parent_sha256='a', table_sha256='b',
        inputs_manifest_sha256='c')


def test_closed_cli_and_all_actual_layers():
    required = ['--model', 'm', '--inputs', 'i', '--parent', 'p', '--output-dir', 'o']
    assert runner.parser().parse_args(required).phase == 'primary'
    for extra in (['--prior-run', 'x'], ['--phase', 'position'], ['--layer', '0'],
                  ['--span', 'entity'], ['--target-tickers', 'AAA'], ['--max-new-tokens', '1']):
        with pytest.raises(SystemExit):
            runner.parser().parse_args(required + extra)
    table = full_table()
    runner.validate_parent(table, parent_bindings())
    for count in (32, 48):
        desc = grouped.descriptor(table, 'primary', count, 0, 1, {})
        assert desc['layers'] == list(range(count))
        assert desc['hook_site'] == 'post'
        assert desc['arms'] == [[span, 'full'] for span in grouped.SPANS]
        assert sum(1 for _ in grouped.cells(table, desc)) == 3016 * count * 4
        parts = [grouped.descriptor(table, 'primary', count, i, 4, {}) for i in range(4)]
        assert sorted(layer for part in parts for layer in part['layers']) == list(range(count))


@pytest.mark.parametrize('field,value', [('max_new_tokens', 4096), ('timeout_seconds', 1200),
    ('use_cache', False), ('channel_policy', 'harmony_no_tools')])
def test_original_parent_policy_only(field, value):
    bindings = parent_bindings()
    bindings['generation_policy']['policy'][field] = value
    with pytest.raises(ValueError, match='parent policy'):
        runner.validate_parent(full_table(), bindings)


def test_full_grid_and_native_parent_required():
    table = full_table()
    table.pairs = table.pairs[:-1]
    with pytest.raises(ValueError, match='full'):
        runner.validate_parent(table, parent_bindings())
    bindings = parent_bindings()
    bindings['backend']['requested_dtype'] = 'bf16'
    with pytest.raises(ValueError, match='native BF16'):
        runner.validate_parent(full_table(), bindings)


def test_runtime_exact_parent_and_source_inventory(tmp_path, monkeypatch):
    metadata = dict(model=dict(resolved_path=str(tmp_path), metadata_file_sha256={'config.json': 'a'}),
        backend={name: name for name in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda',
            'kernel_policy', 'cudnn', 'deterministic_algorithms', 'python')},
        code=dict(source_sha256={'llm_bias/core/model.py': 'existing'}))
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: deepcopy(metadata))
    bindings = deepcopy(metadata)
    actual = runner.bind_runtime(tmp_path, bindings)
    assert actual['code']['source_sha256']['llm_bias/core/model.py'] == 'existing'
    for name in ('run_stance_localization_plain_crossmodel.py', 'run_stance_localization_grouped.py',
                 'run_stance_localization.py', 'recover_stance_baseline_truncations.py'):
        path = runner.ROOT / 'scripts' / name
        assert actual['code']['source_sha256']['scripts/' + name] == sha256_bytes(path.read_bytes())
    bindings['model']['resolved_path'] = '/another'
    with pytest.raises(ValueError, match='checkpoint'):
        runner.bind_runtime(tmp_path, bindings)
    bindings = deepcopy(metadata)
    bindings['backend']['torch'] = 'another'
    with pytest.raises(ValueError, match='backend'):
        runner.bind_runtime(tmp_path, bindings)


def test_composed_actual_fake_grid_completion_failure_and_resume(tmp_path, primary):
    g = primary
    root = tmp_path / 'new'
    args = SimpleNamespace(output_dir=root)
    def fail(pair, cells):
        g.setup['model'].hf_model.fail = (g.setup['model'].hf_model.calls + 3, 0)
        return g.group(pair, cells)
    assert grouped.execute_run(args, g.desc, g.table, g.prompts, g.parent, g.gate, fail, None) == 0
    report = read_file(root / 'summary.json')
    assert report['counts']['planned'] == report['counts']['executed'] == 8
    assert report['counts']['failure'] == 1
    assert report['complete_global'] and report['research_eligible'] is False
    assert report['origins'] == {'current': 8, 'prior': 0}
    calls = g.setup['model'].hf_model.calls
    grouped.execute_run(args, g.desc, g.table, g.prompts, g.parent, g.gate, g.group, None)
    assert g.setup['model'].hf_model.calls == calls
    clean(g.setup['model'].hf_model)


def test_failed_gate_never_completes(tmp_path, primary):
    g = primary
    root = tmp_path / 'new'
    def drift(*args):
        g.setup['model'].hf_model.target = tuple(g.setup['tokenizer'].encode(
            '{"decision":"sell","reason":"drift"}', add_special_tokens=False))
        return g.gate(*args)
    with pytest.raises(RuntimeError):
        grouped.execute_run(SimpleNamespace(output_dir=root), g.desc, g.table, g.prompts,
                            g.parent, drift, g.group, None)
    assert not (root / 'summary.json').exists() and not g.groups
    clean(g.setup['model'].hf_model)


@pytest.mark.parametrize('requested_dtype', ['native', 'bfloat16'])
def test_runtime_loader_preserves_mode_and_checks_wrapper_before_store(tmp_path, monkeypatch, requested_dtype):
    identity = write_config(tmp_path)
    bindings = parent_bindings()
    bindings['backend']['requested_dtype'] = requested_dtype
    parent = SimpleNamespace(metadata={'bindings': bindings})
    monkeypatch.setattr(runner, 'load_baseline_inputs', lambda p: object())
    monkeypatch.setattr(runner, 'load_completed_baseline', lambda p, inputs: parent)
    monkeypatch.setattr(runner, 'build_localization_pairs', lambda i, p: full_table())
    monkeypatch.setattr(runner, 'bind_runtime', lambda p, b: dict(model=dict(
        metadata_file_sha256={'config.json': identity['config_sha256']})))
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: True)
    calls = []
    hf = SimpleNamespace(eval=lambda: None, config=SimpleNamespace(
        model_type='qwen3_5_text', num_hidden_layers=32))
    fake = SimpleNamespace(hf_model=hf, layers=[object() for _ in range(31)], tokenizer=object())
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return fake, None, 'cuda:0'
    monkeypatch.setattr(runner, 'load_model', load)
    args = SimpleNamespace(phase='primary', inputs='i', parent='p', model=tmp_path,
                           output_dir=tmp_path / 'output', shard_index=0, num_shards=1)
    with pytest.raises(ValueError, match='wrapper'):
        runner.run(args)
    assert calls == [(str(tmp_path.resolve()), dict(device_map=None,
        dtype='native' if requested_dtype == 'native' else runner.torch.bfloat16))]
    assert not args.output_dir.exists()


@pytest.mark.parametrize('slug,run_number', [('qwen3.5-4b', 16), ('gemma4-12b-it', 17)])
def test_real_recorded_parent_bindings(slug, run_number):
    path = (runner.ROOT / 'artifacts' / slug / 'concept-cone-steering' / 'runs' /
            f'diagnostic-baseline-resumed-stance-rb260930-{run_number}' /
            'shard-0-of-1.execution_metadata.json')
    if not path.exists():
        pytest.skip('read-only recorded parent artifact not installed')
    bindings = json.loads(path.read_text())['bindings']
    before = deepcopy(bindings)
    assert bindings['backend']['requested_dtype'] == 'bfloat16'
    assert bindings['backend']['head_dtype'] == 'torch.bfloat16'
    assert bindings['backend']['embedding_dtype'] == 'torch.bfloat16'
    policy = runner.validate_parent(full_table(), bindings)
    assert policy.channel_policy == 'plain_json'
    assert bindings == before


def test_gemma_unified_outer_bf16_config_and_wrapper(tmp_path):
    config = dict(model_type='gemma4_unified', dtype='bfloat16',
                  text_config=dict(model_type='gemma4_unified_text', num_hidden_layers=48))
    (tmp_path / 'config.json').write_text(json.dumps(config))
    identity = runner.config_identity(tmp_path)
    assert identity['model_type'] == 'gemma4_unified_text'
    assert identity['outer_model_type'] == 'gemma4_unified'
    assert identity['configured_layer_count'] == 48
    assert identity['declared_dtype'] == 'bfloat16'
    native = SimpleNamespace(model_type='gemma4_unified', get_text_config=lambda:
        SimpleNamespace(model_type='gemma4_unified_text', num_hidden_layers=48))
    fake = SimpleNamespace(hf_model=SimpleNamespace(config=native),
                           layers=[object() for _ in range(48)])
    assert runner.authenticate_layers(fake, identity) == 48
    native.model_type = 'gemma4'
    with pytest.raises(ValueError, match='outer family'):
        runner.authenticate_layers(fake, identity)
    native.model_type = 'gemma4_unified'
    native.get_text_config = lambda: SimpleNamespace(model_type='gemma4_text', num_hidden_layers=48)
    with pytest.raises(ValueError, match='family'):
        runner.authenticate_layers(fake, identity)
    config['model_type'] = 'unknown'
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='outer config'):
        runner.config_identity(tmp_path)
    config['model_type'] = 'gemma4_unified'
    config['dtype'] = 'float32'
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='native BF16'):
        runner.config_identity(tmp_path)


@pytest.mark.parametrize('requested_dtype', ['native', 'bfloat16'])
def test_public_run_records_exact_requested_mode(tmp_path, monkeypatch, requested_dtype):
    identity = write_config(tmp_path)
    bindings = parent_bindings()
    bindings['backend'].update(requested_dtype=requested_dtype, attention_implementation='sdpa')
    bindings['template'] = dict(actual_wrapper_record={})
    reference = SimpleNamespace(provenance={'stop_token_ids': [1]})
    parent = SimpleNamespace(metadata={'bindings': bindings},
        plan=SimpleNamespace(keys=['reference']), generation_for=lambda key: reference)
    monkeypatch.setattr(runner, 'load_baseline_inputs', lambda p: SimpleNamespace(members=[]))
    monkeypatch.setattr(runner, 'load_completed_baseline', lambda p, inputs: parent)
    monkeypatch.setattr(runner, 'build_localization_pairs', lambda i, p: full_table())
    monkeypatch.setattr(runner, 'bind_runtime', lambda p, b: dict(backend={}, model=dict(
        metadata_file_sha256={'config.json': identity['config_sha256']})))
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(runner.torch.cuda, 'get_device_name', lambda d: 'fake GPU')
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'fake GPU inventory')
    weight = SimpleNamespace(device=runner.torch.device('cuda:0'), dtype=runner.torch.bfloat16,
                             shape=(100, 4))
    hf = SimpleNamespace(eval=lambda: None, config=SimpleNamespace(
        model_type='qwen3_5_text', num_hidden_layers=32, _attn_implementation='sdpa'),
        get_output_embeddings=lambda: SimpleNamespace(weight=weight),
        get_input_embeddings=lambda: SimpleNamespace(weight=weight),
        parameters=lambda: [SimpleNamespace(device=weight.device, dtype=weight.dtype,
                                            is_floating_point=lambda: True)])
    fake = SimpleNamespace(hf_model=hf, layers=[object() for _ in range(32)], tokenizer=object())
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        return fake, None, 'cuda:0'
    monkeypatch.setattr(runner, 'load_model', load)
    monkeypatch.setattr(runner, 'generation_adapter', lambda m: m)
    monkeypatch.setattr(runner, 'compile_smoke_grammar', lambda *args: object())
    monkeypatch.setattr(runner, 'check_capability', lambda *args: parent.plan.keys.clear())
    monkeypatch.setattr(runner, 'WrapperPolicy', lambda **kwargs: object())
    captured = {}
    def execute(args, desc, *rest):
        captured.update(desc)
        return 0
    monkeypatch.setattr(runner.grouped, 'execute_run', execute)
    args = SimpleNamespace(phase='primary', inputs='i', parent='p', model=tmp_path,
                           output_dir=tmp_path / 'output', shard_index=0, num_shards=1)
    assert runner.run(args) == 0
    assert calls == [dict(device_map=None,
        dtype='native' if requested_dtype == 'native' else runner.torch.bfloat16)]
    assert captured['bindings']['runtime']['backend']['requested_dtype'] == requested_dtype


@pytest.mark.parametrize('field,value', [
    ('requested_dtype', 'bf16'), ('requested_dtype', 'float32'),
    ('head_dtype', 'torch.float32'), ('embedding_dtype', 'torch.float32')])
def test_parent_rejects_dtype_drift(field, value):
    bindings = parent_bindings()
    bindings['backend']['requested_dtype'] = 'bfloat16'
    bindings['backend'][field] = value
    with pytest.raises(ValueError, match='native BF16'):
        runner.validate_parent(full_table(), bindings)
