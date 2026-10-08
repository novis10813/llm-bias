"""CPU-only actual Harmony callbacks and immutable grouped persistence."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from test_stance_localization_harmony_grouped import setup, policies
from test_stance_localization_execution import clean
from test_stance_localization_grouped import reset
from scripts import run_stance_localization_harmony as runner
from scripts import run_stance_localization_grouped as grouped
from scripts.recover_stance_baseline_truncations import read_file, bind_relocated_tokenizer
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.experiment_contract import RowKey
from llm_bias.core.stance_localization_pairs import LocalizationPair


def full_table():
    return SimpleNamespace(pairs=tuple(SimpleNamespace(role='fit' if i < 3016 else 'evaluation',
        pair_sha256=str(i)) for i in range(4024)), parent_sha256='a', table_sha256='b',
        inputs_manifest_sha256='c')


def full_parent():
    keys = tuple(RowKey('baseline', f'T{i:04}', '++', 'trial', 'baseline', '0') for i in range(2012))
    records, sources = {}, {}
    for i, key in enumerate(keys):
        recovered = i < 31
        records[key] = dict(max_new_tokens=4096 if recovered else 1024,
            timeout_seconds=1200 if recovered else 300, use_cache=True,
            pad_token_id=0, channel_policy='harmony_no_tools')
        sources[key] = dict(origin='recovery' if recovered else 'original',
                            source_sha256='a' * 64, record_sha256='b' * 64)
    return SimpleNamespace(plan=SimpleNamespace(keys=keys), rows=keys,
        summary={'class_counts': {'buy': 518, 'sell': 1494}},
        metadata={'original_metadata': {'bindings': {'backend': {'requested_dtype': 'native'}}}},
        row_policy_for=lambda k: deepcopy(records[k]), row_source_for=lambda k: deepcopy(sources[k]),
        generation_for=lambda k: SimpleNamespace(provenance={'generation_policy': deepcopy(records[k])}),
        records=records, sources=sources)


def test_merged_shape_accepts_all_retries_and_policies():
    parent = full_parent()
    bindings, bound, inventory = runner.validate_parent(full_table(), parent)
    assert len(bound) == len(inventory) == 2012
    assert sum(r['source']['origin'] == 'recovery' for r in inventory) == 31
    assert {p.max_new_tokens for p in bound.values()} == {1024, 4096}
    assert bindings['backend']['requested_dtype'] == 'native'


@pytest.mark.parametrize('mutation', ['count', 'classes', 'source', 'budget', 'timeout', 'cache',
                                      'channel', 'native', 'coverage', 'provenance'])
def test_effective_parent_guards(mutation):
    parent, table = full_parent(), full_table()
    key = parent.plan.keys[0]
    if mutation == 'count': parent.rows = parent.rows[:-1]
    elif mutation == 'classes': parent.summary['class_counts']['buy'] = 517
    elif mutation == 'source': parent.sources[key]['origin'] = 'foreign'
    elif mutation == 'budget': parent.records[key]['max_new_tokens'] = 1024
    elif mutation == 'timeout': parent.records[key]['timeout_seconds'] = 300
    elif mutation == 'cache': parent.records[key]['use_cache'] = False
    elif mutation == 'channel': parent.records[key]['channel_policy'] = 'plain_json'
    elif mutation == 'native': parent.metadata['original_metadata']['bindings']['backend']['requested_dtype'] = 'bf16'
    elif mutation == 'coverage': table.pairs = table.pairs[:-1]
    else: parent.generation_for = lambda k: SimpleNamespace(provenance={'generation_policy': {}})
    with pytest.raises(ValueError): runner.validate_parent(table, parent)


def test_closed_cli():
    required = ['--model', 'm', '--inputs', 'i', '--parent', 'original12', '--recovery', 'recovery20',
                '--output-dir', 'o']
    args = runner.parser().parse_args(required)
    assert str(args.parent) == 'original12' and str(args.recovery) == 'recovery20'
    for extra in (['--prior-run', 'x'], ['--phase', 'position'], ['--layer', '0'], ['--span', 'entity'],
                  ['--max-new-tokens', '4096'], ['--timeout-seconds', '1200']):
        with pytest.raises(SystemExit): runner.parser().parse_args(required + extra)
    with pytest.raises(SystemExit): runner.parser().parse_args(required[:6] + required[8:])


def write_config(path, count=24, **extra):
    config = dict(model_type='gpt_oss', num_hidden_layers=count, dtype='bfloat16',
                  quantization_config={'quant_method': 'mxfp4'}) | extra
    (path / 'config.json').write_text(json.dumps(config))
    return runner.config_identity(path)


def test_authentic_actual_layer_grid(tmp_path):
    identity = write_config(tmp_path)
    layers = [object() for _ in range(24)]
    native = SimpleNamespace(model_type='gpt_oss', num_hidden_layers=24)
    model = SimpleNamespace(layers=layers, hf_model=SimpleNamespace(config=native,
                                                                   model=SimpleNamespace(layers=layers)))
    assert runner.authenticate_layers(model, identity) == 24
    assert identity['quantization_config'] == {'quant_method': 'mxfp4'}
    model.layers = [object() for _ in range(24)]
    with pytest.raises(ValueError, match='inventory'): runner.authenticate_layers(model, identity)
    model.layers = layers[:-1]
    with pytest.raises(ValueError, match='wrapper'): runner.authenticate_layers(model, identity)
    for count in (24, 36):
        desc = grouped.descriptor(full_table(), 'primary', count, 0, 1, {})
        assert desc['layers'] == list(range(count)) and desc['hook_site'] == 'post'
        assert sum(1 for _ in grouped.cells(full_table(), desc)) == 3016 * 4 * count
        parts = [grouped.descriptor(full_table(), 'primary', count, i, 4, {}) for i in range(4)]
        assert sorted(l for p in parts for l in p['layers']) == list(range(count))


@pytest.mark.parametrize('extra', [dict(model_type='glm'), dict(num_hidden_layers=True),
                                  dict(quantization_config={}), dict(dtype=None)])
def test_config_rejects(tmp_path, extra):
    with pytest.raises(ValueError): write_config(tmp_path, **extra)


def test_metadata_relocation_and_source_inventory(tmp_path, monkeypatch):
    names = ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
             'cudnn', 'deterministic_algorithms', 'python')
    original = dict(model=dict(resolved_path='/original', metadata_file_sha256={'config.json': 'a'}),
                    backend={n: n for n in names})
    actual = deepcopy(original)
    actual['model']['resolved_path'] = str(tmp_path)
    actual['code'] = {'source_sha256': {'llm_bias/core/model.py': 'existing'}}
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: deepcopy(actual))
    bound = runner.bind_runtime(tmp_path, original, {'metadata': {'backend': original['backend']}})
    for name in ('run_stance_localization_harmony.py', 'run_stance_localization_grouped.py',
                 'run_stance_localization.py', 'recover_stance_baseline_truncations.py'):
        assert bound['code']['source_sha256']['scripts/' + name] == sha256_bytes(
            (runner.ROOT / 'scripts' / name).read_bytes())
    tokenizer = SimpleNamespace(name_or_path=str(tmp_path))
    relocation = bind_relocated_tokenizer(tokenizer, bound['model'], original['model'])
    assert tokenizer.name_or_path == '/original' and relocation['relocation_metadata_sha256_verified']
    original['model']['metadata_file_sha256']['config.json'] = 'foreign'
    with pytest.raises(ValueError, match='checkpoint'): runner.bind_runtime(tmp_path, original, {'metadata': {'backend': original['backend']}})


@pytest.fixture(params=[(1024, 4096), (4096, 1024)])
def native_grid(setup, request):
    s = setup
    config = policies(s, *request.param)
    # Match actual original/recovery timeout contracts, not a shared pair policy.
    from llm_bias.core.inference.harmony_generation import generate_harmony_structured
    import torch
    s['model'].tokenizer = s['tokenizer']
    donor = RowKey('baseline', 'AAA', '++', 'trial', 'baseline', '0')
    target = RowKey('baseline', 'BBB', '++', 'trial', 'baseline', '0')
    prompts = {donor: s['donor_prompt'], target: s['target_prompt']}
    outputs, bound = {}, {}
    for key, label in ((donor, 'donor'), (target, 'target')):
        policy = config[label + '_policy']
        policy = replace(policy, timeout_seconds=300 if policy.max_new_tokens == 1024 else 1200)
        bound[key] = policy
        outputs[key] = generate_harmony_structured(s['model'], s['tokenizer'],
            torch.tensor([prompts[key].inference_token_ids]), s['capability'], policy=policy)
    reset(s)
    parent = SimpleNamespace(plan=SimpleNamespace(keys=(donor, target)), generation_for=outputs.__getitem__)
    runner.bind_rows(parent, s['capability'], bound)
    payload = dict(target_key=target.to_dict(), donor_key=donor.to_dict(), role='fit',
                   family='cross_company', contrast='entity_context', clean_relation='same')
    pair = LocalizationPair(target, donor, 'fit', 'cross_company', 'entity_context', 'same', sha256_json(payload))
    table = SimpleNamespace(pairs=(pair,), parent_sha256='a' * 64, table_sha256='b' * 64,
                            inputs_manifest_sha256='c' * 64)
    desc = grouped.descriptor(table, 'primary', 2, 0, 1,
        {'effective_parent': {'row_inventory': [asdict(p) for p in bound.values()]}})
    gate, group = runner.callbacks(s['model'], s['capability'], prompts, parent, bound)
    return SimpleNamespace(setup=s, parent=parent, prompts=prompts, policies=bound, pair=pair,
                           table=table, desc=desc, gate=gate, group=group)


def execute(root, g, gate=None, group=None):
    return grouped.execute_run(SimpleNamespace(output_dir=root), g.desc, g.table, g.prompts,
                               g.parent, gate or g.gate, group or g.group, None)


def test_genuine_callbacks_failure_resume_and_no_raw_tensors(tmp_path, native_grid):
    g, root = native_grid, tmp_path / 'run'
    hf = g.setup['model'].hf_model
    def fail(pair, coordinates):
        hf.fail = (hf.calls + 3, 1)
        return g.group(pair, coordinates)
    assert execute(root, g, group=fail) == 0
    report = read_file(root / 'summary.json')
    assert report['counts']['executed'] == 8 and report['counts']['failure'] == 1
    assert report['gates'] == 8 and report['origins'] == {'current': 8, 'prior': 0}
    assert hf.calls == 4 * 8 + 2 + 8
    cells = [read_file(p) for p in (root / 'records').iterdir() if not p.name.startswith('gate_')]
    for item in cells:
        row = item['execution']
        assert row['donor']['provenance']['generation_policy'] == asdict(g.policies[g.pair.donor_key])
        for arm in ('target_clean', 'intervention'):
            assert row[arm]['provenance']['generation_policy'] == asdict(g.policies[g.pair.target_key])
        assert 'analysis retained' in row['donor']['generated_text']
    assert not list(root.rglob('*.pt')) and not list(root.rglob('*.npy'))
    for path in root.rglob('*.json'):
        raw = path.read_text()
        assert 'tensor(' not in raw and 'raw_activations' not in raw and 'past_key_values' not in raw
    calls = hf.calls
    assert execute(root, g) == 0 and hf.calls == calls
    clean(hf)


@pytest.mark.parametrize('kind', ['halt', 'gate'])
def test_clean_abort_or_gate_failure_no_fabricated_cells_and_resume(tmp_path, native_grid, kind):
    g, root = native_grid, tmp_path / 'run'
    hf = g.setup['model'].hf_model
    if kind == 'halt':
        def fail(pair, cells):
            hf.fail = (hf.calls + 1, 1)
            return g.group(pair, cells)
        kwargs = {'group': fail}
    else:
        def fail(*args):
            hf.fail = (hf.calls + 1, 1)
            return g.gate(*args)
        kwargs = {'gate': fail}
    with pytest.raises(RuntimeError): execute(root, g, **kwargs)
    assert not (root / 'summary.json').exists()
    files = list((root / 'records').iterdir())
    assert all(p.name.startswith(('gate_', 'halt_')) for p in files)
    if kind == 'halt':
        halt = read_file(next(p for p in files if p.name.startswith('halt_')))
        assert halt['status'] == 'donor_failed' and 'execution' not in halt
    calls = hf.calls
    with pytest.raises(RuntimeError): execute(root, g)
    assert hf.calls == calls
    clean(hf)


@pytest.mark.parametrize('field', ['max_new_tokens', 'timeout_seconds', 'use_cache', 'channel_policy'])
def test_all_effective_policy_binding_before_generation(native_grid, field):
    g = native_grid
    bad = dict(g.policies)
    key = g.pair.target_key
    value = dict(max_new_tokens=2048, timeout_seconds=1, use_cache=False, channel_policy='plain_json')[field]
    bad[key] = replace(bad[key], **{field: value})
    with pytest.raises(ValueError): runner.bind_rows(g.parent, g.setup['capability'], bad)
    assert g.setup['model'].hf_model.calls == 0


@pytest.mark.parametrize('field', ['stop_token_ids', 'channel_contract_sha256', 'channel_policy_sha256',
                                  'schema_bytes_sha256', 'tokenizer_sha256', 'head_vocab_size'])
def test_all_effective_capability_bindings(native_grid, field):
    g = native_grid
    key = g.pair.target_key
    output = g.parent.generation_for(key)
    provenance = output.provenance
    provenance[field] = [] if field == 'stop_token_ids' else 'foreign'
    corrupted = replace(output, _provenance_bytes=canonical_json_bytes(provenance))
    parent = SimpleNamespace(plan=g.parent.plan, generation_for=lambda k: corrupted if k == key
                             else g.parent.generation_for(k))
    with pytest.raises(ValueError): runner.bind_rows(parent, g.setup['capability'], g.policies)
    assert g.setup['model'].hf_model.calls == 0


@pytest.mark.parametrize('kind', ['policy', 'foreign', 'staging', 'missing_gate'])
def test_resume_corruption_before_effects(tmp_path, native_grid, kind):
    g, root = native_grid, tmp_path / 'run'
    execute(root, g)
    if kind == 'policy':
        item = read_file(root / 'registration.json')
        item['descriptor']['bindings']['effective_parent']['row_inventory'][0]['max_new_tokens'] = 1
        (root / 'registration.json').write_bytes(canonical_json_bytes(item) + b'\n')
    elif kind == 'missing_gate': next((root / 'records').glob('gate_*.json')).unlink()
    else: (root / 'records' / ('foreign.json' if kind == 'foreign' else '.pending-crash.tmp')).write_text('{}')
    calls = g.setup['model'].hf_model.calls
    with pytest.raises(ValueError): execute(root, g)
    assert g.setup['model'].hf_model.calls == calls
    clean(g.setup['model'].hf_model)


@pytest.mark.parametrize('cuda', [False, True])
def test_public_loader_uses_strict_merged_sources_and_native_no_cpu_fallback(tmp_path, monkeypatch, cuda):
    identity = write_config(tmp_path)
    parent = full_parent()
    merged_calls, loader_calls = [], []
    monkeypatch.setattr(runner, 'load_baseline_inputs', lambda p: 'approved-inputs')
    def merged(original, recovery, *, inputs):
        merged_calls.append((original, recovery, inputs))
        return parent
    monkeypatch.setattr(runner, 'load_merged_baseline', merged)
    monkeypatch.setattr(runner, 'build_localization_pairs', lambda i, p: full_table())
    monkeypatch.setattr(runner, 'bind_runtime', lambda p, b, r: dict(model=dict(
        metadata_file_sha256={'config.json': identity['config_sha256']})))
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: cuda)
    fake = SimpleNamespace(hf_model=SimpleNamespace(eval=lambda: None, config=SimpleNamespace(
        model_type='gpt_oss', num_hidden_layers=24)), layers=[object() for _ in range(23)])
    monkeypatch.setattr(runner, 'generation_adapter', lambda m: m)
    def load(path, **kwargs):
        loader_calls.append((path, kwargs))
        return fake, None, 'cuda:0'
    monkeypatch.setattr(runner, 'load_model', load)
    args = SimpleNamespace(phase='primary', inputs='inputs', parent='original12', recovery='recovery20',
        model=tmp_path, output_dir=tmp_path / 'output', shard_index=0, num_shards=1)
    with pytest.raises((ValueError, RuntimeError), match='wrapper' if cuda else 'CUDA required'):
        runner.run(args)
    assert merged_calls == [('original12', 'recovery20', 'approved-inputs')]
    assert loader_calls == ([(str(tmp_path.resolve()), {'device_map': None, 'dtype': 'native'})] if cuda else [])
    assert not args.output_dir.exists()


def test_four_arm_gate_uses_actual_fixed_fit_policy(native_grid):
    g = native_grid
    gate = g.gate(g.pair.target_key, 0, 'entity', 'a' * 64)
    assert gate.passed
    expected = g.parent.generation_for(g.pair.target_key)
    for arm in (gate.baseline, gate.repeat, gate.zero, gate.self_replacement):
        assert arm.generated_token_ids == expected.generated_token_ids
        assert arm.generated_text == expected.generated_text
        assert arm.provenance['generation_policy'] == asdict(g.policies[g.pair.target_key])
    assert gate.zero_diagnostics['changed_token_count'] == 0
    assert g.setup['model'].hf_model.calls == 4
    clean(g.setup['model'].hf_model)


# Exact upstream bytes match original12's recorded config metadata hash.
REAL_CONFIG = runner.ROOT / 'tests/fixtures/gpt_oss_20b/config.json'
REAL_CONFIG_SHA256 = '3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce'


def test_recorded_native_config_without_dtype(tmp_path):
    raw = REAL_CONFIG.read_bytes()
    assert sha256_bytes(raw) == REAL_CONFIG_SHA256
    config = json.loads(raw)
    assert config['hidden_size'] == 2880
    assert 'dtype' not in config and 'torch_dtype' not in config
    (tmp_path / 'config.json').write_bytes(raw)
    identity = runner.config_identity(tmp_path)
    assert identity['declared_dtype'] is None
    assert identity['declared_dtype_source'] == 'absent'
    assert identity['config_sha256'] == REAL_CONFIG_SHA256
    assert identity['configured_layer_count'] == 24
    assert identity['quantization_config'] == config['quantization_config']
    assert (tmp_path / 'config.json').read_bytes() == raw


@pytest.mark.parametrize('scope', ['config', 'text_config'])
@pytest.mark.parametrize('name', ['dtype', 'torch_dtype'])
@pytest.mark.parametrize('value', [None, '', '  ', 16, False, [], {}])
def test_declared_dtype_malformed_rejects(tmp_path, scope, name, value):
    config = json.loads(REAL_CONFIG.read_bytes())
    if scope == 'text_config':
        config = {'text_config': config, 'quantization_config': config['quantization_config']}
    record = config if scope == 'config' else config['text_config']
    record[name] = value
    # Even an unselected declaration must not silently hide invalid metadata.
    if name == 'torch_dtype': record['dtype'] = 'bfloat16'
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match=f'{scope}.{name} must be a nonempty dtype string'):
        runner.config_identity(tmp_path)


@pytest.mark.parametrize('scope', ['config', 'text_config'])
@pytest.mark.parametrize('name', ['dtype', 'torch_dtype'])
def test_declared_dtype_source(tmp_path, scope, name):
    config = json.loads(REAL_CONFIG.read_bytes())
    config[name] = 'float16'
    if scope == 'text_config':
        config = {'text_config': config, 'quantization_config': config['quantization_config']}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    identity = runner.config_identity(tmp_path)
    assert identity['declared_dtype'] == 'float16'
    assert identity['declared_dtype_source'] == f'{scope}.{name}'


@pytest.mark.parametrize('mutation', ['none', 'quantization', 'layers', 'embedding_dtype', 'head_dtype'])
def test_public_missing_dtype_reaches_actual_native_authentication(tmp_path, monkeypatch, mutation):
    import torch
    raw = REAL_CONFIG.read_bytes()
    if mutation == 'quantization':
        config = json.loads(raw)
        config['quantization_config']['quant_method'] = 'foreign'
        raw = json.dumps(config).encode()
    (tmp_path / 'config.json').write_bytes(raw)
    parent = full_parent()
    parent.metadata['original_metadata']['bindings']['model'] = {}
    backend = parent.metadata['original_metadata']['bindings']['backend']
    backend.update(embedding_dtype='torch.bfloat16', head_dtype='torch.bfloat16',
                   attention_implementation='eager')
    monkeypatch.setattr(runner, 'load_baseline_inputs', lambda p: 'inputs')
    monkeypatch.setattr(runner, 'load_merged_baseline', lambda *a, **k: parent)
    monkeypatch.setattr(runner, 'build_localization_pairs', lambda *a: full_table())
    monkeypatch.setattr(runner, 'bind_runtime', lambda *a: dict(model=dict(
        metadata_file_sha256={'config.json': sha256_bytes(raw)})))
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: True)
    layers = [object() for _ in range(24)]
    weight = lambda name: SimpleNamespace(device=torch.device('cuda:0'), shape=(201088, 2880),
        dtype=torch.float16 if mutation == name else torch.bfloat16)
    hf = SimpleNamespace(eval=lambda: None, config=SimpleNamespace(model_type='gpt_oss',
        num_hidden_layers=23 if mutation == 'layers' else 24, _attn_implementation='eager'),
        model=SimpleNamespace(layers=layers), parameters=lambda: [weight('parameter')],
        get_input_embeddings=lambda: SimpleNamespace(weight=weight('embedding_dtype')),
        get_output_embeddings=lambda: SimpleNamespace(weight=weight('head_dtype')))
    model = SimpleNamespace(hf_model=hf, layers=layers, tokenizer=object())
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        return model, None, 'cuda:0'
    monkeypatch.setattr(runner, 'load_model', load)
    monkeypatch.setattr(runner, 'generation_adapter', lambda m: m)
    monkeypatch.setattr(runner, 'bind_relocated_tokenizer', lambda *a: {})
    class ActualDtypesAuthenticated(Exception): pass
    # This sentinel occurs strictly after both actual embedding/head checks.
    parent.generation_for = lambda k: SimpleNamespace(provenance={
        'generation_policy': parent.records[k], 'stop_token_ids': []})
    def compiled(*a): raise ActualDtypesAuthenticated()
    monkeypatch.setattr(runner, 'compile_smoke_grammar', compiled)
    args = SimpleNamespace(phase='primary', inputs='inputs', parent='original12', recovery='recovery20',
        model=tmp_path, output_dir=tmp_path / 'output', shard_index=0, num_shards=1)
    error = ActualDtypesAuthenticated if mutation == 'none' else ValueError
    match = {'quantization': 'MXFP4', 'layers': 'depth',
             'embedding_dtype': 'actual dtype', 'head_dtype': 'actual dtype'}.get(mutation)
    with pytest.raises(error, match=match): runner.run(args)
    assert calls == ([] if mutation == 'quantization' else [{'device_map': None, 'dtype': 'native'}])
    assert not args.output_dir.exists()
    assert (tmp_path / 'config.json').read_bytes() == raw


@pytest.mark.parametrize('quantization', [None, [], 'mxfp4', {'quant_method': 'bf16'}])
def test_native_quantization_config_required(tmp_path, quantization):
    with pytest.raises(ValueError, match='native MXFP4'):
        write_config(tmp_path, quantization_config=quantization)


@pytest.mark.parametrize('mutation', ['changed', 'missing', 'extra'])
def test_all_raw_metadata_hashes_remain_bound(tmp_path, monkeypatch, mutation):
    names = ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
             'cudnn', 'deterministic_algorithms', 'python')
    parent = dict(model=dict(metadata_file_sha256={
        'config.json': REAL_CONFIG_SHA256, 'tokenizer.json': 'tokenizer',
        'generation_config.json': 'generation'}), backend={n: n for n in names})
    actual = deepcopy(parent)
    hashes = actual['model']['metadata_file_sha256']
    if mutation == 'changed': hashes['tokenizer.json'] = 'foreign'
    elif mutation == 'missing': del hashes['generation_config.json']
    else: hashes['added_tokens.json'] = 'extra'
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: actual)
    with pytest.raises(ValueError, match='checkpoint metadata differs from parent'):
        runner.bind_runtime(tmp_path, parent, {'metadata': {'backend': parent['backend']}})


@pytest.fixture
def recorded_runtime():
    return json.loads((REAL_CONFIG.parent / 'merged_runtime_metadata.json').read_text())


def current_runtime(recorded):
    bindings = recorded['original_metadata']['bindings']
    return dict(model=deepcopy(bindings['model']),
                backend=deepcopy(recorded['recovery_registration']['metadata']['backend']),
                code={'source_sha256': {}})


def test_recorded_mixed_python_selects_exact_recovery(tmp_path, monkeypatch, recorded_runtime):
    before = deepcopy(recorded_runtime)
    actual = current_runtime(recorded_runtime)
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: deepcopy(actual))
    bound = runner.bind_runtime(tmp_path, recorded_runtime['original_metadata']['bindings'],
                                recorded_runtime['recovery_registration'])
    assert bound['backend']['python'] == '3.13.14'
    assert bound['mixed_runtime_python'] == dict(selected_backend_mode='recorded_recovery',
        current='3.13.14', original='3.13.15', recovery='3.13.14')
    assert recorded_runtime == before


@pytest.mark.parametrize('mutation', ['missing_registration', 'missing_metadata', 'missing_backend',
    'missing_python', 'empty_python', 'current_original', 'current_random', 'recovery_mismatch'])
def test_recovery_python_binding_rejects(tmp_path, monkeypatch, recorded_runtime, mutation):
    actual = current_runtime(recorded_runtime)
    recovery = recorded_runtime['recovery_registration']
    if mutation == 'missing_registration': recovery = None
    elif mutation == 'missing_metadata': del recovery['metadata']
    elif mutation == 'missing_backend': del recovery['metadata']['backend']
    elif mutation == 'missing_python': del recovery['metadata']['backend']['python']
    elif mutation == 'empty_python': recovery['metadata']['backend']['python'] = ''
    elif mutation == 'current_original': actual['backend']['python'] = '3.13.15'
    elif mutation == 'current_random': actual['backend']['python'] = '3.12.9'
    else: recovery['metadata']['backend']['python'] = '3.13.13'
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: actual)
    with pytest.raises(ValueError, match='[Pp]ython|python'):
        runner.bind_runtime(tmp_path, recorded_runtime['original_metadata']['bindings'], recovery)


@pytest.mark.parametrize('field', ['torch', 'transformers', 'xgrammar', 'jlens', 'cuda',
    'cudnn', 'kernel_policy', 'deterministic_algorithms'])
@pytest.mark.parametrize('source', ['recovery', 'current'])
def test_non_python_backend_stays_strict(tmp_path, monkeypatch, recorded_runtime, field, source):
    actual = current_runtime(recorded_runtime)
    recovery = recorded_runtime['recovery_registration']
    backend = actual['backend'] if source == 'current' else recovery['metadata']['backend']
    backend[field] = 'foreign'
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: actual)
    with pytest.raises(ValueError, match='backend differs: ' + field):
        runner.bind_runtime(tmp_path, recorded_runtime['original_metadata']['bindings'], recovery)


def test_public_run_records_selected_python_without_generation(tmp_path, monkeypatch, recorded_runtime):
    import torch
    (tmp_path / 'config.json').write_bytes(REAL_CONFIG.read_bytes())
    parent = full_parent()
    parent.metadata.update(deepcopy(recorded_runtime))
    bindings = parent.metadata['original_metadata']['bindings']
    bindings['template'] = {'actual_wrapper_record': asdict(runner.WrapperPolicy(use_chat_template=True, add_special_tokens=False))}
    parent.content_sha256, parent.file_sha256 = 'merged', {'original': {}, 'recovery': {}}
    parent.plan.to_dict = lambda: {'keys': [k.to_dict() for k in parent.plan.keys]}
    parent.plan.identity = SimpleNamespace(schema_sha256='schema')
    table = full_table()
    for pair in table.pairs: pair.target_key = parent.plan.keys[0]
    monkeypatch.setattr(runner, 'load_baseline_inputs', lambda p: SimpleNamespace(
        members=[SimpleNamespace(ticker=k.ticker) for k in parent.plan.keys],
        pair_for=lambda *a: None))
    monkeypatch.setattr(runner, 'load_merged_baseline', lambda *a, **k: parent)
    monkeypatch.setattr(runner, 'build_localization_pairs', lambda *a: table)
    actual = current_runtime(recorded_runtime)
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: deepcopy(actual))
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(runner.torch.cuda, 'get_device_name', lambda d: 'fake CUDA device')
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'fake placement')
    layers = [object() for _ in range(24)]
    weight = SimpleNamespace(device=torch.device('cuda:0'), dtype=torch.bfloat16, shape=(201088, 2880))
    hf = SimpleNamespace(eval=lambda: None, config=SimpleNamespace(model_type='gpt_oss',
        num_hidden_layers=24, _attn_implementation='eager'), model=SimpleNamespace(layers=layers),
        parameters=lambda: [weight], get_input_embeddings=lambda: SimpleNamespace(weight=weight),
        get_output_embeddings=lambda: SimpleNamespace(weight=weight))
    model = SimpleNamespace(hf_model=hf, layers=layers, tokenizer=object())
    monkeypatch.setattr(runner, 'load_model', lambda *a, **k: (model, None, 'cuda:0'))
    monkeypatch.setattr(runner, 'generation_adapter', lambda m: m)
    monkeypatch.setattr(runner, 'bind_relocated_tokenizer', lambda *a: {})
    parent.generation_for = lambda k: SimpleNamespace(provenance={
        'generation_policy': parent.records[k], 'stop_token_ids': []})
    monkeypatch.setattr(runner, 'compile_smoke_grammar', lambda *a: object())
    monkeypatch.setattr(runner, 'bind_rows', lambda *a: None)
    monkeypatch.setattr(runner, 'render_decision_prompt', lambda *a, **k: SimpleNamespace(schema_sha256='schema'))
    monkeypatch.setattr(runner, 'check_prompt', lambda *a: None)
    monkeypatch.setattr(runner, 'callbacks', lambda *a: (None, None))
    captured = []
    def execute(args, desc, *rest):
        captured.append(desc)
        return 0
    monkeypatch.setattr(grouped, 'execute_run', execute)
    args = SimpleNamespace(phase='primary', inputs='inputs', parent='original12', recovery='recovery20',
        model=tmp_path, output_dir=tmp_path / 'unused', shard_index=0, num_shards=1)
    assert runner.run(args) == 0
    desc = captured[0]
    runtime = desc['bindings']['runtime']
    assert runtime['mixed_runtime_python'] == dict(selected_backend_mode='recorded_recovery',
        current='3.13.14', original='3.13.15', recovery='3.13.14')
    assert runtime['backend']['python'] == '3.13.14'
    effective = desc['bindings']['effective_parent']
    assert effective['metadata']['original_metadata']['bindings']['backend']['python'] == '3.13.15'
    assert len(effective['row_inventory']) == 2012
    assert sum(r['source']['origin'] == 'recovery' for r in effective['row_inventory']) == 31
    assert desc['layers'] == list(range(24))
    assert not args.output_dir.exists()


@pytest.mark.parametrize('field', ['requested_dtype', 'head_dtype', 'embedding_dtype',
                                   'attention_implementation', 'use_cache'])
def test_recovery_native_controls_stay_strict(tmp_path, monkeypatch, recorded_runtime, field):
    actual = current_runtime(recorded_runtime)
    if field == 'use_cache':
        recorded_runtime['original_metadata']['bindings']['backend'][field] = True
    recorded_runtime['recovery_registration']['metadata']['backend'][field] = 'foreign'
    monkeypatch.setattr(runner, 'runtime_metadata', lambda p: actual)
    with pytest.raises(ValueError, match='recovery backend differs: ' + field):
        runner.bind_runtime(tmp_path, recorded_runtime['original_metadata']['bindings'],
                            recorded_runtime['recovery_registration'])
