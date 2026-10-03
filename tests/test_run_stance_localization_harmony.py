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
    bound = runner.bind_runtime(tmp_path, original)
    for name in ('run_stance_localization_harmony.py', 'run_stance_localization_grouped.py',
                 'run_stance_localization.py', 'recover_stance_baseline_truncations.py'):
        assert bound['code']['source_sha256']['scripts/' + name] == sha256_bytes(
            (runner.ROOT / 'scripts' / name).read_bytes())
    tokenizer = SimpleNamespace(name_or_path=str(tmp_path))
    relocation = bind_relocated_tokenizer(tokenizer, bound['model'], original['model'])
    assert tokenizer.name_or_path == '/original' and relocation['relocation_metadata_sha256_verified']
    original['model']['metadata_file_sha256']['config.json'] = 'foreign'
    with pytest.raises(ValueError, match='checkpoint'): runner.bind_runtime(tmp_path, original)


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
    monkeypatch.setattr(runner, 'bind_runtime', lambda p, b: dict(model=dict(
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
