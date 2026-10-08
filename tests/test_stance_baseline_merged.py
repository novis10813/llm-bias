"""Full-cohort synthetic fixtures, never checkpoint or GPU execution."""
from dataclasses import asdict
from pathlib import Path
import shutil

import pytest

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_merged import load_merged_baseline, materialize_merged_baseline
from test_stance_baseline_parent import (parent_bundle, _records_for, _summary_for, _write_parent,
                                        _generation_for)
from scripts.recover_stance_baseline_truncations import make_record, run_recovery, recovery_policy


@pytest.fixture(scope='module')
def bundle():
    # The prepared worktree has no local pinned pack. Read shared bytes only.
    local = Path(__file__).resolve().parents[1] / 'data/concept-cone-steering/rebuild-v1/compiled'
    pack = local if local.is_dir() else Path('/mnt/raid1/novis/llm-bias/data/concept-cone-steering/rebuild-v1/compiled')
    return load_baseline_inputs(pack), None


@pytest.fixture
def sources(tmp_path, parent_bundle):
    from llm_bias.core.stance_baseline_parent import load_completed_baseline
    from llm_bias.core.stance_baseline_merged import RECOVERY_PAIRS
    keys = parent_bundle['plan'].keys
    selected = {k for k in keys if (k.ticker, k.condition) in RECOVERY_PAIRS}
    successes = [k for k in keys if k not in selected]
    buy = set(successes[:514])
    specs = {i: (('failure', 'truncated', 'token_budget', '') if k in selected
                 else ('success', 'buy' if k in buy else 'sell')) for i, k in enumerate(keys)}
    records, rows, _ = _records_for(parent_bundle, specs)
    original = tmp_path / 'original'
    _write_parent(original, parent_bundle, records, _summary_for(parent_bundle['plan'], rows))
    parent = load_completed_baseline(original, inputs=parent_bundle['inputs'])
    policy = recovery_policy(parent, 4096, 1200.)
    from llm_bias.core.inference.structured_output import _hf_controls
    gp = {'policy': asdict(policy), 'hf_controls': _hf_controls(policy, (199999,))}
    recovery = tmp_path / 'recovery'
    (recovery / 'records').mkdir(parents=True)
    (recovery / 'store.lock').write_bytes(b'')
    registration = dict(kind='adaptive_budget_baseline_recovery_v1', research_eligible=False,
        parent_sha256=parent.parent_sha256, parent_plan_hash=parent.plan.plan_hash,
        inputs_manifest_sha256=parent_bundle['inputs'].manifest_sha256,
        selected_keys=[k.to_dict() for k in sorted(selected)],
        original_policy=parent.metadata['bindings']['generation_policy']['policy'],
        recovery_policy=asdict(policy), metadata={'model': parent.metadata['bindings']['model']})
    def save(path, obj):
        path.write_bytes(canonical_json_bytes(obj) + b'\n')
    save(recovery / 'registration.json', registration)
    for i, k in enumerate(sorted(selected)):
        g = _generation_for(('success', 'buy' if i < 4 else 'sell'), gp)
        save(recovery / 'records' / (sha256_json(k.to_dict()) + '.json'),
             make_record(k, g, parent, parent_bundle['inputs']))
    summary = run_recovery(recovery / 'records', tuple(sorted(selected)), parent,
                           parent_bundle['inputs'], None, policy=policy)
    save(recovery / 'summary.json', summary)
    return original, recovery, parent_bundle['inputs']


def test_complete_sources_and_snapshot(sources, tmp_path):
    original, recovery, inputs = sources
    before = {p: p.read_bytes() for root in (original, recovery) for p in root.rglob('*') if p.is_file()}
    view = load_merged_baseline(original, recovery, inputs=inputs)
    assert len(view.rows) == 2012
    assert {r.key for r in view.rows} == set(view.plan.keys)
    assert view.summary['class_counts'] == {'buy': 518, 'sell': 1494}
    replacements = [r.key for r in view.rows if view.row_source_for(r.key)['origin'] == 'recovery']
    assert len(replacements) == 31
    from llm_bias.core.stance_localization_pairs import build_localization_pairs
    assert build_localization_pairs(inputs, view).pairs
    assert all(view.row_policy_for(k)['max_new_tokens'] == 4096 for k in replacements)
    assert all(view.generation_for(r.key).decision == r.outcome.decision for r in view.rows)
    metadata = view.metadata
    metadata['mixed_generation_policy'] = False
    assert view.metadata['mixed_generation_policy'] is True
    source = view.row_source_for(replacements[0]); source['origin'] = 'changed'
    assert view.row_source_for(replacements[0])['origin'] == 'recovery'
    assert before == {p: p.read_bytes() for p in before}
    copy = tmp_path / 'copy'; shutil.copytree(recovery, copy)
    assert load_merged_baseline(original, copy, inputs=inputs).parent_sha256 == view.parent_sha256
    output = tmp_path / 'merged'
    materialize_merged_baseline(view, output)
    assert (output / 'manifest.json').is_file()
    import json
    from llm_bias.core.artifact_paths import sha256_bytes
    manifest = json.loads((output / 'manifest.json').read_bytes())
    assert manifest['content_sha256'] == view.content_sha256
    assert all(sha256_bytes((output / name).read_bytes()) == digest
               for name, digest in manifest['files'].items())
    with pytest.raises(ValueError):
        materialize_merged_baseline(view, output)
    record = next((recovery / 'records').iterdir()); record.write_bytes(b'corrupt')
    assert view.generation_for(replacements[0]).decision in ('buy', 'sell')
    with pytest.raises(ValueError):
        load_merged_baseline(original, recovery, inputs=inputs)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'issuer', 'policy', 'grammar',
    'hash', 'parser', 'failure', 'registration', 'summary'])
def test_recovery_rejects(sources, mutation):
    import json
    original, recovery, inputs = sources
    path = next((recovery / 'records').iterdir())
    item = json.loads(path.read_bytes())
    if mutation == 'missing':
        path.unlink()
    elif mutation == 'extra':
        shutil.copyfile(path, path.with_name('foreign.json'))
    else:
        if mutation == 'issuer': item['outcome']['issuer_id'] = 'foreign'
        if mutation == 'policy': item['generation']['provenance']['generation_policy']['max_new_tokens'] = 1024
        if mutation == 'grammar': item['generation']['provenance']['grammar_sha256'] = '0' * 64
        if mutation == 'hash': item['original_generated_token_sha256'] = '0' * 64
        if mutation == 'parser': item['generation']['json_payload'] = '{"decision":"hold","reason":"evidence"}'
        if mutation == 'failure': item['generation']['failure_type'] = 'truncated'
        if mutation == 'registration':
            path = recovery / 'registration.json'; item = json.loads(path.read_bytes()); item['selected_keys'].pop()
        if mutation == 'summary':
            path = recovery / 'summary.json'; item = json.loads(path.read_bytes()); item['executed'] = 30
        path.write_bytes(canonical_json_bytes(item) + b'\n')
    with pytest.raises(ValueError): load_merged_baseline(original, recovery, inputs=inputs)


def test_full_harmony_outcome_guards():
    from dataclasses import replace
    from llm_bias.core.stance_baseline_merged import _harmony
    from test_stance_baseline_adapter import result
    contract = dict(prompt_suffix_ids=[5,6], initial_analysis_header_ids=[1,2,3],
        initial_final_header_ids=[1,4,3], message_end_id=7, assistant_restart_ids=[5,6],
        final_stop_id=8, forbidden_control_ids=[9])
    tokens = (1,2,3,10,7,5,6,1,4,3,11,8)
    prov = dict(channel_contract=contract, channel_contract_sha256=sha256_json(contract),
        channel_policy_sha256=sha256_json({'policy':'harmony_no_tools',
            'contract_sha256':sha256_json(contract)}), final_content_start=10,
        final_content_end=11, analysis_segments=[[3,4]], generated_token_count=12, final_token_count=1)
    g = result(generated_token_ids=tokens, generated_token_sha256=sha256_json(list(tokens)),
        generated_text='<|channel|>analysis<|message|>think<|end|><|start|>assistant'
            '<|channel|>final<|message|>{"decision":"buy","reason":"evidence"}<|return|>',
        _provenance_bytes=canonical_json_bytes(prov))
    _harmony(g)
    for changed in (replace(g, generated_text=g.generated_text + 'extra'),
                    replace(g, generated_token_ids=tokens[:-1]),
                    replace(g, _provenance_bytes=canonical_json_bytes(prov | {'final_content_start':9}))):
        with pytest.raises(ValueError): _harmony(changed)


@pytest.mark.parametrize('mutation', ['other_failure', 'valid_original_replaced'])
def test_original_rejects(sources, mutation):
    import json
    original, recovery, inputs = sources
    path = next((original / 'shard-0-of-1/records').iterdir())
    item = json.loads(path.read_bytes())
    if mutation == 'other_failure':
        item['generation']['failure_type'] = 'exception'
    else:
        # Attempt to recover an already valid original instead of a selected retry.
        valid_key = item['row']['key']
        path = recovery / 'registration.json'
        item = json.loads(path.read_bytes())
        item['selected_keys'][0] = valid_key
    path.write_bytes(canonical_json_bytes(item) + b'\n')
    with pytest.raises(ValueError): load_merged_baseline(original, recovery, inputs=inputs)
