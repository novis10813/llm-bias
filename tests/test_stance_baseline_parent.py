"""LC0 read-only complete baseline parent snapshot with unverified synthetic fixtures.

The fixture uses the actual pinned full input pack and all 2012 synthetic record
exports written directly (never through the writable store). Fixture provenance
is labeled synthetic unverified; it certifies record structure, not models.
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict

import pytest

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.experiment_contract import ExecutionRow, GenerationOutcome, PlanIdentity, RowKey
from llm_bias.core.inference.structured_output import (StructuredGenerationPolicy,
    StructuredGenerationResult, _hf_controls, validate_decision_payload)
from llm_bias.core.prompt_input.decision_prompt import (DECISION_INSTRUCTION, DECISION_SCHEMA,
    DECISION_TEMPLATE, _ENTITY_TEMPLATE)
from llm_bias.core.stance_baseline_adapter import _validate_generation_result
from llm_bias.core.stance_baseline_parent import CompletedBaseline, load_completed_baseline
from llm_bias.core.stance_baseline_plan import build_baseline_plan
from test_stance_baseline_adapter import bundle

SCHEMA_SHA256 = sha256_json({'schema': DECISION_SCHEMA, 'key_order': ['decision', 'reason']})
BODY_TEMPLATE_SHA256 = sha256_json({'template': DECISION_TEMPLATE, 'entity_template': _ENTITY_TEMPLATE,
                                   'instruction': DECISION_INSTRUCTION})
HEAD = 200000
TOKENS = (11, 22, 33, 199999)
STOPS = (199999,)
POLICY = StructuredGenerationPolicy(max_new_tokens=512, use_cache=True, pad_token_id=199998,
                                    timeout_seconds=180.0, channel_policy='plain_json')
HARMONY_PREFIX = ('<|channel|>analysis<|message|>think<|end|><|start|>assistant'
                  '<|channel|>final<|message|>')
PARENT_KIND = 'completed_baseline_parent_v1'


def _provenance(gp_record):
    return {
        'backend': 'xgrammar', 'backend_version': '0.2.8',
        'model_binding': {'declarations': {'tokenizer': ['model.tokenizer'],
                                           'head': ['hf_model.output_head'],
                                           'stops': ['hf_model.config.eos_token_id']},
                          'attested_missing': [], 'verification_reference': None},
        'byte_policy': 'hf-fast-bpe-bytelevel-strict-utf8-v1',
        'schema_sha256': SCHEMA_SHA256,
        'schema_bytes_sha256': sha256_json('synthetic unverified schema_bytes_sha256'),
        'tokenizer_sha256': sha256_json('synthetic unverified tokenizer_sha256'),
        'tokenizer_info_sha256': sha256_json('synthetic unverified tokenizer_info_sha256'),
        'head_vocab_size': HEAD,
        'stop_token_ids': list(STOPS),
        'compiler_policy': {'strict_mode': True, 'any_order': False, 'any_whitespace': True},
        'grammar_correction': 'xgrammar-0.2.8-minLength-json-escapes-v1',
        'grammar_sha256': sha256_json('synthetic unverified grammar_sha256'),
        'mask_backend': 'cpu',
        'generation_policy': gp_record['policy'],
        'hf_controls': gp_record['hf_controls'],
        'generation_policy_sha256': sha256_json(gp_record),
    }


def _generation_for(spec, gp_record):
    if spec[0] in ('success', 'harmony'):
        decision = spec[1]
        payload = '{"decision":"%s","reason":"evidence"}' % decision
        text = (HARMONY_PREFIX + payload + '\n') if spec[0] == 'harmony' else payload + '<eos>'
        failure, finish, error = None, 'eos', None
    else:
        _, failure, finish, payload = spec
        text, error = 'partial', 'execution diagnostic'
    parsed = validate_decision_payload(payload)
    return StructuredGenerationResult(
        text, TOKENS, sha256_json(list(TOKENS)), payload,
        parsed.decision if failure is None else None,
        parsed.reason if failure is None else None,
        parsed.decision_complete, parsed.schema_complete, parsed.reason_valid,
        finish, failure, 0.125, canonical_json_bytes(_provenance(gp_record)), error, None)


def _row_for(issuers, key, generated):
    status = _validate_generation_result(generated)
    outcome = GenerationOutcome(**(status.to_dict() | {
        'ticker': key.ticker, 'issuer_id': issuers[key.ticker],
        'condition': key.condition, 'trial_id': key.trial_id}))
    return ExecutionRow(key, outcome)


def _records_for(parent_bundle, specs):
    inputs, plan, gp = parent_bundle['inputs'], parent_bundle['plan'], parent_bundle['gp_record']
    plan_hash = plan.plan_hash
    issuers = {m.ticker: m.issuer_id for m in inputs.members}
    records, rows, generations = {}, {}, {}
    for index, key in enumerate(plan.keys):
        spec = specs.get(index, ('success', 'buy' if index % 2 == 0 else 'sell'))
        generated = _generation_for(spec, gp)
        row = _row_for(issuers, key, generated)
        envelope = {'schema_version': 1, 'plan_hash': plan_hash, 'shard_index': 0,
                    'num_shards': 1, 'row': row.to_dict(), 'generation': generated.to_dict()}
        records[sha256_json(key.to_dict()) + '.json'] = canonical_json_bytes(envelope) + b'\n'
        rows[key] = row
        generations[key] = generated
    return records, rows, generations


def _summary_for(plan, rows):
    outcomes = [row.outcome for row in rows.values()]
    return {'kind': 'diagnostic_full_baseline_v1', 'research_eligible': False,
            'steering_flips_claimed': False, 'plan_hash': plan.plan_hash,
            'planned_global': 2012, 'shard_index': 0, 'num_shards': 1,
            'assigned_count': len(plan.keys), 'executed_count': len(outcomes),
            'missing_count': 0, 'complete_shard': True, 'complete_global': True,
            'class_counts': dict(Counter(o.decision or 'no_decision' for o in outcomes)),
            'failure_counts': dict(Counter(o.failure_type or 'none' for o in outcomes)),
            'finish_counts': dict(Counter(o.finish_reason for o in outcomes)),
            'gates': {name: False for name in plan.gate_names}}


def _write_parent(root, parent_bundle, records, summary):
    shard = root / 'shard-0-of-1'
    (shard / 'records').mkdir(parents=True)
    (root / 'shard-0-of-1.execution_metadata.json').write_bytes(
        canonical_json_bytes(parent_bundle['metadata']) + b'\n')
    (root / 'shard-0-of-1.summary.json').write_bytes(canonical_json_bytes(summary) + b'\n')
    (shard / 'registration.json').write_bytes(canonical_json_bytes(parent_bundle['registration']) + b'\n')
    (shard / 'store.lock').write_bytes(b'')
    for filename, data in records.items():
        (shard / 'records' / filename).write_bytes(data)


def _parent_sha256(parent_bundle, records, summary):
    files = {
        'shard-0-of-1.execution_metadata.json':
            sha256_bytes(canonical_json_bytes(parent_bundle['metadata']) + b'\n'),
        'shard-0-of-1.summary.json': sha256_bytes(canonical_json_bytes(summary) + b'\n'),
        'shard-0-of-1/registration.json':
            sha256_bytes(canonical_json_bytes(parent_bundle['registration']) + b'\n'),
    }
    files.update({'shard-0-of-1/records/' + name: sha256_bytes(data) for name, data in records.items()})
    return sha256_json({'kind': PARENT_KIND, 'files': files})


# One row per priority failure cause, plus buy/sell/harmony successes.
BASE_SPECS = {
    0: ('success', 'buy'),
    1: ('success', 'sell'),
    2: ('failure', 'exception', 'exception', ''),
    3: ('failure', 'timeout', 'timeout', ''),
    4: ('failure', 'no_legal_token', 'no_legal_token', ''),
    5: ('failure', 'truncated', 'token_budget', '{malformed'),
    6: ('failure', 'unsupported_channel', 'unsupported', ''),
    7: ('failure', 'unsupported_tokenizer', 'unsupported', ''),
    8: ('failure', 'invalid_json', 'eos', '{malformed'),
    9: ('failure', 'invalid_schema', 'schema_complete', '{"decision":"hold","reason":"evidence"}'),
    10: ('failure', 'invalid_reason', 'schema_complete', '{"decision":"buy","reason":"  "}'),
    11: ('harmony', 'buy'),
}


@pytest.fixture(scope='module')
def parent_bundle(bundle):
    inputs, _ = bundle
    gp_record = {'policy': asdict(POLICY), 'hf_controls': _hf_controls(POLICY, STOPS)}
    template_record = {'body_template_sha256': BODY_TEMPLATE_SHA256,
                       'actual_wrapper_record': {
                           'use_chat_template': True, 'add_special_tokens': False,
                           'system_message': None, 'enable_thinking': False,
                           'chat_template_kwargs': {},
                           'tokenizer_chat_template': '[gMASK]<sop> synthetic unverified template'}}
    protocol_record = {'kind': 'diagnostic_full_baseline_v1', 'research_eligible': False,
                       'planned_global': 2012, 'members': 503,
                       'role_assignment': inputs.roles_sha256, 'evidence': inputs.evidence_sha256,
                       'sampling': 'all_approved_pairs_once',
                       'timeout_budget_seconds_per_row': POLICY.timeout_seconds,
                       'model_route': POLICY.channel_policy, 'efficacy_claim': False,
                       'protocol_freeze_claim': False}
    model_record = {'resolved_path': '/synthetic/unverified-checkpoint',
                    'metadata_file_sha256': {'config.json': 'a' * 64},
                    'identity_scope': 'metadata_only_not_full_weights'}
    code_record = {'source_sha256': {'scripts/run_stance_baseline.py': 'b' * 64},
                   'git_head': 'c' * 40, 'git_head_kind': 'git_object_id', 'jlens_git_head': 'd' * 40}
    backend_record = {'python': '3.13.14', 'torch': '2.9.1+cu128', 'transformers': '5.14.1',
                      'xgrammar': '0.2.8', 'jlens': '0.1.0', 'cuda': '12.8',
                      'kernel_policy': 'synthetic', 'cudnn': 91002, 'deterministic_algorithms': False,
                      'device': 'cuda:0', 'embedding_device': 'cuda:0', 'head_device': 'cuda:0',
                      'requested_dtype': 'bfloat16', 'head_dtype': 'torch.bfloat16',
                      'embedding_dtype': 'torch.bfloat16', 'gpu_name': 'synthetic',
                      'attention_implementation': 'sdpa'}
    identity = PlanIdentity(
        protocol_sha256=sha256_json(protocol_record),
        population_sha256=inputs.membership_sha256,
        issuer_sha256=inputs.issuer_mapping_sha256,
        roles_sha256=inputs.roles_sha256, evidence_sha256=inputs.evidence_sha256,
        schema_sha256=SCHEMA_SHA256, template_sha256=sha256_json(template_record),
        model_sha256=sha256_json(model_record), code_sha256=sha256_json(code_record),
        backend_sha256=sha256_json(backend_record),
        generation_policy_sha256=sha256_json(gp_record),
        operator_sha256='not_applicable', parent_sha256='not_applicable')
    plan = build_baseline_plan(inputs, identity)
    metadata = {'identity': identity.to_dict(),
                'bindings': {'protocol': protocol_record, 'template': template_record,
                             'generation_policy': gp_record, 'model': model_record,
                             'code': code_record, 'backend': backend_record},
                'inputs_manifest_sha256': inputs.manifest_sha256}
    registration = {'schema_version': 1, 'plan': plan.to_dict(),
                    'inputs_manifest_sha256': inputs.manifest_sha256,
                    'shard_index': 0, 'num_shards': 1}
    return {'inputs': inputs, 'plan': plan, 'gp_record': gp_record,
            'metadata': metadata, 'registration': registration}


@pytest.fixture(scope='module')
def parent_artifact(tmp_path_factory, parent_bundle):
    root = tmp_path_factory.mktemp('parent') / 'parent'
    records, rows, generations = _records_for(parent_bundle, BASE_SPECS)
    summary = _summary_for(parent_bundle['plan'], rows)
    _write_parent(root, parent_bundle, records, summary)
    return {'root': root, 'records': records, 'rows': rows, 'generations': generations,
            'summary': summary, 'first_record': min(records),
            'parent_sha256': _parent_sha256(parent_bundle, records, summary)}


@pytest.fixture
def parent_dir(tmp_path, parent_artifact):
    target = tmp_path / 'parent'
    shutil.copytree(parent_artifact['root'], target)
    return target


def _load(parent_dir, parent_bundle):
    return load_completed_baseline(parent_dir, inputs=parent_bundle['inputs'])


def _rewrite(path, record, *, allow_nan=False):
    if allow_nan:
        data = json.dumps(record, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=True).encode('utf-8') + b'\n'
    else:
        data = canonical_json_bytes(record) + b'\n'
    path.write_bytes(data)


# ------------------------------------------------------------------ happy path


def test_complete_parent_loads(parent_dir, parent_bundle, parent_artifact):
    started = time.monotonic()
    parent = _load(parent_dir, parent_bundle)
    elapsed = time.monotonic() - started
    print(f'\nLC0 synthetic parent load wall time: {elapsed:.2f}s')
    assert isinstance(parent, CompletedBaseline)
    assert parent.plan.to_json() == parent_bundle['plan'].to_json()
    assert len(parent.rows) == 2012
    expected = tuple(parent_artifact['rows'][key] for key in sorted(parent_artifact['rows']))
    assert parent.rows == expected
    assert parent.parent_sha256 == parent_artifact['parent_sha256']
    assert set(parent.file_sha256) == {'shard-0-of-1.execution_metadata.json',
                                       'shard-0-of-1.summary.json', 'shard-0-of-1/registration.json'} \
        | {'shard-0-of-1/records/' + name for name in parent_artifact['records']}
    assert len(parent.file_sha256) == 2015


def test_generation_for_reimports_each_access(parent_dir, parent_bundle, parent_artifact):
    parent = _load(parent_dir, parent_bundle)
    plan = parent_bundle['plan']
    for index in (0, 1, 11):
        key = plan.keys[index]
        for _ in range(2):
            assert parent.generation_for(key).to_dict() == parent_artifact['generations'][key].to_dict()


def test_full_text_preserved_plain_and_harmony(parent_dir, parent_bundle, parent_artifact):
    parent = _load(parent_dir, parent_bundle)
    plan = parent_bundle['plan']
    plain = parent.generation_for(plan.keys[0])
    harmony = parent.generation_for(plan.keys[11])
    assert plain.generated_text.endswith('<eos>') and '<|channel|>' not in plain.generated_text
    assert harmony.generated_text == HARMONY_PREFIX + harmony.json_payload + '\n'
    assert plain.generated_token_ids == harmony.generated_token_ids == TOKENS


def test_single_plan_reconstruction_and_no_row_rebuild(parent_dir, parent_bundle, monkeypatch):
    import llm_bias.core.stance_baseline_parent as module
    calls = []
    original = module.build_baseline_plan
    def counting(inputs, identity):
        calls.append(identity)
        return original(inputs, identity)
    monkeypatch.setattr(module, 'build_baseline_plan', counting)
    _load(parent_dir, parent_bundle)
    assert calls == [parent_bundle['plan'].identity]
    assert 'baseline_execution_row' not in vars(module)


def test_source_changes_do_not_alter_snapshot(parent_dir, parent_bundle):
    parent = _load(parent_dir, parent_bundle)
    key = parent_bundle['plan'].keys[0]
    before = (parent.parent_sha256, parent.metadata, parent.summary, parent.file_sha256,
              parent.generation_for(key).to_dict(), parent.rows[0])
    record = parent_dir / 'shard-0-of-1' / 'records' / (sha256_json(key.to_dict()) + '.json')
    record.write_bytes(b'garbage')
    (parent_dir / 'shard-0-of-1.summary.json').write_bytes(b'garbage')
    assert (parent.parent_sha256, parent.metadata, parent.summary, parent.file_sha256,
            parent.generation_for(key).to_dict(), parent.rows[0]) == before


def test_copied_directory_same_parent_hash(tmp_path, parent_artifact, parent_bundle):
    original = load_completed_baseline(parent_artifact['root'],
                                       inputs=parent_bundle['inputs']).parent_sha256
    copy = tmp_path / 'copy'
    shutil.copytree(parent_artifact['root'], copy)
    moved = load_completed_baseline(copy, inputs=parent_bundle['inputs'])
    assert moved.parent_sha256 == original == parent_artifact['parent_sha256']
    assert moved.file_sha256 == load_completed_baseline(parent_artifact['root'],
                                                        inputs=parent_bundle['inputs']).file_sha256


def test_changed_record_changes_parent_identity(tmp_path, parent_dir, parent_bundle):
    base = _load(parent_dir, parent_bundle).parent_sha256
    key = parent_bundle['plan'].keys[12]
    path = parent_dir / 'shard-0-of-1' / 'records' / (sha256_json(key.to_dict()) + '.json')
    envelope = json.loads(path.read_bytes())
    for container in (envelope['generation'],):
        container['json_payload'] = '{"decision":"buy","reason":"changed reason"}'
        container['generated_text'] = container['json_payload'] + '<eos>'
        container['reason'] = 'changed reason'
    path.write_bytes(canonical_json_bytes(envelope) + b'\n')
    changed = _load(parent_dir, parent_bundle)
    assert changed.parent_sha256 != base
    assert changed.generation_for(key).json_payload == '{"decision":"buy","reason":"changed reason"}'


def test_defensive_copies(parent_dir, parent_bundle):
    parent = _load(parent_dir, parent_bundle)
    key = parent_bundle['plan'].keys[0]
    metadata = parent.metadata
    metadata['identity']['protocol_sha256'] = 'f' * 64
    metadata['bindings']['protocol']['kind'] = 'forged'
    assert parent.metadata['identity']['protocol_sha256'] != 'f' * 64
    assert parent.metadata['bindings']['protocol']['kind'] == 'diagnostic_full_baseline_v1'
    summary = parent.summary
    summary['gates']['no_op'] = True
    assert parent.summary['gates']['no_op'] is False
    files = parent.file_sha256
    files['shard-0-of-1.summary.json'] = '0' * 64
    assert parent.file_sha256['shard-0-of-1.summary.json'] != '0' * 64
    parent.generation_for(key).provenance['schema_sha256'] = 'f' * 64
    assert parent.generation_for(key).provenance['schema_sha256'] == SCHEMA_SHA256
    with pytest.raises(Exception):
        parent.rows[0].key.ticker = 'X'
    with pytest.raises(Exception):
        parent.rows = ()


def test_unknown_key_rejected(parent_dir, parent_bundle):
    parent = _load(parent_dir, parent_bundle)
    first = parent_bundle['plan'].keys[0]
    foreign = RowKey('localization', first.ticker, first.condition, first.trial_id, 'baseline', '0')
    with pytest.raises(ValueError):
        parent.generation_for(foreign)
    with pytest.raises(ValueError):
        parent.generation_for(object())


def test_no_writable_surface(parent_dir, parent_bundle):
    parent = _load(parent_dir, parent_bundle)
    for name in ('record', 'close', '__enter__', '__exit__', 'finalize'):
        assert not hasattr(parent, name), name
    with pytest.raises(Exception):
        parent.__enter__()


def test_public_api_has_no_cohort_override():
    import inspect
    import llm_bias.core.stance_baseline_parent as module
    parameters = list(inspect.signature(module.load_completed_baseline).parameters)
    assert parameters == ['directory', 'inputs']
    assert inspect.signature(module.load_completed_baseline).parameters['inputs'].kind \
        is inspect.Parameter.KEYWORD_ONLY
    assert not hasattr(module, 'load_partial_baseline')


def test_direct_construction_is_trusted_not_attested():
    parent = CompletedBaseline(plan=object(), rows=(), parent_sha256='0' * 64,
                               _metadata_bytes=b'{}\n', _summary_bytes=b'{}\n',
                               _file_hashes_bytes=b'{}\n', _record_bytes={})
    assert parent.file_sha256 == {}
    with pytest.raises(ValueError):
        parent.generation_for(object())


# ------------------------------------------------------------------- lock mode


def test_active_writer_rejected(parent_dir, parent_bundle):
    code = ('import fcntl,os,sys,time\n'
            'fd=os.open(sys.argv[1],os.O_RDWR)\n'
            'fcntl.flock(fd,fcntl.LOCK_EX)\n'
            'print("held",flush=True)\n'
            'time.sleep(30)\n')
    process = subprocess.Popen([sys.executable, '-c', code,
                                str(parent_dir / 'shard-0-of-1' / 'store.lock')],
                               stdout=subprocess.PIPE)
    try:
        assert process.stdout.readline() == b'held\n'
        with pytest.raises(ValueError, match='writable'):
            _load(parent_dir, parent_bundle)
    finally:
        process.kill()
        process.wait()


def test_shared_reader_coexists(parent_dir, parent_bundle):
    fd = os.open(str(parent_dir / 'shard-0-of-1' / 'store.lock'), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        parent = _load(parent_dir, parent_bundle)
        assert len(parent.rows) == 2012
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_lock_inode_swap_rejected(parent_dir, parent_bundle, monkeypatch):
    import llm_bias.core.stance_baseline_parent as module
    original = module.fcntl.flock
    lock = parent_dir / 'shard-0-of-1' / 'store.lock'
    def swapped(fd, flags):
        original(fd, flags)
        lock.rename(lock.with_name('old-lock'))
        lock.write_bytes(b'')
    monkeypatch.setattr(module.fcntl, 'flock', swapped)
    with pytest.raises(ValueError, match='inode'):
        _load(parent_dir, parent_bundle)


def test_failed_load_releases_lock_and_leaves_files(parent_dir, parent_bundle):
    snapshot = {}
    for path in parent_dir.rglob('*'):
        if path.is_file() and path.name != 'shard-0-of-1.summary.json':
            snapshot[str(path.relative_to(parent_dir))] = path.read_bytes()
    _rewrite(parent_dir / 'shard-0-of-1.summary.json', {'forged': True})
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)
    fd = os.open(str(parent_dir / 'shard-0-of-1' / 'store.lock'), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    for name, data in snapshot.items():
        assert (parent_dir / name).read_bytes() == data


# ------------------------------------------------------------------- bad layout


@pytest.mark.parametrize('layout', ['extra-root', 'missing-summary', 'missing-metadata',
    'missing-registration', 'extra-shard-file', 'missing-lock', 'lock-directory',
    'multi-shard-names', 'symlink-root', 'symlink-registration', 'symlink-lock',
    'symlink-record', 'records-symlink', 'records-file'])
def test_bad_layout(parent_dir, parent_bundle, layout):
    root = parent_dir
    shard = root / 'shard-0-of-1'
    if layout == 'extra-root':
        (root / 'notes.txt').write_bytes(b'x')
    elif layout == 'missing-summary':
        (root / 'shard-0-of-1.summary.json').unlink()
    elif layout == 'missing-metadata':
        (root / 'shard-0-of-1.execution_metadata.json').unlink()
    elif layout == 'missing-registration':
        (shard / 'registration.json').unlink()
    elif layout == 'extra-shard-file':
        (shard / 'extra.json').write_bytes(b'{}\n')
    elif layout == 'missing-lock':
        (shard / 'store.lock').unlink()
    elif layout == 'lock-directory':
        (shard / 'store.lock').unlink()
        (shard / 'store.lock').mkdir()
    elif layout == 'multi-shard-names':
        for name in ('shard-0-of-1', 'shard-0-of-1.execution_metadata.json',
                     'shard-0-of-1.summary.json'):
            (root / name).rename(root / name.replace('shard-0-of-1', 'shard-0-of-2'))
    elif layout == 'symlink-root':
        moved = root.with_name('moved')
        root.rename(moved)
        root.symlink_to(moved)
    elif layout == 'symlink-registration':
        (shard / 'registration.json').rename(shard / 'registration-real.json')
        (shard / 'registration.json').symlink_to('registration-real.json')
    elif layout == 'symlink-lock':
        (shard / 'store.lock').rename(shard / 'lock-real')
        (shard / 'store.lock').symlink_to('lock-real')
    elif layout == 'symlink-record':
        name = next(iter((shard / 'records').iterdir()))
        (name).rename(parent_dir / 'record-real')
        (name).symlink_to(parent_dir / 'record-real')
    elif layout == 'records-symlink':
        (shard / 'records').rename(shard / 'records-real')
        (shard / 'records').symlink_to('records-real')
    else:
        shutil.rmtree(shard / 'records')
        (shard / 'records').write_bytes(b'not a directory')
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


def test_staging_leftover_rejected(parent_dir, parent_bundle):
    (parent_dir / 'shard-0-of-1' / 'records' / '.pending-abrupt.tmp').write_bytes(b'torn')
    with pytest.raises(ValueError, match='staging'):
        _load(parent_dir, parent_bundle)


def test_missing_summary_is_not_complete(parent_dir, parent_bundle):
    (parent_dir / 'shard-0-of-1.summary.json').unlink()
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


# --------------------------------------------------------------- registration


@pytest.mark.parametrize('corruption', ['utf8', 'duplicate', 'nan', 'whitespace', 'torn',
    'extra-field', 'missing-field', 'schema-version', 'bool-version', 'shard-index',
    'num-shards', 'manifest', 'plan-hash', 'plan-keys', 'plan-identity', 'plan-gates'])
def test_corrupt_registration(parent_dir, parent_bundle, corruption):
    path = parent_dir / 'shard-0-of-1' / 'registration.json'
    data = json.loads(path.read_bytes())
    raw = None
    if corruption == 'utf8':
        raw = b'\xff'
    elif corruption == 'duplicate':
        raw = b'{"a":1,"a":2}\n'
    elif corruption == 'nan':
        raw = b'{"a":NaN}\n'
    elif corruption == 'whitespace':
        raw = b' ' + path.read_bytes()
    elif corruption == 'torn':
        raw = path.read_bytes()[:-5]
    elif corruption == 'extra-field':
        data['extra'] = 1
    elif corruption == 'missing-field':
        del data['shard_index']
    elif corruption == 'schema-version':
        data['schema_version'] = 2
    elif corruption == 'bool-version':
        data['schema_version'] = True
    elif corruption == 'shard-index':
        data['shard_index'] = 1
    elif corruption == 'num-shards':
        data['num_shards'] = 2
    elif corruption == 'manifest':
        data['inputs_manifest_sha256'] = '0' * 64
    elif corruption == 'plan-hash':
        data['plan']['plan_hash'] = '0' * 64
    elif corruption == 'plan-keys':
        data['plan']['keys'] = data['plan']['keys'][:-1]
    elif corruption == 'plan-identity':
        data['plan']['identity']['evidence_sha256'] = '0' * 64
    elif corruption == 'plan-gates':
        data['plan']['gate_names'] = ['no_op']
    path.write_bytes(raw if raw is not None else canonical_json_bytes(data) + b'\n')
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


# --------------------------------------------------------------------- records


def _record_path(parent_dir, name):
    return parent_dir / 'shard-0-of-1' / 'records' / name


def _success_record_name(parent_bundle):
    # plan.keys[0] is spec ('success', 'buy'); corrupting it exercises every branch.
    return sha256_json(parent_bundle['plan'].keys[0].to_dict()) + '.json'


@pytest.mark.parametrize('corruption', ['utf8', 'duplicate', 'nan', 'whitespace', 'torn',
    'extra-field', 'missing-field', 'bool-version', 'shard-index', 'num-shards',
    'plan-hash', 'row-decision', 'row-issuer', 'row-key', 'filename', 'tokens-string',
    'token-hash', 'payload-decision', 'flags', 'success-parse-failure',
    'payload-failure-mismatch', 'failed-primary-set', 'provenance-schema',
    'provenance-policy-hash', 'provenance-hf', 'provenance-missing'])
def test_corrupt_record(parent_dir, parent_bundle, corruption):
    name = _success_record_name(parent_bundle)
    path = _record_path(parent_dir, name)
    data = json.loads(path.read_bytes())
    raw = None
    if corruption == 'utf8':
        raw = b'\xff'
    elif corruption == 'duplicate':
        raw = b'{"a":1,"a":2}\n'
    elif corruption == 'nan':
        raw = b'{"a":NaN}\n'
    elif corruption == 'whitespace':
        raw = b' ' + path.read_bytes()
    elif corruption == 'torn':
        raw = path.read_bytes()[:-5]
    elif corruption == 'extra-field':
        data['extra'] = 1
    elif corruption == 'missing-field':
        del data['generation']
    elif corruption == 'bool-version':
        data['schema_version'] = True
    elif corruption == 'shard-index':
        data['shard_index'] = 1
    elif corruption == 'num-shards':
        data['num_shards'] = 2
    elif corruption == 'plan-hash':
        data['plan_hash'] = '0' * 64
    elif corruption == 'row-decision':
        data['row']['outcome']['decision'] = 'sell'
    elif corruption == 'row-issuer':
        data['row']['outcome']['issuer_id'] = 'issuer:other'
    elif corruption == 'row-key':
        data['row']['key']['condition'] = '-+' if data['row']['key']['condition'] != '-+' else '--'
    elif corruption == 'filename':
        pass
    elif corruption == 'tokens-string':
        data['generation']['generated_token_ids'] = '11'
    elif corruption == 'token-hash':
        data['generation']['generated_token_sha256'] = '0' * 64
    elif corruption == 'payload-decision':
        data['generation']['json_payload'] = '{"decision":"sell","reason":"evidence"}'
        data['generation']['generated_text'] = data['generation']['json_payload'] + '<eos>'
    elif corruption == 'flags':
        data['generation']['reason_valid'] = False
    elif corruption == 'success-parse-failure':
        data['generation']['json_payload'] = '{malformed'
    elif corruption == 'payload-failure-mismatch':
        # Rewrite as an invalid_schema failure row whose payload parses as invalid_json.
        data['generation']['json_payload'] = '{malformed'
        data['generation']['generated_text'] = 'partial'
        data['generation']['decision'] = None
        data['generation']['reason'] = None
        data['generation']['decision_complete'] = False
        data['generation']['schema_complete'] = False
        data['generation']['reason_valid'] = False
        data['generation']['finish_reason'] = 'eos'
        data['generation']['failure_type'] = 'invalid_schema'
        data['generation']['error_message'] = 'execution diagnostic'
        data['row']['outcome']['decision'] = None
        data['row']['outcome']['decision_complete'] = False
        data['row']['outcome']['schema_complete'] = False
        data['row']['outcome']['reason_valid'] = False
        data['row']['outcome']['finish_reason'] = 'eos'
        data['row']['outcome']['failure_type'] = 'invalid_schema'
    elif corruption == 'failed-primary-set':
        # A failed row that retained a non-null primary decision.
        data['generation']['json_payload'] = ''
        data['generation']['generated_text'] = 'partial'
        data['generation']['decision'] = 'buy'
        data['generation']['reason'] = None
        data['generation']['decision_complete'] = False
        data['generation']['schema_complete'] = False
        data['generation']['reason_valid'] = False
        data['generation']['finish_reason'] = 'exception'
        data['generation']['failure_type'] = 'exception'
        data['generation']['error_message'] = 'execution diagnostic'
        data['row']['outcome']['decision'] = 'buy'
        data['row']['outcome']['decision_complete'] = False
        data['row']['outcome']['schema_complete'] = False
        data['row']['outcome']['reason_valid'] = False
        data['row']['outcome']['finish_reason'] = 'exception'
        data['row']['outcome']['failure_type'] = 'exception'
    elif corruption == 'provenance-schema':
        data['generation']['provenance']['schema_sha256'] = '0' * 64
    elif corruption == 'provenance-policy-hash':
        data['generation']['provenance']['generation_policy_sha256'] = '0' * 64
    elif corruption == 'provenance-hf':
        data['generation']['provenance']['hf_controls']['do_sample'] = True
    elif corruption == 'provenance-missing':
        del data['generation']['provenance']['stop_token_ids']
    if corruption == 'filename':
        path.rename(_record_path(parent_dir, '0' * 64 + '.json'))
    elif raw is not None:
        path.write_bytes(raw)
    elif corruption != 'filename':
        path.write_bytes(canonical_json_bytes(data) + b'\n')
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


def parent_artifact_first_record(parent_dir):
    """Lexicographically first record name (fast failure for record corruptions)."""
    return min(os.listdir(parent_dir / 'shard-0-of-1' / 'records'))


def test_missing_record_rejected(parent_dir, parent_bundle):
    name = parent_artifact_first_record(parent_dir)
    _record_path(parent_dir, name).unlink()
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


def test_extra_record_rejected(parent_dir, parent_bundle):
    name = parent_artifact_first_record(parent_dir)
    data = _record_path(parent_dir, name).read_bytes()
    (_record_path(parent_dir, 'f' * 64 + '.json')).write_bytes(data)
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


# -------------------------------------------------------------------- metadata


@pytest.mark.parametrize('corruption', ['outer-extra', 'outer-missing', 'binding-extra',
    'binding-missing', 'identity-mismatch', 'manifest', 'protocol-hash', 'template-hash',
    'model-hash', 'code-hash', 'backend-hash', 'policy-hash', 'schema-hash',
    'body-template-hash', 'template-missing', 'wrapper-type', 'protocol-kind',
    'protocol-members', 'protocol-planned', 'protocol-sampling', 'protocol-claims',
    'protocol-roles', 'protocol-evidence', 'protocol-timeout', 'protocol-route',
    'protocol-bool-int'])
def test_corrupt_metadata(parent_dir, parent_bundle, corruption):
    path = parent_dir / 'shard-0-of-1.execution_metadata.json'
    data = json.loads(path.read_bytes())
    identity = data['identity']
    bindings = data['bindings']
    if corruption == 'outer-extra':
        data['extra'] = 1
    elif corruption == 'outer-missing':
        del data['inputs_manifest_sha256']
    elif corruption == 'binding-extra':
        bindings['extra'] = {}
    elif corruption == 'binding-missing':
        del bindings['model']
    elif corruption == 'identity-mismatch':
        identity['model_sha256'] = '0' * 64
    elif corruption == 'manifest':
        data['inputs_manifest_sha256'] = '0' * 64
    elif corruption == 'protocol-hash':
        identity['protocol_sha256'] = '0' * 64
    elif corruption == 'template-hash':
        identity['template_sha256'] = '0' * 64
    elif corruption == 'model-hash':
        identity['model_sha256'] = '0' * 64
    elif corruption == 'code-hash':
        identity['code_sha256'] = '0' * 64
    elif corruption == 'backend-hash':
        identity['backend_sha256'] = '0' * 64
    elif corruption == 'policy-hash':
        identity['generation_policy_sha256'] = '0' * 64
    elif corruption == 'schema-hash':
        identity['schema_sha256'] = '0' * 64
    elif corruption == 'body-template-hash':
        bindings['template']['body_template_sha256'] = '0' * 64
    elif corruption == 'template-missing':
        del bindings['template']['actual_wrapper_record']
    elif corruption == 'wrapper-type':
        bindings['template']['actual_wrapper_record'] = 'not a record'
    elif corruption == 'protocol-kind':
        bindings['protocol']['kind'] = 'diagnostic_smoke_v1'
    elif corruption == 'protocol-members':
        bindings['protocol']['members'] = 500
    elif corruption == 'protocol-planned':
        bindings['protocol']['planned_global'] = 2011
    elif corruption == 'protocol-sampling':
        bindings['protocol']['sampling'] = 'sampled_pairs'
    elif corruption == 'protocol-claims':
        bindings['protocol']['research_eligible'] = True
    elif corruption == 'protocol-roles':
        bindings['protocol']['role_assignment'] = '0' * 64
    elif corruption == 'protocol-evidence':
        bindings['protocol']['evidence'] = '0' * 64
    elif corruption == 'protocol-timeout':
        bindings['protocol']['timeout_budget_seconds_per_row'] = 181.0
    elif corruption == 'protocol-route':
        bindings['protocol']['model_route'] = 'harmony_no_tools'
    elif corruption == 'protocol-bool-int':
        bindings['protocol']['research_eligible'] = 0
    path.write_bytes(canonical_json_bytes(data) + b'\n')
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


@pytest.mark.parametrize('change', [{'timeout_seconds': True}, {'timeout_seconds': '180'},
    {'timeout_seconds': float('nan')}, {'timeout_seconds': 0}, {'timeout_seconds': -1},
    {'max_new_tokens': True}, {'max_new_tokens': 0}, {'use_cache': 1},
    {'pad_token_id': True}, {'pad_token_id': HEAD}, {'channel_policy': 'unsupported'},
    {'channel_policy': 'unknown'}, {'extra-field': 1}])
def test_bad_policy(parent_dir, parent_bundle, change):
    path = parent_dir / 'shard-0-of-1.execution_metadata.json'
    data = json.loads(path.read_bytes())
    policy = data['bindings']['generation_policy']['policy']
    if 'extra-field' in change:
        policy['extra_field'] = change['extra-field']
    else:
        policy[next(iter(change))] = change[next(iter(change))]
    # Keep the binding hash consistent so the field-level check is exercised;
    # a NaN policy is nonfinite JSON and is rejected before any hash check.
    is_nan = any(isinstance(v, float) and v != v for v in change.values())
    if not is_nan:
        data['identity']['generation_policy_sha256'] = sha256_json(data['bindings']['generation_policy'])
    _rewrite(path, data, allow_nan=is_nan)
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


# ------------------------------------------------------------- provenance rows


@pytest.mark.parametrize('mutation', ['tokenizer', 'tokenizer-info', 'schema-bytes',
    'grammar', 'head-size', 'stop-ids', 'stop-out-of-range', 'hf-controls'])
def test_provenance_row_mismatch(parent_dir, parent_bundle, mutation):
    name = parent_artifact_first_record(parent_dir)
    path = _record_path(parent_dir, name)
    data = json.loads(path.read_bytes())
    prov = data['generation']['provenance']
    if mutation == 'tokenizer':
        prov['tokenizer_sha256'] = '0' * 64
    elif mutation == 'tokenizer-info':
        prov['tokenizer_info_sha256'] = '0' * 64
    elif mutation == 'schema-bytes':
        prov['schema_bytes_sha256'] = '0' * 64
    elif mutation == 'grammar':
        prov['grammar_sha256'] = '0' * 64
    elif mutation == 'head-size':
        prov['head_vocab_size'] = HEAD + 1
    elif mutation == 'stop-ids':
        prov['stop_token_ids'] = [199998]
    elif mutation == 'stop-out-of-range':
        prov['stop_token_ids'] = [HEAD]
    elif mutation == 'hf-controls':
        prov['hf_controls']['do_sample'] = True
    path.write_bytes(canonical_json_bytes(data) + b'\n')
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)


# --------------------------------------------------------------------- summary


@pytest.mark.parametrize('corruption', ['class-counts', 'failure-counts', 'finish-counts',
    'plan-hash', 'assigned', 'executed', 'missing', 'complete-shard', 'complete-global',
    'gate-true', 'research-eligible', 'steering-true', 'kind', 'shard-index', 'num-shards',
    'planned', 'bool-int', 'noncanonical', 'extra-field', 'missing-field'])
def test_corrupt_summary(parent_dir, parent_bundle, corruption):
    path = parent_dir / 'shard-0-of-1.summary.json'
    data = json.loads(path.read_bytes())
    raw = None
    if corruption == 'noncanonical':
        raw = b' ' + path.read_bytes()
    elif corruption == 'class-counts':
        data['class_counts']['buy'] += 1
    elif corruption == 'failure-counts':
        data['failure_counts']['none'] -= 1
    elif corruption == 'finish-counts':
        data['finish_counts']['eos'] -= 1
    elif corruption == 'plan-hash':
        data['plan_hash'] = '0' * 64
    elif corruption == 'assigned':
        data['assigned_count'] = 2011
    elif corruption == 'executed':
        data['executed_count'] = 2011
    elif corruption == 'missing':
        data['missing_count'] = 1
    elif corruption == 'complete-shard':
        data['complete_shard'] = False
    elif corruption == 'complete-global':
        data['complete_global'] = False
    elif corruption == 'gate-true':
        data['gates']['no_op'] = True
    elif corruption == 'research-eligible':
        data['research_eligible'] = True
    elif corruption == 'steering-true':
        data['steering_flips_claimed'] = True
    elif corruption == 'kind':
        data['kind'] = 'diagnostic_smoke_v1'
    elif corruption == 'shard-index':
        data['shard_index'] = 1
    elif corruption == 'num-shards':
        data['num_shards'] = 2
    elif corruption == 'planned':
        data['planned_global'] = 2011
    elif corruption == 'bool-int':
        data['complete_shard'] = 1
    elif corruption == 'extra-field':
        data['extra'] = 1
    elif corruption == 'missing-field':
        del data['finish_counts']
    path.write_bytes(raw if raw is not None else canonical_json_bytes(data) + b'\n')
    with pytest.raises(ValueError):
        _load(parent_dir, parent_bundle)
