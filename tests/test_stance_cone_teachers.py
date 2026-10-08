"""Full-plan synthetic teachers with actual CPU tokenizer/grammar, no weights."""
from dataclasses import replace
from types import MappingProxyType
import json

import pytest

from test_stance_baseline_adapter import bundle
from test_stance_baseline_parent import parent_bundle
from test_structured_output import make_tokenizer
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.stance_baseline_parent import CompletedBaseline
from llm_bias.core.inference.structured_output import compile_decision_grammar, StructuredGenerationResult
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.stance_cone_teachers import compile_cone_teachers, write_teacher_pack


@pytest.fixture(scope='module')
def case(parent_bundle):
    inputs, plan = parent_bundle['inputs'], parent_bundle['plan']
    tokenizer = make_tokenizer()
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    cap = compile_decision_grammar(tokenizer, len(tokenizer), [tokenizer.eos_token_id])
    wrapper = WrapperPolicy(use_chat_template=True, add_special_tokens=False)
    pair = inputs.pairs[0]
    prompt = render_decision_prompt(tokenizer, inputs.members[0], pair, wrapper_policy=wrapper)
    metadata = json.loads(json.dumps(parent_bundle['metadata']))
    metadata['bindings']['template'] = dict(body_template_sha256=prompt.template_sha256,
                                            actual_wrapper_record=prompt.wrapper_policy)
    gp = metadata['bindings']['generation_policy']
    provenance = dict(backend='xgrammar', backend_version=cap.backend_version,
        byte_policy=cap.byte_policy, schema_sha256=cap.schema_sha256,
        schema_bytes_sha256=cap.schema_bytes_sha256, tokenizer_sha256=cap.tokenizer_sha256,
        tokenizer_info_sha256=cap.tokenizer_info_sha256, head_vocab_size=cap.head_vocab_size,
        stop_token_ids=list(cap.stop_token_ids), compiler_policy=dict(strict_mode=True,
        any_order=False, any_whitespace=True), grammar_correction='xgrammar-0.2.8-minLength-json-escapes-v1',
        grammar_sha256=cap.grammar_sha256, generation_policy=gp['policy'],
        hf_controls=gp['hf_controls'], generation_policy_sha256=sha256_json(gp))
    # The pure entry point still checks exact full plan identity.
    from llm_bias.core.stance_baseline_plan import build_baseline_plan
    identity = replace(plan.identity, template_sha256=sha256_json(metadata['bindings']['template']))
    plan = build_baseline_plan(inputs, identity)
    metadata['identity'] = identity.to_dict()
    records = {}
    rows = []
    from test_stance_baseline_parent import _row_for
    for key in plan.keys:
        decision = 'sell' if key.condition == '--' else 'buy'
        payload = '{"decision": "%s", \n"reason": "full reason \\n with escape"}' % decision
        ids = tuple(tokenizer.encode(payload, add_special_tokens=False)) + (tokenizer.eos_token_id,)
        generated = StructuredGenerationResult(tokenizer.decode(ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False), ids, sha256_json(list(ids)), payload,
            decision, 'full reason \n with escape', True, True, True, 'eos', None, .01,
            canonical_json_bytes(provenance))
        rows.append(_row_for(inputs.issuer_by_ticker, key, generated))
        records[sha256_json(key.to_dict()) + '.json'] = canonical_json_bytes(dict(
            schema_version=1, plan_hash=plan.plan_hash, shard_index=0, num_shards=1,
            row={}, generation=generated.to_dict())) + b'\n'
    # Trusted synthetic snapshot, not a production authenticity attestation.
    parent = CompletedBaseline(plan, tuple(rows), sha256_json('synthetic parent'),
        canonical_json_bytes(metadata) + b'\n', b'{}\n', canonical_json_bytes({'synthetic': sha256_json('raw')}),
        MappingProxyType(records))
    return inputs, parent, tokenizer


def changed(case, condition, change):
    inputs, parent, tokenizer = case
    fit = min(t for t, r in inputs.roles['assignments'].items() if r == 'fit')
    key = next(k for k in parent.plan.keys if k.ticker == fit and k.condition == condition)
    records = dict(parent._record_bytes)
    filename = sha256_json(key.to_dict()) + '.json'
    data = json.loads(records[filename])
    data['generation'] = change(parent.generation_for(key)).to_dict()
    records[filename] = canonical_json_bytes(data) + b'\n'
    return inputs, replace(parent, _record_bytes=MappingProxyType(records)), tokenizer


def test_full_coverage_and_full_masks(case):
    pack = compile_cone_teachers(*case)
    assert pack['manifest']['accepted'] is True
    assert len(pack['teachers']) == 906
    assert pack['manifest']['purpose_counts'] == dict(addition=302, ablation=302, retain=302)
    roles = {t for t, r in case[0].roles['assignments'].items() if r == 'fit'}
    source_hash = sha256_json(pack['sources'])
    review_hash = sha256_json(pack['review'])
    for teacher, review in zip(pack['teachers'], pack['review']['cells']):
        assert teacher['ticker'] in roles
        mask = teacher['response_mask']
        start = review['prompt_token_count']
        assert mask == [False] * start + [True] * review['response_token_count']
        assert case[2].eos_token_id not in teacher['token_ids'][start:]
        assert teacher['source_sha256'] == source_hash
        assert teacher['review_sha256'] == review_hash
        assert teacher['review_sha256'] == sha256_json(pack['review'])
        assert review['removed_terminal_stop_ids'] == [case[2].eos_token_id]
        payload = case[2].decode(teacher['token_ids'][start:], clean_up_tokenization_spaces=False)
        assert '\n"reason"' in payload and '\\n with escape' in payload
    assert pack['review']['adaptation'] == 'cross_evidence_finance_targets_v1'


def test_wrong_class_unavailable_not_partial_pack(case):
    def wrong_class(g):
        payload = g.json_payload.replace('buy', 'sell')
        ids = tuple(case[2].encode(payload, add_special_tokens=False)) + (case[2].eos_token_id,)
        return replace(g, decision='sell', json_payload=payload, generated_token_ids=ids,
            generated_token_sha256=sha256_json(list(ids)), generated_text=case[2].decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    mutated = changed(case, '++', wrong_class)
    pack = compile_cone_teachers(*mutated)
    assert not pack['manifest']['accepted'] and pack['teachers'] == []
    assert len(pack['unavailable']) == 1


@pytest.mark.parametrize('mutation', [
    lambda g: replace(g, generated_text=g.generated_text + ' '),
    lambda g: replace(g, json_payload=g.json_payload.replace('\n', '')),
    lambda g: replace(g, failure_type='truncated', decision=None, reason=None,
                      finish_reason='token_budget', schema_complete=False),
])
def test_bad_needed_source_unavailable(case, mutation):
    pack = compile_cone_teachers(*changed(case, '+-', mutation))
    assert not pack['manifest']['accepted'] and not pack['teachers']


def test_harmony_explicitly_unsupported(case):
    inputs, parent, tok = case
    metadata = parent.metadata
    metadata['bindings']['generation_policy']['policy']['channel_policy'] = 'harmony_no_tools'
    pack = compile_cone_teachers(inputs, replace(parent, _metadata_bytes=canonical_json_bytes(metadata) + b'\n'), tok)
    assert pack['manifest']['status'] == 'unsupported_harmony_full_channel_teacher_policy'
    assert pack['teachers'] == [] and pack['manifest']['planned_count'] == 906


def test_tokenizer_and_template_mismatch_reject(case):
    tok = make_tokenizer()
    tok.chat_template = 'changed {{ messages[0]["content"] }}'
    with pytest.raises(ValueError, match='binding|template|tokenizer'):
        compile_cone_teachers(case[0], case[1], tok)


def test_fresh_pack_and_hashes(case, tmp_path):
    pack = compile_cone_teachers(*case)
    root = tmp_path / 'pack'
    write_teacher_pack(root, pack, {'synthetic': True})
    assert len((root / 'teachers.jsonl').read_text().splitlines()) == 906
    from llm_bias.core.artifact_paths import sha256_bytes
    manifest = json.loads((root / 'manifest.json').read_text())
    assert all(sha256_bytes((root / name).read_bytes()) == digest
               for name, digest in manifest['file_sha256'].items())
    with pytest.raises(FileExistsError):
        write_teacher_pack(root, pack, {})


@pytest.mark.parametrize('payload', [
    '{"decision":"buy","decision":"sell","reason":"x"}',
    '{"reason":"x","decision":"buy"}',
    '{"decision":"buy","reason":"x"} trailing',
    '{"decision":"buy","reason":"   "}',
    '{"decision":"buy","reason":"unfinished',
    '{"decision":"buy","reason":"<eos> control"}',
])
def test_strict_full_response_rejects_no_salvage(case, payload):
    from llm_bias.core.stance_cone_teachers import _response
    from llm_bias.core.inference.structured_output import _check_capability
    _, parent, tok = case
    cap = compile_decision_grammar(tok, len(tok), [tok.eos_token_id])
    g = parent.generation_for(parent.plan.keys[0])
    ids = tuple(tok.encode(payload, add_special_tokens=False)) + (tok.eos_token_id,)
    mutated = replace(g, generated_token_ids=ids, generated_token_sha256=sha256_json(list(ids)),
                      generated_text=tok.decode(ids, clean_up_tokenization_spaces=False), json_payload=payload)
    with pytest.raises(ValueError):
        _response(mutated, tok, cap, _check_capability(cap), None)


def test_explicit_stop_and_no_stop_boundaries(case):
    from llm_bias.core.stance_cone_teachers import _response
    from llm_bias.core.inference.structured_output import _check_capability
    _, parent, tok = case
    cap = compile_decision_grammar(tok, len(tok), [tok.eos_token_id])
    g = parent.generation_for(parent.plan.keys[0])
    ids = g.generated_token_ids[:-1]
    plain = replace(g, generated_token_ids=ids, generated_token_sha256=sha256_json(list(ids)),
                    generated_text=g.json_payload, finish_reason='schema_complete')
    assert _response(plain, tok, cap, _check_capability(cap), None) == (ids, ())
    with pytest.raises(ValueError, match='explicit_eos'):
        _response(replace(g, finish_reason='schema_complete'), tok, cap, _check_capability(cap), None)


def test_pairings_are_exact_same_ticker(case):
    pack = compile_cone_teachers(*case)
    expected = {'addition': ('--', '++', 'buy'), 'ablation': ('++', '--', 'sell'),
                'retain': ('+-', '+-', 'buy')}
    for cell in pack['review']['cells']:
        input_condition, target_condition, decision = expected[cell['purpose']]
        assert cell['input_key']['ticker'] == cell['target_key']['ticker'] == cell['ticker']
        assert (cell['input_key']['condition'], cell['target_key']['condition'],
                cell['target_decision']) == (input_condition, target_condition, decision)


def test_cli_only_tokenizer_no_model_weight_loader(tmp_path):
    import importlib.util
    path = __import__('pathlib').Path(__file__).resolve().parents[1] / 'scripts/compile_stance_cone_teachers.py'
    spec = importlib.util.spec_from_file_location('compiler_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    (checkpoint / 'tokenizer.json').write_text('{}')
    from llm_bias.core.artifact_paths import sha256_bytes
    record = dict(resolved_path=str(checkpoint), metadata_file_sha256={
        p.name: sha256_bytes(p.read_bytes()) for p in checkpoint.iterdir()})
    assert module.verify_checkpoint_metadata(checkpoint, record) == checkpoint
    (checkpoint / 'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='metadata differs'):
        module.verify_checkpoint_metadata(checkpoint, record)
    assert 'load_model' not in vars(module) and 'AutoModelForCausalLM' not in vars(module)


def test_grammar_source_identity_mismatch_reject(case):
    def mutation(g):
        provenance = g.provenance
        provenance['grammar_sha256'] = '0' * 64
        return replace(g, _provenance_bytes=canonical_json_bytes(provenance))
    with pytest.raises(ValueError, match='grammar_sha256'):
        compile_cone_teachers(*changed(case, '++', mutation))


def test_effective_policy_interface_binds_per_row(case):
    from dataclasses import dataclass
    from llm_bias.core.inference.structured_output import StructuredGenerationPolicy, _hf_controls

    @dataclass(frozen=True, slots=True)
    class EffectiveFixture(CompletedBaseline):
        def row_policy_for(self, key):
            return self.generation_for(key).provenance['generation_policy']

        def row_source_for(self, key):
            return dict(source='synthetic_effective', key=key.to_dict(), recovery_sha256='a' * 64)

    inputs, parent, tok = case
    fit = min(t for t, role in inputs.roles['assignments'].items() if role == 'fit')
    key = next(k for k in parent.plan.keys if k.ticker == fit and k.condition == '+-')
    records = dict(parent._record_bytes)
    g = parent.generation_for(key)
    provenance = g.provenance
    policy = dict(provenance['generation_policy'], max_new_tokens=1024)
    controls = _hf_controls(StructuredGenerationPolicy(**policy), (tok.eos_token_id,))
    provenance.update(generation_policy=policy, hf_controls=controls,
        generation_policy_sha256=sha256_json(dict(policy=policy, hf_controls=controls)))
    data = json.loads(records[sha256_json(key.to_dict()) + '.json'])
    data['generation'] = replace(g, _provenance_bytes=canonical_json_bytes(provenance)).to_dict()
    records[sha256_json(key.to_dict()) + '.json'] = canonical_json_bytes(data) + b'\n'
    effective = EffectiveFixture(parent.plan, parent.rows, parent.parent_sha256,
        parent._metadata_bytes, parent._summary_bytes, parent._file_hashes_bytes, MappingProxyType(records))
    pack = compile_cone_teachers(inputs, effective, tok)
    assert pack['manifest']['accepted']
    row = next(row for row in pack['sources']['selected_records'] if row['key'] == key.to_dict())
    assert row['policy']['policy']['max_new_tokens'] == 1024
    assert row['source']['source'] == 'synthetic_effective'


def test_actual_merged_metadata_shape_reports_harmony_unsupported(case):
    inputs, parent, tok = case
    metadata = parent.metadata
    metadata['bindings']['generation_policy']['policy']['channel_policy'] = 'harmony_no_tools'
    effective = replace(parent, _metadata_bytes=canonical_json_bytes({
        'kind': 'effective_merged_baseline_v1', 'original_metadata': metadata,
        'recovery_registration': {}}) + b'\n')
    pack = compile_cone_teachers(inputs, effective, None)
    assert pack['manifest']['status'] == 'unsupported_harmony_full_channel_teacher_policy'
    assert pack['manifest']['unavailable_count'] == 906
    assert pack['teachers'] == []
