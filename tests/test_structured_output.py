"""CPU contract tests using the installed xgrammar matcher, never a checkpoint."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LogitsProcessorList, PreTrainedTokenizerFast, StoppingCriteriaList
import xgrammar as xgr

from llm_bias.core.artifact_paths import sha256_bytes, sha256_json
from llm_bias.core.inference.structured_output import (
    CANONICAL_SCHEMA_PATH,
    FiniteLegalTokenGuard,
    NoLegalTokenError,
    StructuredGenerationPolicy,
    compile_decision_grammar,
    generate_structured,
    load_decision_schema,
    validate_decision_payload,
)
from llm_bias.core.prompt_input.decision_prompt import DECISION_SCHEMA


def make_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    extras = ['<unk>', '<pad>', '<eos>', '<stop>', '<|analysis|>',
              '<|final|>', '<thought>', '"decision":', '"extra":']
    vocab = {token: i for i, token in enumerate(alphabet + extras)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>', eos_token='<eos>',
        additional_special_tokens=['<stop>', '<|analysis|>', '<|final|>', '<thought>'],
    )


@pytest.fixture(scope='module')
def tokenizer():
    return make_tokenizer()


@pytest.fixture(scope='module')
def capability(tokenizer):
    return compile_decision_grammar(
        tokenizer, len(tokenizer) + 35,
        [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<stop>')],
    )


def policy(tokenizer, **kwargs):
    return replace(StructuredGenerationPolicy(
        max_new_tokens=200, use_cache=True, pad_token_id=tokenizer.pad_token_id,
        timeout_seconds=30.0, channel_policy='plain_json',
    ), **kwargs)


class FakeGenerate:
    """Small HF-shaped greedy loop; a real forward hook changes the chosen enum."""
    def __init__(self, tokenizer, capability, *, decision='buy', reason='market evidence',
                 logits_mode='normal', raise_at=None, stop_id=None):
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(vocab_size=capability.head_vocab_size,
                                      eos_token_id=list(capability.stop_token_ids))
        self.generation_config = SimpleNamespace(eos_token_id=list(capability.stop_token_ids))
        self.capability = capability
        self.decision = decision
        self.reason = reason
        self.logits_mode = logits_mode
        self.raise_at = raise_at
        self.stop_id = stop_id or tokenizer.eos_token_id
        self.block = torch.nn.Identity()
        self.calls = []
        self.selected = []
        self.mask_checks = []

    def target_ids(self, decision=None):
        text = json.dumps({'decision': decision or self.decision, 'reason': self.reason},
                          ensure_ascii=False, separators=(',', ':'))
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        assert isinstance(kwargs['logits_processor'], LogitsProcessorList)
        assert isinstance(kwargs['stopping_criteria'], StoppingCriteriaList)
        assert kwargs['do_sample'] is False and kwargs['num_beams'] == 1
        assert kwargs['generation_config'].forced_eos_token_id is None
        assert kwargs['generation_config'].forced_bos_token_id is None
        assert kwargs['generation_config'].repetition_penalty == 1.0
        assert kwargs['generation_config'].suppress_tokens is None
        assert kwargs['generation_config'].output_scores is False
        ids = input_ids.clone()
        self.selected = []
        for step in range(kwargs['max_new_tokens']):
            hidden = self.block(torch.zeros(1, 1, 1))
            if step == self.raise_at:
                raise RuntimeError('fake forward failure')
            choice = 'sell' if hidden.item() > 0 else self.decision
            target = self.target_ids(choice)
            scores = torch.full((1, self.capability.head_vocab_size), -20.0)
            desired = target[step] if step < len(target) else self.stop_id
            scores[0, desired] = 50.0
            # NUL is illegal both outside and inside a JSON string. Channel and
            # padded IDs must stay masked throughout; premature EOS must too.
            nul = self.tokenizer.encode('\x00', add_special_tokens=False)[0]
            channel = self.tokenizer.convert_tokens_to_ids('<|analysis|>')
            scores[0, nul] = 100.0
            scores[0, channel] = 200.0
            scores[0, self.capability.head_vocab_size - 1] = 300.0
            scores[0, self.stop_id] = 90.0
            if self.logits_mode == 'no_legal':
                scores.fill_(-float('inf'))
                scores[0, nul] = 100.0
            elif self.logits_mode == 'nan':
                scores.fill_(float('nan'))
            elif self.logits_mode == 'mixed_nan':
                scores[0, self.tokenizer.encode(' ', add_special_tokens=False)[0]] = float('nan')
            elif self.logits_mode == 'positive_infinity':
                scores[0, desired] = float('inf')
            scores = kwargs['logits_processor'](ids, scores)
            self.mask_checks.append(tuple(torch.isneginf(scores[0, i]).item()
                                          for i in (nul, channel, self.capability.head_vocab_size - 1)))
            selected = scores.argmax(dim=-1).reshape(1, 1)
            ids = torch.cat((ids, selected), dim=-1)
            self.selected.append(selected.item())
            if kwargs['stopping_criteria'](ids, scores).all():
                break
            if selected.item() in kwargs['eos_token_id']:
                break
        return SimpleNamespace(sequences=ids)


def run(fake, tokenizer, capability, *, controls=None, transforms=None):
    wrapped = SimpleNamespace(hf_model=fake, layers=[fake.block])
    return generate_structured(
        wrapped, tokenizer, torch.tensor([[tokenizer.pad_token_id]]), capability,
        policy=controls or policy(tokenizer), transforms=transforms,
    )


def masked(capability, prefix, tokenizer):
    matcher = xgr.GrammarMatcher(capability.compiled_grammar)
    for token in tokenizer.encode(prefix, add_special_tokens=False):
        assert matcher.accept_token(token)
    bitmask = xgr.allocate_token_bitmask(1, capability.head_vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    scores = torch.zeros(1, capability.head_vocab_size)
    xgr.apply_token_bitmask_inplace(scores, bitmask, vocab_size=capability.head_vocab_size,
                                   backend='cpu')
    return matcher, scores


def test_one_schema_identity_and_provenance(tokenizer, capability):
    schema = load_decision_schema()
    assert schema == DECISION_SCHEMA
    assert list(schema['properties']) == ['decision', 'reason']
    assert schema['properties']['reason'] == {'type': 'string', 'minLength': 1}
    assert capability.schema_sha256 == sha256_json({'schema': DECISION_SCHEMA,
                                                    'key_order': ['decision', 'reason']})
    assert capability.schema_bytes_sha256 == sha256_bytes(CANONICAL_SCHEMA_PATH.read_bytes())
    assert capability.backend_version == '0.2.8'
    schema['properties']['decision']['enum'].pop()
    assert load_decision_schema()['properties']['decision']['enum'] == ['buy', 'sell']


@pytest.mark.parametrize('mutation', ['enum', 'pattern', 'order', 'required', 'extra', 'bool_length'])
def test_reject_different_schema(tmp_path, mutation):
    schema = json.loads(json.dumps(DECISION_SCHEMA))
    if mutation == 'enum':
        schema['properties']['decision']['enum'] = ['buy']
    elif mutation == 'pattern':
        schema['properties']['reason']['pattern'] = '\\S'
    elif mutation == 'order':
        schema['properties'] = dict(reversed(list(schema['properties'].items())))
    elif mutation == 'required':
        schema['required'] = ['decision']
    elif mutation == 'bool_length':
        schema['properties']['reason']['minLength'] = True
    else:
        schema['title'] = 'different'
    path = tmp_path / 'schema.json'
    path.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match='canonical'):
        load_decision_schema(path)


@pytest.mark.parametrize('head,stops', [(0, [1]), (True, [1]), (2, [1]),
    (400, []), (400, [1, 1]), (400, [True]), (400, [-1]), (400, [400]), (400, [1.5])])
def test_invalid_vocab_and_stop_bindings(tokenizer, head, stops):
    with pytest.raises(ValueError):
        compile_decision_grammar(tokenizer, head, stops)


@pytest.mark.parametrize('decision', ['buy', 'sell'])
def test_both_enum_paths_with_adversarial_preferences(tokenizer, capability, decision):
    fake = FakeGenerate(tokenizer, capability, decision=decision,
                        reason='escapes: "quote" \\ newline\n市場 café')
    result = run(fake, tokenizer, capability)
    assert result.decision == decision and result.reason == fake.reason
    assert result.decision_complete and result.schema_complete and result.reason_valid
    assert result.failure_type is None and result.finish_reason == 'eos'
    assert result.generated_token_ids == tuple(fake.selected)
    assert result.generated_token_ids[:-1] == tuple(fake.target_ids())
    assert len(tokenizer.encode(decision, add_special_tokens=False)) > 1
    assert result.generated_token_sha256 == sha256_json(fake.selected)
    assert result.generated_text.endswith('<eos>')
    assert json.loads(result.json_payload)['decision'] == decision
    assert all(all(checks) for checks in fake.mask_checks)
    assert result.provenance['head_vocab_size'] == capability.head_vocab_size
    assert result.provenance['stop_token_ids'] == list(capability.stop_token_ids)
    assert result.provenance['schema_sha256'] == capability.schema_sha256
    assert result.provenance['tokenizer_sha256'] == capability.tokenizer_sha256
    assert result.provenance['backend_version'] == '0.2.8'
    assert result.provenance['compiler_policy'] == {'strict_mode': True, 'any_order': False,
                                                 'any_whitespace': True}
    assert result.elapsed_seconds >= 0
    json.dumps(result.to_dict(), allow_nan=False)  # Compact, tensor-free artifact record.


def test_real_matcher_blocks_missing_duplicate_extra_reordered_and_early_stop(tokenizer, capability):
    for prefix, forbidden in [('', '<eos>'), ('{"decision":"buy"', '}'),
                              ('{"decision":"buy",', '"decision":'),
                              ('{"decision":"buy",', '"extra":'),
                              ('{', '"extra":')]:
        matcher, scores = masked(capability, prefix, tokenizer)
        token = tokenizer.convert_tokens_to_ids(forbidden)
        assert torch.isneginf(scores[0, token])
        assert not matcher.accept_token(token)
    for text in ['{"reason":"x","decision":"buy"}',
                 '{"decision":"buy","reason":"x","decision":"sell"}',
                 '{"decision":"buy","reason":"x","extra":1}',
                 '{"decision":"buy"}', '{"decision":"buy","reason":""}']:
        matcher = xgr.GrammarMatcher(capability.compiled_grammar)
        assert not all(matcher.accept_token(i) for i in tokenizer.encode(text, add_special_tokens=False))
    matcher, scores = masked(capability, '{"decision":"sell","reason":"x"}', tokenizer)
    assert matcher.is_completed() and not matcher.is_terminated()
    assert set(torch.where(torch.isfinite(scores[0]))[0].tolist()) == set(capability.stop_token_ids)
    assert matcher.accept_token(capability.stop_token_ids[1])
    assert matcher.is_terminated()


@pytest.mark.parametrize('text,failure,schema_complete', [
    ('```json\n{"decision":"buy","reason":"x"}\n```', 'invalid_json', False),
    ('prefix {"decision":"buy","reason":"x"}', 'invalid_json', False),
    ('{"decision":"buy","reason":"x"} tail', 'invalid_json', False),
    ('{"decision":"buy","decision":"sell","reason":"x"}', 'invalid_schema', False),
    ('{"reason":"x","decision":"buy"}', 'invalid_schema', False),
    ('{"decision":"hold","reason":"x"}', 'invalid_schema', False),
    ('{"decision":"buy"}', 'invalid_schema', False),
    ('{"decision":"buy","reason":5}', 'invalid_schema', False),
    ('{"decision":"buy","reason":"x","extra":1}', 'invalid_schema', False),
    ('{"decision":"buy","reason":""}', 'invalid_reason', False),
    ('{"decision":"buy","reason":" \\n\\t"}', 'invalid_reason', True),
    ('{"decision":"buy","reason":NaN}', 'invalid_json', False),
    ('{"decision":"buy",', 'invalid_json', False),
])
def test_strict_validator_never_salvages(text, failure, schema_complete):
    outcome = validate_decision_payload(text)
    assert outcome.failure_type == failure
    assert outcome.schema_complete is schema_complete
    assert outcome.decision is None and outcome.reason is None and not outcome.reason_valid


def test_parser_whitespace_and_unicode():
    result = validate_decision_payload(' \n{"decision":"sell","reason":"市場\\n\\\"yes\\\""}\t')
    assert result.failure_type is None and result.decision == 'sell'
    assert result.reason == '市場\n"yes"'


def test_semantic_blank_reason_is_not_primary_valid(tokenizer, capability):
    result = run(FakeGenerate(tokenizer, capability, reason=' \n\t'), tokenizer, capability)
    assert result.schema_complete and result.decision_complete
    assert not result.reason_valid and result.failure_type == 'invalid_reason'
    assert result.decision is None and result.reason is None


@pytest.mark.parametrize('use_cache', [True, False])
def test_exact_budget_valid_and_unfinished_is_truncated(tokenizer, capability, use_cache):
    fake = FakeGenerate(tokenizer, capability)
    budget = len(fake.target_ids())
    controls = policy(tokenizer, max_new_tokens=budget, use_cache=use_cache)
    complete = run(fake, tokenizer, capability, controls=controls)
    assert complete.finish_reason == 'schema_complete' and complete.failure_type is None
    assert complete.decision == 'buy' and complete.generated_text.endswith('}')
    assert complete.generated_text == complete.json_payload
    assert fake.calls[-1]['use_cache'] is use_cache
    assert complete.provenance['generation_policy']['use_cache'] is use_cache
    partial = run(fake, tokenizer, capability, controls=replace(controls, max_new_tokens=budget - 1))
    assert partial.failure_type == 'truncated' and partial.finish_reason == 'token_budget'
    assert partial.decision is None and partial.reason is None
    assert not partial.schema_complete and not partial.decision_complete


@pytest.mark.parametrize('mode', ['no_legal', 'nan', 'positive_infinity'])
def test_no_legal_finite_token_outcome_and_typed_guard(tokenizer, capability, mode):
    result = run(FakeGenerate(tokenizer, capability, logits_mode=mode), tokenizer, capability)
    if mode == 'positive_infinity':
        # Other finite legal candidates remain; nonfinite preferences cannot win.
        assert result.failure_type != 'exception'
    else:
        assert result.failure_type == result.finish_reason == 'no_legal_token'
        assert result.generated_token_ids == () and result.decision is None
    processors = LogitsProcessorList([
        capability.new_processor(prompt_length=1, deadline=None), FiniteLegalTokenGuard(),
    ])
    with pytest.raises(NoLegalTokenError):
        processors(torch.tensor([[tokenizer.pad_token_id]]),
                   torch.full((1, capability.head_vocab_size), float('nan')))


def test_nan_cannot_win_over_finite_legal_token(tokenizer, capability):
    result = run(FakeGenerate(tokenizer, capability, logits_mode='mixed_nan'), tokenizer, capability)
    assert result.decision == 'buy' and result.failure_type is None


def test_fresh_matchers_and_processor_state(tokenizer, capability):
    first = capability.new_processor(prompt_length=1, deadline=None)
    second = capability.new_processor(prompt_length=1, deadline=None)
    assert first is not second and first.matcher is not second.matcher
    first.matcher.accept_token(tokenizer.encode('{', add_special_tokens=False)[0])
    assert not second.matcher.is_completed()
    buy_fake = FakeGenerate(tokenizer, capability)
    sell_fake = FakeGenerate(tokenizer, capability, decision='sell')
    buy = run(buy_fake, tokenizer, capability)
    sell = run(sell_fake, tokenizer, capability)
    assert buy.decision == 'buy' and sell.decision == 'sell'
    assert buy_fake.calls[0]['logits_processor'][0] is not sell_fake.calls[0]['logits_processor'][0]
    assert buy.generated_token_sha256 != sell.generated_token_sha256


def test_hook_effect_and_cleanup_success_no_legal_and_exception(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability)
    result = run(fake, tokenizer, capability, transforms={0: lambda hidden: hidden + 1})
    assert result.decision == 'sell' and not fake.block._forward_hooks
    fake.logits_mode = 'no_legal'
    result = run(fake, tokenizer, capability, transforms={0: lambda hidden: hidden + 1})
    assert result.failure_type == 'no_legal_token' and not fake.block._forward_hooks
    fake.logits_mode = 'normal'
    fake.raise_at = 5
    result = run(fake, tokenizer, capability, transforms={0: lambda hidden: hidden + 1})
    assert result.failure_type == 'exception' and result.finish_reason == 'exception'
    assert result.decision is None and len(result.generated_token_ids) == 5
    assert result.error_message == 'RuntimeError: fake forward failure'
    assert not fake.block._forward_hooks


def test_monotonic_timeout_keeps_partial_ids_and_cleans_hooks(tokenizer, capability, monkeypatch):
    import llm_bias.core.inference.structured_output as module
    clock = iter([0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock, 2.0))
    fake = FakeGenerate(tokenizer, capability)
    result = run(fake, tokenizer, capability, controls=policy(tokenizer, timeout_seconds=1.0),
                 transforms={0: lambda hidden: hidden + 1})
    assert result.failure_type == result.finish_reason == 'timeout'
    assert result.decision is None and result.generated_token_ids
    assert result.elapsed_seconds == 2.0 and not fake.block._forward_hooks


def test_explicit_unsupported_policy_never_calls_generate(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability)
    result = run(fake, tokenizer, capability, controls=policy(tokenizer, channel_policy='unsupported'))
    assert result.failure_type == 'unsupported_channel' and result.finish_reason == 'unsupported'
    assert not fake.calls and result.generated_token_ids == ()


@pytest.mark.parametrize('marker', ['<|analysis|>', '<|final|>', '<thought>', '<eos>'])
def test_returned_channel_or_interior_stop_not_stripped(tokenizer, capability, marker):
    fake = FakeGenerate(tokenizer, capability)
    content = fake.target_ids()
    inserted = content[:4] + [tokenizer.convert_tokens_to_ids(marker)] + content[4:]
    fake.generate = lambda ids, **kwargs: torch.cat((ids, torch.tensor([inserted])), dim=-1)
    result = run(fake, tokenizer, capability)
    assert result.failure_type == 'unsupported_channel' and result.decision is None
    assert marker in result.generated_text and marker in result.json_payload


def test_only_recognized_terminal_stop_suffix_removed(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability, stop_id=capability.stop_token_ids[1])
    result = run(fake, tokenizer, capability)
    assert result.failure_type is None and result.generated_text.endswith('<stop>')
    assert result.json_payload.endswith('}') and '<stop>' not in result.json_payload


@pytest.mark.parametrize('changes', [{'max_new_tokens': 0}, {'max_new_tokens': True},
    {'use_cache': 1}, {'timeout_seconds': 0}, {'timeout_seconds': float('nan')},
    {'pad_token_id': -1}, {'channel_policy': 'auto'}])
def test_policy_validation_before_generate(tokenizer, capability, changes):
    fake = FakeGenerate(tokenizer, capability)
    with pytest.raises(ValueError):
        run(fake, tokenizer, capability, controls=policy(tokenizer, **changes))
    assert not fake.calls


def test_tokenizer_identity_mismatch_rejected(tokenizer, capability):
    changed = make_tokenizer()
    changed.add_tokens(['new token'])
    fake = FakeGenerate(changed, capability)
    with pytest.raises(ValueError, match='tokenizer'):
        run(fake, changed, capability)
    assert not fake.calls


def test_batch_and_head_shape_checked(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability)
    with pytest.raises(ValueError, match='batch'):
        generate_structured(SimpleNamespace(hf_model=fake), tokenizer,
                            torch.ones(2, 3, dtype=torch.long), capability, policy=policy(tokenizer))
    processor = capability.new_processor(prompt_length=1, deadline=None)
    with pytest.raises(ValueError, match='head'):
        processor(torch.ones(1, 1, dtype=torch.long), torch.zeros(1, capability.head_vocab_size + 1))


@pytest.mark.parametrize('reason_literal,expected', [
    ('"\\u5e02\\u5834"', '市場'),
    ('"\\\"\\\\\\/\\b\\f\\n\\r\\t"', '"\\/\b\f\n\r\t'),
    ('"\\u0000"', '\x00'),
])
def test_corrected_real_backend_accepts_json_escape_paths(tokenizer, capability, reason_literal, expected):
    text = '{"decision":"buy","reason":' + reason_literal + '}'
    matcher = xgr.GrammarMatcher(capability.compiled_grammar)
    for token in tokenizer.encode(text, add_special_tokens=False):
        assert matcher.accept_token(token)
    assert matcher.is_completed()
    assert validate_decision_payload(text).reason == expected
    for byte in range(32):
        matcher, scores = masked(capability, '{"decision":"buy","reason":"', tokenizer)
        token = tokenizer.encode(chr(byte), add_special_tokens=False)[0]
        assert torch.isneginf(scores[0, token])
        assert not matcher.accept_token(token)


def test_grammar_correction_fails_closed_on_unknown_conversion(tokenizer):
    from llm_bias.core.inference.structured_output import _correct_reason_string_rule
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, stop_token_ids=[tokenizer.eos_token_id])
    compiler = xgr.GrammarCompiler(info)
    unexpected = compiler.compile_builtin_json_grammar()
    with pytest.raises(ValueError, match='cannot safely compile'):
        _correct_reason_string_rule(compiler, unexpected)


def test_state_isolation_with_unfinished_then_complete(tokenizer, capability):
    unfinished_fake = FakeGenerate(tokenizer, capability)
    unfinished = run(unfinished_fake, tokenizer, capability,
                     controls=policy(tokenizer, max_new_tokens=4))
    complete = run(FakeGenerate(tokenizer, capability, decision='sell'), tokenizer, capability)
    assert unfinished.failure_type == 'truncated'
    assert complete.failure_type is None and complete.decision == 'sell'


def test_unsupported_harmony_has_no_slug_policy_or_salvage(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability)
    wrapped = SimpleNamespace(hf_model=fake, model_name='openai/gpt-oss-20b')
    result = generate_structured(wrapped, tokenizer, torch.tensor([[tokenizer.pad_token_id]]),
                                 capability, policy=policy(tokenizer, channel_policy='unsupported'))
    assert result.failure_type == 'unsupported_channel' and not fake.calls
    with pytest.raises(TypeError, match='policy'):
        generate_structured(wrapped, tokenizer, torch.tensor([[tokenizer.pad_token_id]]), capability)
    assert validate_decision_payload('analysis\nfinal\n{"decision":"buy","reason":"x"}').failure_type == 'invalid_json'


@pytest.mark.parametrize('stop_count', [1, 2])
def test_terminal_stop_suffix_and_interior_stop_after_complete(tokenizer, capability, stop_count):
    fake = FakeGenerate(tokenizer, capability)
    ids = fake.target_ids() + list(capability.stop_token_ids[:stop_count])
    fake.generate = lambda prompt, **kwargs: torch.cat((prompt, torch.tensor([ids])), dim=-1)
    result = run(fake, tokenizer, capability)
    assert result.failure_type is None and result.finish_reason == 'eos'
    assert result.generated_token_ids == tuple(ids) and result.json_payload.endswith('}')
    ids += tokenizer.encode('tail', add_special_tokens=False)
    result = run(fake, tokenizer, capability)
    assert result.failure_type == 'unsupported_channel'
    assert '<eos>' in result.json_payload and result.decision is None


def test_invalid_transform_mapping_cannot_leak_hook(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability)
    with pytest.raises(ValueError, match='transforms'):
        run(fake, tokenizer, capability, transforms={0: lambda state: state + 1,
                                                    1: lambda state: state})
    assert not fake.block._forward_hooks and not fake.calls


def test_exact_budget_blank_reason_finish(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability, reason=' ')
    result = run(fake, tokenizer, capability,
                 controls=policy(tokenizer, max_new_tokens=len(fake.target_ids())))
    assert result.finish_reason == 'schema_complete' and result.failure_type == 'invalid_reason'
    assert result.schema_complete and not result.reason_valid and result.decision is None


def test_checkpoint_generation_defaults_cannot_force_an_answer(tokenizer, capability):
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    fake = FakeGenerate(tokenizer, capability)
    run(fake, tokenizer, capability)
    sent = fake.calls[0]
    hostile = GenerationConfig(
        forced_bos_token_id=10, forced_eos_token_id=11,
        suppress_tokens=[12], bad_words_ids=[[13]], stop_strings=['buy'],
        sequence_bias={(14,): 3.0}, repetition_penalty=2.0, do_sample=True,
        penalty_alpha=0.6,
    )
    holder = SimpleNamespace(generation_config=hostile)
    actual, kwargs = GenerationMixin._prepare_generation_config(
        holder, sent['generation_config'],
        **{key: value for key, value in sent.items() if key != 'generation_config'},
    )
    assert actual.forced_bos_token_id is None and actual.forced_eos_token_id is None
    assert actual.bad_words_ids is None and actual.suppress_tokens is None
    assert actual.stop_strings is None and actual.sequence_bias is None
    assert actual.repetition_penalty == 1.0 and actual.do_sample is False
    assert actual.num_beams == 1 and actual.eos_token_id == list(capability.stop_token_ids)
    assert actual.get_generation_mode().value == 'greedy_search'
    assert isinstance(kwargs['logits_processor'], LogitsProcessorList)


def byte_ids(tokenizer, data):
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, stop_token_ids=[tokenizer.eos_token_id])
    pieces = {piece: i for i, piece in enumerate(info.decoded_vocab) if len(piece) == 1}
    return [pieces[bytes([byte])] for byte in data]


def raw_fake(tokenizer, capability, content):
    fake = FakeGenerate(tokenizer, capability)
    fake.target_ids = lambda decision=None: byte_ids(tokenizer, content)
    return fake


@pytest.mark.parametrize('literal', [r'\ud800', r'\udfff', r'\ud800x', r'\udc00\ud800'])
def test_lone_surrogate_is_invalid_and_utf8_persistable(tokenizer, capability, literal, tmp_path):
    raw = ('{"decision":"buy","reason":"' + literal + '"}').encode()
    result = run(raw_fake(tokenizer, capability, raw), tokenizer, capability)
    assert result.failure_type == 'invalid_reason'
    assert result.decision is None and result.reason is None and not result.reason_valid
    assert result.schema_complete and result.decision_complete
    from llm_bias.core.artifact_paths import canonical_json_bytes
    (tmp_path / 'result.json').write_bytes(canonical_json_bytes(result.to_dict()))
    assert validate_decision_payload(raw.decode()).failure_type == 'invalid_reason'


@pytest.mark.parametrize('reason', ['😀', '市場 café', '\ufffd', r'\ud83d\ude00'])
@pytest.mark.parametrize('decision', ['buy', 'sell'])
def test_correct_unicode_and_literal_replacement_survive(tokenizer, capability, reason, decision):
    raw = ('{"decision":"' + decision + '","reason":"' + reason + '"}').encode()
    result = run(raw_fake(tokenizer, capability, raw), tokenizer, capability)
    assert result.failure_type is None and result.decision == decision
    assert result.reason == json.loads(raw)['reason']


@pytest.mark.parametrize('data', [b'\xed\xa0\x80', b'\xff', b'\xc0\xaf'])
def test_raw_invalid_utf8_cannot_be_replacement_decoded_valid(tokenizer, capability, data):
    raw = b'{"decision":"buy","reason":"' + data + b'"}'
    fake = raw_fake(tokenizer, capability, raw)
    # Bypass callback enforcement too: malformed custom return must still be
    # checked against the matcher and exact bytes, never replacement-decoded.
    fake.generate = lambda ids, **kwargs: torch.cat((ids, torch.tensor([fake.target_ids()])), dim=-1)
    result = run(fake, tokenizer, capability)
    assert result.decision is None and result.failure_type is not None
    assert result.decode_error is not None
    assert result.generated_token_ids


def test_public_provenance_and_exports_are_defensive(tokenizer, capability):
    result = run(FakeGenerate(tokenizer, capability), tokenizer, capability)
    original = json.dumps(result.provenance, sort_keys=True)
    exported = result.provenance
    exported['hf_controls']['eos_token_id'].clear()
    exported['generation_policy']['use_cache'] = 'forged'
    result.to_dict()['provenance']['stop_token_ids'].append(999)
    assert json.dumps(result.provenance, sort_keys=True) == original
    assert isinstance(result._provenance_bytes, bytes)


@pytest.mark.parametrize('forgery', ['grammar', 'info', 'metadata'])
def test_forged_compiled_capability_is_rejected_before_forward(tokenizer, capability, forgery):
    other = make_tokenizer()
    other.add_tokens(['other'])
    info = xgr.TokenizerInfo.from_huggingface(
        other if forgery == 'info' else tokenizer,
        vocab_size=capability.head_vocab_size, stop_token_ids=list(capability.stop_token_ids),
    )
    compiler = xgr.GrammarCompiler(info)
    compiled = (compiler.compile_json_schema(load_decision_schema(), strict_mode=True, any_order=False)
                if forgery == 'info' else compiler.compile_builtin_json_grammar())
    forged = replace(capability, compiled_grammar=compiled)
    if forgery == 'metadata':
        forged = replace(forged, grammar_sha256=sha256_bytes(str(compiled.grammar).encode()),
                         tokenizer_info_sha256=sha256_json(json.loads(info.serialize_json())))
    fake = FakeGenerate(tokenizer, capability)
    with pytest.raises(ValueError, match='capability|compiled'):
        run(fake, tokenizer, forged)
    assert not fake.calls


@pytest.mark.parametrize('mode,root', [('no_legal', 'no_legal_token'), ('exception', 'exception'),
                                      ('timeout', 'timeout')])
def test_first_failure_survives_secondary_decode_error(tokenizer, capability, monkeypatch, mode, root):
    fake = FakeGenerate(tokenizer, capability, logits_mode='no_legal' if mode == 'no_legal' else 'normal',
                        raise_at=3 if mode == 'exception' else None)
    if mode == 'timeout':
        import llm_bias.core.inference.structured_output as module
        clock = iter([0.0, 0.0, 0.0, 0.0, 2.0])
        monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock, 2.0))
    def bad_decode(*args, **kwargs):
        raise RuntimeError('secondary text decode')
    monkeypatch.setattr(tokenizer, 'decode', bad_decode)
    result = run(fake, tokenizer, capability, controls=policy(tokenizer, timeout_seconds=1.0))
    assert result.failure_type == result.finish_reason == root
    assert result.error_message and 'secondary text decode' not in result.error_message
    assert result.decode_error == 'RuntimeError: secondary text decode'
    if mode != 'no_legal':
        assert result.generated_token_ids


@pytest.mark.parametrize('stop', ['ordinary', 'padded'])
def test_ordinary_or_padded_stop_is_not_a_capability(tokenizer, stop):
    stop_id = tokenizer.encode('b', add_special_tokens=False)[0] if stop == 'ordinary' else len(tokenizer) + 5
    with pytest.raises(ValueError, match='stop'):
        compile_decision_grammar(tokenizer, len(tokenizer) + 35, [stop_id])


@pytest.mark.parametrize('kind', ['tokenizer', 'head', 'generation_stop', 'config_stop'])
def test_attached_model_declarations_checked_before_forward(tokenizer, capability, kind):
    fake = FakeGenerate(tokenizer, capability)
    fake.config = SimpleNamespace(vocab_size=capability.head_vocab_size,
                                  eos_token_id=list(capability.stop_token_ids))
    fake.generation_config = SimpleNamespace(eos_token_id=list(capability.stop_token_ids))
    wrapper = SimpleNamespace(hf_model=fake, layers=[fake.block], tokenizer=tokenizer)
    if kind == 'tokenizer':
        wrapper.tokenizer = make_tokenizer()
        wrapper.tokenizer.add_tokens(['mismatch'])
    elif kind == 'head':
        fake.get_output_embeddings = lambda: torch.nn.Linear(4, capability.head_vocab_size + 1)
    elif kind == 'generation_stop':
        fake.generation_config.eos_token_id = [tokenizer.pad_token_id]
    else:
        fake.config.eos_token_id = [tokenizer.pad_token_id]
    with pytest.raises(ValueError, match='tokenizer|head|stop'):
        generate_structured(wrapper, tokenizer, torch.tensor([[tokenizer.pad_token_id]]),
                            capability, policy=policy(tokenizer))
    assert not fake.calls and not fake.block._forward_hooks


@pytest.mark.parametrize('kind', ['over_budget', 'prefix', 'early_partial', 'float_ids'])
def test_malformed_returned_sequence_is_exception(tokenizer, capability, kind):
    fake = FakeGenerate(tokenizer, capability)
    content = fake.target_ids()
    controls = policy(tokenizer)
    if kind == 'over_budget':
        controls = policy(tokenizer, max_new_tokens=len(content) - 1)
    elif kind == 'early_partial':
        content = content[:5]
    def malformed(ids, **kwargs):
        result = torch.cat((ids, torch.tensor([content])), dim=-1)
        if kind == 'prefix':
            result[0, 0] = tokenizer.eos_token_id
        elif kind == 'float_ids':
            result = result.float()
        return result
    fake.generate = malformed
    result = run(fake, tokenizer, capability, controls=controls)
    assert result.failure_type == result.finish_reason == 'exception'
    assert result.decision is None and result.error_message


@pytest.mark.parametrize('control', list(range(32)))
def test_compound_added_c0_token_cannot_bypass_backend(tokenizer, control):
    added = make_tokenizer()
    bad = 'prefix' + chr(control) + 'suffix'
    added.add_tokens([bad])
    cap = compile_decision_grammar(added, len(added), [added.eos_token_id])
    processor = cap.new_processor(prompt_length=1, deadline=None)
    prefix = added.encode('{"decision":"buy","reason":"', add_special_tokens=False)
    scores = processor(torch.tensor([[added.pad_token_id] + prefix]), torch.zeros(1, len(added)))
    assert torch.isneginf(scores[0, added.convert_tokens_to_ids(bad)])


def test_correction_requires_exact_pinned_native_shape(tokenizer):
    from llm_bias.core.inference.structured_output import _correct_reason_string_rule
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, stop_token_ids=[tokenizer.eos_token_id])
    compiler = xgr.GrammarCompiler(info)
    native = compiler.compile_json_schema(load_decision_schema(), strict_mode=True, any_order=False,
                                          any_whitespace=True)
    changed = compiler.compile_grammar(xgr.Grammar.from_ebnf(
        str(native.grammar).replace(r'"\"sell\""', r'"\"hold\""')))
    with pytest.raises(ValueError, match='cannot safely compile'):
        _correct_reason_string_rule(compiler, changed)


def tiny_hf(tokenizer, capability, target, *, fail_after=None, mode=None):
    """Actual random tiny CPU causal model and HF generate, not a fake loop."""
    from transformers import GPT2Config, GPT2LMHeadModel
    hf = GPT2LMHeadModel(GPT2Config(
        vocab_size=capability.head_vocab_size, n_positions=256, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=list(capability.stop_token_ids),
    )).eval()
    count = 0
    def prefer(_module, _args, output):
        nonlocal count
        if count == fail_after:
            raise RuntimeError('tiny HF root forward failure')
        output.logits.fill_(-20.0)
        chosen = target[count] if count < len(target) else tokenizer.eos_token_id
        output.logits[:, -1, chosen] = 50.0
        output.logits[:, -1, capability.head_vocab_size - 1] = 100.0
        if mode == 'no_legal' and count == 3:
            output.logits.fill_(-float('inf'))
        count += 1
    hf.register_forward_hook(prefer)
    return SimpleNamespace(hf_model=hf, tokenizer=tokenizer, layers=list(hf.transformer.h))


@pytest.fixture
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('decision,use_cache', [('buy', True), ('buy', False), ('sell', True), ('sell', False)])
@pytest.mark.parametrize('reason', [b'\xed\xa0\x80', b'\\ud800', '😀 �'.encode(), b'\\ud83d\\ude00'])
def test_actual_tiny_hf_unicode_paths(tokenizer, capability, single_cpu_thread, decision, use_cache, reason,
                                      tmp_path):
    content = b'{"decision":"' + decision.encode() + b'","reason":"' + reason + b'"}'
    model = tiny_hf(tokenizer, capability, byte_ids(tokenizer, content))
    result = generate_structured(model, tokenizer, torch.tensor([[tokenizer.pad_token_id]]), capability,
                                 policy=policy(tokenizer, use_cache=use_cache))
    if reason == b'\\ud800':
        assert result.failure_type == 'invalid_reason' and result.decision is None
    elif reason == b'\xed\xa0\x80':
        assert result.failure_type is not None and result.decision is None and result.decode_error
    else:
        assert result.failure_type is None and result.decision == decision
    from llm_bias.core.artifact_paths import canonical_json_bytes
    (tmp_path / 'tiny.json').write_bytes(canonical_json_bytes(result.to_dict()))


@pytest.mark.parametrize('root', ['exception', 'no_legal_token'])
def test_actual_tiny_hf_root_and_secondary_error(tokenizer, capability, single_cpu_thread, monkeypatch, root):
    target = tokenizer.encode('{"decision":"buy","reason":"x"}', add_special_tokens=False)
    model = tiny_hf(tokenizer, capability, target, fail_after=3 if root == 'exception' else None,
                    mode='no_legal' if root == 'no_legal_token' else None)
    monkeypatch.setattr(tokenizer, 'decode', lambda *a, **kw: (_ for _ in ()).throw(ValueError('secondary')))
    result = generate_structured(model, tokenizer, torch.tensor([[tokenizer.pad_token_id]]), capability,
                                 policy=policy(tokenizer), transforms={0: lambda hidden: hidden})
    assert result.failure_type == result.finish_reason == root
    assert result.generated_token_ids and result.decode_error == 'ValueError: secondary'
    assert 'secondary' not in result.error_message
    assert not model.layers[0]._forward_hooks


@pytest.mark.parametrize('missing', ['tokenizer', 'head', 'stops', 'all'])
def test_missing_declarations_need_explicit_verified_binding(tokenizer, capability, missing):
    from llm_bias.core.inference.structured_output import UnsupportedModelBindingError, VerifiedModelBinding
    fake = FakeGenerate(tokenizer, capability)
    if missing in ('tokenizer', 'all'):
        del fake.tokenizer
        # Leave the loop's test-only tokenizer accessible through its closure.
        target = FakeGenerate(tokenizer, capability).target_ids()
        fake.generate = lambda ids, **kwargs: torch.cat((ids, torch.tensor([target])), dim=-1)
    if missing in ('head', 'all'):
        fake.config.vocab_size = None
    if missing in ('stops', 'all'):
        fake.config.eos_token_id = fake.generation_config.eos_token_id = None
    wrapped = SimpleNamespace(hf_model=fake, layers=[fake.block])
    prompt = torch.tensor([[tokenizer.pad_token_id]])
    with pytest.raises(UnsupportedModelBindingError, match='missing'):
        generate_structured(wrapped, tokenizer, prompt, capability, policy=policy(tokenizer))
    assert not fake.calls
    verified = VerifiedModelBinding(fake, capability.tokenizer_sha256, capability.head_vocab_size,
                                    capability.stop_token_ids, 'checkpoint-free synthetic-loop contract')
    result = generate_structured(wrapped, tokenizer, prompt, capability, policy=policy(tokenizer),
                                 verified_binding=verified)
    assert result.failure_type is None and result.decision == 'buy'
    names = ['tokenizer', 'head', 'stops'] if missing == 'all' else [missing]
    assert result.provenance['model_binding']['attested_missing'] == names
    assert result.provenance['model_binding']['verification_reference'] == verified.verification_reference
    with pytest.raises(ValueError, match='verified model binding'):
        generate_structured(wrapped, tokenizer, prompt, capability, policy=policy(tokenizer),
                            verified_binding=replace(verified, hf_model=object()))


def test_verified_binding_never_overrides_declared_mismatch(tokenizer, capability):
    from llm_bias.core.inference.structured_output import VerifiedModelBinding
    fake = FakeGenerate(tokenizer, capability)
    fake.config.vocab_size += 1
    verified = VerifiedModelBinding(fake, capability.tokenizer_sha256, capability.head_vocab_size,
                                    capability.stop_token_ids, 'cannot override conflicting config')
    with pytest.raises(ValueError, match='head'):
        generate_structured(SimpleNamespace(hf_model=fake), tokenizer,
                            torch.tensor([[tokenizer.pad_token_id]]), capability,
                            policy=policy(tokenizer), verified_binding=verified)
    assert not fake.calls


@pytest.mark.parametrize('decoder', ['sequence', 'none', 'wordpiece'])
def test_untested_lossless_decoder_fails_explicitly(decoder):
    from llm_bias.core.inference.structured_output import UnsupportedTokenizerError
    tokenizer = make_tokenizer()
    tokenizer.backend_tokenizer.decoder = {
        'sequence': decoders.Sequence([decoders.ByteLevel()]),
        'none': None, 'wordpiece': decoders.WordPiece(),
    }[decoder]
    with pytest.raises(UnsupportedTokenizerError, match='unsupported lossless decoder'):
        compile_decision_grammar(tokenizer, len(tokenizer), [tokenizer.eos_token_id])


def test_prefix_space_adjustment_fails_explicitly():
    from llm_bias.core.inference.structured_output import UnsupportedTokenizerError
    tokenizer = make_tokenizer()
    tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    with pytest.raises(UnsupportedTokenizerError, match='without prefix adjustment'):
        compile_decision_grammar(tokenizer, len(tokenizer), [tokenizer.eos_token_id])


@pytest.mark.parametrize('mutation', ['grammar', 'info'])
def test_actual_native_handle_identity_is_checked(tokenizer, capability, monkeypatch, mutation):
    local = compile_decision_grammar(tokenizer, capability.head_vocab_size, capability.stop_token_ids)
    changed = make_tokenizer()
    if mutation == 'info':
        changed.add_tokens(['changed native identity'])
    info = xgr.TokenizerInfo.from_huggingface(changed, vocab_size=capability.head_vocab_size,
                                           stop_token_ids=list(capability.stop_token_ids))
    compiler = xgr.GrammarCompiler(info)
    compiled = compiler.compile_grammar(local.compiled_grammar.grammar) if mutation == 'info' else compiler.compile_builtin_json_grammar()
    monkeypatch.setattr(local.compiled_grammar, '_XGRObject__handle', compiled._handle)
    fake = FakeGenerate(tokenizer, capability)
    with pytest.raises(ValueError, match='actual compiled'):
        run(fake, tokenizer, local)
    assert not fake.calls


def test_provenance_is_frozen_before_execution(tokenizer, capability):
    fake = FakeGenerate(tokenizer, capability)
    original = fake.generate
    def mutate_controls(ids, **kwargs):
        result = original(ids, **kwargs)
        kwargs['eos_token_id'].clear()
        return result
    fake.generate = mutate_controls
    result = run(fake, tokenizer, capability)
    assert result.failure_type is None
    assert result.provenance['hf_controls']['eos_token_id'] == list(capability.stop_token_ids)


@pytest.mark.parametrize('root', ['exception', 'no_legal_token'])
def test_secondary_non_utf8_text_is_not_exported(tokenizer, capability, monkeypatch, root):
    from llm_bias.core.artifact_paths import canonical_json_bytes
    fake = FakeGenerate(tokenizer, capability, logits_mode='no_legal' if root == 'no_legal_token' else 'normal',
                        raise_at=3 if root == 'exception' else None)
    monkeypatch.setattr(tokenizer, 'decode', lambda *a, **kwargs: '\ud800')
    result = run(fake, tokenizer, capability)
    assert result.failure_type == result.finish_reason == root and result.decode_error
    assert result.generated_text == result.json_payload == ''
    canonical_json_bytes(result.to_dict())


@pytest.mark.parametrize('kind', ['tokenizer', 'head', 'config_stop', 'generation_stop'])
def test_actual_tiny_hf_binding_rejects_before_forward(tokenizer, capability, single_cpu_thread, kind):
    model = tiny_hf(tokenizer, capability, [], fail_after=0)
    if kind == 'tokenizer':
        model.tokenizer = make_tokenizer()
        model.tokenizer.add_tokens(['wrong tokenizer'])
    elif kind == 'head':
        model.hf_model.set_output_embeddings(torch.nn.Linear(8, capability.head_vocab_size + 1))
    elif kind == 'config_stop':
        model.hf_model.config.eos_token_id = tokenizer.pad_token_id
    else:
        model.hf_model.generation_config.eos_token_id = tokenizer.pad_token_id
    with pytest.raises(ValueError, match='tokenizer|head|stop'):
        generate_structured(model, tokenizer, torch.tensor([[tokenizer.pad_token_id]]), capability,
                            policy=policy(tokenizer))
    assert not model.layers[0]._forward_hooks


@pytest.mark.parametrize('control', [0, 1, 31])
def test_actual_tiny_hf_added_c0_is_masked(tokenizer, single_cpu_thread, control):
    added = make_tokenizer()
    bad = 'prefix' + chr(control) + 'suffix'
    added.add_tokens([bad])
    cap = compile_decision_grammar(added, len(added), [added.eos_token_id])
    prefix = added.encode('{"decision":"buy","reason":"', add_special_tokens=False)
    target = prefix + [added.convert_tokens_to_ids(bad)]
    model = tiny_hf(added, cap, target)
    result = generate_structured(model, added, torch.tensor([[added.pad_token_id]]), cap,
                                 policy=policy(added, max_new_tokens=len(target)))
    assert result.failure_type == 'truncated' and result.decision is None
    assert added.convert_tokens_to_ids(bad) not in result.generated_token_ids


@pytest.mark.parametrize('kind', ['over_budget', 'prefix', 'early_partial'])
def test_actual_tiny_hf_malformed_return_is_exception(tokenizer, capability, single_cpu_thread, kind):
    target = tokenizer.encode('{"decision":"buy","reason":"x"}', add_special_tokens=False)
    model = tiny_hf(tokenizer, capability, target)
    actual = model.hf_model.generate
    def malformed_return(ids, **kwargs):
        if kind == 'over_budget':
            kwargs['max_new_tokens'] += 1
        output = actual(ids, **kwargs)
        if kind == 'prefix':
            output[0, 0] = tokenizer.eos_token_id
        elif kind == 'early_partial':
            output = output[:, :6]
        return output
    model.hf_model.generate = malformed_return
    result = generate_structured(model, tokenizer, torch.tensor([[tokenizer.pad_token_id]]), capability,
                                 policy=policy(tokenizer, max_new_tokens=len(target)))
    assert result.failure_type == result.finish_reason == 'exception' and result.decision is None
    assert result.generated_token_ids
