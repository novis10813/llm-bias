"""Native tokenizer/xgrammar byte-integrity contracts; no checkpoints or GPU."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import xgrammar as xgr

from llm_bias.core.inference.structured_output import (
    StructuredGenerationPolicy, UnsupportedTokenizerError, VerifiedModelBinding,
    compile_decision_grammar, generate_structured,
)


BYTELEVEL_POLICY = 'hf-fast-bpe-bytelevel-strict-utf8-v1'
BYTEFALLBACK_POLICY = 'hf-fast-bpe-bytefallback-strict-utf8-v1'


def fallback_decoder():
    return decoders.Sequence([
        decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse(),
    ])


def make_tokenizer(*, extras=(), added=(), decoder=None, gap=False):
    tokens = [f'<0x{i:02X}>' for i in range(256)] + [
        '<unk>', '<pad>', '<eos>', '▁market', 'x▁y', '�',
    ] + list(extras)
    vocab = {token: i for i, token in enumerate(tokens)}
    if gap:
        vocab['gap'] = len(vocab) + 10
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>', byte_fallback=True))
    backend.decoder = decoder or fallback_decoder()
    backend.add_tokens(list(added))
    # Declared in the backend only, deliberately absent from HF named specials.
    backend.add_special_tokens([AddedToken('<backend_stop>', special=True),
                                AddedToken('<channel>', special=True)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                  pad_token='<pad>', eos_token='<eos>')


def compile_tokenizer(tokenizer, *, stops=None):
    return compile_decision_grammar(tokenizer, max(tokenizer.get_vocab().values()) + 4,
                                    stops or [tokenizer.eos_token_id])


def byte_ids(tokenizer, text):
    data = text.encode('utf-8') if isinstance(text, str) else text
    return [tokenizer.get_vocab()[f'<0x{byte:02X}>'] for byte in data]


class NativeGreedyLoop:
    """CPU HF-shaped harness that runs the real grammar processors/callbacks."""
    def __init__(self, tokenizer, capability, target, *, bypass=False):
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(vocab_size=capability.head_vocab_size,
                                      eos_token_id=list(capability.stop_token_ids))
        self.generation_config = SimpleNamespace(eos_token_id=list(capability.stop_token_ids))
        self.capability = capability
        self.target = target
        self.bypass = bypass
        self.calls = 0

    def generate(self, ids, **kwargs):
        self.calls += 1
        if self.bypass:
            return torch.cat((ids, torch.tensor([self.target])), dim=1)
        for i in range(kwargs['max_new_tokens']):
            scores = torch.full((1, self.capability.head_vocab_size), -float('inf'))
            scores[0, self.target[i]] = 1.0
            for bad in self.capability.blocked_token_ids:
                scores[0, bad] = 100.0
            scores = kwargs['logits_processor'](ids, scores)
            assert all(torch.isneginf(scores[0, bad]) for bad in self.capability.blocked_token_ids)
            ids = torch.cat((ids, scores.argmax(-1).reshape(1, 1)), dim=1)
            if kwargs['stopping_criteria'](ids, scores).all():
                break
        return ids


def generate(tokenizer, capability, target, *, bypass=False):
    fake = NativeGreedyLoop(tokenizer, capability, target, bypass=bypass)
    policy = StructuredGenerationPolicy(len(target), True, tokenizer.pad_token_id, 30.0, 'plain_json')
    return generate_structured(SimpleNamespace(hf_model=fake), tokenizer,
                               torch.tensor([[tokenizer.pad_token_id]]), capability, policy=policy)


def test_full_native_pipeline_is_admitted():
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    assert capability.byte_policy == BYTEFALLBACK_POLICY
    assert capability.compiled_grammar.tokenizer_info.vocab_type == xgr.VocabType.BYTE_FALLBACK


def test_backend_only_special_stop_is_admitted():
    tokenizer = make_tokenizer()
    stop = tokenizer.get_vocab()['<backend_stop>']
    assert stop not in tokenizer.all_special_ids
    capability = compile_tokenizer(tokenizer, stops=[tokenizer.eos_token_id, stop])
    assert stop in capability.special_token_ids and stop not in capability.blocked_token_ids
    assert tokenizer.get_vocab()['<channel>'] in capability.blocked_token_ids
    text = '{"decision":"buy","reason":"native"}'
    result = generate(tokenizer, capability, byte_ids(tokenizer, text) + [stop])
    assert result.decision == 'buy' and result.finish_reason == 'eos'
    assert result.json_payload == text


@pytest.mark.parametrize('decision', ['buy', 'sell'])
@pytest.mark.parametrize('with_stop', [False, True])
def test_native_unicode_generation_and_exact_budget(decision, with_stop):
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    reason = '市場 café 😀 � escapes: "quote" \\ newline\n'
    text = json.dumps({'decision': decision, 'reason': reason}, ensure_ascii=False,
                      separators=(',', ':'))
    target = byte_ids(tokenizer, text)
    if with_stop:
        target.append(tokenizer.eos_token_id)
    result = generate(tokenizer, capability, target)
    assert result.failure_type is None and result.decision == decision and result.reason == reason
    assert result.finish_reason == ('eos' if with_stop else 'schema_complete')
    assert result.generated_token_ids == tuple(target)
    assert result.json_payload == text
    assert result.provenance['byte_policy'] == BYTEFALLBACK_POLICY
    exported = result.provenance
    exported['byte_policy'] = 'forged'
    assert result.to_dict()['provenance']['byte_policy'] == BYTEFALLBACK_POLICY


def test_all_256_bytes_are_native_bytes_not_replacement_strings():
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    info = capability.compiled_grammar.tokenizer_info
    for byte in range(256):
        token_id = tokenizer.get_vocab()[f'<0x{byte:02X}>']
        assert info.decoded_vocab[token_id] == bytes([byte])
        assert token_id not in capability.blocked_token_ids
    # Native replacement decoding is not the vocabulary's byte representation.
    assert tokenizer.decode(byte_ids(tokenizer, b'\xff')) == '�'
    assert info.decoded_vocab[tokenizer.get_vocab()['�']] == '�'.encode()


@pytest.mark.parametrize('spelling,native,blocked', [
    ('<0x4a>', 'J', True), ('<0x0a>', '\n', True),
    ('<0X41>', '<0X41>', False), ('<0x1>', '<0x1>', False),
    ('<0x041>', '<0x041>', False), ('<0x4g>', '<0x4g>', True),
])
def test_native_byte_spelling_and_backend_disagreement(spelling, native, blocked):
    tokenizer = make_tokenizer(extras=[spelling])
    token_id = tokenizer.get_vocab()[spelling]
    assert fallback_decoder().decode([spelling]) == native
    assert tokenizer.decode([token_id], clean_up_tokenization_spaces=False) == native
    capability = compile_tokenizer(tokenizer)
    assert (token_id in capability.blocked_token_ids) is blocked
    if blocked:
        prefix = byte_ids(tokenizer, '{"decision":"buy","reason":"')
        processor = capability.new_processor(prompt_length=1, deadline=None)
        scores = processor(torch.tensor([[tokenizer.pad_token_id] + prefix]),
                           torch.zeros(1, capability.head_vocab_size))
        assert torch.isneginf(scores[0, token_id])
        with pytest.raises(ValueError, match='blocked'):
            processor.observe(torch.tensor([[tokenizer.pad_token_id] + prefix + [token_id]]))


@pytest.mark.parametrize('spelling,native', [('<0xff>', '�'), ('<0x+1>', '\x01'),
                                              ('<0x 1>', '<0x 1>'), ('<0x-1>', '<0x-1>')])
def test_native_spellings_that_backend_cannot_construct_fail_closed(spelling, native):
    tokenizer = make_tokenizer(extras=[spelling])
    assert fallback_decoder().decode([spelling]) == native
    with pytest.raises(UnsupportedTokenizerError, match='bytes could not be established'):
        compile_tokenizer(tokenizer)


@pytest.mark.parametrize('content,blocked', [('▁literal', True), ('<0x41>', True),
                                            ('ordinary', False), ('�', False)])
def test_added_contents_are_scrutinized_as_literals(content, blocked):
    tokenizer = make_tokenizer(added=[AddedToken(content, normalized=False)])
    capability = compile_tokenizer(tokenizer)
    token_id = tokenizer.get_vocab()[content]
    assert (token_id in capability.blocked_token_ids) is blocked
    if not blocked:
        prefix = byte_ids(tokenizer, '{"decision":"sell","reason":"')
        suffix = byte_ids(tokenizer, '"}')
        result = generate(tokenizer, capability, prefix + [token_id] + suffix)
        assert result.decision == 'sell' and result.reason == content
    else:
        assert tokenizer.decode([token_id], clean_up_tokenization_spaces=False) != content


def test_native_token_boundaries_and_mixed_ordinary_byte_added_sequences():
    tokenizer = make_tokenizer(added=[AddedToken('literal', normalized=False)])
    capability = compile_tokenizer(tokenizer)
    vocab = tokenizer.get_vocab()
    target = (byte_ids(tokenizer, '{"decision":"buy","reason":"')
              + [vocab['▁market'], vocab['x▁y']] + byte_ids(tokenizer, ' 😀 ')
              + [vocab['literal'], vocab['�']] + byte_ids(tokenizer, '"}'))
    text = '{"decision":"buy","reason":" marketx y 😀 literal�"}'
    assert tokenizer.backend_tokenizer.decode(target, skip_special_tokens=False) == text
    result = generate(tokenizer, capability, target)
    assert result.failure_type is None and result.json_payload == text
    assert result.reason == ' marketx y 😀 literal�'


@pytest.mark.parametrize('bad', [b'\xff', b'\xc3', b'\xed\xa0\x80', b'\xf0\x9f'])
def test_invalid_or_incomplete_utf8_never_becomes_primary_valid(bad):
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    data = b'{"decision":"buy","reason":"' + bad + b'"}'
    target = byte_ids(tokenizer, data)
    assert '�' in tokenizer.decode(target, clean_up_tokenization_spaces=False)
    result = generate(tokenizer, capability, target, bypass=True)
    assert result.failure_type is not None and result.decision is None and result.reason is None
    assert result.decode_error and 'UnicodeDecodeError' in result.decode_error


@pytest.mark.parametrize('literal', [r'\ud800', r'\udfff'])
def test_escaped_lone_surrogates_stay_invalid(literal):
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    target = byte_ids(tokenizer, '{"decision":"buy","reason":"' + literal + '"}')
    result = generate(tokenizer, capability, target)
    assert result.failure_type == 'invalid_reason' and result.decision is None
    json.dumps(result.to_dict()).encode('utf-8')


def test_runtime_hf_equality_remains_mandatory(monkeypatch):
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    decode = tokenizer.decode
    monkeypatch.setattr(tokenizer, 'decode', lambda *a, **kw: decode(*a, **kw).replace('buy', 'sell'))
    target = byte_ids(tokenizer, '{"decision":"buy","reason":"valid"}')
    result = generate(tokenizer, capability, target)
    assert result.failure_type == 'unsupported_tokenizer' and result.decision is None
    assert 'differs from exact backend token bytes' in result.decode_error


def test_secondary_decode_error_preserves_root_failure(monkeypatch):
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    decode = tokenizer.decode
    monkeypatch.setattr(tokenizer, 'decode', lambda *a, **kw: decode(*a, **kw) + '\ud800')
    target = [tokenizer.get_vocab()['<channel>']]
    result = generate(tokenizer, capability, target, bypass=True)
    assert result.failure_type == 'unsupported_channel'
    assert 'non-stop special token' in result.error_message
    assert result.decode_error and result.generated_text == result.json_payload == ''
    json.dumps(result.to_dict()).encode('utf-8')


@pytest.mark.parametrize('marker', ['<channel>', '<backend_stop>'])
def test_nonstop_backend_special_cannot_be_json_payload(marker):
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    target = (byte_ids(tokenizer, '{"decision":"buy","reason":"')
              + [tokenizer.get_vocab()[marker]] + byte_ids(tokenizer, '"}'))
    result = generate(tokenizer, capability, target, bypass=True)
    assert result.failure_type == 'unsupported_channel' and result.decision is None
    assert marker in result.json_payload


@pytest.mark.parametrize('kind', ['ordinary', 'padded', 'gap'])
def test_ordinary_padded_and_gap_stops_remain_invalid(kind):
    tokenizer = make_tokenizer(gap=True)
    head = max(tokenizer.get_vocab().values()) + 4
    stop = {'ordinary': tokenizer.get_vocab()['▁market'], 'padded': head - 1,
            'gap': tokenizer.get_vocab()['gap'] - 1}[kind]
    with pytest.raises(ValueError, match='known special'):
        compile_decision_grammar(tokenizer, head, [stop])
    capability = compile_tokenizer(tokenizer)
    assert set(range(head)) - set(tokenizer.get_vocab().values()) <= set(capability.blocked_token_ids)


@pytest.mark.parametrize('content', ['▁unsafe_stop', '<0x41>'])
def test_disagreeing_declared_stop_fails_compilation(content):
    tokenizer = make_tokenizer(added=[AddedToken(content, special=True)])
    # add_tokens preserves special:true even without a named HF special.
    assert tokenizer.get_vocab()[content] not in tokenizer.all_special_ids
    with pytest.raises(UnsupportedTokenizerError, match='stop token bytes disagree'):
        compile_tokenizer(tokenizer, stops=[tokenizer.get_vocab()[content]])


@pytest.mark.parametrize('mutation', ['copy', 'policy', 'specials', 'inplace_policy', 'inplace_specials'])
def test_factory_identity_covers_policy_and_special_union(mutation):
    tokenizer = make_tokenizer()
    capability = compile_tokenizer(tokenizer)
    if mutation == 'copy':
        forged = replace(capability)
    elif mutation in ('policy', 'specials'):
        changes = {'byte_policy': BYTELEVEL_POLICY} if mutation == 'policy' else {'special_token_ids': ()}
        forged = replace(capability, **changes)
    else:
        forged = capability
        name, value = ('byte_policy', BYTELEVEL_POLICY) if mutation == 'inplace_policy' else ('special_token_ids', ())
        object.__setattr__(forged, name, value)
    with pytest.raises(ValueError, match='factory-owned identity'):
        forged.new_processor(prompt_length=1, deadline=None)


def test_backend_stop_never_relaxes_model_stop_binding():
    tokenizer = make_tokenizer()
    stop = tokenizer.get_vocab()['<backend_stop>']
    capability = compile_tokenizer(tokenizer, stops=[stop])
    fake = NativeGreedyLoop(tokenizer, capability, [])
    fake.config.eos_token_id = [tokenizer.eos_token_id]
    verified = VerifiedModelBinding(fake, capability.tokenizer_sha256, capability.head_vocab_size,
                                    capability.stop_token_ids, 'synthetic binding')
    with pytest.raises(ValueError, match='stop declarations conflict'):
        generate_structured(SimpleNamespace(hf_model=fake), tokenizer,
                            torch.tensor([[tokenizer.pad_token_id]]), capability,
                            policy=StructuredGenerationPolicy(1, True, tokenizer.pad_token_id, 30., 'plain_json'),
                            verified_binding=verified)
    assert fake.calls == 0


def make_bytelevel_tokenizer(*, added=()):
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet()) + ['<pad>', '<eos>']
    backend = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=[]))
    backend.decoder = decoders.ByteLevel()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.add_tokens(list(added))
    backend.add_special_tokens([AddedToken('<backend_stop>', special=True),
                                AddedToken('<channel>', special=True)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='<eos>')


def test_plain_bytelevel_policy_and_backend_special_union_are_preserved():
    tokenizer = make_bytelevel_tokenizer()
    stop = tokenizer.get_vocab()['<backend_stop>']
    assert stop not in tokenizer.all_special_ids
    capability = compile_tokenizer(tokenizer, stops=[stop])
    text = '{"decision":"sell","reason":"bytelevel"}'
    result = generate(tokenizer, capability, tokenizer.encode(text, add_special_tokens=False) + [stop])
    assert capability.byte_policy == result.provenance['byte_policy'] == BYTELEVEL_POLICY
    assert result.failure_type is None and result.decision == 'sell'
    assert tokenizer.get_vocab()['<channel>'] in capability.blocked_token_ids


@pytest.mark.parametrize('family', ['bytelevel', 'bytefallback'])
def test_nul_bytes_are_compared_not_laundered(family):
    maker = make_bytelevel_tokenizer if family == 'bytelevel' else make_tokenizer
    tokenizer = maker(added=[AddedToken('before\x00after', normalized=False)])
    token_id = tokenizer.get_vocab()['before\x00after']
    assert tokenizer.decode([token_id], clean_up_tokenization_spaces=False) == 'before\x00after'
    capability = compile_tokenizer(tokenizer)
    backend_bytes = capability.compiled_grammar.tokenizer_info.decoded_vocab[token_id]
    # ByteLevel truncates this spelling in the installed backend; fallback does not.
    assert (token_id in capability.blocked_token_ids) is (family == 'bytelevel')
    assert (backend_bytes != b'before\x00after') is (family == 'bytelevel')
    prefix = (tokenizer.encode('{"decision":"buy","reason":"', add_special_tokens=False)
              if family == 'bytelevel' else byte_ids(tokenizer, '{"decision":"buy","reason":"'))
    processor = capability.new_processor(prompt_length=1, deadline=None)
    scores = processor(torch.tensor([[tokenizer.pad_token_id] + prefix]),
                       torch.zeros(1, capability.head_vocab_size))
    # Fallback's exact NUL remains illegal JSON even without a byte discrepancy.
    assert torch.isneginf(scores[0, token_id])


@pytest.mark.parametrize('decoder', [
    decoders.ByteFallback(), decoders.Sequence([decoders.ByteLevel()]),
    decoders.Sequence([decoders.Replace('▁', ' '), decoders.Fuse(), decoders.ByteFallback()]),
    decoders.Sequence([decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse(),
                       decoders.Strip(' ', 1, 0)]),
    decoders.Sequence([decoders.Replace('▁', '_'), decoders.ByteFallback(), decoders.Fuse()]),
    decoders.Metaspace(),
])
def test_unsupported_native_decoders_remain_unsupported(decoder):
    tokenizer = make_tokenizer(decoder=decoder)
    with pytest.raises(UnsupportedTokenizerError, match='unsupported lossless decoder'):
        compile_tokenizer(tokenizer)


def patch_descriptor(monkeypatch, tokenizer, mutate):
    """Alter only serialized metadata for negative validation; never supply bytes."""
    original = Tokenizer.to_str
    native = tokenizer.backend_tokenizer
    descriptor = json.loads(native.to_str())
    mutate(descriptor)
    monkeypatch.setattr(Tokenizer, 'to_str',
                        lambda self, *a, **kw: json.dumps(descriptor) if self is native else original(self, *a, **kw))


@pytest.mark.parametrize('mutation', ['regex', 'extra_stage_key', 'extra_pipeline_key',
                                     'missing_stage', 'not_dict', 'not_list', 'unigram', 'prefix'])
def test_malformed_or_untested_descriptors_fail_before_backend_compilation(monkeypatch, mutation):
    tokenizer = make_tokenizer()

    def mutate(record):
        decoder = record['decoder']
        if mutation == 'regex':
            decoder['decoders'][0]['pattern'] = {'Regex': '▁'}
        elif mutation == 'extra_stage_key':
            decoder['decoders'][1]['extra'] = True
        elif mutation == 'extra_pipeline_key':
            decoder['extra'] = True
        elif mutation == 'missing_stage':
            decoder['decoders'].pop()
        elif mutation == 'not_dict':
            record['decoder'] = ['ByteFallback']
        elif mutation == 'not_list':
            decoder['decoders'] = {'type': 'ByteFallback'}
        elif mutation == 'unigram':
            record['model']['type'] = 'Unigram'
        else:
            record['pre_tokenizer'] = {'type': 'ByteLevel', 'add_prefix_space': True}

    patch_descriptor(monkeypatch, tokenizer, mutate)
    with pytest.raises(UnsupportedTokenizerError, match='unsupported lossless decoder'):
        compile_tokenizer(tokenizer)


@pytest.mark.parametrize('kind', ['bool_id', 'string_id', 'negative_id', 'outside_id',
                                 'wrong_content', 'wrong_id', 'not_bool_special', 'duplicate',
                                 'not_dict', 'not_list'])
def test_malformed_backend_added_declarations_fail_explicitly(monkeypatch, kind):
    tokenizer = make_tokenizer()

    def mutate(backend):
        records = backend['added_tokens']
        record = next(item for item in records if item['content'] == '<backend_stop>')
        if kind == 'bool_id':
            record['id'] = True
        elif kind == 'string_id':
            record['id'] = str(record['id'])
        elif kind == 'negative_id':
            record['id'] = -1
        elif kind == 'outside_id':
            record['id'] = 10000
        elif kind == 'wrong_content':
            record['content'] = 'absent'
        elif kind == 'wrong_id':
            record['id'] = tokenizer.eos_token_id
        elif kind == 'not_bool_special':
            record['special'] = 1
        elif kind == 'duplicate':
            records.append(dict(record))
        elif kind == 'not_dict':
            records.append('invalid')
        else:
            backend['added_tokens'] = {}

    patch_descriptor(monkeypatch, tokenizer, mutate)
    with pytest.raises(UnsupportedTokenizerError, match='backend added-token'):
        compile_tokenizer(tokenizer)


@pytest.mark.parametrize('vocab_type,prefix', [(xgr.VocabType.RAW, False),
                                              (xgr.VocabType.BYTE_LEVEL, False),
                                              (xgr.VocabType.BYTE_FALLBACK, True)])
def test_backend_raw_wrong_type_and_prefix_adjustment_rejected(monkeypatch, vocab_type, prefix):
    tokenizer = make_tokenizer()
    info = xgr.TokenizerInfo(list(tokenizer.get_vocab()), vocab_type=vocab_type,
                             stop_token_ids=[tokenizer.eos_token_id], add_prefix_space=prefix)
    monkeypatch.setattr(xgr.TokenizerInfo, 'from_huggingface', lambda *a, **kw: info)
    with pytest.raises(UnsupportedTokenizerError, match='vocabulary type mismatch or prefix adjustment'):
        compile_tokenizer(tokenizer)


def test_compile_does_not_rewrite_normalizer_pretokenizer_or_template():
    from tokenizers import normalizers
    tokenizer = make_tokenizer()
    tokenizer.backend_tokenizer.normalizer = normalizers.NFC()
    tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Split(' ', 'isolated')
    tokenizer.chat_template = '{{ messages }}'
    before = tokenizer.backend_tokenizer.to_str()
    compile_tokenizer(tokenizer)
    assert tokenizer.backend_tokenizer.to_str() == before
    assert tokenizer.chat_template == '{{ messages }}'


def test_native_fallback_flushes_invalid_bytes_at_ordinary_and_added_boundaries():
    tokenizer = make_tokenizer(added=[AddedToken('literal', normalized=False)])
    capability = compile_tokenizer(tokenizer)
    vocab = tokenizer.get_vocab()
    # Split bytes can assemble an emoji, but an ordinary/added token flushes
    # incomplete native ByteFallback sequences rather than making them valid.
    emoji = byte_ids(tokenizer, '😀')
    assert tokenizer.decode(emoji) == '😀'
    for boundary in [vocab['x▁y'], vocab['literal']]:
        ids = emoji[:2] + [boundary] + emoji[2:]
        assert '�' in tokenizer.decode(ids)
        with pytest.raises(UnicodeDecodeError):
            b''.join(capability.compiled_grammar.tokenizer_info.decoded_vocab[i]
                     for i in ids).decode('utf-8', errors='strict')


@pytest.mark.parametrize('serialized', ['{', '[]', '{"decoder":null,"decoder":null}'])
def test_malformed_backend_json_fails_explicitly(monkeypatch, serialized):
    tokenizer = make_tokenizer()
    monkeypatch.setattr(Tokenizer, 'to_str', lambda self, *a, **kw: serialized)
    with pytest.raises(UnsupportedTokenizerError, match='malformed tokenizer backend descriptor'):
        compile_tokenizer(tokenizer)
