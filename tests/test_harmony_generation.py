"""C2 CPU engineering contracts; synthetic native tokens, not GPT certification."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.inference.harmony_channels import HarmonyTokenContract, UnsupportedHarmonyChannel
from llm_bias.core.inference.harmony_generation import (
    compile_harmony_decision_grammar, generate_harmony_structured,
)
from llm_bias.core.inference.structured_output import (
    FiniteLegalTokenGuard, NoLegalTokenError, StructuredGenerationPolicy,
    UnsupportedTokenizerError, VerifiedModelBinding, compile_decision_grammar, generate_structured,
)


def make_tokenizer(*, fallback=False, bad_end=False):
    controls = ['<|start|>', '<|channel|>', '<|message|>', '<|end|>', '<|return|>', '<|tool|>']
    alphabet = ([f'<0x{i:02X}>' for i in range(256)] if fallback
                else sorted(pre_tokenizers.ByteLevel.alphabet()))
    vocab = {t: i for i, t in enumerate(alphabet + ['<unk>', '<pad>'] + controls)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>', byte_fallback=fallback))
    if fallback:
        backend.decoder = decoders.Sequence([decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse()])
    else:
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
    if bad_end:
        # A declared literal with bytes disagreed on by xgrammar's C-string path.
        controls[3] = '<|end|>\x00'
        backend.add_special_tokens([AddedToken(controls[3], special=True)])
    # Backend-only tool control proves specials aren't limited to HF named IDs.
    backend.add_special_tokens([AddedToken('<|tool|>', special=True)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>',
                                  eos_token='<|return|>', additional_special_tokens=controls[:-1])


def ids(tokenizer, text):
    if tokenizer.backend_tokenizer.decoder.__class__.__name__ == 'Sequence':
        return tuple(tokenizer.get_vocab()[f'<0x{i:02X}>'] for i in text.encode())
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def contract_for(t):
    v = t.get_vocab()
    restart = (v['<|start|>'],) + ids(t, 'assistant')
    return HarmonyTokenContract(restart,
        (v['<|channel|>'],) + ids(t, 'analysis') + (v['<|message|>'],),
        (v['<|channel|>'],) + ids(t, 'final') + (v['<|message|>'],),
        v['<|end|>'], restart, v['<|return|>'], (v['<|tool|>'],))


@pytest.fixture(scope='module', params=[False, True], ids=['bytelevel', 'bytefallback'])
def setup(request):
    t = make_tokenizer(fallback=request.param)
    c = contract_for(t)
    cap = compile_harmony_decision_grammar(t, len(t) + 7, c)
    return t, c, cap


def policy(t, **kw):
    return replace(StructuredGenerationPolicy(300, True, t.pad_token_id, 30.0, 'harmony_no_tools'), **kw)


def turn(t, c, *, analysis=None, reason='市場 café 😀 �', stop=True):
    body = ids(t, json.dumps({'decision': 'buy', 'reason': reason}, ensure_ascii=False, separators=(',', ':')))
    prefix = ()
    for text in analysis or []:
        prefix += c.initial_analysis_header_ids + ids(t, text) + (c.message_end_id,) + c.assistant_restart_ids
    return prefix + c.initial_final_header_ids + body + ((c.final_stop_id,) if stop else ())


class NativeLoop:
    def __init__(self, t, cap, target, *, bypass=False, fail_at=None, nonfinite_at=None):
        self.tokenizer = t
        self.config = SimpleNamespace(vocab_size=cap.json_capability.head_vocab_size,
                                      eos_token_id=[cap.contract.final_stop_id])
        self.generation_config = SimpleNamespace(eos_token_id=[cap.contract.final_stop_id])
        self.target, self.bypass, self.fail_at, self.nonfinite_at = target, bypass, fail_at, nonfinite_at
        self.calls = []
        self.head = cap.json_capability.head_vocab_size

    def generate(self, prompt, **kw):
        self.calls.append(kw)
        if self.bypass:
            return torch.cat((prompt, torch.tensor([self.target], dtype=torch.long)), dim=1)
        output = prompt.clone()
        for step in range(kw['max_new_tokens']):
            if step == self.fail_at:
                raise RuntimeError('root forward failure')
            desired = self.target[step] if step < len(self.target) else self.tokenizer.eos_token_id
            scores = torch.full((1, self.head), -20.0)
            scores[0, desired] = 51.0
            scores[0, self.head - 1] = 200.0
            # Before final the return must not steal generation from analysis.
            if desired != self.tokenizer.eos_token_id:
                scores[0, self.tokenizer.eos_token_id] = 100.0
            if step == self.nonfinite_at:
                scores.fill_(-float('inf'))
                scores[0, desired] = float('nan')
            scores = kw['logits_processor'](output, scores)
            assert scores[0, desired] == 51.0  # No native legal logit rewriting.
            output = torch.cat((output, scores.argmax(-1).reshape(1, 1)), dim=1)
            if kw['stopping_criteria'](output, scores).all():
                break
            if output[0, -1] in kw['eos_token_id']:
                break
        return output


def run(setup, target, *, controls=None, bypass=False, **kw):
    t, c, cap = setup
    fake = NativeLoop(t, cap, target, bypass=bypass, **kw)
    result = generate_harmony_structured(SimpleNamespace(hf_model=fake), t,
        torch.tensor([c.prompt_suffix_ids]), cap, policy=controls or policy(t))
    return result, fake


def test_native_turn_and_offsets(setup):
    t, c, cap = setup
    target = turn(t, c, analysis=['final {"decision":"sell","reason":"fake"}', '', 'more'])
    result, fake = run(setup, target)
    assert result.failure_type is None and result.finish_reason == 'eos' and result.decision == 'buy'
    assert result.generated_token_ids == target and result.generated_token_sha256 == sha256_json(target)
    assert result.generated_text == t.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert json.loads(result.json_payload)['reason'] == '市場 café 😀 �'
    assert len(fake.calls) == 1
    p = result.provenance
    assert p['channel_contract'] == json.loads(json.dumps(asdict(c)))
    assert p['channel_contract_sha256'] == sha256_json(asdict(c)) == cap.contract_sha256
    assert p['channel_policy_sha256'] == sha256_json({'policy': 'harmony_no_tools', 'contract_sha256': cap.contract_sha256})
    start, end = p['final_content_start'], p['final_content_end']
    assert target[start:end] == ids(t, result.json_payload)
    assert p['generated_token_count'] == len(target) and p['final_token_count'] == end - start
    offset = len(c.initial_analysis_header_ids)
    for text, span in zip(['final {"decision":"sell","reason":"fake"}', '', 'more'], p['analysis_segments']):
        assert span == [offset, offset + len(ids(t, text))]
        offset = span[1] + 1 + len(c.assistant_restart_ids) + len(c.initial_analysis_header_ids)
    p['channel_contract']['prompt_suffix_ids'].clear()
    assert result.provenance['channel_contract']['prompt_suffix_ids']
    canonical_json_bytes(result.to_dict())


@pytest.mark.parametrize('cache', [True, False])
def test_exact_total_budget_and_analysis_truncation(setup, cache):
    t, c, cap = setup
    full = turn(t, c, analysis=['long native thoughts'], stop=False)
    result, fake = run(setup, full, controls=policy(t, max_new_tokens=len(full), use_cache=cache))
    assert result.finish_reason == 'schema_complete' and result.failure_type is None
    assert result.provenance['final_content_end'] is None
    assert fake.calls[0]['use_cache'] is cache
    for n in [1, len(c.initial_analysis_header_ids), len(c.initial_analysis_header_ids) + 4, len(full) - 1]:
        result, _ = run(setup, full, controls=policy(t, max_new_tokens=n))
        assert result.failure_type == 'truncated' and result.finish_reason == 'token_budget'
        assert result.decision is result.reason is None
        if n <= len(c.initial_analysis_header_ids) + 4:
            assert result.json_payload == '' and result.provenance['analysis_segments'] == []


def test_fresh_processors_chunks_and_read_only_boundary(setup):
    t, c, cap = setup
    target = turn(t, c, analysis=['one', 'two'])
    for chunk in [1, 2, 7, len(target)]:
        p = cap.new_processor(prompt_length=len(c.prompt_suffix_ids), deadline=None)
        for end in range(chunk, len(target) + chunk, chunk):
            prefix = target[:end]
            tensor = torch.tensor([c.prompt_suffix_ids + prefix])
            p.observe(tensor)
            p.observe(tensor)
        assert p.generated_token_ids == target and p.boundary_tracker.is_terminated
        assert p.matcher.is_terminated()
        with pytest.raises(AttributeError):
            p.boundary_tracker = None
    assert cap.new_processor(prompt_length=1, deadline=None).matcher is not p.matcher


@pytest.mark.parametrize('location', ['header', 'analysis', 'final', 'after_return', 'restart'])
def test_unmasked_malformed_protocol_retains_raw_not_accepted(setup, location):
    t, c, cap = setup
    if location == 'header':
        target = (c.final_stop_id,)
    elif location == 'analysis':
        target = c.initial_analysis_header_ids + ids(t, 'hello') + (c.final_stop_id,)
    elif location == 'final':
        target = c.initial_final_header_ids + ids(t, '{') + (c.final_stop_id,)
    elif location == 'restart':
        target = c.initial_analysis_header_ids + (c.message_end_id, c.initial_final_header_ids[0])
    else:
        target = turn(t, c) + ids(t, 'tail')
    p = cap.new_processor(prompt_length=len(c.prompt_suffix_ids), deadline=None)
    with pytest.raises(UnsupportedHarmonyChannel):
        p.observe(torch.tensor([c.prompt_suffix_ids + target]))
    assert p.generated_token_ids == target
    assert len(p.boundary_tracker.observed_token_ids) < len(target)
    if location == 'final':
        assert not p.matcher.is_terminated() and not p.boundary_tracker.is_terminated
    result, _ = run(setup, target, bypass=True)
    assert result.generated_token_ids == target and result.failure_type == 'unsupported_channel'
    assert result.finish_reason == 'unsupported' and result.decision is None


def test_analysis_return_mask_and_original_logits(setup):
    t, c, cap = setup
    p = cap.new_processor(prompt_length=len(c.prompt_suffix_ids), deadline=None)
    prefix = torch.tensor([c.prompt_suffix_ids + c.initial_analysis_header_ids])
    raw = torch.arange(cap.json_capability.head_vocab_size, dtype=torch.float).reshape(1, -1)
    scores = p(prefix, raw.clone())
    blocked = (set(cap.json_capability.blocked_token_ids) - {c.message_end_id}) | {c.final_stop_id}
    assert all(torch.isneginf(scores[0, i]) for i in blocked)
    assert scores[0, c.message_end_id] == raw[0, c.message_end_id]
    allowed = set(range(raw.shape[1])) - blocked
    assert all(scores[0, i] == raw[0, i] for i in allowed)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('phase', ['header', 'analysis', 'final_complete'])
def test_no_synthetic_finite_logits(setup, value, phase):
    t, c, cap = setup
    prefix = () if phase == 'header' else (c.initial_analysis_header_ids if phase == 'analysis' else turn(t, c, stop=False))
    p = cap.new_processor(prompt_length=len(c.prompt_suffix_ids), deadline=None)
    tensor = torch.tensor([c.prompt_suffix_ids + prefix])
    scores = p(tensor, torch.full((1, cap.json_capability.head_vocab_size), value))
    with pytest.raises(NoLegalTokenError):
        FiniteLegalTokenGuard()(tensor, scores)


def test_noop_full_turn_changes_when_analysis_changes(setup):
    t, c, cap = setup
    a, _ = run(setup, turn(t, c, analysis=['one']))
    b, _ = run(setup, turn(t, c, analysis=['two']))
    assert a.json_payload == b.json_payload and a.decision == b.decision
    assert a.generated_token_ids != b.generated_token_ids
    assert a.generated_token_sha256 != b.generated_token_sha256


@pytest.mark.parametrize('forgery', ['copy', 'nested_copy', 'contract_copy', 'nested_mutation', 'contract_mutation'])
def test_factory_nested_identity_rejected_before_forward(setup, forgery):
    t, c, original = setup
    cap = compile_harmony_decision_grammar(t, original.json_capability.head_vocab_size, c)
    if forgery == 'copy':
        cap = replace(cap)
    elif forgery == 'nested_copy':
        object.__setattr__(cap, 'json_capability', replace(cap.json_capability))
    elif forgery == 'contract_copy':
        object.__setattr__(cap, 'contract', replace(c))
    elif forgery == 'nested_mutation':
        object.__setattr__(cap.json_capability, 'blocked_token_ids', ())
    else:
        # Keep the shared fixture contract intact even with hostile object mutation.
        cap = compile_harmony_decision_grammar(t, original.json_capability.head_vocab_size, replace(c))
        object.__setattr__(cap.contract, 'message_end_id', c.final_stop_id)
    fake = NativeLoop(t, original, turn(t, c))
    with pytest.raises(ValueError, match='identity|capability'):
        generate_harmony_structured(SimpleNamespace(hf_model=fake), t, torch.tensor([c.prompt_suffix_ids]),
                                    cap, policy=policy(t))
    with pytest.raises(ValueError, match='identity|capability'):
        cap.new_processor(prompt_length=1, deadline=None)
    assert fake.calls == []


@pytest.mark.parametrize('kind', ['huge', 'padded', 'unknown_vocab', 'ordinary_control', 'special_word', 'wrong_literal', 'unsafe_end'])
def test_compile_bounds_and_exact_binding(kind, monkeypatch):
    import llm_bias.core.inference.harmony_generation as module
    t = make_tokenizer(bad_end=kind == 'unsafe_end')
    c = contract_for(t)
    head = len(t) + 7
    if kind in ('huge', 'padded', 'unknown_vocab'):
        c = replace(c, message_end_id=10 ** 5000 if kind == 'huge' else head - 1)
        monkeypatch.setattr(module, 'sha256_json', lambda *a: pytest.fail('must check bounds before hashing'))
    elif kind == 'ordinary_control':
        c = replace(c, message_end_id=ids(t, 'z')[0])
    elif kind == 'special_word':
        c = replace(c, initial_analysis_header_ids=(c.initial_analysis_header_ids[0], t.pad_token_id,
                                                   c.initial_analysis_header_ids[-1]))
    elif kind == 'wrong_literal':
        c = replace(c, initial_analysis_header_ids=(c.initial_analysis_header_ids[0],) + ids(t, 'commentary')
                                                   + (c.initial_analysis_header_ids[-1],))
    else:
        c = replace(c, message_end_id=t.get_vocab()['<|end|>\x00'])
    with pytest.raises(ValueError):
        compile_harmony_decision_grammar(t, head, c)


@pytest.mark.parametrize('kind', ['policy', 'plain_routing', 'prompt_suffix', 'float_prompt', 'batch',
                                  'head', 'stops', 'tokenizer', 'missing_binding'])
def test_input_errors_before_generate(setup, kind):
    t, c, cap = setup
    fake = NativeLoop(t, cap, turn(t, c))
    wrapper = SimpleNamespace(hf_model=fake)
    prompt = torch.tensor([c.prompt_suffix_ids])
    controls = policy(t)
    use_tokenizer = t
    if kind == 'policy':
        controls = policy(t, channel_policy='plain_json')
    elif kind == 'prompt_suffix':
        prompt[0, -1] = t.pad_token_id
    elif kind == 'float_prompt':
        prompt = prompt.float()
    elif kind == 'batch':
        prompt = prompt.repeat(2, 1)
    elif kind == 'head':
        fake.config.vocab_size += 1
    elif kind == 'stops':
        fake.config.eos_token_id = [c.message_end_id]
    elif kind == 'tokenizer':
        use_tokenizer = make_tokenizer()
        use_tokenizer.add_tokens(['new token'])
    elif kind == 'missing_binding':
        del fake.tokenizer
    with pytest.raises(ValueError):
        if kind == 'plain_routing':
            generate_structured(wrapper, t, prompt, cap.json_capability, policy=controls)
        else:
            generate_harmony_structured(wrapper, use_tokenizer, prompt, cap, policy=controls)
    assert fake.calls == []


def test_verified_binding_missing_declaration_not_contradiction(setup):
    t, c, cap = setup
    fake = NativeLoop(t, cap, turn(t, c))
    del fake.tokenizer
    fake.bypass = True  # Returned-sequence path doesn't need a loop tokenizer.
    nested = cap.json_capability
    binding = VerifiedModelBinding(fake, nested.tokenizer_sha256, nested.head_vocab_size,
                                   nested.stop_token_ids, 'synthetic native loop inspection')
    result = generate_harmony_structured(SimpleNamespace(hf_model=fake), t, torch.tensor([c.prompt_suffix_ids]),
                                        cap, policy=policy(t), verified_binding=binding)
    assert result.decision == 'buy' and result.provenance['model_binding']['attested_missing'] == ['tokenizer']
    fake.config.eos_token_id = [c.message_end_id]
    with pytest.raises(ValueError, match='stop'):
        generate_harmony_structured(SimpleNamespace(hf_model=fake), t, torch.tensor([c.prompt_suffix_ids]),
                                    cap, policy=policy(t), verified_binding=binding)
    assert len(fake.calls) == 1


@pytest.mark.parametrize('prompt_length,deadline', [(True, None), (0, None), (1, True), (1, float('nan')),
                                                  (1, float('inf')), (1, 'tomorrow')])
def test_processor_argument_validation(setup, prompt_length, deadline):
    with pytest.raises(ValueError):
        setup[2].new_processor(prompt_length=prompt_length, deadline=deadline)


@pytest.mark.parametrize('kind', ['rewritten', 'dtype', 'batch', 'out_of_head', 'padded', 'logits_head'])
def test_processor_observation_validation(setup, kind):
    t, c, cap = setup
    p = cap.new_processor(prompt_length=len(c.prompt_suffix_ids), deadline=None)
    prefix = c.prompt_suffix_ids + c.initial_analysis_header_ids
    tensor = torch.tensor([prefix])
    p.observe(tensor)
    if kind == 'rewritten':
        tensor[0, -1] = t.pad_token_id
    elif kind == 'dtype':
        tensor = tensor.float()
    elif kind == 'batch':
        tensor = tensor.repeat(2, 1)
    elif kind in ('out_of_head', 'padded'):
        bad = cap.json_capability.head_vocab_size - (kind == 'padded')
        tensor = torch.cat((tensor, torch.tensor([[bad]])), dim=1)
    with pytest.raises(ValueError):
        if kind == 'logits_head':
            p(tensor, torch.zeros(1, cap.json_capability.head_vocab_size + 1))
        else:
            p.observe(tensor)
    if kind in ('out_of_head', 'padded'):
        assert p.generated_token_ids[-1] == bad
        assert p.boundary_tracker.observed_token_ids == c.initial_analysis_header_ids


@pytest.mark.parametrize('kind', ['over_budget', 'prefix', 'early_header', 'early_final', 'float_return', 'complete_no_return'])
def test_custom_generate_return_validation(setup, kind):
    t, c, cap = setup
    target = turn(t, c, stop=False)
    fake = NativeLoop(t, cap, target)
    prompt = torch.tensor([c.prompt_suffix_ids])
    controls = policy(t)
    if kind == 'over_budget':
        controls = policy(t, max_new_tokens=len(target) - 1)
    elif kind == 'early_header':
        target = target[:1]
    elif kind == 'early_final':
        target = target[:-1]
    def custom(ids, **kw):
        output = torch.cat((ids, torch.tensor([target])), dim=1)
        if kind == 'prefix':
            output[0, 0] = t.pad_token_id
        elif kind == 'float_return':
            output = output.float()
        return output
    fake.generate = custom
    result = generate_harmony_structured(SimpleNamespace(hf_model=fake), t, prompt, cap, policy=controls)
    if kind == 'complete_no_return':
        assert result.failure_type is None and result.finish_reason == 'schema_complete'
    else:
        assert result.failure_type == result.finish_reason == 'exception' and result.decision is None


@pytest.mark.parametrize('root', ['exception', 'no_legal_token', 'timeout', 'unsupported_channel'])
@pytest.mark.parametrize('secondary', ['raises', 'mismatch', 'surrogate'])
def test_root_failure_survives_decode(setup, monkeypatch, root, secondary):
    import llm_bias.core.inference.harmony_generation as module
    t, c, cap = setup
    # Freeze factory identity before patching the runtime decoder.
    target = turn(t, c, analysis=['native thoughts'])
    fake = NativeLoop(t, cap, target, fail_at=3 if root == 'exception' else None,
                      nonfinite_at=3 if root == 'no_legal_token' else None)
    if root == 'unsupported_channel':
        fake.bypass = True
        fake.target = c.initial_final_header_ids + (c.final_stop_id,)
    if root == 'timeout':
        clock = iter([0.0, 0.0, 0.0, 0.0, 2.0])
        monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock, 2.0))
    def bad_decode(*a, **kw):
        if secondary == 'raises':
            raise RuntimeError('secondary decode failure')
        return '\ud800' if secondary == 'surrogate' else 'not exact bytes'
    monkeypatch.setattr(t, 'decode', bad_decode)
    result = generate_harmony_structured(SimpleNamespace(hf_model=fake), t, torch.tensor([c.prompt_suffix_ids]),
                                        cap, policy=policy(t, timeout_seconds=1.0))
    assert result.failure_type == root and result.decode_error and result.error_message
    assert 'secondary' not in result.error_message
    assert result.decision is result.reason is None
    canonical_json_bytes(result.to_dict())


def test_success_decode_mismatch_is_existing_unsupported_tokenizer(setup, monkeypatch):
    t, c, cap = setup
    target = turn(t, c)
    monkeypatch.setattr(t, 'decode', lambda *a, **kw: 'mismatch')
    result, _ = run(setup, target, bypass=True)
    assert result.failure_type == 'unsupported_tokenizer' and result.finish_reason == 'unsupported'
    assert result.decode_error and result.decision is None


@pytest.mark.parametrize('where', ['analysis', 'final'])
def test_whole_turn_strict_utf8_including_analysis(setup, where):
    t, c, cap = setup
    vocab = cap.json_capability.compiled_grammar.tokenizer_info.decoded_vocab
    bad = next(i for i, b in enumerate(vocab) if b == b'\xff')
    if where == 'analysis':
        target = c.initial_analysis_header_ids + (bad, c.message_end_id) + c.assistant_restart_ids + turn(t, c)
    else:
        target = c.initial_final_header_ids + ids(t, '{"decision":"buy","reason":"') + (bad,) + ids(t, '"}') + (c.final_stop_id,)
    result, _ = run(setup, target, bypass=True)
    assert result.generated_token_ids == target and result.decode_error
    assert result.failure_type is not None and result.decision is result.reason is None


def test_provenance_bound_before_execution(setup):
    t, c, cap = setup
    fake = NativeLoop(t, cap, turn(t, c))
    original = fake.generate
    def mutate(prompt, **kw):
        result = original(prompt, **kw)
        kw['eos_token_id'].clear()
        return result
    fake.generate = mutate
    result = generate_harmony_structured(SimpleNamespace(hf_model=fake), t, torch.tensor([c.prompt_suffix_ids]),
                                        cap, policy=policy(t))
    assert result.failure_type is None
    assert result.provenance['hf_controls']['eos_token_id'] == [c.final_stop_id]


@pytest.mark.parametrize('cache', [True, False])
@pytest.mark.parametrize('completion', ['return', 'budget'])
def test_actual_tiny_hf_cpu_generate(setup, cache, completion):
    from transformers import GPT2Config, GPT2LMHeadModel
    t, c, cap = setup
    target = turn(t, c, analysis=['final fake JSON {"decision":"sell"}', ''], stop=completion == 'return')
    hf = GPT2LMHeadModel(GPT2Config(
        vocab_size=cap.json_capability.head_vocab_size, n_positions=512, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=t.pad_token_id, pad_token_id=t.pad_token_id, eos_token_id=c.final_stop_id,
    )).eval()
    # Hostile checkpoint defaults must not inject headers, return or an answer.
    hf.generation_config.forced_bos_token_id = c.final_stop_id
    hf.generation_config.forced_eos_token_id = c.message_end_id
    hf.generation_config.suppress_tokens = [target[0]]
    forwards = 0
    def prefer(_module, _args, output):
        nonlocal forwards
        output.logits.fill_(-20.0)
        output.logits[:, -1, target[forwards]] = 51.0
        output.logits[:, -1, cap.json_capability.head_vocab_size - 1] = 100.0
        if target[forwards] != c.final_stop_id:
            output.logits[:, -1, c.final_stop_id] = 80.0
        forwards += 1
    handle = hf.register_forward_hook(prefer)
    wrapper = SimpleNamespace(hf_model=hf, tokenizer=t)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = generate_harmony_structured(wrapper, t, torch.tensor([c.prompt_suffix_ids]), cap,
            policy=policy(t, use_cache=cache, max_new_tokens=len(target)))
    finally:
        torch.set_num_threads(old)
        handle.remove()
    assert result.failure_type is None and result.decision == 'buy'
    assert result.generated_token_ids == target and forwards == len(target)
    assert result.finish_reason == ('eos' if completion == 'return' else 'schema_complete')
    assert len(result.provenance['analysis_segments']) == 2
    assert not hf._forward_hooks


@pytest.mark.parametrize('phase', ['header', 'analysis', 'restart', 'final'])
def test_backend_only_tool_controls_cannot_bypass_masks_or_returns(setup, phase):
    t, c, cap = setup
    tool = c.forbidden_control_ids[0]
    assert tool not in t.all_special_ids and tool in cap.json_capability.special_token_ids
    prefix = {'header': (), 'analysis': c.initial_analysis_header_ids,
              'restart': c.initial_analysis_header_ids + (c.message_end_id,),
              'final': c.initial_final_header_ids}[phase]
    p = cap.new_processor(prompt_length=len(c.prompt_suffix_ids), deadline=None)
    tensor = torch.tensor([c.prompt_suffix_ids + prefix])
    scores = p(tensor, torch.zeros(1, cap.json_capability.head_vocab_size))
    assert torch.isneginf(scores[0, tool])
    result, _ = run(setup, prefix + (tool,), bypass=True)
    assert result.failure_type == 'unsupported_channel' and result.decision is None


@pytest.mark.parametrize('phase', ['analysis', 'final'])
def test_mismatched_ordinary_token_remains_blocked(phase):
    t = make_tokenizer(fallback=True)
    # Native lower-case hex spelling disagrees with installed backend bytes.
    t.add_tokens(['<0x4a>'])
    c = contract_for(t)
    cap = compile_harmony_decision_grammar(t, len(t) + 7, c)
    bad = t.get_vocab()['<0x4a>']
    assert bad in cap.json_capability.blocked_token_ids
    prefix = c.initial_analysis_header_ids if phase == 'analysis' else c.initial_final_header_ids
    p = cap.new_processor(prompt_length=len(c.prompt_suffix_ids), deadline=None)
    tensor = torch.tensor([c.prompt_suffix_ids + prefix])
    assert torch.isneginf(p(tensor, torch.zeros(1, cap.json_capability.head_vocab_size))[0, bad])
    with pytest.raises(ValueError, match='mismatched'):
        p.observe(torch.tensor([c.prompt_suffix_ids + prefix + (bad,)]))
    assert p.generated_token_ids[-1] == bad and p.boundary_tracker.observed_token_ids == prefix


@pytest.mark.parametrize('reason,failure', [(' \n\t', 'invalid_reason'), ('\\ud800', 'invalid_reason')])
def test_final_validation_never_promotes_invalid_reason(setup, reason, failure):
    t, c, cap = setup
    # Supply the escape literally for the surrogate case, not a Python surrogate.
    if reason == '\\ud800':
        target = c.initial_final_header_ids + ids(t, '{"decision":"buy","reason":"\\ud800"}') + (c.final_stop_id,)
    else:
        target = turn(t, c, reason=reason)
    result, _ = run(setup, target)
    assert result.failure_type == failure and result.decision is result.reason is None
    assert result.schema_complete and result.decision_complete and not result.reason_valid


def test_partial_final_diagnostics_do_not_replace_root_failure(setup):
    t, c, cap = setup
    target = turn(t, c, analysis=['thinking'])
    # Failure after a complete final object but before its native return.
    result, _ = run(setup, target, fail_at=len(target) - 1)
    assert result.failure_type == result.finish_reason == 'exception'
    assert result.schema_complete and result.decision_complete and result.reason_valid
    assert result.decision is result.reason is None
    assert result.json_payload.endswith('}') and result.provenance['final_content_end'] is None


def test_complete_schema_cannot_synthesize_nonfinite_return(setup):
    t, c, cap = setup
    target = turn(t, c)
    result, _ = run(setup, target, nonfinite_at=len(target) - 1)
    assert result.failure_type == result.finish_reason == 'no_legal_token'
    assert result.schema_complete and result.decision is None
    assert result.generated_token_ids == target[:-1]
    assert result.provenance['final_content_end'] is None


def test_c2_preserves_all_nine_existing_failure_types():
    from typing import get_args
    from llm_bias.core.inference.structured_output import FailureType
    assert set(get_args(FailureType)) == {
        'truncated', 'timeout', 'exception', 'no_legal_token', 'invalid_json',
        'invalid_schema', 'invalid_reason', 'unsupported_channel', 'unsupported_tokenizer',
    }
