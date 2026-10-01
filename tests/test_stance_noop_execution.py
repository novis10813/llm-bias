"""Real CPU root forwards and native grammar-driven autoregressive decoding."""
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest
import torch
from test_harmony_generation import make_tokenizer, contract_for, turn
from llm_bias.core.inference.harmony_generation import compile_harmony_decision_grammar
from llm_bias.core.inference.structured_output import compile_decision_grammar, StructuredGenerationPolicy
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop, PromptNoOpExecution


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = torch.nn.Identity()

    def forward(self, hidden_states):
        return self.post_attention_layernorm(hidden_states + 1) + 2


class Root(torch.nn.Module):
    def __init__(self, tokenizer, head, stops, target):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(vocab_size=head, eos_token_id=stops)
        self.generation_config = SimpleNamespace(eos_token_id=stops)
        self.head, self.target = head, target
        self.calls = 0
        self.fail = None
        self.bypass = False
        self.bad_positions = False
        self.mutate = None
        self.legacy = False

    def forward(self, input_ids, **kwargs):
        hidden = input_ids.double().unsqueeze(-1).expand(-1, -1, 4)
        for block in self.layers:
            hidden = block(hidden)
        return hidden

    def generate(self, prompt, **kw):
        self.calls += 1
        output = prompt.clone()
        if self.bypass:
            return torch.cat((output, torch.tensor([self.target])), dim=1)
        for step in range(kw['max_new_tokens']):
            values = output if not kw['use_cache'] or step == 0 else output[:, -1:]
            meta = {}
            if kw['use_cache']:
                offset = 0 if step == 0 else output.shape[1] - 1
                if self.legacy:
                    kv = torch.zeros(1, 1, offset, 1)
                    meta['past_key_values'] = ((kv, kv),) if offset else None
                else:
                    meta['cache_position'] = torch.arange(offset, offset + values.shape[1])
            if self.bad_positions:
                meta['cache_position'] = torch.zeros(values.shape[1])
            self(values, **meta)
            if self.mutate:
                self.mutate()
            if self.fail == (self.calls, step):
                raise RuntimeError('real root generation failure')
            scores = torch.full((1, self.head), -20.)
            desired = self.target[step] if step < len(self.target) else self.tokenizer.eos_token_id
            scores[0, desired] = 50.
            scores = kw['logits_processor'](output, scores)
            output = torch.cat((output, scores.argmax(-1).reshape(1, 1)), dim=1)
            if kw['stopping_criteria'](output, scores).all() or output[0, -1] in kw['eos_token_id']:
                break
        return output


@pytest.fixture(params=['plain', 'harmony'])
def setup(request):
    t = make_tokenizer()
    if request.param == 'harmony':
        contract = contract_for(t)
        cap = compile_harmony_decision_grammar(t, len(t) + 7, contract)
        target = turn(t, contract, analysis=['analysis retained'])
        prompt = torch.tensor([contract.prompt_suffix_ids])
        head, stops = cap.json_capability.head_vocab_size, [contract.final_stop_id]
        channel = 'harmony_no_tools'
    else:
        cap = compile_decision_grammar(t, len(t) + 7, [t.eos_token_id])
        target = tuple(t.encode('{"decision":"buy","reason":"evidence"}', add_special_tokens=False)) + (t.eos_token_id,)
        prompt = torch.tensor([[t.pad_token_id] * 3])
        head, stops, channel = cap.head_vocab_size, [t.eos_token_id], 'plain_json'
    root = Root(t, head, stops, target)
    model = SimpleNamespace(hf_model=root, layers=root.layers)
    policy = StructuredGenerationPolicy(300, True, t.pad_token_id, 30., channel)
    return model, t, prompt, cap, dict(policy=policy, layer=0, hook_site='pre', zero_vector=torch.ones(4), config_hash='a' * 64)


def run(setup, **overrides):
    model, t, prompt, cap, kwargs = setup
    return execute_prompt_noop(model, t, prompt, cap, **(kwargs | overrides))


def clean(root):
    for module in root.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


@pytest.mark.parametrize('site', ['pre', 'mid', 'post'])
@pytest.mark.parametrize('cache', [True, False])
@pytest.mark.parametrize('positions', [None, [2, 0]])
def test_four_real_generations(setup, site, cache, positions):
    result = run(setup, hook_site=site, prompt_positions=positions, policy=replace(setup[-1]['policy'], use_cache=cache))
    root = setup[0].hf_model
    assert root.calls == 4 and result.completed and result.passed
    for arm in (result.baseline, result.repeat, result.zero, result.self_replacement):
        assert arm.generated_token_ids == root.target
        assert arm.generated_text == result.baseline.generated_text
    for d in (result.zero_diagnostics, result.self_diagnostics):
        assert d['selected_token_opportunities'] > 0 and d['changed_token_count'] == 0
        assert d['delta_l2_sum'] == 0
    d = result.zero_diagnostics
    d['layer'] = 100
    assert result.zero_diagnostics['layer'] == 0
    if setup[-1]['policy'].channel_policy == 'harmony_no_tools':
        assert 'analysis retained' in result.baseline.generated_text
    assert not hasattr(result, 'source') and not hasattr(result, 'vector')
    clean(root)


@pytest.mark.parametrize('arm', [1, 2, 3, 4])
@pytest.mark.parametrize('step', [0, 1])
def test_runtime_failure(setup, arm, step):
    root = setup[0].hf_model
    root.fail = (arm, step)
    result = run(setup)
    assert not result.passed
    if arm == 1:
        assert result.halt_reason == 'baseline_failed' and root.calls == 1
        assert result.repeat is result.zero is result.self_replacement is result.gate is None
        assert result.zero_diagnostics is result.self_diagnostics is None
    else:
        assert result.completed and root.calls == 4
        failed = (result.repeat, result.zero, result.self_replacement)[arm - 2]
        assert failed.failure_type == 'exception' and failed.finish_reason == 'exception'
        assert failed.error_message == 'RuntimeError: real root generation failure'
    clean(root)


def test_no_capture(setup):
    root = setup[0].hf_model
    root.bypass = True
    result = run(setup)
    assert result.halt_reason == 'source_unavailable' and root.calls == 1
    clean(root)


@pytest.mark.parametrize('vector', [torch.zeros(4), torch.ones(2, 2), torch.tensor([float('nan')]), torch.ones(4, dtype=torch.long), torch.ones(5)])
def test_invalid_vector(setup, vector):
    with pytest.raises(ValueError):
        run(setup, zero_vector=vector)
    assert setup[0].hf_model.calls == (1 if vector.shape == (5,) else 0)
    clean(setup[0].hf_model)


@pytest.mark.parametrize('kw', [dict(layer=True), dict(hook_site='bad'), dict(prompt_positions=[]), dict(prompt_positions=[0, 0]), dict(config_hash='A' * 64)])
def test_invalid_config(setup, kw):
    with pytest.raises(ValueError):
        run(setup, **kw)
    assert setup[0].hf_model.calls == 0
    clean(setup[0].hf_model)


def test_constructor_negative(setup):
    r = run(setup)
    for kw in [dict(repeat=None), dict(halt_reason='baseline_failed'), dict(baseline=None), dict(_zero_diagnostics=b'{}'), dict(_self_diagnostics=None)]:
        with pytest.raises((ValueError, TypeError)):
            replace(r, **kw)
    d = r.zero_diagnostics
    for raw in [json.dumps(d).encode(), b'{"layer":0,"layer":0}', b'{"value":NaN}']:
        with pytest.raises(ValueError):
            replace(r, _zero_diagnostics=raw)
    from llm_bias.core.stance_baseline_adapter import baseline_gate_input
    from llm_bias.core.stance_gates import evaluate_noop_gate
    failed = replace(r.repeat, decision=None, reason=None, decision_complete=False,
                     schema_complete=False, reason_valid=False, failure_type='exception',
                     finish_reason='exception', error_message='synthetic failure')
    # A separately valid gate cannot be substituted merely because passed agrees.
    unrelated = evaluate_noop_gate(*(baseline_gate_input(x) for x in
        (r.baseline, failed, r.zero, r.self_replacement)), config_hash=r.gate.config_hash)
    with pytest.raises(ValueError):
        replace(r, gate=unrelated)
    for malformed in [replace(r.baseline, generated_token_ids=[1]),
                      replace(r.baseline, _provenance_bytes=b'{"bad":NaN}')]:
        with pytest.raises((ValueError, TypeError)):
            replace(r, baseline=malformed)
    for name, bad in [('layer', True), ('finite_count', -1), ('scope', 'decode_only'),
                      ('delta_l2_sum', float('inf')), ('hook_site', 'bad')]:
        from llm_bias.core.artifact_paths import canonical_json_bytes
        changed = r.zero_diagnostics | {name: bad}
        with pytest.raises(ValueError):
            replace(r, _zero_diagnostics=canonical_json_bytes(changed))


@pytest.mark.parametrize('field', ['delta_l2_sum', 'delta_l2_max',
                                  'relative_delta_l2_sum', 'relative_delta_l2_max'])
@pytest.mark.parametrize('storage', ['_zero_diagnostics', '_self_diagnostics'])
def test_oversized_diagnostic_reduction_is_validation_error(setup, field, storage):
    from llm_bias.core.artifact_paths import canonical_json_bytes
    result = run(setup)
    diagnostics = result.zero_diagnostics if storage == '_zero_diagnostics' else result.self_diagnostics
    with pytest.raises(ValueError):
        replace(result, **{storage: canonical_json_bytes(diagnostics | {field: 10**400})})


def test_snapshot_and_legacy_cache(setup):
    root = setup[0].hf_model
    root.legacy = True
    vector = setup[-1]['zero_vector']
    root.mutate = lambda: vector.zero_()
    assert run(setup).passed
    assert root.calls == 4
    clean(root)


@pytest.mark.parametrize('registration', ['register_forward_hook', 'register_forward_pre_hook'])
def test_registration_failure(setup, monkeypatch, registration):
    root = setup[0].hf_model
    def fail(*args, **kwargs):
        raise RuntimeError('registration failed')
    monkeypatch.setattr(root.layers[0], registration, fail)
    site = 'post' if registration == 'register_forward_hook' else 'pre'
    with pytest.raises(RuntimeError, match='registration failed'):
        run(setup, hook_site=site)
    assert root.calls == 0
    clean(root)


def test_capture_release_on_exception(setup, monkeypatch):
    import llm_bias.core.inference.stance_noop_execution as executor
    from contextlib import contextmanager
    original = executor.capture_prompt_residual
    holders = []
    @contextmanager
    def observe(*args, **kwargs):
        with original(*args, **kwargs) as holder:
            holders.append(holder)
            yield holder
    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    with pytest.raises(ValueError):
        run(setup, zero_vector=torch.ones(5))
    assert holders[0]._source is None
    with pytest.raises(ValueError):
        holders[0].require_source()
    clean(setup[0].hf_model)


def test_malformed_driver_result_raises(setup, monkeypatch):
    import llm_bias.core.inference.stance_noop_execution as executor
    name = 'generate_structured' if setup[-1]['policy'].channel_policy == 'plain_json' else 'generate_harmony_structured'
    original = getattr(executor, name)
    def malformed(*args, **kwargs):
        return replace(original(*args, **kwargs), generated_token_ids=[1])
    monkeypatch.setattr(executor, name, malformed)
    with pytest.raises(ValueError):
        run(setup)
    assert setup[0].hf_model.calls == 1
    clean(setup[0].hf_model)


def test_tracking_failure_and_external_hook(setup):
    root = setup[0].hf_model
    handle = root.layers[1].register_forward_hook(lambda *args: None)
    root.bad_positions = True
    result = run(setup)
    assert result.halt_reason == 'baseline_failed' and root.calls == 1
    assert list(root.layers[1]._forward_hooks) == [handle.id]
    handle.remove()
    clean(root)
