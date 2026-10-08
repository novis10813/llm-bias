"""LC2: genuine donor-to-target replacement execution on CPU fakes.

Red phase: this file fails while
``llm_bias.core.inference.stance_localization_execution`` is absent (import
error). Green phase: three real greedy calls (donor capture, clean recipient,
patched recipient) with fresh trackers, actual prompt tensors differing per
route, donor-order gather repeats, target-label replacement editing only
mapped positions, preflight violations before any generation, same-token-text
drift, retained runtime failures without retry, an actual third-generation
flip, helper-exception cleanup, and frozen/defensive result validation for
both the plain-JSON and native Harmony routes.
"""
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import json

import pytest
import torch

from test_harmony_generation import contract_for, make_tokenizer, turn
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.inference.harmony_generation import (
    compile_harmony_decision_grammar, generate_harmony_structured,
)
from llm_bias.core.inference.structured_output import (
    StructuredGenerationPolicy, compile_decision_grammar, generate_structured,
)
from llm_bias.core.inference.stance_localization_execution import (
    PromptReplacementExecution, ReplacementMatchChecks, execute_prompt_replacement,
)
from llm_bias.core.prompt_input.decision_prompt import DecisionPrompt, DecisionSpan
from llm_bias.core.stance_localization_alignment import SpanAlignment, build_span_alignment

# ---------------------------------------------------------------------------
# CPU fake extending the NO1/Harmony fixtures: real torch.nn.Embedding-backed
# accessor, per-arm prompt recording, block-0 output observation, and a
# prompt-hidden sentinel that makes the third generation an actual flip.
# ---------------------------------------------------------------------------

DONOR_IDS = (51, 123, 124, 125, 52, 53)            # plain donor: entity (1,4)
TARGET_IDS = (61, 62, 63, 130, 131, 132, 133, 134, 64)  # plain target: entity (3,8)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = torch.nn.Identity()

    def forward(self, hidden_states):
        return self.post_attention_layernorm(hidden_states + 1) + 2


class Root(torch.nn.Module):
    def __init__(self, tokenizer, head, stops, target, *, flip_target=None,
                 sensitive=None, sentinel=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(vocab_size=head, eos_token_id=stops)
        self.generation_config = SimpleNamespace(eos_token_id=stops)
        self.head = head
        self.target = target
        self.flip_target = flip_target
        self.sensitive = sensitive
        self.sentinel = sentinel
        self.embeddings = torch.nn.Embedding(head, 4)
        self.calls = 0
        self.fail = None
        self.bypass = False
        self.prompts = []
        self.block0_outputs = []
        self.prompt_len = 0
        self._observed = None

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, **kwargs):
        hidden = input_ids.double().unsqueeze(-1).expand(-1, -1, 4)
        for index, block in enumerate(self.layers):
            hidden = block(hidden)
            if index == 0:
                self.block0_outputs.append(hidden)
        if self.sensitive is not None and hidden.shape[1] >= self.prompt_len:
            self._observed = float(hidden[0, self.sensitive, 0])
        return hidden

    def _flips(self):
        return (self.flip_target is not None and self.sentinel is not None
                and self._observed == self.sentinel)

    def generate(self, prompt, **kw):
        self.calls += 1
        self.prompt_len = prompt.shape[1]
        self.prompts.append(prompt[0].tolist())
        self._observed = None
        output = prompt.clone()
        if self.bypass:
            return torch.cat((output, torch.tensor([self.target])), dim=1)
        for step in range(kw['max_new_tokens']):
            values = output if not kw['use_cache'] or step == 0 else output[:, -1:]
            meta = {}
            if kw['use_cache']:
                offset = 0 if step == 0 else output.shape[1] - 1
                meta['cache_position'] = torch.arange(offset, offset + values.shape[1])
            self(values, **meta)
            if self.fail == (self.calls, step):
                raise RuntimeError('real root generation failure')
            current = self.flip_target if self._flips() else self.target
            scores = torch.full((1, self.head), -20.)
            desired = current[step] if step < len(current) else self.tokenizer.eos_token_id
            scores[0, desired] = 50.
            scores = kw['logits_processor'](output, scores)
            output = torch.cat((output, scores.argmax(-1).reshape(1, 1)), dim=1)
            if kw['stopping_criteria'](output, scores).all() or output[0, -1] in kw['eos_token_id']:
                break
        return output


def clean(root):
    for module in root.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


# ---------------------------------------------------------------------------
# DecisionPrompt record fixtures: real schema/template/wrapper binding, with
# verified span slices over the actual inference ID tuples.
# ---------------------------------------------------------------------------

WRAPPER_POLICY = {
    'use_chat_template': False, 'add_special_tokens': False,
    'system_message': None, 'enable_thinking': False,
    'chat_template_kwargs': {}, 'tokenizer_chat_template': None,
}


def _span(role, start, end, ids):
    token_ids = tuple(ids[start:end])
    return DecisionSpan(role, start, end, start, end, token_ids,
                        sha256_json(list(token_ids)),
                        sha256_json({'span': role, 'start': start, 'end': end}))


def make_prompt(token_ids, spans, schema_sha256, template_sha256, tag):
    token_ids = tuple(token_ids)
    records = {role: _span(role, start, end, token_ids) for role, (start, end) in spans.items()}
    return DecisionPrompt(
        raw_text=f'raw-{tag}', rendered_text=f'rendered-{tag}',
        inference_token_ids=token_ids,
        entity_span=records['entity'], evidence1_span=records['evidence1'],
        evidence2_span=records['evidence2'], instruction_span=records['instruction'],
        prompt_sha256=sha256_json({'fixture': 'lc2', 'tag': tag, 'ids': token_ids}),
        template_sha256=template_sha256, schema_sha256=schema_sha256,
        wrapper_policy=dict(WRAPPER_POLICY),
    )


PLAIN_DONOR_SPANS = {'entity': (1, 4), 'evidence1': (0, 1), 'evidence2': (4, 5),
                     'instruction': (5, 6)}
PLAIN_TARGET_SPANS = {'entity': (3, 8), 'evidence1': (0, 1), 'evidence2': (8, 9),
                      'instruction': (1, 2)}


@pytest.fixture(params=['plain', 'harmony'])
def setup(request):
    t = make_tokenizer()
    head = len(t) + 7
    template = sha256_json({'template': 'lc2'})
    if request.param == 'harmony':
        c = contract_for(t)
        cap = compile_harmony_decision_grammar(t, head, c)
        schema = cap.json_capability.schema_sha256
        suffix = tuple(c.prompt_suffix_ids)
        donor_p = make_prompt(suffix, {'entity': (0, len(suffix)), 'evidence1': (0, 1),
                                       'evidence2': (1, 2),
                                       'instruction': (len(suffix) - 2, len(suffix) - 1)},
                              schema, template, 'harmony-donor')
        target_ids = (t.pad_token_id, t.pad_token_id) + suffix
        target_p = make_prompt(target_ids, {'entity': (2, 12), 'evidence1': (0, 1),
                                            'evidence2': (1, 2), 'instruction': (0, 1)},
                               schema, template, 'harmony-target')
        alignment = build_span_alignment(donor_p, target_p, span='entity', policy='exact_tokens')
        target = turn(t, c, analysis=['analysis retained'])
        flip, sensitive, sentinel = None, None, None
        policy = StructuredGenerationPolicy(300, True, t.pad_token_id, 30.0, 'harmony_no_tools')
        stops = [c.final_stop_id]
    else:
        cap = compile_decision_grammar(t, head, [t.eos_token_id])
        schema = cap.schema_sha256
        donor_p = make_prompt(DONOR_IDS, PLAIN_DONOR_SPANS, schema, template, 'plain-donor')
        target_p = make_prompt(TARGET_IDS, PLAIN_TARGET_SPANS, schema, template, 'plain-target')
        alignment = build_span_alignment(donor_p, target_p, span='entity', policy='relative_rank')
        buy = tuple(t.encode('{"decision":"buy","reason":"evidence"}', add_special_tokens=False))
        sell = tuple(t.encode('{"decision":"sell","reason":"evidence"}', add_special_tokens=False))
        target, flip = buy + (t.eos_token_id,), sell + (t.eos_token_id,)
        # Donor row for target position 5 is donor position 2 (ID 124); final
        # hidden adds +6 per two blocks, so the sentinel detects the patched arm.
        sensitive, sentinel = 5, 124.0 + 6.0
        policy = StructuredGenerationPolicy(300, True, t.pad_token_id, 30.0, 'plain_json')
        stops = [t.eos_token_id]
    root = Root(t, head, stops, target, flip_target=flip, sensitive=sensitive, sentinel=sentinel)
    model = SimpleNamespace(hf_model=root, layers=root.layers)
    driver = (generate_harmony_structured if request.param == 'harmony'
              else generate_structured)
    donor_t = torch.tensor([donor_p.inference_token_ids])
    target_t = torch.tensor([target_p.inference_token_ids])
    expected_donor = driver(model, t, donor_t, cap, policy=policy)
    expected_target = driver(model, t, target_t, cap, policy=policy)
    assert expected_donor.failure_type is None and expected_target.failure_type is None
    root.calls = 0
    root.prompts.clear()
    root.block0_outputs.clear()
    return dict(route=request.param, model=model, tokenizer=t, capability=cap,
                donor_prompt=donor_p, target_prompt=target_p, alignment=alignment,
                expected_donor=expected_donor, expected_target=expected_target,
                policy=policy)


def run(setup, **overrides):
    model = overrides.pop('model', setup['model'])
    tokenizer = overrides.pop('tokenizer', setup['tokenizer'])
    donor_prompt = overrides.pop('donor_prompt', setup['donor_prompt'])
    target_prompt = overrides.pop('target_prompt', setup['target_prompt'])
    capability = overrides.pop('capability', setup['capability'])
    base = dict(policy=setup['policy'], layer=0, hook_site='pre',
                alignment=setup['alignment'], expected_donor=setup['expected_donor'],
                expected_target=setup['expected_target'])
    return execute_prompt_replacement(
        model, tokenizer, donor_prompt, target_prompt, capability, **(base | overrides))


def driver_name(setup):
    return ('generate_harmony_structured' if setup['route'] == 'harmony'
            else 'generate_structured')


def _failed(result, label):
    return replace(result, decision=None, reason=None, decision_complete=False,
                   schema_complete=False, reason_valid=False, failure_type='exception',
                   finish_reason='exception', error_message=f'{label} failure')


def _forged_alignment(setup, mutate):
    record = {key: (tuple(value) if key in ('donor_capture_positions', 'target_positions',
                                            'source_indices') else value)
              for key, value in setup['alignment'].to_dict().items()}
    mutate(record)
    payload = {key: value for key, value in record.items() if key != 'mapping_sha256'}
    return SpanAlignment(mapping_sha256=sha256_json(payload), **payload)


def _embedding_model(setup, getter):
    root = setup['model'].hf_model
    return SimpleNamespace(hf_model=SimpleNamespace(get_input_embeddings=getter),
                           layers=root.layers)


# ---------------------------------------------------------------------------
# Happy path: three real greedy calls, full continuation preservation, flip.
# ---------------------------------------------------------------------------

def test_executed_preserves_full_outputs(setup):
    result = run(setup)
    assert isinstance(result, PromptReplacementExecution)
    root = setup['model'].hf_model
    assert result.status == 'executed' and result.executed
    assert root.calls == 3
    exp_d, exp_t = setup['expected_donor'], setup['expected_target']
    assert result.donor.generated_token_ids == exp_d.generated_token_ids
    assert result.target_clean.generated_token_ids == exp_t.generated_token_ids
    assert result.diagnostics['layer'] == 0
    assert result.match_checks == ReplacementMatchChecks(True, True)
    assert not hasattr(result, 'source') and not hasattr(result, 'gathered')
    clean(root)


def test_actual_prompts_differ_and_call_order(setup):
    result = run(setup)
    root = setup['model'].hf_model
    donor_ids = setup['donor_prompt'].inference_token_ids
    target_ids = setup['target_prompt'].inference_token_ids
    assert donor_ids != target_ids
    assert root.prompts == [list(donor_ids), list(target_ids), list(target_ids)]
    assert result.status == 'executed'
    clean(root)


def test_third_generation_actual_flip(setup):
    if setup['route'] != 'plain':
        pytest.skip('flip fixture is plain route only')
    result = run(setup)
    assert result.status == 'executed'
    assert result.donor.decision == 'buy'
    assert result.target_clean.decision == 'buy'
    assert result.intervention.decision == 'sell'
    assert result.intervention.generated_token_ids == tuple(setup['model'].hf_model.flip_target)
    assert result.intervention.generated_token_ids != result.target_clean.generated_token_ids
    assert json.loads(result.intervention.json_payload)['decision'] == 'sell'
    # The clean recipient still matched its parent; the flip is not a mismatch.
    assert result.match_checks == ReplacementMatchChecks(True, True)
    clean(setup['model'].hf_model)


def test_harmony_preserves_full_ids(setup):
    if setup['route'] != 'harmony':
        pytest.skip('Harmony route only')
    result = run(setup)
    assert result.status == 'executed'
    exp_t = setup['expected_target']
    assert result.intervention.generated_token_ids == exp_t.generated_token_ids
    assert result.target_clean.generated_token_ids == exp_t.generated_token_ids
    assert 'analysis retained' in result.intervention.generated_text
    assert result.intervention.decision == 'buy'
    stops = [setup['capability'].contract.final_stop_id]
    assert result.intervention.generated_token_ids[-1] in stops
    # Donor rows are the identical shared suffix (a true self-replacement): the
    # hook must still select and record every mapped target position, while the
    # values themselves stay unchanged.
    assert result.diagnostics['changed_token_count'] == 0
    assert (result.diagnostics['selected_token_opportunities']
            == len(setup['alignment'].target_positions))
    clean(setup['model'].hf_model)


@pytest.mark.parametrize('site', ['pre', 'mid', 'post'])
@pytest.mark.parametrize('cache', [True, False])
def test_all_sites_and_cache_modes(setup, site, cache):
    policy = replace(setup['policy'], use_cache=cache)
    if policy.use_cache != setup['policy'].use_cache:
        # The expected outputs bind the exact policy; regenerate them for this
        # cache mode with genuine greedy calls.
        driver = (generate_harmony_structured if setup['route'] == 'harmony'
                  else generate_structured)
        model, t, cap = setup['model'], setup['tokenizer'], setup['capability']
        expected_donor = driver(model, t, torch.tensor([setup['donor_prompt'].inference_token_ids]),
                                cap, policy=policy)
        expected_target = driver(model, t, torch.tensor([setup['target_prompt'].inference_token_ids]),
                                 cap, policy=policy)
        assert expected_donor.failure_type is None and expected_target.failure_type is None
        model.hf_model.calls = 0
        model.hf_model.prompts.clear()
        model.hf_model.block0_outputs.clear()
        overrides = dict(expected_donor=expected_donor, expected_target=expected_target)
    else:
        overrides = {}
    result = run(setup, hook_site=site, policy=policy, **overrides)
    root = setup['model'].hf_model
    assert result.status == 'executed' and root.calls == 3
    diagnostics = result.diagnostics
    assert diagnostics['layer'] == 0 and diagnostics['hook_site'] == site
    assert diagnostics['operation'] == 'replacement' and diagnostics['scope'] == 'prompt_only'
    bound = len(setup['alignment'].target_positions) * (
        1 + len(result.intervention.generated_token_ids))
    assert 0 < diagnostics['selected_token_opportunities'] <= bound
    assert result.match_checks == ReplacementMatchChecks(True, True)
    clean(root)


# ---------------------------------------------------------------------------
# Capture, gather, and replacement mechanics.
# ---------------------------------------------------------------------------

def test_capture_once_at_declared_layer_site(setup, monkeypatch):
    import llm_bias.core.inference.stance_localization_execution as executor
    original = executor.capture_prompt_residual
    calls, holders = [], []

    @contextmanager
    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        with original(*args, **kwargs) as holder:
            holders.append(holder)
            yield holder

    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    result = run(setup)
    assert result.status == 'executed'
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is setup['model']
    assert kwargs['layer'] == 0 and kwargs['hook_site'] == 'pre'
    assert kwargs['prompt_positions'] == list(setup['alignment'].donor_capture_positions)
    tracker = kwargs['tracker']
    assert tracker.prompt_length == len(setup['donor_prompt'].inference_token_ids)
    holder = holders[0]
    # Exactly one donor prompt forward was captured; recipient forwards are inert.
    assert holder._observations == 1
    with pytest.raises(ValueError):
        holder.require_source()
    clean(setup['model'].hf_model)


def test_fresh_trackers_per_arm(setup, monkeypatch):
    import llm_bias.core.inference.stance_localization_execution as executor
    original = executor.GenerationPositionTracker
    created = []

    class Tracking(original):
        def __init__(self, prompt_length, use_cache):
            super().__init__(prompt_length, use_cache)
            created.append(self)

    monkeypatch.setattr(executor, 'GenerationPositionTracker', Tracking)
    result = run(setup)
    assert result.status == 'executed'
    donor_len = len(setup['donor_prompt'].inference_token_ids)
    target_len = len(setup['target_prompt'].inference_token_ids)
    assert [(t.prompt_length, t.use_cache) for t in created] == [
        (donor_len, True), (target_len, True), (target_len, True)]
    assert all(t._used and not t._tracking for t in created)
    clean(setup['model'].hf_model)


def test_source_released_on_all_paths(setup, monkeypatch):
    import llm_bias.core.inference.stance_localization_execution as executor
    original = executor.capture_prompt_residual
    holders = []

    @contextmanager
    def observe(*args, **kwargs):
        with original(*args, **kwargs) as holder:
            holders.append(holder)
            yield holder

    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    # Successful path first.
    assert run(setup).status == 'executed'
    holder = holders[-1]
    assert holder._source is None
    with pytest.raises(ValueError):
        holder.require_source()
    # Donor-failure path also releases the capture lifetime.
    holders.clear()
    root = setup['model'].hf_model
    root.calls = 0
    root.fail = (1, 1)
    result = run(setup)
    assert result.status == 'donor_failed'
    assert holders[-1]._source is None
    with pytest.raises(ValueError):
        holders[-1].require_source()
    clean(root)


def test_donor_order_gather_repeats_edit_only_mapped_positions(setup):
    if setup['route'] != 'plain':
        pytest.skip('gather-repeat fixture is plain route only')
    result = run(setup)
    assert result.status == 'executed'
    root = setup['model'].hf_model
    full = [h for h in root.block0_outputs if h.shape[1] == len(TARGET_IDS)]
    assert len(full) == 2
    rows = [[h[0, p, 0].item() for p in range(len(TARGET_IDS))] for h in full]
    clean_row, patched_row = rows
    assert clean_row == [i + 3 for i in TARGET_IDS]
    # Gathered donor rows repeat in donor order and land on target labels only.
    assert patched_row == [64, 65, 66, 126, 126, 127, 128, 128, 67]
    donor_rows = [[h[0, p, 0].item() for p in range(len(DONOR_IDS))]
                  for h in root.block0_outputs if h.shape[1] == len(DONOR_IDS)]
    assert donor_rows == [[i + 3 for i in DONOR_IDS]]
    assert result.diagnostics['changed_token_count'] == len(setup['alignment'].target_positions)
    clean(root)


# ---------------------------------------------------------------------------
# Preflight: every configuration violation raises before any generation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', [
    'layer_float', 'layer_too_big', 'layer_negative', 'site', 'route',
    'policy_channel', 'policy_max_tokens', 'policy_use_cache',
    'prompt_not_record', 'schema_hash', 'template_hash', 'wrapper',
    'alignment_forged_record', 'alignment_selector_claim', 'alignment_wrong_type',
    'parent_donor', 'parent_target', 'provenance_policy', 'provenance_bytes',
    'id_over_head', 'embedding_rows', 'embedding_not_callable', 'embedding_not_module',
    'embedding_not_tensor', 'embedding_1d', 'embedding_nonfinite', 'embedding_int',
    'embedding_zero_rows',
])
def test_preflight_rejects_before_generation(setup, case, monkeypatch):
    cap, policy = setup['capability'], setup['policy']
    donor_p, target_p = setup['donor_prompt'], setup['target_prompt']
    root = setup['model'].hf_model
    if case == 'layer_float':
        override = dict(layer=True)
    elif case == 'layer_too_big':
        override = dict(layer=2)
    elif case == 'layer_negative':
        override = dict(layer=-1)
    elif case == 'site':
        override = dict(hook_site='bad')
    elif case == 'route':
        other = 'harmony_no_tools' if setup['route'] == 'plain' else 'plain_json'
        override = dict(policy=replace(policy, channel_policy=other))
    elif case == 'policy_channel':
        override = dict(policy=replace(policy, channel_policy='unsupported'))
    elif case == 'policy_max_tokens':
        override = dict(policy=replace(policy, max_new_tokens=0))
    elif case == 'policy_use_cache':
        override = dict(policy=replace(policy, use_cache='yes'))
    elif case == 'prompt_not_record':
        override = dict(donor_prompt='not a record')
    elif case == 'schema_hash':
        override = dict(donor_prompt=replace(donor_p, schema_sha256='0' * 64))
    elif case == 'template_hash':
        override = dict(target_prompt=replace(target_p, template_sha256='f' * 64))
    elif case == 'wrapper':
        override = dict(target_prompt=replace(target_p, wrapper_policy={
            'use_chat_template': True, 'add_special_tokens': True, 'system_message': None,
            'enable_thinking': True, 'chat_template_kwargs': {},
            'tokenizer_chat_template': None}))
    elif case == 'alignment_forged_record':
        override = dict(alignment=_forged_alignment(
            setup, lambda record: record.update(
                target_positions=tuple(p + 1 for p in record['target_positions']))))
    elif case == 'alignment_selector_claim':
        override = dict(alignment=_forged_alignment(
            setup, lambda record: record.update(selector='last')))
    elif case == 'alignment_wrong_type':
        override = dict(alignment=SimpleNamespace(span='entity', policy='relative_rank',
                                                  selector='full'))
    elif case == 'parent_donor':
        override = dict(expected_donor=_failed(setup['expected_donor'], 'donor'))
    elif case == 'parent_target':
        override = dict(expected_target=_failed(setup['expected_target'], 'target'))
    elif case == 'provenance_policy':
        driver = (generate_harmony_structured if setup['route'] == 'harmony'
                  else generate_structured)
        prompt_t = (torch.tensor([target_p.inference_token_ids])
                    if setup['route'] == 'harmony'
                    else torch.tensor([donor_p.inference_token_ids]))
        other = driver(setup['model'], setup['tokenizer'], prompt_t, cap,
                       policy=replace(policy, max_new_tokens=250))
        root.calls = 0
        override = (dict(expected_target=other) if setup['route'] == 'harmony'
                    else dict(expected_donor=other))
    elif case == 'provenance_bytes':
        provenance = dict(setup['expected_donor'].provenance)
        provenance['schema_bytes_sha256'] = '0' * 64
        override = dict(expected_donor=replace(setup['expected_donor'],
                                               _provenance_bytes=canonical_json_bytes(provenance)))
    elif case == 'id_over_head':
        broken = list(DONOR_IDS)
        broken[5] = 9999
        override = dict(donor_prompt=make_prompt(broken, PLAIN_DONOR_SPANS,
                                                 cap.json_capability.schema_sha256
                                                 if setup['route'] == 'harmony'
                                                 else cap.schema_sha256,
                                                 donor_p.template_sha256, 'over-head'))
    elif case == 'embedding_rows':
        monkeypatch.setattr(root, 'embeddings', torch.nn.Embedding(50, 4))
        override = {}
    elif case == 'embedding_not_callable':
        override = dict(model=_embedding_model(setup, 5))
    elif case == 'embedding_not_module':
        override = dict(model=_embedding_model(setup, lambda: 3))
    elif case == 'embedding_not_tensor':
        override = dict(model=_embedding_model(
            setup, lambda: SimpleNamespace(weight=1)))
    elif case == 'embedding_1d':
        override = dict(model=_embedding_model(
            setup, lambda: SimpleNamespace(weight=torch.ones(4))))
    elif case == 'embedding_nonfinite':
        override = dict(model=_embedding_model(
            setup, lambda: SimpleNamespace(weight=torch.full((8, 4), float('nan')))))
    elif case == 'embedding_int':
        override = dict(model=_embedding_model(
            setup, lambda: SimpleNamespace(weight=torch.ones((8, 4), dtype=torch.int64))))
    elif case == 'embedding_zero_rows':
        override = dict(model=_embedding_model(
            setup, lambda: SimpleNamespace(weight=torch.ones((0, 4)))))
    else:
        raise AssertionError(case)
    with pytest.raises(ValueError):
        run(setup, **override)
    assert root.calls == 0
    clean(root)


@pytest.mark.parametrize('side', ['donor', 'target'])
def test_harmony_suffix_must_match(setup, side):
    if setup['route'] != 'harmony':
        pytest.skip('Harmony route only')
    c = setup['capability'].contract
    suffix = list(c.prompt_suffix_ids)
    suffix[-1] += 1
    prompt = make_prompt(suffix, {'entity': (0, len(suffix)), 'evidence1': (0, 1),
                                  'evidence2': (1, 2),
                                  'instruction': (len(suffix) - 2, len(suffix) - 1)},
                         setup['capability'].json_capability.schema_sha256,
                         setup['donor_prompt'].template_sha256, 'broken-suffix')
    override = dict(donor_prompt=prompt) if side == 'donor' else dict(target_prompt=prompt)
    if side == 'target':
        override['target_prompt'] = make_prompt(
            (setup['tokenizer'].pad_token_id, setup['tokenizer'].pad_token_id) + tuple(suffix),
            {'entity': (2, 12), 'evidence1': (0, 1), 'evidence2': (1, 2),
             'instruction': (0, 1)},
            setup['capability'].json_capability.schema_sha256,
            setup['donor_prompt'].template_sha256, 'broken-suffix-target')
    with pytest.raises(ValueError):
        run(setup, **override)
    assert setup['model'].hf_model.calls == 0
    clean(setup['model'].hf_model)


@pytest.mark.parametrize('route', ['plain', 'harmony'])
def test_unverified_factory_capability(setup, route):
    if setup['route'] != route:
        pytest.skip(f'{route} capability only')
    cap = setup['capability']
    forged = (replace(cap, grammar_sha256='0' * 64) if route == 'plain'
              else replace(cap, contract_sha256='0' * 64))
    with pytest.raises(ValueError):
        run(setup, capability=forged)
    assert setup['model'].hf_model.calls == 0
    clean(setup['model'].hf_model)


def test_invalid_parent_is_invocation_error(setup):
    # An invalid donor parent raises; it is never fabricated as a donor_failed
    # execution result.
    with pytest.raises(ValueError):
        run(setup, expected_donor=_failed(setup['expected_donor'], 'donor'))
    with pytest.raises(ValueError):
        run(setup, expected_target=_failed(setup['expected_target'], 'target'))
    assert setup['model'].hf_model.calls == 0
    clean(setup['model'].hf_model)


# ---------------------------------------------------------------------------
# Drift, failure, and source-availability outcomes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('field', ['generated_text', 'json_payload'])
def test_same_token_text_drift_is_donor_mismatch(setup, field):
    drifted = replace(setup['expected_donor'], **{field: 'drifted'})
    result = run(setup, expected_donor=drifted)
    root = setup['model'].hf_model
    assert result.status == 'donor_mismatch' and root.calls == 1
    assert result.donor.failure_type is None
    assert result.donor.generated_token_ids == setup['expected_donor'].generated_token_ids
    assert result.donor.generated_token_sha256 == setup['expected_donor'].generated_token_sha256
    assert getattr(result.donor, field) != getattr(drifted, field)
    assert result.target_clean is None and result.intervention is None
    assert result.diagnostics is None
    assert result.match_checks == ReplacementMatchChecks(False, None)
    clean(root)


@pytest.mark.parametrize('step', [0, 1])
def test_donor_failure_retained(setup, step):
    root = setup['model'].hf_model
    root.fail = (1, step)
    result = run(setup)
    assert result.status == 'donor_failed' and not result.executed
    assert root.calls == 1
    assert result.donor.failure_type == 'exception'
    assert result.donor.error_message == 'RuntimeError: real root generation failure'
    assert result.target_clean is None and result.intervention is None
    assert result.diagnostics is None
    assert result.match_checks == ReplacementMatchChecks(False, None)
    clean(root)


@pytest.mark.parametrize('step', [0, 1])
def test_target_failure_retained(setup, step):
    root = setup['model'].hf_model
    root.fail = (2, step)
    result = run(setup)
    assert result.status == 'target_failed' and root.calls == 2
    assert result.donor.failure_type is None
    assert result.target_clean.failure_type == 'exception'
    assert result.target_clean.error_message == 'RuntimeError: real root generation failure'
    assert result.intervention is None and result.diagnostics is None
    assert result.match_checks == ReplacementMatchChecks(True, False)
    clean(root)


def test_target_mismatch(setup):
    drifted = replace(setup['expected_target'], generated_text='drifted')
    result = run(setup, expected_target=drifted)
    root = setup['model'].hf_model
    assert result.status == 'target_mismatch' and root.calls == 2
    assert result.donor.failure_type is None
    assert result.target_clean.failure_type is None
    assert result.intervention is None and result.diagnostics is None
    assert result.match_checks == ReplacementMatchChecks(True, False)
    clean(root)


def test_source_unavailable(setup):
    root = setup['model'].hf_model
    root.bypass = True
    result = run(setup)
    assert result.status == 'source_unavailable' and root.calls == 1
    assert result.donor.failure_type is None
    assert result.target_clean is None and result.intervention is None
    assert result.match_checks == ReplacementMatchChecks(True, None)
    clean(root)


@pytest.mark.parametrize('step', [0, 1])
def test_intervention_failure_retained_without_retry(setup, step):
    root = setup['model'].hf_model
    root.fail = (3, step)
    result = run(setup)
    # Failure is a generation outcome: the arm stays executed and is not
    # resampled, and its actual failure record is preserved.
    assert result.status == 'executed' and root.calls == 3
    assert result.intervention.failure_type == 'exception'
    assert result.intervention.error_message == 'RuntimeError: real root generation failure'
    assert result.donor.failure_type is None
    assert result.target_clean.failure_type is None
    diagnostics = result.diagnostics
    assert diagnostics is not None and diagnostics['operation'] == 'replacement'
    bound = len(setup['alignment'].target_positions) * (
        1 + len(result.intervention.generated_token_ids))
    assert diagnostics['selected_token_opportunities'] <= bound
    assert result.match_checks == ReplacementMatchChecks(True, True)
    clean(root)


# ---------------------------------------------------------------------------
# Helper exceptions raise; cleanup still runs on every path.
# ---------------------------------------------------------------------------

def test_malformed_driver_result_raises(setup, monkeypatch):
    import llm_bias.core.inference.stance_localization_execution as executor
    name = driver_name(setup)
    original = getattr(executor, name)

    def malformed(*args, **kwargs):
        return replace(original(*args, **kwargs), generated_token_ids=[1])

    monkeypatch.setattr(executor, name, malformed)
    with pytest.raises(ValueError):
        run(setup)
    assert setup['model'].hf_model.calls == 1
    clean(setup['model'].hf_model)


@pytest.mark.parametrize('arm', [1, 2, 3])
def test_helper_exception_in_any_arm_cleans_up(setup, monkeypatch, arm):
    import llm_bias.core.inference.stance_localization_execution as executor
    name = driver_name(setup)
    original = getattr(executor, name)
    seen = {'calls': 0}

    def flaky(*args, **kwargs):
        seen['calls'] += 1
        if seen['calls'] == arm:
            raise RuntimeError(f'helper failure in arm {arm}')
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, name, flaky)
    with pytest.raises(RuntimeError, match='helper failure'):
        run(setup)
    # Arms before the failing one ran their single real greedy call.
    assert setup['model'].hf_model.calls == arm - 1
    clean(setup['model'].hf_model)


def test_capture_refusal_cleans_up(setup, monkeypatch):
    import llm_bias.core.inference.stance_localization_execution as executor

    @contextmanager
    def broken(*args, **kwargs):
        raise ValueError('capture refused')
        yield

    monkeypatch.setattr(executor, 'capture_prompt_residual', broken)
    with pytest.raises(ValueError, match='capture refused'):
        run(setup)
    assert setup['model'].hf_model.calls == 0
    clean(setup['model'].hf_model)


def test_intervention_refusal_cleans_up(setup, monkeypatch):
    import llm_bias.core.inference.stance_localization_execution as executor

    @contextmanager
    def broken(*args, **kwargs):
        raise RuntimeError('intervention refused')
        yield

    monkeypatch.setattr(executor, 'scoped_residual_intervention', broken)
    with pytest.raises(RuntimeError, match='intervention refused'):
        run(setup)
    assert setup['model'].hf_model.calls == 2
    clean(setup['model'].hf_model)


# ---------------------------------------------------------------------------
# Frozen/defensive result validation: status invariants and contradiction
# rejection.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('changes', [
    dict(status='done'),
    dict(status='donor_failed'),
    dict(status='donor_mismatch'),
    dict(status='source_unavailable'),
    dict(status='target_failed'),
    dict(status='target_mismatch'),
    dict(intervention=None),
    dict(target_clean=None),
    dict(_diagnostics_bytes=None),
    dict(expected_donor='not a result'),
    dict(match_checks=ReplacementMatchChecks(True, False)),
    dict(match_checks=ReplacementMatchChecks(True, None)),
    dict(match_checks=ReplacementMatchChecks(False, True)),
    dict(donor='not a result'),
    dict(expected_donor='not a result'),
    dict(match_checks=object()),
    dict(alignment=SimpleNamespace()),
])
def test_executed_rejects_contradictions(setup, changes):
    base = run(setup)
    with pytest.raises((ValueError, TypeError)):
        replace(base, **changes)
    clean(setup['model'].hf_model)


def test_executed_rejects_stored_output_drift(setup):
    base = run(setup)
    with pytest.raises(ValueError):
        replace(base, expected_donor=replace(setup['expected_donor'],
                                             generated_text='drifted'))
    with pytest.raises(ValueError):
        replace(base, donor=replace(base.donor, generated_text='drifted'))
    with pytest.raises(ValueError):
        replace(base, target_clean=replace(base.target_clean, json_payload='drifted'))
    clean(setup['model'].hf_model)


def test_executed_rejects_bad_diagnostics(setup):
    base = run(setup)
    diagnostics = base.diagnostics
    bound = len(setup['alignment'].target_positions) * (
        1 + len(base.intervention.generated_token_ids))
    for bad in [
        b'{}',
        b'{"layer":0,"layer":0}',
        b'{"value":NaN}',
        canonical_json_bytes(diagnostics | {'selected_token_opportunities': bound + 1}),
        canonical_json_bytes(diagnostics | {'operation': 'addition'}),
        canonical_json_bytes(diagnostics | {'scope': 'decode_only'}),
        canonical_json_bytes(diagnostics | {'layer': '0'}),
        canonical_json_bytes(diagnostics | {'finite_count': -1}),
        canonical_json_bytes(diagnostics | {'delta_l2_sum': 10 ** 400}),
    ]:
        with pytest.raises(ValueError):
            replace(base, _diagnostics_bytes=bad)
    clean(setup['model'].hf_model)


def test_failure_status_rejects_later_outputs(setup):
    executed = run(setup)
    root = setup['model'].hf_model
    root.calls = 0
    root.fail = (1, 1)
    donor_failed = run(setup)
    assert donor_failed.status == 'donor_failed'
    for changes in [
        dict(target_clean=executed.target_clean),
        dict(intervention=executed.intervention),
        dict(_diagnostics_bytes=executed._diagnostics_bytes),
        dict(status='source_unavailable'),
        dict(status='target_failed'),
        dict(match_checks=ReplacementMatchChecks(True, None)),
    ]:
        with pytest.raises((ValueError, TypeError)):
            replace(donor_failed, **changes)

    root.fail = (2, 0)
    root.calls = 0
    target_failed = run(setup)
    for changes in [
        dict(status='target_mismatch'),
        dict(intervention=executed.intervention),
        dict(_diagnostics_bytes=executed._diagnostics_bytes),
    ]:
        with pytest.raises((ValueError, TypeError)):
            replace(target_failed, **changes)
    root.fail = None
    root.calls = 0

    root_target_drift = replace(setup['expected_target'], generated_text='drifted')
    target_mismatch = run(setup, expected_target=root_target_drift)
    for changes in [
        dict(status='target_failed'),
        dict(intervention=executed.intervention),
    ]:
        with pytest.raises((ValueError, TypeError)):
            replace(target_mismatch, **changes)
    clean(root)


@pytest.mark.parametrize('donor,target', [
    (1, None), (0, None), (True, 1), (True, 'yes'), (False, 0),
])
def test_match_checks_require_genuine_bools(donor, target):
    with pytest.raises(ValueError):
        ReplacementMatchChecks(donor, target)


def test_match_checks_export_is_fresh():
    checks = ReplacementMatchChecks(False, True)
    first = checks.to_dict()
    assert first == {'donor': False, 'target': True}
    first['donor'] = True
    first['target'] = None
    assert checks.to_dict() == {'donor': False, 'target': True}
    assert ReplacementMatchChecks(True).target is None
    with pytest.raises(FrozenInstanceError):
        checks.donor = True


def test_frozen_result(setup):
    result = run(setup)
    with pytest.raises(FrozenInstanceError):
        result.status = 'donor_failed'
    with pytest.raises(FrozenInstanceError):
        result.donor = None
    with pytest.raises(FrozenInstanceError):
        result.match_checks.donor = False


def test_diagnostics_parsed_fresh(setup):
    result = run(setup)
    assert result._diagnostics_bytes is not None
    first = result.diagnostics
    first['layer'] = 99
    assert result.diagnostics is not first
    assert result.diagnostics['layer'] == 0


def test_to_dict_exact_export(setup):
    result = run(setup)
    export = result.to_dict()
    assert set(export) == {'status', 'executed', 'donor', 'expected_donor',
                           'target_clean', 'expected_target', 'intervention',
                           'match_checks', 'alignment', 'diagnostics'}
    assert '_diagnostics_bytes' not in export
    json.dumps(export, allow_nan=False)
    assert export['status'] == 'executed' and export['executed'] is True
    assert export['match_checks'] == {'donor': True, 'target': True}
    assert export['alignment']['target_positions'] == list(result.alignment.target_positions)
    assert export['diagnostics'] == result.diagnostics
    assert isinstance(export['donor']['generated_token_ids'], (list, tuple))
    assert 'provenance' in export['donor']
    # Defensive: exports are fresh copies, never the stored objects.
    export['alignment']['target_positions'].append(999)
    assert list(result.alignment.target_positions)[-1] != 999
    export['donor']['provenance']['schema_sha256'] = 'forged'
    assert result.donor.provenance['schema_sha256'] != 'forged'


def test_to_dict_failure_state_exports(setup):
    root = setup['model'].hf_model
    root.fail = (1, 1)
    result = run(setup)
    export = result.to_dict()
    assert export['status'] == 'donor_failed' and export['executed'] is False
    assert export['target_clean'] is None
    assert export['intervention'] is None
    assert export['diagnostics'] is None
    assert export['match_checks'] == {'donor': False, 'target': None}
    assert export['donor']['failure_type'] == 'exception'
    clean(root)
