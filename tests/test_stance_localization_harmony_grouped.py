"""Actual CPU Harmony driver, factory grammar, transient capture and patch hooks."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_stance_localization_execution import setup as lc2_setup, clean, run
from test_stance_localization_grouped import panel, reset, without_elapsed
from llm_bias.core.artifact_paths import canonical_json_bytes
from llm_bias.core.inference import stance_localization_harmony_grouped as executor
from llm_bias.core.inference.stance_localization_execution import _execution


@pytest.fixture
def setup():
    return lc2_setup.__wrapped__(SimpleNamespace(param='harmony'))


def grouped(s, cells=None, **overrides):
    kwargs = dict(donor_policy=s['policy'], target_policy=s['policy'],
                  expected_donor=s['expected_donor'], expected_target=s['expected_target'])
    kwargs.update(overrides)
    return executor.execute_harmony_grouped_prompt_replacement(
        s['model'], s['tokenizer'], s['donor_prompt'], s['target_prompt'],
        s['capability'], cells=panel(s) if cells is None else cells, **kwargs)


def policies(s, donor_budget, target_budget, donor_cache=True, target_cache=True):
    donor = replace(s['policy'], max_new_tokens=donor_budget, use_cache=donor_cache,
                    timeout_seconds=15.0)
    target = replace(s['policy'], max_new_tokens=target_budget, use_cache=target_cache,
                     timeout_seconds=40.0)
    parents = {}
    for label, policy in (('donor', donor), ('target', target)):
        parents['expected_' + label] = executor.generate_harmony_structured(
            s['model'], s['tokenizer'],
            torch.tensor([s[label + '_prompt'].inference_token_ids]), s['capability'],
            policy=policy)
        assert parents['expected_' + label].failure_type is None
    reset(s)
    return dict(donor_policy=donor, target_policy=target, **parents)


def three_arm(s, cell, config):
    """Independent single-cell oracle using each arm's native driver policy."""
    model, tokenizer, cap = s['model'], s['tokenizer'], s['capability']
    donor_ids = torch.tensor([s['donor_prompt'].inference_token_ids])
    target_ids = torch.tensor([s['target_prompt'].inference_token_ids])
    dp, tp = config['donor_policy'], config['target_policy']
    a = cell.alignment
    tracker = executor.GenerationPositionTracker(donor_ids.shape[1], dp.use_cache)
    with tracker.track(model), executor.capture_prompt_residual(
        model, tracker=tracker, layer=cell.layer, hook_site=cell.hook_site,
        prompt_positions=list(a.donor_capture_positions),
    ) as holder:
        donor = executor.generate_harmony_structured(model, tokenizer, donor_ids, cap, policy=dp)
        source = holder.require_source()[:, list(a.source_indices), :]
    with executor.GenerationPositionTracker(target_ids.shape[1], tp.use_cache).track(model):
        target = executor.generate_harmony_structured(model, tokenizer, target_ids, cap, policy=tp)
    tracker = executor.GenerationPositionTracker(target_ids.shape[1], tp.use_cache)
    with tracker.track(model), executor.scoped_residual_intervention(
        model, tracker=tracker, layer=cell.layer, hook_site=cell.hook_site,
        scope='prompt_only', operation='replacement', source=source,
        source_positions=list(a.target_positions), prompt_positions=list(a.target_positions), dose=1,
    ) as diagnostics:
        patched = executor.generate_harmony_structured(model, tokenizer, target_ids, cap, policy=tp)
    return _execution('executed', donor, target, patched, config['expected_donor'],
                      config['expected_target'], a,
                      diagnostics=canonical_json_bytes(diagnostics.to_dict()))


@pytest.mark.parametrize('donor_budget,target_budget', [(1024, 4096), (4096, 1024)])
@pytest.mark.parametrize('donor_cache,target_cache', [(True, False), (False, True)])
def test_mixed_policies_full_three_arm_equivalence(setup, donor_budget, target_budget,
                                                  donor_cache, target_cache, monkeypatch):
    config = policies(setup, donor_budget, target_budget, donor_cache, target_cache)
    cells = panel(setup)
    expected = [three_arm(setup, c, config) for c in cells]
    root = setup['model'].hf_model
    assert root.calls == 3 * len(cells)
    reset(setup)
    trackers, observed = [], []
    tracker_class = executor.GenerationPositionTracker
    driver = executor.generate_harmony_structured

    class Tracker(tracker_class):
        def __init__(self, *args):
            super().__init__(*args)
            trackers.append(self)

    def observe(*args, **kwargs):
        observed.append(kwargs['policy'])
        return driver(*args, **kwargs)

    monkeypatch.setattr(executor, 'GenerationPositionTracker', Tracker)
    monkeypatch.setattr(executor, 'generate_harmony_structured', observe)
    actual = grouped(setup, cells, **config)
    assert actual.status == 'executed'
    assert root.calls == 2 + len(cells)
    assert observed == [config['donor_policy']] + [config['target_policy']] * (1 + len(cells))
    assert [t.use_cache for t in trackers] == [donor_cache] + [target_cache] * (1 + len(cells))
    assert all(t._used and not t._tracking for t in trackers)
    assert [without_elapsed(r.to_dict()) for r in actual.executions] == [
        without_elapsed(r.to_dict()) for r in expected]
    for row in actual.executions:
        assert row.donor is actual.donor and row.target_clean is actual.target_clean
        for arm in (row.donor, row.target_clean, row.intervention):
            assert 'analysis retained' in arm.generated_text
            assert '<|channel|>final<|message|>' in arm.generated_text
            assert arm.generated_token_ids[-1] == setup['capability'].contract.final_stop_id
            assert arm.json_payload == config['expected_target'].json_payload
    clean(root)


def test_equal_policy_existing_lc2_equivalence(setup):
    cells = panel(setup)
    expected = [run(setup, layer=c.layer, hook_site=c.hook_site, alignment=c.alignment)
                for c in cells]
    reset(setup)
    actual = grouped(setup, cells)
    assert [without_elapsed(r.to_dict()) for r in actual.executions] == [
        without_elapsed(r.to_dict()) for r in expected]
    clean(setup['model'].hf_model)


@pytest.mark.parametrize('case,status,calls', [
    ('donor_failed', 'donor_failed', 1), ('target_failed', 'target_failed', 2),
    ('donor_text', 'donor_mismatch', 1), ('donor_tokens', 'donor_mismatch', 1),
    ('target_text', 'target_mismatch', 2), ('target_tokens', 'target_mismatch', 2),
    ('capture', 'source_unavailable', 1),
])
def test_parent_abort_cleanup(setup, case, status, calls, monkeypatch):
    config = policies(setup, 1024, 4096)
    root = setup['model'].hf_model
    if case.endswith('failed'):
        root.fail = (1 if case.startswith('donor') else 2, 1)
    elif case == 'capture':
        root.bypass = True
    else:
        label, field = case.split('_')
        parent = config['expected_' + label]
        changes = {'generated_text': 'changed analysis, same decision'}
        if field == 'tokens':
            from llm_bias.core.artifact_paths import sha256_json
            tokens = parent.generated_token_ids + (setup['tokenizer'].pad_token_id,)
            changes = dict(generated_token_ids=tokens, generated_token_sha256=sha256_json(tokens))
        config['expected_' + label] = replace(parent, **changes)
    holders = []
    capture = executor.capture_prompt_residual

    @contextmanager
    def observe(*args, **kwargs):
        with capture(*args, **kwargs) as holder:
            holders.append(holder)
            yield holder

    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    actual = grouped(setup, **config)
    assert actual.status == status and actual.executions == ()
    assert root.calls == calls
    assert all(h._source is None for h in holders)
    clean(root)


def test_failed_intervention_is_genuine_cell_and_continues(setup):
    config = policies(setup, 4096, 1024)
    root = setup['model'].hf_model
    root.fail = (3, 1)
    actual = grouped(setup, **config)
    assert actual.status == 'executed'
    first = actual.executions[0]
    assert first.intervention.failure_type == 'exception'
    assert first.intervention.error_message == 'RuntimeError: real root generation failure'
    assert first.intervention.generated_token_ids
    assert first.diagnostics is not None
    assert all(r.intervention.failure_type is None for r in actual.executions[1:])
    assert root.calls == 2 + len(actual.cells)
    clean(root)


@pytest.mark.parametrize('case', ['donor_policy', 'target_policy', 'timeout', 'cache',
                                  'channel', 'capability', 'contract', 'schema', 'stops',
                                  'tokenizer', 'head', 'suffix', 'duplicate'])
def test_binding_before_forward(setup, case):
    config = policies(setup, 1024, 4096)
    cells = panel(setup)
    if case in ('donor_policy', 'target_policy'):
        config[case] = replace(config[case], max_new_tokens=2048)
    elif case in ('timeout', 'cache', 'channel'):
        changes = {'timeout_seconds': 41} if case == 'timeout' else (
            {'use_cache': False} if case == 'cache' else {'channel_policy': 'plain_json'})
        config['target_policy'] = replace(config['target_policy'], **changes)
    elif case == 'capability':
        setup['capability'] = replace(setup['capability'])
    elif case in ('contract', 'schema', 'stops'):
        parent = config['expected_target']
        provenance = parent.provenance
        key = {'contract': 'channel_contract_sha256', 'schema': 'schema_bytes_sha256',
               'stops': 'stop_token_ids'}[case]
        provenance[key] = [] if case == 'stops' else '0' * 64
        config['expected_target'] = replace(parent, _provenance_bytes=canonical_json_bytes(provenance))
    elif case == 'tokenizer':
        from test_harmony_generation import make_tokenizer
        setup['tokenizer'] = make_tokenizer()
        setup['tokenizer'].add_tokens(['foreign'])
    elif case == 'head':
        setup['model'].hf_model.config.vocab_size += 1
    elif case == 'suffix':
        # A different prompt still has valid spans, but lacks the native suffix.
        p = setup['target_prompt']
        setup['target_prompt'] = replace(p, inference_token_ids=p.inference_token_ids + (0,))
    else:
        cells += (cells[0],)
    with pytest.raises(ValueError):
        grouped(setup, cells, **config)
    assert setup['model'].hf_model.calls == 0
    clean(setup['model'].hf_model)


@pytest.mark.parametrize('arm', [1, 2, 3, 5])
def test_helper_exception_cleans_every_lifetime(setup, monkeypatch, arm):
    capture, driver = executor.capture_prompt_residual, executor.generate_harmony_structured
    holders = []
    calls = 0

    @contextmanager
    def observe(*args, **kwargs):
        with capture(*args, **kwargs) as holder:
            holders.append(holder)
            yield holder

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == arm:
            raise RuntimeError('helper error')
        return driver(*args, **kwargs)

    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    monkeypatch.setattr(executor, 'generate_harmony_structured', fail)
    with pytest.raises(RuntimeError, match='helper error'):
        grouped(setup)
    assert holders and all(h._source is None for h in holders)
    clean(setup['model'].hf_model)


def test_union_capture_once_and_released(setup, monkeypatch):
    capture = executor.capture_prompt_residual
    holders = []
    cells = panel(setup)

    @contextmanager
    def observe(*args, **kwargs):
        with capture(*args, **kwargs) as holder:
            holders.append(holder)
            yield holder

    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    assert grouped(setup, cells).status == 'executed'
    assert len(holders) == 3
    for holder in holders:
        assert holder.positions == tuple(sorted({p for c in cells
            if (c.layer, c.hook_site) == (holder.layer, holder.hook_site)
            for p in c.alignment.donor_capture_positions}))
        assert holder._observations == 1 and holder._source is None
        with pytest.raises(ValueError):
            holder.require_source()
    clean(setup['model'].hf_model)


def test_actual_harmony_patch_changes_generated_decision(setup):
    from test_harmony_generation import turn
    from llm_bias.core.inference.stance_localization_grouped import ReplacementCell
    from llm_bias.core.stance_localization_alignment import build_span_alignment
    root = setup['model'].hf_model
    c, t = setup['capability'].contract, setup['tokenizer']
    root.sensitive = 0
    root.sentinel = float(setup['donor_prompt'].inference_token_ids[0] + 6)
    buy = t.encode('{"decision":"buy"', add_special_tokens=False)
    sell = t.encode('{"decision":"sell"', add_special_tokens=False)
    target = list(turn(t, c, analysis=['analysis retained']))
    start = next(i for i in range(len(target)) if target[i:i + len(buy)] == buy)
    root.flip_target = tuple(target[:start] + sell + target[start + len(buy):])
    config = policies(setup, 1024, 4096)
    alignment = build_span_alignment(setup['donor_prompt'], setup['target_prompt'],
                                     span='evidence1', policy='relative_rank')
    actual = grouped(setup, (ReplacementCell(0, 'pre', alignment),), **config)
    row = actual.executions[0]
    assert row.target_clean.decision == 'buy' and row.intervention.decision == 'sell'
    assert row.intervention.generated_token_ids == root.flip_target
    assert 'analysis retained' in row.intervention.generated_text
    assert row.diagnostics['changed_token_count'] == 1
    clean(root)
