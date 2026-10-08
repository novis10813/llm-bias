"""Grouped execution uses actual CPU greedy drivers and transient torch hooks."""
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from test_stance_localization_execution import setup, clean, run, driver_name, _forged_alignment
from llm_bias.core.inference.stance_localization_grouped import (
    ReplacementCell, execute_grouped_prompt_replacement,
)
from llm_bias.core.stance_localization_alignment import build_span_alignment


def panel(s):
    alignments = [s['alignment']]
    for span in ('evidence1', 'instruction'):
        alignments.append(build_span_alignment(s['donor_prompt'], s['target_prompt'],
                                              span=span, policy='relative_rank'))
    alignments.append(build_span_alignment(s['donor_prompt'], s['target_prompt'],
                                          span='entity', policy='relative_rank',
                                          selector='last'))
    return tuple(ReplacementCell(layer, site, a)
                 for layer, site in ((0, 'pre'), (0, 'mid'), (1, 'post'))
                 for a in alignments)


def grouped(s, cells=None, **overrides):
    args = {k: s[k] for k in ('policy', 'expected_donor', 'expected_target')}
    args.update(overrides)
    return execute_grouped_prompt_replacement(
        s['model'], s['tokenizer'], s['donor_prompt'], s['target_prompt'],
        s['capability'], cells=panel(s) if cells is None else cells, **args)


def reset(s):
    root = s['model'].hf_model
    root.calls = 0
    root.prompts.clear()
    root.block0_outputs.clear()


def without_elapsed(value):
    if isinstance(value, dict):
        return {k: without_elapsed(v) for k, v in value.items() if k != 'elapsed_seconds'}
    if isinstance(value, (tuple, list)):
        return [without_elapsed(v) for v in value]
    return value


@pytest.mark.parametrize('cache', [True, False])
def test_actual_full_lc2_equivalence_and_amortization(setup, cache):
    import llm_bias.core.inference.stance_localization_grouped as executor
    if not cache:
        policy = replace(setup['policy'], use_cache=False)
        driver = getattr(executor, driver_name(setup))
        for name, prompt in (('expected_donor', 'donor_prompt'),
                             ('expected_target', 'target_prompt')):
            setup[name] = driver(setup['model'], setup['tokenizer'],
                                 torch.tensor([setup[prompt].inference_token_ids]),
                                 setup['capability'], policy=policy)
        setup['policy'] = policy
    cells = panel(setup)
    reset(setup)
    expected = [run(setup, layer=c.layer, hook_site=c.hook_site, alignment=c.alignment)
                for c in cells]
    assert setup['model'].hf_model.calls == 3 * len(cells)
    reset(setup)
    actual = grouped(setup, cells)
    assert actual.status == 'executed'
    assert actual.cells == cells
    assert setup['model'].hf_model.calls == 2 + len(cells)
    assert [without_elapsed(r.to_dict()) for r in actual.executions] == [
        without_elapsed(r.to_dict()) for r in expected]
    assert all(r.donor is actual.donor and r.target_clean is actual.target_clean
               for r in actual.executions)
    clean(setup['model'].hf_model)


def test_union_capture_inert_lifetime_and_fresh_trackers(setup, monkeypatch):
    import llm_bias.core.inference.stance_localization_grouped as executor
    capture = executor.capture_prompt_residual
    tracking = executor.GenerationPositionTracker
    holders, trackers, captures = [], [], []

    class Tracker(tracking):
        def __init__(self, *args):
            super().__init__(*args)
            trackers.append(self)

    @contextmanager
    def observe(*args, **kwargs):
        with capture(*args, **kwargs) as holder:
            holders.append(holder)
            captures.append(kwargs)
            yield holder

    monkeypatch.setattr(executor, 'GenerationPositionTracker', Tracker)
    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    cells = panel(setup)
    result = grouped(setup, cells)
    assert result.status == 'executed'
    assert len(holders) == 3
    for holder, kwargs in zip(holders, captures):
        union = sorted({p for c in cells
                        if (c.layer, c.hook_site) == (holder.layer, holder.hook_site)
                        for p in c.alignment.donor_capture_positions})
        assert kwargs['prompt_positions'] == union
        assert holder._observations == 1
        assert holder._source is None
        with pytest.raises(ValueError):
            holder.require_source()
    assert len(trackers) == 2 + len(cells)
    assert all(t._used and not t._tracking for t in trackers)
    clean(setup['model'].hf_model)


@pytest.mark.parametrize('case,status,calls', [
    ('donor_failure', 'donor_failed', 1), ('target_failure', 'target_failed', 2),
    ('donor_drift', 'donor_mismatch', 1), ('target_drift', 'target_mismatch', 2),
    ('bypass', 'source_unavailable', 1),
])
def test_abort_without_fake_cell_rows(setup, case, status, calls):
    root = setup['model'].hf_model
    overrides = {}
    if case.endswith('failure'):
        root.fail = (1 if case.startswith('donor') else 2, 1)
    elif case.endswith('drift'):
        key = 'expected_donor' if case.startswith('donor') else 'expected_target'
        overrides[key] = replace(setup[key], generated_text='drifted')
    else:
        root.bypass = True
    result = grouped(setup, **overrides)
    assert result.status == status
    assert result.executions == ()
    assert root.calls == calls
    clean(root)


def test_failed_intervention_continues(setup):
    root = setup['model'].hf_model
    root.fail = (3, 1)
    result = grouped(setup)
    assert result.status == 'executed'
    assert result.executions[0].intervention.failure_type == 'exception'
    assert all(r.intervention.failure_type is None for r in result.executions[1:])
    assert root.calls == 2 + len(result.cells)
    clean(root)


@pytest.mark.parametrize('arm', [1, 2, 3, 5])
def test_caller_error_releases_captures(setup, monkeypatch, arm):
    import llm_bias.core.inference.stance_localization_grouped as executor
    capture = executor.capture_prompt_residual
    driver = getattr(executor, driver_name(setup))
    holders = []
    count = 0

    @contextmanager
    def observe(*args, **kwargs):
        with capture(*args, **kwargs) as holder:
            holders.append(holder)
            yield holder

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == arm:
            raise RuntimeError('caller kernel error')
        return driver(*args, **kwargs)

    monkeypatch.setattr(executor, 'capture_prompt_residual', observe)
    monkeypatch.setattr(executor, driver_name(setup), fail)
    with pytest.raises(RuntimeError, match='caller kernel error'):
        grouped(setup)
    assert holders and all(h._source is None for h in holders)
    clean(setup['model'].hf_model)


@pytest.mark.parametrize('case', ['duplicate', 'mutable', 'empty', 'layer', 'alignment',
                                 'policy', 'parent', 'capability'])
def test_preflight_before_forward(setup, case):
    cells = panel(setup)
    overrides = {}
    if case == 'duplicate':
        cells += (cells[0],)
    elif case == 'mutable':
        cells = list(cells)
    elif case == 'empty':
        cells = ()
    elif case == 'layer':
        cells += (ReplacementCell(999, 'post', setup['alignment']),)
    elif case == 'alignment':
        cells += (ReplacementCell(0, 'post', _forged_alignment(setup, lambda record: record.update(selector='last'))),)
    elif case == 'policy':
        overrides['policy'] = replace(setup['policy'], max_new_tokens=250)
    elif case == 'parent':
        overrides['expected_target'] = 'bad'
    else:
        setup['capability'] = object()
    with pytest.raises((ValueError, TypeError)):
        grouped(setup, cells, **overrides)
    assert setup['model'].hf_model.calls == 0
    clean(setup['model'].hf_model)


def test_real_embedding_gradients_and_captured_replacement(setup, monkeypatch):
    """Exercise actual differentiable hidden values, not hand-written row formulas."""
    if setup['route'] != 'plain':
        pytest.skip('gradient sentinel fixture is plain only')
    root = setup['model'].hf_model
    with torch.no_grad():
        root.embeddings.weight.copy_(torch.arange(root.head).unsqueeze(1).expand(-1, 4))

    def forward(input_ids, **kwargs):
        hidden = root.embeddings(input_ids)
        for block in root.layers:
            hidden = block(hidden)
        if hidden.shape[1] >= root.prompt_len:
            root._observed = float(hidden[0, root.sensitive, 0].detach())
        return hidden

    monkeypatch.setattr(root, 'forward', forward)
    ids = torch.tensor([setup['donor_prompt'].inference_token_ids])
    root(ids).sum().backward()
    assert root.embeddings.weight.grad[ids[0]].abs().sum() > 0
    cells = panel(setup)
    expected = [run(setup, layer=c.layer, hook_site=c.hook_site, alignment=c.alignment)
                for c in cells]
    reset(setup)
    result = grouped(setup, cells)
    assert [without_elapsed(r.to_dict()) for r in result.executions] == [
        without_elapsed(r.to_dict()) for r in expected]
    assert result.executions[0].intervention.decision == 'sell'
    clean(root)
