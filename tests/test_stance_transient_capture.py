from contextlib import ExitStack

import pytest
import torch

from llm_bias.core.inference.stance_interventions import (
    GenerationPositionTracker, scoped_residual_intervention,
)
from llm_bias.core.inference.stance_transient_capture import capture_prompt_residual


class Block(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.post_attention_layernorm = torch.nn.Identity()
        self.kind = kind
        self.output = None

    def forward(self, hidden_states):
        value = self.post_attention_layernorm(hidden_states + 1) + 2
        self.output = value if self.kind == 'tensor' else (value, 'extra')
        if self.kind == 'list':
            self.output = list(self.output)
        return self.output


class Root(torch.nn.Module):
    def __init__(self, kind='tensor'):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(kind), Block(kind)])
        self.mode = ''
        self.check_identity = True

    def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        hidden = inputs_embeds
        for index, block in enumerate(self.layers):
            if index == 0 and self.mode == 'skip':
                continue
            output = block(hidden)
            if self.check_identity:
                assert output is block.output
            hidden = output if torch.is_tensor(output) else output[0]
            if index == 0 and self.mode == 'duplicate':
                block(hidden)
        if self.mode == 'fail':
            raise RuntimeError('late failure')
        return None if self.mode == 'none' else hidden


def clean(root):
    for module in root.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def x(length=3):
    return torch.arange(length * 2, dtype=torch.float64).reshape(1, length, 2).requires_grad_()


@pytest.mark.parametrize('site', ['pre', 'mid', 'post'])
@pytest.mark.parametrize('kind', ['tensor', 'tuple', 'list'])
@pytest.mark.parametrize('positions', [None, [2, 0]])
def test_capture_and_fresh_replay(site, kind, positions):
    root = Root(kind)
    tracker = GenerationPositionTracker(3, True)
    values = x()
    with ExitStack() as tracking, ExitStack() as lifetime:
        tracking.enter_context(tracker.track(root))
        holder = lifetime.enter_context(capture_prompt_residual(
            root, tracker=tracker, layer=0, hook_site=site, prompt_positions=positions))
        with pytest.raises(ValueError):
            holder.require_source()
        baseline = root(inputs_embeds=values)
        source = holder.require_source()
        order = [0, 1, 2] if positions is None else positions
        offset = {'pre': 0, 'mid': 1, 'post': 3}[site]
        assert torch.equal(source, (values + offset)[:, order])
        assert source.dtype == values.dtype and source.device == values.device
        assert not source.requires_grad and source.grad_fn is None
        assert source.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
        assert 'tensor(' not in repr(holder)
        assert holder.ready and holder.positions == tuple(order)
        assert holder.layer == 0 and holder.hook_site == site and holder.prompt_length == 3
        with pytest.raises(AttributeError):
            holder.positions = ()
        tracking.close()
        root.check_identity = False
        replay_tracker = GenerationPositionTracker(3, True)
        with replay_tracker.track(root), scoped_residual_intervention(
            root, tracker=replay_tracker, layer=0, hook_site=site,
            scope='prompt_only', operation='replacement', source=source,
            source_positions=holder.positions, prompt_positions=holder.positions,
        ) as diagnostics:
            assert torch.equal(root(inputs_embeds=values), baseline)
            # Invalid original metadata must be irrelevant once its tracker exits.
            root(inputs_embeds=x(1), cache_position=torch.tensor([3]))
        assert diagnostics.changed_token_count == 0
        assert holder.require_source() is source
    assert not holder.ready
    with pytest.raises(ValueError):
        holder.require_source()
    clean(root)


@pytest.mark.parametrize('site', ['pre', 'mid', 'post'])
@pytest.mark.parametrize('cached', [True, False])
def test_first_source_authority(site, cached):
    root = Root()
    tracker = GenerationPositionTracker(3, cached)
    with tracker.track(root), capture_prompt_residual(root, tracker=tracker, layer=0, hook_site=site) as holder:
        root(inputs_embeds=x())
        source = holder.require_source()
        saved = source.clone()
        root(inputs_embeds=x(1) + 100, cache_position=torch.tensor([3])) if cached else root(inputs_embeds=x(4) + 100)
        assert holder.require_source() is source
        assert torch.equal(source, saved)
    clean(root)


@pytest.mark.parametrize('site', ['pre', 'mid', 'post'])
@pytest.mark.parametrize('mode', ['skip', 'duplicate', 'fail', 'none'])
def test_failed_forward_caught_by_driver(site, mode):
    root = Root()
    root.mode = mode
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), capture_prompt_residual(root, tracker=tracker, layer=0, hook_site=site) as holder:
        try:
            root(inputs_embeds=x())
        except (ValueError, RuntimeError):
            pass
        assert not holder.ready
        with pytest.raises(ValueError):
            holder.require_source()
        root.mode = ''
        with pytest.raises(ValueError):
            root(inputs_embeds=x())
    clean(root)


@pytest.mark.parametrize('kwargs,length', [({'cache_position': torch.tensor([0])}, 1), ({}, 1),
                                          ({'cache_position': torch.tensor([0., 1., 2.])}, 3)])
def test_first_metadata_failure(kwargs, length):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre') as holder:
        with pytest.raises(ValueError):
            root(inputs_embeds=x(length), **kwargs)
        with pytest.raises(ValueError):
            holder.require_source()
    clean(root)


@pytest.mark.parametrize('positions', [[], [True], [0, 0], [-1], [3], [1.], '0'])
def test_bad_positions(positions):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), pytest.raises(ValueError):
        with capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre', prompt_positions=positions):
            pass
    clean(root)


@pytest.mark.parametrize('layer,site', [(True, 'pre'), (0., 'pre'), (-1, 'pre'), (2, 'pre'), (0, 'bad')])
def test_bad_coordinates(layer, site):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), pytest.raises(ValueError):
        with capture_prompt_residual(root, tracker=tracker, layer=layer, hook_site=site):
            pass
    clean(root)


def test_tracker_gate():
    root, other = Root(), Root()
    tracker = GenerationPositionTracker(3, True)
    with pytest.raises(ValueError):
        with capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre'):
            pass
    with tracker.track(other), pytest.raises(ValueError):
        with capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre'):
            pass
    clean(root)
    clean(other)


@pytest.mark.parametrize('bad', [torch.ones(1, 3, 2, dtype=torch.long), torch.full((1, 3, 2), float('nan')),
                               torch.zeros(2, 3, 2), torch.zeros(1, 2, 2), torch.zeros(3, 2), torch.zeros(1, 3, 0)])
def test_bad_site_values(bad):
    root = Root()
    external = root.layers[0].register_forward_pre_hook(lambda m, a: (bad,))
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre') as holder:
        with pytest.raises(ValueError):
            root(inputs_embeds=x())
        with pytest.raises(ValueError):
            holder.require_source()
    assert external.id in root.layers[0]._forward_pre_hooks
    external.remove()
    clean(root)


def test_require_during_forward_and_untracked_site():
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre') as holder:
        def check(*args):
            with pytest.raises(ValueError):
                holder.require_source()
        handle = root.layers[0].register_forward_hook(check)
        root(inputs_embeds=x())
        handle.remove()
        with pytest.raises(ValueError):
            root.layers[0](x())
        with pytest.raises(ValueError):
            holder.require_source()
    clean(root)


@pytest.mark.parametrize('registration', ['register_forward_hook', 'site'])
def test_registration_rollback(monkeypatch, registration):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root):
        before = (dict(root._forward_pre_hooks), dict(root._forward_hooks))
        def fail(*args, **kwargs):
            raise RuntimeError('registration failed')
        target = root if registration != 'site' else root.layers[0]
        name = registration if registration != 'site' else 'register_forward_pre_hook'
        monkeypatch.setattr(target, name, fail)
        with pytest.raises(RuntimeError, match='registration failed'):
            with capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre'):
                pass
        assert before == (dict(root._forward_pre_hooks), dict(root._forward_hooks))
    clean(root)


def test_body_exception_release():
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with pytest.raises(RuntimeError):
        with tracker.track(root), capture_prompt_residual(root, tracker=tracker, layer=0, hook_site='pre') as holder:
            root(inputs_embeds=x())
            raise RuntimeError('body')
    with pytest.raises(ValueError):
        holder.require_source()
    assert holder._source is None
    clean(root)
