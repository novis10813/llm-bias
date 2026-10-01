from contextlib import ExitStack
from types import SimpleNamespace
import json

import pytest
import torch

from llm_bias.core.inference.stance_interventions import (
    GenerationPositionTracker,
    UnsupportedPositionMetadata,
    scoped_mlp_addition,
    scoped_residual_intervention,
)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.mlp.down_proj.weight.copy_(torch.tensor([[1., 2., 3.], [-2., 1., 4.]]))
        self.native_inputs = []
        self.pre_inputs = []

    def forward(self, hidden_states):
        self.pre_inputs.append(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        native = torch.cat((hidden_states, hidden_states[..., :1]), dim=-1)
        self.native_inputs.append(native)
        return hidden_states + self.mlp.down_proj(native)


class Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.fail = False
        self.positions = []
        self.tracker = None

    def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        if self.tracker is not None:
            self.positions.append(self.tracker.active_positions)
        if self.fail:
            raise RuntimeError('root failure')
        hidden = inputs_embeds if inputs_embeds is not None else input_ids.double().unsqueeze(-1).expand(-1, -1, 2)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class Cache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


def legacy_cache(length=3):
    return tuple((torch.zeros(1, 2, length, 4), torch.zeros(1, 2, length, 4)) for _ in range(2))


def clean(root):
    for module in root.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def run_selected(calls, *, use_cache, scope, prompt_positions=None):
    root = Root()
    tracker = GenerationPositionTracker(3, use_cache)
    root.tracker = tracker
    masks = []
    with tracker.track(root):
        with scoped_residual_intervention(root, tracker=tracker, layer=0, hook_site='pre',
                                          scope=scope, operation='addition', vector=torch.tensor([1., 0.]),
                                          prompt_positions=prompt_positions) as diagnostics:
            for length, kwargs in calls:
                x = torch.zeros(1, length, 2, dtype=torch.float64)
                root(inputs_embeds=x, **kwargs)
                masks.append(root.layers[0].pre_inputs[-1][0, :, 0].tolist())
                assert tracker.active_positions is None
        json.dumps(diagnostics.to_dict(), allow_nan=False)
    clean(root)
    return root.positions, masks, diagnostics.to_dict()


@pytest.mark.parametrize('scope,expected', [
    ('prompt_only', [[0., 1., 0.], [0.], [0., 0.]]),
    ('decode_only', [[0., 0., 0.], [1.], [1., 1.]]),
    ('prompt_and_decode', [[0., 1., 0.], [1.], [1., 1.]]),
])
@pytest.mark.parametrize('cache_kind', ['explicit', 'modern', 'legacy'])
def test_cached_absolute_scopes(scope, expected, cache_kind):
    if cache_kind == 'explicit':
        calls = [(3, {'cache_position': torch.arange(3)}), (1, {'cache_position': torch.tensor([3])}),
                 (2, {'cache_position': torch.tensor([4, 5])})]
    else:
        factory = Cache if cache_kind == 'modern' else legacy_cache
        calls = [(3, {}), (1, {'past_key_values': factory(3)}), (2, {'past_key_values': factory(4)})]
    positions, masks, diagnostics = run_selected(calls, use_cache=True, scope=scope,
                                                 prompt_positions=None if scope == 'decode_only' else [1])
    assert positions == [(0, 1, 2), (3,), (4, 5)]
    assert masks == expected
    assert diagnostics['selected_token_opportunities'] == sum(sum(row) for row in expected)


@pytest.mark.parametrize('scope,expected', [
    ('prompt_only', [[0., 1., 0.], [0., 1., 0., 0.], [0., 1., 0., 0., 0.]]),
    ('decode_only', [[0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 1., 1.]]),
    ('prompt_and_decode', [[0., 1., 0.], [0., 1., 0., 1.], [0., 1., 0., 1., 1.]]),
])
def test_no_cache_recomputation(scope, expected):
    positions, masks, _ = run_selected([(3, {}), (4, {}), (5, {})], use_cache=False, scope=scope,
                                       prompt_positions=None if scope == 'decode_only' else [1])
    assert positions == [(0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4)]
    assert masks == expected


def test_chunked_one_token_prefill_is_not_decode():
    positions, masks, _ = run_selected([
        (1, {'cache_position': torch.tensor([0])}),
        (2, {'cache_position': torch.tensor([1, 2]), 'past_key_values': object()}),
        (1, {'position_ids': torch.tensor([[3]])}),
    ], use_cache=True, scope='prompt_only', prompt_positions=[0, 2])
    assert positions == [(0,), (1, 2), (3,)]
    assert masks == [[1.], [0., 1.], [0.]]


@pytest.mark.parametrize('length,kwargs', [
    (1, {}), (2, {'past_key_values': ()}), (1, {'past_key_values': []}),
    (1, {'past_key_values': object()}), (1, {'past_key_values': Cache(True)}),
    (1, {'past_key_values': Cache(-1)}), (1, {'past_key_values': Cache(3.0)}),
    (1, {'past_key_values': ((torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),)}),
    (1, {'past_key_values': ((torch.zeros(2, 1, 3, 2), torch.zeros(2, 1, 3, 2)),)}),
    (1, {'past_key_values': legacy_cache(3) + legacy_cache(4)}),
    (1, {'past_key_values': ((torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 4, 4)),)}),
    (1, {'cache_position': torch.tensor([False])}),
    (1, {'cache_position': torch.tensor([0.])}),
    (1, {'cache_position': torch.tensor([-1])}),
    (2, {'cache_position': torch.tensor([0, 0])}),
    (2, {'cache_position': torch.tensor([0])}),
    (1, {'cache_position': torch.tensor([[0]])}),
    (1, {'position_ids': torch.tensor([0])}),
    (1, {'cache_position': torch.tensor([0]), 'position_ids': torch.tensor([[1]])}),
])
def test_unsupported_metadata_fails_before_edits(length, kwargs):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), scoped_mlp_addition(root, tracker=tracker, layer=0, neuron=1,
                                                delta=0., scope='prompt_only'):
        with pytest.raises(UnsupportedPositionMetadata):
            root(torch.zeros(1, length, dtype=torch.long), **kwargs)
        assert tracker.active_positions is None
        assert not root.layers[0].pre_inputs
    clean(root)


@pytest.mark.parametrize('prompt_length,use_cache', [(True, True), (0, True), (-1, True), (3., True), (3, 1)])
def test_tracker_constructor_validation(prompt_length, use_cache):
    with pytest.raises(ValueError):
        GenerationPositionTracker(prompt_length, use_cache)


@pytest.mark.parametrize('kwargs', [
    {'input_ids': torch.zeros(2, 3, dtype=torch.long)},
    {'input_ids': torch.zeros(1, 3, 1, dtype=torch.long)},
    {'inputs_embeds': torch.zeros(1, 3)},
    {'input_ids': torch.zeros(1, 3), 'inputs_embeds': torch.zeros(1, 3, 2)},
    {},
])
def test_root_shapes_and_exclusivity(kwargs):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), pytest.raises(UnsupportedPositionMetadata):
        root(**kwargs)
    clean(root)


@pytest.mark.parametrize('cache', [None, (), Cache(0)])
def test_full_initial_prefill_empty_cache(cache):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    root.tracker = tracker
    with tracker.track(root):
        root(torch.zeros(1, 3, dtype=torch.long), past_key_values=cache)
    assert root.positions == [(0, 1, 2)]


def test_no_cache_partial_prefill_rejected():
    root = Root()
    with GenerationPositionTracker(3, False).track(root), pytest.raises(UnsupportedPositionMetadata):
        root(torch.zeros(1, 2, dtype=torch.long))
    clean(root)


@pytest.mark.parametrize('wrapper', ['bare', 'hf_model', '_hf_model'])
def test_root_resolution(wrapper):
    root = Root()
    model = root if wrapper == 'bare' else SimpleNamespace(**{wrapper: root})
    tracker = GenerationPositionTracker(3, True)
    root.tracker = tracker
    with tracker.track(model):
        root(torch.zeros(1, 3, dtype=torch.long), position_ids=torch.tensor([[2, 0, 1]]),
             cache_position=torch.tensor([2, 0, 1]))
    assert root.positions == [(2, 0, 1)]
    clean(root)


def test_tracker_lifetime_and_exception_cleanup(monkeypatch):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with pytest.raises(ValueError):
        tracker.select(torch.zeros(1, 3, 2), scope='prompt_only')
    with tracker.track(root):
        with pytest.raises(ValueError):
            with tracker.track(root):
                pass
        root.fail = True
        with pytest.raises(RuntimeError, match='root failure'):
            root(torch.zeros(1, 3, dtype=torch.long))
        assert tracker.active_positions is None
    clean(root)
    with pytest.raises(ValueError):
        with tracker.track(root):
            pass
    tracker = GenerationPositionTracker(3, True)
    def fail(*args, **kwargs):
        raise RuntimeError('install failed')
    monkeypatch.setattr(root, 'register_forward_hook', fail)
    with pytest.raises(RuntimeError, match='install failed'):
        with tracker.track(root):
            pass
    clean(root)
    assert tracker.active_positions is None


@pytest.mark.parametrize('scope,positions', [('unknown', None), ('decode_only', [0]),
                                            ('prompt_only', [True]), ('prompt_only', [1, 1]),
                                            ('prompt_only', [-1]), ('prompt_only', [3])])
def test_invalid_scope_selection(scope, positions):
    with pytest.raises(ValueError):
        run_selected([(3, {})], use_cache=True, scope=scope, prompt_positions=positions)


@pytest.mark.parametrize('shape', [(2, 3, 2), (1, 2, 2), (3, 2)])
def test_hidden_shape_must_match_root(shape):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root):
        handle = root.layers[0].register_forward_pre_hook(
            lambda *_: tracker.select(torch.zeros(shape), scope='prompt_only'))
        try:
            with pytest.raises(ValueError):
                root(torch.zeros(1, 3, dtype=torch.long))
        finally:
            handle.remove()
        assert tracker.active_positions is None
    clean(root)


@pytest.mark.parametrize('site', ['pre', 'mid', 'post'])
@pytest.mark.parametrize('operation', ['addition', 'ablation', 'replacement'])
def test_residual_operations_exact_alignment(site, operation):
    root = Root()
    x = torch.tensor([[[2., 3.], [4., 5.], [6., 7.]]], dtype=torch.float64)
    target = root.layers[0] if site != 'mid' else root.layers[0].post_attention_layernorm
    observed = []
    registration = target.register_forward_hook if site == 'post' else target.register_forward_pre_hook
    capture = (lambda m, a, out: observed.append(out.clone())) if site == 'post' else (lambda m, a: observed.append(a[0].clone()))
    before = registration(capture)
    root(inputs_embeds=x)
    baseline = observed.pop()
    before.remove()
    tracker = GenerationPositionTracker(3, True)
    args = {'vector': torch.tensor([1., 0.]), 'dose': 2. if operation == 'addition' else 1.}
    if operation == 'replacement':
        args = {'source': torch.tensor([[[30., 40.], [10., 20.]]], dtype=torch.float64),
                'source_positions': [2, 0]}
    with tracker.track(root), scoped_residual_intervention(
        root, tracker=tracker, layer=0, hook_site=site, scope='prompt_only',
        operation=operation, prompt_positions=[0, 2], **args
    ) as diagnostics:
        after = registration(capture)
        try:
            root(inputs_embeds=x)
        finally:
            after.remove()
    actual = observed.pop()
    expected = baseline.clone()
    if operation == 'addition':
        expected[:, [0, 2], 0] += 2.
    elif operation == 'ablation':
        expected[:, [0, 2], 0] = 0.
    else:
        expected[:, [0, 2]] = torch.tensor([[[10., 20.], [30., 40.]]], dtype=torch.float64)
    assert torch.equal(actual, expected)
    data = diagnostics.to_dict()
    assert data['selected_token_opportunities'] == 2
    assert data['changed_token_count'] == 2
    assert data['finite_count'] == 2
    assert data['layer'] == 0 and data['hook_site'] == site
    json.dumps(data, allow_nan=False)
    data['changed_token_count'] = -1
    assert diagnostics.to_dict()['changed_token_count'] == 2
    assert all(not torch.is_tensor(value) for value in vars(diagnostics).values())
    clean(root)


@pytest.mark.parametrize('scale', [1e200, 1e-200, -1e200, -1e-200])
def test_projection_ablation_is_invariant_to_extreme_direction_scale(scale):
    root = Root()
    x = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]], dtype=torch.float64)
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), scoped_residual_intervention(
        root, tracker=tracker, layer=0, hook_site='pre', scope='prompt_only',
        operation='ablation', vector=torch.tensor([scale, 0.], dtype=torch.float64),
        prompt_positions=[0, 2]
    ) as diagnostics:
        root(inputs_embeds=x)
    expected = torch.tensor([[[0., 2.], [3., 4.], [0., 6.]]], dtype=torch.float64)
    assert torch.equal(root.layers[0].pre_inputs[-1], expected)
    assert diagnostics.to_dict()['changed_token_count'] == 2
    assert diagnostics.to_dict()['delta_l2_sum'] == pytest.approx(6.)
    assert tracker.active_positions is None
    json.dumps(diagnostics.to_dict(), allow_nan=False)
    clean(root)


def test_projection_ablation_not_coordinate_zeroing():
    root = Root()
    x = torch.tensor([[[3., 1.], [0., 0.], [7., 8.]]], dtype=torch.float64)
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), scoped_residual_intervention(
        root, tracker=tracker, layer=0, hook_site='pre', scope='prompt_only',
        operation='ablation', vector=torch.tensor([1., 1.]), prompt_positions=[0, 1]
    ) as diagnostics:
        root(inputs_embeds=x)
    assert torch.equal(root.layers[0].pre_inputs[-1], torch.tensor([[[1., -1.], [0., 0.], [7., 8.]]]))
    data = diagnostics.to_dict()
    assert data['delta_l2_sum'] == pytest.approx(8 ** .5)
    assert data['delta_l2_max'] == pytest.approx(8 ** .5)
    assert data['residual_zero_count'] == 1
    assert data['relative_delta_l2_sum'] == pytest.approx((8 / 10) ** .5)


@pytest.mark.parametrize('operation', ['addition', 'replacement'])
def test_zero_and_copied_self_return_original_tensor(operation):
    root = Root()
    x = torch.tensor([[[2., 3.], [4., 5.], [6., 7.]]], dtype=torch.float64)
    tracker = GenerationPositionTracker(3, True)
    args = {'vector': torch.tensor([1., 2.]), 'dose': 0.} if operation == 'addition' else {
        'source': x[:, [2, 0]].clone(), 'source_positions': [2, 0]}
    with tracker.track(root), scoped_residual_intervention(
        root, tracker=tracker, layer=0, hook_site='pre', scope='prompt_only', operation=operation,
        prompt_positions=[0, 2], **args
    ) as diagnostics:
        root(inputs_embeds=x)
        assert root.layers[0].pre_inputs[-1] is x
    assert diagnostics.to_dict()['changed_token_count'] == 0
    assert diagnostics.to_dict()['delta_l2_sum'] == 0
    clean(root)


def test_replacement_chunks_use_exact_absolute_mapping():
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), scoped_residual_intervention(
        root, tracker=tracker, layer=0, hook_site='pre', scope='prompt_only', operation='replacement',
        prompt_positions=[0, 2], source_positions=[2, 0],
        source=torch.tensor([[[20., 21.], [10., 11.]]], dtype=torch.float64)
    ):
        root(inputs_embeds=torch.zeros(1, 1, 2, dtype=torch.float64), cache_position=torch.tensor([0]))
        assert root.layers[0].pre_inputs[-1].tolist() == [[[10., 11.]]]
        root(inputs_embeds=torch.zeros(1, 2, 2, dtype=torch.float64), cache_position=torch.tensor([1, 2]))
        assert root.layers[0].pre_inputs[-1].tolist() == [[[0., 0.], [20., 21.]]]
    clean(root)


@pytest.mark.parametrize('kwargs', [
    {'layer': True}, {'layer': 0.}, {'layer': '0'}, {'layer': -1}, {'hook_site': 'bad'},
    {'operation': 'bad'}, {'dose': True}, {'dose': float('inf')}, {'vector': torch.zeros(2)},
    {'vector': torch.tensor([float('nan'), 0.])}, {'vector': torch.zeros(1, 2)},
    {'operation': 'ablation', 'dose': 0.},
    {'operation': 'replacement', 'source': torch.zeros(1, 1, 2), 'source_positions': [0]},
    {'operation': 'replacement', 'vector': None, 'source': torch.zeros(1, 2, 2), 'source_positions': [0, 1]},
    {'operation': 'replacement', 'vector': None, 'source': torch.zeros(1, 2, 2), 'source_positions': [0, 0]},
    {'operation': 'replacement', 'vector': None, 'source': torch.zeros(1, 2, 2), 'source_positions': [False, 2]},
    {'operation': 'replacement', 'vector': None, 'scope': 'decode_only', 'prompt_positions': None,
     'source': torch.zeros(1, 3, 2), 'source_positions': [0, 1, 2]},
])
def test_operation_validation(kwargs):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    args = dict(layer=0, hook_site='pre', scope='prompt_only', operation='addition',
                vector=torch.ones(2), prompt_positions=[0, 2])
    args.update(kwargs)
    with tracker.track(root), pytest.raises(ValueError):
        with scoped_residual_intervention(root, tracker=tracker, **args):
            root(inputs_embeds=torch.zeros(1, 3, 2, dtype=torch.float64))
    clean(root)


@pytest.mark.parametrize('args', [
    {'vector': torch.ones(3), 'dose': 0.},
    {'vector': torch.tensor([1e100, 1.], dtype=torch.float64)},
    {'operation': 'replacement', 'vector': None, 'source': torch.zeros(1, 3, 2, dtype=torch.float64),
     'source_positions': [0, 1, 2]},
])
def test_runtime_dimension_cast_and_dtype_validation(args):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    kwargs = dict(layer=0, hook_site='pre', scope='prompt_only', operation='addition', vector=torch.ones(2))
    kwargs.update(args)
    with pytest.raises(ValueError):
        with tracker.track(root), scoped_residual_intervention(root, tracker=tracker, **kwargs):
            root(inputs_embeds=torch.zeros(1, 3, 2, dtype=torch.float32))
    assert tracker.active_positions is None
    clean(root)


def test_native_neuron_projection_and_shared_root_positions():
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    x = torch.zeros(1, 3, 2, dtype=torch.float64)
    with tracker.track(root), ExitStack() as stack:
        metadata = stack.enter_context(scoped_mlp_addition(root, tracker=tracker, layer=0, neuron=1,
                                                          delta=2., scope='prompt_only', prompt_positions=[1]))
        stack.enter_context(scoped_mlp_addition(root, tracker=tracker, layer=1, neuron=2,
                                              delta=3., scope='decode_only', prompt_positions=[]))
        output = root(inputs_embeds=x)
        # Layer 0's real down projection contributes [4, 2]; layer 1 then contributes
        # [4+4+12, -8+2+16], on top of its incoming residual.
        assert torch.equal(output, torch.tensor([[[0., 0.], [24., 12.], [0., 0.]]]))
        decode = root(inputs_embeds=torch.zeros(1, 1, 2, dtype=torch.float64), past_key_values=Cache(3))
        assert torch.equal(decode, torch.tensor([[[9., 12.]]]))
        assert metadata['hook_site'] == 'mlp_down_proj_input'
        assert all(not torch.is_tensor(value) for value in metadata.values())
    clean(root)


def test_scoped_zero_mlp_requires_active_root_and_cleanup():
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with tracker.track(root), scoped_mlp_addition(root, tracker=tracker, layer=0, neuron=1,
                                                delta=0., scope='prompt_only'):
        with pytest.raises(ValueError):
            root.layers[0].mlp.down_proj(torch.zeros(1, 3, 3, dtype=torch.float64))
    clean(root)


def test_consumer_exception_and_root_layer_cleanup():
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    with pytest.raises(ValueError):
        with tracker.track(root), scoped_residual_intervention(
            root, tracker=tracker, layer=0, hook_site='pre', scope='prompt_only',
            operation='addition', vector=torch.ones(3)
        ), scoped_mlp_addition(root, tracker=tracker, layer=1, neuron=1, delta=1., scope='decode_only'):
            root(inputs_embeds=torch.zeros(1, 3, 2, dtype=torch.float64))
    assert tracker.active_positions is None
    clean(root)


def test_root_resolution_prefers_explicit_hf_model():
    root, other = Root(), Root()
    tracker = GenerationPositionTracker(3, True)
    root.tracker = tracker
    with tracker.track(SimpleNamespace(hf_model=root, _hf_model=other)):
        root(torch.zeros(1, 3, dtype=torch.long))
        assert not other._forward_hooks and not other._forward_pre_hooks
    assert root.positions == [(0, 1, 2)]
    clean(root)


def test_unhookable_root_fails_closed():
    tracker = GenerationPositionTracker(3, True)
    with pytest.raises(ValueError):
        with tracker.track(SimpleNamespace(hf_model=object())):
            pass
    assert tracker.active_positions is None


@pytest.mark.parametrize('operation', ['addition', 'replacement'])
def test_identity_diagnostics_do_not_subtract_or_cast(monkeypatch, operation):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    x = torch.ones(1, 3, 2, dtype=torch.float64)
    args = {'vector': torch.tensor([1, 2]), 'dose': 0.} if operation == 'addition' else {
        'source': x.clone(), 'source_positions': [0, 1, 2]}
    with tracker.track(root), scoped_residual_intervention(
        root, tracker=tracker, layer=0, hook_site='pre', scope='prompt_only',
        operation=operation, **args
    ) as diagnostics:
        def fail(*args):
            raise AssertionError('identity must not compute deltas')
        monkeypatch.setattr(diagnostics, '_record', fail)
        root(inputs_embeds=x)
        assert root.layers[0].pre_inputs[-1] is x
    assert diagnostics.to_dict()['selected_token_opportunities'] == 3
    assert diagnostics.to_dict()['finite_count'] == 3


def test_empty_selection_returns_original_and_no_opportunities():
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    x = torch.zeros(1, 1, 2, dtype=torch.float64)
    with tracker.track(root), scoped_residual_intervention(
        root, tracker=tracker, layer=0, hook_site='pre', scope='decode_only',
        operation='addition', vector=torch.tensor([1, 2]), prompt_positions=[]
    ) as diagnostics:
        root(inputs_embeds=x, cache_position=torch.tensor([0]))
        assert root.layers[0].pre_inputs[-1] is x
    assert diagnostics.to_dict()['selected_token_opportunities'] == 0
    clean(root)


def test_scoped_consumer_requires_matching_active_tracking_context():
    root, other = Root(), Root()
    tracker = GenerationPositionTracker(3, True)
    args = dict(tracker=tracker, layer=0, neuron=0, delta=0., scope='prompt_only')
    with pytest.raises(ValueError):
        with scoped_mlp_addition(root, **args):
            pass
    with tracker.track(root), pytest.raises(ValueError):
        with scoped_mlp_addition(other, **args):
            pass
    clean(root)
    clean(other)


def test_scoped_mlp_install_failure_removes_root_hooks(monkeypatch):
    root = Root()
    tracker = GenerationPositionTracker(3, True)
    def fail(*args, **kwargs):
        raise RuntimeError('native hook install failed')
    monkeypatch.setattr(root.layers[0].mlp.down_proj, 'register_forward_pre_hook', fail)
    with pytest.raises(RuntimeError, match='native hook install failed'):
        with tracker.track(root), scoped_mlp_addition(
            root, tracker=tracker, layer=0, neuron=0, delta=0., scope='prompt_only'
        ):
            pass
    assert tracker.active_positions is None
    clean(root)
