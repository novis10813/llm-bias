from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.inference.stance_interventions import GenerationPositionTracker
from llm_bias.core.stance_neuron_panel import (
    NATIVE_DOSES, SCOPES, NeuronDiscoveryPlan, NeuronPanel, NeuronRole,
    build_neuron_panel, native_neuron_edit,
)


def model(widths=(20, 7)):
    return SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(
        down_proj=torch.nn.Linear(width, 2, bias=False))) for width in widths])


def roles():
    return tuple(NeuronRole(ticker, ticker, role) for ticker, role in zip(
        ('A', 'B', 'C', 'D'), ('fit', 'validation', 'calibration', 'evaluation')))


def plan(panel=None, role_rows=None):
    return NeuronDiscoveryPlan(panel or build_neuron_panel(model()),
                               roles() if role_rows is None else role_rows,
                               'a' * 64, 'b' * 64, 'c' * 64, 'd' * 64)


def test_all_layers_hash_rank_cap_and_fixed_grid():
    panel = build_neuron_panel(model())
    assert panel.widths == (20, 7)
    assert panel.coverage == ((0, 16, 20), (1, 7, 7))
    for layer, width in enumerate(panel.widths):
        ranked = sorted(range(width), key=lambda neuron: (sha256_json({
            'seed': 20261003, 'layer': layer, 'neuron': neuron}), neuron))
        assert panel.coordinates[layer] == tuple(ranked[:min(width, 16)])
    assert len(set(panel.coordinates[0])) == 16
    assert NATIVE_DOSES == (-8, -2, -0.5, 0, 0.5, 2, 8)
    assert SCOPES == ('prompt_only', 'prompt_and_decode')
    assert panel == build_neuron_panel(model())
    assert panel.panel_sha256 == sha256_json(panel.to_dict())
    assert build_neuron_panel(model((21, 7))).panel_sha256 != panel.panel_sha256


@pytest.mark.parametrize('widths', [(True,), (16.0,), (0,), (-1,), (), [16]])
def test_genuine_width_counts(widths):
    with pytest.raises(ValueError):
        NeuronPanel('supported', widths, ((),), '')


@pytest.mark.parametrize('coords', [((True,), ()), ((1.0,), ()), ((0, 0), ()),
                                    ((20,), ()), ((0,),), [(), ()]])
def test_reject_invalid_or_partial_coordinates(coords):
    with pytest.raises(ValueError):
        replace(build_neuron_panel(model()), coordinates=coords)


def test_records_immutable_and_export_detached():
    record = plan()
    with pytest.raises(FrozenInstanceError):
        record.panel.widths = (1,)
    exported = record.to_dict()
    exported['roles'][0]['ticker'] = 'changed'
    assert record.roles[0].ticker == 'A'
    assert record.plan_sha256 == sha256_json(record.to_dict())
    assert hash(record)
    assert replace(record, parent_sha256='e' * 64).plan_sha256 != record.plan_sha256


def test_roles_separate_no_issuer_or_ticker_leakage():
    record = plan()
    for phase, expected in [('discovery', 'A'), ('selection', 'B'),
                            ('calibration', 'C'), ('evaluation', 'D')]:
        assert record.role_ids(phase) == (expected,)
    with pytest.raises(ValueError):
        record.role_ids('fit_and_validation')
    with pytest.raises(ValueError):
        plan(role_rows=roles()[:-1])
    with pytest.raises(ValueError):
        plan(role_rows=roles() + (NeuronRole('A', 'X', 'fit'),))
    with pytest.raises(ValueError):
        plan(role_rows=(roles()[0], NeuronRole('B', 'A', 'validation'), *roles()[2:]))
    with pytest.raises(ValueError):
        plan(role_rows=list(roles()))
    with pytest.raises(ValueError):
        replace(record, inputs_manifest_sha256='invalid')


def test_arms_complete_and_role_independent():
    record = plan()
    arms = tuple(record.arms())
    assert len(arms) == 23 * 7 * 2
    assert len(set(arms)) == len(arms)
    assert {arm.scope for arm in arms} == set(SCOPES)
    assert {arm.delta for arm in arms} == set(NATIVE_DOSES)
    assert all(arm.arm_sha256 == sha256_json(arm.to_dict()) for arm in arms)
    assert tuple(plan(role_rows=tuple(replace(r, issuer_id='X' + r.issuer_id)
                                      for r in roles())).arms()) == arms


def test_unsupported_gpt_moe_no_residual_substitute():
    root = model()
    root.config = SimpleNamespace(model_type='gpt_oss')
    panel = build_neuron_panel(root)
    assert panel.status == 'unsupported'
    assert panel.widths == panel.coordinates == ()
    assert 'routed' in panel.reason
    assert tuple(plan(panel).arms()) == ()
    root = model()
    root.layers[1].mlp.experts = torch.nn.ModuleList([])
    assert build_neuron_panel(root).status == 'unsupported'
    root = model()
    del root.layers[1].mlp.down_proj
    assert build_neuron_panel(root).status == 'unsupported'


class Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [SimpleNamespace(mlp=SimpleNamespace(
            down_proj=torch.nn.Linear(5, 2, bias=False, dtype=torch.float64)))]
        self.projection = self.layers[0].mlp.down_proj
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(10).reshape(2, 5))

    def forward(self, inputs_embeds, **kwargs):
        return self.projection(inputs_embeds)


@pytest.mark.parametrize('scope', SCOPES)
@pytest.mark.parametrize('delta', NATIVE_DOSES)
@pytest.mark.parametrize('use_cache', [False, True])
def test_actual_native_coordinate_scope_zero_and_cleanup(scope, delta, use_cache):
    root = Root()
    record = plan(build_neuron_panel(root))
    arm = next(a for a in record.arms() if a.neuron == 3 and a.scope == scope and a.delta == delta)
    tracker = GenerationPositionTracker(3, use_cache)
    seen = []
    with tracker.track(root), native_neuron_edit(root, panel=record.panel, arm=arm, tracker=tracker):
        handle = root.projection.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
        try:
            for positions in ((0, 1, 2), (3,) if use_cache else (0, 1, 2, 3)):
                x = torch.zeros(1, len(positions), 5, dtype=torch.float64)
                output = root(inputs_embeds=x, cache_position=torch.tensor(positions))
                expected = x.clone()
                for index, position in enumerate(positions):
                    if position < 3 or scope == 'prompt_and_decode':
                        expected[:, index, 3] += delta
                assert torch.equal(seen[-1], expected)
                assert torch.equal(output, torch.nn.functional.linear(expected, root.projection.weight))
                assert not x.any()
        finally:
            handle.remove()
    assert not root.projection._forward_pre_hooks
    assert not root._forward_pre_hooks and not root._forward_hooks


def test_exception_cleanup_and_model_binding():
    root = Root()
    panel = build_neuron_panel(root)
    arm = next(plan(panel).arms())
    tracker = GenerationPositionTracker(3, True)
    with pytest.raises(RuntimeError, match='failure'):
        with tracker.track(root), native_neuron_edit(root, panel=panel, arm=arm, tracker=tracker):
            raise RuntimeError('failure')
    assert not root.projection._forward_pre_hooks
    assert not root._forward_pre_hooks and not root._forward_hooks
    with pytest.raises(ValueError):
        with native_neuron_edit(model(), panel=panel, arm=arm, tracker=tracker):
            pass
    for changes in ({'neuron': True}, {'layer': 0.0}, {'delta': True}, {'delta': 1}, {'scope': 'decode_only'}):
        with pytest.raises(ValueError):
            replace(arm, **changes)


def test_native_width_wrappers_and_invalid_width_capability():
    root = model((19, 3))
    root.layers = [SimpleNamespace(_hf_layer=layer) for layer in root.layers]
    panel = build_neuron_panel(root)
    assert panel.widths == (19, 3)  # output/residual width is 2, not the native width
    root.layers[0]._hf_layer.mlp.down_proj.in_features = True
    assert build_neuron_panel(root).status == 'unsupported'
    with pytest.raises(ValueError):
        build_neuron_panel(model(()))


def test_hashes_ignore_supplied_role_order_but_bind_assignments():
    original = plan()
    assert plan(role_rows=tuple(reversed(roles()))).plan_sha256 == original.plan_sha256
    assert plan(role_rows=tuple(replace(r, issuer_id='X' + r.issuer_id)
                                for r in roles())).plan_sha256 != original.plan_sha256


def test_edit_rejects_outside_panel_and_unsupported_without_hooks():
    root = Root()
    panel = build_neuron_panel(root)
    arm = next(plan(panel).arms())
    for changes in ({'neuron': 5}, {'layer': 1}, {'panel_sha256': 'e' * 64}):
        with pytest.raises(ValueError):
            with native_neuron_edit(root, panel=panel, arm=replace(arm, **changes),
                                    tracker=GenerationPositionTracker(3, True)):
                pass
    unsupported = NeuronPanel('unsupported', (), (), 'routed experts unsupported')
    with pytest.raises(ValueError):
        with native_neuron_edit(root, panel=unsupported, arm=arm,
                                tracker=GenerationPositionTracker(3, True)):
            pass
    assert not root.projection._forward_pre_hooks
