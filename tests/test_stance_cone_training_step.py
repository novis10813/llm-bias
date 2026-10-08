"""Actual CPU frozen-LM forwards, residual captures and backward evidence."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.stance_cone_objectives import TeacherResponse
from llm_bias.core.inference.stance_cone_training_step import (
    TrainingBatch, UnsupportedTrainingModel, stance_cone_training_step,
)


class Block(nn.Module):
    def forward(self, hidden):
        return (hidden * 2, 'untouched')


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 3, dtype=torch.float64)
        self.layers = nn.ModuleList([Block()])
        self.head = nn.Linear(3, 5, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([[1., 2., 0.], [0., 1., 2.],
                [2., 0., 1.], [1., 0., 0.], [0., 0., 0.]]))
            self.head.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.],
                [0., 0., 1.], [-1., 1., 0.], [0., -1., 1.]]))
        self.requires_grad_(False)
        self.eval()
        self.calls = []
        self.captures = []

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, *, attention_mask, use_cache):
        self.calls.append((input_ids.clone(), attention_mask.clone(), use_cache, torch.is_grad_enabled()))
        raw = self.embedding(input_ids) * 2
        output, tail = self.layers[0](self.embedding(input_ids))
        assert tail == 'untouched'
        self.captures.append((raw.detach().clone(), output.detach().clone()))
        # Causal downstream mixing makes prompt edits affect response predictions.
        logits = self.head(output.cumsum(dim=1))
        return SimpleNamespace(logits=logits)


def setup():
    root = TinyLM()
    model = SimpleNamespace(hf_model=root, layers=root.layers)
    batches = {}
    for purpose in ('addition', 'ablation', 'retain'):
        tokens = (0, 1, 2, 3, 4)
        record = TeacherResponse(ticker='A', role='fit', purpose=purpose,
            token_ids=tokens, response_mask=(False, False, True, True, False),
            source_sha256='a'*64, tokenizer_sha256='b'*64, schema_sha256='c'*64,
            token_ids_sha256=sha256_json(list(tokens)), response_source='construction_generated',
            target_decision='sell' if purpose == 'ablation' else 'buy', review_sha256='d'*64)
        batches[purpose] = TrainingBatch((record,), torch.tensor([[1, 1, 1, 1, 0]]),
            torch.tensor([[True, True, False, False, False]]))
    basis = torch.tensor([[1., 2., 1.], [2., -1., 1.]], dtype=torch.float64, requires_grad=True)
    coefficients = torch.tensor([[1., 2.], [3., 1.], [2., 2.]], dtype=torch.float64)
    return model, batches, basis, coefficients


def run(model, batches, basis, coefficients):
    return stance_cone_training_step(model, batches=batches, roles={'A': 'fit'},
        basis=basis, coefficients=coefficients, layer=0, dose=.7)


def analytic_losses(root, batches, direction):
    # Tiny causal LM closed-form predictions, independent of hooks/objective helpers.
    values = []
    for purpose in ('addition', 'ablation', 'retain'):
        batch = batches[purpose]
        tokens = torch.tensor([r.token_ids for r in batch.records])
        clean = root.embedding(tokens) * 2
        delta = .7 * direction if purpose != 'ablation' else -(clean @ direction)[..., None] * direction
        edited = clean + torch.where(batch.prompt_position_mask[..., None], delta, 0.)
        logp = (edited.cumsum(1) @ root.head.weight.T).log_softmax(-1)
        if purpose == 'retain':
            clean_log = (clean.cumsum(1) @ root.head.weight.T).log_softmax(-1)
            losses = (clean_log.exp() * (clean_log - logp)).sum(-1)[:, 1:3]
        else:
            losses = -torch.stack((logp[:, 1, 2], logp[:, 2, 3]), dim=1)
        values.append(losses.mean())
    return values


def test_real_forward_loss_gradients_and_residual_capture():
    model, batches, basis, coefficients = setup()
    before = [p.clone() for p in model.hf_model.parameters()]
    result = run(model, batches, basis, coefficients)
    directions = basis / basis.norm(dim=-1, keepdim=True)
    rays = (coefficients / coefficients.sum(-1, keepdim=True)) @ directions
    rays = rays / rays.norm(dim=-1, keepdim=True)
    expected = []
    for panel in (directions, rays):
        expected.append(torch.stack([torch.stack(analytic_losses(model.hf_model, batches, d)) for d in panel]).mean(0))
    terms = expected[0] + expected[1]
    torch.testing.assert_close(torch.stack((result.addition, result.ablation, result.retain)), terms)
    torch.testing.assert_close(result.total, terms.sum())
    assert result.weights == (1., 1., 1.)
    result.total.backward()
    assert torch.isfinite(basis.grad).all() and (basis.grad.norm(dim=1) > 0).all()
    for original, parameter in zip(before, model.hf_model.parameters()):
        assert parameter.grad is None and not parameter.requires_grad
        torch.testing.assert_close(original, parameter)
    assert len(model.hf_model.calls) == 16  # one clean + three objectives per basis/ray
    assert sum(not call[3] for call in model.hf_model.calls) == 1
    for _, attention, cache, _ in model.hf_model.calls:
        assert cache is False
        assert torch.equal(attention, batches['retain'].attention_mask)
    clean, unedited = model.hf_model.captures[0]
    torch.testing.assert_close(clean, unedited)
    raw, addition = model.hf_model.captures[1]
    torch.testing.assert_close(addition[:, :2], raw[:, :2] + .7 * directions[0].detach())
    torch.testing.assert_close(addition[:, 2:], raw[:, 2:])
    raw, ablation = model.hf_model.captures[2]
    torch.testing.assert_close(ablation[:, :2] @ directions[0].detach(), torch.zeros(1, 2, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(ablation[:, 2:], raw[:, 2:])
    assert not model.layers[0]._forward_hooks


def test_retain_alone_has_real_gradient_and_clean_is_detached():
    model, batches, basis, coefficients = setup()
    seen = []
    handle = model.hf_model.head.register_forward_hook(lambda _m, _a, output: seen.append(output.requires_grad))
    result = run(model, batches, basis, coefficients)
    handle.remove()
    result.retain.backward()
    assert result.retain.item() > 0 and basis.grad.abs().sum() > 0
    assert seen[0] is False and all(seen[1:])
    assert all(p.grad is None for p in model.hf_model.parameters())


def test_equal_example_mean_with_different_response_lengths():
    model, batches, basis, coefficients = setup()
    single = run(model, batches, basis, coefficients)
    for name, batch in batches.items():
        second = replace(batch.records[0], ticker='B', response_mask=(False, False, True, False, False))
        batches[name] = TrainingBatch((batch.records[0], second), batch.attention_mask.repeat(2, 1), batch.prompt_position_mask.repeat(2, 1))
    result = stance_cone_training_step(model, batches=batches, roles={'A': 'fit', 'B': 'fit'},
        basis=basis, coefficients=coefficients, layer=0, dose=.7)
    shorter = {name: TrainingBatch((batch.records[1],), batch.attention_mask[:1], batch.prompt_position_mask[:1]) for name, batch in batches.items()}
    other = stance_cone_training_step(model, batches=shorter, roles={'B': 'fit'}, basis=basis, coefficients=coefficients, layer=0, dose=.7)
    torch.testing.assert_close(result.total, (single.total + other.total) / 2)


@pytest.mark.parametrize('case', ['roles', 'mask_dtype', 'mask_shape', 'response_position', 'padding_position',
    'response_padding', 'empty_positions', 'negative_coefficients', 'zero_basis', 'nan_basis', 'dtype',
    'dimension', 'dose', 'layer', 'trainable', 'training', 'token', 'purpose', 'missing_objective',
    'attention_value', 'unequal_lengths', 'binding', 'cancelled_ray', 'no_basis_grad'])
def test_invalid_input_rejected_before_forward(case):
    model, batches, basis, coefficients = setup()
    kwargs = dict(model=model, batches=batches, roles={'A': 'fit'}, basis=basis, coefficients=coefficients, layer=0, dose=.7)
    batch = batches['addition']
    if case == 'roles': kwargs['roles'] = {'A': 'evaluation'}
    if case == 'mask_dtype': batches['addition'] = replace(batch, prompt_position_mask=batch.prompt_position_mask.long())
    if case == 'mask_shape': batches['addition'] = replace(batch, attention_mask=torch.ones(1, 4, dtype=torch.long))
    if case in ('response_position', 'padding_position'):
        mask = batch.prompt_position_mask.clone(); mask[0, 2 if case == 'response_position' else 4] = True
        batches['addition'] = replace(batch, prompt_position_mask=mask)
    if case == 'response_padding':
        mask = batch.attention_mask.clone(); mask[0, 2] = 0
        batches['addition'] = replace(batch, attention_mask=mask)
    if case == 'empty_positions': batches['addition'] = replace(batch, prompt_position_mask=torch.zeros_like(batch.prompt_position_mask))
    if case == 'negative_coefficients': kwargs['coefficients'] = -coefficients
    if case == 'zero_basis': kwargs['basis'] = torch.zeros_like(basis, requires_grad=True)
    if case == 'nan_basis': kwargs['basis'] = torch.full_like(basis, float('nan'), requires_grad=True)
    if case == 'dtype': kwargs['basis'] = basis.float()
    if case == 'dimension': kwargs['basis'] = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
    if case == 'dose': kwargs['dose'] = 0
    if case == 'layer': kwargs['layer'] = 1
    if case == 'trainable': model.hf_model.head.weight.requires_grad_(True)
    if case == 'training': model.hf_model.train()
    if case == 'token':
        tokens = (0, 1, 9, 3, 4)
        batches['addition'] = replace(batch, records=(replace(batch.records[0], token_ids=tokens, token_ids_sha256=sha256_json(list(tokens))),))
    if case == 'purpose': batches['addition'] = batches['retain']
    if case == 'missing_objective': del batches['retain']
    if case == 'attention_value': batches['addition'] = replace(batch, attention_mask=batch.attention_mask * 2)
    if case == 'unequal_lengths':
        tokens = (0, 1, 2, 3)
        second = replace(batch.records[0], token_ids=tokens, response_mask=(False, False, True, True), token_ids_sha256=sha256_json(list(tokens)))
        batches['addition'] = replace(batch, records=(batch.records[0], second))
    if case == 'binding': batches['addition'] = replace(batch, records=(replace(batch.records[0], tokenizer_sha256='e'*64),))
    if case == 'cancelled_ray':
        kwargs['basis'] = torch.tensor([[1., 0., 0.], [-1., 0., 0.]], dtype=torch.float64, requires_grad=True)
        kwargs['coefficients'] = torch.ones(1, 2, dtype=torch.float64)
    if case == 'no_basis_grad': kwargs['basis'] = basis.detach()
    with pytest.raises(ValueError): stance_cone_training_step(**kwargs)
    assert model.hf_model.calls == []
    assert not model.layers[0]._forward_hooks


def test_exception_cleanup_and_model_state_preservation():
    model, batches, basis, coefficients = setup()
    def fail_on_edited(_module, _args, _output):
        if len(model.hf_model.calls) == 2:
            raise RuntimeError('primary hook failure')
    handle = model.hf_model.head.register_forward_hook(fail_on_edited)
    with pytest.raises(RuntimeError, match='primary hook failure'):
        run(model, batches, basis, coefficients)
    handle.remove()
    assert not model.layers[0]._forward_hooks
    assert not model.hf_model.training
    run(model, batches, basis, coefficients).total.backward()
    assert basis.grad.abs().sum() > 0


def test_unsupported_model_and_disabled_autograd():
    model, batches, basis, coefficients = setup()
    with pytest.raises(UnsupportedTrainingModel):
        run(SimpleNamespace(layers=model.layers), batches, basis, coefficients)
    with torch.no_grad(), pytest.raises(ValueError, match='autograd'):
        run(model, batches, basis, coefficients)


def test_bad_residual_shape_is_capability_error_and_hook_removed():
    model, batches, basis, coefficients = setup()
    # Make the unsupported shape appear only on edited forwards.
    def wrong_shape(_m, _a, output):
        return (output[0].transpose(1, 2), output[1]) if len(model.hf_model.calls) > 1 else output
    handle = model.layers[0].register_forward_hook(wrong_shape)
    with pytest.raises(UnsupportedTrainingModel, match='batch-first'):
        run(model, batches, basis, coefficients)
    assert len(model.layers[0]._forward_hooks) == 1
    handle.remove()
