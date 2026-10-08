"""CPU sequence and cone contracts, without DIM or real checkpoints."""
from dataclasses import FrozenInstanceError, replace
import math

import pytest
import torch

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.stance_cone_objectives import (
    TeacherResponse, sequence_ce, sequence_kl, cone_objective,
    initialize_basis, positive_rays, negative_cone_adaptation,
)


def teacher(purpose='addition', tokens=(0, 1, 2, 1), mask=(False, False, True, True), ticker='A'):
    return TeacherResponse(ticker=ticker, role='fit', purpose=purpose,
        token_ids=tokens, response_mask=mask, source_sha256='a' * 64,
        tokenizer_sha256='b' * 64, schema_sha256='c' * 64,
        token_ids_sha256=sha256_json(list(tokens)), response_source='construction_generated',
        target_decision='buy', review_sha256='d' * 64)


ROLES = {'A': 'fit', 'B': 'fit'}


def test_ce_shift_mask_gradient_and_analytic_value():
    logits = torch.zeros(2, 1, 4, 3, dtype=torch.float64, requires_grad=True)
    losses = sequence_ce(logits, (teacher(),), roles=ROLES, purpose='addition')
    assert losses.shape == (2,)
    torch.testing.assert_close(losses, torch.full((2,), math.log(3), dtype=torch.float64))
    losses.sum().backward()
    expected = torch.zeros_like(logits)
    expected[:, 0, 1:3, :] = 1 / 6
    expected[:, 0, 1, 2] -= .5
    expected[:, 0, 2, 1] -= .5
    torch.testing.assert_close(logits.grad, expected)


def test_length_normalization_before_example_and_direction_mean():
    records = (teacher(), teacher(tokens=(0, 2, 1, 0), mask=(False, True, False, False), ticker='B'))
    logits = torch.zeros(2, 2, 4, 3, dtype=torch.float64)
    logits[0, 0, 1, 2] = math.log(4)
    logits[0, 0, 2, 1] = math.log(4)
    logits[0, 1, 0, 2] = math.log(2)
    losses = sequence_ce(logits, records, roles=ROLES, purpose='addition')
    torch.testing.assert_close(losses, torch.tensor([
        (math.log(1.5) + math.log(2)) / 2, math.log(3)], dtype=torch.float64))


def test_kl_sequence_analytic_gradient_and_frozen_teacher():
    record = teacher('retain')
    clean = torch.zeros(1, 4, 3, dtype=torch.float64, requires_grad=True)
    edited = torch.zeros(1, 1, 4, 3, dtype=torch.float64)
    edited[:, :, 1:3, 0] = math.log(4)
    edited.requires_grad_()
    loss = sequence_kl(edited, clean, (record,), roles=ROLES)
    expected = (math.log(.5) + 2 * math.log(2)) / 3
    torch.testing.assert_close(loss, torch.tensor([expected], dtype=torch.float64))
    loss.sum().backward()
    assert clean.grad is None
    torch.testing.assert_close(edited.grad[0, 0, 1], torch.tensor([1/6, -1/12, -1/12], dtype=torch.float64))
    assert edited.grad[0, 0, (0, 3)].abs().sum() == 0
    torch.testing.assert_close(sequence_kl(clean.detach()[None], clean, (record,), roles=ROLES), torch.zeros(1, dtype=torch.float64))


def test_separate_basis_sample_means_and_unit_weights():
    basis = {name: torch.tensor([1., 3.], requires_grad=True) for name in ('addition', 'ablation', 'retain')}
    samples = {name: torch.tensor([10.], requires_grad=True) for name in basis}
    result = cone_objective(basis, samples)
    assert result.weights == (1., 1., 1.)
    assert result.total.item() == 36
    result.total.backward()
    for value in basis.values():
        torch.testing.assert_close(value.grad, torch.full((2,), .5))
    for value in samples.values():
        torch.testing.assert_close(value.grad, torch.ones(1))


@pytest.mark.parametrize('purpose', ['addition', 'ablation'])
def test_ce_both_named_operations(purpose):
    logits = torch.zeros(1, 1, 4, 3, requires_grad=True)
    sequence_ce(logits, (teacher(purpose),), roles=ROLES, purpose=purpose).sum().backward()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize('role', ['validation', 'calibration', 'evaluation', 'construction', 'Fit'])
def test_teacher_strict_role_and_authoritative_role_guard(role):
    with pytest.raises(ValueError):
        replace(teacher(), role=role)
    with pytest.raises(ValueError):
        sequence_ce(torch.zeros(1, 1, 4, 3), (teacher(),), roles={'A': role}, purpose='addition')


def test_immutable_and_source_token_provenance():
    record = teacher()
    with pytest.raises(FrozenInstanceError):
        record.role = 'evaluation'
    for changes in ({'token_ids': [0, 1, 2, 1]}, {'token_ids_sha256': 'f'*64},
                    {'source_sha256': ''}, {'response_source': 'evaluation_generated'},
                    {'response_mask': (True, False, True, True)},
                    {'response_mask': (False, False, False, False)}):
        with pytest.raises(ValueError):
            replace(record, **changes)


@pytest.mark.parametrize('case', ['empty', 'shape', 'nan', 'vocab', 'role_missing', 'purpose'])
def test_ce_rejects_invalid_inputs(case):
    logits, records, roles, purpose = torch.zeros(1, 1, 4, 3), (teacher(),), ROLES, 'addition'
    if case == 'empty': records = ()
    if case == 'shape': logits = torch.zeros(1, 1, 3, 3)
    if case == 'nan': logits[0, 0, 3, 0] = float('nan')
    if case == 'vocab': logits = torch.zeros(1, 1, 4, 2)
    if case == 'role_missing': roles = {}
    if case == 'purpose': purpose = 'ablation'
    with pytest.raises(ValueError):
        sequence_ce(logits, records, roles=roles, purpose=purpose)


def test_kl_and_reduction_reject_shapes_nonfinite_empty():
    for clean in (torch.zeros(1, 3, 3), torch.full((1, 4, 3), float('inf'))):
        with pytest.raises(ValueError):
            sequence_kl(torch.zeros(1, 1, 4, 3), clean, (teacher('retain'),), roles=ROLES)
    valid = {name: torch.ones(2) for name in ('addition', 'ablation', 'retain')}
    for bad in (torch.empty(0), torch.tensor([float('nan')]), torch.ones(2, 1)):
        with pytest.raises(ValueError):
            cone_objective(valid | {'retain': bad}, valid)
    with pytest.raises(ValueError):
        cone_objective(valid | {'retain': torch.ones(3)}, valid)


@pytest.mark.parametrize('dimension', [2, 4])
@pytest.mark.parametrize('seed', [20261003, 20261004, 20261005])
def test_independent_seeded_basis_and_positive_negative_rays(dimension, seed):
    torch.manual_seed(42)
    state = torch.random.get_rng_state().clone()
    basis = initialize_basis(8, dimension=dimension, seed=seed, dtype=torch.float64)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(basis, initialize_basis(8, dimension=dimension, seed=seed, dtype=torch.float64))
    other_seed = 20261003 if seed != 20261003 else 20261004
    assert not torch.equal(basis, initialize_basis(8, dimension=dimension, seed=other_seed, dtype=torch.float64))
    torch.testing.assert_close(basis.norm(dim=1), torch.ones(dimension, dtype=torch.float64))
    basis.requires_grad_()
    coefficients = torch.arange(1, dimension+1, dtype=torch.float64)[None].requires_grad_()
    normalized, rays = positive_rays(basis, coefficients)
    torch.testing.assert_close(normalized.sum(1), torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(rays.norm(dim=1), torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(negative_cone_adaptation(rays), -rays)
    rays[:, 0].sum().backward()
    assert basis.grad.abs().sum() > 0 and coefficients.grad.abs().sum() > 0


def test_init_and_rays_reject_invalid_inputs():
    for dimension, seed in ((1, 20261003), (2, 1)):
        with pytest.raises(ValueError): initialize_basis(8, dimension=dimension, seed=seed)
    for coefficients in (torch.zeros(1, 2), torch.tensor([[1., -1.]]), torch.ones(0, 2), torch.ones(1, 3)):
        with pytest.raises(ValueError): positive_rays(torch.eye(2), coefficients)
    with pytest.raises(ValueError): positive_rays(torch.tensor([[1., 0.], [-1., 0.]]), torch.ones(1, 2))
