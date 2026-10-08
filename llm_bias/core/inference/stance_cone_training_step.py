"""One transient, differentiable teacher-forced cone loss, not a training loop.

The runner owns initialization, dose, layer, prompt masks, backward and optimizer.
No inference-only hooks, artifacts, clean distributions or LM gradients escape.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Real

import torch

from ..stance_cone_objectives import (
    ConeObjective, TeacherResponse, cone_objective, positive_rays, sequence_ce,
    sequence_kl,
)


class UnsupportedTrainingModel(TypeError):
    """The supplied architecture cannot expose the required residual contract."""


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Equal-width full input teachers and explicit original-prompt positions.

    attention_mask is binary bool/long [B,T], prompt_position_mask is bool [B,T].
    Response targets are the complete-input masks in the immutable records.
    Selected positions must precede the first response, never padding/response.
    Tokenization and source/target adaptation remain the teacher compiler's job.
    """

    records: tuple[TeacherResponse, ...]
    attention_mask: torch.Tensor
    prompt_position_mask: torch.Tensor


_OBJECTIVES = ('addition', 'ablation', 'retain')


def _capabilities(model):
    root = getattr(model, 'hf_model', None)
    layers = getattr(model, 'layers', None)
    if not isinstance(root, torch.nn.Module) or layers is None:
        raise UnsupportedTrainingModel('require hf_model nn.Module and decoder layers')
    try:
        embedding = root.get_input_embeddings()
        head = root.get_output_embeddings()
    except (AttributeError, NotImplementedError) as exc:
        raise UnsupportedTrainingModel('require input/output embedding metadata') from exc
    if (not isinstance(embedding, torch.nn.Module) or not isinstance(head, torch.nn.Module)
            or not torch.is_tensor(getattr(embedding, 'weight', None))
            or not torch.is_tensor(getattr(head, 'weight', None))
            or embedding.weight.ndim != 2 or head.weight.ndim != 2):
        raise UnsupportedTrainingModel('require rank-two input/output embedding weights')
    return root, layers, embedding.weight, head.weight


def _validate_batches(batches, roles, *, device, input_vocab, output_vocab):
    if (not isinstance(batches, Mapping) or set(batches) != set(_OBJECTIVES)
            or not isinstance(roles, Mapping)):
        raise ValueError('require addition/ablation/retain batches and authoritative roles')
    bindings = set()
    for purpose in _OBJECTIVES:
        batch = batches[purpose]
        if (type(batch) is not TrainingBatch or type(batch.records) is not tuple
                or not batch.records):
            raise ValueError('require nonempty immutable TrainingBatch teachers')
        for record in batch.records:
            if type(record) is not TeacherResponse:
                raise ValueError('expected TeacherResponse')
            record.__post_init__()
            if (record.purpose != purpose or roles.get(record.ticker) != 'fit'
                    or max(record.token_ids) >= min(input_vocab, output_vocab)):
                raise ValueError('invalid teacher purpose, authoritative role or token vocabulary')
            bindings.add((record.tokenizer_sha256, record.schema_sha256))
        length = len(batch.records[0].token_ids)
        if any(len(r.token_ids) != length for r in batch.records):
            raise ValueError('teachers must have equal padded lengths within a batch')
        shape = (len(batch.records), length)
        attention, prompt = batch.attention_mask, batch.prompt_position_mask
        if (not torch.is_tensor(attention) or attention.shape != shape
                or attention.dtype not in (torch.bool, torch.long)
                or attention.device != device or not ((attention == 0) | (attention == 1)).all().item()
                or not torch.is_tensor(prompt) or prompt.shape != shape
                or prompt.dtype != torch.bool or prompt.device != device):
            raise ValueError('invalid attention/prompt mask shape, dtype, device or values')
        response = torch.tensor([r.response_mask for r in batch.records], device=device)
        # The predictor immediately preceding each response target must also be live.
        if (response & ~attention.bool()).any().item() or (
                response[:, 1:] & ~attention[:, :-1].bool()).any().item():
            raise ValueError('response target or its causal predictor is padding')
        positions = torch.arange(length, device=device)[None]
        first_response = torch.where(response, positions, length).min(-1).values[:, None]
        if (not prompt.any(-1).all().item()
                or (prompt & (~attention.bool() | (positions >= first_response))).any().item()):
            raise ValueError('positions must select nonempty original prompt tokens only')
    if len(bindings) != 1:
        raise ValueError('all batches must share tokenizer and schema bindings')


@contextmanager
def _post_edit(block, direction, prompt, *, operation, dose):
    calls = 0

    def hook(_module, _args, output):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise UnsupportedTrainingModel('selected decoder block must execute once per forward')
        if torch.is_tensor(output):
            hidden = output
        elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            hidden = output[0]
        else:
            raise UnsupportedTrainingModel('require tensor or tensor-first tuple/list block output')
        if (hidden.shape != (*prompt.shape, direction.numel()) or not hidden.is_floating_point()
                or hidden.device != direction.device or hidden.dtype != direction.dtype):
            raise UnsupportedTrainingModel('require matching batch-first [B,T,H] residual dtype/device')
        if not torch.isfinite(hidden).all().item():
            raise ValueError('nonfinite post-block residual')
        delta = (dose * direction if operation == 'addition'
                 else -(hidden * direction).sum(-1, keepdim=True) * direction)
        # Out-of-place edit keeps the original tensor and all unselected tokens intact.
        edited = hidden + torch.where(prompt[..., None], delta, 0.)
        if torch.is_tensor(output):
            return edited
        return (edited, *output[1:]) if isinstance(output, tuple) else [edited, *output[1:]]

    handle = block.register_forward_hook(hook)
    try:
        yield
        if calls != 1:
            raise UnsupportedTrainingModel('selected decoder block did not execute')
    finally:
        handle.remove()


def _forward(root, batch, index, *, device, vocab):
    ids = torch.tensor([batch.records[index].token_ids], dtype=torch.long, device=device)
    output = root(input_ids=ids, attention_mask=batch.attention_mask[index:index + 1], use_cache=False)
    logits = getattr(output, 'logits', None)
    if (not torch.is_tensor(logits) or logits.shape != (1, ids.shape[1], vocab)
            or not logits.is_floating_point() or logits.device != device):
        raise UnsupportedTrainingModel('require full-sequence batch-first output.logits')
    if not torch.isfinite(logits).all().item():
        raise ValueError('nonfinite sequence logits')
    return logits


def stance_cone_training_step(
    model, *, batches: Mapping[str, TrainingBatch], roles: Mapping[str, str],
    basis: torch.Tensor, coefficients: torch.Tensor, layer: int, dose: float,
) -> ConeObjective:
    """Return differentiable 1/1/1 loss terms for one frozen-LM mini-batch.

    basis [K,H] requires gradients. Strictly positive coefficients [R,K] declare
    sampled rays. Rows and rays are normalized differentiably. Basis and sample
    losses are separately averaged by the accepted cone objective. Each example
    is forwarded alone to avoid assuming padding/batching equivalence.

    The model must already be frozen and in eval mode. This function changes no
    model state, installs only temporary post-block hooks and does not backward,
    update parameters or serialize tensors. Caller invokes result.total.backward().
    Clean retain predictions run under no_grad and are reused across directions.
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError('edited forwards require enabled autograd')
    root, layers, embedding, head = _capabilities(model)
    if any(p.requires_grad for p in root.parameters()):
        raise ValueError('all LM parameters must already be frozen')
    if any(module.training for module in root.modules()):
        raise ValueError('LM must already be in eval mode')
    if type(layer) is not int or not 0 <= layer < len(layers):
        raise ValueError('invalid decoder layer')
    block = layers[layer]
    if not isinstance(block, torch.nn.Module) or not any(block is module for module in root.modules()):
        raise UnsupportedTrainingModel('selected layer must be a module in hf_model')
    if isinstance(dose, bool) or not isinstance(dose, Real) or not math.isfinite(dose) or dose <= 0:
        raise ValueError('dose must be a positive finite scalar')
    if (not torch.is_tensor(basis) or not basis.requires_grad or basis.ndim != 2
            or basis.shape[1] != embedding.shape[1] or basis.device != embedding.device
            or basis.dtype != embedding.dtype):
        raise ValueError('require grad-compatible basis matching input embedding dimension/dtype/device')
    _, rays = positive_rays(basis, coefficients)
    units = basis / basis.norm(dim=-1, keepdim=True)
    _validate_batches(batches, roles, device=basis.device,
                      input_vocab=embedding.shape[0], output_vocab=head.shape[0])
    clean = []
    try:
        with torch.no_grad():
            for index in range(len(batches['retain'].records)):
                clean.append(_forward(root, batches['retain'], index,
                                      device=basis.device, vocab=head.shape[0]).detach())
        panels = []
        for directions in (units, rays):
            losses = {name: [] for name in _OBJECTIVES}
            for direction in directions:
                for purpose in _OBJECTIVES:
                    batch = batches[purpose]
                    examples = []
                    for index, record in enumerate(batch.records):
                        with _post_edit(block, direction, batch.prompt_position_mask[index:index + 1],
                                        operation='ablation' if purpose == 'ablation' else 'addition', dose=dose):
                            logits = _forward(root, batch, index, device=basis.device, vocab=head.shape[0])
                        if purpose == 'retain':
                            loss = sequence_kl(logits[None], clean[index], (record,), roles=roles)
                        else:
                            loss = sequence_ce(logits[None], (record,), roles=roles, purpose=purpose)
                        examples.append(loss[0])
                    losses[purpose].append(torch.stack(examples).mean())
            panels.append({name: torch.stack(values) for name, values in losses.items()})
        return cone_objective(*panels)
    finally:
        clean.clear()
