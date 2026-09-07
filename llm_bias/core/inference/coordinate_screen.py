"""Differentiable last-token objectives for transient coordinate screening."""
from contextlib import contextmanager, nullcontext

import torch

from .mlp_addition import mlp_summed_derivatives, mlp_addition


@contextmanager
def frozen_eval(model):
    """Restore per-module training flags and parameter grad flags on every exit."""
    raw = getattr(model, "_hf_model", model)
    training = [(module, module.training) for module in raw.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in raw.parameters()]
    try:
        raw.eval()
        raw.requires_grad_(False)
        yield raw
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, flag in training:
            module.training = flag


def next_token_margin(model, ids, positive_id, negative_id, device):
    raw = getattr(model, "_hf_model", model)
    output = raw(input_ids=torch.tensor([ids], device=device), use_cache=False, return_dict=True)
    logits = output.logits[0, -1].float()
    return logits[positive_id] - logits[negative_id]


def coordinate_derivatives(model, ids, positive_id, negative_id, device, layers, *, save_on_cpu=False):
    """Return transient signed token-summed vectors; never serialize directly.

    Optionally keep tensors saved for backward in CPU RAM, never on disk.
    This leaves the objective, precision, and returned derivatives unchanged.
    """
    saved_tensors = torch.autograd.graph.save_on_cpu(pin_memory=False) if save_on_cpu else nullcontext()
    with frozen_eval(model), torch.enable_grad(), saved_tensors:
        with mlp_summed_derivatives(model, layers) as records:
            margin = next_token_margin(model, ids, positive_id, negative_id, device)
            value = float(margin.detach())
            margin.backward()
        if set(records) != set(layers):
            raise RuntimeError("not all selected MLP derivatives were observed")
    return value, records


def coordinate_finite_difference(model, ids, positive_id, negative_id, device, layer, neuron, epsilon):
    """Compact numerical derivative of an all-token coordinate addition."""
    import math
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    with frozen_eval(model), torch.no_grad():
        with mlp_addition(model, layer, neuron, epsilon):
            plus = float(next_token_margin(model, ids, positive_id, negative_id, device))
        with mlp_addition(model, layer, neuron, -epsilon):
            minus = float(next_token_margin(model, ids, positive_id, negative_id, device))
    return {"epsilon": epsilon, "plus_margin": plus, "minus_margin": minus,
            "finite_difference": (plus - minus) / (2 * epsilon)}
