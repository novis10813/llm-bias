"""All-position dense MLP additions and transient summed derivatives."""
from collections.abc import Sequence
from contextlib import contextmanager
import math

import torch

from llm_bias.core.inference.mlp import dense_down_projection


def _selection_mask(selection, values):
    """Validate a local position selector without coercing coordinates."""
    length = values.shape[1]
    if torch.is_tensor(selection):
        if selection.dtype != torch.bool or selection.ndim != 1 or selection.numel() != length:
            raise ValueError("selector must return a sequence-length 1D bool tensor")
        return selection.to(device=values.device)
    if not isinstance(selection, Sequence) or isinstance(selection, (str, bytes)):
        raise ValueError("selector must return integer indices or a bool tensor")
    indices = list(selection)
    if any(type(index) is not int or not 0 <= index < length for index in indices):
        raise ValueError("invalid selector index")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate selector indices")
    mask = torch.zeros(length, dtype=torch.bool, device=values.device)
    if indices:
        mask[indices] = True
    return mask


@contextmanager
def mlp_addition(model, layer: int, neuron: int, delta: float, *, selector=None):
    """Add a native-unit scalar before the dense down projection.

    The legacy default edits all positions, including cached decoding. An
    optional selector supplies validated local indices or a boolean mask.
    """
    if (type(layer) is not int or type(neuron) is not int
            or not 0 <= layer < len(model.layers) or neuron < 0
            or isinstance(delta, bool) or not math.isfinite(delta)):
        raise ValueError("invalid coordinate or delta")
    if selector is not None and not callable(selector):
        raise ValueError("selector must be callable")
    module = dense_down_projection(model.layers[layer])

    def hook(_module, args):
        values = args[0]
        if values.ndim != 3 or neuron >= values.shape[-1]:
            raise ValueError("invalid MLP shape or neuron")
        mask = None if selector is None else _selection_mask(selector(values), values)
        if delta == 0 or (mask is not None and not mask.any()):
            return None
        result = values.clone()
        if mask is None:
            result[..., neuron] += delta
        else:
            result[:, mask, neuron] += delta
        return (result, *args[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def mlp_summed_derivatives(model, layers: list[int]):
    """Capture d(objective)/d(uniform coordinate offset), summed over tokens.

    Run one batch-one differentiable forward and backward inside this context.
    Detached activations are never retained. Results are transient signed vectors;
    callers must average trials within ticker, then tickers, BEFORE taking abs.
    This function does not disable parameter gradients or choose the objective.
    """
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("require unique nonempty layers")
    if any(not 0 <= layer < len(model.layers) for layer in layers):
        raise ValueError("layer out of range")
    records = {}
    handles = []
    try:
        for layer in layers:
            def hook(_module, args, layer=layer):
                values = args[0]
                if values.ndim != 3 or values.shape[0] != 1:
                    raise ValueError("require batch-one MLP input")
                if not torch.is_grad_enabled():
                    raise RuntimeError("gradient recording requires grad mode")
                # Frozen models still need a graph starting at the first hook.
                if not values.requires_grad:
                    values = values.detach().requires_grad_(True)

                def collect(gradient):
                    summed = gradient.detach().float().sum(dim=(0, 1)).cpu()
                    if not torch.isfinite(summed).all():
                        raise ValueError("non-finite MLP derivative")
                    records[layer] = records.get(layer, 0) + summed

                handles.append(values.register_hook(collect))
                return (values, *args[1:])
            handles.append(dense_down_projection(model.layers[layer]).register_forward_pre_hook(hook))
        yield records
    finally:
        for handle in handles:
            handle.remove()
