"""All-position dense MLP additions and transient summed derivatives."""
from contextlib import contextmanager
import math

import torch

from llm_bias.core.inference.mlp import dense_down_projection


@contextmanager
def mlp_addition(model, layer: int, neuron: int, delta: float):
    """Add a native-unit scalar at every position, including cached decoding."""
    if not 0 <= layer < len(model.layers) or neuron < 0 or not math.isfinite(delta):
        raise ValueError("invalid coordinate or delta")
    module = dense_down_projection(model.layers[layer])

    def hook(_module, args):
        values = args[0]
        if values.ndim != 3 or neuron >= values.shape[-1]:
            raise ValueError("invalid MLP shape or neuron")
        if delta == 0:
            return None
        result = values.clone()
        result[..., neuron] += delta
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
