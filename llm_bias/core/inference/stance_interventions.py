"""Absolute-position scopes for transient residual and native MLP edits.

Only explicit positions or validated cache lengths locate cached inputs. No
sequence-length decode heuristic, guessed counter, or activation persistence.
"""
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch

from .interventions import (
    mid_residual_interventions,
    pre_residual_interventions,
    residual_interventions,
)
from .mlp_addition import mlp_addition


class UnsupportedPositionMetadata(ValueError):
    """The root input cannot be mapped reliably to absolute token positions."""


def _root(model):
    root = getattr(model, 'hf_model', None)
    if root is None:
        root = getattr(model, '_hf_model', None)
    if root is None and isinstance(model, torch.nn.Module):
        root = model
    if root is None or not all(callable(getattr(root, name, None)) for name in (
        'register_forward_pre_hook', 'register_forward_hook'
    )):
        raise ValueError('model has no hookable HF root')
    return root


def _indices(positions, *, upper=None):
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        raise ValueError('positions must be a unique Python integer sequence')
    result = tuple(positions)
    if any(type(position) is not int or position < 0
           or (upper is not None and position >= upper) for position in result):
        raise ValueError('invalid absolute position')
    if len(set(result)) != len(result):
        raise ValueError('duplicate absolute positions')
    return result


def _prompt_selection(prompt_length, scope, prompt_positions):
    if scope not in ('prompt_only', 'decode_only', 'prompt_and_decode'):
        raise ValueError('invalid intervention scope')
    if prompt_positions is None:
        return () if scope == 'decode_only' else tuple(range(prompt_length))
    positions = _indices(prompt_positions, upper=prompt_length)
    if scope == 'decode_only' and positions:
        raise ValueError('decode_only cannot select prompt positions')
    return positions


def _explicit_positions(value, *, length, name):
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.uint16, torch.uint32, torch.uint64)
    shape = (length,) if name == 'cache_position' else (1, length)
    if not torch.is_tensor(value) or value.dtype not in integer_dtypes or tuple(value.shape) != shape:
        raise UnsupportedPositionMetadata(f'{name} must be an integer tensor of shape {shape}')
    positions = tuple(value.detach().reshape(-1).tolist())
    if any(position < 0 for position in positions) or len(set(positions)) != length:
        raise UnsupportedPositionMetadata(f'{name} must contain unique nonnegative positions')
    return positions


def _cache_length(cache):
    if cache is None or (isinstance(cache, tuple) and not cache):
        return 0
    getter = getattr(cache, 'get_seq_length', None)
    if callable(getter):
        try:
            length = getter()
        except Exception as exc:
            raise UnsupportedPositionMetadata('cache length lookup failed') from exc
        if type(length) is not int or length < 0:
            raise UnsupportedPositionMetadata('cache length must be a nonnegative integer')
        return length
    if not isinstance(cache, (tuple, list)) or not cache:
        raise UnsupportedPositionMetadata('unsupported cache layout')
    lengths = []
    for pair in cache:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise UnsupportedPositionMetadata('legacy cache must contain key/value pairs')
        if any(not torch.is_tensor(tensor) or tensor.ndim != 4 or tensor.shape[0] != 1
               for tensor in pair):
            raise UnsupportedPositionMetadata('legacy cache requires batch-one rank4 key/value tensors')
        if pair[0].shape != pair[1].shape:
            raise UnsupportedPositionMetadata('legacy key/value shapes disagree')
        lengths.append(pair[0].shape[2])
    if len(set(lengths)) != 1:
        raise UnsupportedPositionMetadata('legacy cache layer lengths disagree')
    return lengths[0]


class GenerationPositionTracker:
    """One generation's root-forward absolute positions, never guessed."""

    def __init__(self, prompt_length: int, use_cache: bool):
        if type(prompt_length) is not int or prompt_length <= 0 or type(use_cache) is not bool:
            raise ValueError('require positive integer prompt length and boolean cache mode')
        self.prompt_length = prompt_length
        self.use_cache = use_cache
        self.active_positions = None
        self._used = False
        self._tracking = False
        self._root = None

    def _positions(self, args, kwargs):
        if args and kwargs.get('input_ids') is not None:
            raise UnsupportedPositionMetadata('input_ids supplied twice')
        input_ids = args[0] if args else kwargs.get('input_ids')
        embeds = kwargs.get('inputs_embeds')
        if input_ids is not None and embeds is not None:
            raise UnsupportedPositionMetadata('input_ids and inputs_embeds are mutually exclusive')
        values = input_ids if input_ids is not None else embeds
        ndim = 2 if input_ids is not None else 3
        if (not torch.is_tensor(values) or values.ndim != ndim
                or values.shape[0] != 1 or values.shape[1] <= 0):
            raise UnsupportedPositionMetadata('require a nonempty batch-one root input')
        length = values.shape[1]
        explicit = []
        for name in ('cache_position', 'position_ids'):
            if kwargs.get(name) is not None:
                explicit.append(_explicit_positions(kwargs[name], length=length, name=name))
        if explicit:
            if any(positions != explicit[0] for positions in explicit[1:]):
                raise UnsupportedPositionMetadata('cache_position and position_ids disagree')
            return explicit[0]
        if not self.use_cache:
            if length < self.prompt_length:
                raise UnsupportedPositionMetadata('implicit no-cache input must contain the full prompt')
            return tuple(range(length))
        offset = _cache_length(kwargs.get('past_key_values'))
        if offset == 0 and length != self.prompt_length:
            raise UnsupportedPositionMetadata('implicit empty-cache input must be the full initial prompt')
        return tuple(range(offset, offset + length))

    @contextmanager
    def track(self, model):
        """Install a kwargs-aware root prehook and an always-call clearing hook."""
        if self._used or self._tracking:
            raise ValueError('a tracker cannot be nested or reused across generations')
        self._used = True
        handles = []
        try:
            self._root = _root(model)
            self._tracking = True

            def prehook(_module, args, kwargs):
                if self.active_positions is not None:
                    raise UnsupportedPositionMetadata('recursive root forwards are unsupported')
                self.active_positions = self._positions(args, kwargs)

            def posthook(_module, args, kwargs, output):
                self.active_positions = None

            handles.append(self._root.register_forward_pre_hook(prehook, with_kwargs=True))
            handles.append(self._root.register_forward_hook(posthook, with_kwargs=True, always_call=True))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.active_positions = None
            self._tracking = False
            self._root = None

    def select(self, values, *, scope, prompt_positions=None):
        """Select local tensor tokens by their active absolute positions."""
        if self.active_positions is None:
            raise ValueError('selection requires an active tracked root forward')
        if (not torch.is_tensor(values) or values.ndim != 3 or values.shape[0] != 1
                or values.shape[1] != len(self.active_positions)):
            raise ValueError('hidden sequence shape does not match the active root input')
        prompts = set(_prompt_selection(self.prompt_length, scope, prompt_positions))
        return torch.tensor([
            (scope != 'decode_only' and position in prompts)
            or (scope != 'prompt_only' and position >= self.prompt_length)
            for position in self.active_positions
        ], dtype=torch.bool, device=values.device)


def _consumer(model, tracker, layer, scope, prompt_positions):
    if not isinstance(tracker, GenerationPositionTracker) or not tracker._tracking:
        raise ValueError('nest scoped interventions inside tracker.track(model)')
    if _root(model) is not tracker._root:
        raise ValueError('intervention model differs from the tracked root')
    if type(layer) is not int or not 0 <= layer < len(model.layers):
        raise ValueError('invalid layer coordinate')
    return _prompt_selection(tracker.prompt_length, scope, prompt_positions)


@dataclass
class PerturbationDiagnostics:
    """Scalar reductions only; no source, masks, or hidden arrays are retained."""

    layer: int
    hook_site: str
    scope: str
    operation: str
    selected_token_opportunities: int = 0
    changed_token_count: int = 0
    finite_count: int = 0
    delta_l2_sum: float = 0.0
    delta_l2_max: float = 0.0
    relative_delta_l2_sum: float = 0.0
    relative_delta_l2_max: float = 0.0
    relative_finite_count: int = 0
    residual_zero_count: int = 0

    def to_dict(self):
        return asdict(self)

    def _record_identity(self, values):
        # Zero dose and exact self replacement need no subtraction or casts.
        finite = torch.isfinite(values).all(dim=-1)
        zero = (values == 0).all(dim=-1)
        self.selected_token_opportunities += finite.numel()
        self.finite_count += int(finite.sum().item())
        self.residual_zero_count += int(zero.sum().item())
        self.relative_finite_count += int((finite & ~zero).sum().item())

    def _record(self, before, after):
        # Detached float64 temporaries reduce norm overflow and never escape.
        before = before.detach().double()
        after = after.detach().double()
        delta = after - before
        norms = torch.linalg.vector_norm(delta, dim=-1)
        residual_norms = torch.linalg.vector_norm(before, dim=-1)
        finite = torch.isfinite(norms)
        self.selected_token_opportunities += norms.numel()
        self.changed_token_count += int((before != after).any(dim=-1).sum().item())
        self.finite_count += int(finite.sum().item())
        if finite.any():
            self.delta_l2_sum += norms[finite].sum().item()
            self.delta_l2_max = max(self.delta_l2_max, norms[finite].max().item())
        self.residual_zero_count += int((residual_norms == 0).sum().item())
        valid = finite & torch.isfinite(residual_norms) & (residual_norms > 0)
        relative = norms[valid] / residual_norms[valid]
        relative = relative[torch.isfinite(relative)]
        self.relative_finite_count += relative.numel()
        if relative.numel():
            self.relative_delta_l2_sum += relative.sum().item()
            self.relative_delta_l2_max = max(self.relative_delta_l2_max, relative.max().item())


def _dose(dose):
    if isinstance(dose, bool) or not isinstance(dose, Real) or not math.isfinite(dose):
        raise ValueError('dose must be a finite real nonbool scalar')


def _vector(vector):
    if (not torch.is_tensor(vector) or vector.ndim != 1 or vector.is_complex()
            or vector.dtype == torch.bool or not torch.isfinite(vector).all()
            or not (vector != 0).any()):
        raise ValueError('require a finite nonzero real direction vector')


@contextmanager
def scoped_residual_intervention(
    model, *, tracker, layer, hook_site, scope, operation, vector=None, dose=1.0,
    prompt_positions=None, source=None, source_positions=None
):
    """Apply an exactly aligned, absolute-position residual transformation."""
    prompts = _consumer(model, tracker, layer, scope, prompt_positions)
    contexts = {'pre': pre_residual_interventions, 'mid': mid_residual_interventions,
                'post': residual_interventions}
    if hook_site not in contexts or operation not in ('addition', 'ablation', 'replacement'):
        raise ValueError('invalid residual hook site or operation')
    _dose(dose)
    mapping = None
    if operation == 'replacement':
        if scope != 'prompt_only' or vector is not None or dose != 1:
            raise ValueError('replacement requires prompt_only, no vector and unit dose')
        if (not torch.is_tensor(source) or source.ndim != 3 or source.shape[0] != 1
                or not source.is_floating_point() or not torch.isfinite(source).all()):
            raise ValueError('replacement source must be a finite [1,n_source,d_model] tensor')
        positions = _indices(source_positions, upper=tracker.prompt_length)
        if len(positions) != source.shape[1] or set(positions) != set(prompts):
            raise ValueError('source positions must map exactly to the selected prompt positions')
        mapping = {position: index for index, position in enumerate(positions)}
    else:
        _vector(vector)
        if source is not None or source_positions is not None:
            raise ValueError('source is only valid for replacement')
        if operation == 'ablation' and dose != 1:
            raise ValueError('ablation requires unit dose')
    diagnostics = PerturbationDiagnostics(layer, hook_site, scope, operation)

    def transform(values):
        mask = tracker.select(values, scope=scope, prompt_positions=prompts)
        if not values.is_floating_point():
            raise ValueError('residual must be floating point')
        if operation == 'replacement':
            if source.shape[-1] != values.shape[-1] or source.dtype != values.dtype or source.device != values.device:
                raise ValueError('replacement source must match hidden dimension, dtype and device')
        else:
            if vector.shape[0] != values.shape[-1]:
                raise ValueError('direction dimension does not match residual')
            direction = vector.to(device=values.device, dtype=values.dtype)
            if not torch.isfinite(direction).all() or not (direction != 0).any():
                raise ValueError('direction is nonfinite or zero after casting')
        if not mask.any():
            return values
        selected = values[:, mask, :]
        if operation == 'addition':
            if dose == 0:
                diagnostics._record_identity(selected)
                return values
            edited = selected + dose * direction
        elif operation == 'ablation':
            # Projection is scale-invariant. Bound the direction before its
            # squared norm so even finite float64 extremes cannot overflow or
            # underflow the denominator.
            u = direction.double()
            u = u / u.abs().max()
            x = selected.double()
            edited = (x - ((x * u).sum(dim=-1, keepdim=True) / (u * u).sum()) * u).to(values.dtype)
        else:
            indices = [mapping[position] for position, keep in zip(tracker.active_positions, mask.tolist()) if keep]
            edited = source[:, indices, :]
        if torch.equal(selected, edited):
            diagnostics._record_identity(selected)
            return values
        diagnostics._record(selected, edited)
        result = values.clone()
        result[:, mask, :] = edited
        return result

    with contexts[hook_site](model, {layer: transform}):
        yield diagnostics


@contextmanager
def scoped_mlp_addition(model, *, tracker, layer, neuron, delta, scope, prompt_positions=None):
    """Select absolute tokens for the actual dense down-projection input hook."""
    prompts = _consumer(model, tracker, layer, scope, prompt_positions)
    _dose(delta)
    if type(neuron) is not int or neuron < 0:
        raise ValueError('invalid native neuron coordinate')
    metadata = {'layer': layer, 'neuron': neuron, 'hook_site': 'mlp_down_proj_input',
                'scope': scope, 'operation': 'addition'}
    with mlp_addition(model, layer, neuron, delta,
                      selector=lambda values: tracker.select(values, scope=scope, prompt_positions=prompts)):
        yield metadata
