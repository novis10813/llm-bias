"""Memory-only observation of one baseline prompt forward.

Source availability is not baseline success or evidence that external hooks
are absent. The caller validates generation separately and owns replay policy.
"""
from contextlib import contextmanager

import torch

from .interventions import _block_hidden, _post_attention_norm
from .stance_interventions import GenerationPositionTracker, _indices, _root


class TransientPromptCapture:
    """A scoped source reference, never a persistence or eligibility API."""

    def __init__(self, *, layer, hook_site, prompt_length, positions):
        self._layer = layer
        self._hook_site = hook_site
        self._prompt_length = prompt_length
        self._positions = positions
        self._source = None
        self._state = 'unobserved'
        self._in_forward = False
        self._first_finished = False
        self._observations = 0

    @property
    def layer(self):
        return self._layer

    @property
    def hook_site(self):
        return self._hook_site

    @property
    def prompt_length(self):
        return self._prompt_length

    @property
    def positions(self):
        return self._positions

    @property
    def ready(self):
        return self._state == 'captured' and not self._in_forward

    def require_source(self) -> torch.Tensor:
        """Return the live source; callers must also check baseline generation."""
        if not self.ready:
            raise ValueError('prompt capture is incomplete, invalid, released or in-forward')
        return self._source

    def _invalidate(self):
        self._state = 'invalid'
        self._source = None


@contextmanager
def capture_prompt_residual(model, *, tracker: GenerationPositionTracker,
                            layer: int, hook_site: str, prompt_positions=None):
    """Observe exactly one full initial prompt without replacing any output."""
    root = _root(model)
    if (not isinstance(tracker, GenerationPositionTracker) or not tracker._tracking
            or tracker._root is not root):
        raise ValueError('capture requires an active tracker on the same root')
    layers = getattr(model, 'layers', None)
    if layers is None:
        raise ValueError('model does not expose decoder layers')
    if type(layer) is not int or not 0 <= layer < len(layers):
        raise ValueError('invalid layer coordinate')
    if hook_site not in ('pre', 'mid', 'post'):
        raise ValueError('invalid residual hook site')
    positions = (tuple(range(tracker.prompt_length)) if prompt_positions is None
                 else _indices(prompt_positions, upper=tracker.prompt_length))
    if not positions:
        raise ValueError('capture requires nonempty prompt positions')
    target = _post_attention_norm(model, layer) if hook_site == 'mid' else layers[layer]
    registration = 'register_forward_hook' if hook_site == 'post' else 'register_forward_pre_hook'
    if not callable(getattr(target, registration, None)):
        raise ValueError('unsupported capture hook target')
    holder = TransientPromptCapture(layer=layer, hook_site=hook_site,
                                    prompt_length=tracker.prompt_length, positions=positions)
    handles = []

    def active():
        # Capture lifetime can outlast the original tracker for fresh replay.
        return tracker._tracking and tracker._root is root

    def root_pre(_module, args, kwargs):
        if not active():
            return None
        if holder._state == 'invalid' or holder._in_forward:
            holder._invalidate()
            raise ValueError('invalid or recursive capture forward')
        holder._in_forward = True
        if not holder._first_finished and tracker.active_positions != tuple(range(holder.prompt_length)):
            holder._invalidate()
            raise ValueError('first capture forward must contain the complete ordered prompt')
        return None

    def root_post(_module, args, kwargs, output):
        if not active():
            return None
        # This also executes when the earlier tracker prehook failed, before
        # root_pre could set our flag. Tracker post has already cleared positions.
        try:
            if not holder._first_finished:
                holder._first_finished = True
                if (not holder._in_forward or output is None
                        or holder._observations != 1 or holder._state != 'captured'):
                    holder._invalidate()
                    raise ValueError('initial prompt capture forward did not complete exactly once')
        finally:
            holder._in_forward = False
        return None

    def observe(extract):
        if not active():
            return None
        try:
            if holder._state == 'invalid' or not holder._in_forward:
                raise ValueError('observation requires a valid tracked root forward')
            values = extract()
            tracker.select(values, scope='prompt_only', prompt_positions=positions)
            if (not values.is_floating_point() or values.shape[-1] <= 0
                    or not torch.isfinite(values).all()):
                raise ValueError('capture requires finite floating hidden values with positive width')
            if holder._first_finished:
                return None
            holder._observations += 1
            if holder._observations != 1:
                raise ValueError('duplicate initial prompt observation')
            holder._source = values[:, positions, :].detach().clone()
            holder._state = 'captured'
        except Exception:
            holder._invalidate()
            raise
        return None

    def site_pre(_module, args, kwargs):
        return observe(lambda: _block_hidden(args, kwargs))

    def site_post(_module, args, output):
        def extract():
            if torch.is_tensor(output):
                return output
            if isinstance(output, (tuple, list)) and output:
                return output[0]
            raise ValueError('unsupported decoder block output')
        return observe(extract)

    try:
        handles.append(root.register_forward_pre_hook(root_pre, with_kwargs=True))
        handles.append(root.register_forward_hook(root_post, with_kwargs=True, always_call=True))
        if hook_site == 'post':
            handles.append(target.register_forward_hook(site_post))
        else:
            handles.append(target.register_forward_pre_hook(site_pre, with_kwargs=True))
        yield holder
    finally:
        for handle in handles:
            handle.remove()
        holder._source = None
        holder._state = 'released'
        holder._in_forward = False
