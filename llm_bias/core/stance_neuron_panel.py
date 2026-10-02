"""Independent native dense-neuron candidate records, not discovery outcomes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import re

from .artifact_paths import sha256_json
from .inference.mlp import dense_down_projection
from .inference.stance_interventions import scoped_mlp_addition

SEED = 20261003
PER_LAYER = 16
NATIVE_DOSES = (-8, -2, -0.5, 0, 0.5, 2, 8)
SCOPES = ('prompt_only', 'prompt_and_decode')
_ROLES = ('fit', 'validation', 'calibration', 'evaluation')
_PHASE_ROLE = dict(discovery='fit', selection='validation', calibration='calibration', evaluation='evaluation')


def _ranking(layer, width):
    return tuple(sorted(range(width), key=lambda neuron: (sha256_json(
        {'seed': SEED, 'layer': layer, 'neuron': neuron}), neuron))[:min(width, PER_LAYER)])


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected lowercase SHA-256')


@dataclass(frozen=True, slots=True)
class NeuronPanel:
    status: str
    widths: tuple[int, ...]
    coordinates: tuple[tuple[int, ...], ...]
    reason: str = ''

    def __post_init__(self):
        if type(self.widths) is not tuple or type(self.coordinates) is not tuple:
            raise ValueError('widths and coordinates must be immutable tuples')
        if self.status == 'unsupported':
            if self.widths or self.coordinates or type(self.reason) is not str or not self.reason.strip():
                raise ValueError('unsupported panel requires reason and no candidates')
            return
        if self.status != 'supported' or self.reason != '' or not self.widths:
            raise ValueError('invalid panel status or empty layer coverage')
        if any(type(width) is not int or width <= 0 for width in self.widths):
            raise ValueError('native widths must be genuine positive integers')
        if len(self.coordinates) != len(self.widths):
            raise ValueError('candidate panel must cover all layers')
        for layer, (width, coordinates) in enumerate(zip(self.widths, self.coordinates)):
            if (type(coordinates) is not tuple or any(type(n) is not int for n in coordinates)
                    or coordinates != _ranking(layer, width)):
                raise ValueError('coordinates must equal the independent sorted SHA ranking')

    @property
    def coverage(self):
        """(layer, candidate count, native width), including width<16 caps."""
        return tuple((layer, len(coords), self.widths[layer])
                     for layer, coords in enumerate(self.coordinates))

    def to_dict(self):
        return dict(kind='native_neuron_panel_v1', status=self.status, reason=self.reason,
                    seed=SEED, per_layer=PER_LAYER, widths=list(self.widths),
                    coordinates=[list(coords) for coords in self.coordinates],
                    hook_site='mlp_down_proj_input', ranking='sha256_json_seed_layer_neuron_ascending')

    @property
    def panel_sha256(self):
        return sha256_json(self.to_dict())


def build_neuron_panel(model) -> NeuronPanel:
    """Inspect native down_proj input widths, never residual width or DIM weights.

    Unsupported architecture yields an empty panel, not a partial layer subset.
    Widths come from down_proj.in_features. Unknown projection layouts require
    a future explicit capability adapter rather than a guessed residual width.
    """
    configs = [getattr(owner, 'config', None) for owner in
               (model, getattr(model, 'hf_model', None), getattr(model, '_hf_model', None))]
    if any(getattr(config, 'model_type', None) == 'gpt_oss' for config in configs):
        return NeuronPanel('unsupported', (), (), 'GPT routed MoE native-neuron hook unsupported')
    layers = getattr(model, 'layers', None)
    if layers is None or not len(layers):
        raise ValueError('model requires nonempty ordered layers')
    widths = []
    for layer, block in enumerate(layers):
        owner = getattr(block, '_hf_layer', block)
        mlp = getattr(owner, 'mlp', None)
        if any(getattr(mlp, name, None) is not None for name in ('experts', 'router', 'gate')):
            return NeuronPanel('unsupported', (), (), f'layer {layer}: routed MoE native-neuron hook unsupported')
        try:
            projection = dense_down_projection(block)
        except TypeError:
            return NeuronPanel('unsupported', (), (), f'layer {layer}: no dense native down_proj input')
        width = getattr(projection, 'in_features', None)
        if type(width) is not int or width <= 0:
            return NeuronPanel('unsupported', (), (), f'layer {layer}: native input width unavailable')
        widths.append(width)
    return NeuronPanel('supported', tuple(widths), tuple(_ranking(layer, width)
                                                       for layer, width in enumerate(widths)))


@dataclass(frozen=True, slots=True)
class NeuronRole:
    ticker: str
    issuer_id: str
    role: str

    def __post_init__(self):
        if any(type(value) is not str or not value.strip() for value in (self.ticker, self.issuer_id)):
            raise ValueError('ticker and issuer require nonblank scalar strings')
        if self.role not in _ROLES:
            raise ValueError('unknown role')


@dataclass(frozen=True, slots=True)
class NeuronArm:
    panel_sha256: str
    layer: int
    neuron: int
    delta: float
    scope: str

    def __post_init__(self):
        _digest(self.panel_sha256)
        if any(type(value) is not int or value < 0 for value in (self.layer, self.neuron)):
            raise ValueError('coordinates require genuine nonnegative integers')
        if type(self.delta) not in (int, float) or self.delta not in NATIVE_DOSES or self.scope not in SCOPES:
            raise ValueError('require declared signed native dose and named scope')
        # Canonicalize 0.0/0, etc., so equivalent arms have identical hashes.
        object.__setattr__(self, 'delta', next(d for d in NATIVE_DOSES if d == self.delta))

    def to_dict(self):
        return asdict(self)

    @property
    def arm_sha256(self):
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class NeuronDiscoveryPlan:
    panel: NeuronPanel
    roles: tuple[NeuronRole, ...]
    inputs_manifest_sha256: str
    parent_sha256: str
    model_sha256: str
    protocol_sha256: str

    def __post_init__(self):
        if type(self.panel) is not NeuronPanel:
            raise ValueError('expected immutable NeuronPanel')
        for value in (self.inputs_manifest_sha256, self.parent_sha256, self.model_sha256, self.protocol_sha256):
            _digest(value)
        if type(self.roles) is not tuple or not self.roles or any(type(r) is not NeuronRole for r in self.roles):
            raise ValueError('expected immutable role records')
        if {r.role for r in self.roles} != set(_ROLES) or len({r.ticker for r in self.roles}) != len(self.roles):
            raise ValueError('require four separate nonempty roles and unique tickers')
        issuer_roles = {}
        for row in self.roles:
            if issuer_roles.setdefault(row.issuer_id, row.role) != row.role:
                raise ValueError('issuer crosses roles')
        object.__setattr__(self, 'roles', tuple(sorted(self.roles, key=lambda r: r.ticker)))

    def role_ids(self, phase):
        if phase not in _PHASE_ROLE:
            raise ValueError('unknown phase')
        return tuple(row.ticker for row in self.roles if row.role == _PHASE_ROLE[phase])

    def arms(self):
        """Fixed Cartesian candidate/dose/scope budget, no outcomes or ranking."""
        for layer, coordinates in enumerate(self.panel.coordinates):
            for neuron in coordinates:
                for scope in SCOPES:
                    for delta in NATIVE_DOSES:
                        yield NeuronArm(self.panel.panel_sha256, layer, neuron, delta, scope)

    def to_dict(self):
        return dict(kind='native_neuron_discovery_plan_v1', panel=self.panel.to_dict(),
                    roles=[asdict(row) for row in self.roles], doses=list(NATIVE_DOSES), scopes=list(SCOPES),
                    inputs_manifest_sha256=self.inputs_manifest_sha256, parent_sha256=self.parent_sha256,
                    model_sha256=self.model_sha256, protocol_sha256=self.protocol_sha256)

    @property
    def plan_sha256(self):
        return sha256_json(self.to_dict())


@contextmanager
def native_neuron_edit(model, *, panel, arm, tracker):
    """Apply one declared arm via the accepted actual activation-edit context."""
    if type(panel) is not NeuronPanel or type(arm) is not NeuronArm:
        raise ValueError('expected immutable panel and arm records')
    if (panel.status != 'supported' or build_neuron_panel(model) != panel
            or arm.panel_sha256 != panel.panel_sha256 or arm.layer >= len(panel.widths)
            or arm.neuron not in panel.coordinates[arm.layer]):
        raise ValueError('arm/model not bound to supported native panel')
    with scoped_mlp_addition(model, tracker=tracker, layer=arm.layer, neuron=arm.neuron,
                             delta=arm.delta, scope=arm.scope) as metadata:
        yield metadata
