"""Memory-only four-arm execution; structural consistency is not certification.

config_hash is an unverified caller configuration identity. The caller is also
responsible for the absence of unrelated external perturbation hooks.
"""
from contextlib import ExitStack
from dataclasses import dataclass, fields
import json
import math

import torch

from ..artifact_paths import canonical_json_bytes
from ..stance_baseline_adapter import baseline_gate_input
from ..stance_gates import NoOpGateResult, evaluate_noop_gate, _require_hash
from .structured_output import (
    CompiledDecisionGrammar, StructuredGenerationResult, generate_structured,
    _check_capability, _validate_policy, _tokenizer_identity, _bind_model,
)
from .harmony_generation import (
    CompiledHarmonyDecisionGrammar, _check_harmony_capability, generate_harmony_structured,
)
from .stance_interventions import (
    GenerationPositionTracker, PerturbationDiagnostics, _prompt_selection,
    scoped_residual_intervention,
)
from .stance_transient_capture import capture_prompt_residual


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate diagnostic key')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('nonfinite diagnostic JSON')


def _diagnostics(raw, operation):
    if type(raw) is not bytes:
        raise ValueError('diagnostics must be canonical JSON bytes')
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid diagnostic JSON') from exc
    keys = {field.name for field in fields(PerturbationDiagnostics)}
    if type(value) is not dict or set(value) != keys or canonical_json_bytes(value) != raw:
        raise ValueError('diagnostics require the exact canonical scalar fields')
    if (type(value['layer']) is not int or value['layer'] < 0
            or value['hook_site'] not in ('pre', 'mid', 'post')
            or value['scope'] != 'prompt_only' or value['operation'] != operation):
        raise ValueError('invalid diagnostic coordinates or operation')
    for key in keys - {'layer', 'hook_site', 'scope', 'operation'}:
        number = value[key]
        if key.endswith('count') or key == 'selected_token_opportunities':
            if type(number) is not int or number < 0:
                raise ValueError('diagnostic counts must be nonnegative integers')
        else:
            try:
                valid = type(number) in (int, float) and math.isfinite(number) and number >= 0
            except OverflowError as exc:
                raise ValueError('diagnostic reductions must be finite nonnegative numbers') from exc
            if not valid:
                raise ValueError('diagnostic reductions must be finite nonnegative numbers')
    return value


@dataclass(frozen=True, slots=True)
class PromptNoOpExecution:
    baseline: StructuredGenerationResult
    repeat: StructuredGenerationResult | None = None
    zero: StructuredGenerationResult | None = None
    self_replacement: StructuredGenerationResult | None = None
    gate: NoOpGateResult | None = None
    halt_reason: str | None = None
    _zero_diagnostics: bytes | None = None
    _self_diagnostics: bytes | None = None

    def __post_init__(self):
        baseline = baseline_gate_input(self.baseline)
        later = (self.repeat, self.zero, self.self_replacement)
        inputs = [baseline]
        for arm in later:
            if arm is not None:
                inputs.append(baseline_gate_input(arm))
        if self.halt_reason is not None:
            if (self.halt_reason not in ('baseline_failed', 'source_unavailable')
                    or any(arm is not None for arm in later)
                    or self.gate is not None or self._zero_diagnostics is not None
                    or self._self_diagnostics is not None
                    or (self.halt_reason == 'baseline_failed') != (baseline.failure_type is not None)):
                raise ValueError('contradictory halted execution')
            return
        if baseline.failure_type is not None or any(arm is None for arm in later) or not isinstance(self.gate, NoOpGateResult):
            raise ValueError('completed execution requires successful baseline and all arms/gate')
        expected = evaluate_noop_gate(*inputs, config_hash=self.gate.config_hash)
        if expected.to_dict() != self.gate.to_dict():
            raise ValueError('gate does not match complete generation results')
        zero = _diagnostics(self._zero_diagnostics, 'addition')
        replacement = _diagnostics(self._self_diagnostics, 'replacement')
        if (zero['layer'], zero['hook_site']) != (replacement['layer'], replacement['hook_site']):
            raise ValueError('diagnostic sites differ')

    @property
    def completed(self):
        return self.halt_reason is None and all(value is not None for value in (
            self.baseline, self.repeat, self.zero, self.self_replacement, self.gate))

    @property
    def passed(self):
        return self.completed and self.gate.passed

    @property
    def zero_diagnostics(self):
        return None if self._zero_diagnostics is None else _diagnostics(self._zero_diagnostics, 'addition')

    @property
    def self_diagnostics(self):
        return None if self._self_diagnostics is None else _diagnostics(self._self_diagnostics, 'replacement')


def execute_prompt_noop(model, tokenizer, prompt_ids, capability, *, policy, layer,
                        hook_site, zero_vector, prompt_positions=None, config_hash,
                        verified_binding=None):
    """Execute only genuine attempts, using a fresh tracker per HF generation."""
    _require_hash(config_hash, 'config_hash')
    if type(layer) is not int or not 0 <= layer < len(model.layers):
        raise ValueError('invalid layer coordinate')
    if hook_site not in ('pre', 'mid', 'post'):
        raise ValueError('unsupported hook site')
    if isinstance(capability, CompiledHarmonyDecisionGrammar):
        _check_harmony_capability(capability)
        nested = capability.json_capability
        driver = generate_harmony_structured
        channel = 'harmony_no_tools'
    elif isinstance(capability, CompiledDecisionGrammar):
        _check_capability(capability)
        nested = capability
        driver = generate_structured
        channel = 'plain_json'
    else:
        raise ValueError('require a factory grammar capability')
    _validate_policy(policy, nested.head_vocab_size)
    if policy.channel_policy != channel:
        raise ValueError('policy does not match explicit capability route')
    if (not torch.is_tensor(prompt_ids) or prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1
            or prompt_ids.shape[1] == 0 or prompt_ids.dtype not in (torch.int32, torch.int64)
            or (prompt_ids < 0).any() or (prompt_ids >= nested.head_vocab_size).any()):
        raise ValueError('require batch-one integer prompt inside head range')
    if channel == 'harmony_no_tools':
        suffix = capability.contract.prompt_suffix_ids
        if tuple(prompt_ids[0, -len(suffix):].tolist()) != suffix:
            raise ValueError('prompt must end in bound Harmony suffix')
    if _tokenizer_identity(tokenizer) != nested.tokenizer_sha256:
        raise ValueError('tokenizer identity differs from capability')
    _bind_model(model, nested, verified_binding)
    positions = _prompt_selection(prompt_ids.shape[1], 'prompt_only', prompt_positions)
    if not positions:
        raise ValueError('prompt selection cannot be empty')
    if (not torch.is_tensor(zero_vector) or not zero_vector.is_floating_point()
            or zero_vector.ndim != 1 or zero_vector.numel() == 0
            or not torch.isfinite(zero_vector).all() or not (zero_vector != 0).any()):
        raise ValueError('require finite nonzero floating direction')
    snapshot = zero_vector.detach().clone()

    def generate():
        kwargs = dict(policy=policy, verified_binding=verified_binding)
        if driver is generate_structured:
            kwargs['transforms'] = None
        result = driver(model, tokenizer, prompt_ids, capability, **kwargs)
        baseline_gate_input(result)
        return result

    def tracker():
        return GenerationPositionTracker(prompt_ids.shape[1], policy.use_cache)

    source = None
    try:
        with ExitStack() as lifetime:
            with ExitStack() as tracking:
                original = tracking.enter_context(tracker().track(model))
                holder = lifetime.enter_context(capture_prompt_residual(
                    model, tracker=original, layer=layer, hook_site=hook_site,
                    prompt_positions=positions))
                baseline = generate()
            if baseline.failure_type is not None:
                return PromptNoOpExecution(baseline, halt_reason='baseline_failed')
            try:
                source = holder.require_source()
            except ValueError:
                return PromptNoOpExecution(baseline, halt_reason='source_unavailable')
            if snapshot.shape[0] != source.shape[-1]:
                raise ValueError('direction dimension does not match captured residual')
            cast = snapshot.to(device=source.device, dtype=source.dtype)
            if not torch.isfinite(cast).all() or not (cast != 0).any():
                raise ValueError('direction is nonfinite or zero after casting')
            del cast
            with tracker().track(model):
                repeat = generate()
            zero_tracker = tracker()
            with zero_tracker.track(model), scoped_residual_intervention(
                model, tracker=zero_tracker, layer=layer, hook_site=hook_site,
                scope='prompt_only', operation='addition', vector=snapshot,
                dose=0.0, prompt_positions=positions,
            ) as zero_diag:
                zero = generate()
            self_tracker = tracker()
            with self_tracker.track(model), scoped_residual_intervention(
                model, tracker=self_tracker, layer=layer, hook_site=hook_site,
                scope='prompt_only', operation='replacement', source=source,
                source_positions=holder.positions, prompt_positions=holder.positions, dose=1,
            ) as self_diag:
                replacement = generate()
            gate = evaluate_noop_gate(*(baseline_gate_input(arm) for arm in (
                baseline, repeat, zero, replacement)), config_hash=config_hash)
            return PromptNoOpExecution(baseline, repeat, zero, replacement, gate,
                _zero_diagnostics=canonical_json_bytes(zero_diag.to_dict()),
                _self_diagnostics=canonical_json_bytes(self_diag.to_dict()))
    finally:
        source = None
        snapshot = None
