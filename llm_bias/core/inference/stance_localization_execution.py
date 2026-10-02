"""Genuine donor-to-target cross-prompt replacement, executed in memory only.

One explicitly aligned cross-prompt replacement is executed with the accepted
schema drivers, transient capture and the unchanged scoped replacement hook.
Parent matching detects clean-generation drift; it does not certify model
authenticity, replace phase-level no-op gates, or make a result self-
certifying. Expected outputs bind recorded configuration to the nested
capability; validation is integrity only and never establishes research
eligibility. No residual, source or gathered tensor escapes this module.
"""
from contextlib import ExitStack
from dataclasses import asdict, dataclass

import torch

from ..artifact_paths import canonical_json_bytes, sha256_json
from ..prompt_input.decision_prompt import DecisionPrompt
from ..stance_baseline_adapter import baseline_gate_input
from ..stance_localization_alignment import SpanAlignment, build_span_alignment
from .harmony_generation import (
    CompiledHarmonyDecisionGrammar, _check_harmony_capability, generate_harmony_structured,
)
from .stance_interventions import GenerationPositionTracker, scoped_residual_intervention
from .stance_noop_execution import _diagnostics
from .stance_transient_capture import capture_prompt_residual
from .structured_output import (
    CompiledDecisionGrammar, StructuredGenerationPolicy, StructuredGenerationResult,
    _bind_model, _check_capability, _hf_controls, _tokenizer_identity, _validate_policy,
    generate_structured,
)

__all__ = ['PromptReplacementExecution', 'ReplacementMatchChecks',
           'execute_prompt_replacement']

_STATUSES = ('donor_failed', 'donor_mismatch', 'source_unavailable',
             'target_failed', 'target_mismatch', 'executed')


def _same_full_output(actual, expected):
    """Complete-output equality; elapsed seconds and provenance are excluded."""
    return (
        actual.generated_token_ids == expected.generated_token_ids
        and actual.generated_text == expected.generated_text
        and actual.json_payload == expected.json_payload
        and actual.decision == expected.decision
        and actual.reason == expected.reason
        and (actual.decision_complete, actual.schema_complete, actual.reason_valid)
        == (expected.decision_complete, expected.schema_complete, expected.reason_valid)
        and actual.finish_reason == expected.finish_reason
        and actual.failure_type == expected.failure_type
        and actual.error_message == expected.error_message
        and actual.decode_error == expected.decode_error
    )


def _canonically_equal(left, right):
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _input_embedding_weight(model):
    getter = getattr(model.hf_model, 'get_input_embeddings', None)
    if not callable(getter):
        raise ValueError('model.hf_model must expose a callable get_input_embeddings')
    weight = getattr(getter(), 'weight', None)
    if (not torch.is_tensor(weight) or weight.ndim != 2
            or not weight.is_floating_point()
            or weight.shape[0] <= 0 or weight.shape[1] <= 0
            or not torch.isfinite(weight).all()):
        raise ValueError('input embedding weight must be a finite floating two-dimensional tensor')
    return weight


def _prompt_tensor(prompt, device, upper, label):
    ids = prompt.inference_token_ids
    if not isinstance(ids, tuple) or not ids:
        raise ValueError(f'{label} prompt must contain a nonempty token tuple')
    if any(type(i) is not int or not 0 <= i < upper for i in ids):
        raise ValueError(f'{label} prompt IDs must lie inside the embedding rows and head range')
    return torch.tensor(ids, dtype=torch.int64, device=device).unsqueeze(0)


def _bind_prompt_records(donor_prompt, target_prompt, nested):
    for label, prompt in (('donor', donor_prompt), ('target', target_prompt)):
        if not isinstance(prompt, DecisionPrompt):
            raise ValueError(f'{label} prompt must be a DecisionPrompt record')
        if prompt.schema_sha256 != nested.schema_sha256:
            raise ValueError(f'{label} prompt schema hash differs from the capability schema')
    if (donor_prompt.template_sha256 != target_prompt.template_sha256
            or donor_prompt.wrapper_policy != target_prompt.wrapper_policy):
        raise ValueError('donor and target prompt wrapper records and template hashes must match')


def _bind_alignment(donor_prompt, target_prompt, alignment):
    if not isinstance(alignment, SpanAlignment):
        raise ValueError('alignment must be a SpanAlignment record')
    rebuilt = build_span_alignment(donor_prompt, target_prompt, span=alignment.span,
                                   policy=alignment.policy, selector=alignment.selector)
    if rebuilt != alignment:
        raise ValueError('alignment differs from the recomputed span mapping')


def _bind_expected(result, nested, policy, label):
    gate = baseline_gate_input(result)
    if not gate.primary_valid:
        raise ValueError(f'{label} expected result must be primary-valid')
    provenance = result.provenance
    controls = _hf_controls(policy, nested.stop_token_ids)
    policy_record = asdict(policy)
    if (provenance.get('schema_sha256') != nested.schema_sha256
            or provenance.get('schema_bytes_sha256') != nested.schema_bytes_sha256
            or provenance.get('tokenizer_sha256') != nested.tokenizer_sha256
            or provenance.get('head_vocab_size') != nested.head_vocab_size
            or provenance.get('stop_token_ids') != list(nested.stop_token_ids)
            or not _canonically_equal(provenance.get('generation_policy'), policy_record)
            or provenance.get('hf_controls') != controls
            or provenance.get('generation_policy_sha256')
            != sha256_json({'policy': policy_record, 'hf_controls': controls})):
        raise ValueError(f'{label} expected provenance is not bound to the declared configuration')
    return gate


def _execution(status, donor, target_clean, intervention, expected_donor,
               expected_target, alignment, diagnostics=None):
    checks = ReplacementMatchChecks(
        _same_full_output(donor, expected_donor),
        None if target_clean is None else _same_full_output(target_clean, expected_target))
    return PromptReplacementExecution(status, donor, expected_donor, target_clean,
                                      expected_target, intervention, checks, alignment,
                                      _diagnostics_bytes=diagnostics)


@dataclass(frozen=True, slots=True)
class ReplacementMatchChecks:
    """Genuine parent-match flags recomputed from stored full outputs."""

    donor: bool
    target: bool | None = None

    def __post_init__(self):
        if type(self.donor) is not bool:
            raise ValueError('donor match must be a genuine bool')
        if self.target is not None and type(self.target) is not bool:
            raise ValueError('target match must be a genuine bool or None')

    def to_dict(self):
        return {'donor': self.donor, 'target': self.target}


@dataclass(frozen=True, slots=True)
class PromptReplacementExecution:
    """Frozen three-arm execution record; expected outputs are always present.

    Integrity validation recomputes match checks against the stored full
    outputs and enforces the status invariants; no config hash self-certifies
    the record and validation never certifies research eligibility.
    """

    status: str
    donor: StructuredGenerationResult
    expected_donor: StructuredGenerationResult
    target_clean: StructuredGenerationResult | None
    expected_target: StructuredGenerationResult
    intervention: StructuredGenerationResult | None
    match_checks: ReplacementMatchChecks
    alignment: SpanAlignment
    _diagnostics_bytes: bytes | None = None

    def __post_init__(self):
        if type(self.status) is not str or self.status not in _STATUSES:
            raise ValueError('invalid replacement execution status')
        if not isinstance(self.alignment, SpanAlignment):
            raise ValueError('alignment must be a SpanAlignment record')
        if not isinstance(self.match_checks, ReplacementMatchChecks):
            raise ValueError('match_checks must be ReplacementMatchChecks')
        expected_donor = baseline_gate_input(self.expected_donor)
        expected_target = baseline_gate_input(self.expected_target)
        if not expected_donor.primary_valid or not expected_target.primary_valid:
            raise ValueError('expected results must be primary-valid in every state')
        donor = baseline_gate_input(self.donor)
        target = None
        if self.target_clean is not None:
            target = baseline_gate_input(self.target_clean)
        if self.intervention is not None:
            baseline_gate_input(self.intervention)
        donor_match = _same_full_output(self.donor, self.expected_donor)
        target_match = (None if self.target_clean is None
                        else _same_full_output(self.target_clean, self.expected_target))
        if self.match_checks.donor is not donor_match:
            raise ValueError('match_checks donor disagrees with the stored full outputs')
        if self.match_checks.target is not target_match:
            raise ValueError('match_checks target disagrees with the stored full outputs')
        no_later = (self.target_clean is None and self.intervention is None
                    and self._diagnostics_bytes is None)
        if self.status == 'donor_failed':
            if donor.failure_type is None or not no_later:
                raise ValueError('donor_failed requires a failed donor and no later outputs')
        elif self.status == 'donor_mismatch':
            if donor.failure_type is not None or donor_match or not no_later:
                raise ValueError('donor_mismatch requires a valid nonmatching donor and no later outputs')
        elif self.status == 'source_unavailable':
            if donor.failure_type is not None or not donor_match or not no_later:
                raise ValueError('source_unavailable requires a valid matching donor and no later outputs')
        elif self.status == 'target_failed':
            if (donor.failure_type is not None or not donor_match or target is None
                    or target.failure_type is None or self.intervention is not None
                    or self._diagnostics_bytes is not None):
                raise ValueError('target_failed requires a valid matching donor and a failed clean target only')
        elif self.status == 'target_mismatch':
            if (donor.failure_type is not None or not donor_match or target is None
                    or target.failure_type is not None or target_match
                    or self.intervention is not None or self._diagnostics_bytes is not None):
                raise ValueError('target_mismatch requires a valid nonmatching clean target and no intervention')
        else:
            if (donor.failure_type is not None or not donor_match or target is None
                    or target.failure_type is not None or not target_match
                    or self.intervention is None or self._diagnostics_bytes is None):
                raise ValueError('executed requires matching valid donor/target, an intervention and diagnostics')
            diagnostics = _diagnostics(self._diagnostics_bytes, 'replacement')
            bound = (len(self.alignment.target_positions)
                     * (1 + len(self.intervention.generated_token_ids)))
            if diagnostics['selected_token_opportunities'] > bound:
                raise ValueError('replacement diagnostics exceed the selected-token bound')

    @property
    def executed(self):
        return self.status == 'executed'

    @property
    def diagnostics(self):
        return None if self._diagnostics_bytes is None else _diagnostics(self._diagnostics_bytes, 'replacement')

    def to_dict(self):
        return {
            'status': self.status,
            'executed': self.executed,
            'donor': self.donor.to_dict(),
            'expected_donor': self.expected_donor.to_dict(),
            'target_clean': None if self.target_clean is None else self.target_clean.to_dict(),
            'expected_target': self.expected_target.to_dict(),
            'intervention': None if self.intervention is None else self.intervention.to_dict(),
            'match_checks': self.match_checks.to_dict(),
            'alignment': self.alignment.to_dict(),
            'diagnostics': self.diagnostics,
        }


def execute_prompt_replacement(model, tokenizer, donor_prompt: DecisionPrompt,
                               target_prompt: DecisionPrompt, capability, *,
                               policy: StructuredGenerationPolicy,
                               layer: int, hook_site: str, alignment: SpanAlignment,
                               expected_donor: StructuredGenerationResult,
                               expected_target: StructuredGenerationResult,
                               verified_binding=None) -> PromptReplacementExecution:
    """Execute one genuine donor capture, clean recipient, patched recipient.

    All configuration binding happens before any model generation. Each arm
    runs one real greedy driver call under a fresh tracker; a runtime failure
    is retained as the arm's actual outcome and never resampled. Caller and
    helper errors raise, but hooks and source references are released on every
    path and no activation is cached.
    """
    if type(layer) is not int or not 0 <= layer < len(model.layers):
        raise ValueError('invalid layer coordinate')
    if hook_site not in ('pre', 'mid', 'post'):
        raise ValueError('unsupported hook site')
    if isinstance(capability, CompiledHarmonyDecisionGrammar):
        _check_harmony_capability(capability)
        nested = capability.json_capability
        driver = generate_harmony_structured
        channel = 'harmony_no_tools'
        suffix = capability.contract.prompt_suffix_ids
    elif isinstance(capability, CompiledDecisionGrammar):
        _check_capability(capability)
        nested = capability
        driver = generate_structured
        channel = 'plain_json'
        suffix = None
    else:
        raise ValueError('require a factory grammar capability')
    _validate_policy(policy, nested.head_vocab_size)
    if policy.channel_policy != channel:
        raise ValueError('policy does not match explicit capability route')
    _bind_prompt_records(donor_prompt, target_prompt, nested)
    weight = _input_embedding_weight(model)
    upper = min(weight.shape[0], nested.head_vocab_size)
    donor_ids = _prompt_tensor(donor_prompt, weight.device, upper, 'donor')
    target_ids = _prompt_tensor(target_prompt, weight.device, upper, 'target')
    if suffix is not None:
        for label, ids in (('donor', donor_ids), ('target', target_ids)):
            if tuple(ids[0, -len(suffix):].tolist()) != suffix:
                raise ValueError(f'{label} prompt must end in bound Harmony suffix')
    if _tokenizer_identity(tokenizer) != nested.tokenizer_sha256:
        raise ValueError('tokenizer identity differs from capability')
    _bind_model(model, nested, verified_binding)
    _bind_alignment(donor_prompt, target_prompt, alignment)
    _bind_expected(expected_donor, nested, policy, 'donor')
    _bind_expected(expected_target, nested, policy, 'target')

    def generate(prompt_ids):
        kwargs = dict(policy=policy, verified_binding=verified_binding)
        if driver is generate_structured:
            kwargs['transforms'] = None
        result = driver(model, tokenizer, prompt_ids, capability, **kwargs)
        baseline_gate_input(result)
        return result

    source = None
    gathered = None
    try:
        with ExitStack() as lifetime:
            with ExitStack() as tracking:
                donor_tracker = GenerationPositionTracker(donor_ids.shape[1], policy.use_cache)
                tracking.enter_context(donor_tracker.track(model))
                holder = lifetime.enter_context(capture_prompt_residual(
                    model, tracker=donor_tracker, layer=layer, hook_site=hook_site,
                    prompt_positions=list(alignment.donor_capture_positions)))
                donor = generate(donor_ids)
            if donor.failure_type is not None:
                return _execution('donor_failed', donor, None, None, expected_donor,
                                  expected_target, alignment)
            if not _same_full_output(donor, expected_donor):
                return _execution('donor_mismatch', donor, None, None, expected_donor,
                                  expected_target, alignment)
            try:
                source = holder.require_source()
            except ValueError:
                return _execution('source_unavailable', donor, None, None, expected_donor,
                                  expected_target, alignment)
            # Transient gather into target order; advanced indexing copies, and
            # the copy survives the capture lifetime closing on return.
            gathered = source[:, list(alignment.source_indices), :]
            with GenerationPositionTracker(target_ids.shape[1], policy.use_cache).track(model):
                target_clean = generate(target_ids)
            if target_clean.failure_type is not None:
                return _execution('target_failed', donor, target_clean, None, expected_donor,
                                  expected_target, alignment)
            if not _same_full_output(target_clean, expected_target):
                return _execution('target_mismatch', donor, target_clean, None, expected_donor,
                                  expected_target, alignment)
            patch_tracker = GenerationPositionTracker(target_ids.shape[1], policy.use_cache)
            with patch_tracker.track(model), scoped_residual_intervention(
                model, tracker=patch_tracker, layer=layer, hook_site=hook_site,
                scope='prompt_only', operation='replacement', source=gathered,
                source_positions=list(alignment.target_positions),
                prompt_positions=list(alignment.target_positions), dose=1,
            ) as intervention_diagnostics:
                intervention = generate(target_ids)
            return _execution(
                'executed', donor, target_clean, intervention, expected_donor,
                expected_target, alignment,
                diagnostics=canonical_json_bytes(intervention_diagnostics.to_dict()))
    finally:
        source = None
        gathered = None
