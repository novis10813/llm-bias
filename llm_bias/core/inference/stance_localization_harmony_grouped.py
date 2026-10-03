"""Native Harmony pair-scoped replacement with independently bound arm policies.

Only compact LC2 records escape. Original donor sources remain transient and
inert during target generation. This module does not implement a runner or
certify parent cohort eligibility.
"""
from contextlib import ExitStack
from dataclasses import asdict

from ..artifact_paths import canonical_json_bytes, sha256_json
from ..stance_baseline_adapter import baseline_gate_input
from ..stance_localization_alignment import SpanAlignment
from .harmony_generation import _check_harmony_capability, generate_harmony_structured
from .structured_output import _validate_policy, _tokenizer_identity, _bind_model
from .stance_interventions import GenerationPositionTracker, scoped_residual_intervention
from .stance_transient_capture import capture_prompt_residual
from .stance_localization_grouped import ReplacementCell, GroupedReplacementExecution
from .stance_localization_execution import (
    _bind_prompt_records, _input_embedding_weight, _prompt_tensor, _bind_alignment,
    _bind_expected, _same_full_output, _execution, _canonically_equal,
)

__all__ = ['execute_harmony_grouped_prompt_replacement']


def execute_harmony_grouped_prompt_replacement(
    model, tokenizer, donor_prompt, target_prompt, capability, *, donor_policy,
    target_policy, cells, expected_donor, expected_target, verified_binding=None,
):
    """Share two clean generations across a nonempty tuple of ReplacementCells.

    Binding completes before generation. Intervention runtime failures remain
    executed outcomes and do not stop later cells. Caller errors propagate
    after cleanup. Captures never escape the call or observe target forwards.
    """
    if not isinstance(cells, tuple) or not cells:
        raise ValueError('cells must be a nonempty immutable tuple')
    keys = set()
    for cell in cells:
        if not isinstance(cell, ReplacementCell):
            raise ValueError('cells must contain ReplacementCell records')
        if type(cell.layer) is not int or not 0 <= cell.layer < len(model.layers):
            raise ValueError('invalid layer coordinate')
        if cell.hook_site not in ('pre', 'mid', 'post'):
            raise ValueError('unsupported hook site')
        if not isinstance(cell.alignment, SpanAlignment):
            raise ValueError('alignment must be a SpanAlignment record')
        key = (cell.layer, cell.hook_site, cell.alignment.mapping_sha256)
        if key in keys:
            raise ValueError('duplicate cell coordinate')
        keys.add(key)
    _check_harmony_capability(capability)
    nested = capability.json_capability
    suffix = capability.contract.prompt_suffix_ids
    for policy in (donor_policy, target_policy):
        _validate_policy(policy, nested.head_vocab_size)
        if policy.channel_policy != 'harmony_no_tools':
            raise ValueError('policy does not match native Harmony capability')
    _bind_prompt_records(donor_prompt, target_prompt, nested)
    weight = _input_embedding_weight(model)
    upper = min(weight.shape[0], nested.head_vocab_size)
    donor_ids = _prompt_tensor(donor_prompt, weight.device, upper, 'donor')
    target_ids = _prompt_tensor(target_prompt, weight.device, upper, 'target')
    for label, ids in (('donor', donor_ids), ('target', target_ids)):
        if tuple(ids[0, -len(suffix):].tolist()) != suffix:
            raise ValueError(f'{label} prompt must end in bound Harmony suffix')
    if _tokenizer_identity(tokenizer) != nested.tokenizer_sha256:
        raise ValueError('tokenizer identity differs from capability')
    _bind_model(model, nested, verified_binding)
    for cell in cells:
        _bind_alignment(donor_prompt, target_prompt, cell.alignment)
    _bind_expected(expected_donor, nested, donor_policy, 'donor')
    _bind_expected(expected_target, nested, target_policy, 'target')

    # LC2 binds nested schema/head/stops and each arm's complete policy.
    # Native parents additionally bind the factory-owned channel contract.
    for label, expected in (('donor', expected_donor), ('target', expected_target)):
        provenance = expected.provenance
        if (not _canonically_equal(provenance.get('channel_contract'), asdict(capability.contract))
                or provenance.get('channel_contract_sha256') != capability.contract_sha256
                or provenance.get('channel_policy_sha256') != sha256_json({
                    'policy': 'harmony_no_tools', 'contract_sha256': capability.contract_sha256})):
            raise ValueError(f'{label} expected Harmony contract differs from capability')

    def generate(prompt_ids, policy):
        result = generate_harmony_structured(
            model, tokenizer, prompt_ids, capability, policy=policy,
            verified_binding=verified_binding)
        baseline_gate_input(result)
        return result

    unions = {}
    for cell in cells:
        unions.setdefault((cell.layer, cell.hook_site), set()).update(
            cell.alignment.donor_capture_positions)
    unions = {key: tuple(sorted(positions)) for key, positions in unions.items()}
    holders, sources = {}, {}
    source = gathered = None
    donor = target_clean = None

    def abort(status):
        return GroupedReplacementExecution(status, cells, donor, target_clean, ())

    try:
        with ExitStack() as lifetime:
            with ExitStack() as tracking:
                donor_tracker = GenerationPositionTracker(donor_ids.shape[1], donor_policy.use_cache)
                tracking.enter_context(donor_tracker.track(model))
                for (layer, site), positions in unions.items():
                    holders[layer, site] = lifetime.enter_context(capture_prompt_residual(
                        model, tracker=donor_tracker, layer=layer, hook_site=site,
                        prompt_positions=list(positions)))
                donor = generate(donor_ids, donor_policy)
            if donor.failure_type is not None:
                return abort('donor_failed')
            if not _same_full_output(donor, expected_donor):
                return abort('donor_mismatch')
            try:
                for key, holder in holders.items():
                    sources[key] = holder.require_source()
            except ValueError:
                return abort('source_unavailable')
            with GenerationPositionTracker(target_ids.shape[1], target_policy.use_cache).track(model):
                target_clean = generate(target_ids, target_policy)
            if target_clean.failure_type is not None:
                return abort('target_failed')
            if not _same_full_output(target_clean, expected_target):
                return abort('target_mismatch')
            executions = []
            for cell in cells:
                key = (cell.layer, cell.hook_site)
                alignment = cell.alignment
                source = sources[key]
                union_indices = {position: i for i, position in enumerate(unions[key])}
                # Expand the accepted cell mapping into the union's capture order.
                indices = [union_indices[alignment.donor_capture_positions[i]]
                           for i in alignment.source_indices]
                gathered = source[:, indices, :]
                tracker = GenerationPositionTracker(target_ids.shape[1], target_policy.use_cache)
                with tracker.track(model), scoped_residual_intervention(
                    model, tracker=tracker, layer=cell.layer, hook_site=cell.hook_site,
                    scope='prompt_only', operation='replacement', source=gathered,
                    source_positions=list(alignment.target_positions),
                    prompt_positions=list(alignment.target_positions), dose=1,
                ) as diagnostics:
                    intervention = generate(target_ids, target_policy)
                executions.append(_execution(
                    'executed', donor, target_clean, intervention, expected_donor,
                    expected_target, alignment,
                    diagnostics=canonical_json_bytes(diagnostics.to_dict())))
                source = gathered = None
            return GroupedReplacementExecution('executed', cells, donor, target_clean,
                                               tuple(executions))
    finally:
        source = gathered = None
        sources.clear()
        holders.clear()
