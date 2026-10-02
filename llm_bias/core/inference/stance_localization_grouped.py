"""Pair-scoped LC2 execution. Sources live only within this call."""
from contextlib import ExitStack
from dataclasses import dataclass

from ..artifact_paths import canonical_json_bytes
from ..stance_baseline_adapter import baseline_gate_input
from ..stance_localization_alignment import SpanAlignment
from .harmony_generation import (
    CompiledHarmonyDecisionGrammar, _check_harmony_capability, generate_harmony_structured,
)
from .structured_output import (
    CompiledDecisionGrammar, StructuredGenerationResult, _check_capability,
    _validate_policy, _tokenizer_identity, _bind_model, generate_structured,
)
from .stance_interventions import GenerationPositionTracker, scoped_residual_intervention
from .stance_transient_capture import capture_prompt_residual
from .stance_localization_execution import (
    PromptReplacementExecution, _bind_prompt_records, _input_embedding_weight,
    _prompt_tensor, _bind_alignment, _bind_expected, _same_full_output, _execution,
)

__all__ = ['ReplacementCell', 'GroupedReplacementExecution',
           'execute_grouped_prompt_replacement']


@dataclass(frozen=True, slots=True)
class ReplacementCell:
    """One immutable layer/site/alignment coordinate in execution order."""

    layer: int
    hook_site: str
    alignment: SpanAlignment


@dataclass(frozen=True, slots=True)
class GroupedReplacementExecution:
    """Shared actual clean outcomes and LC2 records, with no tensor references.

    Clean-arm aborts use LC2 status names and contain no per-cell rows.
    Coordinates preserve scope alongside the ordered LC2 execution records.
    Parent matching does not certify phase-level gates.
    """

    status: str
    cells: tuple[ReplacementCell, ...]
    donor: StructuredGenerationResult
    target_clean: StructuredGenerationResult | None
    executions: tuple[PromptReplacementExecution, ...]


def execute_grouped_prompt_replacement(model, tokenizer, donor_prompt, target_prompt,
                                       capability, *, policy, cells,
                                       expected_donor, expected_target,
                                       verified_binding=None):
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
    for cell in cells:
        _bind_alignment(donor_prompt, target_prompt, cell.alignment)
    _bind_expected(expected_donor, nested, policy, 'donor')
    _bind_expected(expected_target, nested, policy, 'target')

    def generate(prompt_ids):
        kwargs = dict(policy=policy, verified_binding=verified_binding)
        if driver is generate_structured:
            kwargs['transforms'] = None
        result = driver(model, tokenizer, prompt_ids, capability, **kwargs)
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
                donor_tracker = GenerationPositionTracker(donor_ids.shape[1], policy.use_cache)
                tracking.enter_context(donor_tracker.track(model))
                for (layer, site), positions in unions.items():
                    holders[layer, site] = lifetime.enter_context(capture_prompt_residual(
                        model, tracker=donor_tracker, layer=layer, hook_site=site,
                        prompt_positions=list(positions)))
                donor = generate(donor_ids)
            if donor.failure_type is not None:
                return abort('donor_failed')
            if not _same_full_output(donor, expected_donor):
                return abort('donor_mismatch')
            try:
                for key, holder in holders.items():
                    sources[key] = holder.require_source()
            except ValueError:
                return abort('source_unavailable')
            with GenerationPositionTracker(target_ids.shape[1], policy.use_cache).track(model):
                target_clean = generate(target_ids)
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
                tracker = GenerationPositionTracker(target_ids.shape[1], policy.use_cache)
                with tracker.track(model), scoped_residual_intervention(
                    model, tracker=tracker, layer=cell.layer, hook_site=cell.hook_site,
                    scope='prompt_only', operation='replacement', source=gathered,
                    source_positions=list(alignment.target_positions),
                    prompt_positions=list(alignment.target_positions), dose=1,
                ) as diagnostics:
                    intervention = generate(target_ids)
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
