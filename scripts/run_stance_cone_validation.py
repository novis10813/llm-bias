"""Fixed full prospective cone validation. No operator acceptance or selection."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.run_stance_baseline import runtime_metadata
from scripts.smoke_stance_checkpoint import generation_adapter, compile_smoke_grammar
from scripts.recover_stance_baseline_truncations import (
    recovery_store, publish, read_file, check_prompt, check_capability,
)
from scripts.run_stance_localization import gate_record, import_gate, require_gate, _provenance
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.model import load_model
from llm_bias.core.experiment_contract import RowKey
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline
from llm_bias.core.stance_baseline_store import _generation
from llm_bias.core.stance_baseline_adapter import baseline_gate_input
from llm_bias.core.stance_cone_validation_plan import compile_cone_validation_plan
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy, generate_structured
from llm_bias.core.inference.harmony_generation import generate_harmony_structured
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop, _diagnostics
from llm_bias.core.inference.stance_localization_execution import _same_full_output
from llm_bias.core.inference.stance_interventions import (
    GenerationPositionTracker, scoped_residual_intervention,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('inputs', 'parent', 'model', 'training-dir', 'training-audit', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    return p


def assigned_cells(plan, index, count):
    directions = plan['panel']['directions']
    if not 1 <= count <= len(directions) or not 0 <= index < count:
        raise ValueError('invalid direction-modulo shard')
    assigned = {(d['cone'], d['family']) for i, d in enumerate(directions) if i % count == index}
    return {sha256_json(c) + '.json': c for c in plan['cells']
            if (c['cone'], c['family']) in assigned}


def envelope(config_hash, payload):
    return dict(kind='stance_cone_validation_record_v1', config_hash=config_hash,
                payload=payload, payload_sha256=sha256_json(payload))


def unwrap(item, config_hash):
    if item != envelope(config_hash, item['payload']):
        raise ValueError('record hash/configuration differs')
    return item['payload']


def addition(model, tokenizer, prompt, capability, policy, vector, dose, *, layer=19):
    """One actual driver call with fresh absolute-position tracking, even at zero."""
    weight = model.hf_model.get_input_embeddings().weight
    vector = vector.detach().to(device=weight.device, dtype=weight.dtype)
    if not torch.isfinite(vector).all() or not (vector != 0).any():
        raise ValueError('invalid native direction')
    ids = torch.tensor([prompt.inference_token_ids], device=weight.device, dtype=torch.long)
    span = prompt.instruction_span
    positions = list(range(span.token_start, span.token_end))
    tracker = GenerationPositionTracker(ids.shape[1], policy.use_cache)
    driver = generate_harmony_structured if policy.channel_policy == 'harmony_no_tools' else generate_structured
    with torch.inference_mode(), tracker.track(model), scoped_residual_intervention(
            model, tracker=tracker, layer=layer, hook_site='post', scope='prompt_only',
            operation='addition', vector=vector, dose=dose, prompt_positions=positions) as diag:
        result = driver(model, tokenizer, ids, capability, policy=policy)
    baseline_gate_input(result)
    return result, diag.to_dict()


def execute(records, plan, parent, inputs, config_hash, index, count, gate, clean, intervene):
    """One O(N) import pass and one keyed execution pass. Tiny plans are internal only."""
    cells = assigned_cells(plan, index, count)
    keys = {}
    for cell in plan['cells']:
        h = sha256_json(cell['row'])
        if h not in keys:
            keys[h] = RowKey(**cell['row'])
    first = next(iter(keys.values()))
    expected_names = {'gate.json'} | set(cells)
    expected_names |= {prefix + h + '.json' for h in keys for prefix in ('clean-', 'zero-')}
    saved = {}
    for path in records.iterdir():
        if path.name in ('plan.json', 'runtime.json'):
            continue
        if path.name not in expected_names:
            raise ValueError('foreign/legacy record or staging leftover')
        saved[path.name] = unwrap(read_file(path), config_hash)

    def put(name, payload):
        payload = json.loads(canonical_json_bytes(payload))
        # Publish immediately, retaining every genuine failed generation too.
        publish(records, name, envelope(config_hash, payload))
        saved[name] = payload

    def generation(payload, key, dose=None):
        result = _generation(json.loads(canonical_json_bytes(payload['generation'])))
        baseline_gate_input(result)
        _provenance(result, parent.generation_for(key))
        allowed = {'row', 'generation'} if dose is None else {'row', 'generation', 'diagnostics', 'dose'}
        if set(payload) != allowed or payload['row'] != key.to_dict():
            raise ValueError('generation row/configuration differs')
        if dose is not None:
            diag = _diagnostics(canonical_json_bytes(payload['diagnostics']), 'addition')
            if (payload['dose'] != dose or diag['layer'] != 19 or diag['hook_site'] != 'post'):
                raise ValueError('addition coordinates differ')
            if result.failure_type is None and diag['selected_token_opportunities'] == 0:
                raise ValueError('successful generation bypassed intervention')
            if dose == 0 and (diag['changed_token_count'] or diag['delta_l2_max']):
                raise ValueError('zero hook changed residual')
        return result

    # Validate every imported row before trusting it, not just its self digest.
    for h, key in keys.items():
        for prefix, dose in (('clean-', None), ('zero-', 0)):
            name = prefix + h + '.json'
            if name in saved:
                generation(saved[name], key, dose)
    for name, cell in cells.items():
        if name not in saved:
            continue
        item = saved[name]
        if item.get('cell') != cell:
            raise ValueError('cell identity differs')
        key = keys[sha256_json(cell['row'])]
        if cell['dose']:
            if set(item) != {'cell', 'actual'}:
                raise ValueError('invalid actual cell')
            generation(item['actual'], key, cell['dose'])
        else:
            validate_zero(item, cell, saved, config_hash)
    if any(n in saved for n in cells) and ('gate.json' not in saved or any(
            prefix + h + '.json' not in saved for h in keys for prefix in ('clean-', 'zero-'))):
        raise ValueError('effect records without complete prerequisite records')
    halted = None
    replay_passed = False
    try:
        if 'gate.json' not in saved:
            put('gate.json', gate_record(first, parent.generation_for(first), gate(first), config_hash))
        execution = import_gate(saved['gate.json'], first, parent, 19, config_hash)
        require_gate(execution, parent.generation_for(first))
        # A gate baseline and genuine zero arm are reused explicitly, not rerun.
        for h, key in keys.items():
            cname, zname = 'clean-' + h + '.json', 'zero-' + h + '.json'
            if cname not in saved:
                result = execution.baseline if key == first else clean(key)
                put(cname, dict(row=key.to_dict(), generation=result.to_dict()))
            actual = generation(saved[cname], key)
            if actual.failure_type is not None or not _same_full_output(actual, parent.generation_for(key)):
                halted = 'clean_failure_or_parent_drift'
                break
            if zname not in saved:
                if key == first:
                    result, diag = execution.zero, execution.zero_diagnostics
                else:
                    result, diag = intervene(key, None, 0)
                put(zname, dict(row=key.to_dict(), generation=result.to_dict(), diagnostics=diag, dose=0))
            zero = generation(saved[zname], key, 0)
            if zero.failure_type is not None or not _same_full_output(zero, actual):
                halted = 'zero_failure_or_parent_drift'
                break
        if halted is None:
            replay_passed = True
            for name, cell in cells.items():
                if name in saved:
                    continue
                key = keys[sha256_json(cell['row'])]
                if cell['dose'] == 0:
                    item = zero_reference(cell, saved, config_hash)
                else:
                    result, diag = intervene(key, (cell['cone'], cell['family']), cell['dose'])
                    actual = dict(row=key.to_dict(), generation=result.to_dict(), diagnostics=diag, dose=cell['dose'])
                    generation(actual, key, cell['dose'])
                    item = dict(cell=cell, actual=actual)
                put(name, item)
    except (Exception, KeyboardInterrupt) as exc:
        halted = type(exc).__name__
    # Existing effect records cannot precede valid full clean/zero coverage.
    if any(n in saved for n in cells) and (not replay_passed or any(
            prefix + h + '.json' not in saved for h in keys for prefix in ('clean-', 'zero-'))):
        raise ValueError('effect records without passed gate and complete exact replay')
    groups = defaultdict(Counter)
    companies = defaultdict(lambda: defaultdict(Counter))
    for name, cell in cells.items():
        key = keys[sha256_json(cell['row'])]
        source = parent.generation_for(key)
        source_class = source.decision if source.failure_type is None else 'unknown'
        group_key = (cell['cone'], cell['family'], cell['dose'], cell['arm'], key.condition, source_class)
        stats = groups[group_key]
        stats['planned'] += 1
        stats['unknown_baseline'] += int(source_class == 'unknown')
        stats['directional_denominator'] += int(source_class == ('sell' if cell['dose'] > 0 else 'buy') and cell['dose'] != 0)
        company = companies[group_key][inputs.issuer_by_ticker[key.ticker]]
        company['planned'] += 1
        company['unknown_baseline'] += int(source_class == 'unknown')
        company['directional_denominator'] += int(source_class == ('sell' if cell['dose'] > 0 else 'buy') and cell['dose'] != 0)
        if name not in saved:
            stats['missing'] += 1
            company['missing'] += 1
            continue
        item = saved[name]
        result = (_generation(item['actual']['generation']) if cell['dose'] else
                  _generation(saved[item['source_generation']['zero_name']]['generation']))
        stats['executed'] += 1
        stats['failure'] += int(result.failure_type is not None)
        stats['invalid'] += int(result.failure_type is not None)
        company['executed'] += 1
        for field in ('schema_complete', 'reason_valid', 'decision_complete'):
            stats[field] += int(getattr(result, field))
            company[field] += int(getattr(result, field))
        flip = source_class != 'unknown' and result.failure_type is None and result.decision != source_class
        directional = flip and source_class == ('sell' if cell['dose'] > 0 else 'buy') and cell['dose'] != 0
        stats['flip'] += int(flip)
        stats['directional_flip'] += int(directional)
        company['flip'] += int(flip)
        company['directional_flip'] += int(directional)
        company['failure'] += int(result.failure_type is not None)
    completed = sum(n in saved for n in cells)
    return dict(kind='stance_cone_validation_summary_v1', config_hash=config_hash,
        planned_global=129600, planned_shard=len(cells), executed_logical=completed,
        missing=len(cells) - completed, complete_shard=completed == len(cells) and halted is None,
        complete_global=count == 1 and completed == 129600 and halted is None,
        halt_reason=halted, accepted_operator=False, research_eligible=False, acceptance_threshold=None, efficacy_threshold=None,
        denominator_policy='fixed_parent_source_class_ITT;missing_and_invalid_are_not_flips',
        independent_zero_generations=sum(n.startswith('zero-') for n in saved),
        zero_references_are_not_independent_replicates=True,
        groups=[dict(cone=k[0], family=k[1], dose=k[2], arm=k[3], condition=k[4], source_class=k[5],
                     counts=dict(v), issuer_first=[dict(issuer=i, **dict(s)) for i, s in sorted(companies[k].items())])
                for k, v in sorted(groups.items())])


def zero_reference(cell, saved, config_hash):
    h = sha256_json(cell['row'])
    clean, zero = 'clean-' + h + '.json', 'zero-' + h + '.json'
    return dict(cell=cell, source_generation=dict(
        baseline_name=clean, baseline_record_sha256=sha256_bytes(canonical_json_bytes(envelope(config_hash, saved[clean])) + b'\n'),
        zero_name=zero, zero_record_sha256=sha256_bytes(canonical_json_bytes(envelope(config_hash, saved[zero])) + b'\n'),
        configuration_sha256=config_hash, reuse=True, independent_replicate=False,
        generation_origin='actual_parent_replay_and_actual_zero_hook',
        gate_arm_reuse=(saved.get('gate.json', {}).get('key') == cell['row'])))


def validate_zero(item, cell, saved, config_hash):
    if item != zero_reference(cell, saved, config_hash):
        raise ValueError('zero source reference/hash differs')


def run(args):
    inputs = load_baseline_inputs(args.inputs)
    parent = load_completed_baseline(args.parent, inputs=inputs)
    bindings = parent.metadata['bindings']
    compiled = compile_cone_validation_plan(inputs, parent.plan, args.training_dir, args.training_audit)
    plan = compiled.to_dict()
    if plan['parent_sha256'] != parent.parent_sha256:
        raise ValueError('audited parent differs')
    assigned_count = len(assigned_cells(plan, args.shard_index, args.num_shards))
    checkpoint = args.model.resolve(strict=True)
    metadata = runtime_metadata(checkpoint)
    if metadata['model'] != bindings['model'] or 'glm4-9b-0414' not in str(checkpoint).lower():
        raise ValueError('original GLM parent checkpoint required')
    for field in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy', 'cudnn', 'deterministic_algorithms'):
        if metadata['backend'][field] != bindings['backend'][field]:
            raise ValueError('parent backend differs: ' + field)
    if bindings['backend']['requested_dtype'] not in ('native', 'bfloat16'):
        raise ValueError('native/BF16 parent requested mode required')
    for name in ('scripts/run_stance_cone_validation.py', 'scripts/run_stance_localization.py',
                 'scripts/recover_stance_baseline_truncations.py'):
        metadata['code']['source_sha256'][name] = sha256_bytes((ROOT / name).read_bytes())
    # Source inventory is current-file hash verified. It is not parent code equivalence.
    for name, digest in metadata['code']['source_sha256'].items():
        if sha256_bytes((ROOT / name).read_bytes()) != digest:
            raise ValueError('current source identity changed')
    preflight = dict(kind='stance_cone_validation_preflight_v1', plan_id=compiled.plan_id,
        panel_id=compiled.panel_id, shard_index=args.shard_index, num_shards=args.num_shards,
        runtime=metadata, parent_sha256=parent.parent_sha256, bindings=bindings,
        roles_sha256=inputs.roles_sha256, inputs_manifest_sha256=inputs.manifest_sha256,
        accepted_operator=False, research_eligible=False)
    registration = dict(preflight=preflight, preflight_sha256=sha256_json(preflight))
    with recovery_store(args.output_dir, registration) as records:
        # Write the entire fixed plan and derived panel before allocating checkpoint.
        if (records / 'plan.json').exists():
            if read_file(records / 'plan.json') != plan:
                raise ValueError('saved immutable plan differs')
        else:
            if any(records.iterdir()):
                raise ValueError('partial registration has no plan; no repair')
            publish(records, 'plan.json', plan)
        if (args.output_dir / 'summary.json').exists():
            prior_summary = read_file(args.output_dir / 'summary.json')
            if not prior_summary.get('complete_shard'):
                raise ValueError('terminal incomplete attempt cannot be repaired')
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; no CPU fallback')
            loaded, _, device = load_model(str(checkpoint), device_map=None,
                dtype='native' if bindings['backend']['requested_dtype'] == 'native' else torch.bfloat16)
            model = generation_adapter(loaded)
            model.hf_model.eval().requires_grad_(False)
            weight = model.hf_model.get_input_embeddings().weight
            head = model.hf_model.get_output_embeddings().weight
            device = torch.device(device)
            if (len(model.layers) != 40 or weight.shape[1] != 4096 or head.shape[1] != 4096
                    or weight.dtype != torch.bfloat16 or head.dtype != torch.bfloat16
                    or device.type != 'cuda' or weight.device != device or head.device != device):
                raise ValueError('single GPU 40-layer H4096 BF16 checkpoint required')
            if (str(weight.dtype) != bindings['backend']['embedding_dtype']
                    or str(head.dtype) != bindings['backend']['head_dtype']
                    or getattr(model.hf_model.config, '_attn_implementation', None) != bindings['backend']['attention_implementation']):
                raise ValueError('actual parent dtype/attention differs')
            policy = StructuredGenerationPolicy(**bindings['generation_policy']['policy'])
            reference = parent.generation_for(parent.plan.keys[0])
            cap = compile_smoke_grammar(model.tokenizer, head.shape[0], reference.provenance['stop_token_ids'], policy.channel_policy)
            check_capability(cap, reference.provenance)
            wrapper = WrapperPolicy(**{k: v for k, v in bindings['template']['actual_wrapper_record'].items() if k != 'tokenizer_chat_template'})
            members = {m.ticker: m for m in inputs.members}
            prompts = {}
            for key in parent.plan.keys:
                if inputs.roles['assignments'][key.ticker] != 'validation':
                    continue
                prompt = render_decision_prompt(model.tokenizer, members[key.ticker],
                    inputs.pair_for(key.ticker, key.condition, key.trial_id), wrapper_policy=wrapper)
                check_prompt(prompt, bindings)
                if prompt.schema_sha256 != parent.plan.identity.schema_sha256:
                    raise ValueError('schema differs')
                prompts[key] = prompt
            runtime = dict(preflight_sha256=sha256_json(preflight), plan_id=compiled.plan_id,
                actual_embedding_dtype=str(weight.dtype), actual_head_dtype=str(head.dtype),
                device=str(device), gpu_name=torch.cuda.get_device_name(device),
                attention_implementation=model.hf_model.config._attn_implementation,
                prompt_sha256={sha256_json(k.to_dict()): sha256_json(dict(ids=list(p.inference_token_ids),
                    text=p.rendered_text)) for k, p in prompts.items()})
            if (records / 'runtime.json').exists():
                if read_file(records / 'runtime.json') != runtime:
                    raise ValueError('runtime resume binding differs')
            else:
                if any(p.name != 'plan.json' for p in records.iterdir()):
                    raise ValueError('records without runtime binding')
                publish(records, 'runtime.json', runtime)
            config_hash = sha256_json(dict(preflight=preflight, runtime=runtime))
            directions = {(d['cone'], d['family']): torch.tensor(d['direction'], device=device,
                dtype=weight.dtype) for d in plan['panel']['directions']}
            zero_vector = next(iter(directions.values()))
            driver = generate_harmony_structured if policy.channel_policy == 'harmony_no_tools' else generate_structured
            def clean(key):
                ids = torch.tensor([prompts[key].inference_token_ids], device=device, dtype=torch.long)
                with torch.inference_mode():
                    return driver(model, model.tokenizer, ids, cap, policy=policy)
            def gate(key):
                prompt = prompts[key]
                ids = torch.tensor([prompt.inference_token_ids], device=device, dtype=torch.long)
                span = prompt.instruction_span
                with torch.inference_mode():
                    return execute_prompt_noop(model, model.tokenizer, ids, cap, policy=policy,
                        layer=19, hook_site='post', zero_vector=zero_vector,
                        prompt_positions=list(range(span.token_start, span.token_end)), config_hash=config_hash)
            def intervene(key, direction, dose):
                return addition(model, model.tokenizer, prompts[key], cap, policy,
                    zero_vector if direction is None else directions[direction], dose)
            report = execute(records, plan, parent, inputs, config_hash, args.shard_index,
                             args.num_shards, gate, clean, intervene)
        except (Exception, KeyboardInterrupt) as exc:
            if any(p.name not in ('plan.json', 'runtime.json') for p in records.iterdir()):
                raise
            report = dict(kind='stance_cone_validation_preflight_failure_v1', error_type=type(exc).__name__,
                complete_shard=False, complete_global=False, accepted_operator=False, research_eligible=False,
                planned_global=129600, planned_shard=assigned_count,
                executed_logical=0, missing=assigned_count,
                preflight_sha256=sha256_json(preflight))
        summary = args.output_dir / 'summary.json'
        if summary.exists():
            if read_file(summary) != report:
                raise ValueError('immutable summary differs; failed attempts cannot be repaired')
        else:
            publish(args.output_dir, 'summary.json', report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return 0 if report.get('complete_shard') else 1


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (Exception, KeyboardInterrupt) as exc:
        print(canonical_json_bytes(dict(error_type=type(exc).__name__, accepted_operator=False,
                                       research_eligible=False)).decode(), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
