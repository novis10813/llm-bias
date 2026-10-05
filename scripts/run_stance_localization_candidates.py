"""Fixed historical candidate-panel native-BF16 development localization.

Compose the accepted grouped grid, executor, gates and immutable store. No
historical-result migration or public cohort/layer/span override is supported.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts import run_stance_localization_grouped as grouped
from scripts import run_stance_localization as logical
from scripts import run_stance_localization_plain_crossmodel as plain
from llm_bias.core.stance_localization_candidate_panel import build_localization_candidate_panel
from llm_bias.core.artifact_paths import sha256_json
from scripts.recover_stance_baseline_truncations import recovery_store, publish, read_file
from scripts.run_stance_localization import (cells as original_cells, alignment_for, cell_record,
    import_cell, gate_record, import_gate, require_gate, _equal, SPANS, ROLES)
from llm_bias.core.inference.stance_localization_grouped import ReplacementCell
from scripts.smoke_stance_checkpoint import generation_adapter, compile_smoke_grammar
from scripts.recover_stance_baseline_truncations import check_capability, check_prompt
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline
from llm_bias.core.stance_localization_pairs import build_localization_pairs
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy
from llm_bias.core.inference.stance_localization_execution import _bind_expected
from llm_bias.core.inference.stance_localization_grouped import execute_grouped_prompt_replacement
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop


def parser():
    p = logical.parser()
    next(a for a in p._actions if a.dest == 'phase').choices = ('primary',)
    p.add_argument("--model-slug", required=True, choices=tuple(MODEL_COUNTS))
    return p


def config_identity(checkpoint, slug):
    if slug == 'gpt-oss-20b':
        raise NotImplementedError('GPT candidate route unsupported: strict merged Harmony replay remains blocked')
    if slug != 'glm4-9b-0414':
        identity = plain.config_identity(checkpoint)
        families = {'qwen3.5-4b': ('qwen3_5', 'qwen3_5_text'),
                    'gemma4-12b-it': ('gemma4', 'gemma4_text', 'gemma4_unified_text')}
        if identity['model_type'] not in families[slug]:
            raise ValueError('canonical model slug differs from checkpoint family')
    else:
        raw = (checkpoint / 'config.json').read_bytes()
        config = json.loads(raw)
        if config.get('model_type') != 'glm4':
            raise ValueError('GLM requires native glm4 config')
        dtype = config.get('dtype', config.get('torch_dtype'))
        if dtype not in ('bfloat16', 'torch.bfloat16'):
            raise ValueError('checkpoint config must declare native BF16')
        identity = dict(model_type='glm4', config_sha256=sha256_bytes(raw),
                        configured_layer_count=config.get('num_hidden_layers'), declared_dtype=dtype)
    if type(identity['configured_layer_count']) is not int or identity['configured_layer_count'] != MODEL_COUNTS[slug]:
        raise ValueError('configured depth differs from canonical model')
    return identity


def authenticate_layers(model, identity):
    return plain.authenticate_layers(model, identity)


def validate_parent(table, bindings, slug):
    if len(table.pairs) != 4024 or sum(p.role in logical.ROLES for p in table.pairs) != 3016:
        raise ValueError('require full 4024-pair / 3016-development grid')
    if Counter(p.role for p in table.pairs) != dict(fit=2416, validation=600, calibration=208, evaluation=800):
        raise ValueError('require unchanged full role counts')
    policy = StructuredGenerationPolicy(**bindings['generation_policy']['policy'])
    if (policy.channel_policy != 'plain_json' or policy.max_new_tokens != 512
            or policy.timeout_seconds != 180 or policy.use_cache is not True):
        raise ValueError('require original 512-token/180-second/cache/plainJSON parent policy')
    backend = bindings['backend']
    if (backend['requested_dtype'] not in ('native', 'bfloat16') or backend['head_dtype'] != 'torch.bfloat16'
            or backend['embedding_dtype'] != 'torch.bfloat16'):
        raise ValueError('require native BF16 baseline parent')
    if slug == 'glm4-9b-0414' and backend['requested_dtype'] != 'bfloat16':
        raise ValueError('GLM original parent requires requested BF16')
    return policy


def bind_runtime(checkpoint, bindings):
    metadata = plain.bind_runtime(checkpoint, bindings)
    metadata['code']['source_sha256']['scripts/run_stance_localization_candidates.py'] = sha256_bytes(Path(__file__).read_bytes())
    return metadata


def run(args):
    if args.phase != 'primary' or getattr(args, 'prior_run', None) is not None:
        raise ValueError('only fresh primary execution, no prior migration')
    if args.model_slug == 'gpt-oss-20b':
        raise NotImplementedError('GPT candidate route unsupported: strict merged Harmony replay remains blocked')
    rt = load_runtime(args)
    panel = build_localization_candidate_panel(model_slug=args.model_slug, actual_layer_count=rt.count)
    shard_panel(panel.layers, args.shard_index, args.num_shards)
    desc = descriptor(rt.table, panel, args.shard_index, args.num_shards, rt.bindings_record)
    model, policy, capability, prompts, parent = rt.model, rt.policy, rt.capability, rt.prompts, rt.parent
    embedding = rt.embedding

    def gate(key, layer, span, config_hash):
        prompt = prompts[key]
        record = getattr(prompt, span + '_span')
        return execute_prompt_noop(model, model.tokenizer,
            torch.tensor([prompt.inference_token_ids], device=embedding.device, dtype=torch.long),
            capability, policy=policy, layer=layer, hook_site='post',
            zero_vector=torch.ones(embedding.shape[1], device=embedding.device, dtype=embedding.dtype),
            prompt_positions=list(range(record.token_start, record.token_end)), config_hash=config_hash)

    def group(pair, coordinates):
        return execute_grouped_prompt_replacement(model, model.tokenizer, prompts[pair.donor_key],
            prompts[pair.target_key], capability, policy=policy, cells=coordinates,
            expected_donor=parent.generation_for(pair.donor_key),
            expected_target=parent.generation_for(pair.target_key))

    return execute_run(args, desc, rt.table, prompts, parent, gate, group)


def load_runtime(args):
    """Validated parent, authenticated CUDA BF16 model, grammar and bound prompts."""
    inputs = load_baseline_inputs(args.inputs)
    parent = load_completed_baseline(args.parent, inputs=inputs)
    table = build_localization_pairs(inputs, parent)
    bindings = parent.metadata['bindings']
    policy = validate_parent(table, bindings, args.model_slug)
    checkpoint = args.model.resolve(strict=True)
    identity = config_identity(checkpoint, args.model_slug)
    metadata = bind_runtime(checkpoint, bindings)
    logical._equal(identity['config_sha256'],
        metadata['model']['metadata_file_sha256'].get('config.json'), 'config bytes changed during preflight')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    requested_dtype = bindings['backend']['requested_dtype']
    loaded, _, device = load_model(str(checkpoint), device_map=None,
                                  dtype='native' if requested_dtype == 'native' else torch.bfloat16)
    model = generation_adapter(loaded)
    model.hf_model.eval()
    count = authenticate_layers(model, identity)
    head = model.hf_model.get_output_embeddings().weight
    embedding = model.hf_model.get_input_embeddings().weight
    if any(torch.device(d).type != 'cuda' for d in (device, head.device, embedding.device)):
        raise ValueError('actual model must be on CUDA')
    if any(p.device.type != 'cuda' or (p.is_floating_point() and p.dtype != torch.bfloat16)
           for p in model.hf_model.parameters()):
        raise ValueError('all native model parameters must be CUDA BF16')
    for value, name in ((head.dtype, 'head_dtype'), (embedding.dtype, 'embedding_dtype')):
        logical._equal(str(value), bindings['backend'][name], 'actual dtype differs from parent')
    attention = getattr(model.hf_model.config, '_attn_implementation', None)
    logical._equal(attention, bindings['backend']['attention_implementation'], 'attention policy differs')
    reference = parent.generation_for(parent.plan.keys[0])
    capability = compile_smoke_grammar(model.tokenizer, head.shape[0],
                                       reference.provenance['stop_token_ids'], policy.channel_policy)
    check_capability(capability, reference.provenance)
    wrapper_record = bindings['template']['actual_wrapper_record']
    wrapper = WrapperPolicy(**{k: v for k, v in wrapper_record.items() if k != 'tokenizer_chat_template'})
    members = {m.ticker: m for m in inputs.members}
    prompts = {}
    for key in parent.plan.keys:
        prompt = render_decision_prompt(model.tokenizer, members[key.ticker],
            inputs.pair_for(key.ticker, key.condition, key.trial_id), wrapper_policy=wrapper)
        check_prompt(prompt, bindings)
        logical._equal(prompt.schema_sha256, parent.plan.identity.schema_sha256, 'prompt schema differs')
        expected = parent.generation_for(key)
        if expected.failure_type is None:
            _bind_expected(expected, capability, policy, 'parent')
        prompts[key] = prompt
    metadata['backend'].update(device=str(device), head_dtype=str(head.dtype),
        embedding_dtype=str(embedding.dtype), requested_dtype=requested_dtype, use_cache=policy.use_cache,
        attention_implementation=attention, gpu_name=torch.cuda.get_device_name(embedding.device),
        physical_gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,driver_version',
            '--format=csv,noheader'], text=True).strip(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    bindings_record = dict(runtime=metadata, template=bindings['template'],
        generation_policy=bindings['generation_policy'], grammar=reference.provenance,
        model_layer_authentication=identity, issuer_by_ticker=inputs.issuer_by_ticker)
    return SimpleNamespace(inputs=inputs, parent=parent, table=table, policy=policy, model=model,
        count=count, embedding=embedding, capability=capability, prompts=prompts,
        bindings_record=bindings_record)


MODEL_COUNTS = {'glm4-9b-0414': 40, 'qwen3.5-4b': 32, 'gemma4-12b-it': 48, 'gpt-oss-20b': 24}
EXECUTION_VERSION = 'candidate_grouped_pair_v2'


def shard_panel(layers, index, shards):
    if any(type(v) is not int for v in (index, shards)) or not 0 <= index < shards <= len(layers):
        raise ValueError('require 0 <= shard index < num shards <= panel size')
    return tuple(layer for order, layer in enumerate(layers) if order % shards == index)


def descriptor(table, panel, index, shards, bindings):
    desc = dict(kind='stance_localization_candidates_v2', research_eligible=False,
        parent_sha256=table.parent_sha256, pair_table_sha256=table.table_sha256,
        inputs_manifest_sha256=table.inputs_manifest_sha256, phase='primary',
        execution_roles=list(ROLES), full_table_roles=['fit', 'validation', 'calibration', 'evaluation'],
        hook_site='post', actual_layer_count=panel.actual_layer_count,
        layers=list(shard_panel(panel.layers, index, shards)), arms=[[s, 'full'] for s in SPANS],
        mapping_rule='exact_tokens_if_complete_span_equal_else_relative_rank',
        shard_index=index, num_shards=shards, shard_rule='panel_order_modulo',
        candidate_panel=panel.to_dict(), panel_sha256=panel.panel_sha256,
        global_planned_cells=panel.development_cell_count, bindings=bindings,
        execution_version=EXECUTION_VERSION)
    return desc | dict(logical_grid_sha256=sha256_json(grouped.logical_descriptor(desc)))


def cells(table, desc):
    for _, key, pair in original_cells(table, desc):
        key = key | dict(logical_version='stance_localization_candidates_v2', panel_sha256=desc['panel_sha256'])
        yield sha256_json(key) + '.json', key, pair


def _plan(table, desc):
    panel = build_localization_candidate_panel(model_slug=desc['candidate_panel']['model_slug'],
                                              actual_layer_count=desc['actual_layer_count'])
    _equal(desc, descriptor(table, panel, desc['shard_index'], desc['num_shards'], desc['bindings']),
           'candidate descriptor differs')
    plan = {name: (key, pair) for name, key, pair in cells(table, desc)}
    gate_key = min(p.target_key for p in table.pairs if p.role == 'fit')
    gates = {f'gate_{layer}_{span}.json': (layer, span) for layer in desc['layers'] for span in SPANS}
    return plan, gate_key, gates


def execute_run(args, desc, table, prompts, parent, run_gate, run_group):
    registration = dict(descriptor=desc, descriptor_sha256=sha256_json(desc))
    with recovery_store(args.output_dir, registration) as records:
        report = run_grid(records, table, desc, prompts, parent, run_gate, run_group)
        summary = args.output_dir / 'summary.json'
        if summary.exists() or summary.is_symlink():
            _equal(read_file(summary), report, 'summary differs')
        else:
            publish(args.output_dir, 'summary.json', report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return 0


def run_grid(records, table, desc, prompts, parent, run_gate, run_group, *, snapshot=None):
    """Validate all records, run true gates, then group only remaining cells."""
    if snapshot is not None:
        raise ValueError('candidate runs prohibit prior migration')
    plan, gate_key, gates = _plan(table, desc)
    config_hash = sha256_json(desc)
    saved, passed = {}, {}
    halted = []
    summary_path = records.parent / 'summary.json'
    summary = read_file(summary_path) if summary_path.exists() or summary_path.is_symlink() else None
    for path in sorted(records.iterdir()):
        item = read_file(path)
        if path.name in gates:
            layer, _ = gates[path.name]
            passed[path.name] = import_gate(item, gate_key, parent, layer, config_hash)
        elif path.name in plan:
            key, pair = plan[path.name]
            mapping = alignment_for(pair, prompts, 'primary', key['span'], key['selector'])
            saved[path.name] = import_cell(item, key, pair, mapping, parent, config_hash)
        elif path.name.startswith('halt_'):
            grouped._validate_halt(item, path.name, plan, parent, desc)
            halted.append(item)
        else:
            raise ValueError('unexpected record, foreign key or staging file')
    if summary is not None and (set(saved) != set(plan) or set(passed) != set(gates)):
        raise ValueError('summary with incomplete gate/cell coverage')
    for execution in passed.values():
        require_gate(execution, parent.generation_for(gate_key))
    for name, execution in saved.items():
        if execution is not None and not execution.executed:
            raise RuntimeError('recorded halted cell')
        key, _ = plan[name]
        if f"gate_{key['layer']}_{key['span']}.json" not in passed:
            raise ValueError('cell has no recorded current phase gate')
    for item in halted:
        for key in item['keys']:
            if f"gate_{key['layer']}_{key['span']}.json" not in passed:
                raise ValueError('halted group missing current phase gate')
            if sha256_json(key) + '.json' in saved:
                raise ValueError('halted coordinate also has cell execution')
    if halted:
        raise RuntimeError('recorded halted group: ' + halted[0]['status'])
    for name, (layer, span) in gates.items():
        if name not in passed:
            execution = run_gate(gate_key, layer, span, config_hash)
            publish(records, name, gate_record(gate_key, parent.generation_for(gate_key), execution, config_hash))
            passed[name] = import_gate(read_file(records / name), gate_key, parent, layer, config_hash)
            require_gate(passed[name], parent.generation_for(gate_key))
    pending = defaultdict(list)
    for name, (key, pair) in plan.items():
        if name not in saved:
            pending[pair.pair_sha256].append((name, key, pair))
    for entries in pending.values():
        pair = entries[0][2]
        coordinates = tuple(ReplacementCell(key['layer'], 'post', alignment_for(
            pair, prompts, 'primary', key['span'], key['selector'])) for _, key, _ in entries)
        if pair.clean_relation == 'invalid_parent':
            executions = (None,) * len(entries)
        else:
            result = run_group(pair, coordinates)
            if result.cells != coordinates:
                raise ValueError('group coordinates differ')
            if result.status != 'executed':
                if result.executions:
                    raise ValueError('halted group has fabricated cell executions')
                item = dict(config_hash=config_hash, pair=pair.to_dict(),
                    keys=[key for _, key, _ in entries], status=result.status,
                    donor=result.donor.to_dict(), target_clean=None if result.target_clean is None
                    else result.target_clean.to_dict())
                name = 'halt_' + pair.pair_sha256 + '.json'
                item = json.loads(canonical_json_bytes(item))
                grouped._validate_halt(item, name, plan, parent, desc)
                publish(records, name, item)
                raise RuntimeError('halted group: ' + result.status)
            executions = result.executions
            if len(executions) != len(entries):
                raise ValueError('group execution coverage differs')
        for (name, key, pair), execution in zip(entries, executions, strict=True):
            item = json.loads(canonical_json_bytes(cell_record(key, pair, execution, config_hash)))
            mapping = alignment_for(pair, prompts, 'primary', key['span'], key['selector'])
            import_cell(item, key, pair, mapping, parent, config_hash)
            publish(records, name, item)
            saved[name] = import_cell(read_file(records / name), key, pair, mapping, parent, config_hash)
    if set(saved) != set(plan) or set(passed) != set(gates):
        raise ValueError('incomplete gate/cell coverage')
    counts = Counter(planned=len(plan), executed=0, invalid_parent=0, halted=0, failure=0)
    groups = defaultdict(Counter)
    origins = Counter(current=0, prior=0)
    for name, (key, pair) in plan.items():
        execution = saved[name]
        origins['prior' if 'original_record' in read_file(records / name) else 'current'] += 1
        counts['invalid_parent' if execution is None else 'executed'] += 1
        failure = execution is not None and execution.intervention.failure_type is not None
        counts['failure'] += int(failure)
        group = groups[(pair.role, pair.family, pair.contrast, key['span'], key['selector'], key['layer'])]
        group['planned'] += 1
        group['invalid_parent'] += int(execution is None)
        group['failure'] += int(failure)
        group['eligible'] += int(pair.clean_relation == 'opposite')
        group['flip'] += int(execution is not None and pair.clean_relation == 'opposite'
            and execution.intervention.failure_type is None
            and execution.intervention.decision == execution.expected_donor.decision)
    report = dict(kind=desc['kind'], config_hash=config_hash, execution_version=EXECUTION_VERSION,
        logical_grid_sha256=desc['logical_grid_sha256'], research_eligible=False,
        complete_shard=True, complete_global=desc['num_shards'] == 1, phase='primary',
        counts=dict(counts), origins=dict(origins), prior_source=desc.get('prior_source'),
        gates=len(passed), groups=[dict(role=k[0], family=k[1], contrast=k[2], span=k[3],
            selector=k[4], layer=k[5], **dict(v)) for k, v in sorted(groups.items())])
    report['candidate_panel_complete'] = desc['num_shards'] == 1
    report['global_planned_cells'] = desc['global_planned_cells']
    report['objective'] = desc['candidate_panel']['objective']
    report['directional_groups'] = directional_summary(plan, saved)
    report['company_issuer_groups'] = company_issuer_summary(plan, saved, desc['bindings'].get('issuer_by_ticker', {}))
    if summary is not None:
        _equal(summary, report, 'summary differs')
    return report


def directional_summary(plan, saved):
    """Both source classes remain visible, including NA fixed denominators."""
    groups = defaultdict(lambda: {d: Counter(eligible=0, flip=0, failure=0) for d in ('buy', 'sell')})
    for name, (key, pair) in plan.items():
        group = groups[(pair.role, pair.family, pair.contrast, key['span'], key['layer'])]
        execution = saved[name]
        if pair.clean_relation != 'opposite' or execution is None:
            continue
        stats = group[execution.expected_donor.decision]
        stats['eligible'] += 1
        failed = execution.intervention.failure_type is not None
        stats['failure'] += int(failed)
        stats['flip'] += int(not failed and execution.intervention.decision == execution.expected_donor.decision)
    return [dict(role=k[0], family=k[1], contrast=k[2], span=k[3], layer=k[4], source=d,
                 **dict(v), itt=None if not v['eligible'] else v['flip'] / v['eligible'])
            for k, values in sorted(groups.items()) for d, v in values.items()]


def company_issuer_summary(plan, saved, issuers):
    """Company-first ITT, then mean companies within issuer, then mean issuers."""
    groups = defaultdict(lambda: defaultdict(lambda: Counter(eligible=0, flip=0)))
    for name, (key, pair) in plan.items():
        for source in ('buy', 'sell'):
            groups[(pair.role, pair.family, pair.contrast, key['span'], key['layer'], source)]
        execution = saved[name]
        if pair.clean_relation != 'opposite' or execution is None:
            continue
        group = groups[(pair.role, pair.family, pair.contrast, key['span'], key['layer'],
                        execution.expected_donor.decision)]
        stats = group[pair.target_key.ticker]
        stats['eligible'] += 1
        stats['flip'] += int(execution.intervention.failure_type is None and
                             execution.intervention.decision == execution.expected_donor.decision)
    result = []
    for k, companies in sorted(groups.items()):
        scores = {ticker: v['flip'] / v['eligible'] for ticker, v in companies.items()}
        issuer_scores = defaultdict(list)
        for ticker, score in scores.items():
            if ticker not in issuers:
                raise ValueError('missing planned company issuer binding')
            issuer_scores[issuers[ticker]].append(score)
        means = [sum(v) / len(v) for v in issuer_scores.values()]
        result.append(dict(role=k[0], family=k[1], contrast=k[2], span=k[3], layer=k[4], source=k[5],
            companies=len(scores), issuers=len(means),
            company_first_itt=None if not scores else sum(scores.values()) / len(scores),
            issuer_balanced_itt=None if not means else sum(means) / len(means)))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(canonical_json_bytes(dict(kind='localization_candidates_failure', research_eligible=False,
            error_type=type(exc).__name__, message=str(exc))).decode(), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
