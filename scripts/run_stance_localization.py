"""Complete development localization grids, with immutable layer-modulo shards.

Internal helpers accept tiny fake tables for tests. The public entry always loads
validated full inputs and a complete baseline before loading a checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
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
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline
from llm_bias.core.stance_baseline_store import _generation
from llm_bias.core.stance_localization_pairs import build_localization_pairs
from llm_bias.core.stance_localization_alignment import build_span_alignment
from llm_bias.core.stance_gates import NoOpGateResult
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy
from llm_bias.core.inference.stance_localization_execution import (
    PromptReplacementExecution, ReplacementMatchChecks, execute_prompt_replacement,
    _same_full_output, _bind_expected,
)
from llm_bias.core.inference.stance_noop_execution import PromptNoOpExecution, execute_prompt_noop

SPANS = ('entity', 'evidence1', 'evidence2', 'instruction')
ROLES = ('fit', 'validation')
STAGES = {
    'primary': tuple((span, 'full') for span in SPANS),
    'position': tuple((span, selector) for span in SPANS
                      for selector in ('first', 'middle', 'last', 'tail4')),
    'alignment_sensitivity': (('entity', 'full'),),
}
PROVENANCE_KEYS = ('schema_sha256', 'schema_bytes_sha256', 'tokenizer_sha256',
                   'tokenizer_info_sha256', 'head_vocab_size', 'grammar_sha256',
                   'stop_token_ids', 'generation_policy', 'hf_controls',
                   'generation_policy_sha256')


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('model', 'inputs', 'parent', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--phase', choices=tuple(STAGES), default='primary')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    return p


def shard_layers(count, index, shards):
    if (any(type(v) is not int for v in (count, index, shards))
            or not 0 <= index < shards <= count):
        raise ValueError('require 0 <= shard index < num shards <= actual layer count')
    return tuple(layer for layer in range(count) if layer % shards == index)


def descriptor(table, phase, count, index, shards, bindings):
    layers = shard_layers(count, index, shards)
    return dict(kind='stance_localization_v1', research_eligible=False,
        parent_sha256=table.parent_sha256, pair_table_sha256=table.table_sha256,
        inputs_manifest_sha256=table.inputs_manifest_sha256,
        phase=phase, execution_roles=list(ROLES), full_table_roles=[
            'fit', 'validation', 'calibration', 'evaluation'], hook_site='post',
        actual_layer_count=count, layers=list(layers), arms=[list(a) for a in STAGES[phase]],
        mapping_rule=('tail_overlap' if phase == 'alignment_sensitivity'
                      else 'exact_tokens_if_complete_span_equal_else_relative_rank'),
        shard_index=index, num_shards=shards, bindings=bindings)


def cell_key(pair, layer, span, selector):
    return dict(pair_sha256=pair.pair_sha256, layer=layer, span=span, selector=selector)


def cells(table, desc):
    for layer in desc['layers']:
        for span, selector in STAGES[desc['phase']]:
            for pair in table.pairs:
                if pair.role in ROLES:
                    key = cell_key(pair, layer, span, selector)
                    yield sha256_json(key) + '.json', key, pair


def alignment_for(pair, prompts, phase, span, selector):
    donor, target = prompts[pair.donor_key], prompts[pair.target_key]
    equal = getattr(donor, span + '_span').token_ids == getattr(target, span + '_span').token_ids
    policy = ('tail_overlap' if phase == 'alignment_sensitivity'
              else 'exact_tokens' if equal else 'relative_rank')
    return build_span_alignment(donor, target, span=span, policy=policy, selector=selector)


def _equal(a, b, message):
    if canonical_json_bytes(a) != canonical_json_bytes(b):
        raise ValueError(message)


def _provenance(generation, reference):
    for key in PROVENANCE_KEYS:
        _equal(generation.provenance.get(key), reference.provenance.get(key),
               'generation configuration differs: ' + key)


def _coordinates(diagnostics, layer, operation):
    if diagnostics is not None and (diagnostics['layer'], diagnostics['hook_site'],
            diagnostics['scope'], diagnostics['operation']) != (layer, 'post', 'prompt_only', operation):
        raise ValueError('diagnostic coordinates differ')


def cell_record(key, pair, execution, config_hash):
    return dict(key=key, pair=pair.to_dict(), config_hash=config_hash,
                execution=None if execution is None else execution.to_dict(),
                status='invalid_parent' if execution is None else execution.status)


def import_cell(item, key, pair, mapping, parent, config_hash):
    if pair.clean_relation == 'invalid_parent':
        expected = cell_record(key, pair, None, config_hash)
        _equal(item, expected, 'invalid parent cell differs')
        return None
    value = item['execution']
    result = lambda name: None if value[name] is None else _generation(value[name])
    execution = PromptReplacementExecution(value['status'], result('donor'),
        result('expected_donor'), result('target_clean'), result('expected_target'),
        result('intervention'), ReplacementMatchChecks(**value['match_checks']), mapping,
        None if value['diagnostics'] is None else canonical_json_bytes(value['diagnostics']))
    _equal(execution.expected_donor.to_dict(), parent.generation_for(pair.donor_key).to_dict(),
           'stored donor parent snapshot differs')
    _equal(execution.expected_target.to_dict(), parent.generation_for(pair.target_key).to_dict(),
           'stored target parent snapshot differs')
    for actual, reference in ((execution.donor, execution.expected_donor),
            (execution.target_clean, execution.expected_target),
            (execution.intervention, execution.expected_target)):
        if actual is not None:
            _provenance(actual, reference)
    _coordinates(execution.diagnostics, key['layer'], 'replacement')
    _equal(item, cell_record(key, pair, execution, config_hash), 'cell record differs')
    return execution


def gate_record(key, expected, execution, config_hash):
    return dict(key=key.to_dict(), expected=expected.to_dict(), config_hash=config_hash,
        baseline=execution.baseline.to_dict(),
        repeat=None if execution.repeat is None else execution.repeat.to_dict(),
        zero=None if execution.zero is None else execution.zero.to_dict(),
        self_replacement=None if execution.self_replacement is None else execution.self_replacement.to_dict(),
        gate=None if execution.gate is None else execution.gate.to_dict(),
        halt_reason=execution.halt_reason, zero_diagnostics=execution.zero_diagnostics,
        self_diagnostics=execution.self_diagnostics)


def import_gate(item, key, parent, layer, config_hash):
    expected = parent.generation_for(key)
    arm = lambda name: None if item[name] is None else _generation(item[name])
    gate = None if item['gate'] is None else NoOpGateResult(**item['gate'])
    execution = PromptNoOpExecution(arm('baseline'), arm('repeat'), arm('zero'),
        arm('self_replacement'), gate, item['halt_reason'],
        None if item['zero_diagnostics'] is None else canonical_json_bytes(item['zero_diagnostics']),
        None if item['self_diagnostics'] is None else canonical_json_bytes(item['self_diagnostics']))
    for actual in (execution.baseline, execution.repeat, execution.zero, execution.self_replacement):
        if actual is not None:
            _provenance(actual, expected)
    for diagnostics, operation in ((execution.zero_diagnostics, 'addition'),
                                   (execution.self_diagnostics, 'replacement')):
        _coordinates(diagnostics, layer, operation)
    if gate is not None and gate.config_hash != config_hash:
        raise ValueError('gate configuration differs')
    _equal(item, gate_record(key, expected, execution, config_hash), 'gate record differs')
    return execution


def require_gate(execution, expected):
    if not execution.passed or not _same_full_output(execution.baseline, expected):
        raise RuntimeError('phase no-op gate failed or baseline parent drift')
    if not all(_same_full_output(arm, expected) for arm in (
            execution.repeat, execution.zero, execution.self_replacement)):
        raise RuntimeError('phase full-output no-op mismatch')


def run_grid(records, table, desc, prompts, parent, run_gate, run_cell):
    """Re-import every saved record before executing anything. Halts never retry."""
    config_hash = sha256_json(desc)
    plan = {name: (key, pair) for name, key, pair in cells(table, desc)}
    gate_key = min(p.target_key for p in table.pairs if p.role in ROLES)
    gates = {f'gate_{layer}_{span}.json': (layer, span) for layer in desc['layers']
             for span in dict.fromkeys(span for span, _ in STAGES[desc['phase']])}
    saved_cells, saved_gates = {}, {}
    summary_path = records.parent / 'summary.json'
    saved_summary = (read_file(summary_path) if summary_path.exists() or summary_path.is_symlink()
                     else None)
    for path in sorted(records.iterdir()):
        if path.name not in plan and path.name not in gates:
            raise ValueError('unexpected record, foreign key or staging file')
        item = read_file(path)
        if path.name in plan:
            key, pair = plan[path.name]
            mapping = alignment_for(pair, prompts, desc['phase'], key['span'], key['selector'])
            saved_cells[path.name] = import_cell(item, key, pair, mapping, parent, config_hash)
        elif path.name in gates:
            layer, _ = gates[path.name]
            saved_gates[path.name] = import_gate(item, gate_key, parent, layer, config_hash)
        else:
            raise ValueError('unexpected record, foreign key or staging file')
    if saved_summary is not None and (set(saved_cells) != set(plan) or set(saved_gates) != set(gates)):
        raise ValueError('summary with incomplete gate/cell coverage')
    for name, execution in saved_gates.items():
        require_gate(execution, parent.generation_for(gate_key))
    for execution in saved_cells.values():
        if execution is not None and not execution.executed:
            raise RuntimeError('recorded halted cell: ' + execution.status)
    # Saved cells cannot precede their immutable passed gate.
    for name in saved_cells:
        key, _ = plan[name]
        if f"gate_{key['layer']}_{key['span']}.json" not in saved_gates:
            raise ValueError('cell has no recorded phase gate')
    groups = defaultdict(Counter)
    counts = Counter(planned=len(plan), executed=0, invalid_parent=0, halted=0, failure=0)
    for name, (key, pair) in plan.items():
        gate_name = f"gate_{key['layer']}_{key['span']}.json"
        if gate_name not in saved_gates:
            execution = run_gate(gate_key, key['layer'], key['span'], config_hash)
            item = gate_record(gate_key, parent.generation_for(gate_key), execution, config_hash)
            publish(records, gate_name, item)
            execution = import_gate(read_file(records / gate_name), gate_key, parent, key['layer'], config_hash)
            require_gate(execution, parent.generation_for(gate_key))
            saved_gates[gate_name] = execution
        mapping = alignment_for(pair, prompts, desc['phase'], key['span'], key['selector'])
        if name not in saved_cells:
            execution = None if pair.clean_relation == 'invalid_parent' else run_cell(pair, key, mapping)
            item = cell_record(key, pair, execution, config_hash)
            publish(records, name, item)
            saved_cells[name] = import_cell(read_file(records / name), key, pair, mapping, parent, config_hash)
        execution = saved_cells[name]
        if execution is not None and not execution.executed:
            raise RuntimeError('halted cell: ' + execution.status)
        counts['invalid_parent' if execution is None else 'executed'] += 1
        failure = execution is not None and execution.intervention.failure_type is not None
        counts['failure'] += int(failure)
        group = groups[(pair.role, pair.family, pair.contrast, key['span'], key['selector'], key['layer'])]
        group['planned'] += 1
        group['invalid_parent'] += int(execution is None)
        group['failure'] += int(failure)
        group['eligible'] += int(pair.clean_relation == 'opposite')
        flip = (execution is not None and pair.clean_relation == 'opposite'
                and execution.intervention.failure_type is None
                and execution.intervention.decision == execution.expected_donor.decision)
        group['flip'] += int(flip)
        print(canonical_json_bytes(dict(key=key, status='invalid_parent' if execution is None
            else execution.status, failure=bool(failure), flip=bool(flip))).decode(), flush=True)
    if set(saved_cells) != set(plan) or set(saved_gates) != set(gates):
        raise ValueError('incomplete gate/cell coverage')
    report = dict(kind=desc['kind'], config_hash=config_hash, research_eligible=False,
        complete_shard=True, complete_global=desc['num_shards'] == 1, phase=desc['phase'],
        counts=dict(counts), gates=len(saved_gates), groups=[dict(role=k[0], family=k[1],
            contrast=k[2], span=k[3], selector=k[4], layer=k[5], **dict(v))
            for k, v in sorted(groups.items())])
    if saved_summary is not None:
        _equal(saved_summary, report, 'summary differs')
    return report


def run(args):
    inputs = load_baseline_inputs(args.inputs)
    parent = load_completed_baseline(args.parent, inputs=inputs)
    table = build_localization_pairs(inputs, parent)
    bindings = parent.metadata['bindings']
    policy = StructuredGenerationPolicy(**bindings['generation_policy']['policy'])
    if policy.channel_policy != 'plain_json':
        raise ValueError('first localization GPU runner requires plain_json')
    checkpoint = args.model.resolve(strict=True)
    metadata = runtime_metadata(checkpoint)
    _equal(metadata['model'], bindings['model'], 'original checkpoint path/metadata differs')
    for name in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
                 'cudnn', 'deterministic_algorithms'):
        _equal(metadata['backend'][name], bindings['backend'][name],
               'parent backend differs: ' + name)
    metadata['code']['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha256_bytes(Path(__file__).read_bytes())
    metadata['code']['source_sha256']['scripts/recover_stance_baseline_truncations.py'] = sha256_bytes(
        (ROOT / 'scripts/recover_stance_baseline_truncations.py').read_bytes())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    loaded, _, device = load_model(str(checkpoint), device_map=None,
        dtype='native' if bindings['backend']['requested_dtype'] == 'native' else torch.bfloat16)
    model = generation_adapter(loaded)
    model.hf_model.eval()
    tokenizer = model.tokenizer
    head = model.hf_model.get_output_embeddings().weight
    embedding = model.hf_model.get_input_embeddings().weight
    if any(torch.device(d).type != 'cuda' for d in (device, head.device, embedding.device)):
        raise ValueError('actual model must be on CUDA')
    for value, name in ((head.dtype, 'head_dtype'), (embedding.dtype, 'embedding_dtype')):
        if str(value) != bindings['backend'][name]:
            raise ValueError('actual dtype differs from parent')
    _equal(getattr(model.hf_model.config, '_attn_implementation', None),
           bindings['backend']['attention_implementation'], 'attention policy differs')
    shard_layers(len(model.layers), args.shard_index, args.num_shards)
    reference = parent.generation_for(parent.plan.keys[0])
    capability = compile_smoke_grammar(tokenizer, head.shape[0], reference.provenance['stop_token_ids'],
                                       policy.channel_policy)
    check_capability(capability, reference.provenance)
    wrapper_record = bindings['template']['actual_wrapper_record']
    wrapper = WrapperPolicy(**{k: v for k, v in wrapper_record.items() if k != 'tokenizer_chat_template'})
    members = {m.ticker: m for m in inputs.members}
    prompts = {}
    for key in parent.plan.keys:
        prompt = render_decision_prompt(tokenizer, members[key.ticker],
            inputs.pair_for(key.ticker, key.condition, key.trial_id), wrapper_policy=wrapper)
        check_prompt(prompt, bindings)
        if prompt.schema_sha256 != parent.plan.identity.schema_sha256:
            raise ValueError('prompt schema differs from parent')
        expected = parent.generation_for(key)
        if expected.failure_type is None:
            _bind_expected(expected, capability, policy, 'parent')
        prompts[key] = prompt
    backend = metadata['backend']
    backend.update(device=str(device), head_dtype=str(head.dtype), embedding_dtype=str(embedding.dtype),
        requested_dtype=bindings['backend']['requested_dtype'], use_cache=policy.use_cache,
        attention_implementation=getattr(model.hf_model.config, '_attn_implementation', None),
        gpu_name=torch.cuda.get_device_name(embedding.device),
        physical_gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,driver_version',
                                              '--format=csv,noheader'], text=True).strip(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    desc = descriptor(table, args.phase, len(model.layers), args.shard_index, args.num_shards,
        dict(runtime=metadata, template=bindings['template'], generation_policy=bindings['generation_policy'],
             grammar=reference.provenance))
    def run_gate(key, layer, span, config_hash):
        prompt = prompts[key]
        record = getattr(prompt, span + '_span')
        ids = torch.tensor([prompt.inference_token_ids], device=embedding.device, dtype=torch.long)
        # NO1 requires a nonzero direction, then executes a genuine dose-zero hook.
        direction = torch.ones(embedding.shape[1], device=embedding.device, dtype=embedding.dtype)
        return execute_prompt_noop(model, tokenizer, ids, capability, policy=policy, layer=layer,
            hook_site='post', zero_vector=direction, prompt_positions=list(range(record.token_start,
                record.token_end)), config_hash=config_hash)
    def run_cell(pair, key, mapping):
        return execute_prompt_replacement(model, tokenizer, prompts[pair.donor_key], prompts[pair.target_key],
            capability, policy=policy, layer=key['layer'], hook_site='post', alignment=mapping,
            expected_donor=parent.generation_for(pair.donor_key), expected_target=parent.generation_for(pair.target_key))
    registration = dict(descriptor=desc, descriptor_sha256=sha256_json(desc))
    with recovery_store(args.output_dir, registration) as records:
        report = run_grid(records, table, desc, prompts, parent, run_gate, run_cell)
        summary = args.output_dir / 'summary.json'
        if summary.exists() or summary.is_symlink():
            _equal(read_file(summary), report, 'summary differs')
        else:
            publish(args.output_dir, 'summary.json', report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(canonical_json_bytes(dict(kind='localization_failure', research_eligible=False,
            error_type=type(exc).__name__, message=str(exc))).decode(), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
