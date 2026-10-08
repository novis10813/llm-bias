"""Fixed independent native GLM fit discovery, with immutable layer shards."""
from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.run_stance_localization import (
    shard_layers, _equal, _provenance, runtime_metadata, generation_adapter,
    compile_smoke_grammar, check_capability, check_prompt, _bind_expected,
    load_model, render_decision_prompt, WrapperPolicy,
)
from scripts.recover_stance_baseline_truncations import recovery_store, publish, read_file
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline, _validated_generation_payload
from llm_bias.core.stance_baseline_store import _generation
from llm_bias.core.stance_baseline_adapter import baseline_gate_input
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy, generate_structured
from llm_bias.core.inference.stance_localization_execution import _same_full_output
from llm_bias.core.inference.stance_interventions import GenerationPositionTracker, scoped_mlp_addition
from llm_bias.core.stance_neuron_panel import (
    build_neuron_panel, NeuronDiscoveryPlan, NeuronRole, NeuronArm,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('model', 'inputs', 'parent', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    return p


def descriptor(plan, keys, index, shards, bindings):
    fit = plan.role_ids('discovery')
    if (len(keys) != len(fit) or {k.ticker for k in keys} != set(fit)
            or any(k.condition != '+-' for k in keys)):
        raise ValueError('require exact one mixed fit key per ticker')
    return dict(kind='stance_neuron_discovery_v1', research_eligible=False,
        plan=plan.to_dict(), plan_sha256=plan.plan_sha256,
        fit_keys=[k.to_dict() for k in sorted(keys)], doses=[-2, 2], scope='prompt_and_decode',
        layers=list(shard_layers(len(plan.panel.widths), index, shards)),
        shard_index=index, num_shards=shards, bindings=bindings,
        planned_global=sum(len(c) for c in plan.panel.coordinates) * len(keys) * 2)


def arm_for(panel, layer, neuron, dose):
    return NeuronArm(panel.panel_sha256, layer, neuron, dose, 'prompt_and_decode')


def cells(plan, keys, desc):
    for layer in desc['layers']:
        for neuron in plan.panel.coordinates[layer]:
            for dose in (-2, 2):
                arm = arm_for(plan.panel, layer, neuron, dose)
                for target in sorted(keys):
                    key = dict(panel_sha256=plan.panel.panel_sha256, layer=layer,
                        neuron=neuron, ticker=target.ticker, condition=target.condition,
                        trial_id=target.trial_id, dose=dose, scope=arm.scope)
                    yield sha256_json(key) + '.json', key, target, arm


def execute_native(model, tokenizer, ids, capability, policy, panel, arm=None):
    if arm is None:
        return generate_structured(model, tokenizer, ids, capability, policy=policy)
    # The runtime inspects the complete panel once. Re-ranking every native
    # width on all40layers per cell would repeat millions of hash operations.
    if (arm.panel_sha256 != panel.panel_sha256 or panel.status != 'supported'
            or arm.layer >= len(panel.widths) or arm.neuron not in panel.coordinates[arm.layer]):
        raise ValueError('native arm differs from inspected panel')
    tracker = GenerationPositionTracker(ids.shape[1], policy.use_cache)
    with tracker.track(model), scoped_mlp_addition(model, tracker=tracker,
            layer=arm.layer, neuron=arm.neuron, delta=arm.delta, scope=arm.scope):
        return generate_structured(model, tokenizer, ids, capability, policy=policy)


def execute_gate(generate, panel, layer):
    return (generate(), generate(), generate(arm_for(panel, layer, panel.coordinates[layer][0], 0)))


def envelope(payload):
    return dict(payload=payload, payload_sha256=sha256_json(payload))


def unpack(item):
    if set(item) != {'payload', 'payload_sha256'} or sha256_json(item['payload']) != item['payload_sha256']:
        raise ValueError('record payload hash differs')
    return item['payload']


def validate_generation(value, target, expected):
    generation = _generation(value)
    baseline_gate_input(generation)
    _validated_generation_payload(target, generation)
    _provenance(generation, expected)
    _equal(generation.provenance['model_binding'], expected.provenance['model_binding'],
           'generation model binding differs')
    return generation


def run_grid(records, plan, keys, desc, parent, run_gate, run_cell):
    config_hash = sha256_json(desc)
    planned = {name: (key, target, arm) for name, key, target, arm in cells(plan, keys, desc)}
    gates = {f'gate_{layer}.json': layer for layer in desc['layers']}
    gate_key = min(keys)
    expected_gate = parent.generation_for(gate_key)
    saved, saved_gates = {}, {}
    summary_path = records.parent / 'summary.json'
    summary = read_file(summary_path) if summary_path.exists() or summary_path.is_symlink() else None

    def import_gate(item, layer):
        payload = unpack(item)
        generations = tuple(validate_generation(v, gate_key, expected_gate) for v in payload['generations'])
        _equal(payload, dict(config_hash=config_hash, layer=layer, key=gate_key.to_dict(),
            expected=expected_gate.to_dict(), generations=[g.to_dict() for g in generations]),
            'gate record differs')
        if len(generations) != 3:
            raise ValueError('gate requires baseline/repeat/native zero')
        return generations

    def import_cell(item, key, target, arm):
        payload = unpack(item)
        expected = parent.generation_for(target)
        generation = validate_generation(payload['generation'], target, expected)
        _equal(payload, dict(config_hash=config_hash, key=key, expected=expected.to_dict(),
            generation=generation.to_dict(), native_hook=dict(layer=arm.layer, neuron=arm.neuron,
                hook_site='mlp_down_proj_input', scope=arm.scope, operation='addition')),
            'cell record differs')
        return generation

    for path in sorted(records.iterdir()):
        if path.name in planned:
            saved[path.name] = import_cell(read_file(path), *planned[path.name])
        elif path.name in gates:
            saved_gates[path.name] = import_gate(read_file(path), gates[path.name])
        else:
            raise ValueError('unexpected record or staging leftover')
    if summary is not None and (set(saved) != set(planned) or set(saved_gates) != set(gates)):
        raise ValueError('summary with incomplete coverage')
    for name in saved:
        if f"gate_{planned[name][0]['layer']}.json" not in saved_gates:
            raise ValueError('cell missing layer gate')

    def require_gate(values):
        if not all(g.failure_type is None and _same_full_output(g, expected_gate) for g in values):
            raise RuntimeError('native zero/repeat/parent gate failed')
    for values in saved_gates.values():
        require_gate(values)
    print(canonical_json_bytes(dict(planned=len(planned), executed=len(saved),
        missing=len(planned)-len(saved), research_eligible=False)).decode(), flush=True)
    for name, layer in gates.items():
        if name not in saved_gates:
            values = run_gate(gate_key, layer)
            publish(records, name, envelope(dict(config_hash=config_hash, layer=layer,
                key=gate_key.to_dict(), expected=expected_gate.to_dict(),
                generations=[g.to_dict() for g in values])))
            saved_gates[name] = import_gate(read_file(records / name), layer)
        require_gate(saved_gates[name])
    groups = defaultdict(Counter)
    elapsed = sum(g.elapsed_seconds for name in gates for g in saved_gates[name])
    for name, (key, target, arm) in planned.items():
        if name not in saved:
            generation = run_cell(target, arm)
            publish(records, name, envelope(dict(config_hash=config_hash, key=key,
                expected=parent.generation_for(target).to_dict(), generation=generation.to_dict(),
                native_hook=dict(layer=arm.layer, neuron=arm.neuron, hook_site='mlp_down_proj_input',
                                 scope=arm.scope, operation='addition'))))
            saved[name] = import_cell(read_file(records / name), key, target, arm)
            print(canonical_json_bytes(dict(executed=len(saved), missing=len(planned)-len(saved),
                key=key, failure_type=saved[name].failure_type)).decode(), flush=True)
        actual, expected = saved[name], parent.generation_for(target)
        elapsed += actual.elapsed_seconds
        group = groups[(arm.layer, arm.neuron, arm.delta)]
        group['planned'] += 1
        group['baseline_' + (expected.decision if expected.failure_type is None else 'unknown')] += 1
        group['schema_complete'] += int(actual.schema_complete)
        group['failure'] += int(actual.failure_type is not None)
        group['sell_denominator'] += int(expected.failure_type is None and expected.decision == 'sell')
        group['buy_denominator'] += int(expected.failure_type is None and expected.decision == 'buy')
        flip = (actual.failure_type is None and expected.failure_type is None
                and actual.decision != expected.decision)
        group['any_flip'] += int(flip)
        group['sell_to_buy'] += int(flip and expected.decision == 'sell')
        group['buy_to_sell'] += int(flip and expected.decision == 'buy')
    if set(saved) != set(planned) or set(saved_gates) != set(gates):
        raise ValueError('incomplete native coverage')
    rows = []
    for (layer, neuron, dose), group in sorted(groups.items()):
        for field in ('baseline_buy', 'baseline_sell', 'baseline_unknown'):
            group.setdefault(field, 0)
        rows.append(dict(layer=layer, neuron=neuron, dose=dose, **dict(group),
            sell_to_buy_rate=group['sell_to_buy']/group['sell_denominator'] if group['sell_denominator'] else None,
            buy_to_sell_rate=group['buy_to_sell']/group['buy_denominator'] if group['buy_denominator'] else None))
    report = dict(kind=desc['kind'], config_hash=config_hash, research_eligible=False,
        complete_shard=True, complete_global=desc['num_shards'] == 1,
        planned=len(planned), planned_global=desc['planned_global'], executed=len(saved), missing=0,
        failure=sum(g.failure_type is not None for g in saved.values()), gates=len(saved_gates),
        recorded_generation_seconds=elapsed, groups=rows, native_coverage=[list(c) for c in plan.panel.coverage])
    if summary is not None:
        _equal(summary, report, 'summary differs')
    return report


def run(args):
    inputs = load_baseline_inputs(args.inputs)
    parent = load_completed_baseline(args.parent, inputs=inputs)
    bindings = parent.metadata['bindings']
    policy = StructuredGenerationPolicy(**bindings['generation_policy']['policy'])
    if policy.channel_policy != 'plain_json':
        raise ValueError('native discovery runner requires plain_json')
    checkpoint = args.model.resolve(strict=True)
    metadata = runtime_metadata(checkpoint)
    _equal(metadata['model'], bindings['model'], 'original checkpoint path/metadata differs')
    for name in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
                 'cudnn', 'deterministic_algorithms'):
        _equal(metadata['backend'][name], bindings['backend'][name],
               'parent backend differs: ' + name)
    metadata['code']['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha256_bytes(Path(__file__).read_bytes())
    for source in ('scripts/run_stance_localization.py',
                   'scripts/recover_stance_baseline_truncations.py',
                   'llm_bias/core/stance_neuron_panel.py'):
        metadata['code']['source_sha256'][source] = sha256_bytes((ROOT / source).read_bytes())
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
    if len(model.layers) != 40 or model.hf_model.config.model_type != 'glm4':
        raise ValueError('requires actual 40-layer dense GLM')
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
    if (policy.max_new_tokens, policy.timeout_seconds, policy.use_cache) != (512, 180, True):
        raise ValueError('requires original GLM 512/180/cache generation policy')
    panel = build_neuron_panel(model)
    if panel.status != 'supported' or any(len(c) != 16 for c in panel.coordinates):
        raise ValueError('requires full independent16 dense panel')
    assignments = {ticker: role for role, tickers in inputs.roles['roles'].items() for ticker in tickers}
    roles = tuple(NeuronRole(m.ticker, m.issuer_id, assignments[m.ticker]) for m in inputs.members)
    protocol = dict(kind='native_glm_fit_discovery_v1', condition='+-', doses=[-2, 2],
                    scope='prompt_and_decode', seed=20261003, per_layer=16, research_eligible=False)
    plan = NeuronDiscoveryPlan(panel, roles, inputs.manifest_sha256, parent.parent_sha256,
        parent.plan.identity.model_sha256, sha256_json(protocol))
    keys = tuple(k for k in parent.plan.keys if k.ticker in plan.role_ids('discovery') and k.condition == '+-')
    if len(keys) != 302:
        raise ValueError('requires exact 302 fit tickers')
    desc = descriptor(plan, keys, args.shard_index, args.num_shards,
        dict(runtime=metadata, template=bindings['template'], generation_policy=bindings['generation_policy'],
             grammar=reference.provenance, protocol=protocol))
    def generate(key, arm=None):
        ids = torch.tensor([prompts[key].inference_token_ids], device=embedding.device, dtype=torch.long)
        return execute_native(model, tokenizer, ids, capability, policy, panel, arm)
    def gate(key, layer):
        return execute_gate(lambda arm=None: generate(key, arm), panel, layer)
    registration = dict(descriptor=desc, descriptor_sha256=sha256_json(desc))
    with recovery_store(args.output_dir, registration) as records:
        report = run_grid(records, plan, keys, desc, parent, gate, generate)
        if not (args.output_dir / 'summary.json').exists():
            publish(args.output_dir, 'summary.json', report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except Exception as exc:
        print(canonical_json_bytes(dict(kind='neuron_discovery_failure', research_eligible=False,
            error_type=type(exc).__name__, message=str(exc))).decode(), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
