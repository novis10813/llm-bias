"""Native Harmony primary localization from the effective original/recovery parent.

Only the approved complete cohort and layer-modulo shards are public. No prior
localization import, generation override, prefix forcing or analysis removal.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts import run_stance_localization as logical
from scripts import run_stance_localization_grouped as grouped
from scripts.run_stance_baseline import runtime_metadata
from scripts.smoke_stance_checkpoint import generation_adapter, compile_smoke_grammar
from scripts.recover_stance_baseline_truncations import (
    check_capability, check_prompt, bind_relocated_tokenizer,
)
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_merged import load_merged_baseline
from llm_bias.core.stance_localization_pairs import build_localization_pairs
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy
from llm_bias.core.inference.stance_localization_execution import _bind_expected
from llm_bias.core.inference.stance_localization_harmony_grouped import execute_harmony_grouped_prompt_replacement
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop


def parser():
    p = logical.parser()
    next(a for a in p._actions if a.dest == 'phase').choices = ('primary',)
    p.add_argument('--recovery', required=True, type=Path)
    return p


def config_identity(checkpoint):
    raw = (checkpoint / 'config.json').read_bytes()
    config = json.loads(raw)
    text = config.get('text_config', config)
    family = text.get('model_type', config.get('model_type'))
    if family != 'gpt_oss':
        raise ValueError('require GPT-OSS checkpoint config')
    count = text.get('num_hidden_layers')
    if type(count) is not int or count <= 0:
        raise ValueError('config requires a positive integer text layer count')
    # Native quantized configs may omit floating dtype declarations entirely.
    # Absence is provenance, not permission to infer a dtype or cast weights.
    dtype, dtype_source = None, 'absent'
    scopes = [('text_config', text), ('config', config)] if text is not config else [('config', config)]
    for scope, record in scopes:
        for name in ('dtype', 'torch_dtype'):
            if name not in record:
                continue
            value = record[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'checkpoint {scope}.{name} must be a nonempty dtype string')
            if dtype_source == 'absent':
                dtype, dtype_source = value, f'{scope}.{name}'
    quantization = config.get('quantization_config', {})
    if not isinstance(quantization, dict) or quantization.get('quant_method') != 'mxfp4':
        raise ValueError('require native MXFP4 checkpoint representation')
    return dict(model_type=family, config_sha256=sha256_bytes(raw),
                configured_layer_count=count, declared_dtype=dtype,
                declared_dtype_source=dtype_source, quantization_config=quantization)


def authenticate_layers(model, identity):
    config = model.hf_model.config
    text = config.get_text_config() if callable(getattr(config, 'get_text_config', None)) else config
    logical._equal(getattr(text, 'num_hidden_layers', None), identity['configured_layer_count'],
                   'native config depth differs')
    logical._equal(getattr(text, 'model_type', None), identity['model_type'], 'native config family differs')
    count = identity['configured_layer_count']
    if len(model.layers) != count or len({id(layer) for layer in model.layers}) != count:
        raise ValueError('native wrapper must expose every distinct configured text layer')
    # Verify the inventory against HF itself, not merely a same-length wrapper.
    native_layers = model.hf_model.model.layers
    if len(native_layers) != count or any(a is not b for a, b in zip(model.layers, native_layers, strict=True)):
        raise ValueError('wrapper inventory differs from actual HF text layers')
    return count


def validate_parent(table, parent):
    if len(table.pairs) != 4024 or sum(p.role in logical.ROLES for p in table.pairs) != 3016:
        raise ValueError('require full 4024-pair / 3016-development grid')
    if (len(parent.plan.keys) != 2012 or len(parent.rows) != 2012
            or parent.summary['class_counts'] != {'buy': 518, 'sell': 1494}):
        raise ValueError('require effective 2012 parent with 518buy/1494sell')
    bindings = parent.metadata['original_metadata']['bindings']
    if bindings['backend']['requested_dtype'] != 'native':
        raise ValueError('require native parent representation')
    policies, inventory = {}, []
    for key in parent.plan.keys:
        policy = StructuredGenerationPolicy(**parent.row_policy_for(key))
        source = parent.row_source_for(key)
        budget = (1024, 300) if source['origin'] == 'original' else (4096, 1200)
        if (source['origin'] not in ('original', 'recovery')
                or (policy.max_new_tokens, policy.timeout_seconds) != budget
                or policy.channel_policy != 'harmony_no_tools' or policy.use_cache is not True):
            raise ValueError('effective row source/policy differs from approved parent')
        logical._equal(asdict(policy), parent.generation_for(key).provenance['generation_policy'],
                       'effective policy differs from generation provenance')
        policies[key] = policy
        inventory.append(dict(key=key.to_dict(), source=source, policy=asdict(policy)))
    if Counter(r['source']['origin'] for r in inventory) != {'original': 1981, 'recovery': 31}:
        raise ValueError('require all 31 recovery rows')
    return bindings, policies, inventory


def bind_rows(parent, capability, policies):
    """Bind every effective policy and native contract before opening a store."""
    for key in parent.plan.keys:
        expected = parent.generation_for(key)
        check_capability(capability, expected.provenance)
        _bind_expected(expected, capability.json_capability, policies[key], 'parent')
        logical._equal(expected.provenance['channel_contract_sha256'], capability.contract_sha256,
                       'parent Harmony contract hash differs')
        logical._equal(expected.provenance['channel_policy_sha256'], sha256_json({
            'policy': 'harmony_no_tools', 'contract_sha256': capability.contract_sha256}),
            'parent Harmony channel policy hash differs')


def bind_runtime(checkpoint, bindings):
    metadata = runtime_metadata(checkpoint)
    logical._equal(metadata['model']['metadata_file_sha256'], bindings['model']['metadata_file_sha256'],
                   'checkpoint metadata differs from parent')
    for name in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
                 'cudnn', 'deterministic_algorithms', 'python'):
        logical._equal(metadata['backend'][name], bindings['backend'][name], 'parent backend differs: ' + name)
    for name in ('run_stance_localization_harmony.py', 'run_stance_localization_grouped.py',
                 'run_stance_localization.py', 'recover_stance_baseline_truncations.py'):
        path = ROOT / 'scripts' / name
        metadata['code']['source_sha256']['scripts/' + name] = sha256_bytes(path.read_bytes())
    return metadata


def callbacks(model, capability, prompts, parent, policies):
    embedding = model.hf_model.get_input_embeddings().weight

    def gate(key, layer, span, config_hash):
        prompt = prompts[key]
        record = getattr(prompt, span + '_span')
        return execute_prompt_noop(model, model.tokenizer,
            torch.tensor([prompt.inference_token_ids], device=embedding.device, dtype=torch.long),
            capability, policy=policies[key], layer=layer, hook_site='post',
            zero_vector=torch.ones(embedding.shape[1], device=embedding.device, dtype=embedding.dtype),
            prompt_positions=list(range(record.token_start, record.token_end)), config_hash=config_hash)

    def group(pair, coordinates):
        return execute_harmony_grouped_prompt_replacement(model, model.tokenizer,
            prompts[pair.donor_key], prompts[pair.target_key], capability,
            donor_policy=policies[pair.donor_key], target_policy=policies[pair.target_key],
            cells=coordinates, expected_donor=parent.generation_for(pair.donor_key),
            expected_target=parent.generation_for(pair.target_key))

    return gate, group


def run(args):
    if args.phase != 'primary' or getattr(args, 'prior_run', None) is not None:
        raise ValueError('only fresh primary execution, no prior migration')
    inputs = load_baseline_inputs(args.inputs)
    parent = load_merged_baseline(args.parent, args.recovery, inputs=inputs)
    table = build_localization_pairs(inputs, parent)
    bindings, policies, inventory = validate_parent(table, parent)
    checkpoint = args.model.resolve(strict=True)
    identity = config_identity(checkpoint)
    metadata = bind_runtime(checkpoint, bindings)
    logical._equal(identity['config_sha256'], metadata['model']['metadata_file_sha256'].get('config.json'),
                   'config bytes changed during preflight')
    logical.shard_layers(identity['configured_layer_count'], args.shard_index, args.num_shards)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    loaded, _, device = load_model(str(checkpoint), device_map=None, dtype='native')
    model = generation_adapter(loaded)
    model.hf_model.eval()
    count = authenticate_layers(model, identity)
    metadata['tokenizer_relocation'] = bind_relocated_tokenizer(
        model.tokenizer, metadata['model'], bindings['model'])
    head = model.hf_model.get_output_embeddings().weight
    embedding = model.hf_model.get_input_embeddings().weight
    if (any(torch.device(d).type != 'cuda' for d in (device, head.device, embedding.device))
            or any(p.device.type != 'cuda' for p in model.hf_model.parameters())):
        raise ValueError('actual native model must be CUDA only, no CPU offload')
    for value, name in ((head.dtype, 'head_dtype'), (embedding.dtype, 'embedding_dtype')):
        logical._equal(str(value), bindings['backend'][name], 'actual dtype differs from parent')
    attention = getattr(model.hf_model.config, '_attn_implementation', None)
    logical._equal(attention, bindings['backend']['attention_implementation'], 'attention policy differs')
    reference = parent.generation_for(parent.plan.keys[0])
    capability = compile_smoke_grammar(model.tokenizer, head.shape[0],
                                       reference.provenance['stop_token_ids'], 'harmony_no_tools')
    bind_rows(parent, capability, policies)
    wrapper_record = bindings['template']['actual_wrapper_record']
    wrapper = WrapperPolicy(**{k: v for k, v in wrapper_record.items() if k != 'tokenizer_chat_template'})
    members = {m.ticker: m for m in inputs.members}
    prompts = {}
    for key in parent.plan.keys:
        prompt = render_decision_prompt(model.tokenizer, members[key.ticker],
            inputs.pair_for(key.ticker, key.condition, key.trial_id), wrapper_policy=wrapper)
        check_prompt(prompt, bindings)
        logical._equal(prompt.schema_sha256, parent.plan.identity.schema_sha256, 'prompt schema differs')
        prompts[key] = prompt
    metadata['backend'].update(device=str(device), head_dtype=str(head.dtype),
        embedding_dtype=str(embedding.dtype), requested_dtype='native', use_cache=True,
        attention_implementation=attention, gpu_name=torch.cuda.get_device_name(embedding.device),
        physical_gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,driver_version',
            '--format=csv,noheader'], text=True).strip(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    desc = grouped.descriptor(table, 'primary', count, args.shard_index, args.num_shards,
        dict(runtime=metadata, template=bindings['template'], grammar=reference.provenance,
             model_layer_authentication=identity,
             effective_parent=dict(content_sha256=parent.content_sha256, metadata=parent.metadata,
                 files=parent.file_sha256, plan=parent.plan.to_dict(), row_inventory=inventory)))
    gate, group = callbacks(model, capability, prompts, parent, policies)
    # Approved first fit key coincides with the accepted grouped development key.
    fit_key = min(p.target_key for p in table.pairs if p.role == 'fit')
    if fit_key != min(p.target_key for p in table.pairs if p.role in logical.ROLES):
        raise ValueError('accepted grouped gate key differs from first fixed fit key')
    return grouped.execute_run(args, desc, table, prompts, parent, gate, group, None)


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
