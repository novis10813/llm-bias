"""Separate native-BF16 plain-JSON primary localization for Qwen3.5/Gemma4.

Compose the accepted grouped grid, executor, gates and immutable store. No
historical-result migration or public cohort/layer/span override is supported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts import run_stance_localization_grouped as grouped
from scripts import run_stance_localization as logical
from scripts.run_stance_baseline import runtime_metadata
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
    return p


def config_identity(checkpoint):
    """Read the checkpoint text depth, not a CLI default or model-slug heuristic."""
    raw = (checkpoint / 'config.json').read_bytes()
    config = json.loads(raw)
    text = config.get('text_config', config)
    family = text.get('model_type', config.get('model_type'))
    if family not in ('qwen3_5', 'qwen3_5_text', 'gemma4', 'gemma4_text',
                      'gemma4_unified_text'):
        raise ValueError('require Qwen3.5 or Gemma4 checkpoint config')
    if (family == 'gemma4_unified_text' or config.get('model_type') == 'gemma4_unified'):
        if family != 'gemma4_unified_text' or config.get('model_type') != 'gemma4_unified':
            raise ValueError('Gemma4 unified text requires matching gemma4_unified outer config')
    count = text.get('num_hidden_layers')
    if type(count) is not int or count <= 0:
        raise ValueError('config requires a positive integer text layer count')
    if family.startswith('qwen') and count != 32:
        raise ValueError('Qwen3.5-4B route requires 32 configured text layers')
    dtype = text.get('dtype', text.get('torch_dtype', config.get('dtype', config.get('torch_dtype'))))
    if dtype not in ('bfloat16', 'torch.bfloat16'):
        raise ValueError('checkpoint config must declare native BF16')
    identity = dict(model_type=family, config_sha256=sha256_bytes(raw),
                    configured_layer_count=count, declared_dtype=dtype)
    if family == 'gemma4_unified_text':
        identity['outer_model_type'] = config['model_type']
    return identity


def authenticate_layers(model, identity):
    config = model.hf_model.config
    if identity['model_type'] == 'gemma4_unified_text':
        logical._equal(getattr(config, 'model_type', None), identity['outer_model_type'],
                       'native config outer family differs')
    text = config.get_text_config() if callable(getattr(config, 'get_text_config', None)) else config
    count = getattr(text, 'num_hidden_layers', None)
    logical._equal(count, identity['configured_layer_count'], 'native config depth differs')
    logical._equal(getattr(text, 'model_type', None), identity['model_type'], 'native config family differs')
    if len(model.layers) != count or len({id(layer) for layer in model.layers}) != count:
        raise ValueError('native wrapper must expose every distinct configured text layer')
    return count


def validate_parent(table, bindings):
    if len(table.pairs) != 4024 or sum(p.role in logical.ROLES for p in table.pairs) != 3016:
        raise ValueError('require full 4024-pair / 3016-development grid')
    policy = StructuredGenerationPolicy(**bindings['generation_policy']['policy'])
    if (policy.channel_policy != 'plain_json' or policy.max_new_tokens != 512
            or policy.timeout_seconds != 180 or policy.use_cache is not True):
        raise ValueError('require original 512-token/180-second/cache/plainJSON parent policy')
    backend = bindings['backend']
    if (backend['requested_dtype'] not in ('native', 'bfloat16') or backend['head_dtype'] != 'torch.bfloat16'
            or backend['embedding_dtype'] != 'torch.bfloat16'):
        raise ValueError('require native BF16 baseline parent')
    return policy


def bind_runtime(checkpoint, bindings):
    metadata = runtime_metadata(checkpoint)
    logical._equal(metadata['model'], bindings['model'], 'checkpoint path/metadata differs from parent')
    for name in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
                 'cudnn', 'deterministic_algorithms', 'python'):
        logical._equal(metadata['backend'][name], bindings['backend'][name], 'parent backend differs: ' + name)
    for name in ('run_stance_localization_plain_crossmodel.py', 'run_stance_localization_grouped.py',
                 'run_stance_localization.py', 'recover_stance_baseline_truncations.py'):
        path = ROOT / 'scripts' / name
        metadata['code']['source_sha256']['scripts/' + name] = sha256_bytes(path.read_bytes())
    return metadata


def run(args):
    if args.phase != 'primary' or getattr(args, 'prior_run', None) is not None:
        raise ValueError('only fresh primary execution, no prior migration')
    inputs = load_baseline_inputs(args.inputs)
    parent = load_completed_baseline(args.parent, inputs=inputs)
    table = build_localization_pairs(inputs, parent)
    bindings = parent.metadata['bindings']
    policy = validate_parent(table, bindings)
    checkpoint = args.model.resolve(strict=True)
    identity = config_identity(checkpoint)
    metadata = bind_runtime(checkpoint, bindings)
    logical._equal(identity['config_sha256'],
        metadata['model']['metadata_file_sha256'].get('config.json'), 'config bytes changed during preflight')
    logical.shard_layers(identity['configured_layer_count'], args.shard_index, args.num_shards)
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
    desc = grouped.descriptor(table, 'primary', count, args.shard_index, args.num_shards,
        dict(runtime=metadata, template=bindings['template'], generation_policy=bindings['generation_policy'],
             grammar=reference.provenance, model_layer_authentication=identity))

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
