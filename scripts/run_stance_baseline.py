"""Full approved baseline diagnostic; no steering or research certification."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from importlib.metadata import version
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.smoke_stance_checkpoint import generation_adapter, compile_smoke_grammar
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.model import load_model
from llm_bias.core.experiment_contract import PlanIdentity
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_plan import build_baseline_plan
from llm_bias.core.stance_baseline_store import open_baseline_store
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy, _hf_controls, generate_structured
from llm_bias.core.inference.harmony_generation import generate_harmony_structured


def boolean(text):
    if text.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('expected true or false')
    return text.lower() == 'true'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'inputs', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--stop-token-id', required=True, action='append', type=int)
    p.add_argument('--channel-policy', choices=('plain_json', 'harmony_no_tools'), default='plain_json')
    p.add_argument('--dtype', choices=('bfloat16', 'native'), default='bfloat16')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--timeout-seconds', type=float, default=180)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--use-cache', type=boolean, default=True)
    return p


def immutable_json(path, record):
    data = canonical_json_bytes(record) + b'\n'
    try:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f'immutable metadata differs: {path.name}')


def runtime_metadata(checkpoint):
    files = {p.name: sha256_bytes(p.read_bytes()) for p in checkpoint.iterdir()
             if p.is_file() and (p.name in ('config.json', 'generation_config.json',
                 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
                 'chat_template.jinja') or p.name.startswith('tokenizer'))}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('expected 40-hex git HEAD provenance')
    sources = [ROOT / 'scripts/run_stance_baseline.py', ROOT / 'scripts/smoke_stance_checkpoint.py',
               ROOT / 'uv.lock', *sorted((ROOT / 'llm_bias/core').rglob('*.py')),
               *sorted((ROOT / 'third_party/jacobian-lens').rglob('*.py'))]
    return dict(model=dict(resolved_path=str(checkpoint), metadata_file_sha256=files,
                           identity_scope='metadata_only_not_full_weights'),
                code=dict(source_sha256={str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes())
                                         for p in sources},
                          git_head=head, git_head_kind='git_object_id',
                          jlens_git_head=subprocess.check_output(
                              ['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'third_party/jacobian-lens',
                              text=True).strip()),
                backend=dict(python=platform.python_version(), torch=torch.__version__,
                             transformers=version('transformers'), xgrammar=version('xgrammar'),
                             jlens=version('jlens'), cuda=torch.version.cuda,
                             kernel_policy='native_hf_default; xgrammar_auto_cuda_mask',
                             cudnn=torch.backends.cudnn.version(),
                             deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))


def build_identity(inputs, prompt, capability, policy, metadata, route):
    nested = getattr(capability, 'json_capability', capability)
    if prompt.schema_sha256 != nested.schema_sha256:
        raise ValueError('renderer schema differs from compiled capability')
    protocol = dict(kind='diagnostic_full_baseline_v1', research_eligible=False,
                    planned_global=2012, members=503, role_assignment=inputs.roles_sha256,
                    evidence=inputs.evidence_sha256, sampling='all_approved_pairs_once',
                    timeout_budget_seconds_per_row=policy.timeout_seconds, model_route=route,
                    efficacy_claim=False, protocol_freeze_claim=False)
    records = dict(protocol=protocol,
                   template=dict(body_template_sha256=prompt.template_sha256,
                                 actual_wrapper_record=prompt.wrapper_policy),
                   generation_policy=dict(policy=asdict(policy),
                                          hf_controls=_hf_controls(policy, nested.stop_token_ids)),
                   **metadata)
    identity = PlanIdentity(protocol_sha256=sha256_json(records['protocol']),
        population_sha256=inputs.membership_sha256, issuer_sha256=inputs.issuer_mapping_sha256,
        roles_sha256=inputs.roles_sha256, evidence_sha256=inputs.evidence_sha256,
        schema_sha256=nested.schema_sha256, template_sha256=sha256_json(records['template']),
        model_sha256=sha256_json(records['model']), code_sha256=sha256_json(records['code']),
        backend_sha256=sha256_json(records['backend']),
        generation_policy_sha256=sha256_json(records['generation_policy']),
        operator_sha256='not_applicable', parent_sha256='not_applicable')
    return identity, records


def run_rows(store, inputs, first_prompt, tokenizer, wrapper, model, device, capability, policy):
    members = {m.ticker: m for m in inputs.members}
    pairs = {(p.ticker, p.condition, p.trial_id): p for p in inputs.pairs}
    driver = generate_harmony_structured if policy.channel_policy == 'harmony_no_tools' else generate_structured
    for key in store.pending_keys:
        pair = pairs[key.ticker, key.condition, key.trial_id]
        prompt = render_decision_prompt(tokenizer, members[key.ticker], pair, wrapper_policy=wrapper)
        if (prompt.wrapper_policy != first_prompt.wrapper_policy or
                prompt.template_sha256 != first_prompt.template_sha256 or
                prompt.schema_sha256 != first_prompt.schema_sha256):
            raise ValueError('row prompt binding changed')
        ids = torch.tensor([prompt.inference_token_ids], dtype=torch.long, device=device)
        generated = driver(model, tokenizer, ids, capability, policy=policy)
        store.record(key, generated)
        print(canonical_json_bytes(dict(ticker=key.ticker, condition=key.condition,
            decision=generated.decision, failure_type=generated.failure_type,
            finish_reason=generated.finish_reason)).decode(), file=sys.stderr, flush=True)


def summary(store, plan, index, count):
    generations = [store.generation_for(k) for k in store.recorded_keys]
    return dict(kind='diagnostic_full_baseline_v1', research_eligible=False,
                steering_flips_claimed=False, plan_hash=plan.plan_hash,
                planned_global=2012, shard_index=index, num_shards=count,
                assigned_count=len(plan.keys_for_shard(index, count)),
                executed_count=len(generations), missing_count=len(store.pending_keys),
                complete_shard=not store.pending_keys,
                complete_global=count == 1 and not store.pending_keys,
                class_counts=dict(Counter(g.decision or 'no_decision' for g in generations)),
                failure_counts=dict(Counter(g.failure_type or 'none' for g in generations)),
                finish_counts=dict(Counter(g.finish_reason for g in generations)),
                gates={name: False for name in plan.gate_names})


def run(args):
    inputs = load_baseline_inputs(args.inputs)
    checkpoint = args.model.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError('model must be a local checkpoint directory')
    metadata = runtime_metadata(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    loaded, _, device = load_model(str(checkpoint), device_map=None,
                                  dtype='native' if args.dtype == 'native' else torch.bfloat16)
    model = generation_adapter(loaded)
    model.hf_model.eval()
    tokenizer = model.tokenizer
    head = model.hf_model.get_output_embeddings().weight
    embedding = model.hf_model.get_input_embeddings().weight
    if (torch.device(device).type != 'cuda' or head.device.type != 'cuda'
            or embedding.device.type != 'cuda'):
        raise ValueError('actual model device must be CUDA')
    if not head.dtype.is_floating_point or not embedding.dtype.is_floating_point:
        raise ValueError('floating head and embedding required')
    capability = compile_smoke_grammar(tokenizer, head.shape[0], args.stop_token_id, args.channel_policy)
    wrapper = WrapperPolicy(use_chat_template=True, add_special_tokens=False,
                            enable_thinking=args.channel_policy == 'harmony_no_tools')
    pair = inputs.pairs[0]
    member = next(m for m in inputs.members if m.ticker == pair.ticker)
    first = render_decision_prompt(tokenizer, member, pair, wrapper_policy=wrapper)
    policy = StructuredGenerationPolicy(max_new_tokens=args.max_new_tokens, use_cache=args.use_cache,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else args.stop_token_id[0],
        timeout_seconds=args.timeout_seconds, channel_policy=args.channel_policy)
    metadata['backend'].update(device=str(device), embedding_device=str(embedding.device),
                              head_device=str(head.device), requested_dtype=args.dtype,
                              head_dtype=str(head.dtype), embedding_dtype=str(embedding.dtype),
                              gpu_name=torch.cuda.get_device_name(embedding.device),
                              attention_implementation=getattr(model.hf_model.config, '_attn_implementation', None))
    identity, records = build_identity(inputs, first, capability, policy, metadata, args.channel_policy)
    plan = build_baseline_plan(inputs, identity)
    plan.keys_for_shard(args.shard_index, args.num_shards)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'shard-{args.shard_index}-of-{args.num_shards}'
    with open_baseline_store(args.output_dir / stem, inputs=inputs, plan=plan,
                             shard_index=args.shard_index, num_shards=args.num_shards) as store:
        immutable_json(args.output_dir / f'{stem}.execution_metadata.json',
                       dict(identity=identity.to_dict(), bindings=records,
                            inputs_manifest_sha256=inputs.manifest_sha256))
        run_rows(store, inputs, first, tokenizer, wrapper, model, device, capability, policy)
        report = summary(store, plan, args.shard_index, args.num_shards)
        immutable_json(args.output_dir / f'{stem}.summary.json', report)
    print(canonical_json_bytes(report).decode())
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            return run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        record = dict(kind='diagnostic_full_baseline_failure', research_eligible=False,
                      complete_shard=False, error_type=type(exc).__name__,
                      message=str(exc).encode('utf8', errors='replace').decode('utf8'))
        immutable_json(args.output_dir / f'error-{uuid.uuid4().hex}.json', record)
        print(canonical_json_bytes(record).decode(), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
