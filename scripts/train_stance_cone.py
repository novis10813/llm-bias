"""Fixed fit-only frozen-checkpoint cone optimization, never validation acceptance.

The public diagnostic authenticates the same complete pack and runs exactly one
complete ticker. C3 graphs are used unchanged. OOM is a failed attempt, not a
request to truncate teachers or silently substitute another backward algorithm.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.run_stance_baseline import runtime_metadata
from scripts.smoke_stance_checkpoint import generation_adapter
from scripts.recover_stance_baseline_truncations import publish, check_prompt
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.model import load_model
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline
from llm_bias.core.stance_cone_teachers import compile_cone_teachers
from llm_bias.core.stance_cone_objectives import (
    TeacherResponse, initialize_basis, positive_rays, CONE_DIMENSIONS, INITIALIZATION_SEEDS,
)
from llm_bias.core.inference.stance_cone_training_step import TrainingBatch, stance_cone_training_step
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt

PURPOSES = ('addition', 'ablation', 'retain')
PACK_FILES = {'config.json', 'source_manifest.json', 'review_manifest.json',
              'teachers.jsonl', 'unavailable.json'}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('inputs', 'parent', 'teachers', 'model', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--dimension', required=True, type=int, choices=CONE_DIMENSIONS)
    p.add_argument('--seed', required=True, type=int, choices=INITIALIZATION_SEEDS)
    p.add_argument('--diagnostic-one-step', action='store_true')
    return p


def ticker_order(tickers, seed):
    if seed not in INITIALIZATION_SEEDS:
        raise ValueError('undeclared seed')
    return tuple(sorted(tickers, key=lambda t: (sha256_json(dict(seed=seed, ticker=t)), t)))


def read_raw(path):
    """Read regular raw bytes without following a final-component symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('teacher artifact must be a regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def load_teachers(directory, inputs, parent, tokenizer):
    """Raw file hashes plus exact fresh compiler equality, not self-attestation.

    Recompilation authenticates original prompt prefixes, full source responses,
    policies, grammar, review, authoritative roles and complete 906 coverage.
    Compiler config is historical provenance, not permission to change targets.
    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('teacher pack must be a directory, not a symlink')
    if {p.name for p in directory.iterdir()} != PACK_FILES | {'manifest.json'}:
        raise ValueError('teacher pack exact file layout differs')
    raw_manifest = read_raw(directory / 'manifest.json')
    manifest = json.loads(raw_manifest)
    if set(manifest['file_sha256']) != PACK_FILES:
        raise ValueError('teacher pack file hash coverage differs')
    raw = {name: read_raw(directory / name) for name in PACK_FILES}
    if any(sha256_bytes(data) != manifest['file_sha256'][name] for name, data in raw.items()):
        raise ValueError('teacher pack raw file hash differs')
    expected = compile_cone_teachers(inputs, parent, tokenizer)
    if expected['manifest']['accepted'] is not True or len(expected['teachers']) != 906:
        raise ValueError('complete accepted 906 fit teachers required')
    if manifest != expected['manifest'] | {'file_sha256': manifest['file_sha256']}:
        raise ValueError('teacher manifest differs from authenticated parent')
    for filename, key in (('source_manifest.json', 'sources'), ('review_manifest.json', 'review'),
                          ('unavailable.json', 'unavailable')):
        if canonical_json_bytes(json.loads(raw[filename])) != canonical_json_bytes(expected[key]):
            raise ValueError('teacher source/review/coverage differs from parent')
    rows = [json.loads(line) for line in raw['teachers.jsonl'].splitlines()]
    if rows != expected['teachers']:
        raise ValueError('teachers differ from freshly authenticated sources and masks')
    config = json.loads(raw['config.json'])
    bindings = parent.metadata.get('original_metadata', parent.metadata)['bindings']
    for name, value in dict(kind='stance_cone_teacher_compiler_config_v1',
        inputs_manifest_sha256=inputs.manifest_sha256, parent_sha256=parent.parent_sha256,
        model=bindings['model'], logical_tokenizer_name=bindings['model']['resolved_path'],
        execution='CPU_tokenizer_and_grammar_only_no_checkpoint_weights').items():
        if config.get(name) != value:
            raise ValueError('teacher compiler provenance binding differs: ' + name)
    records = tuple(TeacherResponse(**(r | dict(token_ids=tuple(r['token_ids']),
                         response_mask=tuple(r['response_mask'])))) for r in rows)
    return records, dict(manifest_sha256=sha256_bytes(raw_manifest), manifest=manifest,
                        compiler_config=config)


def make_batches(records, prompts, device):
    """One full response per purpose. No padding, truncation or response editing."""
    batches = {}
    for record in records:
        prompt = prompts[record.ticker, record.purpose]
        ids = tuple(prompt.inference_token_ids)
        n = len(ids)
        if (record.token_ids[:n] != ids or record.response_mask !=
                (False,) * n + (True,) * (len(record.token_ids) - n)):
            raise ValueError('teacher must contain exact prompt and complete response mask')
        span = prompt.instruction_span
        if not 0 <= span.token_start < span.token_end <= n:
            raise ValueError('instruction span must select original prompt positions')
        mask = torch.zeros(1, len(record.token_ids), dtype=torch.bool, device=device)
        mask[:, span.token_start:span.token_end] = True
        if record.purpose in batches:
            raise ValueError('duplicate ticker objective')
        batches[record.purpose] = TrainingBatch((record,), torch.ones_like(mask, dtype=torch.long), mask)
    if set(batches) != set(PURPOSES):
        raise ValueError('complete ticker objectives required')
    return batches


def resources(device, started):
    value = dict(elapsed_seconds=time.monotonic() - started)
    if device is not None and device.type == 'cuda':
        value.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return value


def failure(exc):
    # Arbitrary runtime messages may contain tensors, paths or full responses.
    # Keep a fixed compact message instead of serializing exception repr/str.
    oom = isinstance(exc, torch.cuda.OutOfMemoryError)
    return dict(error_type=type(exc).__name__, error_message=(
        'C3 full graph exceeded available memory; separately verified reduction/backward required'
        if oom else 'execution failed; no training acceptance'), capability_failure=True,
        oom=oom)


def optimize(directory, model, records, prompts, roles, config, *, step=stance_cone_training_step):
    """Internal CPU-testable loop. Public run always authenticates full capabilities."""
    started = time.monotonic()
    weight = model.hf_model.get_input_embeddings().weight
    device = weight.device
    # Public run resets before checkpoint loading, so peaks include load cost.
    completed = attempted = 0
    report = dict(training_completed=False, accepted_operator=False, research_eligible=False,
                  diagnostic_one_step=config['diagnostic_one_step'], planned_steps=302,
                  seed=config['seed'], dimension=config['dimension'], config_sha256=sha256_json(config))
    try:
        tickers = ticker_order({r.ticker for r in records}, config['seed'])
        if len(tickers) != 302 or len(records) != 906 or any(
                roles.get(r.ticker) != 'fit' for r in records):
            raise ValueError('complete 302 fit ticker epoch required')
        grouped = {t: tuple(r for r in records if r.ticker == t) for t in tickers}
        basis = initialize_basis(weight.shape[1], dimension=config['dimension'], seed=config['seed'])
        basis = basis.to(device=device, dtype=weight.dtype).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([basis], lr=0.01)
        generator = torch.Generator(device='cpu').manual_seed(config['seed'])
        for index, ticker in enumerate(tickers):
            attempted += 1
            step_started = time.monotonic()
            record = dict(index=index, ticker=ticker, config_sha256=report['config_sha256'])
            try:
                batches = make_batches(grouped[ticker], prompts, device)
                coefficients = (torch.rand(1, config['dimension'], generator=generator) + .01).to(
                    device=device, dtype=weight.dtype)
                optimizer.zero_grad(set_to_none=True)
                result = step(model, batches=batches, roles=roles, basis=basis,
                              coefficients=coefficients, layer=config['layer'], dose=32.)
                result.total.backward()
                if basis.grad is None or not torch.isfinite(basis.grad).all().item():
                    raise ValueError('nonfinite or missing basis gradient')
                record.update(losses={name: float(getattr(result, name).detach())
                              for name in (*PURPOSES, 'total')},
                              gradient_norm=float(basis.grad.detach().float().norm()))
                optimizer.step()
                with torch.no_grad():
                    norms = basis.float().norm(dim=-1, keepdim=True)
                    if not torch.isfinite(norms).all().item() or (norms <= 0).any().item():
                        raise ValueError('zero or nonfinite updated basis')
                    basis.copy_((basis.float() / norms).to(basis.dtype))
                    if not torch.isfinite(basis).all().item():
                        raise ValueError('nonfinite normalized basis')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                record['status'] = 'optimizer_step_completed'
                del result, batches
            except Exception as exc:
                record.update(status='failed', **failure(exc), **resources(device, step_started))
                publish(directory / 'records', f'{index:03d}-{ticker}.json', record)
                raise
            record.update(resources(device, step_started))
            publish(directory / 'records', f'{index:03d}-{ticker}.json', record)
            completed += 1
            if config['diagnostic_one_step']:
                break
        full = completed == 302 and not config['diagnostic_one_step']
        if full:
            # Only derived basis and deterministic positive-ray policy are exported.
            positive_rays(basis, coefficients)  # reject cancelling/nonfinite final ray
            publish(directory, 'operator.json', dict(basis=basis.detach().float().cpu().tolist(),
                config_sha256=report['config_sha256'], seed=config['seed'], dimension=config['dimension'],
                layer=config['layer'], scope='original_instruction_post_block', dose=32.,
                ray_policy=config['ray_policy'], optimizer=config['optimizer'],
                training_completed=True, accepted_operator=False, research_eligible=False))
        report.update(status='training_completed' if full else 'diagnostic_capability_success',
                      training_completed=full, capability_success=True)
    except Exception as exc:
        report.update(status='failed', capability_success=False, **failure(exc))
    report.update(attempted_steps=attempted, completed_steps=completed, **resources(device, started))
    publish(directory, 'summary.json', report)
    return report


def run(args):
    # Reserve before loading. All failed attempts remain immutable and visible.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'records').mkdir()
    started = time.monotonic()
    device = None
    try:
        publish(args.output_dir, 'attempt.json', dict(arguments={k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()}, training_completed=False, accepted_operator=False,
            research_eligible=False))
        inputs = load_baseline_inputs(args.inputs)
        parent = load_completed_baseline(args.parent, inputs=inputs)
        bindings = parent.metadata['bindings']
        checkpoint = args.model.resolve(strict=True)
        metadata = runtime_metadata(checkpoint)
        if metadata['model'] != bindings['model'] or 'glm4-9b-0414' not in str(checkpoint).lower():
            raise ValueError('require original parent GLM4-9B-0414 checkpoint path/metadata')
        for name in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
                     'cudnn', 'deterministic_algorithms'):
            if metadata['backend'][name] != bindings['backend'][name]:
                raise ValueError('parent backend differs: ' + name)
        metadata['code']['source_sha256']['scripts/train_stance_cone.py'] = sha256_bytes(Path(__file__).read_bytes())
        if not torch.cuda.is_available():
            raise RuntimeError('actual checkpoint training requires CUDA')
        device = torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        loaded, _, _ = load_model(str(checkpoint), device_map=None, dtype='native', trust_remote_code=False)
        model = generation_adapter(loaded)
        model.hf_model.eval().requires_grad_(False)
        weight = model.hf_model.get_input_embeddings().weight
        head = model.hf_model.get_output_embeddings().weight
        if len(model.layers) != 40 or weight.dtype != torch.bfloat16 or head.dtype != torch.bfloat16:
            raise ValueError('require 40-layer native BF16 GLM checkpoint')
        if weight.device != device or head.device != device:
            raise ValueError('require single CUDA device')
        if getattr(model.hf_model.config, '_attn_implementation', None) != bindings['backend']['attention_implementation']:
            raise ValueError('actual attention implementation differs')
        records, teacher_binding = load_teachers(args.teachers, inputs, parent, model.tokenizer)
        wrapper_record = bindings['template']['actual_wrapper_record']
        wrapper = WrapperPolicy(**{k: v for k, v in wrapper_record.items() if k != 'tokenizer_chat_template'})
        members = {m.ticker: m for m in inputs.members}
        keys = {(k.ticker, k.condition): k for k in parent.plan.keys}
        prompts = {}
        for r in records:
            condition = {'addition': '--', 'ablation': '++', 'retain': '+-'}[r.purpose]
            key = keys[r.ticker, condition]
            prompt = render_decision_prompt(model.tokenizer, members[r.ticker],
                inputs.pair_for(r.ticker, condition, key.trial_id), wrapper_policy=wrapper)
            check_prompt(prompt, bindings)
            prompts[r.ticker, r.purpose] = prompt
        metadata['backend'].update(actual_embedding_dtype=str(weight.dtype), actual_head_dtype=str(head.dtype),
                                   gpu_name=torch.cuda.get_device_name(device))
        config = dict(kind='stance_cone_training_v1', dimension=args.dimension, seed=args.seed,
            declared_grid=dict(dimensions=list(CONE_DIMENSIONS), seeds=list(INITIALIZATION_SEEDS)),
            diagnostic_one_step=args.diagnostic_one_step, layer=19, scope='original_instruction_post_block',
            dose=32., planned_steps=302, equal_example_weight=True, objective_weights=[1., 1., 1.],
            panels='separate_basis_and_one_sample_means',
            optimizer=dict(name='Adam', lr=.01, betas=[.9, .999], eps=1e-8, weight_decay=0,
                           state_dtype=str(weight.dtype), unit_normalize_after_step=True),
            ray_policy='local_CPU_generator_seed;uniform(0,1)+0.01;one_per_step;L1_then_L2',
            ticker_order=list(ticker_order({r.ticker for r in records}, args.seed)),
            teacher_binding=teacher_binding, runtime=metadata,
            inputs_manifest_sha256=inputs.manifest_sha256, roles_sha256=inputs.roles_sha256,
            parent_sha256=parent.parent_sha256, plan_hash=parent.plan.plan_hash,
            full_plan_identity=parent.plan.identity.to_dict(), accepted_operator=False, research_eligible=False)
        publish(args.output_dir, 'config.json', config)
        report = optimize(args.output_dir, model, records, prompts, inputs.roles['assignments'], config)
        return 0 if report['capability_success'] else 1
    except Exception as exc:
        if not (args.output_dir / 'summary.json').exists():
            publish(args.output_dir, 'summary.json', dict(status='preflight_failure', training_completed=False,
                accepted_operator=False, research_eligible=False, diagnostic_one_step=args.diagnostic_one_step,
                attempted_steps=0, completed_steps=0, **failure(exc), **resources(device, started)))
        return 1


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except Exception as exc:
        print(canonical_json_bytes(failure(exc)).decode(), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
