"""Adaptive-budget diagnostic recovery; never rewrites the full baseline parent."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path
import stat
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.run_stance_baseline import runtime_metadata
from scripts.smoke_stance_checkpoint import generation_adapter, compile_smoke_grammar
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.experiment_contract import GenerationOutcome, RowKey
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy, generate_structured
from llm_bias.core.inference.harmony_generation import generate_harmony_structured
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.stance_baseline_adapter import baseline_gate_input
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline
from llm_bias.core.stance_baseline_store import _generation, _parse


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'parent', 'model', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--timeout-seconds', type=float, default=1200)
    return p


def recovery_keys(parent):
    return tuple(row.key for row in sorted(parent.rows, key=lambda row: row.key)
                 if row.outcome.failure_type == 'truncated' and row.outcome.finish_reason == 'token_budget')


def recovery_policy(parent, max_new_tokens, timeout_seconds):
    old = parent.metadata['bindings']['generation_policy']['policy']
    if type(max_new_tokens) is not int or max_new_tokens <= old['max_new_tokens']:
        raise ValueError('recovery budget must be an integer greater than the original')
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError('recovery timeout must be finite and positive')
    return StructuredGenerationPolicy(**(old | dict(max_new_tokens=max_new_tokens,
                                                   timeout_seconds=timeout_seconds)))


def publish(directory, name, value):
    """Write once. A staging leftover after interruption is never silently removed."""
    raw = canonical_json_bytes(value) + b'\n'
    staging = directory / ('.pending-' + uuid.uuid4().hex + '.tmp')
    with staging.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(staging, directory / name, follow_symlinks=False)
    staging.unlink()
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_file(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('recovery artifact must be a regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            return _parse(stream.read())
    finally:
        os.close(fd)


def check_prompt(prompt, bindings):
    if (prompt.wrapper_policy != bindings['template']['actual_wrapper_record']
            or prompt.template_sha256 != bindings['template']['body_template_sha256']):
        raise ValueError('recovery prompt wrapper/template differs from parent')


def bind_relocated_tokenizer(tokenizer, model_record, parent_model_record):
    """Preserve the parent logical tokenizer name only for byte-verified relocation.

    The actual checkpoint path remains in model/runtime metadata. The accepted
    tokenizer identity includes name_or_path, so a moved identical checkpoint
    needs this explicitly recorded logical-name binding, not a skipped hash.
    """
    if model_record['metadata_file_sha256'] != parent_model_record['metadata_file_sha256']:
        raise ValueError('relocated tokenizer metadata differs')
    actual = getattr(tokenizer, 'name_or_path', None)
    logical = parent_model_record['resolved_path']
    if actual != model_record['resolved_path']:
        raise ValueError('loaded tokenizer does not name the actual checkpoint')
    tokenizer.name_or_path = logical
    return dict(actual_loaded_name_or_path=actual, logical_parent_name_or_path=logical,
                relocation_metadata_sha256_verified=True)


def check_capability(capability, reference):
    nested = getattr(capability, 'json_capability', capability)
    for name in ('schema_sha256', 'schema_bytes_sha256', 'tokenizer_sha256',
                 'tokenizer_info_sha256', 'head_vocab_size', 'grammar_sha256'):
        if getattr(nested, name) != reference[name]:
            raise ValueError(f'recovery grammar binding differs: {name}')
    if list(nested.stop_token_ids) != reference['stop_token_ids']:
        raise ValueError('recovery stops differ')
    if (hasattr(capability, 'contract') and canonical_json_bytes(asdict(capability.contract))
            != canonical_json_bytes(reference['channel_contract'])):
        raise ValueError('recovery Harmony contract differs')


@contextmanager
def recovery_store(directory, registration):
    """Persistent nonblocking lock; no writer or automatic partial-init repair."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(); fresh = True
    except FileExistsError:
        fresh = False
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('invalid recovery directory')
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    if fresh:
        flags |= os.O_CREAT | os.O_EXCL
    lock = os.open(directory / 'store.lock', flags, 0o600)
    try:
        inode = os.fstat(lock)
        if not stat.S_ISREG(inode.st_mode):
            raise ValueError('invalid recovery lock')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entry = os.stat(directory / 'store.lock', follow_symlinks=False)
        if (inode.st_dev, inode.st_ino) != (entry.st_dev, entry.st_ino):
            raise ValueError('lock inode changed')
        if fresh:
            (directory / 'records').mkdir()
            publish(directory, 'registration.json', registration)
        else:
            allowed = {'registration.json', 'store.lock', 'records'}
            if (set(p.name for p in directory.iterdir()) - {'summary.json'} != allowed
                    or canonical_json_bytes(read_file(directory / 'registration.json'))
                    != canonical_json_bytes(registration)):
                raise ValueError('recovery registration or layout differs')
        records = directory / 'records'
        if records.is_symlink() or not records.is_dir():
            raise ValueError('invalid recovery records directory')
        yield records
    finally:
        os.close(lock)


def make_record(key, generated, parent, inputs):
    baseline_gate_input(generated)
    outcome = GenerationOutcome(key.ticker, inputs.issuer_by_ticker[key.ticker], key.condition,
        key.trial_id, generated.decision, generated.decision_complete, generated.schema_complete,
        generated.reason_valid, generated.finish_reason, generated.failure_type)
    return dict(key=key.to_dict(), parent_sha256=parent.parent_sha256,
                original_generated_token_sha256=parent.generation_for(key).generated_token_sha256,
                outcome=outcome.to_dict(), generation=generated.to_dict())


def run_recovery(records, keys, parent, inputs, generate, *, policy):
    expected = {sha256_json(k.to_dict()) + '.json': k for k in keys}
    found = {}
    for path in records.iterdir():
        if path.name not in expected:
            raise ValueError('unexpected recovery file or staging leftover')
        item = read_file(path)
        generated = _generation(item['generation'])
        recomputed = make_record(expected[path.name], generated, parent, inputs)
        if canonical_json_bytes(item) != canonical_json_bytes(recomputed):
            raise ValueError('recovery record binding differs')
        if generated.provenance['generation_policy'] != asdict(policy):
            raise ValueError('saved recovery policy differs')
        found[path.name] = generated
    for name, key in expected.items():
        if name in found:
            continue
        generated = generate(key)
        if generated.provenance['generation_policy'] != asdict(policy):
            raise ValueError('recovery driver policy differs')
        publish(records, name, make_record(key, generated, parent, inputs))
        found[name] = generated
        print(canonical_json_bytes(dict(ticker=key.ticker, condition=key.condition,
            decision=generated.decision, failure_type=generated.failure_type)).decode(), flush=True)
    return dict(kind='adaptive_budget_baseline_recovery_v1', research_eligible=False,
        homogeneous_original_policy=False, parent_sha256=parent.parent_sha256,
        original_planned=len(parent.plan.keys), selected_truncated=len(keys), executed=len(found),
        complete=True, adaptive_selection='only_parent_truncated_token_budget',
        original_policy=parent.metadata['bindings']['generation_policy']['policy'],
        recovery_policy=asdict(policy), class_counts=dict(Counter(g.decision or 'no_decision' for g in found.values())),
        failure_counts=dict(Counter(g.failure_type or 'none' for g in found.values())),
        original_failure_counts=parent.summary['failure_counts'])


def run(args):
    inputs = load_baseline_inputs(args.inputs)
    parent = load_completed_baseline(args.parent, inputs=inputs)
    policy = recovery_policy(parent, args.max_new_tokens, args.timeout_seconds)
    keys = recovery_keys(parent)
    bindings = parent.metadata['bindings']
    metadata = dict(empty_recovery=True)
    generate = None
    if keys:
        checkpoint = args.model.resolve(strict=True)
        metadata = runtime_metadata(checkpoint)
        if metadata['model']['metadata_file_sha256'] != bindings['model']['metadata_file_sha256']:
            raise ValueError('checkpoint metadata differs from parent')
        metadata['code']['source_sha256']['scripts/recover_stance_baseline_truncations.py'] = sha256_bytes(Path(__file__).read_bytes())
        if not torch.cuda.is_available():
            raise ValueError('recovery requires CUDA')
        loaded, _, device = load_model(str(checkpoint), device_map=None,
            dtype='native' if bindings['backend']['requested_dtype'] == 'native' else torch.bfloat16)
        model = generation_adapter(loaded); model.hf_model.eval(); tokenizer = model.tokenizer
        metadata['tokenizer_relocation'] = bind_relocated_tokenizer(
            tokenizer, metadata['model'], bindings['model'])
        head = model.hf_model.get_output_embeddings().weight
        if head.device.type != 'cuda':
            raise ValueError('actual recovery model must be on CUDA')
        reference = parent.generation_for(keys[0]).provenance
        capability = compile_smoke_grammar(tokenizer, head.shape[0], reference['stop_token_ids'], policy.channel_policy)
        check_capability(capability, reference)
        metadata['backend'].update(device=str(device), gpu_name=torch.cuda.get_device_name(head.device),
            head_dtype=str(head.dtype), driver_version=os.environ.get('LAB_RECOVERY_DRIVER_VERSION'),
            environment={k: v for k, v in os.environ.items() if k.startswith('LAB') or k == 'CUDA_VISIBLE_DEVICES'})
        wrapper = WrapperPolicy(use_chat_template=True, add_special_tokens=False,
                                enable_thinking=policy.channel_policy == 'harmony_no_tools')
        members = {m.ticker: m for m in inputs.members}
        prompts = {}
        for key in keys:
            prompt = render_decision_prompt(tokenizer, members[key.ticker],
                inputs.pair_for(key.ticker, key.condition, key.trial_id), wrapper_policy=wrapper)
            check_prompt(prompt, bindings)
            if prompt.schema_sha256 != parent.plan.identity.schema_sha256:
                raise ValueError('recovery schema differs')
            prompts[key] = prompt
        driver = generate_harmony_structured if policy.channel_policy == 'harmony_no_tools' else generate_structured
        def generate(key):
            ids = torch.tensor([prompts[key].inference_token_ids], device=device, dtype=torch.long)
            return driver(model, tokenizer, ids, capability, policy=policy)
    registration = dict(kind='adaptive_budget_baseline_recovery_v1', research_eligible=False,
        parent_sha256=parent.parent_sha256, parent_plan_hash=parent.plan.plan_hash,
        inputs_manifest_sha256=inputs.manifest_sha256, selected_keys=[k.to_dict() for k in keys],
        original_policy=bindings['generation_policy']['policy'], recovery_policy=asdict(policy), metadata=metadata)
    with recovery_store(args.output_dir, registration) as records:
        report = run_recovery(records, keys, parent, inputs, generate, policy=policy)
        summary = args.output_dir / 'summary.json'
        if summary.exists():
            if canonical_json_bytes(read_file(summary)) != canonical_json_bytes(report):
                raise ValueError('recovery summary differs')
        else:
            publish(args.output_dir, 'summary.json', report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except Exception as exc:
        print(canonical_json_bytes(dict(kind='recovery_failure', research_eligible=False,
            error_type=type(exc).__name__, message=str(exc))).decode(), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
