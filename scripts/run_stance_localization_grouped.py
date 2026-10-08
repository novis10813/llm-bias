"""Pair-grouped primary development localization, with immutable layer-modulo shards.

Internal helpers accept tiny fake tables for tests. The public entry always loads
validated full inputs and a complete baseline before loading a checkpoint.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
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
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy
from llm_bias.core.inference.stance_localization_execution import (
    _same_full_output, _bind_expected,
)
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop

from scripts import run_stance_localization as original
from scripts.run_stance_localization import (
    SPANS, ROLES, shard_layers, cells, alignment_for, cell_record,
    import_cell, gate_record, import_gate, require_gate, _equal,
)
from llm_bias.core.inference.stance_localization_grouped import (
    ReplacementCell, execute_grouped_prompt_replacement,
)
from contextlib import contextmanager
import fcntl
import stat

EXECUTION_VERSION = 'grouped_pair_v1'
APPROVED_LC4_COMMIT = '6c553c6c635f88a40b68a9bdcec3a090a7870c1f'


@lru_cache(maxsize=1)
def approved_lc4_sources():
    """Exact original inventory and bytes from the immutable approved Git tree.

    New independent core modules are not retroactively required in old records.
    Every module that did exist in LC4 remains mandatory, never intersection-only.
    """
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
        APPROVED_LC4_COMMIT, '--', 'llm_bias/core'], cwd=ROOT, text=True).splitlines()
    names = [n for n in names if n.endswith('.py')]
    names += ['scripts/run_stance_baseline.py', 'scripts/smoke_stance_checkpoint.py',
              'scripts/run_stance_localization.py', 'scripts/recover_stance_baseline_truncations.py',
              'uv.lock']
    result = {n: sha256_bytes(subprocess.check_output(
        ['git', 'show', APPROVED_LC4_COMMIT + ':' + n], cwd=ROOT)) for n in names}
    jlens = ROOT / 'third_party/jacobian-lens'
    jlens_head = '581d398613e5602a5af361e1c34d3a92ea82ba8e'
    for n in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', jlens_head],
                                      cwd=jlens, text=True).splitlines():
        if n.endswith('.py'):
            result['third_party/jacobian-lens/' + n] = sha256_bytes(subprocess.check_output(
                ['git', 'show', jlens_head + ':' + n], cwd=jlens))
    return result


def parser():
    p = original.parser()
    next(a for a in p._actions if a.dest == 'phase').choices = ('primary',)
    p.add_argument('--prior-run', type=Path)
    return p


def logical_descriptor(desc):
    return {k: v for k, v in desc.items() if k not in (
        'bindings', 'execution_version', 'logical_grid_sha256', 'prior_source')}


def descriptor(table, phase, count, index, shards, bindings):
    if phase != 'primary':
        raise ValueError('grouped runner supports primary only')
    desc = original.descriptor(table, phase, count, index, shards, bindings)
    return desc | dict(execution_version=EXECUTION_VERSION,
        logical_grid_sha256=sha256_json(logical_descriptor(desc)))


def run(args):
    inputs = load_baseline_inputs(args.inputs)
    parent = load_completed_baseline(args.parent, inputs=inputs)
    table = build_localization_pairs(inputs, parent)
    if len(table.pairs) != 4024 or sum(p.role in ROLES for p in table.pairs) != 3016:
        raise ValueError('require original full 4024-pair / 3016-development grid')
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
    metadata['code']['source_sha256']['scripts/run_stance_localization.py'] = sha256_bytes(
        (ROOT / 'scripts/run_stance_localization.py').read_bytes())
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
    if len(model.layers) != 40:
        raise ValueError('primary production grid requires all 40 layers')
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
    def run_group(pair, coordinates):
        return execute_grouped_prompt_replacement(model, tokenizer, prompts[pair.donor_key],
            prompts[pair.target_key], capability, policy=policy, cells=coordinates,
            expected_donor=parent.generation_for(pair.donor_key),
            expected_target=parent.generation_for(pair.target_key))
    if args.prior_run is not None:
        if args.prior_run.resolve() == args.output_dir.resolve():
            raise ValueError('prior and output must be distinct')
        with prior_snapshot(args.prior_run, table, desc, prompts, parent) as snapshot:
            return execute_run(args, desc, table, prompts, parent, run_gate, run_group, snapshot)
    return execute_run(args, desc, table, prompts, parent, run_gate, run_group, None)


def execute_run(args, desc, table, prompts, parent, run_gate, run_group, snapshot):
    if snapshot is not None:
        desc = desc | {'prior_source': snapshot['source']}
    registration = dict(descriptor=desc, descriptor_sha256=sha256_json(desc))
    with recovery_store(args.output_dir, registration) as records:
        report = run_grid(records, table, desc, prompts, parent, run_gate, run_group, snapshot=snapshot)
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


@contextmanager
def prior_snapshot(root, table, desc, prompts, parent):
    """Read-only locked LC4 snapshot. A live exclusive writer is rejected."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('invalid prior directory')
    fd = os.open(root / 'store.lock', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        inode = os.fstat(fd)
        if not stat.S_ISREG(inode.st_mode):
            raise ValueError('invalid prior lock')
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        entry = os.stat(root / 'store.lock', follow_symlinks=False)
        if (inode.st_dev, inode.st_ino) != (entry.st_dev, entry.st_ino):
            raise ValueError('prior lock inode changed')
        if set(p.name for p in root.iterdir()) - {'summary.json'} != {
                'store.lock', 'registration.json', 'records'}:
            raise ValueError('prior layout or staging differs')
        records = root / 'records'
        if records.is_symlink() or not records.is_dir():
            raise ValueError('invalid prior records directory')
        registration = read_file(root / 'registration.json')
        old = registration['descriptor']
        _equal(registration, dict(descriptor=old, descriptor_sha256=sha256_json(old)),
               'prior registration differs')
        expected = original.descriptor(table, 'primary', desc['actual_layer_count'],
            desc['shard_index'], desc['num_shards'], old['bindings'])
        _equal(old, expected, 'prior logical descriptor differs')
        _equal(logical_descriptor(old), logical_descriptor(desc), 'prior logical grid differs')
        for name in ('template', 'generation_policy', 'grammar'):
            _equal(old['bindings'][name], desc['bindings'][name], 'prior binding differs: ' + name)
        for name in ('model', 'backend'):
            _equal(old['bindings']['runtime'][name], desc['bindings']['runtime'][name],
                   'prior runtime differs: ' + name)
        old_sources = old['bindings']['runtime']['code'].get('source_sha256')
        if not isinstance(old_sources, dict):
            raise ValueError('prior code identity missing')
        _equal(old_sources.get('scripts/run_stance_localization.py'),
               sha256_bytes((ROOT / 'scripts/run_stance_localization.py').read_bytes()),
               'unapproved original runner code')
        current_sources = desc['bindings']['runtime']['code']['source_sha256']
        approved = approved_lc4_sources()
        _equal(old_sources, approved, 'prior shared code inventory or approved hashes differ')
        for name, digest in approved.items():
            _equal(current_sources.get(name), digest, 'prior shared code differs: ' + name)
        plan, gate_key, gates = _plan(table, desc)
        saved, passed = {}, {}
        hashes = {'registration.json': sha256_bytes((root / 'registration.json').read_bytes())}
        for path in sorted(records.iterdir()):
            item = read_file(path)
            hashes['records/' + path.name] = sha256_bytes(path.read_bytes())
            if path.name in gates:
                layer, _ = gates[path.name]
                execution = import_gate(item, gate_key, parent, layer, sha256_json(old))
                require_gate(execution, parent.generation_for(gate_key))
                passed[path.name] = execution
            elif path.name in plan:
                key, pair = plan[path.name]
                mapping = alignment_for(pair, prompts, 'primary', key['span'], key['selector'])
                execution = import_cell(item, key, pair, mapping, parent, sha256_json(old))
                if execution is not None and not execution.executed:
                    raise ValueError('prior contains halted cell')
                saved[path.name] = item
            else:
                raise ValueError('foreign prior record or staging')
        for name in saved:
            key, _ = plan[name]
            if f"gate_{key['layer']}_{key['span']}.json" not in passed:
                raise ValueError('prior cell missing gate')
        if (root / 'summary.json').exists() or (root / 'summary.json').is_symlink():
            def forbidden(*args):
                raise ValueError('prior summary has incomplete coverage')
            original.run_grid(records, table, old, prompts, parent, forbidden, forbidden)
            hashes['summary.json'] = sha256_bytes((root / 'summary.json').read_bytes())
        yield dict(source=dict(path=str(root.resolve()), descriptor=old,
            descriptor_sha256=sha256_json(old), content_sha256=hashes), records=saved)
    finally:
        os.close(fd)


def _plan(table, desc):
    expected = descriptor(table, desc['phase'], desc['actual_layer_count'],
                          desc['shard_index'], desc['num_shards'], desc['bindings'])
    if 'prior_source' in desc:
        expected['prior_source'] = desc['prior_source']
    _equal(desc, expected, 'grouped descriptor differs')
    plan = {name: (key, pair) for name, key, pair in cells(table, desc)}
    gate_key = min(p.target_key for p in table.pairs if p.role in ROLES)
    gates = {f'gate_{layer}_{span}.json': (layer, span)
             for layer in desc['layers'] for span in SPANS}
    return plan, gate_key, gates


def imported_record(name, raw, desc):
    source = desc['prior_source']
    return dict(config_hash=sha256_json(desc), execution_version=EXECUTION_VERSION,
        source_reference=dict(path=source['path'], filename='records/' + name,
            file_sha256=source['content_sha256']['records/' + name],
            descriptor_sha256=source['descriptor_sha256']), original_record=raw)


def _import_current(item, name, key, pair, mapping, parent, desc):
    if 'original_record' not in item:
        return import_cell(item, key, pair, mapping, parent, sha256_json(desc))
    if 'prior_source' not in desc:
        raise ValueError('unexpected prior origin')
    _equal(item, imported_record(name, item['original_record'], desc), 'source reference differs')
    _equal(sha256_bytes(canonical_json_bytes(item['original_record']) + b'\n'),
           desc['prior_source']['content_sha256']['records/' + name], 'prior payload differs')
    return import_cell(item['original_record'], key, pair, mapping, parent,
                       desc['prior_source']['descriptor_sha256'])


def run_grid(records, table, desc, prompts, parent, run_gate, run_group, *, snapshot=None):
    """Validate all records, run true gates, then group only remaining cells."""
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
            saved[path.name] = _import_current(item, path.name, key, pair, mapping, parent, desc)
        elif path.name.startswith('halt_'):
            _validate_halt(item, path.name, plan, parent, desc)
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
    if snapshot is not None:
        _equal(snapshot['source'], desc.get('prior_source'), 'snapshot identity differs')
        for name, raw in snapshot['records'].items():
            if name not in saved:
                publish(records, name, imported_record(name, raw, desc))
                key, pair = plan[name]
                mapping = alignment_for(pair, prompts, 'primary', key['span'], key['selector'])
                saved[name] = _import_current(read_file(records / name), name, key, pair, mapping, parent, desc)
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
                _validate_halt(item, name, plan, parent, desc)
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
            saved[name] = _import_current(read_file(records / name), name, key, pair, mapping, parent, desc)
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
    if summary is not None:
        _equal(summary, report, 'summary differs')
    return report


def _validate_halt(item, name, plan, parent, desc):
    """Closed group-clean diagnostic, not a fabricated per-cell execution."""
    statuses = {'donor_failed', 'donor_mismatch', 'source_unavailable',
                'target_failed', 'target_mismatch'}
    keys = item['keys']
    if not keys or item['status'] not in statuses:
        raise ValueError('invalid group halt')
    entries = []
    for key in keys:
        cell_name = sha256_json(key) + '.json'
        if cell_name not in plan or plan[cell_name][0] != key:
            raise ValueError('foreign halted coordinate')
        entries.append(plan[cell_name])
    pair = entries[0][1]
    if len({sha256_json(k) for k in keys}) != len(keys) or any(p != pair for _, p in entries):
        raise ValueError('halt pair or duplicate coordinate differs')
    if name != 'halt_' + pair.pair_sha256 + '.json':
        raise ValueError('halt filename differs')
    donor = _generation(item['donor'])
    target = None if item['target_clean'] is None else _generation(item['target_clean'])
    original._provenance(donor, parent.generation_for(pair.donor_key))
    if target is not None:
        original._provenance(target, parent.generation_for(pair.target_key))
    status = item['status']
    donor_ok = donor.failure_type is None and _same_full_output(donor, parent.generation_for(pair.donor_key))
    if ((status == 'donor_failed' and donor.failure_type is None)
            or (status == 'donor_mismatch' and (donor.failure_type is not None or donor_ok))
            or (status in {'source_unavailable', 'target_failed', 'target_mismatch'} and not donor_ok)
            or ((status.startswith('donor') or status == 'source_unavailable') and target is not None)
            or (status == 'target_failed' and (target is None or target.failure_type is None))
            or (status == 'target_mismatch' and (target is None or target.failure_type is not None
                or _same_full_output(target, parent.generation_for(pair.target_key))))):
        raise ValueError('halt outcome differs')
    _equal(item, dict(config_hash=sha256_json(desc), pair=pair.to_dict(), keys=keys,
        status=status, donor=donor.to_dict(), target_clean=None if target is None else target.to_dict()),
        'halt record differs')


if __name__ == '__main__':
    raise SystemExit(main())
