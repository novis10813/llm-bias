"""Read-only effective baseline. LC0 remains the original-parent boundary."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import fcntl
import os
from pathlib import Path
from types import MappingProxyType

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.experiment_contract import ExecutionRow, GenerationOutcome
from llm_bias.core.inference.harmony_channels import HarmonyBoundaryTracker, HarmonyTokenContract, _control_ids
from llm_bias.core.inference.structured_output import _hf_controls
from llm_bias.core.stance_baseline_adapter import _validate_generation_result
from llm_bias.core.stance_baseline_parent import (
    CompletedBaseline, load_completed_baseline, _open_root, _open_directory,
    _open_regular, _read_file, _validated_policy, _validated_generation_payload,
    _validated_provenance)
from llm_bias.core.stance_baseline_store import _generation, _parse

RECOVERY_PAIRS = frozenset((ticker, condition) for ticker, condition in (
    ('ADM','-+'), ('AWK','-+'), ('AXON','+-'), ('CBOE','+-'), ('CHD','-+'),
    ('CI','-+'), ('CL','-+'), ('CNP','-+'), ('CPB','-+'), ('D','-+'),
    ('DTE','+-'), ('DUK','-+'), ('EIX','-+'), ('EXC','-+'), ('FTV','-+'),
    ('HSY','+-'), ('HUBB','-+'), ('ISRG','+-'), ('JNJ','-+'), ('KEYS','-+'),
    ('KMI','-+'), ('KMX','-+'), ('KO','+-'), ('LNT','-+'), ('MDLZ','-+'),
    ('MPC','-+'), ('NUE','+-'), ('PM','-+'), ('PPL','-+'), ('ROL','-+'), ('XEL','-+')))
_RULE = 'preserve_all_primary_valid_originals_replace_exact31_truncated_token_budget_v1'
_GRAMMAR = ('backend', 'backend_version', 'byte_policy', 'compiler_policy', 'grammar_correction',
            'schema_sha256', 'schema_bytes_sha256', 'grammar_sha256', 'tokenizer_sha256',
            'tokenizer_info_sha256', 'head_vocab_size', 'stop_token_ids', 'channel_contract',
            'channel_contract_sha256', 'channel_policy_sha256')


def _same(a, b, label):
    if canonical_json_bytes(a) != canonical_json_bytes(b):
        raise ValueError(f'{label} differs')


def _harmony(g):
    """Replay full continuation boundaries without checkpoint/tokenizer loading."""
    p = g.provenance
    try:
        contract = HarmonyTokenContract(**p['channel_contract'])
        _same(p['channel_contract_sha256'], sha256_json(p['channel_contract']), 'Harmony contract hash')
        _same(p['channel_policy_sha256'], sha256_json({'policy': 'harmony_no_tools',
              'contract_sha256': p['channel_contract_sha256']}), 'Harmony policy hash')
        tracker = HarmonyBoundaryTracker(contract, special_token_ids=_control_ids(contract))
        tracker.observe(g.generated_token_ids)
        expected = dict(final_content_start=tracker.final_content_start,
            final_content_end=tracker.final_content_end,
            analysis_segments=[list(x) for x in tracker.analysis_segments],
            generated_token_count=len(g.generated_token_ids),
            final_token_count=(0 if tracker.final_content_start is None else
                len(g.generated_token_ids[tracker.final_content_start:tracker.final_content_end])))
        for name, value in expected.items():
            _same(p[name], value, 'Harmony ' + name)
        if g.failure_type is None:
            if not tracker.is_terminated or g.finish_reason != 'eos':
                raise ValueError('successful Harmony output must terminate the full turn')
            # No arbitrary stripping or regex rescue. The final text must end in the
            # exact saved payload and native final-stop spelling.
            marker = '<|channel|>final<|message|>'
            if marker not in g.generated_text:
                raise ValueError('missing Harmony final text header')
            final = g.generated_text.split(marker, 1)[1]
            if final != g.json_payload + '<|return|>':
                raise ValueError('Harmony full text differs from final payload')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('malformed Harmony provenance') from exc


@dataclass(frozen=True, slots=True)
class MergedBaseline(CompletedBaseline):
    """CompletedBaseline-compatible effective rows, with explicit mixed provenance.

    The plan remains the original key/role plan, not a claim of homogeneous
    generation. parent_sha256 is the merged content hash, not the original hash.
    """
    _sources_bytes: bytes = field(repr=False)
    _policies_bytes: bytes = field(repr=False)
    _source_records: MappingProxyType = field(repr=False)
    _policy_records: MappingProxyType = field(repr=False)

    @property
    def content_sha256(self):
        return self.parent_sha256

    def row_source_for(self, key):
        self.generation_for(key)  # Same strict unknown-key boundary.
        return _parse(self._source_records[sha256_json(key.to_dict())])

    def row_policy_for(self, key):
        self.generation_for(key)
        return _parse(self._policy_records[sha256_json(key.to_dict())])


def load_merged_baseline(original, recovery, *, inputs) -> MergedBaseline:
    """LC0 full original plus exact31 completed recovery records, read-only."""
    parent = load_completed_baseline(original, inputs=inputs)
    old = parent.metadata['bindings']['generation_policy']['policy']
    selected = tuple(sorted(r.key for r in parent.rows if r.outcome.failure_type == 'truncated'
                            and r.outcome.finish_reason == 'token_budget'))
    expected = tuple(sorted(k for k in parent.plan.keys if (k.ticker, k.condition) in RECOVERY_PAIRS))
    if len(selected) != 31 or selected != expected:
        raise ValueError('original must have the exact registered31 recovery keys')
    if any(r.outcome.failure_type is not None and r.key not in selected for r in parent.rows):
        raise ValueError('unresolved original failure outside recovery selection')
    for row in parent.rows:
        g = parent.generation_for(row.key)
        if old['channel_policy'] == 'harmony_no_tools': _harmony(g)
    root = _open_root(recovery)
    try:
        if set(os.listdir(root)) != {'registration.json', 'summary.json', 'store.lock', 'records'}:
            raise ValueError('recovery requires exact completed layout')
        lock = _open_regular(root, 'store.lock')
        try:
            try: fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise ValueError('active recovery writer') from exc
            held, entry = os.fstat(lock), os.stat('store.lock', dir_fd=root, follow_symlinks=False)
            if (held.st_dev, held.st_ino) != (entry.st_dev, entry.st_ino):
                raise ValueError('recovery lock inode changed')
            raw = {n: _read_file(root, n) for n in ('registration.json', 'summary.json')}
            reg, report = (_parse(raw[n]) for n in ('registration.json', 'summary.json'))
            fields = {'kind','research_eligible','parent_sha256','parent_plan_hash','inputs_manifest_sha256',
                      'selected_keys','original_policy','recovery_policy','metadata'}
            if type(reg) is not dict or set(reg) != fields: raise ValueError('recovery registration fields')
            for n, value in dict(kind='adaptive_budget_baseline_recovery_v1', research_eligible=False,
                parent_sha256=parent.parent_sha256, parent_plan_hash=parent.plan.plan_hash,
                inputs_manifest_sha256=inputs.manifest_sha256, selected_keys=[k.to_dict() for k in selected],
                original_policy=old).items(): _same(reg[n], value, 'registration ' + n)
            policy = _validated_policy(reg['recovery_policy'])
            if policy.max_new_tokens <= old['max_new_tokens']:
                raise ValueError('recovery requires a higher token budget')
            for n in ('use_cache', 'pad_token_id', 'channel_policy'):
                _same(reg['recovery_policy'][n], old[n], 'recovery policy ' + n)
            _same(reg['metadata']['model']['metadata_file_sha256'],
                  parent.metadata['bindings']['model']['metadata_file_sha256'], 'checkpoint metadata')
            fd = _open_directory(root, 'records')
            replacements = {}
            try:
                names = {sha256_json(k.to_dict()) + '.json': k for k in selected}
                if set(os.listdir(fd)) != set(names): raise ValueError('recovery exact31 record coverage')
                for name, key in names.items():
                    data = _read_file(fd, name); raw['records/' + name] = data
                    item = _parse(data); g = _generation(item['generation'])
                    status = _validate_generation_result(g)
                    if g.failure_type is not None: raise ValueError('unresolved recovery generation')
                    _validated_generation_payload(key, g)
                    ref = parent.generation_for(key).provenance
                    for n in _GRAMMAR:
                        _same(g.provenance.get(n), ref.get(n), 'recovery grammar ' + n)
                    gp = dict(policy=reg['recovery_policy'], hf_controls=_hf_controls(policy, tuple(ref['stop_token_ids'])))
                    # The recovery has its own policy identity, not the original plan identity.
                    plan = replace(parent.plan, identity=replace(parent.plan.identity,
                        generation_policy_sha256=sha256_json(gp)))
                    _validated_provenance(g, None, plan, gp, policy)
                    if len(g.generated_token_ids) > policy.max_new_tokens: raise ValueError('token budget exceeded')
                    if policy.channel_policy == 'harmony_no_tools': _harmony(g)
                    outcome = GenerationOutcome(**(status.to_dict() | dict(ticker=key.ticker,
                        issuer_id=inputs.issuer_by_ticker[key.ticker], condition=key.condition, trial_id=key.trial_id)))
                    expected_item = dict(key=key.to_dict(), parent_sha256=parent.parent_sha256,
                        original_generated_token_sha256=parent.generation_for(key).generated_token_sha256,
                        outcome=outcome.to_dict(), generation=g.to_dict())
                    _same(item, expected_item, 'recovery record')
                    replacements[key] = (ExecutionRow(key, outcome), g)
            finally: os.close(fd)
            expected_report = dict(kind='adaptive_budget_baseline_recovery_v1', research_eligible=False,
                homogeneous_original_policy=False, parent_sha256=parent.parent_sha256,
                original_planned=2012, selected_truncated=31, executed=31, complete=True,
                adaptive_selection='only_parent_truncated_token_budget', original_policy=old,
                recovery_policy=reg['recovery_policy'],
                class_counts=dict(Counter(g.decision for _, g in replacements.values())),
                failure_counts={'none':31}, original_failure_counts=parent.summary['failure_counts'])
            _same(report, expected_report, 'recovery summary')
        finally: os.close(lock)
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError('malformed recovery artifact') from exc
    finally: os.close(root)
    recovery_files = {n: sha256_bytes(b) for n, b in sorted(raw.items())}
    recovery_hash = sha256_json({'kind':'baseline_recovery_source_v1', 'files':recovery_files})
    metadata = dict(kind='effective_merged_baseline_v1', mixed_generation_policy=True,
        homogeneous_original_policy=False, replacement_rule=_RULE,
        original_parent_sha256=parent.parent_sha256, recovery_sha256=recovery_hash,
        original_metadata=parent.metadata, recovery_registration=reg,
        original_files=parent.file_sha256, recovery_files=recovery_files)
    rows, records, sources, policies = [], {}, {}, {}
    for row in parent.rows:
        key = row.key; digest = sha256_json(key.to_dict()); name = digest + '.json'
        if key in replacements:
            row, g = replacements[key]; origin, source_hash = 'recovery', recovery_hash
            file_hash = recovery_files['records/' + name]
        else:
            g = parent.generation_for(key); origin, source_hash = 'original', parent.parent_sha256
            file_hash = parent.file_sha256['shard-0-of-1/records/' + name]
        rows.append(row)
        records[name] = canonical_json_bytes(dict(schema_version=1, plan_hash=parent.plan.plan_hash,
            shard_index=0, num_shards=1, row=row.to_dict(), generation=g.to_dict())) + b'\n'
        sources[digest] = dict(origin=origin, source_sha256=source_hash, record_sha256=file_hash)
        policies[digest] = g.provenance['generation_policy']
    counts = dict(Counter(r.outcome.decision for r in rows))
    if counts != {'buy':518, 'sell':1494}: raise ValueError('effective decisions differ from518buy/1494sell')
    summary = parent.summary
    summary.update(class_counts=counts, failure_counts={'none':2012},
                   finish_counts=dict(Counter(r.outcome.finish_reason for r in rows)))
    content_hash = sha256_json(dict(metadata=metadata, plan=parent.plan.to_dict(),
        effective_records={n:sha256_bytes(b) for n,b in sorted(records.items())}, sources=sources, policies=policies))
    encode = lambda v: canonical_json_bytes(v) + b'\n'
    return MergedBaseline(parent.plan, tuple(rows), content_hash, encode(metadata), encode(summary),
        canonical_json_bytes({'original':parent.file_sha256, 'recovery':recovery_files}),
        MappingProxyType(records), encode(sources), encode(policies),
        MappingProxyType({k:encode(v) for k,v in sources.items()}),
        MappingProxyType({k:encode(v) for k,v in policies.items()}))


def materialize_merged_baseline(view: MergedBaseline, output) -> None:
    """Fresh self-contained export, never a standard homogeneous LC0 artifact."""
    if type(view) is not MergedBaseline: raise ValueError('expected MergedBaseline')
    output = Path(output)
    try: output.mkdir()
    except FileExistsError as exc: raise ValueError('merged output already exists') from exc
    (output / 'records').mkdir()
    payloads = {'metadata.json':view._metadata_bytes, 'summary.json':view._summary_bytes,
        'row_sources.json':view._sources_bytes, 'row_policies.json':view._policies_bytes,
        'plan.json':canonical_json_bytes(view.plan.to_dict()) + b'\n'}
    payloads.update({'records/' + n:b for n,b in view._record_bytes.items()})
    for name, raw in payloads.items():
        with (output / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    manifest = dict(kind='effective_merged_baseline_materialization_v1',
        content_sha256=view.content_sha256, files={n:sha256_bytes(b) for n,b in sorted(payloads.items())})
    with (output / 'manifest.json').open('xb') as stream:
        stream.write(canonical_json_bytes(manifest) + b'\n'); stream.flush(); os.fsync(stream.fileno())
