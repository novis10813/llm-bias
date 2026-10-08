"""Read-only complete baseline parent snapshot for generated localization.

The loader validates one immutable full-population baseline root (single shard
``shard-0-of-1``) exactly once and returns a fully materialized, tensor-free
snapshot. It never opens the writable store, never creates or alters any file,
rebuilds the approved plan exactly once, and never calls the per-row baseline
adapter. Recorded provenance is checked for consistency of its claims only;
this consumer does not certify live checkpoint contents, prompt authenticity
or research eligibility.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import stat
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.experiment_contract import (
    ExecutionRow, ExperimentPlan, GenerationOutcome, RowKey)
from llm_bias.core.inference.structured_output import (
    StructuredGenerationPolicy, StructuredGenerationResult, validate_decision_payload)
from llm_bias.core.prompt_input.decision_prompt import (
    DECISION_INSTRUCTION, DECISION_SCHEMA, DECISION_TEMPLATE, _ENTITY_TEMPLATE)
from llm_bias.core.stance_baseline_adapter import _validate_generation_result
from llm_bias.core.stance_baseline_inputs import BaselineInputs
from llm_bias.core.stance_baseline_plan import build_baseline_plan
from llm_bias.core.stance_baseline_store import _generation, _parse

_KIND = 'completed_baseline_parent_v1'
_METADATA_FILE = 'shard-0-of-1.execution_metadata.json'
_SUMMARY_FILE = 'shard-0-of-1.summary.json'
_SHARD = 'shard-0-of-1'
_ROOT_ENTRIES = frozenset((_METADATA_FILE, _SUMMARY_FILE, _SHARD))
_SHARD_ENTRIES = frozenset(('registration.json', 'store.lock', 'records'))
_METADATA_FIELDS = frozenset(('identity', 'bindings', 'inputs_manifest_sha256'))
_BINDING_FIELDS = frozenset(('protocol', 'template', 'generation_policy', 'model', 'code', 'backend'))
_PROTOCOL_FIELDS = frozenset(('kind', 'research_eligible', 'planned_global', 'members',
                              'role_assignment', 'evidence', 'sampling',
                              'timeout_budget_seconds_per_row', 'model_route',
                              'efficacy_claim', 'protocol_freeze_claim'))
_TEMPLATE_FIELDS = frozenset(('body_template_sha256', 'actual_wrapper_record'))
_POLICY_FIELDS = frozenset(('max_new_tokens', 'use_cache', 'pad_token_id',
                            'timeout_seconds', 'channel_policy'))
_ENVELOPE_FIELDS = frozenset(('schema_version', 'plan_hash', 'shard_index', 'num_shards',
                              'row', 'generation'))
_REGISTRATION_FIELDS = frozenset(('schema_version', 'plan', 'inputs_manifest_sha256',
                                  'shard_index', 'num_shards'))
_PAYLOAD_FAILURES = frozenset(('invalid_json', 'invalid_schema', 'invalid_reason'))
_PROVENANCE_REQUIRED = ('backend', 'backend_version', 'byte_policy', 'compiler_policy',
                        'grammar_correction', 'grammar_sha256', 'generation_policy',
                        'generation_policy_sha256', 'hf_controls', 'head_vocab_size',
                        'model_binding', 'schema_bytes_sha256', 'schema_sha256',
                        'stop_token_ids', 'tokenizer_info_sha256', 'tokenizer_sha256')
_PROVENANCE_CONSISTENT = ('tokenizer_sha256', 'tokenizer_info_sha256', 'schema_bytes_sha256',
                          'grammar_sha256', 'head_vocab_size', 'stop_token_ids', 'hf_controls')
_BINDING_HASHES = (('protocol', 'protocol_sha256'), ('template', 'template_sha256'),
                   ('generation_policy', 'generation_policy_sha256'), ('model', 'model_sha256'),
                   ('code', 'code_sha256'), ('backend', 'backend_sha256'))


def _schema_sha256() -> str:
    return sha256_json({'schema': DECISION_SCHEMA, 'key_order': ['decision', 'reason']})


def _body_template_sha256() -> str:
    return sha256_json({'template': DECISION_TEMPLATE, 'entity_template': _ENTITY_TEMPLATE,
                        'instruction': DECISION_INSTRUCTION})


def _open_root(directory: str | Path):
    try:
        return os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError('parent root must be an existing nonsymlink directory') from exc


def _open_directory(directory_fd, name):
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f'invalid parent directory: {name}') from exc


def _open_regular(directory_fd, name):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, 0, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f'invalid parent file: {name}') from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f'nonregular parent file: {name}')
    return fd


def _read_file(directory_fd, name) -> bytes:
    with os.fdopen(_open_regular(directory_fd, name), 'rb') as stream:
        return stream.read()


def _validated_registration(registration, inputs: BaselineInputs) -> ExperimentPlan:
    if type(registration) is not dict or set(registration) != _REGISTRATION_FIELDS:
        raise ValueError('registration must have the exact shard fields')
    if type(registration['schema_version']) is not int or registration['schema_version'] != 1:
        raise ValueError('registration schema_version must be the integer 1')
    if type(registration['shard_index']) is not int or registration['shard_index'] != 0:
        raise ValueError('parent requires single shard index 0')
    if type(registration['num_shards']) is not int or registration['num_shards'] != 1:
        raise ValueError('parent requires exactly one shard')
    if registration['inputs_manifest_sha256'] != inputs.manifest_sha256:
        raise ValueError('registration pins a different input manifest')
    plan = ExperimentPlan.from_dict(registration['plan'])
    rebuilt = build_baseline_plan(inputs, plan.identity)
    if rebuilt.to_json() != plan.to_json():
        raise ValueError('registration plan differs from approved complete baseline plan')
    return plan


def _validated_policy(record) -> StructuredGenerationPolicy:
    if type(record) is not dict or set(record) != _POLICY_FIELDS:
        raise ValueError('generation policy must have the exact five fields')
    if type(record['max_new_tokens']) is not int or record['max_new_tokens'] <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    if type(record['use_cache']) is not bool:
        raise ValueError('use_cache must be boolean')
    if type(record['pad_token_id']) is not int or record['pad_token_id'] < 0:
        raise ValueError('pad_token_id must be a nonnegative integer')
    timeout = record['timeout_seconds']
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout_seconds must be positive and finite')
    channel = record['channel_policy']
    if channel not in ('plain_json', 'harmony_no_tools'):
        raise ValueError('channel_policy must be a supported baseline route')
    return StructuredGenerationPolicy(record['max_new_tokens'], record['use_cache'],
                                      record['pad_token_id'], timeout, channel)


def _validated_metadata(metadata, plan: ExperimentPlan, inputs: BaselineInputs):
    if type(metadata) is not dict or set(metadata) != _METADATA_FIELDS:
        raise ValueError('execution metadata must have identity, bindings and inputs_manifest_sha256')
    if canonical_json_bytes(metadata['identity']) != canonical_json_bytes(plan.identity.to_dict()):
        raise ValueError('metadata identity differs from registered plan identity')
    if metadata['inputs_manifest_sha256'] != inputs.manifest_sha256:
        raise ValueError('metadata pins a different input manifest')
    bindings = metadata['bindings']
    if type(bindings) is not dict or set(bindings) != _BINDING_FIELDS:
        raise ValueError('bindings must have the exact six binding records')
    for record_name, identity_name in _BINDING_HASHES:
        if sha256_json(bindings[record_name]) != getattr(plan.identity, identity_name):
            raise ValueError(f'{record_name} binding hash differs from plan identity')
    if plan.identity.schema_sha256 != _schema_sha256():
        raise ValueError('plan schema hash differs from canonical decision schema')
    template = bindings['template']
    if type(template) is not dict or set(template) != _TEMPLATE_FIELDS:
        raise ValueError('template binding must have body_template_sha256 and actual_wrapper_record')
    if template['body_template_sha256'] != _body_template_sha256():
        raise ValueError('body template hash differs from decision prompt template')
    if not isinstance(template['actual_wrapper_record'], dict):
        raise ValueError('actual_wrapper_record must be a JSON object')
    gp = bindings['generation_policy']
    if type(gp) is not dict or set(gp) != {'policy', 'hf_controls'}:
        raise ValueError('generation_policy binding must have policy and hf_controls')
    policy = _validated_policy(gp['policy'])
    expected_protocol = {'kind': 'diagnostic_full_baseline_v1', 'research_eligible': False,
                         'planned_global': 2012, 'members': 503,
                         'role_assignment': inputs.roles_sha256, 'evidence': inputs.evidence_sha256,
                         'sampling': 'all_approved_pairs_once',
                         'timeout_budget_seconds_per_row': policy.timeout_seconds,
                         'model_route': policy.channel_policy, 'efficacy_claim': False,
                         'protocol_freeze_claim': False}
    protocol = bindings['protocol']
    if (type(protocol) is not dict or set(protocol) != _PROTOCOL_FIELDS
            or canonical_json_bytes(protocol) != canonical_json_bytes(expected_protocol)):
        raise ValueError('protocol record differs from diagnostic full baseline protocol')
    return policy, gp


def _validated_generation_payload(key: RowKey, generation: StructuredGenerationResult) -> None:
    parsed = validate_decision_payload(generation.json_payload if generation.decode_error is None
                                       else '')
    if ((generation.decision_complete, generation.schema_complete, generation.reason_valid)
            != (parsed.decision_complete, parsed.schema_complete, parsed.reason_valid)):
        raise ValueError(f'diagnostic flags differ from strict payload parse: {key}')
    if generation.failure_type is None:
        if parsed.failure_type is not None:
            raise ValueError(f'success generation has a parse failure: {key}')
        if (generation.decision, generation.reason) != (parsed.decision, parsed.reason):
            raise ValueError(f'success generation differs from parsed decision/reason: {key}')
    else:
        if generation.decision is not None or generation.reason is not None:
            raise ValueError(f'failed generation must retain null primary values: {key}')
        if generation.failure_type in _PAYLOAD_FAILURES and generation.failure_type != parsed.failure_type:
            raise ValueError(f'payload failure differs from parser failure: {key}')


def _validated_provenance(generation: StructuredGenerationResult, reference, plan, gp,
                          policy: StructuredGenerationPolicy):
    prov = generation.provenance
    for name in _PROVENANCE_REQUIRED:
        if name not in prov:
            raise ValueError(f'provenance missing {name}')
    if prov['schema_sha256'] != plan.identity.schema_sha256:
        raise ValueError('provenance schema hash differs from plan identity')
    if prov['generation_policy_sha256'] != plan.identity.generation_policy_sha256:
        raise ValueError('provenance generation policy hash differs from plan identity')
    if canonical_json_bytes(prov['generation_policy']) != canonical_json_bytes(gp['policy']):
        raise ValueError('provenance generation policy differs from binding')
    head = prov['head_vocab_size']
    if type(head) is not int or head <= 0:
        raise ValueError('head_vocab_size must be a positive integer')
    stops = prov['stop_token_ids']
    if (type(stops) is not list or not stops or len(set(stops)) != len(stops)
            or any(type(i) is not int or not 0 <= i < head for i in stops)):
        raise ValueError('stop token IDs must be unique genuine integers inside the head range')
    if any(not 0 <= token < head for token in generation.generated_token_ids):
        raise ValueError('generated token IDs lie outside the declared head range')
    if policy.pad_token_id >= head:
        raise ValueError('pad_token_id lies outside the declared head range')
    if reference is None:
        if canonical_json_bytes(prov['hf_controls']) != canonical_json_bytes(gp['hf_controls']):
            raise ValueError('provenance HF controls differ from binding')
        return {name: prov[name] for name in _PROVENANCE_CONSISTENT}
    for name in _PROVENANCE_CONSISTENT:
        if canonical_json_bytes(prov[name]) != canonical_json_bytes(reference[name]):
            raise ValueError(f'provenance {name} differs across rows')
    return reference


def _validated_records(records_fd, filename_to_key, plan: ExperimentPlan, plan_hash: str,
                       inputs: BaselineInputs,
                       policy: StructuredGenerationPolicy, gp) -> tuple:
    issuers = {member.ticker: member.issuer_id for member in inputs.members}
    rows: dict[RowKey, ExecutionRow] = {}
    record_bytes: dict[str, bytes] = {}
    reference = None
    for filename in sorted(filename_to_key):
        key = filename_to_key[filename]
        data = _read_file(records_fd, filename)
        envelope = _parse(data)
        if type(envelope) is not dict or set(envelope) != _ENVELOPE_FIELDS:
            raise ValueError(f'invalid record fields: {filename}')
        for name, expected in (('schema_version', 1), ('shard_index', 0), ('num_shards', 1)):
            if type(envelope[name]) is not int or envelope[name] != expected:
                raise ValueError(f'invalid record binding {name}: {filename}')
        if envelope['plan_hash'] != plan_hash:
            raise ValueError(f'foreign plan hash: {filename}')
        row = ExecutionRow.from_dict(envelope['row'])
        if row.key != key:
            raise ValueError(f'record key differs from filename: {filename}')
        generation = _generation(envelope['generation'])
        status = _validate_generation_result(generation)
        outcome = GenerationOutcome(**(status.to_dict() | {
            'ticker': key.ticker, 'issuer_id': issuers[key.ticker],
            'condition': key.condition, 'trial_id': key.trial_id}))
        rebuilt_row = ExecutionRow(key, outcome)
        if canonical_json_bytes(rebuilt_row.to_dict()) != canonical_json_bytes(envelope['row']):
            raise ValueError(f'record row differs from issuer-bound outcome: {filename}')
        _validated_generation_payload(key, generation)
        reference = _validated_provenance(generation, reference, plan, gp, policy)
        rows[key] = rebuilt_row
        record_bytes[filename] = data
    return tuple(rows[key] for key in sorted(rows)), record_bytes


def _validated_summary(summary, plan: ExperimentPlan, plan_hash: str, rows) -> None:
    outcomes = [row.outcome for row in rows]
    computed = {'kind': 'diagnostic_full_baseline_v1', 'research_eligible': False,
                'steering_flips_claimed': False, 'plan_hash': plan_hash,
                'planned_global': 2012, 'shard_index': 0, 'num_shards': 1,
                'assigned_count': len(plan.keys), 'executed_count': len(outcomes),
                'missing_count': 0, 'complete_shard': True, 'complete_global': True,
                'class_counts': dict(Counter(outcome.decision or 'no_decision' for outcome in outcomes)),
                'failure_counts': dict(Counter(outcome.failure_type or 'none' for outcome in outcomes)),
                'finish_counts': dict(Counter(outcome.finish_reason for outcome in outcomes)),
                'gates': {name: False for name in plan.gate_names}}
    if canonical_json_bytes(computed) != canonical_json_bytes(summary):
        raise ValueError('summary differs from recomputed runner summary')


@dataclass(frozen=True, slots=True)
class CompletedBaseline:
    """A fully materialized read-only snapshot of a complete baseline parent.

    Direct construction is trusted caller input and does not attest that
    ``load_completed_baseline`` validated a complete parent. Once loaded, the
    snapshot is immutable: later source-file changes cannot alter it. There is
    no writable method, context manager or residual/tensor payload.
    """

    plan: ExperimentPlan
    rows: tuple[ExecutionRow, ...]
    parent_sha256: str
    _metadata_bytes: bytes = field(repr=False)
    _summary_bytes: bytes = field(repr=False)
    _file_hashes_bytes: bytes = field(repr=False)
    _record_bytes: MappingProxyType = field(repr=False)

    @property
    def metadata(self) -> dict:
        """Defensive strict JSON copy of the execution metadata."""
        return _parse(self._metadata_bytes)

    @property
    def summary(self) -> dict:
        """Defensive strict JSON copy of the runner summary."""
        return _parse(self._summary_bytes)

    @property
    def file_sha256(self) -> dict:
        """Defensive copy of root-relative consumed file digests (no lock)."""
        return json.loads(self._file_hashes_bytes)

    def generation_for(self, key: RowKey) -> StructuredGenerationResult:
        """Re-import the stored canonical generation export; unknown keys reject."""
        if type(key) is not RowKey:
            raise ValueError('expected a RowKey')
        data = self._record_bytes.get(sha256_json(key.to_dict()) + '.json')
        if data is None:
            raise ValueError('unknown baseline key')
        envelope = _parse(data)
        if type(envelope) is not dict or set(envelope) != _ENVELOPE_FIELDS:
            raise ValueError('stored record no longer matches the envelope contract')
        return _generation(envelope['generation'])


def load_completed_baseline(directory: str | Path, *, inputs: BaselineInputs) -> CompletedBaseline:
    """Validate one complete single-shard baseline root read-only, exactly once.

    The approved plan is rebuilt once from the pinned inputs; records are then
    checked linearly against it. An active writable store (exclusive lock) is
    rejected immediately. No file is created or altered.
    """
    if type(inputs) is not BaselineInputs:
        raise ValueError('expected BaselineInputs')
    root_fd = _open_root(directory)
    try:
        try:
            if set(os.listdir(root_fd)) != _ROOT_ENTRIES:
                raise ValueError('parent root must contain exactly the single-shard layout')
            shard_fd = _open_directory(root_fd, _SHARD)
            try:
                if set(os.listdir(shard_fd)) != _SHARD_ENTRIES:
                    raise ValueError('parent shard must contain exactly registration, lock and records')
                lock_fd = _open_regular(shard_fd, 'store.lock')
                try:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise ValueError('baseline parent lock is held by a writable store') from exc
                    held = os.fstat(lock_fd)
                    entry = os.stat('store.lock', dir_fd=shard_fd, follow_symlinks=False)
                    if (not stat.S_ISREG(entry.st_mode)
                            or (held.st_dev, held.st_ino) != (entry.st_dev, entry.st_ino)):
                        raise ValueError('baseline parent lock inode changed during acquisition')
                    metadata_raw = _read_file(root_fd, _METADATA_FILE)
                    summary_raw = _read_file(root_fd, _SUMMARY_FILE)
                    registration_raw = _read_file(shard_fd, 'registration.json')
                    metadata = _parse(metadata_raw)
                    summary = _parse(summary_raw)
                    plan = _validated_registration(_parse(registration_raw), inputs)
                    policy, gp = _validated_metadata(metadata, plan, inputs)
                    records_fd = _open_directory(shard_fd, 'records')
                    try:
                        listing = os.listdir(records_fd)
                        if any(name.startswith('.pending-') for name in listing):
                            raise ValueError('incomplete parent: staging leftover')
                        filename_to_key = {sha256_json(key.to_dict()) + '.json': key
                                           for key in plan.keys}
                        if set(listing) != set(filename_to_key):
                            raise ValueError('records must be exactly the complete 2012-key set')
                        rows, record_bytes = _validated_records(
                            records_fd, filename_to_key, plan, plan.plan_hash, inputs, policy, gp)
                        _validated_summary(summary, plan, plan.plan_hash, rows)
                    finally:
                        os.close(records_fd)
                finally:
                    os.close(lock_fd)
            finally:
                os.close(shard_fd)
        except OSError as exc:
            raise ValueError('invalid or incomplete baseline parent filesystem') from exc
        files = {_METADATA_FILE: sha256_bytes(metadata_raw),
                 _SUMMARY_FILE: sha256_bytes(summary_raw),
                 _SHARD + '/registration.json': sha256_bytes(registration_raw)}
        files.update({_SHARD + '/records/' + name: sha256_bytes(data)
                      for name, data in sorted(record_bytes.items())})
        parent_sha256 = sha256_json({'kind': _KIND, 'files': files})
        return CompletedBaseline(plan, rows, parent_sha256, metadata_raw, summary_raw,
                                 canonical_json_bytes(files), MappingProxyType(record_bytes))
    finally:
        os.close(root_fd)


__all__ = ['CompletedBaseline', 'load_completed_baseline']
