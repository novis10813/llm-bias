"""Tensor-free full baseline planning; construction does not certify any gate."""
from __future__ import annotations

from dataclasses import asdict, fields

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.experiment_contract import ExperimentPlan, PlanIdentity, RowKey
from llm_bias.core.population import PopulationMember, validate_roles
from llm_bias.core.stance_baseline_inputs import BaselineInputs
from llm_bias.core.stance_evidence import EvidenceItem, EvidencePair

_MANIFEST_SHA256 = 'b3ee1d47990c9becce37e2eafe4739b9ab1d2544894e3fdfe5febec77819168d'
_GATES = ('input_integrity', 'model_binding', 'no_op', 'protocol_frozen')


def _scalar_record(value, cls) -> None:
    if type(value) is not cls or any(type(getattr(value, f.name)) is not str for f in fields(cls)):
        raise ValueError(f'expected frozen scalar-string {cls.__name__}')


def _validate_baseline_inputs(inputs: BaselineInputs) -> None:
    """Reconstruct the approved bytes, not an attestation of the constructor."""
    if type(inputs) is not BaselineInputs:
        raise ValueError('expected BaselineInputs')
    try:
        if (type(inputs.members) is not tuple or type(inputs.pairs) is not tuple
                or len(inputs.members) != 503 or len(inputs.pairs) != 2012):
            raise ValueError('expected immutable full 503-member/2012-pair bundle')
        for member in inputs.members:
            _scalar_record(member, PopulationMember)
        for pair in inputs.pairs:
            if type(pair) is not EvidencePair:
                raise ValueError('expected frozen EvidencePair')
            for name in ('ticker', 'condition', 'trial_id'):
                if type(getattr(pair, name)) is not str:
                    raise ValueError('pair identity fields must be scalar strings')
            _scalar_record(pair.evidence1, EvidenceItem)
            _scalar_record(pair.evidence2, EvidenceItem)
        for name in ('membership_sha256', 'issuer_mapping_sha256', 'roles_sha256',
                     'evidence_sha256', 'manifest_sha256'):
            if type(getattr(inputs, name)) is not str:
                raise ValueError('input hashes must be strings')
        if type(inputs._roles_bytes) is not bytes or type(inputs._manifest_bytes) is not bytes:
            raise ValueError('exports must have immutable byte storage')
        manifest, roles = inputs.manifest, inputs.roles
        manifest_raw = canonical_json_bytes(manifest) + b'\n'
        if (sha256_bytes(manifest_raw) != _MANIFEST_SHA256
                or inputs.manifest_sha256 != _MANIFEST_SHA256):
            raise ValueError('approved manifest identity mismatch')
        member_rows = [asdict(member) for member in inputs.members]
        population = dict(members=member_rows, source_sha256=manifest['population_source_sha256'],
                          membership_sha256=inputs.membership_sha256)
        payloads = {
            'population.json': canonical_json_bytes(population) + b'\n',
            'roles.json': canonical_json_bytes(roles) + b'\n',
            'evidence_pairs.jsonl': b''.join(canonical_json_bytes(asdict(pair)) + b'\n'
                                           for pair in inputs.pairs),
        }
        for name, raw in payloads.items():
            if sha256_bytes(raw) != manifest['output_hashes'][name]:
                raise ValueError(f'approved {name} identity mismatch')
        membership_hash = sha256_json(member_rows)
        issuer_hash = sha256_json({m.ticker: m.issuer_id for m in inputs.members})
        roles_hash = validate_roles(inputs.members, roles['roles']).assignment_hash
        semantic = manifest['semantic_hashes']
        if (membership_hash != inputs.membership_sha256
                or membership_hash != semantic['membership_hash']
                or issuer_hash != inputs.issuer_mapping_sha256
                or issuer_hash != semantic['issuer_mapping_hash']
                or roles_hash != inputs.roles_sha256 or roles_hash != roles['assignment_hash']
                or roles_hash != semantic['roles_hash']
                or inputs.evidence_sha256 != sha256_bytes(payloads['evidence_pairs.jsonl'])):
            raise ValueError('input semantic identity mismatch')
    except (AttributeError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError('malformed baseline input bundle') from exc


def build_baseline_plan(inputs: BaselineInputs, identity: PlanIdentity) -> ExperimentPlan:
    """Bind all approved pairs and four gate names, without asserting gate truth.

    Non-input identity claims remain supplied, unverified policy/model claims.
    No cohort, stage, dose, gate or protocol overrides are supported.
    """
    _validate_baseline_inputs(inputs)
    if not isinstance(identity, PlanIdentity):
        raise ValueError('expected PlanIdentity')
    identity = PlanIdentity(**{f.name: getattr(identity, f.name) for f in fields(PlanIdentity)})
    for plan_field, input_field in (
        ('population_sha256', 'membership_sha256'), ('issuer_sha256', 'issuer_mapping_sha256'),
        ('roles_sha256', 'roles_sha256'), ('evidence_sha256', 'evidence_sha256'),
    ):
        if getattr(identity, plan_field) != getattr(inputs, input_field):
            raise ValueError(f'{plan_field} differs from approved input identity')
    if identity.operator_sha256 != 'not_applicable' or identity.parent_sha256 != 'not_applicable':
        raise ValueError('baseline operator and parent must be not_applicable')
    keys = tuple(RowKey('baseline', p.ticker, p.condition, p.trial_id, 'baseline', '0')
                 for p in inputs.pairs)
    return ExperimentPlan(identity, keys, _GATES)
