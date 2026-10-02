"""Full-population, role-preserving localization pairs; labels never select donors."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .artifact_paths import canonical_json_bytes, sha256_json
from .experiment_contract import ExecutionRow, RowKey, GenerationOutcome
from .stance_baseline_inputs import BaselineInputs
from .stance_baseline_parent import CompletedBaseline, _validated_summary
from .stance_baseline_plan import build_baseline_plan

_ROLES = ('fit', 'validation', 'calibration', 'evaluation')
_RECIPROCAL = {'++': '--', '--': '++', '+-': '-+', '-+': '+-'}
_CONTRASTS = {'cross_company': 'entity_context', 'cross_evidence': {
    '++': 'polarity_content', '--': 'polarity_content', '+-': 'evidence_order', '-+': 'evidence_order'}}


def pairing_policy() -> dict:
    return {'kind': 'localization_pairs_v1', 'seed': 20261002,
            'cross_company': 'role_issuer_hash_ring_next_lexicographic_ticker',
            'cross_evidence': dict(_RECIPROCAL),
            'contrast_names': json.loads(canonical_json_bytes(_CONTRASTS))}


def _hash(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected a lowercase SHA-256')


def _baseline_key(key):
    if type(key) is not RowKey or (key.stage, key.arm, key.dose) != ('baseline', 'baseline', '0'):
        raise ValueError('expected a baseline key')
    if RowKey.from_dict(key.to_dict()) != key:
        raise ValueError('invalid baseline key')


@dataclass(frozen=True, slots=True)
class LocalizationPair:
    target_key: RowKey
    donor_key: RowKey
    role: str
    family: str
    contrast: str
    clean_relation: str
    pair_sha256: str

    def __post_init__(self):
        _baseline_key(self.target_key); _baseline_key(self.donor_key)
        if self.role not in _ROLES or self.family not in _CONTRASTS:
            raise ValueError('unknown role or pair family')
        if self.clean_relation not in ('same', 'opposite', 'invalid_parent'):
            raise ValueError('unknown clean relation')
        t, d = self.target_key, self.donor_key
        if t.trial_id != d.trial_id:
            raise ValueError('pairs must preserve trial')
        if self.family == 'cross_company':
            if t.ticker == d.ticker or t.condition != d.condition or self.contrast != 'entity_context':
                raise ValueError('invalid cross-company contrast')
        elif (t.ticker != d.ticker or _RECIPROCAL[t.condition] != d.condition
              or self.contrast != _CONTRASTS['cross_evidence'][t.condition]):
            raise ValueError('invalid cross-evidence contrast')
        _hash(self.pair_sha256)
        if self.pair_sha256 != sha256_json(self._payload()):
            raise ValueError('pair hash mismatch')

    def _payload(self):
        return dict(target_key=self.target_key.to_dict(), donor_key=self.donor_key.to_dict(),
                    role=self.role, family=self.family, contrast=self.contrast,
                    clean_relation=self.clean_relation)

    def to_dict(self):
        return self._payload() | {'pair_sha256': self.pair_sha256}


def _order(pair):
    return pair.target_key, pair.family, pair.donor_key


@dataclass(frozen=True, slots=True)
class LocalizationPairTable:
    parent_sha256: str
    inputs_manifest_sha256: str
    policy_sha256: str
    pairs: tuple[LocalizationPair, ...]
    table_sha256: str

    def __post_init__(self):
        for digest in (self.parent_sha256, self.inputs_manifest_sha256,
                       self.policy_sha256, self.table_sha256):
            _hash(digest)
        if (type(self.pairs) is not tuple or not self.pairs
                or any(type(p) is not LocalizationPair for p in self.pairs)):
            raise ValueError('pairs must be a nonempty typed tuple')
        if tuple(sorted(self.pairs, key=_order)) != self.pairs:
            raise ValueError('pairs must be canonically sorted')
        if len({(p.target_key, p.family) for p in self.pairs}) != len(self.pairs):
            raise ValueError('duplicate pair target/family')
        for p in self.pairs:
            p.__post_init__()
        if self.table_sha256 != sha256_json(self._payload()):
            raise ValueError('pair table hash mismatch')

    def _payload(self):
        return dict(parent_sha256=self.parent_sha256, inputs_manifest_sha256=self.inputs_manifest_sha256,
                    policy_sha256=self.policy_sha256, pairs=[p.to_dict() for p in self.pairs])

    def to_dict(self):
        return self._payload() | {'table_sha256': self.table_sha256}


def build_localization_pairs(inputs: BaselineInputs, parent: CompletedBaseline) -> LocalizationPairTable:
    """Validate the full structural parent once, then pair without inspecting labels for selection.

    LC0 owns generation and file integrity. Direct CompletedBaseline construction
    remains trusted caller input, not loader authentication.
    """
    if not isinstance(inputs, BaselineInputs) or not isinstance(parent, CompletedBaseline):
        raise ValueError('expected approved inputs and a completed baseline snapshot')
    _hash(parent.parent_sha256)
    rebuilt = build_baseline_plan(inputs, parent.plan.identity)
    if rebuilt.to_json() != parent.plan.to_json():
        raise ValueError('parent plan differs from the complete approved baseline')
    issuers = inputs.issuer_by_ticker
    roles = inputs.roles['assignments']
    if type(parent.rows) is not tuple or len(parent.rows) != len(rebuilt.keys):
        raise ValueError('incomplete parent rows')
    rows = {}
    for row in parent.rows:
        if type(row) is not ExecutionRow:
            raise ValueError('invalid parent row')
        _baseline_key(row.key)
        if row.key in rows:
            raise ValueError('duplicate parent key')
        validated = GenerationOutcome.from_dict(row.outcome.to_dict())
        expected = {'ticker': row.key.ticker, 'issuer_id': issuers.get(row.key.ticker),
                    'condition': row.key.condition, 'trial_id': row.key.trial_id}
        if any(validated.to_dict()[name] != value for name, value in expected.items()):
            raise ValueError('parent row population/key binding differs')
        rows[row.key] = validated
    if set(rows) != set(rebuilt.keys):
        raise ValueError('parent key coverage differs')
    _validated_summary(parent.summary, rebuilt, rebuilt.plan_hash, parent.rows)

    groups = {role: {} for role in _ROLES}
    for member in inputs.members:
        groups[roles[member.ticker]].setdefault(member.issuer_id, []).append(member.ticker)
    donor_tickers = {}
    for role, issuer_groups in groups.items():
        if len(issuer_groups) < 2:
            raise ValueError('cross-company pairing needs two issuers in every role')
        ring = sorted(issuer_groups, key=lambda issuer: (sha256_json(
            {'seed': 20261002, 'role': role, 'issuer_id': issuer}), issuer))
        for index, issuer in enumerate(ring):
            donor_tickers[issuer] = min(issuer_groups[ring[(index + 1) % len(ring)]])
    pairs = []
    for target in rebuilt.keys:
        role = roles[target.ticker]
        for family in ('cross_company', 'cross_evidence'):
            ticker = donor_tickers[issuers[target.ticker]] if family == 'cross_company' else target.ticker
            condition = target.condition if family == 'cross_company' else _RECIPROCAL[target.condition]
            donor = RowKey('baseline', ticker, condition, target.trial_id, 'baseline', '0')
            if donor not in rows or roles[donor.ticker] != role:
                raise ValueError('missing or cross-role donor')
            if family == 'cross_company' and issuers[ticker] == issuers[target.ticker]:
                raise ValueError('cross-company donor shares target issuer')
            t, d = rows[target], rows[donor]
            relation = ('invalid_parent' if not (t.primary_valid and d.primary_valid)
                        else 'same' if t.decision == d.decision else 'opposite')
            contrast = 'entity_context' if family == 'cross_company' else _CONTRASTS[family][target.condition]
            payload = dict(target_key=target.to_dict(), donor_key=donor.to_dict(), role=role,
                           family=family, contrast=contrast, clean_relation=relation)
            pairs.append(LocalizationPair(target, donor, role, family, contrast, relation, sha256_json(payload)))
    ordered = tuple(sorted(pairs, key=_order))
    payload = dict(parent_sha256=parent.parent_sha256, inputs_manifest_sha256=inputs.manifest_sha256,
                   policy_sha256=sha256_json(pairing_policy()), pairs=[p.to_dict() for p in ordered])
    return LocalizationPairTable(parent.parent_sha256, inputs.manifest_sha256,
                                 payload['policy_sha256'], ordered, sha256_json(payload))
