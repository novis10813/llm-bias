"""Strict, model-independent capability for the approved full baseline input pack."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.population import PopulationMember, assign_roles, validate_members, validate_roles
from llm_bias.core.stance_evidence import EvidenceItem, EvidencePair, EvidenceOrderContrast, validate_evidence_pairs

_MANIFEST_SHA256 = 'b3ee1d47990c9becce37e2eafe4739b9ab1d2544894e3fdfe5febec77819168d'
_FILES = ('population.json', 'roles.json', 'evidence_pairs.jsonl', 'inputs_manifest.json')
_ISSUER_COUNTS = dict(fit=300, validation=75, calibration=25, evaluation=100)
_TICKER_COUNTS = dict(fit=302, validation=75, calibration=26, evaluation=100)
_TRIAL = 'factset-20241115-a'
_SOURCE = '677776cd01ddf02efcfb5fb4b437c336baa825f0dd41f3cdf9f3b1b829078486'
_URL = 'https://advantage.factset.com/hubfs/Website/Resources%20Section/Research%20Desk/Earnings%20Insight/EarningsInsight_111524.pdf'
_REVIEW = 'factset-20241115-machine-factual-v1'
_POPULATION_SOURCE = '27c454d250ce513fda2016b43e7009ccbe0f5381e6c7648d3efe0d418c501a38'


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def _parse_json(raw: bytes):
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_constant=_nonfinite)
        if canonical_json_bytes(value) + b'\n' != raw:
            raise ValueError('noncanonical JSON serialization')
        return value
    except (UnicodeError, TypeError) as exc:
        raise ValueError('invalid UTF8 JSON') from exc


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'expected exact fields: {tuple(names)}')


def _record(cls, value):
    _fields(value, (f.name for f in fields(cls)))
    return cls(**value)


def _validate_documents(population, roles, rows, manifest):
    """Pure full-cohort semantic boundary; never authorizes production bytes."""
    _fields(population, ('members', 'source_sha256', 'membership_sha256'))
    _fields(manifest, ('schema_version', 'config_sha256', 'population_source_sha256',
                      'evidence_source_sha256', 'code_sha256', 'output_hashes', 'counts',
                      'semantic_hashes', 'eligibility', 'provenance'))
    _fields(manifest['output_hashes'], _FILES[:3])
    _fields(manifest['code_sha256'], ('llm_bias/core/artifact_paths.py', 'llm_bias/core/population.py',
                                    'llm_bias/core/stance_evidence.py', 'scripts/compile_stance_inputs.py'))
    _fields(manifest['semantic_hashes'], ('membership_hash', 'roles_hash', 'issuer_mapping_hash'))
    _fields(manifest['counts'], ('population_count', 'issuer_count', 'unspecified_sector_count',
                                'evidence_pair_count', 'ticker_counts', 'issuer_counts'))
    for counts in ('ticker_counts', 'issuer_counts'):
        _fields(manifest['counts'][counts], _ISSUER_COUNTS)
    _fields(manifest['eligibility'], ('note', 'research_eligible'))
    _fields(manifest['provenance'], ('acquisition_time_is_approximate', 'acquisition_time_utc',
                                   'report_date', 'review_id', 'source_byte_size',
                                   'source_pdf_sha256', 'source_url'))
    if not isinstance(population['members'], list) or not isinstance(rows, list):
        raise ValueError('members and evidence rows must be lists')
    members = tuple(_record(PopulationMember, row) for row in population['members'])
    if members != validate_members(members):
        raise ValueError('members must already be sorted')
    mapping = {m.ticker: m.issuer_id for m in members}
    if (len(members) != 503 or len(set(mapping.values())) != 500
            or sum(m.sector == 'Unspecified' for m in members) != 29):
        raise ValueError('expected full 503/500 population with 29 Unspecified sectors')
    expected_mapping = {ticker: f'issuer:{ticker}' for ticker in mapping}
    for tickers, issuer in (
        (('FOX', 'FOXA'), 'issuer:fox-corporation'),
        (('GOOG', 'GOOGL'), 'issuer:alphabet-inc'),
        (('NWS', 'NWSA'), 'issuer:news-corp'),
    ):
        for ticker in tickers:
            expected_mapping[ticker] = issuer
    if mapping != expected_mapping:
        raise ValueError('expected exact registered issuer mapping')
    membership_hash = sha256_json([asdict(m) for m in members])
    mapping_hash = sha256_json(mapping)
    if (population['source_sha256'] != _POPULATION_SOURCE
            or manifest['population_source_sha256'] != _POPULATION_SOURCE
            or population['membership_sha256'] != membership_hash):
        raise ValueError('population identity mismatch')
    _fields(roles, ('roles', 'assignments', 'ticker_counts', 'issuer_counts', 'assignment_hash'))
    expected_roles = assign_roles(members, 20260930, _ISSUER_COUNTS)
    validated_roles = validate_roles(members, roles['roles'])
    if roles != asdict(expected_roles) or roles != asdict(validated_roles):
        raise ValueError('whole role record differs from seeded assignments')
    if expected_roles.ticker_counts != _TICKER_COUNTS:
        raise ValueError('unexpected ticker role counts')
    pairs = []
    for row in rows:
        _fields(row, (f.name for f in fields(EvidencePair)))
        first = _record(EvidenceItem, row['evidence1'])
        second = _record(EvidenceItem, row['evidence2'])
        pairs.append(EvidencePair(row['ticker'], row['trial_id'], row['condition'], first, second))
    pairs = tuple(pairs)
    validated = validate_evidence_pairs(members, (_TRIAL,), pairs,
                                       order_contrasts=(EvidenceOrderContrast(_TRIAL, '+-', '-+'),))
    if pairs != validated or len(pairs) != 2012:
        raise ValueError('evidence rows must be complete and already sorted')
    for pair in pairs:
        for item in (pair.evidence1, pair.evidence2):
            if (item.source_sha256 != _SOURCE or item.source != _URL or item.review_id != _REVIEW):
                raise ValueError('evidence source/review identity mismatch')
    if (manifest['schema_version'] != 1 or isinstance(manifest['schema_version'], bool)
            or manifest['evidence_source_sha256'] != _SOURCE
            or manifest['provenance']['source_pdf_sha256'] != _SOURCE
            or manifest['provenance']['source_url'] != _URL
            or manifest['provenance']['review_id'] != _REVIEW
            or manifest['eligibility']['research_eligible'] is not False):
        raise ValueError('manifest provenance/eligibility mismatch')
    if manifest['semantic_hashes'] != dict(membership_hash=membership_hash,
                                          roles_hash=expected_roles.assignment_hash,
                                          issuer_mapping_hash=mapping_hash):
        raise ValueError('semantic hashes mismatch')
    if manifest['counts'] != dict(population_count=503, issuer_count=500, unspecified_sector_count=29,
                                  evidence_pair_count=2012, ticker_counts=_TICKER_COUNTS,
                                  issuer_counts=_ISSUER_COUNTS):
        raise ValueError('manifest counts mismatch')
    return members, pairs


@dataclass(frozen=True, slots=True)
class BaselineInputs:
    """Trusted input container; loader outputs are immutable, not eligibility proof.

    Direct construction does not attest that the approved loader validated bytes.
    """

    members: tuple[PopulationMember, ...]
    pairs: tuple[EvidencePair, ...]
    membership_sha256: str
    issuer_mapping_sha256: str
    roles_sha256: str
    evidence_sha256: str
    manifest_sha256: str
    _roles_bytes: bytes
    _manifest_bytes: bytes

    @property
    def roles(self) -> dict:
        return json.loads(self._roles_bytes)

    @property
    def manifest(self) -> dict:
        return json.loads(self._manifest_bytes)

    @property
    def issuer_by_ticker(self) -> dict[str, str]:
        return {m.ticker: m.issuer_id for m in self.members}

    def pair_for(self, ticker: str, condition: str, trial_id: str) -> EvidencePair:
        for pair in self.pairs:
            if (pair.ticker, pair.condition, pair.trial_id) == (ticker, condition, trial_id):
                return pair
        raise ValueError('unknown baseline evidence key')


def load_baseline_inputs(directory: str | Path) -> BaselineInputs:
    """Load only the exact approved four-file materialization; no pin overrides."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('input directory must be a nonsymlink directory')
    if {p.name for p in directory.iterdir()} != set(_FILES):
        raise ValueError('input directory must contain exactly four approved files')
    payloads = {}
    for name in _FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('input files must be regular nonsymlink files')
        payloads[name] = path.read_bytes()
    if sha256_bytes(payloads[_FILES[3]]) != _MANIFEST_SHA256:
        raise ValueError('manifest raw SHA-256 mismatch')
    manifest = _parse_json(payloads[_FILES[3]])
    for name in _FILES[:3]:
        if sha256_bytes(payloads[name]) != manifest['output_hashes'][name]:
            raise ValueError(f'{name} raw SHA-256 mismatch')
    population = _parse_json(payloads[_FILES[0]])
    roles = _parse_json(payloads[_FILES[1]])
    raw_rows = payloads[_FILES[2]].splitlines(keepends=True)
    rows = [_parse_json(row) for row in raw_rows]
    members, pairs = _validate_documents(population, roles, rows, manifest)
    hashes = manifest['semantic_hashes']
    return BaselineInputs(members, pairs, hashes['membership_hash'], hashes['issuer_mapping_hash'],
                          hashes['roles_hash'], sha256_bytes(payloads[_FILES[2]]), _MANIFEST_SHA256,
                          payloads[_FILES[1]], payloads[_FILES[3]])
