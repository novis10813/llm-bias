"""CPU-only prospective validation panel. No outcomes or checkpoint loading."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from .experiment_contract import ExperimentPlan
from .stance_baseline_inputs import BaselineInputs
from .stance_baseline_plan import build_baseline_plan

DIMENSIONS = (2, 4)
SEEDS = (20261003, 20261004, 20261005)
RAY_SEED = 20261006
DOSES = (-64, -32, -8, -2, 0, 2, 8, 32, 64)
AUDIT_SHA256 = '42c5300a224e1b8a4ab2f5508173f3749249d4932679b10c3af5d3dfd362220b'
GRID = tuple((k, s, f'k{k}-seed{s}') for k in DIMENSIONS for s in SEEDS)


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('expected regular artifact file: ' + str(path))
    raw = path.read_bytes()
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return raw, json.loads(raw, parse_constant=reject, object_pairs_hook=unique)


def unit_direction(basis, coefficients):
    """Positive combinations only. Basis axes use nonnegative one-hot coefficients."""
    b = torch.tensor(basis, dtype=torch.float64, device='cpu')
    c = torch.tensor(coefficients, dtype=torch.float64, device='cpu')
    if (b.ndim != 2 or c.shape != (b.shape[0],) or not torch.isfinite(b).all()
            or (b.norm(dim=1) <= 0).any()
            or not torch.isfinite(c).all() or (c < 0).any() or c.sum() <= 0):
        raise ValueError('invalid finite nonnegative cone coefficients/basis')
    c = c / c.sum()
    direction = c @ b
    norm = direction.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError('zero or nonfinite cone direction')
    return c.tolist(), (direction / norm).tolist()


@dataclass(frozen=True, slots=True)
class ConeValidationPlan:
    """Immutable canonical byte storage, including all logical planned cells."""
    _bytes: bytes

    @property
    def plan_id(self):
        return sha256_bytes(self._bytes)

    @property
    def panel_id(self):
        return self.to_dict()['panel_id']

    def to_dict(self):
        return json.loads(self._bytes)

    def to_json(self):
        return self._bytes.decode('utf8')


def compile_cone_validation_plan(inputs: BaselineInputs, parent_plan: ExperimentPlan,
                                 training_directory: str | Path, audit_path: str | Path) -> ConeValidationPlan:
    """Consume the pinned accepted saved-training audit, not live weight authentication.

    parent_plan is reconstructed from approved inputs. No parent generated outcomes
    are read here. The next runner must authenticate/replay that parent once.
    """
    rebuilt = build_baseline_plan(inputs, parent_plan.identity)
    if rebuilt.to_json() != parent_plan.to_json():
        raise ValueError('full approved parent plan differs')
    directory = Path(training_directory)
    audit_raw, audit = _read(Path(audit_path))
    if sha256_bytes(audit_raw) != AUDIT_SHA256:
        raise ValueError('accepted training audit SHA mismatch')
    required = dict(kind='stance_cone_training_grid_acceptance_v1', status='passed',
        training_completed=True, grid_count=6, completed_steps=1812,
        authenticated_teachers=906, fit_tickers_per_run=302, accepted_operator=False,
        research_eligible=False, generated_validation_completed=False,
        teacher_content_freshly_authenticated=True, training_artifact_files_unchanged=True,
        training_backend_and_model_bindings_verified=True,
        training_source_inventory_and_hashes_verified=True)
    if any(audit.get(k) != v for k, v in required.items()):
        raise ValueError('training audit contract differs')
    run_names = {name for _, _, name in GRID}
    if {p.name for p in directory.iterdir()} != run_names:
        raise ValueError('exact six training runs required')
    if len(audit['runs']) != 6 or {r['run'] for r in audit['runs']} != run_names:
        raise ValueError('audit grid differs')
    # Recheck every saved artifact digest, including all 1812 compact step records.
    inventory = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if inventory != set(audit['file_sha256']):
        raise ValueError('training artifact manifest inventory differs')
    for name, digest in audit['file_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe artifact manifest path')
        raw, _ = _read(directory / name)
        if sha256_bytes(raw) != digest:
            raise ValueError('training artifact SHA mismatch: ' + name)
    generator = torch.Generator(device='cpu').manual_seed(RAY_SEED)
    directions, provenance = [], []
    fit = {t for t, role in inputs.roles['assignments'].items() if role == 'fit'}
    reference_binding = None
    for k, seed, name in GRID:
        raw_config, config = _read(directory / name / 'config.json')
        raw_operator, operator = _read(directory / name / 'operator.json')
        _, summary = _read(directory / name / 'summary.json')
        config_hash = sha256_json(config)
        audit_run = next(r for r in audit['runs'] if r['run'] == name)
        for record in (config, operator, summary, audit_run):
            if record.get('dimension') != k or record.get('seed') != seed:
                raise ValueError('training seed/dimension differs')
        for record in (operator, summary, audit_run):
            if record.get('config_sha256') != config_hash:
                raise ValueError('config/operator hash binding differs')
        for record in (config, operator, summary):
            if record.get('accepted_operator') is not False or record.get('research_eligible') is not False:
                raise ValueError('training cannot accept an operator')
        for record in (operator, summary):
            if record.get('training_completed') is not True:
                raise ValueError('incomplete training')
        if (summary.get('status') != 'training_completed' or summary.get('completed_steps') != 302
                or summary.get('attempted_steps') != 302 or summary.get('diagnostic_one_step') is not False):
            raise ValueError('full training completion required')
        if (config.get('declared_grid') != dict(dimensions=list(DIMENSIONS), seeds=list(SEEDS))
                or config.get('diagnostic_one_step') is not False or config.get('planned_steps') != 302
                or len(config['ticker_order']) != 302 or set(config['ticker_order']) != fit):
            raise ValueError('construction training grid/roles differ')
        for record in (config, operator):
            if (record.get('layer') != 19 or record.get('scope') != 'original_instruction_post_block'
                    or record.get('dose') != 32):
                raise ValueError('training intervention differs')
        if (config['inputs_manifest_sha256'] != inputs.manifest_sha256
                or config['roles_sha256'] != inputs.roles_sha256
                or config['parent_sha256'] != audit['parent_sha256']
                or config['plan_hash'] != parent_plan.plan_hash
                or config['full_plan_identity'] != parent_plan.identity.to_dict()):
            raise ValueError('current input/parent/token policy identity differs')
        binding = config['teacher_binding']
        manifest = binding['manifest']
        if (manifest['accepted'] is not True or manifest['planned_count'] != 906
                or manifest['parent_sha256'] != audit['parent_sha256']
                or manifest['inputs_manifest_sha256'] != inputs.manifest_sha256
                or manifest['roles_sha256'] != inputs.roles_sha256
                or manifest['plan_hash'] != parent_plan.plan_hash):
            raise ValueError('accepted fit teacher binding differs')
        if reference_binding is None:
            reference_binding = binding
        if binding != reference_binding:
            raise ValueError('teacher/tokenizer binding differs across grid')
        if any(config['runtime']['backend'].get(field) != 'torch.bfloat16'
               for field in ('actual_embedding_dtype', 'actual_head_dtype')):
            raise ValueError('native BF16 basis required')
        basis = torch.tensor(operator['basis'], dtype=torch.float64, device='cpu')
        if (basis.shape != (k, 4096) or not torch.isfinite(basis).all()
                or ((basis.norm(dim=1) - 1).abs() > .01).any()
                or not torch.equal(basis, basis.to(torch.bfloat16).to(torch.float64))):
            raise ValueError('expected finite K x 4096 near-unit BF16 exported basis')
        unit_basis = basis / basis.norm(dim=1, keepdim=True)
        coeffs = [(f'basis-{i}', torch.eye(k, dtype=torch.float64)[i].tolist()) for i in range(k)]
        coeffs.append(('positive-centroid', [1 / k] * k))
        for i in range(4):
            c = torch.rand(k, generator=generator, dtype=torch.float32) + .01
            coeffs.append((f'positive-ray-{i}', (c / c.sum()).tolist()))
        for family, c in coeffs:
            normalized, vector = unit_direction(unit_basis.tolist(), c)
            directions.append(dict(cone=name, family=family, coefficients=normalized, direction=vector))
        provenance.append(dict(cone=name, config_sha256=config_hash,
            config_file_sha256=sha256_bytes(raw_config), operator_file_sha256=sha256_bytes(raw_operator),
            teacher_manifest_sha256=binding['manifest_sha256'],
            source_sha256=manifest['source_sha256'], review_sha256=manifest['review_sha256']))
    rows = [key.to_dict() for key in parent_plan.keys
            if inputs.roles['assignments'][key.ticker] == 'validation']
    if (len(rows) != 300 or len({r['ticker'] for r in rows}) != 75
            or len({inputs.issuer_by_ticker[r['ticker']] for r in rows}) != 75):
        raise ValueError('full 75 x 4 validation coverage required')
    panel = dict(kind='stance_cone_validation_panel_v1', directions=directions,
        ray_seed=RAY_SEED, generation_order='dimension_then_seed_then_ray',
        ray_policy='CPU_torch_float32_uniform_plus_0.01_L1_then_L2_unit_basis',
        torch_version=torch.__version__, provenance=provenance,
        compiler_source_sha256=sha256_bytes(Path(__file__).read_bytes()),
        audit_sha256=sha256_bytes(audit_raw),
        compiler_provenance_discrepancies=audit['compiler_provenance_discrepancies'],
        authentication_scope=audit['validation_scope'])
    panel_id = sha256_json(panel)
    cells = [dict(cone=d['cone'], family=d['family'], dose=dose, row=row,
                  arm='negative_cone_financial_adaptation' if dose < 0 else
                      ('positive_cone' if dose > 0 else 'zero_baseline'))
             for d in directions for dose in DOSES for row in rows]
    coverage = dict(cones=6, directions=48, validation_tickers=75, validation_rows=300,
        validation_issuers=75, conditions=['++', '+-', '-+', '--'],
        role_counts=inputs.roles['ticker_counts'], doses=9, planned_cells=129600, zero_cells=14400, nonzero_cells=115200,
        per_cone={name: (k + 5) * 2700 for k, _, name in GRID})
    if len(cells) != coverage['planned_cells']:
        raise ValueError('planned cell coverage differs')
    plan = dict(kind='stance_cone_validation_plan_v1', panel_id=panel_id, panel=panel,
        parent_sha256=audit['parent_sha256'], parent_plan_hash=parent_plan.plan_hash,
        parent_identity=parent_plan.identity.to_dict(), inputs_manifest_sha256=inputs.manifest_sha256,
        layer=19, site='original_instruction_post', scope='prompt_only', operation='addition',
        dose_semantics='native_signed_unit_L2', doses=list(DOSES), role='validation',
        status='prospective_validation_exploration', accepted_operator=False, research_eligible=False,
        evaluation_seen=False, acceptance_threshold=None, coverage=coverage, cells=cells,
        failure_policy='retain_every_planned_failed_generation_in_ITT;no_resampling_or_row_drop',
        zero_policy='all_logical_cells_retained;one_legitimate_generation_per_input_may_be_referenced_only_with_identical_model_prompt_schema_tokenizer_decoding_hook_config;never_fabricate_outputs',
        runner_requirements='authenticate_parent_and_runtime;run_gates_and_parent_replay_once;execute_full_exact_plan;no_raw_activations')
    return ConeValidationPlan(canonical_json_bytes(plan))


def write_cone_validation_plan(directory: str | Path, plan: ConeValidationPlan) -> None:
    """Fresh immutable export only. Never write into training history."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    value = plan.to_dict()
    payloads = {'plan.json': plan._bytes + b'\n',
                'panel.json': canonical_json_bytes(value['panel']) + b'\n',
                'coverage.json': canonical_json_bytes(value['coverage']) + b'\n'}
    payloads['manifest.json'] = canonical_json_bytes(dict(plan_id=plan.plan_id,
        panel_id=plan.panel_id, file_sha256={n: sha256_bytes(b) for n, b in payloads.items()},
        research_eligible=False, accepted_operator=False)) + b'\n'
    for name, raw in payloads.items():
        with (directory / name).open('xb') as stream:
            stream.write(raw)
