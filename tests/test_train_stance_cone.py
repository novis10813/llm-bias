"""CPU full-pack authentication and actual C3 optimization, no checkpoint."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from test_stance_cone_teachers import case
from test_stance_baseline_parent import parent_bundle
from test_stance_baseline_adapter import bundle
from test_stance_cone_training_step import setup
from scripts.train_stance_cone import (
    load_teachers, make_batches, optimize, parser, ticker_order, run,
)
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes
from llm_bias.core.stance_cone_teachers import compile_cone_teachers, write_teacher_pack
from llm_bias.core.inference.stance_cone_training_step import stance_cone_training_step


def config(diagnostic=True):
    return dict(seed=20261003, dimension=2, layer=0, diagnostic_one_step=diagnostic,
                ray_policy='positive_one_per_step', optimizer=dict(name='Adam', lr=.01))


def loop_case(tmp_path, dtype=torch.float64):
    model, batches, _, _ = setup()
    model.hf_model.to(dtype=dtype)
    records, prompts, roles = [], {}, {}
    for i in range(302):
        ticker = f'T{i:03d}'
        roles[ticker] = 'fit'
        for purpose, batch in batches.items():
            # Unlike C3's padding fixture, production has a complete response.
            record = replace(batch.records[0], ticker=ticker,
                             response_mask=(False, False, True, True, True))
            records.append(record)
            prompts[ticker, purpose] = SimpleNamespace(inference_token_ids=(0, 1),
                instruction_span=SimpleNamespace(token_start=1, token_end=2))
    out = tmp_path / 'run'
    out.mkdir(); (out / 'records').mkdir()
    return out, model, tuple(records), prompts, roles


def teacher_case(tmp_path, case):
    inputs, parent, tokenizer = case
    bindings = parent.metadata['bindings']
    compiler_config = dict(kind='stance_cone_teacher_compiler_config_v1',
        inputs_manifest_sha256=inputs.manifest_sha256, parent_sha256=parent.parent_sha256,
        model=bindings['model'], logical_tokenizer_name=bindings['model']['resolved_path'],
        execution='CPU_tokenizer_and_grammar_only_no_checkpoint_weights')
    pack = compile_cone_teachers(*case)
    path = tmp_path / 'teachers'
    write_teacher_pack(path, pack, compiler_config)
    return path, pack


def test_full906_raw_authentication_and_tuple_fields(tmp_path, case):
    path, _ = teacher_case(tmp_path, case)
    records, binding = load_teachers(path, *case)
    assert len(records) == 906
    assert len({r.ticker for r in records}) == 302
    assert all(type(r.token_ids) is tuple and type(r.response_mask) is tuple for r in records)
    assert binding['manifest']['accepted'] is True


@pytest.mark.parametrize('mutation', ['raw_hash', 'teacher', 'source', 'review', 'roles', 'config', 'missing'])
def test_forged_or_incomplete_pack_rejected_even_rehashed(tmp_path, case, mutation):
    path, _ = teacher_case(tmp_path, case)
    manifest = json.loads((path / 'manifest.json').read_bytes())
    if mutation == 'raw_hash':
        with (path / 'teachers.jsonl').open('ab') as f: f.write(b'\n')
    else:
        filename = {'teacher': 'teachers.jsonl', 'source': 'source_manifest.json',
                    'review': 'review_manifest.json', 'roles': 'teachers.jsonl',
                    'config': 'config.json', 'missing': 'teachers.jsonl'}[mutation]
        if filename == 'teachers.jsonl':
            rows = [json.loads(line) for line in (path / filename).read_bytes().splitlines()]
            if mutation == 'missing': rows.pop()
            elif mutation == 'roles': rows[0]['role'] = 'validation'
            else: rows[0]['response_source'] = 'construction_authored'
            raw = b''.join(canonical_json_bytes(r) + b'\n' for r in rows)
        else:
            value = json.loads((path / filename).read_bytes())
            if mutation == 'source': value['selected_records'][0]['prompt_token_ids'][0] += 1
            elif mutation == 'review': value['cells'][0]['available'] = False
            else: value['logical_tokenizer_name'] = 'other'
            raw = canonical_json_bytes(value) + b'\n'
        (path / filename).write_bytes(raw)
        manifest['file_sha256'][filename] = sha256_bytes(raw)
        (path / 'manifest.json').write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError): load_teachers(path, *case)


def test_cli_has_only_declared_controls():
    common = ['--inputs', 'i', '--parent', 'p', '--teachers', 't', '--model', 'm', '--output-dir', 'o']
    assert parser().parse_args(common + ['--dimension', '4', '--seed', '20261005']).dimension == 4
    for extra in (['--dimension', '3', '--seed', '20261003'],
                  ['--dimension', '2', '--seed', '1'],
                  ['--dimension', '2', '--seed', '20261003', '--steps', '2']):
        with pytest.raises(SystemExit): parser().parse_args(common + extra)
    assert ticker_order(['B', 'A'], 20261003) == ticker_order(['A', 'B'], 20261003)
    assert len({ticker_order([str(i) for i in range(30)], s)
                for s in (20261003, 20261004, 20261005)}) == 3


@pytest.mark.parametrize('dtype', [torch.float64, torch.bfloat16])
@pytest.mark.parametrize('dimension', [2, 4])
def test_one_step_real_autograd_frozen_lm_and_no_operator(tmp_path, dtype, dimension):
    out, model, records, prompts, roles = loop_case(tmp_path, dtype)
    if dimension == 4:
        root = model.hf_model
        root.embedding = torch.nn.Embedding.from_pretrained(
            torch.cat((root.embedding.weight.detach(), torch.ones(5, 1, dtype=dtype)), dim=1), freeze=True)
        old = root.head.weight.detach()
        root.head = torch.nn.Linear(4, 5, bias=False, dtype=dtype)
        with torch.no_grad():
            root.head.weight.copy_(torch.cat((old, torch.ones(5, 1, dtype=dtype)), dim=1))
        root.eval().requires_grad_(False)
    before = [p.detach().clone() for p in model.hf_model.parameters()]
    observed = []
    def actual(*args, **kwargs):
        observed.append(kwargs['basis'])
        return stance_cone_training_step(*args, **kwargs)
    report = optimize(out, model, records, prompts, roles, config() | {'dimension': dimension}, step=actual)
    assert report['status'] == 'diagnostic_capability_success'
    assert report['completed_steps'] == report['attempted_steps'] == 1
    assert not report['training_completed'] and not report['research_eligible']
    assert not (out / 'operator.json').exists()
    basis = observed[0]
    from llm_bias.core.stance_cone_objectives import initialize_basis
    initial = initialize_basis(model.hf_model.embedding.weight.shape[1], dimension=dimension, seed=20261003).to(dtype)
    assert not torch.equal(basis.detach(), initial)
    assert basis.dtype == dtype and torch.isfinite(basis).all()
    torch.testing.assert_close(basis.float().norm(dim=-1), torch.ones(dimension), atol=.006, rtol=.006)
    for old, p in zip(before, model.hf_model.parameters()):
        assert p.grad is None and not p.requires_grad
        torch.testing.assert_close(old, p)
    assert not model.layers[0]._forward_hooks
    row = json.loads(next((out / 'records').iterdir()).read_bytes())
    assert row['gradient_norm'] > 0 and set(row['losses']) == {'addition', 'ablation', 'retain', 'total'}


def test_full_epoch_exact_302_steps_and_determinism(tmp_path):
    operators = []
    for i in range(2):
        root = tmp_path / str(i); root.mkdir()
        out, model, records, prompts, roles = loop_case(root)
        report = optimize(out, model, records, prompts, roles, config(False))
        assert report['training_completed'] and report['completed_steps'] == 302
        assert report['accepted_operator'] is False and report['research_eligible'] is False
        rows = sorted((out / 'records').iterdir())
        assert len(rows) == 302
        assert [json.loads(p.read_bytes())['ticker'] for p in rows] == list(ticker_order(roles, 20261003))
        operators.append(json.loads((out / 'operator.json').read_bytes())['basis'])
    assert operators[0] == operators[1]


def test_failed_attempt_retains_partial_steps_resources_and_cleanup(tmp_path):
    out, model, records, prompts, roles = loop_case(tmp_path)
    calls = 0
    def actual(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            def fail(_m, _a, _o):
                raise torch.cuda.OutOfMemoryError('tensor SECRET must not be serialized')
            handle = model.hf_model.head.register_forward_hook(fail)
            try: return stance_cone_training_step(*args, **kwargs)
            finally: handle.remove()
        return stance_cone_training_step(*args, **kwargs)
    report = optimize(out, model, records, prompts, roles, config(False), step=actual)
    assert report['status'] == 'failed' and report['oom']
    assert report['completed_steps'] == 1 and report['attempted_steps'] == 2
    assert not report['training_completed'] and not (out / 'operator.json').exists()
    rows = [json.loads(p.read_bytes()) for p in sorted((out / 'records').iterdir())]
    assert [r['status'] for r in rows] == ['optimizer_step_completed', 'failed']
    assert all(r['elapsed_seconds'] >= 0 for r in rows)
    assert not model.layers[0]._forward_hooks
    assert 'SECRET' not in (out / 'summary.json').read_text()


def test_batches_exact_full_masks_instruction_only(tmp_path):
    _, model, records, prompts, _ = loop_case(tmp_path)
    first = tuple(r for r in records if r.ticker == records[0].ticker)
    batches = make_batches(first, prompts, torch.device('cpu'))
    for b in batches.values():
        assert b.attention_mask.all() and b.prompt_position_mask.tolist() == [[False, True, False, False, False]]
    bad = (replace(first[0], response_mask=(False, False, True, False, True)), *first[1:])
    with pytest.raises(ValueError): make_batches(bad, prompts, torch.device('cpu'))


def test_public_preflight_failure_persisted_and_path_never_reused(tmp_path):
    args = parser().parse_args(['--inputs', str(tmp_path / 'missing'), '--parent', 'p', '--teachers', 't',
        '--model', 'm', '--output-dir', str(tmp_path / 'out'), '--dimension', '2', '--seed', '20261003',
        '--diagnostic-one-step'])
    assert run(args) == 1
    report = json.loads((args.output_dir / 'summary.json').read_bytes())
    assert report['status'] == 'preflight_failure' and not report['training_completed']
    assert report['diagnostic_one_step'] and report['elapsed_seconds'] >= 0
    assert (args.output_dir / 'attempt.json').exists()
    with pytest.raises(FileExistsError): run(args)
