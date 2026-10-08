"""Fit capture execution on real CPU fake hooks, not a checkpoint."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from test_stance_localization_execution import setup, Block, clean
from test_stance_dim_extraction import case
from scripts import fit_stance_dim as runner
from llm_bias.core.artifact_paths import canonical_json_bytes
from llm_bias.core.inference.structured_output import generate_structured
from llm_bias.core.inference.harmony_generation import generate_harmony_structured


def test_declared_panel_and_closed_cli():
    assert runner.depth_panel(40) == (0, 9, 19, 29, 39)
    assert runner.depth_panel(2) == (0, 1)
    for value in (0, True, 2.0):
        with pytest.raises(ValueError): runner.depth_panel(value)
    required = ['--inputs', 'i', '--parent', 'p', '--model', 'm', '--output-dir', 'o']
    runner.parser().parse_args(required)
    for option in ('--layer', '--target-tickers', '--max-new-tokens', '--condition'):
        with pytest.raises(SystemExit): runner.parser().parse_args(required + [option, '1'])


@pytest.mark.parametrize('cached', [True, False])
def test_actual_simultaneous_five_depth_capture_first_prompt(setup, cached):
    setup['policy'] = replace(setup['policy'], use_cache=cached)
    model = setup['model']; root = model.hf_model
    root.layers = torch.nn.ModuleList([Block() for _ in range(40)])
    model.layers = root.layers
    prompt = setup['donor_prompt']
    ids = torch.tensor([prompt.inference_token_ids])
    driver = generate_structured if setup['route'] == 'plain' else generate_harmony_structured
    expected = driver(model, setup['tokenizer'], ids,
                      setup['capability'], policy=setup['policy'])
    root.calls = 0
    seen = []
    handle = root.register_forward_pre_hook(lambda *a: seen.append([
        len(root.layers[l]._forward_hooks) for l in runner.depth_panel(40)]))
    result, means, status = runner.capture_row(model, setup['tokenizer'], prompt,
        setup['capability'], setup['policy'], expected, runner.depth_panel(40))
    handle.remove()
    assert status == 'matched' and root.calls == 1
    assert seen and all(all(n == 1 for n in counts) for counts in seen)
    assert len(means) == 20
    for (layer, span), mean in means.items():
        record = getattr(prompt, span + '_span')
        wanted = sum(prompt.inference_token_ids[record.token_start:record.token_end]) / len(record.token_ids)
        assert mean == pytest.approx([wanted + 3 * (layer + 1)] * 4)
    clean(root)


@pytest.mark.parametrize('kind', ['text', 'config', 'failure', 'bypass'])
def test_capture_halts_and_cleans_without_pooling(setup, kind):
    root = setup['model'].hf_model
    expected = setup['expected_donor']
    if kind == 'text': expected = replace(expected, generated_text=expected.generated_text + 'drift')
    elif kind == 'config':
        provenance = expected.provenance; provenance['grammar_sha256'] = 'f' * 64
        expected = replace(expected, _provenance_bytes=canonical_json_bytes(provenance))
    elif kind == 'failure': root.fail = (1, 0)
    else: root.bypass = True
    result, means, status = runner.capture_row(setup['model'], setup['tokenizer'],
        setup['donor_prompt'], setup['capability'], setup['policy'], expected, (0, 1))
    assert status != 'matched' and means == {} and root.calls == 1
    clean(root)


def fake_inputs(case):
    return SimpleNamespace(members=case['members'], roles={'roles': case['roles']},
                           manifest_sha256='a' * 64)


def test_fit_all_conditions_roles_no_raw_and_fresh_artifact(tmp_path, case):
    inputs = fake_inputs(case)
    parent = SimpleNamespace(plan=case['plan'], rows=case['rows'], parent_sha256='b' * 64)
    calls = []
    def capture(key):
        calls.append(key)
        return None, {(l, s): case['pooled_residuals'][key] for l in (0, 9, 19, 29, 39)
                      for s in runner.SPANS}, 'matched'
    output = tmp_path / 'fit'
    desc = runner.descriptor(inputs, parent, 40, 2, {})
    runner.fresh_output(output, desc)
    report = runner.fit(output, inputs, parent, desc, capture)
    assert len(calls) == 160 and set(k.condition for k in calls) == {'++', '+-', '-+', '--'}
    assert report['complete_capture'] and not report['research_eligible']
    import json
    operators = json.loads((output / 'operators.json').read_text())
    assert len(operators['extractions']) == 20
    assert all(c['direction'] == [3., 2.] for e in operators['extractions'] for c in e['candidates'])
    assert all(k.ticker in case['roles']['fit'] for k in calls)
    assert 'pooled_residuals' not in (output / 'operators.json').read_text()
    assert all('direction' not in p.read_text() for p in (output / 'records').iterdir())
    with pytest.raises(FileExistsError): runner.fresh_output(output, desc)


def test_drift_persists_blocks_fit_no_retry(tmp_path, case):
    inputs = fake_inputs(case)
    parent = SimpleNamespace(plan=case['plan'], rows=case['rows'], parent_sha256='b' * 64)
    desc = runner.descriptor(inputs, parent, 40, 2, {})
    output = tmp_path / 'fit'; runner.fresh_output(output, desc)
    calls = []
    def capture(key):
        calls.append(key)
        return None, {}, 'parent_mismatch'
    report = runner.fit(output, inputs, parent, desc, capture)
    assert len(calls) == 1 and not report['complete_capture']
    assert report['status'] == 'parent_mismatch' and report['matched_rows'] == 0
    assert (output / 'summary.json').exists() and (output / 'operators.json').exists()


def test_approved_full503_interface_and_1208_fit_rows(tmp_path):
    from pathlib import Path
    from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
    from llm_bias.core.stance_baseline_plan import build_baseline_plan
    from llm_bias.core.experiment_contract import PlanIdentity, ExecutionRow, GenerationOutcome
    inputs = load_baseline_inputs(Path(__file__).resolve().parents[1] /
                                  'data/concept-cone-steering/rebuild-v1/compiled')
    identity = replace(PlanIdentity(*(['a' * 64] * 11),
                                   'not_applicable', 'not_applicable'),
        population_sha256=inputs.membership_sha256, issuer_sha256=inputs.issuer_mapping_sha256,
        roles_sha256=inputs.roles_sha256, evidence_sha256=inputs.evidence_sha256)
    plan = build_baseline_plan(inputs, identity)
    issuers = inputs.issuer_by_ticker
    fit_issuers = sorted({issuers[t] for t in inputs.roles['roles']['fit']})
    buys = set(fit_issuers[:150])
    rows = tuple(ExecutionRow(k, GenerationOutcome(k.ticker, issuers[k.ticker], k.condition,
        k.trial_id, 'buy' if issuers[k.ticker] in buys else 'sell', True, True, True,
        'schema_complete', None)) for k in plan.keys)
    parent = SimpleNamespace(plan=plan, rows=rows, parent_sha256='b' * 64)
    desc = runner.descriptor(inputs, parent, 40, 2, {})
    assert desc['planned_fit_rows'] == 1208
    calls = []
    def capture(key):
        calls.append(key)
        v = (4., 3.) if issuers[key.ticker] in buys else (1., 1.)
        return None, {(l, s): v for l in runner.depth_panel(40) for s in runner.SPANS}, 'matched'
    directory = tmp_path / 'full'; runner.fresh_output(directory, desc)
    report = runner.fit(directory, inputs, parent, desc, capture)
    assert report['complete_capture'] and len(calls) == 1208
    assert report['site_families'] == 20 and report['condition_candidates'] == 80
    assert all(k.ticker in inputs.roles['roles']['fit'] for k in calls)


def test_exception_cleanup_and_no_tensor_serialization(setup, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('raw tensor([12345]) must not enter artifact')
    monkeypatch.setattr(runner, 'pool_original_span', fail)
    _, means, status = runner.capture_row(setup['model'], setup['tokenizer'],
        setup['donor_prompt'], setup['capability'], setup['policy'],
        setup['expected_donor'], (0, 1))
    assert means == {} and status == 'capture_unsupported'
    clean(setup['model'].hf_model)


def test_partial_drift_discards_previously_pooled_vectors(tmp_path, case):
    inputs = fake_inputs(case)
    parent = SimpleNamespace(plan=case['plan'], rows=case['rows'], parent_sha256='b' * 64)
    desc = runner.descriptor(inputs, parent, 40, 2, {})
    directory = tmp_path / 'partial'; runner.fresh_output(directory, desc)
    calls = []
    def capture(key):
        calls.append(key)
        if len(calls) == 2: return None, {}, 'configuration_mismatch'
        return None, {(l, s): case['pooled_residuals'][key] for l in runner.depth_panel(40)
                      for s in runner.SPANS}, 'matched'
    report = runner.fit(directory, inputs, parent, desc, capture)
    assert len(calls) == 2 and report['matched_rows'] == 1 and report['retained_capture_rows'] == 0
    import json
    assert all(c['direction'] is None and c['coverage']['contributing_rows'] == 0
        for e in json.loads((directory / 'operators.json').read_text())['extractions']
        for c in e['candidates'])


def test_public_merged_metadata_reaches_model_loading(tmp_path, monkeypatch):
    """Exercise public --recovery wiring, not an original-shaped fake metadata dict."""
    backend = {name: 'same' for name in ('torch', 'transformers', 'xgrammar', 'jlens',
        'cuda', 'kernel_policy', 'cudnn', 'deterministic_algorithms')}
    backend['requested_dtype'] = 'native'
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    original = {'bindings': {'model': {'resolved_path': '/original/checkpoint',
        'metadata_file_sha256': {'tokenizer.json': 'a' * 64}}, 'backend': backend}}
    parent = SimpleNamespace(metadata={'kind': 'effective_merged_baseline_v1',
        'original_metadata': original, 'recovery_registration': {}})
    monkeypatch.setattr(runner, 'load_baseline_inputs', lambda path: object())
    monkeypatch.setattr(runner, 'load_merged_baseline', lambda *a, **k: parent)
    monkeypatch.setattr(runner, 'runtime_metadata', lambda path: {
        'model': {'resolved_path': str(checkpoint),
                  'metadata_file_sha256': original['bindings']['model']['metadata_file_sha256']},
        'backend': backend.copy(), 'code': {'source_sha256': {}}})
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: True)
    calls = []
    def load(*a, **k):
        calls.append(k['dtype']); raise RuntimeError('model-loading boundary reached')
    monkeypatch.setattr(runner, 'load_model', load)
    args = runner.parser().parse_args(['--inputs', 'i', '--parent', 'p', '--recovery', 'r',
        '--model', str(checkpoint), '--output-dir', str(tmp_path / 'merged-fit')])
    with pytest.raises(RuntimeError, match='model-loading boundary reached'):
        runner.run(args)
    assert calls == ['native']


def test_public_preflight_failure_is_immutable_and_redacted(tmp_path, monkeypatch):
    directory = tmp_path / 'public'
    def fail(path):
        raise ValueError('tensor([12345])')
    monkeypatch.setattr(runner, 'load_baseline_inputs', fail)
    required = ['--inputs', 'i', '--parent', 'p', '--model', 'm', '--output-dir', str(directory)]
    assert runner.main(required) == 1
    import json
    report = json.loads((directory / 'summary.json').read_text())
    assert report['error_type'] == 'ValueError' and '12345' not in str(report)
    assert runner.main(required) == 1
    assert json.loads((directory / 'summary.json').read_text()) == report
