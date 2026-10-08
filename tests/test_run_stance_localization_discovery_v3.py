"""Discovery-v3 CPU contract on the row-batched fake; never GPU evidence."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch

from test_stance_localization_batched import setup  # noqa: F401  (fixture)
from test_stance_localization_execution import clean
from scripts import run_stance_localization_discovery_v3 as runner
from scripts.recover_stance_baseline_truncations import read_file
from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.experiment_contract import RowKey
from llm_bias.core.stance_localization_pairs import LocalizationPair
from llm_bias.core.inference.stance_localization_batched import execute_batched_prompt_replacement
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop
from llm_bias.core.stance_localization_candidate_panel import build_localization_candidate_panel


def make_pair(target, donor, role, relation='opposite'):
    payload = dict(target_key=target.to_dict(), donor_key=donor.to_dict(), role=role,
                   family='cross_company', contrast='entity_context', clean_relation=relation)
    return LocalizationPair(target, donor, role, 'cross_company', 'entity_context', relation,
                            sha256_json(payload))


@pytest.fixture
def g(setup):
    s = setup
    donor = RowKey('baseline', 'AAA', '++', 'trial', 'baseline', '0')
    target = RowKey('baseline', 'BBB', '++', 'trial', 'baseline', '0')
    table = SimpleNamespace(pairs=(make_pair(target, donor, 'fit'), make_pair(target, donor, 'validation')),
                            parent_sha256='a' * 64, table_sha256='b' * 64, inputs_manifest_sha256='c' * 64)
    outputs = {donor: s['expected_donor'], target: s['expected_target']}
    parent = SimpleNamespace(generation_for=outputs.__getitem__)
    prompts = {donor: s['donor'], target: s['target']}
    panel = SimpleNamespace(layers=(0, 1), actual_layer_count=2, panel_sha256='d' * 64,
                            to_dict=lambda: {'model_slug': 'fake', 'layers': [0, 1]})
    bindings = dict(issuer_by_ticker={'AAA': 'ia', 'BBB': 'ib'}, runtime=dict(model={'path': 'm'}),
                    template={'t': 1}, generation_policy={'p': 1}, grammar={'g': 1},
                    model_layer_authentication={'count': 2})
    state = SimpleNamespace(batches=[])

    def gate(key, layer, span, config_hash):
        prompt = prompts[key]
        record = getattr(prompt, span + '_span')
        return execute_prompt_noop(s['model'], s['tokenizer'], torch.tensor([prompt.inference_token_ids]),
            s['capability'], policy=s['policy'], layer=layer, hook_site='post', zero_vector=torch.ones(4),
            prompt_positions=list(range(record.token_start, record.token_end)), config_hash=config_hash)

    def batch(pair, coordinates):
        state.batches.append(coordinates)
        return execute_batched_prompt_replacement(s['model'], s['tokenizer'], prompts[pair.donor_key],
            prompts[pair.target_key], s['capability'], policy=s['policy'], cells=coordinates,
            expected_donor=parent.generation_for(pair.donor_key),
            expected_target=parent.generation_for(pair.target_key), max_rows=state.desc['max_rows'])

    state.__dict__.update(setup=s, table=table, parent=parent, prompts=prompts, panel=panel,
                          bindings=bindings, gate=gate, batch=batch)
    return state


def run_phase(g, root, phase, max_rows, selection=None, gate=None, batch=None):
    g.desc = runner.descriptor(g.table, g.panel, phase, g.bindings, ('ib',), selection, max_rows)
    assert runner.execute_run(SimpleNamespace(output_dir=root), g.desc, g.table, g.prompts, g.parent,
                              gate or g.gate, batch or g.batch) == 0
    return read_file(root / 'summary.json')


def test_closed_cli():
    base = ['--model', 'm', '--inputs', 'i', '--parent', 'p', '--output-dir', 'o',
            '--model-slug', 'glm4-9b-0414', '--phase', 'discovery']
    assert runner.parser().parse_args(base).max_rows == 32
    for extra in (['--layers', '0'], ['--span', 'entity'], ['--target-tickers', 'AAA'],
                  ['--phase', 'evaluation'], ['--max-new-tokens', '1'], ['--num-shards', '2']):
        with pytest.raises(SystemExit):
            runner.parser().parse_args(base + extra)


def test_unsupported_routes_fail_before_io(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.candidates, 'load_runtime', lambda args: pytest.fail('loaded runtime'))
    args = SimpleNamespace(model_slug='gpt-oss-20b', phase='discovery', discovery_run=None,
                           max_rows=32, output_dir=tmp_path / 'new')
    with pytest.raises(NotImplementedError, match='Harmony'):
        runner.run(args)
    for phase, prior, rows in (('discovery', tmp_path, 32), ('validation', None, 32), ('discovery', None, 1)):
        args = SimpleNamespace(model_slug='glm4-9b-0414', phase=phase, discovery_run=prior,
                               max_rows=rows, output_dir=tmp_path / 'new')
        with pytest.raises(ValueError):
            runner.run(args)
    assert not (tmp_path / 'new').exists()


def test_discovery_issuers_fixed_hash_and_share_classes():
    issuers = {f'T{i:03d}': f'I{i:03d}' for i in range(300)}
    issuers.update({'T000B': 'I000', 'T001B': 'I001'})
    roles = {t: 'fit' for t in issuers} | {'V1': 'validation'}
    issuers['V1'] = 'IV'
    chosen = runner.discovery_issuers(issuers, roles)
    assert len(chosen) == 32 and chosen == runner.discovery_issuers(dict(reversed(issuers.items())), roles)
    assert 'IV' not in chosen
    ring = sorted({v for t, v in issuers.items() if roles[t] == 'fit'},
                  key=lambda i: (sha256_json({'seed': 20261007, 'role': 'fit', 'issuer_id': i}), i))
    assert set(chosen) == set(ring[:32])
    pairs = tuple(SimpleNamespace(role=roles[t], target_key=SimpleNamespace(ticker=t)) for t in issuers)
    desc = dict(phase='discovery', discovery_issuers=list(chosen), bindings={'issuer_by_ticker': issuers})
    kept = {p.target_key.ticker for p in runner.phase_pairs(SimpleNamespace(pairs=pairs), desc)}
    assert kept == {t for t, i in issuers.items() if i in chosen and roles[t] == 'fit'}
    with pytest.raises(ValueError):
        runner.discovery_issuers(issuers, roles, count=301)


@pytest.mark.parametrize('slug,count,layers', [('qwen3.5-4b', 32, 6), ('glm4-9b-0414', 40, 6),
                                               ('gemma4-12b-it', 48, 7)])
def test_discovery_cell_count_is_pairs_times_panel_times_spans(slug, count, layers):
    panel = build_localization_candidate_panel(model_slug=slug, actual_layer_count=count)
    pairs = tuple(SimpleNamespace(role='fit', pair_sha256=str(i), family='cross_company',
                                  contrast='entity_context', target_key=SimpleNamespace(ticker=f'T{i % 40}'))
                  for i in range(320))
    issuers = {f'T{i}': f'I{i}' for i in range(40)}
    table = SimpleNamespace(pairs=pairs, parent_sha256='a', table_sha256='b', inputs_manifest_sha256='c')
    desc = runner.descriptor(table, panel, 'discovery', {'issuer_by_ticker': issuers},
                             tuple(f'I{i}' for i in range(32)), None, 32)
    assert desc['planned_cells'] == 256 * layers * 4


def test_selection_rules():
    def row(span, layer, itt, family='cross_company', contrast='entity_context'):
        return dict(family=family, contrast=contrast, span=span, layer=layer, companies=1,
                    eligible=1, flip=0, company_first_itt=itt)
    scores = [row('entity', 0, .5), row('entity', 3, .5), row('entity', 14, .9), row('entity', 31, None),
              row('evidence1', 0, None), row('evidence1', 3, 0.0), row('evidence2', 0, None)]
    sites = {(s['family'], s['contrast'], s['span']): s for s in runner.select_sites(scores, [0, 3, 14, 31])}
    assert len(sites) == 12
    assert sites[('cross_company', 'entity_context', 'entity')]['layers'] == [14, 0]
    assert sites[('cross_company', 'entity_context', 'evidence1')]['layers'] == [3]
    assert sites[('cross_company', 'entity_context', 'evidence2')]['status'] == 'untestable'
    assert sites[('cross_evidence', 'evidence_order', 'instruction')]['layers'] == []
    with pytest.raises(ValueError):
        runner.select_sites([row('entity', 5, .1)], [0, 3])


def test_selection_scores_company_first_with_failures():
    plan, saved = {}, {}
    for i, (ticker, decision, failure, relation) in enumerate([
            ('AAA', 'buy', None, 'opposite'), ('AAA', 'sell', None, 'opposite'),
            ('BBB', 'buy', 'timeout', 'opposite'), ('CCC', 'buy', None, 'same')]):
        pair = SimpleNamespace(family='cross_company', contrast='entity_context', clean_relation=relation,
                               target_key=SimpleNamespace(ticker=ticker))
        plan[str(i)] = (dict(span='entity', layer=0), pair)
        saved[str(i)] = SimpleNamespace(expected_donor=SimpleNamespace(decision='buy'),
            intervention=SimpleNamespace(decision=decision, failure_type=failure))
    (score,) = runner.selection_scores(plan, saved)
    assert score['companies'] == 2 and score['eligible'] == 3 and score['flip'] == 1
    assert score['company_first_itt'] == .25


def test_discovery_then_validation_end_to_end(tmp_path, g):
    root = g.setup['root']
    report = run_phase(g, tmp_path / 'discovery', 'discovery', 3)
    assert report['counts'] == dict(planned=8, executed=8, invalid_parent=0, failure=0)
    assert report['gates'] == 8 and len(g.batches) == 1 and len(g.batches[0]) == 8
    assert report['batch_control'] == dict(calls=4, control_matched=4, cells_in_matched_calls=8,
                                           cells_in_unmatched_calls=0)
    scores = {(r['span'], r['layer']): r['company_first_itt'] for r in report['selection_scores']}
    assert scores[('entity', 0)] == scores[('entity', 1)] == 0.0
    assert scores[('instruction', 1)] == 1.0
    sites = {(s['contrast'], s['span']): s for s in report['selection']}
    assert sites[('entity_context', 'entity')]['layers'] == [0, 1]
    assert sites[('evidence_order', 'entity')]['status'] == 'untestable'
    calls = root.calls
    run_phase(g, tmp_path / 'discovery', 'discovery', 3)
    assert root.calls == calls

    selection = runner.load_selection(tmp_path / 'discovery', g.table, g.panel, g.bindings, ('ib',))
    assert selection['sites'] == report['selection']
    validation = run_phase(g, tmp_path / 'validation', 'validation', 32, selection=selection)
    assert validation['counts']['planned'] == 8 and validation['batch_control']['calls'] == 1
    assert root.rows_seen[-1] == 9
    assert all(row['role'] == 'validation' for row in validation['groups'])
    for name, value in (('grammar', {'g': 2}), ('issuer_by_ticker', {})):
        bad = deepcopy(g.bindings)
        bad[name] = value
        with pytest.raises(ValueError, match='binding'):
            runner.load_selection(tmp_path / 'discovery', g.table, g.panel, bad, ('ib',))
    with pytest.raises(ValueError):
        runner.load_selection(tmp_path / 'discovery', g.table, g.panel, g.bindings, ('ia',))
    clean(root)


def test_missing_or_tampered_batch_record_blocks_resume(tmp_path, g):
    run_phase(g, tmp_path / 'run', 'discovery', 5)
    records = tmp_path / 'run' / 'records'
    batch = sorted(p for p in records.iterdir() if p.name.startswith('batch_'))[0]
    raw = batch.read_text()
    batch.write_text(raw.replace('"control_match":true', '"control_match":false'))
    with pytest.raises(ValueError):
        run_phase(g, tmp_path / 'run', 'discovery', 5)
    batch.unlink()
    calls = g.setup['root'].calls
    with pytest.raises(ValueError, match='batch reference'):
        run_phase(g, tmp_path / 'run', 'discovery', 5)
    assert g.setup['root'].calls == calls
    clean(g.setup['root'])


def test_interrupted_batch_resumes_without_collision(tmp_path, g):
    def interrupted(pair, coordinates):
        result = g.batch(pair, coordinates)
        state['result'] = result
        return result
    state = {}
    original = runner.publish

    def crash_on_first_cell(directory, name, value):
        original(directory, name, value)
        if name.startswith('batch_'):
            raise KeyboardInterrupt
    runner.publish = crash_on_first_cell
    try:
        with pytest.raises(KeyboardInterrupt):
            run_phase(g, tmp_path / 'run', 'discovery', 9, batch=interrupted)
    finally:
        runner.publish = original
    report = run_phase(g, tmp_path / 'run', 'discovery', 9)
    assert report['counts']['executed'] == 8 and report['batch_control']['calls'] == 1
    names = [p.name for p in (tmp_path / 'run' / 'records').iterdir() if p.name.startswith('batch_')]
    assert len(names) == 2
    clean(g.setup['root'])


def test_row_failure_retained_in_itt(tmp_path, g):
    g.setup['root'].poison = {1}
    report = run_phase(g, tmp_path / 'run', 'discovery', 9)
    assert report['counts']['failure'] == 1 and report['counts']['executed'] == 8
    clean(g.setup['root'])


def test_clean_drift_halts_without_rows(tmp_path, g):
    def drift(pair, coordinates):
        g.setup['root'].target = g.setup['sell'] + (g.setup['tokenizer'].eos_token_id,)
        return g.batch(pair, coordinates)
    with pytest.raises(RuntimeError, match='halted group'):
        run_phase(g, tmp_path / 'run', 'discovery', 9, batch=drift)
    files = [p.name for p in (tmp_path / 'run' / 'records').iterdir()]
    assert sum(n.startswith('halt_') for n in files) == 1
    assert not any(n.startswith('batch_') for n in files)
    assert sum(not n.startswith(('halt_', 'gate_')) for n in files) == 0
    clean(g.setup['root'])


def test_batch_drift_is_reported_in_summary(tmp_path, g):
    g.setup['root'].drift = g.setup['sell'] + (g.setup['tokenizer'].eos_token_id,)
    report = run_phase(g, tmp_path / 'run', 'discovery', 5)
    assert report['batch_control'] == dict(calls=2, control_matched=0, cells_in_matched_calls=0,
                                           cells_in_unmatched_calls=8)
    assert all(row['eligible_control_matched'] == 0 for row in report['groups'])
    assert math.isclose(sum(row['flip'] for row in report['groups']), 6)
    clean(g.setup['root'])
