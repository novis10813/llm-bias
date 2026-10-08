"""Lossless B2 conversion with synthetic unverified identities, never checkpoints."""
from dataclasses import fields, replace
from pathlib import Path
import shutil

import pytest

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.experiment_contract import ExperimentPlan, PlanIdentity, progress
from llm_bias.core.inference.structured_output import StructuredGenerationResult
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_plan import build_baseline_plan
from llm_bias.core.stance_baseline_adapter import baseline_execution_row, baseline_gate_input
from llm_bias.core.stance_gates import evaluate_noop_gate


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp('adapter-inputs')
    pack = Path(__file__).resolve().parents[1] / 'data/concept-cone-steering/rebuild-v1/compiled'
    for path in pack.iterdir():
        shutil.copyfile(path, directory / path.name)
    inputs = load_baseline_inputs(directory)
    values = {f.name: sha256_json('synthetic unverified ' + f.name) for f in fields(PlanIdentity)}
    values.update(population_sha256=inputs.membership_sha256, issuer_sha256=inputs.issuer_mapping_sha256,
                  roles_sha256=inputs.roles_sha256, evidence_sha256=inputs.evidence_sha256,
                  operator_sha256='not_applicable', parent_sha256='not_applicable')
    return inputs, build_baseline_plan(inputs, PlanIdentity(**values))


def result(**changes):
    tokens = (11, 22, 33, 2**63 - 1)
    return replace(StructuredGenerationResult(
        '{"decision":"buy","reason":"evidence"}<eos>', tokens, sha256_json(list(tokens)),
        '{"decision":"buy","reason":"evidence"}', 'buy', 'evidence', True, True, True,
        'eos', None, 0.125, canonical_json_bytes({'synthetic': True, 'nested': {'a': [1]}})), **changes)


def assert_lossless(bundle, generated):
    inputs, plan = bundle
    row = baseline_execution_row(plan, inputs, plan.keys[0], generated)
    assert row.key == plan.keys[0]
    assert row.outcome.issuer_id == next(m.issuer_id for m in inputs.members if m.ticker == row.key.ticker)
    for name in ('decision', 'decision_complete', 'schema_complete', 'reason_valid', 'finish_reason', 'failure_type'):
        assert getattr(row.outcome, name) == getattr(generated, name)
    gate = baseline_gate_input(generated)
    assert gate.token_ids == generated.generated_token_ids
    assert gate.text == generated.generated_text
    for name in ('decision', 'decision_complete', 'schema_complete', 'reason_valid', 'failure_type'):
        assert getattr(gate, name) == getattr(generated, name)
    assert row.to_json()
    return row, gate


@pytest.mark.parametrize('decision', ['buy', 'sell'])
@pytest.mark.parametrize('form', ['plain', 'harmony'])
@pytest.mark.parametrize('finish', ['eos', 'schema_complete'])
def test_success(bundle, decision, form, finish):
    payload = '{"decision":"' + decision + '","reason":"evidence"}'
    text = payload + '<eos>' if form == 'plain' else (
        '<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>'
        + payload + '<|fim_suffix|>')
    generated = result(decision=decision, generated_text=text, json_payload=payload, finish_reason=finish)
    _, gate = assert_lossless(bundle, generated)
    assert gate.text != payload
    assert evaluate_noop_gate(gate, gate, gate, gate, config_hash='1' * 64).passed


FAILURES = [('exception', 'exception'), ('timeout', 'timeout'), ('no_legal_token', 'no_legal_token'),
            ('truncated', 'token_budget'), ('unsupported_channel', 'unsupported'),
            ('unsupported_tokenizer', 'unsupported'), ('invalid_json', 'eos'),
            ('invalid_schema', 'schema_complete'), ('invalid_reason', 'schema_complete')]


@pytest.mark.parametrize('failure,finish', FAILURES)
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_failures(bundle, failure, finish, diagnostic, empty):
    schema = diagnostic and failure not in ('truncated', 'invalid_json', 'invalid_schema')
    generated = result(decision=None, reason=None, failure_type=failure, finish_reason=finish,
                       decision_complete=diagnostic, schema_complete=schema,
                       reason_valid=schema and failure != 'invalid_reason',
                       error_message='execution diagnostic', decode_error='secondary diagnostic')
    if empty:
        generated = replace(generated, generated_token_ids=(), generated_token_sha256=sha256_json([]),
                            generated_text='', json_payload='')
    row, gate = assert_lossless(bundle, generated)
    state = progress(bundle[1], (row,))
    assert (state['planned'], state['executed'], state['missing']) == (2012, 1, 2011)
    assert not state['complete'] and not state['eligible']
    assert not evaluate_noop_gate(gate, gate, gate, gate, config_hash='1' * 64).passed


BAD = [
    {'generated_token_ids': [1]}, {'generated_token_ids': (True,)}, {'generated_token_ids': (-1,)},
    {'generated_token_ids': (2**63,)}, {'generated_token_ids': (1.0,)},
    {'generated_token_sha256': '0' * 64}, {'generated_token_sha256': None},
    *[{name: value} for name in ('generated_text', 'json_payload', 'reason', 'error_message', 'decode_error')
      for value in ('\ud800', 7)],
    *[{'elapsed_seconds': value} for value in (-1, float('nan'), float('inf'), float('-inf'), True, '1', None)],
    *[{name: 1} for name in ('decision_complete', 'schema_complete', 'reason_valid')],
    {'decision': 'hold'}, {'decision': None}, {'decision': []}, {'finish_reason': 'unknown'},
    {'failure_type': 'unknown'}, {'finish_reason': []}, {'failure_type': []},
    {'schema_complete': False}, {'decision_complete': False}, {'reason_valid': False},
    {'reason': ''}, {'reason': ' \t\n'}, {'reason': None}, {'error_message': ''}, {'decode_error': ''},
    {'generated_text': ''}, {'json_payload': ''},
    {'generated_token_ids': (), 'generated_token_sha256': sha256_json([])},
    *[{'_provenance_bytes': value} for value in (
        b'{}\n', b'{ "a":1}', b'{"a":1,"a":2}', b'{"a":{"b":1,"b":2}}', b'{"a":NaN}',
        b'{"a":Infinity}', b'{"a":1e999}', b'[]', b'null', b'bad', b'\xff', b'{"a":"\\ud800"}',
        '{}', bytearray(b'{}'))],
]


@pytest.mark.parametrize('changes', BAD)
def test_malformed_result_rejected_by_both(bundle, changes):
    generated = result(**changes)
    inputs, plan = bundle
    with pytest.raises(ValueError):
        baseline_gate_input(generated)
    with pytest.raises(ValueError):
        baseline_execution_row(plan, inputs, plan.keys[0], generated)


@pytest.mark.parametrize('failure,finish', FAILURES)
@pytest.mark.parametrize('forge', ['decision', 'reason', 'finish'])
def test_failed_primary_or_finish_never_repaired(bundle, failure, finish, forge):
    generated = result(decision=None, reason=None, decision_complete=False, schema_complete=False,
                       reason_valid=False, failure_type=failure, finish_reason=finish)
    generated = replace(generated, **{forge if forge != 'finish' else 'finish_reason':
                                    'buy' if forge == 'decision' else 'diagnostic' if forge == 'reason' else 'timeout' if finish != 'timeout' else 'eos'})
    with pytest.raises(ValueError):
        baseline_gate_input(generated)
    with pytest.raises(ValueError):
        baseline_execution_row(bundle[1], bundle[0], bundle[1].keys[0], generated)


@pytest.mark.parametrize('change', ['key', 'stage', 'arm', 'dose', 'inputs', 'input-hash', 'plan-hash',
                                    'truncated-plan', 'gates', 'plan-type', 'key-type', 'result-type'])
def test_caller_violations_raise(bundle, change):
    inputs, plan = bundle
    key, generated = plan.keys[0], result()
    if change == 'key': key = replace(key, ticker='FOREIGN')
    elif change in ('stage', 'arm', 'dose'): key = replace(key, **{change: '1' if change == 'dose' else 'foreign'})
    elif change == 'inputs': inputs = replace(inputs, pairs=inputs.pairs[:-1])
    elif change == 'input-hash': inputs = replace(inputs, issuer_mapping_sha256='0' * 64)
    elif change == 'plan-hash': plan = replace(plan, identity=replace(plan.identity, evidence_sha256='0' * 64))
    elif change == 'truncated-plan': plan = replace(plan, keys=plan.keys[:-1])
    elif change == 'gates': plan = replace(plan, gate_names=('no_op',))
    elif change == 'plan-type': plan = object()
    elif change == 'key-type': key = object()
    elif change == 'result-type': generated = object()
    with pytest.raises(ValueError):
        baseline_execution_row(plan, inputs, key, generated)


def test_non_result_gate():
    with pytest.raises(ValueError):
        baseline_gate_input(object())


@pytest.mark.parametrize('driver', ['plain', 'harmony'])
@pytest.mark.parametrize('failed', [False, True])
def test_genuine_cpu_fake_driver_results(bundle, driver, failed):
    # Reuse existing tiny CPU driver fixtures; no checkpoint or model loader.
    if driver == 'plain':
        import test_structured_output as plain
        tokenizer = plain.make_tokenizer()
        capability = plain.compile_decision_grammar(
            tokenizer, len(tokenizer) + 35,
            [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<stop>')])
        fake = plain.FakeGenerate(tokenizer, capability, raise_at=0 if failed else None)
        generated = plain.run(fake, tokenizer, capability)
    else:
        import test_harmony_generation as harmony
        tokenizer = harmony.make_tokenizer()
        contract = harmony.contract_for(tokenizer)
        capability = harmony.compile_harmony_decision_grammar(tokenizer, len(tokenizer) + 7, contract)
        target = harmony.turn(tokenizer, contract, analysis=['think before final'])
        controls = harmony.policy(tokenizer, max_new_tokens=1) if failed else None
        generated, _ = harmony.run((tokenizer, contract, capability), target, controls=controls)
    assert (generated.failure_type is not None) == failed
    _, gate = assert_lossless(bundle, generated)
    assert evaluate_noop_gate(gate, gate, gate, gate, config_hash='1' * 64).passed == (not failed)


@pytest.mark.parametrize('changes', [
    {'failure_type': 'invalid_reason', 'finish_reason': 'eos', 'reason_valid': True},
    {'failure_type': 'invalid_json', 'finish_reason': 'eos', 'schema_complete': True},
    {'failure_type': 'exception', 'finish_reason': 'exception', 'decision_complete': False,
     'schema_complete': True},
    {'failure_type': 'exception', 'finish_reason': 'exception', 'schema_complete': False,
     'reason_valid': True},
])
def test_invalid_diagnostic_flags(bundle, changes):
    generated = result(decision=None, reason=None, **changes)
    with pytest.raises(ValueError):
        baseline_gate_input(generated)
    with pytest.raises(ValueError):
        baseline_execution_row(bundle[1], bundle[0], bundle[1].keys[0], generated)


def test_directly_forged_plan_and_exact_copy(bundle):
    inputs, plan = bundle
    copied = ExperimentPlan.from_json(plan.to_json())
    assert baseline_execution_row(copied, replace(inputs), copied.keys[0], result()).key == copied.keys[0]
    object.__setattr__(copied, 'keys', copied.keys[:-1])
    with pytest.raises(ValueError):
        baseline_execution_row(copied, inputs, copied.keys[0], result())
