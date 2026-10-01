"""Diagnostic CLI boundaries; no checkpoint loading or GPU execution."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke_stance_checkpoint', ROOT / 'scripts/smoke_stance_checkpoint.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
from llm_bias.core.artifact_paths import sha256_bytes

PACK = Path(__file__).resolve().parents[1] / 'data/concept-cone-steering/rebuild-v1/compiled'


def arguments(tmp_path):
    return ['--model', str(tmp_path), '--inputs', str(PACK), '--output', str(tmp_path / 'smoke.json'),
            '--stop-token-id', '7', '--stop-token-id', '9']


def test_jlens_generation_adapter():
    from types import SimpleNamespace
    hf, tokenizer, layers = object(), object(), []
    wrapped = SimpleNamespace(_hf_model=hf, tokenizer=tokenizer, layers=layers)
    adapted = smoke.generation_adapter(wrapped)
    assert adapted.hf_model is hf and adapted.tokenizer is tokenizer and adapted.layers is layers
    assert not hasattr(wrapped, 'hf_model')
    with pytest.raises(ValueError):
        smoke.generation_adapter(SimpleNamespace(tokenizer=tokenizer, layers=layers))


def test_parser_defaults(tmp_path):
    args = smoke.parser().parse_args(arguments(tmp_path))
    assert args.stop_token_id == [7, 9]
    assert args.max_new_tokens == 512
    assert args.timeout_seconds == 180
    assert args.layer == 0
    assert args.hook_site == 'pre'
    with pytest.raises(SystemExit):
        smoke.parser().parse_args(arguments(tmp_path)[:-4])


def test_approved_full_pack_and_formal_manifest():
    inputs = smoke.load_baseline_inputs(PACK)
    member, pair = smoke.select_diagnostic(inputs)
    assert len(inputs.members) == 503
    assert len(set(inputs.issuer_by_ticker.values())) == 500
    assert len(inputs.pairs) == 2012
    assert inputs.manifest_sha256 == sha256_bytes((PACK / 'inputs_manifest.json').read_bytes())
    assert (pair.ticker, pair.condition, pair.trial_id) == min(
        (p.ticker, p.condition, p.trial_id) for p in inputs.pairs)
    assert member.ticker == pair.ticker


def test_failure_json_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(smoke.torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(smoke, 'load_model', lambda *a, **kw: pytest.fail('must not load on CPU'))
    assert smoke.main(arguments(tmp_path)) == 1
    path = tmp_path / 'smoke.json'
    original = path.read_bytes()
    record = json.loads(original)
    assert record['research_eligible'] is False
    assert record['new_cohort'] is False
    assert record['planned_rows'] == 2012
    assert record['diagnostic_rows'] == 1
    assert record['full_manifest_hash'] == smoke.load_baseline_inputs(PACK).manifest_sha256
    assert record['phase'] == 'load_model'
    assert record['halt_reason'] == 'exception'
    assert record['error']['type'] == 'RuntimeError'
    assert record['passed'] is False
    assert len(record['provenance']['git_head']) == 40
    with pytest.raises(FileExistsError):
        smoke.main(arguments(tmp_path))
    assert path.read_bytes() == original


def test_invalid_inputs_record_failure(tmp_path):
    argv = arguments(tmp_path)
    argv[3] = str(tmp_path)
    assert smoke.main(argv) == 1
    record = json.loads((tmp_path / 'smoke.json').read_bytes())
    assert record['phase'] == 'inputs'
    assert record['error']['type'] == 'ValueError'
    assert record['research_eligible'] is False
    assert 'arms' not in record


def test_utf8_safe_exception(tmp_path, monkeypatch):
    def fail(args, record):
        record['phase'] = 'injected'
        raise ValueError('bad\ud800')
    monkeypatch.setattr(smoke, 'run_smoke', fail)
    assert smoke.main(arguments(tmp_path)) == 1
    record = json.loads((tmp_path / 'smoke.json').read_bytes())
    assert record['error']['message'] == 'bad?'
    assert record['phase'] == 'injected'
