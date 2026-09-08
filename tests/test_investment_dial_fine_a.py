"""Portable A-only diagnostic: no checkpoint or GPU required."""
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('fine_a', Path(__file__).parents[1] / 'scripts/investment_dial_fine_a.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def rows():
    return [dict(ticker=f'T{i}', split='A', positive_count=2) for i in range(85) for _ in range(4)]


def result():
    return {'selected': {'layer': 15, 'neuron': 8490}}


def test_subset_and_coordinate():
    source = rows() + [dict(ticker='excluded', split=s, positive_count=2) for s in ('B', 'test')]
    assert m.select_rows(source, result()) == rows()
    with pytest.raises(ValueError, match='340'):
        m.select_rows(rows()[:-1], result())
    with pytest.raises(ValueError, match='coordinate'):
        m.select_rows(rows(), {'selected': None})


def test_device_guard(monkeypatch):
    args = m.parser().parse_args(['--source-run', 'source', '--model', 'model', '--run-id', 'test'])
    monkeypatch.setattr(m.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='unavailable'):
        m.run(args)
    args.device = 'cpu'
    monkeypatch.setattr(m.torch.cuda, 'is_available', lambda: True)
    with pytest.raises(ValueError, match='CPU execution'):
        m.run(args)


def test_full_mocked_lifecycle(tmp_path, monkeypatch):
    source = rows() + [dict(ticker='excluded', split='test', positive_count=2)]
    bundle = ({'prepare/trials.json': source, 'analyze/result.json': result()}, 'digest')
    monkeypatch.setattr(m.p, 'verified_run', lambda *a: bundle)
    monkeypatch.setattr(m.p, 'load_model', lambda *a, **k: (object(), object(), 'cpu'))
    monkeypatch.setattr(m.p, '_parent', lambda *a: ({'model_identity': {}, 'runtime': {}}, source, result(), 'digest'))
    monkeypatch.setattr(m.p, '_runtime', lambda *a: {'device': 'cpu'})
    monkeypatch.setattr(m.p, '_source', lambda: {})
    monkeypatch.setattr(m.p, 'frozen_eval', lambda *a: nullcontext())
    calls = []

    def decisions(model, tokenizer, device, batch, coordinate, delta, budget):
        assert batch[0]['split'] == 'A' and batch[0]['positive_count'] == 2
        assert coordinate == (15, 8490) and budget == 256
        calls.append(delta)
        return [dict(decision='buy', json_object=True, schema_valid=True, delta=delta)]

    monkeypatch.setattr(m.p, '_decisions', decisions)
    args = m.parser().parse_args(['--source-run', 'source', '--model', 'fake-model',
                                 '--run-id', 'diagnostic', '--device', 'cpu', '--artifact-root', str(tmp_path)])
    directory = m.run(args)
    assert len(calls) == 3060
    assert all(calls.count(d) == 340 for d in m.DELTAS)
    assert len(list((directory / 'forward').glob('*.jsonl'))) == 9
    actual = json.loads((directory / 'analyze/result.json').read_text())
    assert actual['certified'] is False
    assert len(actual['summaries']) == 9
    assert all(s['n'] == 340 and s['pi'] == 1 for s in actual['summaries'])
    assert len(json.loads((directory / 'prepare/trials.json').read_text())) == 340
