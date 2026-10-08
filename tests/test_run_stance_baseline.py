"""Focused diagnostic runner tests without checkpoints."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
from test_stance_baseline_adapter import bundle, result
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_stance_baseline.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def policy(**kw):
    return r.StructuredGenerationPolicy(**(dict(max_new_tokens=512, use_cache=True,
        pad_token_id=0, timeout_seconds=180, channel_policy='plain_json') | kw))


def prompt():
    return NS(template_sha256=r.sha256_json('body'), schema_sha256=r.sha256_json('schema'),
              wrapper_policy={'chat_template': 'actual'}, inference_token_ids=(1, 2))


def identity(inputs, change=None):
    p = prompt()
    metadata = dict(model={'resolved_path': '/fake'}, code={'source_sha256': {'runner': 'source'}}, backend={'python': 'fake'})
    if change == 'source':
        metadata['code']['source_sha256']['runner'] = 'changed'
    if change == 'wrapper':
        p.wrapper_policy = {'chat_template': 'changed'}
    cap = NS(schema_sha256=p.schema_sha256, stop_token_ids=(3,))
    return r.build_identity(inputs, p, cap, policy(use_cache=change != 'policy'), metadata, 'plain_json')[0]


def test_full_plan(bundle):
    inputs, _ = bundle
    plan = r.build_baseline_plan(inputs, identity(inputs))
    assert len(plan.keys) == 2012
    assert plan.identity.parent_sha256 == plan.identity.operator_sha256 == 'not_applicable'


@pytest.mark.parametrize('change', ['source', 'wrapper', 'policy'])
def test_resume_binding(tmp_path, bundle, change):
    inputs, _ = bundle
    plan = r.build_baseline_plan(inputs, identity(inputs))
    with r.open_baseline_store(tmp_path / 's', inputs=inputs, plan=plan, shard_index=0, num_shards=1):
        pass
    changed = r.build_baseline_plan(inputs, identity(inputs, change))
    with pytest.raises(ValueError, match='registration'):
        r.open_baseline_store(tmp_path / 's', inputs=inputs, plan=changed, shard_index=0, num_shards=1)


@pytest.mark.parametrize('route', ['plain_json', 'harmony_no_tools'])
def test_route_failed_retained(tmp_path, bundle, monkeypatch, route):
    inputs, plan = bundle
    key = plan.keys[0]
    calls = []
    generated = result(decision=None, reason=None, failure_type='timeout', finish_reason='timeout',
                       decision_complete=False, schema_complete=False, reason_valid=False)
    def driver(model, tokenizer, ids, capability, *, policy):
        assert ids.tolist() == [[1, 2]] and ids.device.type == 'cpu'
        calls.append(policy.channel_policy)
        return generated
    monkeypatch.setattr(r, 'render_decision_prompt', lambda *a, **k: prompt())
    for name, admitted in [('generate_structured', 'plain_json'), ('generate_harmony_structured', 'harmony_no_tools')]:
        monkeypatch.setattr(r, name, driver if route == admitted else lambda *a, **k: pytest.fail('wrong route'))
    class One:
        def __init__(self, store):
            self.store = store
        @property
        def pending_keys(self):
            return (key,) if key in self.store.pending_keys else ()
        def record(self, *args):
            return self.store.record(*args)
    for _ in range(2):
        with r.open_baseline_store(tmp_path / 's', inputs=inputs, plan=plan, shard_index=0, num_shards=1) as store:
            r.run_rows(One(store), inputs, prompt(), None, None, None, 'cpu', None, policy(channel_policy=route))
            assert store.generation_for(key).to_dict() == generated.to_dict()
            report = r.summary(store, plan, 0, 1)
            assert report['executed_count'] == 1 and report['missing_count'] == 2011
            assert report['failure_counts'] == {'timeout': 1}
            assert not report['complete_global'] and not report['research_eligible']
            assert not any(report['gates'].values())
    assert calls == [route]


def test_prompt_binding(bundle, monkeypatch):
    inputs, plan = bundle
    wrong = prompt()
    wrong.template_sha256 = r.sha256_json('different')
    monkeypatch.setattr(r, 'render_decision_prompt', lambda *a, **k: wrong)
    with pytest.raises(ValueError, match='binding'):
        r.run_rows(NS(pending_keys=(plan.keys[0],)), inputs, prompt(), None, None, None, 'cpu', None, policy())


def test_immutable_metadata(tmp_path):
    path = tmp_path / 'execution_metadata.json'
    r.immutable_json(path, {'fixed': True})
    before = path.read_bytes()
    r.immutable_json(path, {'fixed': True})
    with pytest.raises(ValueError):
        r.immutable_json(path, {'fixed': False})
    assert path.read_bytes() == before


def test_cli():
    args = r.parser().parse_args(['--model', 'm', '--inputs', 'i', '--output-dir', 'o', '--stop-token-id', '1', '--stop-token-id', '2', '--use-cache', 'false'])
    assert args.stop_token_id == [1, 2] and not args.use_cache
    assert args.max_new_tokens == 512 and args.timeout_seconds == 180
    assert args.shard_index == 0 and args.num_shards == 1
    with pytest.raises(SystemExit):
        r.parser().parse_args(['--ticker', 'A'])


def test_exception_reports(tmp_path, monkeypatch):
    def fail(args):
        raise ValueError('binding')
    monkeypatch.setattr(r, 'run', fail)
    argv = ['--model', 'm', '--inputs', 'i', '--output-dir', str(tmp_path), '--stop-token-id', '1']
    assert r.main(argv) == r.main(argv) == 1
    assert len(list(tmp_path.glob('error-*.json'))) == 2
