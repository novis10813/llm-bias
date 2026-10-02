"""Only parent token-budget failures are retried, never overwritten."""
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace as NS
import json
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import recover_stance_baseline_truncations as r
from test_stance_baseline_adapter import bundle, result
from test_stance_baseline_parent import parent_bundle, _records_for, _summary_for, BASE_SPECS
from llm_bias.core.stance_baseline_parent import CompletedBaseline
from llm_bias.core.artifact_paths import canonical_json_bytes


@pytest.fixture(scope='module')
def parent_fixture(parent_bundle):
    records, rows, _ = _records_for(parent_bundle, BASE_SPECS)
    parent = CompletedBaseline(parent_bundle['plan'], tuple(rows.values()), 'a' * 64,
        canonical_json_bytes(parent_bundle['metadata']) + b'\n',
        canonical_json_bytes(_summary_for(parent_bundle['plan'], rows)) + b'\n',
        b'{}', MappingProxyType(records))
    return parent_bundle['inputs'], parent


def generated(policy, failure=False):
    changes = {'_provenance_bytes': canonical_json_bytes({'generation_policy': r.asdict(policy)})}
    if failure:
        changes.update(decision=None, reason=None, failure_type='truncated', finish_reason='token_budget',
            json_payload='{', generated_text='{', decision_complete=False, schema_complete=False,
            reason_valid=False)
    return result(**changes)


def test_select_only_parent_truncations(parent_fixture):
    _, parent = parent_fixture
    keys = r.recovery_keys(parent)
    assert len(keys) == 1
    assert parent.generation_for(keys[0]).failure_type == 'truncated'
    assert len(parent.rows) == 2012


@pytest.mark.parametrize('budget', [True, 512, 1, 4096.0, '4096'])
def test_invalid_budget(parent_fixture, budget):
    with pytest.raises(ValueError):r.recovery_policy(parent_fixture[1], budget, 1200)


@pytest.mark.parametrize('timeout', [True, 0, -1, float('nan'), float('inf'), '1'])
def test_invalid_timeout(parent_fixture, timeout):
    with pytest.raises(ValueError):r.recovery_policy(parent_fixture[1], 4096, timeout)


def test_new_policy_inherits_other_controls(parent_fixture):
    p = r.recovery_policy(parent_fixture[1], 4096, 1200)
    old = parent_fixture[1].metadata['bindings']['generation_policy']['policy']
    assert r.asdict(p) == old | {'max_new_tokens':4096,'timeout_seconds':1200}


@pytest.mark.parametrize('failed', [False, True])
def test_recovery_and_resume_never_retries_committed(tmp_path, parent_fixture, failed):
    inputs, parent = parent_fixture;keys=r.recovery_keys(parent);p=r.recovery_policy(parent,4096,1200)
    calls=[];registration={'parent':parent.parent_sha256,'policy':r.asdict(p)}
    def generate(key):calls.append(key);return generated(p,failed)
    with r.recovery_store(tmp_path/'run',registration) as records:
        summary=r.run_recovery(records,keys,parent,inputs,generate,policy=p)
    raw=(tmp_path/'run/records'/next(iter(records.iterdir())).name).read_bytes()
    with r.recovery_store(tmp_path/'run',registration) as records:
        again=r.run_recovery(records,keys,parent,inputs,generate,policy=p)
    assert len(calls)==1 and summary==again and summary['complete']
    assert summary['executed']==1 and not summary['research_eligible']
    assert summary['failure_counts']==({'truncated':1} if failed else {'none':1})
    assert next(records.iterdir()).read_bytes()==raw
    assert parent.generation_for(keys[0]).failure_type=='truncated'


def test_bad_registration_lock_and_no_overwrite(tmp_path):
    root=tmp_path/'run';registration={'fixed':True}
    with r.recovery_store(root,registration):
        with pytest.raises(BlockingIOError):
            with r.recovery_store(root,registration):pass
    with pytest.raises(ValueError):
        with r.recovery_store(root,{'fixed':False}):pass
    r.publish(root,'file.json',{'a':1})
    with pytest.raises(FileExistsError):r.publish(root,'file.json',{'a':2})
    assert json.loads((root/'file.json').read_bytes())=={'a':1}


@pytest.mark.parametrize('kind',['staging','foreign','tamper','symlink','policy'])
def test_recovery_corruption_rejected(tmp_path,parent_fixture,kind):
    inputs,parent=parent_fixture;keys=r.recovery_keys(parent);p=r.recovery_policy(parent,4096,1200)
    with r.recovery_store(tmp_path/'run',{'fixed':True}) as records:
        r.run_recovery(records,keys,parent,inputs,lambda k:generated(p),policy=p)
        path=next(records.iterdir());value=json.loads(path.read_bytes())
        if kind=='staging':(records/'.pending.tmp').write_text('{}')
        elif kind=='foreign':path.rename(records/'foreign.json')
        elif kind=='symlink':path.unlink();path.symlink_to(tmp_path/'elsewhere')
        elif kind=='policy':
            value['generation']['provenance']['generation_policy']['max_new_tokens']=8192
            path.write_bytes(canonical_json_bytes(value)+b'\n')
        else:
            value['parent_sha256']='b'*64;path.write_bytes(canonical_json_bytes(value)+b'\n')
        with pytest.raises((ValueError,OSError)):
            r.run_recovery(records,keys,parent,inputs,lambda k:pytest.fail('not resume corrupted'),policy=p)


def test_prompt_and_capability_binding():
    prompt=NS(wrapper_policy={'thinking':True},template_sha256='body')
    bindings={'template':{'actual_wrapper_record':prompt.wrapper_policy,'body_template_sha256':'body'}}
    r.check_prompt(prompt,bindings)
    with pytest.raises(ValueError):r.check_prompt(NS(wrapper_policy={},template_sha256='body'),bindings)
    cap=NS(**{n:n for n in ('schema_sha256','schema_bytes_sha256','tokenizer_sha256',
        'tokenizer_info_sha256','head_vocab_size','grammar_sha256')},stop_token_ids=(1,))
    ref=vars(cap).copy();ref['stop_token_ids']=[1]
    r.check_capability(cap,ref)
    ref['tokenizer_sha256']='bad'
    with pytest.raises(ValueError):r.check_capability(cap,ref)


def test_cli_no_cohort_override():
    args=r.parser().parse_args(['--inputs','i','--parent','p','--model','m','--output-dir','o'])
    assert args.max_new_tokens==4096 and args.timeout_seconds==1200
    with pytest.raises(SystemExit):r.parser().parse_args(['--ticker','A'])


def test_empty_recovery(parent_fixture,tmp_path):
    inputs,parent=parent_fixture;p=r.recovery_policy(parent,4096,1200)
    with r.recovery_store(tmp_path/'run',{}) as records:
        summary=r.run_recovery(records,(),parent,inputs,None,policy=p)
    assert summary['complete'] and summary['executed']==summary['selected_truncated']==0
