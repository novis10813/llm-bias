"""BS1 persistence tests with full approved inputs and unverified identities."""
import json
import multiprocessing
import os
import subprocess
import sys
from dataclasses import replace

import pytest
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
import llm_bias.core.stance_baseline_store as module
from test_stance_baseline_adapter import bundle, result, FAILURES
from llm_bias.core.stance_baseline_store import open_baseline_store


def test_resume(tmp_path, bundle):
    inputs, plan = bundle
    path = tmp_path / 'store'
    key = plan.keys[0]
    with open_baseline_store(path, inputs=inputs, plan=plan, shard_index=0, num_shards=1) as store:
        row = store.record(key, result())
        assert store.rows == (row,)
    with open_baseline_store(path, inputs=inputs, plan=plan, shard_index=0, num_shards=1) as store:
        assert store.recorded_keys == (key,)
        assert store.generation_for(key).to_dict() == result().to_dict()
        with pytest.raises(ValueError):
            store.record(key, result())


def opening(path, bundle, **changes):
    inputs, plan = bundle
    return open_baseline_store(path, **(dict(inputs=inputs, plan=plan, shard_index=0, num_shards=1) | changes))


@pytest.mark.parametrize('failure,finish', FAILURES)
def test_failed_is_executed(tmp_path, bundle, failure, finish):
    key = bundle[1].keys[0]
    generated = result(decision=None, reason=None, failure_type=failure, finish_reason=finish,
                       decision_complete=False, schema_complete=False, reason_valid=False,
                       json_payload='{malformed', error_message='diagnostic', decode_error='decode')
    with opening(tmp_path / 's', bundle) as store:
        store.record(key, generated)
        assert key not in store.pending_keys
    with opening(tmp_path / 's', bundle) as store:
        assert store.recorded_keys == (key,)
        assert store.generation_for(key).to_dict() == generated.to_dict()


@pytest.mark.parametrize('harmony', [False, True])
def test_defensive_export_and_closed(tmp_path, bundle, harmony):
    key = bundle[1].keys[0]
    generated = result(generated_text=('<|channel|>analysis<|message|>think<|end|>'
                                      '<|channel|>final<|message|>' + result().generated_text)
                       if harmony else result().generated_text)
    store = opening(tmp_path / 's', bundle)
    assert store.__enter__() is store
    store.record(key, generated)
    export = store.generation_for(key).provenance
    export['nested']['a'].append(99)
    assert store.generation_for(key).to_dict() == generated.to_dict()
    store.close()
    store.close()
    for operation in (lambda: store.rows, lambda: store.recorded_keys, lambda: store.pending_keys,
                      lambda: store.generation_for(key), lambda: store.record(key, generated), store.__enter__):
        with pytest.raises(RuntimeError):
            operation()


@pytest.mark.parametrize('layout', ['empty', 'lock', 'staging', 'extra', 'symlink', 'recordsymlink'])
def test_bad_layout(tmp_path, bundle, layout):
    path = tmp_path / 's'
    if layout in ('empty', 'lock'):
        path.mkdir()
        if layout == 'lock':
            (path / 'store.lock').touch()
    else:
        opening(path, bundle).close()
        if layout == 'staging':
            (path / 'records/.pending-abrupt.tmp').write_bytes(b'torn')
        elif layout == 'extra':
            (path / 'extra').mkdir()
        elif layout == 'symlink':
            (path / 'registration.json').unlink()
            (path / 'registration.json').symlink_to(tmp_path / 'elsewhere')
        else:
            (path / 'records/evil.json').symlink_to(path / 'registration.json')
    with pytest.raises(ValueError):
        opening(path, bundle)


@pytest.mark.parametrize('change', [dict(shard_index=1, num_shards=2), dict(num_shards=2),
                                  dict(shard_index=True), dict(num_shards=1.0)])
def test_binding(tmp_path, bundle, change):
    path = tmp_path / 's'
    opening(path, bundle).close()
    with pytest.raises(ValueError):
        opening(path, bundle, **change)


@pytest.mark.parametrize('kind', ['inputs', 'plan', 'foreign'])
def test_authority_and_ownership(tmp_path, bundle, kind):
    path = tmp_path / 's'
    inputs, plan = bundle
    if kind == 'inputs':
        with pytest.raises(ValueError):
            opening(path, bundle, inputs=replace(inputs, manifest_sha256='0' * 64))
        assert not path.exists()
    elif kind == 'plan':
        opening(path, bundle).close()
        changed = replace(plan, identity=replace(plan.identity, model_sha256='0' * 64))
        with pytest.raises(ValueError):
            opening(path, bundle, plan=changed)
    else:
        with opening(path, bundle, num_shards=2) as store:
            key = next(key for key in plan.keys if key not in store.pending_keys)
            with pytest.raises(ValueError): store.record(key, result())


@pytest.mark.parametrize('stage', ['records', 'file', 'link', 'late'])
def test_initialization_failure_no_recovery(tmp_path, bundle, monkeypatch, stage):
    path = tmp_path / 's'
    fsync, link = os.fsync, os.link
    calls = []
    def failing_fsync(fd):
        calls.append(fd)
        # Empty records directory, registration file, root publication, root, parent.
        target = {'records': 1, 'file': 2, 'late': 5}.get(stage)
        if len(calls) == target: raise OSError('init fsync')
        return fsync(fd)
    def failing_link(*args, **kwargs):
        if stage == 'link': raise OSError('init link')
        return link(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'fsync', failing_fsync)
        patch.setattr(module.os, 'link', failing_link)
        with pytest.raises(ValueError): opening(path, bundle)
    inode = (path / 'store.lock').stat().st_ino
    if stage == 'late':
        with opening(path, bundle) as store: assert store.recorded_keys == ()
    else:
        with pytest.raises(ValueError): opening(path, bundle)
    assert (path / 'store.lock').stat().st_ino == inode


def test_no_overwrite_unindexed_destination(tmp_path, bundle):
    path = tmp_path / 's'
    key = bundle[1].keys[0]
    with opening(path, bundle) as store:
        destination = path / 'records' / (sha256_json(key.to_dict()) + '.json')
        destination.write_bytes(b'external existing history')
        with pytest.raises(ValueError, match='already exists'): store.record(key, result())
        assert destination.read_bytes() == b'external existing history'
        assert not list((path / 'records').glob('.pending-*'))


def test_postpublication_memory_failure(tmp_path, bundle):
    class Broken(dict):
        def __setitem__(self, key, value):
            raise RuntimeError('injected memory update')
    path = tmp_path / 's'
    key = bundle[1].keys[0]
    store = opening(path, bundle)
    store._entries = Broken()
    with pytest.raises(RuntimeError, match='memory update'): store.record(key, result())
    with pytest.raises(RuntimeError, match='poisoned'): _ = store.rows
    store.close()
    with opening(path, bundle) as reopened: assert reopened.recorded_keys == (key,)


def test_empty_shard(tmp_path, bundle):
    plan = bundle[1]
    count = 10000
    index = next(i for i in range(count) if not plan.keys_for_shard(i, count))
    with opening(tmp_path / 's', bundle, shard_index=index, num_shards=count) as store:
        assert store.rows == store.recorded_keys == store.pending_keys == ()
        assert not hasattr(store, 'finalize')
        assert not hasattr(store, 'research_eligible')


@pytest.mark.parametrize('corruption', ['utf8', 'duplicate', 'nan', 'overflow', 'whitespace', 'torn',
                                       'surrogate', 'bool', 'float', 'row', 'filename', 'fields', 'tokens', 'provenance'])
def test_corrupt_record(tmp_path, bundle, corruption):
    path = tmp_path / 's'
    key = bundle[1].keys[0]
    with opening(path, bundle) as store:
        store.record(key, result())
    file = next((path / 'records').iterdir())
    data = json.loads(file.read_bytes())
    raw = None
    if corruption == 'utf8': raw = b'\xff\n'
    elif corruption == 'duplicate': raw = b'{"a":1,"a":2}\n'
    elif corruption == 'nan': raw = b'{"a":NaN}\n'
    elif corruption == 'overflow': raw = b'{"a":1e999}\n'
    elif corruption == 'whitespace': raw = b' ' + file.read_bytes()
    elif corruption == 'torn': raw = file.read_bytes()[:-5]
    elif corruption == 'surrogate': raw = b'{"a":{"\\ud800":"x"}}\n'
    elif corruption == 'bool': data['schema_version'] = True
    elif corruption == 'float': data['num_shards'] = 1.0
    elif corruption == 'row': data['row']['outcome']['decision'] = 'sell'
    elif corruption == 'filename': file = file.rename(file.with_name('0' * 64 + '.json'))
    elif corruption == 'fields': data['generation']['source'] = 'forbidden'
    elif corruption == 'tokens': data['generation']['generated_token_ids'] = '11'
    elif corruption == 'provenance': data['generation']['provenance'] = []
    file.write_bytes(raw if raw is not None else canonical_json_bytes(data) + b'\n')
    with pytest.raises(ValueError):
        opening(path, bundle)


def _contender(path, bundle, connection):
    try:
        opening(path, bundle).close()
        connection.send('unexpected success')
    except RuntimeError as exc:
        connection.send(str(exc))
    finally:
        connection.close()


def test_persistent_lock_and_process_contention(tmp_path, bundle):
    path = tmp_path / 's'
    with opening(path, bundle):
        inode = (path / 'store.lock').stat().st_ino
        code = ('import os,fcntl,sys; f=os.open(sys.argv[1],os.O_RDWR); '
                'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
        process = subprocess.run([sys.executable, '-c', code, str(path / 'store.lock')], capture_output=True)
        assert process.returncode != 0 and b'BlockingIOError' in process.stderr
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        worker = context.Process(target=_contender, args=(path, bundle, child))
        worker.start()
        assert 'already held' in parent.recv()
        worker.join()
        assert worker.exitcode == 0
        parent.close()
        child.close()
        with pytest.raises(RuntimeError, match='already held'):
            opening(path, bundle)
    assert (path / 'store.lock').stat().st_ino == inode
    with opening(path, bundle):
        assert (path / 'store.lock').stat().st_ino == inode


def test_acquisition_inode_mismatch(tmp_path, bundle, monkeypatch):
    path = tmp_path / 's'
    opening(path, bundle).close()
    original = module.fcntl.flock
    def swapped(fd, flags):
        original(fd, flags)
        (path / 'store.lock').rename(path / 'old-lock')
        (path / 'store.lock').touch()
    monkeypatch.setattr(module.fcntl, 'flock', swapped)
    with pytest.raises(RuntimeError, match='inode changed'):
        opening(path, bundle)


@pytest.mark.parametrize('stage', ['file', 'link', 'unlink', 'directory', 'cleanup'])
def test_publication_faults(tmp_path, bundle, monkeypatch, stage):
    path = tmp_path / 's'
    key, historical = bundle[1].keys[:2]
    store = opening(path, bundle)
    store.record(historical, result())
    history = (path / 'records' / (sha256_json(historical.to_dict()) + '.json')).read_bytes()
    real_fsync, real_link, real_unlink = os.fsync, os.link, os.unlink
    original = OSError('original injected failure')
    def fsync(fd):
        if (stage in ('file', 'cleanup') and fd != store._records_fd) or (stage == 'directory' and fd == store._records_fd):
            raise original
        return real_fsync(fd)
    def link(*args, **kwargs):
        if stage == 'link': raise original
        return real_link(*args, **kwargs)
    def unlink(*args, **kwargs):
        if stage in ('unlink', 'cleanup'): raise OSError('unlink injected')
        return real_unlink(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'fsync', fsync)
        patch.setattr(module.os, 'link', link)
        patch.setattr(module.os, 'unlink', unlink)
        with pytest.raises(OSError) as caught:
            store.record(key, result())
        if stage == 'cleanup': assert caught.value is original
        if stage in ('unlink', 'directory', 'cleanup'):
            for operation in (lambda: store.rows, lambda: store.recorded_keys, lambda: store.pending_keys,
                              lambda: store.generation_for(historical), lambda: store.record(key, result()), store.__enter__):
                with pytest.raises(RuntimeError): operation()
        else:
            assert key in store.pending_keys
    store.close()
    assert (path / 'records' / (sha256_json(historical.to_dict()) + '.json')).read_bytes() == history
    destination = path / 'records' / (sha256_json(key.to_dict()) + '.json')
    assert destination.exists() == (stage in ('unlink', 'directory'))
    if stage in ('unlink', 'cleanup'):
        with pytest.raises(ValueError, match='staging'):
            opening(path, bundle)
    else:
        with opening(path, bundle) as reopened:
            assert (key in reopened.recorded_keys) == (stage == 'directory')
            if stage == 'directory':
                before = destination.read_bytes()
                with pytest.raises(ValueError): reopened.record(key, result())
                assert destination.read_bytes() == before
