"""Linux locked, plan-bound write-once baseline generations (no gate claims)."""
from __future__ import annotations

from dataclasses import fields
import fcntl
import json
import os
from pathlib import Path
import stat
import uuid

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from llm_bias.core.experiment_contract import RowKey
from llm_bias.core.inference.structured_output import StructuredGenerationResult
from llm_bias.core.stance_baseline_adapter import baseline_execution_row
from llm_bias.core.stance_baseline_plan import build_baseline_plan


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON')


def _utf8(value):
    if isinstance(value, str):
        value.encode('utf8', errors='strict')
    elif isinstance(value, dict):
        for key, item in value.items():
            _utf8(key)
            _utf8(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _utf8(item)


def _bytes(value):
    _utf8(value)
    return canonical_json_bytes(value)


def _parse(data):
    try:
        value = json.loads(data.decode('utf8', errors='strict'), object_pairs_hook=_pairs,
                           parse_constant=_constant)
        if _bytes(value) + b'\n' != data:
            raise ValueError('noncanonical JSON file')
        return value
    except (TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError('malformed JSON file') from exc


def _generation(value):
    try:
        names = {field.name for field in fields(StructuredGenerationResult)} - {'_provenance_bytes'} | {'provenance'}
        if type(value) is not dict or set(value) != names or type(value['generated_token_ids']) is not list:
            raise ValueError('invalid generation export fields or token array')
        kwargs = value.copy()
        provenance = kwargs.pop('provenance')
        kwargs['generated_token_ids'] = tuple(kwargs['generated_token_ids'])
        result = StructuredGenerationResult(**kwargs, _provenance_bytes=_bytes(provenance))
        if _bytes(result.to_dict()) != _bytes(value):
            raise ValueError('generation export mismatch')
        return result
    except (TypeError, AttributeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError('malformed generation export') from exc


def _open_regular(directory, name, flags=os.O_RDONLY):
    try:
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
    except OSError as exc:
        raise ValueError(f'invalid artifact: {name}') from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f'nonregular artifact: {name}')
    return fd


def _read(directory, name):
    with os.fdopen(_open_regular(directory, name), 'rb') as stream:
        return stream.read()


class BaselineShardStore:
    """An immediately opened shard; only close is permitted after uncertain publication."""

    def _check(self):
        if self._closed or self._poisoned:
            raise RuntimeError('baseline store is closed or poisoned; close and reopen')

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def close(self):
        if self._closed:
            return
        self._closed = True
        for name in ('_records_fd', '_lock_fd', '_root_fd', '_parent_fd'):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def _publish(self, directory, name, data):
        staging = '.pending-' + uuid.uuid4().hex + '.tmp'
        owned = False
        linked = False
        try:
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
            owned = True
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staging, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            linked = True
            os.unlink(staging, dir_fd=directory)
            owned = False
            os.fsync(directory)
        except BaseException:
            if linked:
                self._poisoned = True
            if owned:
                try:
                    os.unlink(staging, dir_fd=directory)
                except BaseException:
                    self._poisoned = True
            raise

    def _validate_entry(self, data, filename):
        try:
            entry = _parse(data)
            if type(entry) is not dict or set(entry) != {'schema_version', 'plan_hash', 'shard_index', 'num_shards', 'row', 'generation'}:
                raise ValueError('invalid record fields')
            for field, expected in (('schema_version', 1), ('shard_index', self._index), ('num_shards', self._count)):
                if type(entry[field]) is not int or entry[field] != expected:
                    raise ValueError('invalid record binding')
            if entry['plan_hash'] != self._plan.plan_hash:
                raise ValueError('foreign plan hash')
            key = RowKey.from_dict(entry['row']['key'])
            if key not in self._assigned or filename != sha256_json(key.to_dict()) + '.json':
                raise ValueError('foreign key or filename')
            generation = _generation(entry['generation'])
            row = baseline_execution_row(self._plan, self._inputs, key, generation)
            if _bytes(row.to_dict()) != _bytes(entry['row']):
                raise ValueError('record row mismatch')
            return key, row, generation
        except (KeyError, TypeError, AttributeError, UnicodeError, OverflowError, RecursionError) as exc:
            raise ValueError('malformed baseline record') from exc

    @property
    def recorded_keys(self):
        self._check()
        return tuple(sorted(self._entries))

    @property
    def pending_keys(self):
        self._check()
        return tuple(sorted(set(self._assigned) - self._entries.keys()))

    @property
    def rows(self):
        self._check()
        return tuple(self._validate_entry(self._entries[key], sha256_json(key.to_dict()) + '.json')[1]
                     for key in sorted(self._entries))

    def generation_for(self, key):
        self._check()
        return self._validate_entry(self._entries[key], sha256_json(key.to_dict()) + '.json')[2]

    def record(self, key, result):
        self._check()
        if key not in self._assigned:
            raise ValueError('key not assigned to shard')
        if key in self._entries:
            raise ValueError('baseline key already recorded')
        row = baseline_execution_row(self._plan, self._inputs, key, result)
        try:
            data = _bytes(dict(schema_version=1, plan_hash=self._plan.plan_hash,
                               shard_index=self._index, num_shards=self._count,
                               row=row.to_dict(), generation=result.to_dict())) + b'\n'
        except (TypeError, UnicodeError, OverflowError, RecursionError) as exc:
            raise ValueError('malformed generation serialization') from exc
        try:
            self._publish(self._records_fd, sha256_json(key.to_dict()) + '.json', data)
        except FileExistsError as exc:
            raise ValueError('baseline record destination already exists') from exc
        try:
            self._entries[key] = data
        except BaseException:
            self._poisoned = True
            raise
        return row


def open_baseline_store(directory, *, inputs, plan, shard_index, num_shards):
    """Validate the full B1 plan, acquire a persistent lock and validate all records."""
    rebuilt = build_baseline_plan(inputs, plan.identity)
    if rebuilt.to_json() != plan.to_json():
        raise ValueError('plan differs from full approved baseline plan')
    assigned = plan.keys_for_shard(shard_index, num_shards)
    expected = _bytes(dict(schema_version=1, plan=plan.to_dict(),
                           inputs_manifest_sha256=inputs.manifest_sha256,
                           shard_index=shard_index, num_shards=num_shards)) + b'\n'
    store = BaselineShardStore()
    store._closed = False
    store._poisoned = False
    store._inputs, store._plan = inputs, plan
    store._index, store._count, store._assigned = shard_index, num_shards, assigned
    store._entries = {}
    path = Path(directory)
    try:
        store._parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.mkdir(path.name, dir_fd=store._parent_fd)
            fresh = True
        except FileExistsError:
            fresh = False
        store._root_fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=store._parent_fd)
        store._lock_fd = _open_regular(store._root_fd, 'store.lock',
                                       os.O_RDWR | (os.O_CREAT | os.O_EXCL if fresh else 0))
        try:
            fcntl.flock(store._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('baseline store lock is already held') from exc
        inode = os.fstat(store._lock_fd)
        entry = os.stat('store.lock', dir_fd=store._root_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode) or (inode.st_dev, inode.st_ino) != (entry.st_dev, entry.st_ino):
            raise RuntimeError('baseline store lock inode changed during acquisition')
        if fresh:
            if set(os.listdir(store._root_fd)) != {'store.lock'}:
                raise ValueError('incomplete store initialization')
            os.mkdir('records', dir_fd=store._root_fd)
            store._records_fd = os.open('records', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                        dir_fd=store._root_fd)
            os.fsync(store._records_fd)
            store._publish(store._root_fd, 'registration.json', expected)
            os.fsync(store._root_fd)
            os.fsync(store._parent_fd)
        else:
            if set(os.listdir(store._root_fd)) != {'store.lock', 'registration.json', 'records'}:
                raise ValueError('incomplete or unexpected store layout')
            actual = _read(store._root_fd, 'registration.json')
            _parse(actual)
            if actual != expected:
                raise ValueError('registration differs from exact plan and shard binding')
            store._records_fd = os.open('records', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                        dir_fd=store._root_fd)
            for filename in os.listdir(store._records_fd):
                if filename.startswith('.pending-'):
                    raise ValueError('incomplete store: staging leftover')
                data = _read(store._records_fd, filename)
                key, _, _ = store._validate_entry(data, filename)
                if key in store._entries:
                    raise ValueError('duplicate record key')
                store._entries[key] = data
        return store
    except BaseException as exc:
        store.close()
        if isinstance(exc, OSError):
            raise ValueError('invalid or incomplete baseline store filesystem') from exc
        raise
