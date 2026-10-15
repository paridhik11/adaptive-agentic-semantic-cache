"""Persistence correctness only: fixture responses, no model quality claims."""
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from src.policy.response_cache import ScopedResponseCache
from src.policy.sqlite_store import SQLiteResponseStore
from tests.test_response_cache import request, response, Provider, stable, run


def cache(path, **kwargs):
    return ScopedResponseCache(eligibility=stable, storage=SQLiteResponseStore(path, namespace='test-v1'), **kwargs)


def test_restart_and_tenant_isolation(tmp_path):
    path = tmp_path / 'cache.sqlite'
    p = Provider()
    assert run(cache(path), request(), p).stored
    assert run(cache(path), request(), p).status == 'HIT'
    assert run(cache(path), request(), p, 'other').status == 'MISS'
    assert p.calls == 2
    # Verify another interpreter reads the committed entry.
    code = "from tests.test_sqlite_response_cache import *; import sys; assert run(cache(sys.argv[1]), request(), Provider()).status == 'HIT'"
    subprocess.run([sys.executable, '-c', code, str(path)], check=True)


def test_expiry_uses_wall_clock_and_reopen(tmp_path):
    now = [1000.0]
    path = tmp_path / 'cache.sqlite'
    def make():
        return ScopedResponseCache(eligibility=stable, ttl_seconds=10,
            storage=SQLiteResponseStore(path, namespace='test', clock=lambda: now[0]))
    assert run(make(), request(), Provider()).stored
    now[0] = 1009
    assert run(make(), request(), Provider()).status == 'HIT'
    now[0] = 1010
    assert make().size == 0
    assert run(make(), request(), Provider()).status == 'MISS'


def test_cross_instance_invalidation_blocks_inflight_admission(tmp_path):
    path = tmp_path / 'cache.sqlite'
    a, b = cache(path), cache(path)
    started, release = threading.Event(), threading.Event()
    def provider(body):
        started.set()
        assert release.wait(5)
        return response()
    with ThreadPoolExecutor() as pool:
        future = pool.submit(run, a, request(), provider)
        assert started.wait(5)
        assert b.invalidate(tenant_id='tenant-a') == 0
        release.set()
        assert not future.result().stored
    assert a.size == b.size == 0


def test_invalidation_is_tenant_scoped_and_visible(tmp_path):
    path = tmp_path / 'cache.sqlite'
    a, b = cache(path), cache(path)
    for tenant in ('tenant-a', 'tenant-b'):
        run(a, request(), Provider(), tenant)
    assert b.invalidate(tenant_id='tenant-a') == 1
    assert run(a, request(), Provider(), 'tenant-b').status == 'HIT'
    assert run(a, request(), Provider()).status == 'MISS'
    assert b.invalidate() == 2
    assert a.size == 0


def test_lru_and_byte_limits(tmp_path):
    c = cache(tmp_path / 'cache.sqlite', max_entries=2)
    p = Provider()
    run(c, request(), p)
    run(c, {**request(), 'seed': 1}, p)
    run(c, request(), p)  # touch first
    run(c, {**request(), 'seed': 2}, p)
    assert c.size == 2
    assert run(c, request(), p).status == 'HIT'
    assert run(c, {**request(), 'seed': 1}, p).status == 'MISS'
    small = cache(tmp_path / 'small.sqlite', max_bytes=400)
    assert run(small, request(), Provider(response('a' * 100))).stored
    assert run(small, {**request(), 'seed': 1}, Provider(response('a' * 100))).stored
    assert small.size == 1
    assert not run(small, {**request(), 'seed': 2}, Provider(response('界' * 400))).stored


def test_wrong_namespace_refuses_startup(tmp_path):
    path = tmp_path / 'cache.sqlite'
    cache(path)
    with pytest.raises(sqlite3.DatabaseError):
        SQLiteResponseStore(path, namespace='different-provider')


@pytest.mark.parametrize('payload', ['{broken', '{}', '{"model":"x","choices":[],"usage":NaN}'])
def test_bad_persisted_payload_bypasses(tmp_path, payload):
    path = tmp_path / 'cache.sqlite'
    c = cache(path)
    run(c, request(), Provider())
    with sqlite3.connect(path) as db:
        db.execute('UPDATE entries SET payload=?', (payload,))
    p = Provider()
    result = run(c, request(), p)
    assert (result.status, result.reason, p.calls) == ('BYPASS', 'cache_storage_unavailable', 1)


def test_storage_failure_fails_open_for_generation_but_not_invalidation(tmp_path):
    path = tmp_path / 'cache.sqlite'
    c = cache(path)
    path.write_bytes(b'not a database')
    p = Provider()
    result = run(c, request(), p)
    assert result.status == 'BYPASS' and p.calls == 1
    with pytest.raises(sqlite3.DatabaseError):
        c.invalidate()


def test_admission_failure_does_not_repeat_provider(tmp_path, monkeypatch):
    c = cache(tmp_path / 'cache.sqlite')
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('disk full')
    monkeypatch.setattr(c._storage, 'admit', fail)
    p = Provider()
    result = run(c, request(), p)
    assert result.reason == 'cache_storage_unavailable'
    assert p.calls == 1 and not result.stored


def test_provider_exception_not_retried(tmp_path):
    c = cache(tmp_path / 'cache.sqlite')
    calls = []
    def fail(body):
        calls.append(1)
        raise RuntimeError('provider failed')
    with pytest.raises(RuntimeError):
        run(c, request(), fail)
    assert len(calls) == 1 and c.size == 0
