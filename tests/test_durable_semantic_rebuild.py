"""Constructed functional fixtures only; no natural-language quality estimate."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import sqlite3
import threading

import pytest

from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache, _serialize
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, ReuseDecision
from src.policy.sqlite_store import SQLiteResponseStore

BODY = {'model': 'fixture-model', 'temperature': 0, 'max_tokens': 64,
        'messages': [{'role': 'user', 'content': 'Explain tuple immutability'}]}
VARIANT = {**BODY, 'messages': [{'role': 'user', 'content': 'Describe immutable tuples'}]}
SCOPE = KnowledgeScope('docs', 'v1', 'public', 'fixture-v1')


class Clock:
    value = 100.
    def __call__(self):
        return self.value


class Encoder:
    identity = 'constant-fixture-only'
    def encode(self, text):
        return [1., 0.]


class Approval:
    identity = 'fixture-approval-not-quality'
    def __call__(self, *args, **kwargs):
        return ReuseDecision(True, 'fixture only')


def build(path, clock, *, mode='guarded', approver=None, scope=SCOPE, **kwargs):
    store = SQLiteResponseStore(path, namespace='rebuild-test', clock=clock)
    exact = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, 'fixture'),
                                storage=store, ttl_seconds=10, max_entries=kwargs.pop('source_entries', 20))
    cache = GuardedSemanticCache(exact, scope=scope, encoder=Encoder(), mode=mode,
                                approver=approver or Approval(), persistent_candidates=True, **kwargs)
    return cache, GatewayService(cache, FixtureChatProvider(), allowed_models={'fixture-model'})


def test_restart_reuses_without_alias_or_ttl_extension(tmp_path):
    clock, path = Clock(), tmp_path/'cache.db'
    _, first = build(path, clock)
    seed = first.complete(BODY, 'a')
    cache, second = build(path, clock)
    assert second.complete(BODY, 'a').tier == 'EXACT'
    clock.value = 109
    hit = second.complete(VARIANT, 'a')
    assert hit.tier == 'SEMANTIC' and hit.provider_calls == 0
    assert hit.response == seed.response and cache.size == 1
    clock.value = 111
    _, third = build(path, clock)
    assert third.complete(VARIANT, 'a').status == 'MISS'


@pytest.mark.parametrize('mutation', ['invalidate', 'evict', 'replace'])
def test_source_change_during_approval_revokes(tmp_path, mutation):
    clock, path = Clock(), tmp_path/'cache.db'
    class Mutate(Approval):
        def __call__(self, *args, **kwargs):
            store = SQLiteResponseStore(path, namespace='rebuild-test', clock=clock)
            if mutation == 'invalidate':
                store.invalidate('a')
            elif mutation == 'evict':
                store.admit('other', 'a', '{}', store.invalidation_epoch, ttl=10, max_entries=1, max_bytes=100000)
            else:
                # Same bytes, same expiry: only the source incarnation changes.
                key = ScopedResponseCache.request_key(BODY, 'a', asdict(SCOPE))
                store.admit(key, 'a', _serialize(FixtureChatProvider()(BODY)), store.invalidation_epoch,
                            ttl=10, max_entries=20, max_bytes=100000)
            return super().__call__(*args, **kwargs)
    _, service = build(path, clock, approver=Mutate())
    service.complete(BODY, 'a')
    result = service.complete(VARIANT, 'a')
    assert result.status == 'MISS'
    assert result.diagnosis['stage'] == 'candidate_revoked'


@pytest.mark.parametrize('field', ['knowledge_base', 'revision', 'access_scope', 'policy_revision'])
def test_scope_change_after_restart_cannot_recover(tmp_path, field):
    clock, path = Clock(), tmp_path/'cache.db'
    _, first = build(path, clock)
    first.complete(BODY, 'a')
    _, second = build(path, clock, scope=replace(SCOPE, **{field: 'changed'}))
    assert second.complete(VARIANT, 'a').diagnosis['candidate'] is None


@pytest.mark.parametrize('change', ['tenant', 'options', 'history', 'approver', 'threshold'])
def test_namespace_isolation(tmp_path, change):
    clock, path = Clock(), tmp_path/'cache.db'
    _, first = build(path, clock)
    first.complete(BODY, 'a')
    kwargs = {}
    if change == 'approver':
        approver = Approval()
        approver.identity = 'different'
        kwargs['approver'] = approver
    if change == 'threshold':
        kwargs['threshold'] = .9
    _, second = build(path, clock, **kwargs)
    body = VARIANT
    if change == 'options':
        body = {**VARIANT, 'max_tokens': 63}
    if change == 'history':
        body = {**VARIANT, 'messages': [{'role': 'system', 'content': 'changed'}, *VARIANT['messages']]}
    result = second.complete(body, 'b' if change == 'tenant' else 'a')
    assert result.diagnosis['candidate'] is None


def test_expiry_during_approval_and_shadow_fall_back(tmp_path):
    clock, path = Clock(), tmp_path/'cache.db'
    class Expire(Approval):
        def __call__(self, *args, **kwargs):
            clock.value += 11
            return super().__call__(*args, **kwargs)
    _, first = build(path, clock, approver=Expire())
    first.complete(BODY, 'a')
    assert first.complete(VARIANT, 'a').diagnosis['stage'] == 'candidate_revoked'
    other = tmp_path/'shadow.db'
    _, shadow = build(other, clock, mode='shadow')
    shadow.complete(BODY, 'a')
    _, restarted = build(other, clock, mode='shadow')
    result = restarted.complete(VARIANT, 'a')
    assert result.status == 'MISS' and result.diagnosis['stage'] == 'shadow_candidate'


def test_legacy_schema_keeps_exact_but_does_not_promote(tmp_path):
    path = tmp_path/'legacy.db'
    key = ScopedResponseCache.request_key(BODY, 'a', asdict(SCOPE))
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE entries (key TEXT PRIMARY KEY,tenant TEXT,payload TEXT,expires REAL,bytes INTEGER,touched INTEGER)')
        db.execute('INSERT INTO entries VALUES (?,?,?,?,?,?)', (key,'a',_serialize(FixtureChatProvider()(BODY)),110,100,1))
    cache, service = build(path, Clock())
    assert service.complete(BODY, 'a').tier == 'EXACT'
    assert service.complete(VARIANT, 'a').diagnosis['candidate'] is None


def test_incomplete_answer_and_durable_bounds(tmp_path):
    clock = Clock()
    cache, service = build(tmp_path/'bad.db', clock)
    def incomplete(body):
        response = FixtureChatProvider()(body)
        response['choices'][0]['finish_reason'] = 'length'
        return response
    service.provider = incomplete
    assert service.complete(BODY, 'a').stored is False
    assert cache.semantic_size == 0
    cache, service = build(tmp_path/'bound.db', clock, mode='shadow', max_entries=1)
    service.complete(BODY, 'a')
    service.complete(VARIANT, 'a')
    with sqlite3.connect(tmp_path/'bound.db') as db:
        assert db.execute('SELECT COUNT(*) FROM semantic_candidates').fetchone()[0] == 1
    tiny, service = build(tmp_path/'tiny.db', clock, max_bytes=1)
    service.complete(BODY, 'a')
    assert tiny.semantic_size == 0 and tiny.size == 1


def test_overlapping_paraphrases_keep_candidate_identity(tmp_path):
    barrier = threading.Barrier(2)
    class Concurrent(Approval):
        def __call__(self, *args, **kwargs):
            barrier.wait(timeout=5)
            return super().__call__(*args, **kwargs)
    cache, service = build(tmp_path/'concurrent.db', Clock(), approver=Concurrent())
    service.complete(BODY, 'a')
    variants = [VARIANT]
    first = int(ScopedResponseCache.request_key(VARIANT,'a',asdict(SCOPE))[:8],16)%64
    for i in range(100):
        body = {**BODY,'messages':[{'role':'user','content':f'paraphrase {i}'}]}
        if int(ScopedResponseCache.request_key(body,'a',asdict(SCOPE))[:8],16)%64 != first:
            variants.append(body)
            break
    assert len(variants) == 2
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda b: service.complete(b,'a'),variants))
    assert [r.tier for r in results] == ['SEMANTIC','SEMANTIC']


def test_storage_failure_and_ineligible_requests_generate(tmp_path):
    cache, service = build(tmp_path/'fallback.db', Clock())
    service.complete(BODY,'a')
    cache.exact._eligibility = lambda _: EligibilityDecision(False,'not-static')
    assert service.complete(VARIANT,'a').status == 'BYPASS'
    cache.exact._eligibility = lambda _: EligibilityDecision(True,'fixture')
    cache.exact._storage.namespace = 'broken'
    assert service.complete(VARIANT,'a').reason == 'cache_storage_unavailable'


def test_candidate_record_removed_during_approval_is_revoked(tmp_path):
    path = tmp_path/'record.db'
    class Remove(Approval):
        def __call__(self, *args, **kwargs):
            with sqlite3.connect(path) as db:
                db.execute('DELETE FROM semantic_candidates')
            return super().__call__(*args, **kwargs)
    _, service = build(path, Clock(), approver=Remove())
    service.complete(BODY, 'a')
    assert service.complete(VARIANT, 'a').diagnosis['stage'] == 'candidate_revoked'


def test_gateway_factory_persists_shadow_candidates(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from src.api.gateway import app_from_env
    prompt = tmp_path/'prompt.txt'
    prompt.write_text('Use only this fixed documentation: tuples are immutable.')
    env = {'GATEWAY_TENANTS_JSON':json.dumps({'test-rebuild-credential':'a'}),
           'GATEWAY_CACHE_POLICY':'pinned_context', 'GATEWAY_DOCUMENT_PROMPT_FILE':str(prompt),
           'GATEWAY_CACHE_MODE':'shadow', 'GATEWAY_PERSIST_SEMANTIC':'true',
           'GATEWAY_CACHE_DB':str(tmp_path/'gateway.db'), 'GATEWAY_CACHE_VERSION':'rebuild-fixture-v1',
           'GATEWAY_KNOWLEDGE_BASE':'docs', 'GATEWAY_KNOWLEDGE_REVISION':'v1',
           'GATEWAY_ACCESS_SCOPE':'public', 'GATEWAY_SCOPE_POLICY_VERSION':'v1'}
    for key,value in env.items():
        monkeypatch.setenv(key,value)
    body = {**BODY,'messages':[{'role':'system','content':prompt.read_text()},*BODY['messages']]}
    headers = {'Authorization':'Bearer test-rebuild-credential'}
    with TestClient(app_from_env()) as client:
        assert client.post('/v1/chat/completions',json=body,headers=headers).headers['x-cache'] == 'MISS'
    with TestClient(app_from_env()) as client:
        assert client.post('/v1/chat/completions',json=body,headers=headers).headers['x-cache'] == 'HIT'
    with sqlite3.connect(tmp_path/'gateway.db') as db:
        assert db.execute('SELECT COUNT(*) FROM semantic_candidates').fetchone()[0] == 1


@pytest.mark.parametrize('mode,db,value', [('exact',True,'true'),('shadow',False,'true'),('shadow',True,'yes')])
def test_invalid_gateway_persistence_configuration(monkeypatch,tmp_path,mode,db,value):
    from src.api.gateway import app_from_env
    monkeypatch.setenv('GATEWAY_TENANTS_JSON',json.dumps({'test-rebuild-credential':'a'}))
    monkeypatch.setenv('GATEWAY_CACHE_MODE',mode)
    monkeypatch.setenv('GATEWAY_PERSIST_SEMANTIC',value)
    if db:
        monkeypatch.setenv('GATEWAY_CACHE_DB',str(tmp_path/'invalid.db'))
    else:
        monkeypatch.delenv('GATEWAY_CACHE_DB',raising=False)
    with pytest.raises(ValueError,match='persistent semantic|GATEWAY_PERSIST_SEMANTIC'):
        app_from_env()
