import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from src.api import demo
from src.api.providers import ProviderError
from src.benchmark.free_pilot import PilotJournal

QUESTION='How do I convert between tuples and lists?'


def client(path):
    c=TestClient(demo.create_demo_app(path))
    shell=c.get('/');assert shell.status_code==200
    token=re.search(r'name="demo-session" content="([^"]+)"',shell.text).group(1)
    c.headers['x-demo-session']=token
    return c


def task(c,question=QUESTION,cache=True):
    response=c.post('/api/tasks',json={'question':question,'cache':cache})
    assert response.status_code==200,response.text
    state=response.json()
    for _ in range(15):
        if state['status'] in demo.TERMINAL:return state
        response=c.post('/api/tasks/'+state['id']+'/advance');assert response.status_code==200
        state=response.json()
    raise AssertionError('Did not reach a bounded terminal state')


def test_real_agent_approval_repeat_exact_reuse_and_sources(tmp_path):
    c=client(tmp_path);first=task(c)
    assert first['status']=='AWAITING_APPROVAL' and first['answer'] is None
    assert first['observed_attempts']==3 and not first['draft_cache_hit']
    assert len(first['source_ids'])==1 and any(t['tool']=='search' for t in first['trace'] if t['stage']=='tool')
    decided=c.post('/api/tasks/'+first['id']+'/decision',json={'decision':'approve'}).json()
    assert decided['status']=='COMPLETED' and decided['answer']==first['draft']
    repeated=task(c)
    assert repeated['draft']==first['draft'] and repeated['draft_cache_hit']
    assert repeated['observed_attempts']==2
    info=c.get('/api/info').json()
    assert info['current_process_attempts']==5 and info['cached_drafts']==1
    assert info['mode']=='reference_replay' and len(info['reference']['documents'])==3


def test_cache_off_bypasses_even_after_warm_cache(tmp_path):
    c=client(tmp_path);task(c);state=task(c,cache=False)
    assert state['observed_attempts']==3 and not state['draft_cache_hit']
    assert next(t for t in state['trace'] if t['stage']=='DRAFTING')['cache_status']=='BYPASS'


def test_cache_and_history_survive_restart_without_auto_inference(tmp_path):
    first=client(tmp_path);s=task(first)
    second=client(tmp_path);info=second.get('/api/info').json()
    assert info['current_process_attempts']==0 and info['history'][0]['id']==s['id']
    repeated=task(second)
    assert repeated['draft_cache_hit'] and repeated['observed_attempts']==2


def test_modify_reject_and_invalid_decision_do_not_call_provider(tmp_path):
    c=client(tmp_path);first=task(c)
    changed=c.post('/api/tasks/'+first['id']+'/decision',json={'decision':'modify','answer':'My edited answer'}).json()
    assert changed['answer']=='My edited answer' and changed['trace'][-1]['decision']=='modify'
    assert c.post('/api/tasks/'+first['id']+'/decision',json={'decision':'approve'}).status_code==400
    second=task(c);rejected=c.post('/api/tasks/'+second['id']+'/decision',json={'decision':'reject'}).json()
    assert rejected['status']=='REJECTED' and rejected['answer'] is None
    assert c.get('/api/info').json()['current_process_attempts']==5


def test_local_session_origin_host_and_asset_boundaries(tmp_path):
    c=client(tmp_path)
    assert c.get('/api/info',headers={'x-demo-session':''}).status_code==401
    assert c.post('/api/tasks',headers={'origin':'https://other.example'},json={'question':QUESTION}).status_code==403
    assert c.get('/',headers={'host':'attacker.example'}).status_code==400
    assert c.get('/assets/workbench.js').status_code==200
    assert c.get('/assets/demo.py').status_code==404
    assert "script-src 'self'" in c.get('/').headers['content-security-policy']


def test_request_validation_and_single_active_task(tmp_path):
    c=client(tmp_path)
    assert c.post('/api/tasks',json={'question':QUESTION,'cache':'false'}).status_code==400
    assert c.post('/api/tasks',json={'question':'x'*4001}).status_code==400
    assert c.post('/api/tasks',content='x'*20001,headers={'content-type':'application/json'}).status_code==400
    first=c.post('/api/tasks',json={'question':QUESTION}).json()
    assert c.post('/api/tasks',json={'question':QUESTION}).status_code==400
    assert c.get('/api/tasks/'+first['id']).json()['status']=='PLANNING'
    assert c.get('/api/tasks/missing').status_code==404


def test_read_source_unavailable_shows_uncertainty(tmp_path):
    c=client(tmp_path);state=task(c,question='Zyxwv qrstuv?')
    assert not state['source_ids'] and 'do not have read source evidence' in state['draft']
    assert state['status']=='AWAITING_APPROVAL'


def test_provider_error_preserves_spent_work_without_automatic_retry(tmp_path,monkeypatch):
    original=demo.ReferenceDemoProvider.__call__
    def fail_review(self,body):
        if 'Check correctness' in body['messages'][0]['content']:
            self.attempts+=1;raise ProviderError('upstream_status',502)
        return original(self,body)
    monkeypatch.setattr(demo.ReferenceDemoProvider,'__call__',fail_review)
    c=client(tmp_path);state=task(c)
    assert state['status']=='FAILED' and state['draft']
    assert state['observed_attempts']==3 and state['provider_error']['code']=='upstream_status'
    again=c.post('/api/tasks/'+state['id']+'/advance').json()
    assert again['observed_attempts']==3


def test_live_requires_existing_journal_and_does_not_read_key_in_replay(tmp_path,monkeypatch):
    import os
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    c=client(tmp_path/'replay');assert c.get('/api/info').json()['mode']=='reference_replay'
    with pytest.raises(ValueError,match='environment'):demo.create_demo_app(tmp_path/'live',live=True,journal_path=tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def test_live_daily_budget_and_run_allowance_cannot_reset(tmp_path):
    path=tmp_path/'journal';now=demo.time.time()
    j=PilotJournal(path,run_id='product-demo-v1',fingerprint='fixed',max_calls=12,clock=lambda:now)
    with sqlite3.connect(path) as db:
        for i in range(10):db.execute('INSERT INTO pilot_calls VALUES (?,?,?,?,?,?)',(j.run_id,i,'hash',now-i*61,'complete','{}'))
    with pytest.raises(ValueError,match='allowance'):demo._live_capacity(j)
    reopened=PilotJournal(path,run_id='product-demo-v1',fingerprint='fixed',max_calls=12)
    assert sum(reopened.status().values())==10

