"""Group 3 functional fixtures, not answer quality or memory utility evidence."""
import json
import sqlite3
import pytest

from src.agents.document_agent import DocumentAgent,DocumentTools
from src.agents.document_orchestration import DocumentToolRegistry
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision
from tests.test_document_agent import Provider,run,SCOPE


class Encoder:
    identity='constant-memory-fixture'
    def __init__(self):self.calls=[];self.fail=False
    def encode(self,text):
        self.calls.append(text)
        if self.fail:raise RuntimeError('fixture encoder failed')
        return [1.,0.]


def build(path,encoder=None,documents=None,**kwargs):
    p=Provider();encoder=encoder or Encoder()
    service=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(False,'stage cache off')),
                           p,allowed_models={'fixture'})
    a=DocumentAgent(service,DocumentTools(documents or {'tuples':'Tuples are immutable.'}),path/'tasks.db',model='fixture',
                    orchestration=True,memory_encoder=encoder,stage_accounting=True,**kwargs)
    return a,p,encoder


def seed(a):
    task=a.create('Explain immutable tuples',standalone_document_task=True,**SCOPE)
    state=run(a,task,**SCOPE)
    assert state['status']=='AWAITING_APPROVAL'
    return a.decide(task,'approve',**SCOPE)


def test_scoped_memory_snapshot_reaches_actual_role_inputs(tmp_path):
    a,p,e=build(tmp_path);first=seed(a)
    task=a.create('Explain previous tuple answer',**SCOPE)
    snapshot=a.inspect(task,**SCOPE)
    assert snapshot['memory_status']=='snapshot'
    assert snapshot['task_memories'][0]['source_task_id']==first['id']
    assert len(e.calls)==2 and p.calls==3
    captured=[]
    original=p.__class__.__call__
    # Observe the real service boundary without replacing its accounting.
    def observed(body):
        captured.append(json.loads(body['messages'][1]['content']))
        return original(p,body)
    a.service.provider=observed
    done=run(a,task,**SCOPE)
    assert p.calls==6 and done['status']=='AWAITING_APPROVAL'
    assert all(row['untrusted_memories']==snapshot['task_memories'] for row in captured)
    assert [row['specialist'] for row in done['trace'] if row['stage']=='supervisor_route']==[
        'planner','source_researcher','source_researcher','answer_writer','reviewer']
    assert {x['stage'] for x in done['work']} >= {'memory_embedding','memory_search','planning','generation','review','tool'}


def test_memory_vectors_survive_restart_and_pending_tasks_are_not_memories(tmp_path):
    a,p,e=build(tmp_path);seed(a)
    task=a.create('tuple query',**SCOPE)
    expected=a.inspect(task,**SCOPE)['task_memories']
    assert len(e.calls)==2
    b,p2,e2=build(tmp_path)
    new=b.create('tuple query',**SCOPE)
    assert b.inspect(new,**SCOPE)['task_memories']==expected
    assert len(e2.calls)==1 and p2.calls==0


@pytest.mark.parametrize('scope',[{'tenant':'other','principal':'alice'},{'tenant':'a','principal':'other'}])
def test_memory_cannot_cross_tenant_or_principal(tmp_path,scope):
    a,p,e=build(tmp_path);seed(a)
    task=a.create('tuple query',**scope)
    assert a.inspect(task,**scope)['task_memories']==[] and e.calls==[]


@pytest.mark.parametrize('change',[{'documents':{'tuples':'Different docs'}},{'access_scope':'private'}, {'temperature':.1}])
def test_changed_source_scope_or_runtime_has_no_prior_memory(tmp_path,change):
    a,p,e=build(tmp_path);seed(a)
    b,p2,e2=build(tmp_path,**change)
    task=b.create('tuple query',**SCOPE)
    assert b.inspect(task,**SCOPE)['task_memories']==[] and p2.calls==0


def test_delete_and_reject_revoke_pending_memory_snapshot(tmp_path):
    a,p,e=build(tmp_path);source=seed(a)
    task=a.create('tuple query',**SCOPE)
    assert a.inspect(task,**SCOPE)['task_memories']
    a.delete(source['id'],**SCOPE)
    state=a.advance(task,**SCOPE)
    assert state['status']=='FAILED' and state['error']=='memory_scope_revoked' and p.calls==3
    fresh=a.create('tuple query',**SCOPE)
    assert a.inspect(fresh,**SCOPE)['task_memories']==[]
    with sqlite3.connect(tmp_path/'tasks.db') as db:
        assert db.execute('SELECT COUNT(*) FROM document_memory_vectors').fetchone()[0]==0


def test_memory_failure_is_observed_and_does_not_use_recent_fallback(tmp_path):
    a,p,e=build(tmp_path);seed(a);e.fail=True
    task=a.create('tuple query',**SCOPE)
    state=a.inspect(task,**SCOPE)
    assert state['task_memories']==[] and state['memory_status']=='unavailable'
    assert any(row['error']=='RuntimeError' for row in state['work'])
    assert run(a,task,**SCOPE)['status']=='AWAITING_APPROVAL' and p.calls==6


def test_unapproved_rejected_and_edited_tasks_do_not_seed_retrieval(tmp_path):
    a,p,e=build(tmp_path)
    pending=a.create('tuple',**SCOPE);run(a,pending,**SCOPE)
    assert a.inspect(a.create('tuple',**SCOPE),**SCOPE)['task_memories']==[]
    a.decide(pending,'modify',modified_answer='Human edit without workflow review',**SCOPE)
    task=a.create('tuple',**SCOPE)
    assert a.inspect(task,**SCOPE)['task_memories']==[]


def test_human_approval_after_failed_workflow_review_is_not_memory_seed(tmp_path):
    a,p,e=build(tmp_path,max_revisions=0)
    p.reject=True
    task=a.create('tuple',**SCOPE);run(a,task,**SCOPE)
    a.decide(task,'approve',**SCOPE)
    next_task=a.create('tuple query',**SCOPE)
    assert a.inspect(next_task,**SCOPE)['task_memories']==[]


def test_standalone_exact_hit_skips_memory_and_all_model_roles(tmp_path):
    from tests.test_before_agent_cache import build as cache_build,run as cache_run,approve
    encoder=Encoder()
    a,p=cache_build(tmp_path,memory_encoder=encoder,orchestration=True)
    approve(a,cache_run(a))
    hit=cache_run(a)
    assert hit['provider_calls']==0 and p.calls and len(p.calls)==3
    assert hit['memory_status']=='standalone_skip' and encoder.calls==[]
    assert [r['specialist'] for r in hit['trace'] if r['stage']=='supervisor_route']==['cache_gateway']
    approve(a,hit)
    assert len(a.list_task_memory(**SCOPE))==1  # cache copies add no source memory


@pytest.mark.parametrize('name,args',[('shell',{'command':'x'}),('read',{'document_id':1}),
    ('search',{'query':'x','extra':'x'}),('read',{'document_id':''}),('search',{'query':'x'*1001})])
def test_typed_registry_rejects_unsupported_calls_before_handler(name,args):
    calls=[]
    registry=DocumentToolRegistry(lambda x:calls.append(x),lambda x:calls.append(x))
    with pytest.raises(ValueError):registry.dispatch(name,args)
    assert calls==[]


def test_clear_memory_preserves_audit_but_retires_future_context(tmp_path):
    a,p,e=build(tmp_path);first=seed(a)
    assert a.list_task_memory(**SCOPE)[0]['source_task_id']==first['id']
    pending=a.create('tuple query',**SCOPE)
    assert a.clear_task_memory(**SCOPE)=={'context_retired':True,'audit_tasks_retained':True}
    assert a.list_task_memory(**SCOPE)==[]
    assert a.inspect(first['id'],**SCOPE)['status']=='COMPLETED'
    assert a.advance(pending,**SCOPE)['error']=='memory_scope_revoked'
    fresh=a.create('tuple query',**SCOPE)
    assert a.inspect(fresh,**SCOPE)['task_memories']==[] and p.calls==3


def test_demo_memory_api_requires_session_and_keeps_cache_hit_zero_call(tmp_path):
    import re
    from fastapi.testclient import TestClient
    from src.api.demo import create_demo_app
    client=TestClient(create_demo_app(tmp_path,whole_answer_cache=True,task_memory=True))
    assert client.get('/api/memory').status_code==401
    client.headers['x-demo-session']=re.search(r'name="demo-session" content="([^"]+)"',client.get('/').text).group(1)
    def ask():
        s=client.post('/api/tasks',json={'question':'How do I convert between tuples and lists?'}).json()
        for _ in range(20):
            if s['status']=='AWAITING_APPROVAL':return s
            response=client.post('/api/tasks/'+s['id']+'/advance');assert response.status_code==200,response.text
            s=response.json()
        raise AssertionError('unbounded HTTP fixture')
    first=ask()
    assert first['observed_attempts']==3
    assert client.post('/api/tasks/'+first['id']+'/decision',json={'decision':'approve'}).status_code==200
    assert len(client.get('/api/memory').json()['items'])==1
    hit=ask();assert hit['observed_attempts']==0 and hit['answer_cache_hit']
    assert client.delete('/api/memory').json()['audit_tasks_retained']
    assert client.get('/api/memory').json()['items']==[]
    assert client.get('/api/tasks/'+first['id']).json()['status']=='COMPLETED'
