"""Before-agent fixtures: call counts are engineering evidence, not accuracy."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3

import pytest

from src.agents.document_agent import DocumentAgent, DocumentTools
from src.agents.document_answer_cache import CompletedDocumentCache
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache, EligibilityDecision
from src.policy.semantic_runtime import (GuardedSemanticCache, KnowledgeScope, ReuseDecision,
                                         AnswerBoundApprovals, approval_binding)
from src.policy.sqlite_store import SQLiteResponseStore
from src.policy.work_ledger import WorkLedger

SCOPE={'tenant':'a','principal':'alice'}
QUESTION='Explain immutable tuples'
PARAPHRASE='Describe tuple immutability'
KNOWLEDGE=KnowledgeScope('docs','v1','public','fixture-v1')


class Encoder:
    identity='constant-fixture-encoder'
    def encode(self,text):return [1.,0.]


class Reject:
    identity='reject-by-default'
    def __call__(self,*args,**kwargs):return ReuseDecision(False,'not qualified')


class Provider(FixtureChatProvider):
    def __init__(self):self.calls=[];self.fail=None;self.reject=False
    def __call__(self,body):
        system=body['messages'][0]['content']
        stage='planning' if 'Plan 1-8' in system else 'review' if 'Check correctness' in system else 'generation'
        self.calls.append({'stage':stage,'body':body})
        if self.fail==stage:raise RuntimeError('fixture failure')
        response=super().__call__(body)
        text=json.dumps({'steps':[{'id':'read','tool':'read','arguments':{'document_id':'tuples'},'depends_on':[]}]}) if stage=='planning' else json.dumps({'approved':not self.reject,'feedback':'fixture review'}) if stage=='review' else 'Tuples are immutable [tuples].'
        response['choices'][0]['message']['content']=text
        return response


def build(path,*,mode='exact',approver=None,provider=None,knowledge=KNOWLEDGE,**agent_kwargs):
    path.mkdir(parents=True,exist_ok=True)
    exact=ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'fixture'),
        storage=SQLiteResponseStore(path/'answers.db',namespace='before-agent-v1'))
    outer=GuardedSemanticCache(exact,scope=knowledge,encoder=Encoder(),mode=mode,
        approver=approver or Reject(),persistent_candidates=mode in {'shadow','guarded'},local_approval_only=True)
    provider=provider or Provider()
    # Disable the inner stage cache to isolate whole-answer work avoidance.
    service=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(False,'fixture stage cache off')),
        provider,allowed_models={'fixture'})
    agent=DocumentAgent(service,DocumentTools({'tuples':'Tuples are immutable.'}),path/'tasks.db',model='fixture',
        answer_cache=CompletedDocumentCache(outer),stage_accounting=True,**agent_kwargs)
    return agent,provider


def run(agent,question=QUESTION,scope=None,**kwargs):
    scope=scope or SCOPE
    tid=agent.create(question,standalone_document_task=True,**scope,**kwargs)
    for _ in range(30):
        state=agent.advance(tid,**scope)
        if state['status'] in {'COMPLETED','AWAITING_APPROVAL','FAILED','REJECTED'}:return state
    raise AssertionError('unbounded fixture')


def approve(agent,state,**kwargs):
    return agent.decide(state['id'],'approve',**SCOPE,**kwargs)


def remote_events(state):
    return [e for e in state['work'] if e['stage'] in {'planning','generation','review'}]


def test_exact_hit_before_agent_creates_new_durable_approval_task(tmp_path):
    agent,p=build(tmp_path)
    seed=run(agent)
    assert len(p.calls)==3 and [e['stage'] for e in remote_events(seed)]==['planning','generation','review']
    assert seed['answer'] is None
    approve(agent,seed)
    agent,p=build(tmp_path)
    hit=run(agent)
    assert hit['status']=='AWAITING_APPROVAL' and hit['id']!=seed['id']
    assert hit['answer'] is None and hit['draft']==seed['draft']
    assert hit['provider_calls']==0 and p.calls==[] and remote_events(hit)==[]
    assert hit['trace'][0]['tier']=='EXACT' and hit['trace'][0]['evidence_reused']
    assert hit['reserved_calls']==0 and not any(e['stage']=='tool' for e in hit['work'])
    assert approve(agent,hit)['status']=='COMPLETED'


def test_semantic_hit_uses_actual_answer_binding_and_no_remote_stages(tmp_path):
    agent,p=build(tmp_path,mode='guarded')
    seed=run(agent)
    approved=approve(agent,seed)
    incoming={**approved,'question':PARAPHRASE}
    request=agent.answer_cache.request(agent,approved,'alice')
    variant=agent.answer_cache.request(agent,incoming,'alice')
    response=agent.answer_cache.response(agent,approved)
    binding=approval_binding(variant,request,response,tenant_id='a',scope=asdict(KNOWLEDGE))
    policy=AnswerBoundApprovals({'schema_version':'answer-bound-approvals-v1','provenance':'authored_fixture',
                               'approvals':[{**binding,'approved':True}]})
    # Approvals are part of the candidate namespace; admit the reviewed seed
    # under the final identified fixture policy, without running models again.
    agent,p=build(tmp_path,mode='guarded',approver=policy)
    assert agent.answer_cache.admit(request,response,'a',agent.answer_cache.epoch())
    hit=run(agent,PARAPHRASE)
    assert hit['trace'][0]['tier']=='SEMANTIC' and hit['matched_question']==QUESTION
    assert hit['status']=='AWAITING_APPROVAL' and p.calls==[] and remote_events(hit)==[]
    assert {'embedding','search','applicability'} <= {e['stage'] for e in hit['work']}


def test_no_answer_admitted_before_required_human_approval(tmp_path):
    agent,p=build(tmp_path)
    seed=run(agent)
    assert agent.answer_cache.cache.size==0
    repeated=run(agent)
    assert len(p.calls)==6 and repeated['provider_calls']==3


@pytest.mark.parametrize('decision',['modify','reject'])
def test_human_correction_revokes_reuse_without_auto_promoting_edits(tmp_path,decision):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    hit=run(agent)
    agent.decide(hit['id'],decision,modified_answer='Human edit',**SCOPE)
    assert agent.answer_cache.cache.size==0
    assert run(agent)['provider_calls']==3 and len(p.calls)==6


def test_rejected_workflow_cannot_be_seeded_by_human_approval(tmp_path):
    agent,p=build(tmp_path,max_revisions=0)
    p.reject=True
    seed=run(agent)
    assert not seed['workflow_approved']
    approve(agent,seed)
    assert agent.answer_cache.cache.size==0


def test_invalidation_during_wait_for_human_prevents_stale_admission(tmp_path):
    agent,p=build(tmp_path)
    seed=run(agent)
    agent.answer_cache.invalidate('a')
    approve(agent,seed)
    assert agent.answer_cache.cache.size==0
    assert run(agent)['provider_calls']==3


@pytest.mark.parametrize('question',['What did you say previous time?','Run the deployment command','What is the latest tuple feature?'])
def test_contextual_or_new_work_requests_bypass_whole_answer_reuse(tmp_path,question):
    agent,p=build(tmp_path)
    agent.service.cache._eligibility=lambda _:EligibilityDecision(True,'stage cache enabled fixture')
    approve(agent,run(agent,question))
    again=run(agent,question)
    assert again['provider_calls']==3 and len(p.calls)==6


@pytest.mark.parametrize('scope',[{'tenant':'b','principal':'alice'},{'tenant':'a','principal':'bob'}])
def test_principal_and_tenant_isolation(tmp_path,scope):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    assert run(agent,scope=scope)['provider_calls']==3


def test_explicit_memory_is_bound_and_automatic_suggestions_are_absent(tmp_path):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    changed=run(agent,relevant_memory=[{'role':'user','content':'Use an example.'}])
    assert changed['provider_calls']==3
    for call in p.calls:
        payload=json.loads(call['body']['messages'][1]['content'])
        assert payload['untrusted_memories']==[]


@pytest.mark.parametrize('stage',['planning','generation','review'])
def test_failures_record_actual_provider_attempts_and_never_seed(tmp_path,stage):
    agent,p=build(tmp_path)
    p.fail=stage
    failed=run(agent)
    assert failed['status']=='FAILED' and agent.answer_cache.cache.size==0
    assert len(remote_events(failed))==len(p.calls)
    assert remote_events(failed)[-1]['error']=='RuntimeError'


def test_remote_judge_attempt_falls_back_without_calling_provider(tmp_path):
    class Remote:
        identity='fixture-attempted-remote-judge'
        def __call__(self,request,*args,provider,**kwargs):
            provider(request)
            return ReuseDecision(True,'must not happen')
    agent,p=build(tmp_path,mode='guarded',approver=Remote())
    approve(agent,run(agent))
    repeated=run(agent,PARAPHRASE)
    assert repeated['provider_calls']==3 and len(p.calls)==6
    checks=[e for e in repeated['work'] if e['stage']=='applicability']
    assert len(checks)==1 and checks[0]['error']=='ValueError'


def test_auto_completion_and_cache_off(tmp_path):
    agent,p=build(tmp_path)
    assert run(agent,require_approval=False)['status']=='COMPLETED'
    assert run(agent,require_approval=False)['provider_calls']==0
    assert run(agent,cache=False)['provider_calls']==3


def test_delete_retires_cached_source_and_ledger_is_detached(tmp_path):
    agent,p=build(tmp_path)
    seed=approve(agent,run(agent))
    agent.delete(seed['id'],**SCOPE)
    assert run(agent)['provider_calls']==3
    ledger=WorkLedger()
    with ledger.measure('embedding'):pass
    snapshot=ledger.snapshot();snapshot[0]['stage']='changed'
    assert ledger.snapshot()[0]['stage']=='embedding'


def test_gateway_validates_accounting_before_provider_work(tmp_path):
    agent,p=build(tmp_path)
    body={'model':'fixture','temperature':0,'max_tokens':64,'messages':[{'role':'user','content':'q'}]}
    with pytest.raises(ValueError):agent.service.complete(body,'a',ledger=WorkLedger())
    with pytest.raises(ValueError):agent.service.complete(body,'a',ledger=WorkLedger(),remote_stage='embedding')
    assert p.calls==[]


def test_durable_rejection_epoch_survives_failed_cache_invalidation(tmp_path,monkeypatch):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    hit=run(agent)
    def fail(_):raise sqlite3.OperationalError('cache locked')
    monkeypatch.setattr(agent.answer_cache,'invalidate',fail)
    rejected=agent.decide(hit['id'],'reject',**SCOPE)
    assert rejected['status']=='REJECTED'
    agent,p=build(tmp_path)
    assert run(agent)['provider_calls']==3 and len(p.calls)==3


def test_pending_lookup_cannot_cross_task_epoch_revocation(tmp_path):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    pending=agent.create(QUESTION,standalone_document_task=True,**SCOPE)
    hit=run(agent)
    agent.decide(hit['id'],'reject',**SCOPE)
    state=agent.advance(pending,**SCOPE)
    assert state['status']=='PLANNING' and state['trace'][0]['reason']=='task_scope_revoked'


def test_hit_approval_never_refreshes_source_expiry(tmp_path):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    def rows():
        with sqlite3.connect(tmp_path/'answers.db') as db:
            return db.execute('SELECT key,source_token,expires FROM entries').fetchall()
    before=rows()
    approve(agent,run(agent))
    assert rows()==before and len(before)==1


def test_invalid_cached_evidence_executes_fresh_work(tmp_path):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    with sqlite3.connect(tmp_path/'answers.db') as db:
        key,payload=db.execute('SELECT key,payload FROM entries').fetchone()
        response=json.loads(payload)
        response['documentation']['tools'][0]['result']='changed evidence'
        db.execute('UPDATE entries SET payload=? WHERE key=?',(json.dumps(response),key))
    result=run(agent)
    assert result['provider_calls']==3 and result['trace'][0]['reason']=='invalid_cached_evidence'


@pytest.mark.parametrize('changed',[{'temperature':.1},{'draft_instruction':'New instruction'}, {'access_scope':'private-docs'}])
def test_changed_runtime_cannot_reuse_prior_answers(tmp_path,changed):
    agent,p=build(tmp_path)
    approve(agent,run(agent))
    agent,p=build(tmp_path,**changed)
    assert run(agent)['provider_calls']==3


def test_opt_in_workbench_http_repeat_has_zero_calls_after_restart(tmp_path):
    import re
    from fastapi.testclient import TestClient
    from src.api.demo import create_demo_app
    def client():
        c=TestClient(create_demo_app(tmp_path,whole_answer_cache=True))
        c.headers['x-demo-session']=re.search(r'name="demo-session" content="([^"]+)"',c.get('/').text).group(1)
        return c
    def ask(c):
        state=c.post('/api/tasks',json={'question':'How do I convert between tuples and lists?'}).json()
        for _ in range(20):
            if state['status']=='AWAITING_APPROVAL':return state
            response=c.post('/api/tasks/'+state['id']+'/advance')
            assert response.status_code==200,response.text
            state=response.json()
        raise AssertionError('unbounded HTTP task')
    c=client();seed=ask(c)
    assert seed['observed_attempts']==3
    assert c.post('/api/tasks/'+seed['id']+'/decision',json={'decision':'approve'}).status_code==200
    c=client();hit=ask(c)
    assert hit['answer_cache_hit'] and hit['observed_attempts']==0 and hit['answer'] is None
    assert hit['matched_question']==seed['question']
    assert c.get('/api/info').json()['cached_answers']==1
    assert c.get('/api/info').json()['current_process_attempts']==0


def test_live_before_agent_demo_is_explicitly_out_of_scope(tmp_path):
    from src.api.demo import create_demo_app
    with pytest.raises(ValueError,match='offline reference mode only'):
        create_demo_app(tmp_path,live=True,whole_answer_cache=True)


def test_answer_cache_failure_does_not_undo_human_decision(tmp_path,monkeypatch):
    agent,p=build(tmp_path)
    seed=run(agent)
    def broken(*args,**kwargs):raise sqlite3.OperationalError('storage unavailable')
    monkeypatch.setattr(agent.answer_cache.store,'admit',broken)
    completed=approve(agent,seed)
    assert completed['status']=='COMPLETED' and completed['trace'][-1]['stored'] is False
    assert agent.inspect(seed['id'],**SCOPE)['status']=='COMPLETED'


def test_provider_usage_is_only_attached_to_actual_calls(tmp_path):
    agent,p=build(tmp_path)
    seed=run(agent)
    assert all(e.get('usage') is not None for e in remote_events(seed))
    approve(agent,seed)
    hit=run(agent)
    assert remote_events(hit)==[] and not any('usage' in e for e in hit['work'])


def test_embedding_failure_retains_approved_exact_answer_and_reports_it(tmp_path,monkeypatch):
    agent,p=build(tmp_path,mode='guarded')
    def broken(_):raise RuntimeError('fixture encoder failure')
    monkeypatch.setattr(agent.answer_cache.cache.encoder,'encode',broken)
    completed=approve(agent,run(agent))
    assert completed['trace'][-1]['stored'] is True
    assert any(e['stage']=='embedding' and e['error']=='RuntimeError' for e in completed['work'])
    assert run(agent)['provider_calls']==0
