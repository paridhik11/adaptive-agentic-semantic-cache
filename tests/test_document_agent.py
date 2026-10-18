import json
import pytest
from src.agents.document_agent import DocumentAgent, DocumentTools, validate_plan
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache, EligibilityDecision

PLAN={'steps':[{'id':'search','tool':'search','arguments':{'query':'tuple'},'depends_on':[]},{'id':'read','tool':'read','arguments':{'document_id':'tuples'},'depends_on':['search']}]}
class Provider(FixtureChatProvider):
    def __init__(self, reject=False, bad=False):
        self.calls=0;self.reject=reject;self.bad=bad
    def __call__(self,body):
        self.calls+=1
        r=super().__call__(body)
        system=body['messages'][0]['content']
        text=json.dumps(PLAN) if 'Plan 1-8' in system else json.dumps({'approved':not self.reject,'feedback':'check evidence'}) if 'Check correctness' in system else 'Tuples are immutable [tuples].'
        if self.bad:text='invalid'
        r['choices'][0]['message']['content']=text
        return r

def build(tmp_path, provider=None, documents=None, **kw):
    p=provider or Provider()
    service=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'test')),p,allowed_models={'fixture'})
    tools=DocumentTools(documents or {'tuples':'Python tuples are immutable sequences.'})
    return DocumentAgent(service,tools,tmp_path/'tasks.sqlite',model='fixture',**kw),p

def run(agent,tid,**scope):
    for _ in range(30):
        state=agent.advance(tid,**scope)
        if state['status'] in ('COMPLETED','AWAITING_APPROVAL','FAILED','REJECTED'):return state
    raise AssertionError('unbounded workflow')

SCOPE={'tenant':'a','principal':'alice'}

def test_tools_restart_approval_memory_and_delete(tmp_path):
    a,p=build(tmp_path)
    tid=a.create('Explain tuple',**SCOPE)
    s=run(a,tid,**SCOPE)
    assert s['status']=='AWAITING_APPROVAL'
    assert len(s['tools'])==2 and s['provider_calls']==3 and p.calls==3
    b,_=build(tmp_path)
    done=b.decide(tid,'approve',**SCOPE)
    assert done['answer']=='Tuples are immutable [tuples].'
    assert b.advance(tid,**SCOPE)==done
    assert len(b.memories(**SCOPE))==1
    b.delete(tid,**SCOPE)
    assert b.memories(**SCOPE)==[]
    with pytest.raises(KeyError):b.inspect(tid,**SCOPE)

@pytest.mark.parametrize('decision',['modify','reject'])
def test_human_decisions(tmp_path,decision):
    a,p=build(tmp_path);tid=a.create('tuple',**SCOPE);run(a,tid,**SCOPE)
    s=a.decide(tid,decision,modified_answer='Corrected answer',**SCOPE)
    assert s['status']==('REJECTED' if decision=='reject' else 'COMPLETED')
    assert p.calls==3
    with pytest.raises(ValueError):a.decide(tid,'approve',**SCOPE)

@pytest.mark.parametrize('scope',[{'tenant':'b','principal':'alice'},{'tenant':'a','principal':'bob'}])
def test_scope(tmp_path,scope):
    a,_=build(tmp_path);tid=a.create('tuple',**SCOPE)
    with pytest.raises(KeyError):a.advance(tid,**scope)
    assert a.memories(**scope)==[]

def test_revision_isolation(tmp_path):
    a,_=build(tmp_path);tid=a.create('tuple',**SCOPE)
    b,_=build(tmp_path,documents={'tuples':'Changed document'})
    with pytest.raises(ValueError):b.advance(tid,**SCOPE)
    assert b.memories(**SCOPE)==[]

def test_review_retry_and_budget(tmp_path):
    a,p=build(tmp_path,Provider(reject=True));tid=a.create('tuple',**SCOPE)
    s=run(a,tid,**SCOPE)
    assert s['status']=='AWAITING_APPROVAL' and s['revisions']==1 and p.calls==5
    a,p=build(tmp_path,Provider(reject=True),max_calls=3)
    tid=a.create('tuple',**SCOPE);s=run(a,tid,**SCOPE)
    assert s['status']=='FAILED' and s['error']=='budget_exhausted' and p.calls==3

def test_invalid_model_plan(tmp_path):
    a,p=build(tmp_path,Provider(bad=True));tid=a.create('tuple',**SCOPE)
    assert run(a,tid,**SCOPE)['status']=='FAILED' and p.calls==1

@pytest.mark.parametrize('plan',[{}, {'steps':[]},{'steps':[{'id':'x','tool':'shell','arguments':{},'depends_on':[]}]},{'steps':[{'id':'x','tool':'read','arguments':{'document_id':'x'},'depends_on':['future']}]}])
def test_plan_validation(plan):
    with pytest.raises(ValueError):validate_plan(plan)

def test_tools_no_paths_or_network():
    tools=DocumentTools({'x':'reference'})
    with pytest.raises(KeyError):tools.dispatch('read',{'document_id':'/etc/passwd'})
    with pytest.raises(ValueError):tools.dispatch('http',{'url':'https://example.org'})

def test_auto_complete_and_invalidate(tmp_path):
    a,_=build(tmp_path);tid=a.create('tuple',require_approval=False,cache=False,**SCOPE)
    assert run(a,tid,**SCOPE)['status']=='COMPLETED'
    a.invalidate(**SCOPE)
    assert a.memories(**SCOPE)==[]

def test_missing_document_fails_before_draft(tmp_path):
    a,p=build(tmp_path,documents={'other':'unrelated'})
    tid=a.create('tuple',**SCOPE)
    s=run(a,tid,**SCOPE)
    assert s['status']=='FAILED' and s['error']=='tool_failed' and p.calls==1

def test_concurrent_runners_do_not_duplicate_tools(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    a,p=build(tmp_path);tid=a.create('tuple',**SCOPE)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _:run(a,tid,**SCOPE),range(4)))
    s=a.inspect(tid,**SCOPE)
    assert s['status']=='AWAITING_APPROVAL' and len(s['tools'])==2 and p.calls==3
    assert s['reserved_calls']<=6

def test_reservations_survive_restart(tmp_path):
    a,p=build(tmp_path,max_calls=3);tid=a.create('tuple',**SCOPE)
    with a._db() as db:
        s=a._load(db,tid,**SCOPE);s['reserved_calls']=3;a._save(db,s)
    b,p=build(tmp_path)
    s=b.advance(tid,**SCOPE)
    assert s['status']=='FAILED' and s['error']=='budget_exhausted' and p.calls==0

def test_repeated_task_draft_uses_existing_gateway_cache(tmp_path):
    a,p=build(tmp_path)
    first=a.create('tuple',**SCOPE);second=a.create('tuple',**SCOPE)
    s1=run(a,first,**SCOPE);s2=run(a,second,**SCOPE)
    assert s1['provider_calls']==3 and s2['provider_calls']==2 and p.calls==5
    assert next(t for t in s2['trace'] if t['stage']=='DRAFTING')['cache_status']=='HIT'

def test_provider_failure_unknown_accounting(tmp_path):
    class Broken(Provider):
        def __call__(self,body):raise RuntimeError('unavailable')
    a,_=build(tmp_path,Broken());tid=a.create('tuple',**SCOPE)
    s=run(a,tid,**SCOPE)
    assert s['status']=='FAILED' and s['provider_calls'] is None and s['reserved_calls']==1

def test_missing_read_cannot_auto_complete(tmp_path):
    class SearchOnly(Provider):
        def __call__(self,body):
            result=super().__call__(body)
            if 'Plan 1-8' in body['messages'][0]['content']:
                result['choices'][0]['message']['content']=json.dumps({'steps':PLAN['steps'][:1]})
            return result
    a,_=build(tmp_path,SearchOnly());tid=a.create('tuple',require_approval=False,**SCOPE)
    assert run(a,tid,**SCOPE)['status']=='AWAITING_APPROVAL'
