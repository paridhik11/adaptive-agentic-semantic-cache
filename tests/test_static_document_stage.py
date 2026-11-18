"""R6 development tests, not the preregistered 44-task criterion run."""
import copy
import json
import pytest
from src.agents.document_agent import DocumentAgent,DocumentTools
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision

SCOPE={'tenant':'a','principal':'alice'}
class RecordingProvider(FixtureChatProvider):
    def __init__(self):self.requests=[]
    def __call__(self,body):
        self.requests.append(copy.deepcopy(body))
        r=super().__call__(body);instruction=body['messages'][0]['content']
        if 'Plan 1-8' in instruction:
            text=json.dumps({'steps':[{'id':'read','tool':'read','arguments':{'document_id':'tuples'},'depends_on':[]}]})
        elif 'Check correctness' in instruction:text=json.dumps({'approved':True,'feedback':'fixture'})
        else:text='Tuples are immutable [tuples].'
        r['choices'][0]['message']['content']=text
        return r

def setup(tmp_path):
    provider=RecordingProvider()
    service=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'fixture')),provider,allowed_models={'fixture','other'})
    return provider,service,DocumentTools({'tuples':'Tuples are immutable sequences.'})

def run(agent,question='Explain tuples',scope=None,**kw):
    scope=scope or SCOPE
    tid=agent.create(question,require_approval=False,static_document_stage=True,**scope,**kw)
    for _ in range(30):
        s=agent.advance(tid,**scope)
        if s['status'] in ('COMPLETED','AWAITING_APPROVAL','FAILED','REJECTED'):return s
    raise AssertionError('unbounded')

def draft(state):return next(t for t in state['trace'] if t['stage']=='DRAFTING')

def test_static_actual_prompt_excludes_only_orchestration_memory(tmp_path):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    first=run(a);second=run(a)
    assert first['provider_calls']==3 and second['provider_calls']==2
    assert draft(second)['cache_status']=='HIT'
    assert len(a.memories(**SCOPE))==2
    planning=[json.loads(r['messages'][1]['content']) for r in p.requests if 'Plan 1-8' in r['messages'][0]['content']]
    assert planning[1]['untrusted_memories']
    payload=json.loads(p.requests[1]['messages'][1]['content'])
    assert 'untrusted_memories' not in payload and payload['stage_contract']==a.STATIC_CONTRACT
    assert payload['relevant_memory']==[]

@pytest.mark.parametrize('change', ['question','tenant','principal','access','policy','knowledge','revision','source_text','source_id','model','temperature','tokens','instructions','contract','relevant_memory'])
def test_answer_relevant_change_prevents_reuse(tmp_path,change):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'first',model='fixture')
    run(a);scope=dict(SCOPE);question='Explain tuples';kw={};ctor={'model':'fixture'}
    if change=='question':question='Explain tuple references'
    elif change=='tenant':scope['tenant']='b'
    elif change=='principal':scope['principal']='bob'
    elif change=='access':ctor['access_scope']='restricted'
    elif change=='policy':ctor['policy_revision']='policy2'
    elif change=='knowledge':ctor['knowledge_base']='another-corpus'
    elif change=='revision':ctor['source_revision']='revision2'
    elif change=='source_text':tools=DocumentTools({'tuples':'A revised reference about tuples.'})
    elif change=='source_id':tools=DocumentTools({'tuples':'Tuples are immutable sequences.','another':'Another reference.'})
    elif change=='model':ctor['model']='other'
    elif change=='temperature':ctor['temperature']=.1
    elif change=='tokens':ctor['max_output_tokens']=512
    elif change=='instructions':ctor['draft_instruction']='Use only the document reference and cite it.'
    elif change=='relevant_memory':kw['relevant_memory']=[{'role':'user','content':'Explain that contained mutable objects may change.'}]
    if change=='contract':
        class NewContract(DocumentAgent):STATIC_CONTRACT='static-document-draft-test-v2'
        cls=NewContract
    else:cls=DocumentAgent
    b=cls(svc,tools,tmp_path/'second',**ctor)
    result=run(b,question,scope=scope,**kw)
    assert result['status']=='COMPLETED' and draft(result)['cache_status']=='MISS'


def test_relevant_memory_values_change_and_are_model_visible(tmp_path):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    first=[{'role':'user','content':'Give a short explanation.'}]
    second=[{'role':'user','content':'Include a mutable-element example.'}]
    assert draft(run(a,relevant_memory=first))['cache_status']=='MISS'
    assert draft(run(a,relevant_memory=second))['cache_status']=='MISS'
    assert draft(run(a,relevant_memory=second))['cache_status']=='HIT'
    drafts=[json.loads(r['messages'][1]['content']) for r in p.requests if 'stage_contract' in r['messages'][1]['content']]
    assert drafts[0]['relevant_memory']==first and drafts[1]['relevant_memory']==second


def test_history_reference_retains_memory_and_bypasses_static_reuse(tmp_path):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    run(a)
    result=run(a,'Explain tuples as you said earlier')
    assert draft(result)['cache_status']=='BYPASS' and draft(result)['context_bypass']
    actual=json.loads(p.requests[-2]['messages'][1]['content'])
    assert actual['untrusted_memories'] and 'stage_contract' not in actual


def test_static_human_decision_survives_restart_without_new_calls(tmp_path):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    tid=a.create('Explain tuples',static_document_stage=True,**SCOPE)
    for _ in range(8):s=a.advance(tid,**SCOPE)
    assert s['status']=='AWAITING_APPROVAL'
    before=len(p.requests)
    b=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    assert b.decide(tid,'modify',modified_answer='Human corrected answer',**SCOPE)['answer']=='Human corrected answer'
    assert b.advance(tid,**SCOPE)['status']=='COMPLETED' and len(p.requests)==before


def test_changed_runtime_cannot_resume_existing_task(tmp_path):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    tid=a.create('Explain tuples',static_document_stage=True,**SCOPE)
    b=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture',access_scope='changed')
    with pytest.raises(ValueError,match='scope/instructions/settings'):b.advance(tid,**SCOPE)
    assert b.memories(**SCOPE)==[]


def test_revision_feedback_and_prior_draft_are_not_dropped(tmp_path):
    class Retry(RecordingProvider):
        def __call__(self,body):
            response=super().__call__(body)
            if 'Check correctness' in body['messages'][0]['content']:
                response['choices'][0]['message']['content']=json.dumps({'approved':False,'feedback':'Add the missing qualification.'})
            return response
    p=Retry();svc=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'fixture')),p,allowed_models={'fixture'})
    a=DocumentAgent(svc,DocumentTools({'tuples':'Tuples are immutable.'}),tmp_path/'tasks',model='fixture')
    result=run(a)
    assert result['status']=='AWAITING_APPROVAL'
    drafts=[json.loads(r['messages'][1]['content']) for r in p.requests if 'stage_contract' in r['messages'][1]['content']]
    assert len(drafts)==2 and drafts[0]['draft'] is None
    assert drafts[1]['draft']=='Tuples are immutable [tuples].'
    assert drafts[1]['feedback']=='Add the missing qualification.'
    assert [t['cache_status'] for t in result['trace'] if t['stage']=='DRAFTING']==['MISS','MISS']


def test_static_budget_is_enforced(tmp_path):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture',max_calls=3)
    tid=a.create('Explain tuples',static_document_stage=True,**SCOPE)
    with a._db() as db:
        s=a._load(db,tid,**SCOPE);s['reserved_calls']=3;a._save(db,s)
    assert a.advance(tid,**SCOPE)['status']=='FAILED' and not p.requests

@pytest.mark.parametrize('decision',['approve','reject'])
def test_static_human_decisions_do_not_become_cached_approval(tmp_path,decision):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    for _ in range(2):
        tid=a.create('Explain tuples',static_document_stage=True,**SCOPE)
        for _ in range(8):state=a.advance(tid,**SCOPE)
        assert state['status']=='AWAITING_APPROVAL'
        state=a.decide(tid,decision,**SCOPE)
        assert state['status']==('COMPLETED' if decision=='approve' else 'REJECTED')


def test_old_unbound_task_memory_not_carried_into_new_access_scope(tmp_path):
    p,svc,tools=setup(tmp_path);a=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture')
    state=run(a)
    with a._db() as db:
        stored=a._load(db,state['id'],**SCOPE)
        stored.pop('runtime_binding');a._save(db,stored)
    # Original default runtime can still inspect old checkpoints.
    assert a.inspect(state['id'],**SCOPE)['status']=='COMPLETED'
    b=DocumentAgent(svc,tools,tmp_path/'tasks',model='fixture',access_scope='new-permissions')
    assert b.memories(**SCOPE)==[]
    with pytest.raises(ValueError,match='scope/instructions/settings'):b.inspect(state['id'],**SCOPE)
