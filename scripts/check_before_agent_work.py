"""Constructed whole-task comparison; never makes external model calls."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agents.document_agent import DocumentAgent,DocumentTools
from src.agents.document_answer_cache import CompletedDocumentCache
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision
from src.policy.semantic_runtime import GuardedSemanticCache,KnowledgeScope,AnswerBoundApprovals,approval_binding
from src.policy.sqlite_store import SQLiteResponseStore

QUESTION='Explain immutable tuples'
PARAPHRASE='Describe tuple immutability'
FRESH='Explain tuple packing'
KNOWLEDGE=KnowledgeScope('docs','v1','public','fixture-v1')
SCOPE={'tenant':'fixture-tenant','principal':'fixture-operator'}
STEP={'id':'read','tool':'read','arguments':{'document_id':'tuples'},'depends_on':[]}
ANSWER='Tuples are immutable [tuples].'
DOCUMENTS={'tuples':'Tuples are immutable.'}


class Encoder:
    identity='constant-encoder-fixture-not-natural-language-evidence'
    def encode(self,text):return [1.,0.]


class Provider(FixtureChatProvider):
    def __init__(self):self.calls=[]
    def __call__(self,body):
        system=body['messages'][0]['content']
        stage='planning' if 'Plan 1-8' in system else 'review' if 'Check correctness' in system else 'generation'
        self.calls.append(stage)
        response=super().__call__(body)
        text=json.dumps({'steps':[STEP]}) if stage=='planning' else json.dumps({'approved':True,'feedback':'fixture'}) if stage=='review' else ANSWER
        response['choices'][0]['message']['content']=text
        return response


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('refusing to overwrite evidence directory')
    results=[]
    with tempfile.TemporaryDirectory() as temporary:
        for mode in ('off','exact','guarded'):
            directory=Path(temporary)/mode;directory.mkdir()
            provider=Provider()
            def build(approver=None):
                exact=ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'fixture'),
                    storage=SQLiteResponseStore(directory/'answers.db',namespace='whole-task-fixture-v1'))
                cache=GuardedSemanticCache(exact,scope=KNOWLEDGE,encoder=Encoder(),mode=mode,
                    approver=approver,persistent_candidates=mode=='guarded',local_approval_only=True)
                service=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(False,'inner cache off')),
                    provider,allowed_models={'fixture'})
                return DocumentAgent(service,DocumentTools(DOCUMENTS),directory/'tasks.db',model='fixture',
                    answer_cache=CompletedDocumentCache(cache),stage_accounting=True)
            policy=None
            if mode=='guarded':
                # Construct and freeze a single authored approval before running
                # this arm. No model judgments or test-label fitting.
                from src.policy.semantic_runtime import ReuseDecision
                class Reject:
                    identity='construction-only-no-serving'
                    def __call__(self,*a,**k):return ReuseDecision(False,'construction')
                draft_agent=build(Reject())
                state={'question':QUESTION,'relevant_memory':[],'answer_namespace_epoch':0,
                       'max_calls':6,'max_revisions':1,'answer':ANSWER,
                       'tools':[{'step':STEP,'result':DOCUMENTS['tuples']}]}
                request=draft_agent.answer_cache.request(draft_agent,state,SCOPE['principal'])
                incoming=draft_agent.answer_cache.request(draft_agent,{**state,'question':PARAPHRASE},SCOPE['principal'])
                response=draft_agent.answer_cache.response(draft_agent,state)
                binding=approval_binding(incoming,request,response,tenant_id=SCOPE['tenant'],scope=asdict(KNOWLEDGE))
                policy=AnswerBoundApprovals({'schema_version':'answer-bound-approvals-v1','provenance':'authored_fixture',
                                            'approvals':[{**binding,'approved':True}]})
            agent=build(policy)
            tasks=[]
            for index,question in enumerate((QUESTION,QUESTION,PARAPHRASE,FRESH)):
                if index==1:agent=build(policy)  # restart after approved seed
                before=len(provider.calls)
                tid=agent.create(question,standalone_document_task=True,**SCOPE)
                for _ in range(30):
                    state=agent.advance(tid,**SCOPE)
                    if state['status'] in {'AWAITING_APPROVAL','FAILED','COMPLETED','REJECTED'}:break
                assert state['status']=='AWAITING_APPROVAL',state
                assert state['answer'] is None  # no inherited human approval
                state=agent.decide(tid,'approve',**SCOPE)  # scripted action, NOT human review
                observed=len(provider.calls)-before
                remote=[e for e in state['work'] if e['stage'] in {'planning','generation','review'}]
                assert observed==len(remote)==state['provider_calls']
                assert not any(e['error'] for e in state['work'])
                tasks.append({'question':question,'actual_fixture_provider_invocations':observed,'state':state})
            results.append({'mode':mode,'tasks':tasks,'actual_fixture_provider_invocations':len(provider.calls),
                            'stages':{stage:provider.calls.count(stage) for stage in ('planning','generation','review')}})
    assert [r['actual_fixture_provider_invocations'] for r in results]==[12,9,6]
    assert [t['actual_fixture_provider_invocations'] for t in results[2]['tasks']]==[3,0,0,3]
    root=Path(__file__).resolve().parents[1]
    sources=['src/agents/document_agent.py','src/agents/document_answer_cache.py','src/policy/work_ledger.py',
             'src/policy/semantic_runtime.py','src/policy/sqlite_store.py','src/api/service.py','scripts/check_before_agent_work.py']
    receipt={'evidence_type':'constructed offline workflow fixtures; scripted approvals are NOT human review',
             'not_measured':['independent natural-language quality','live model utility','real token savings','real cost savings','representative latency'],
             'restart_after_task':1,'inner_stage_cache':'disabled','arms':results,
             'timing_scope':'local fixture stage observations, not production latency',
             'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'work.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps([{'mode':r['mode'],'fixture_calls':r['actual_fixture_provider_invocations'],'stages':r['stages']} for r in results]))


if __name__=='__main__':main()
