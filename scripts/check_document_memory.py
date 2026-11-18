"""Group 3 fixture: persistent context retrieval and role accounting, no live calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.check_before_agent_work import Provider,DOCUMENTS,KNOWLEDGE,SCOPE,QUESTION
from src.agents.document_agent import DocumentAgent,DocumentTools
from src.agents.document_answer_cache import CompletedDocumentCache
from src.api.service import GatewayService
from src.policy.encoders import LexicalEncoder
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision
from src.policy.semantic_runtime import GuardedSemanticCache
from src.policy.sqlite_store import SQLiteResponseStore


class CountEncoder(LexicalEncoder):
    def __init__(self):self.calls=0
    def encode(self,text):
        self.calls+=1
        return super().encode(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('refusing to overwrite evidence')
    provider=Provider()
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)
        def build():
            encoder=CountEncoder()
            exact=ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'fixture'),
                storage=SQLiteResponseStore(path/'answers.db',namespace='group3-fixture-v1'))
            outer=CompletedDocumentCache(GuardedSemanticCache(exact,scope=KNOWLEDGE,mode='exact'))
            service=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(False,'inner cache off')),
                provider,allowed_models={'fixture'})
            return DocumentAgent(service,DocumentTools(DOCUMENTS),path/'tasks.db',model='fixture',
                answer_cache=outer,orchestration=True,memory_encoder=encoder,stage_accounting=True),encoder
        def run(agent,question,**kwargs):
            task=agent.create(question,**SCOPE,**kwargs)
            for _ in range(30):
                state=agent.advance(task,**SCOPE)
                if state['status'] in {'AWAITING_APPROVAL','FAILED','COMPLETED'}:break
            assert state['status']=='AWAITING_APPROVAL'
            return state
        a,e=build()
        seed=run(a,QUESTION,standalone_document_task=True)
        seed=a.decide(seed['id'],'approve',**SCOPE)  # scripted action, no human judgment
        assert len(provider.calls)==3 and e.calls==0
        a,e=build()  # restart, preserving source and memory-compatible tasks
        hit=run(a,QUESTION,standalone_document_task=True)
        assert hit['provider_calls']==0 and e.calls==0 and len(provider.calls)==3
        hit=a.decide(hit['id'],'approve',**SCOPE)
        context=run(a,QUESTION+' as earlier')
        assert context['task_memories'][0]['source_task_id']==seed['id']
        assert context['provider_calls']==3 and len(provider.calls)==6
        cold_embeddings=e.calls
        a,e=build()  # persisted vectors reuse the candidate embedding
        pending=a.create(QUESTION+' as earlier',**SCOPE)
        warm=a.inspect(pending,**SCOPE)
        assert len(warm['task_memories'])==1 and e.calls==1 and cold_embeddings==2
        cleared=a.clear_task_memory(**SCOPE)
        revoked=a.advance(pending,**SCOPE)
        assert revoked['error']=='memory_scope_revoked' and len(provider.calls)==6
        result={'evidence_type':'constructed offline reference fixtures; scripted approvals NOT human labels',
            'memory_encoder':e.identity,'retrieval_type':'lexical vector baseline, not demonstrated natural semantics',
            'actual_fixture_provider_invocations':6,'task_calls':[3,0,3],
            'cold_candidate_plus_query_embeddings':cold_embeddings,'warm_query_embeddings':e.calls,
            'seed':seed,'hit':hit,'context':context,'warm_snapshot':warm,'clear_result':cleared,'revoked':revoked}
    root=Path(__file__).resolve().parents[1]
    sources=['src/agents/document_agent.py','src/agents/document_orchestration.py','src/agents/task_memory.py',
             'src/agents/document_answer_cache.py','src/policy/work_ledger.py','scripts/check_document_memory.py',
             'scripts/check_before_agent_work.py']
    result['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps({k:result[k] for k in ['actual_fixture_provider_invocations','task_calls',
        'cold_candidate_plus_query_embeddings','warm_query_embeddings','retrieval_type']}))


if __name__=='__main__':main()
