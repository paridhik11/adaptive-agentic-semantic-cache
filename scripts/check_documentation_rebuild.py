"""Offline call-avoidance mechanism, not model quality or live savings."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache, EligibilityDecision
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, AnswerBoundApprovals, approval_binding
from src.policy.sqlite_store import SQLiteResponseStore


class FixtureEncoder:
    identity = 'constant-vector-fixture-not-a-semantic-model'
    def encode(self, text):
        return [1., 0.]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    seed = {'model':'fixture-model', 'temperature':0, 'max_tokens':64,
            'messages':[{'role':'user','content':'Explain tuple immutability'}]}
    paraphrase = {**seed,'messages':[{'role':'user','content':'Describe immutable tuples'}]}
    fresh = {**seed,'messages':[{'role':'user','content':'Explain dictionaries'}]}
    scope = KnowledgeScope('fixture-docs','revision-1','public','authored-approvals-v1')
    binding = approval_binding(paraphrase,seed,FixtureChatProvider()(seed),tenant_id='a',scope=asdict(scope))
    approver = AnswerBoundApprovals({'schema_version':'answer-bound-approvals-v1',
                                   'provenance':'authored_fixture','approvals':[{**binding,'approved':True}]})
    arms = []
    with tempfile.TemporaryDirectory() as directory:
        for mode in ('off','exact','guarded'):
            observed = []
            def provider(body):
                observed.append(body)
                return FixtureChatProvider()(body)
            def build():
                store = SQLiteResponseStore(Path(directory)/(mode+'.db'),namespace='rebuild-fixture-v1')
                cache = GuardedSemanticCache(ScopedResponseCache(storage=store,
                    eligibility=lambda _:EligibilityDecision(True,'fixture only')),
                    scope=scope,mode=mode,encoder=FixtureEncoder(),approver=approver,
                    persistent_candidates=mode=='guarded')
                return GatewayService(cache,provider,allowed_models={'fixture-model'})
            service = build()
            results = [service.complete(seed,'a')]
            service = build()  # restart immediately after seed
            results += [service.complete(body,'a') for body in (seed,paraphrase,fresh)]
            assert sum(r.provider_calls for r in results) == len(observed)
            arms.append({'mode':mode,'requests':4,'actual_fixture_provider_invocations':len(observed),
                         'rows':[{'status':r.status,'tier':r.tier,'provider_calls':r.provider_calls,
                                  'response':r.response,'diagnosis':r.diagnosis} for r in results]})
    assert [arm['actual_fixture_provider_invocations'] for arm in arms] == [4,3,2]
    root = Path(__file__).resolve().parents[1]
    sources = ['src/policy/sqlite_store.py','src/policy/response_cache.py',
               'src/policy/semantic_runtime.py','src/api/service.py', 'src/api/providers.py',
               'scripts/check_documentation_rebuild.py']
    receipt = {'evidence_type':'constructed offline functional fixture; builder-authored approvals',
               'not_measured':['natural-language answer quality','live provider work','real tokens','cost','representative latency'],
               'workload':[seed,seed,paraphrase,fresh], 'restart_after_request':1,'arms':arms,
               'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:
        json.dump(receipt,output,indent=2)
        output.write('\n')
    print(json.dumps([{'mode':a['mode'],'fixture_calls':a['actual_fixture_provider_invocations']} for a in arms]))


if __name__ == '__main__':
    main()
