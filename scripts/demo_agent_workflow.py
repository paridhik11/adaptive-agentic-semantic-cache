"""Run the bounded agent integration; fixture default, live calls require explicit opt-in."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agents.workflow import AgentWorkflow
from src.api.providers import FixtureChatProvider,HTTPChatProvider
from src.api.service import GatewayService,ConservativeRulesPolicy
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision


class AgentFixture(FixtureChatProvider):
    def __call__(self,body):
        result=super().__call__(body)
        # Deterministic protocol fixture, not a model's reasoning/quality judgment.
        content='{"approved":true}' if 'You are a reviewer.' in body['messages'][0]['content'] else 'SIMULATED specialist answer: review the request and apply the relevant algorithm.'
        result['choices'][0]['message']['content']=content
        result['usage']['completion_tokens']=len(content.split())
        result['usage']['total_tokens']=result['usage']['prompt_tokens']+result['usage']['completion_tokens']
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--provider',choices=['fixture','http'],default='fixture')
    p.add_argument('--model',default='fixture-model')
    p.add_argument('--question',default='Explain the binary search algorithm.')
    p.add_argument('--allow-live-calls',action='store_true')
    p.add_argument('--max-live-calls',type=int,default=0)
    args=p.parse_args()
    if args.output and args.output.exists():p.error('choose a new output file')
    if args.provider=='http' and (not args.allow_live_calls or args.max_live_calls<6):
        p.error('live demo requires opt-in and a budget covering six provider attempts')
    policy=ConservativeRulesPolicy() if args.provider=='http' else lambda _:EligibilityDecision(True,'explicit_fixture_policy')
    cache=ScopedResponseCache(eligibility=policy)
    provider=HTTPChatProvider(os.environ['UPSTREAM_CHAT_URL'],api_key=os.environ.get('UPSTREAM_API_KEY')) if args.provider=='http' else AgentFixture()
    service=GatewayService(cache,provider,allowed_models={args.model})
    workflow=AgentWorkflow(service,model=args.model)
    runs=[asdict(workflow.run(args.question,tenant_id='demo-a')) for _ in range(2)]
    runs.append(asdict(workflow.run(args.question,tenant_id='demo-b')))
    calls=None if any(r['provider_calls'] is None for r in runs) else sum(r['provider_calls'] for r in runs)
    result={'evaluation_mode':provider.mode,'external_model_calls':calls if args.provider=='http' else 0,
        'note':'Permissive eligibility and reviewer approval are test fixtures only. Production conservative policy bypasses contextual requests.', 'runs':runs}
    if args.output:
        from src.benchmark.schema import write_new
        write_new(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
