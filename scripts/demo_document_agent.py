"""Durable bounded document agent CLI. Default fixture makes no external calls.

The HTTP option targets an operator-managed gateway, never OpenRouter directly.
Its durable quota/free-provider policy must be configured at that gateway.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agents.document_agent import DocumentAgent,DocumentTools
from src.api.providers import FixtureChatProvider,HTTPChatProvider
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision

class DocumentFixture(FixtureChatProvider):
    def __call__(self,body):
        result=super().__call__(body)
        instruction=body['messages'][0]['content']
        payload=json.loads(body['messages'][1]['content'])
        document=payload['catalog'][0]
        if 'Plan 1-8' in instruction:
            content=json.dumps({'steps':[
                {'id':'search','tool':'search','arguments':{'query':payload['question'][:1000]},'depends_on':[]},
                {'id':'read','tool':'read','arguments':{'document_id':document},'depends_on':['search']}]})
        elif 'Check correctness' in instruction:content=json.dumps({'approved':True,'feedback':'Scripted fixture approval; not a quality judgment.'})
        else:content='SIMULATED document answer; inspect reference ['+document+'].'
        result['choices'][0]['message']['content']=content
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--documents',type=Path,required=True,help='JSON object of document ID to text')
    parser.add_argument('--tenant',default='demo');parser.add_argument('--principal',default='operator')
    parser.add_argument('--model',default='fixture-model')
    parser.add_argument('--gateway-url',help='Explicit live gateway /v1/chat/completions endpoint')
    parser.add_argument('--allow-live-calls',action='store_true')
    parser.add_argument('--cache-off',action='store_true')
    commands=parser.add_subparsers(dest='command',required=True)
    start=commands.add_parser('start');start.add_argument('question')
    for name in ('resume','inspect','delete'):
        sub=commands.add_parser(name);sub.add_argument('task_id')
    decide=commands.add_parser('decide');decide.add_argument('task_id');decide.add_argument('decision',choices=['approve','modify','reject']);decide.add_argument('--answer')
    args=parser.parse_args()
    if args.gateway_url and not args.allow_live_calls:parser.error('HTTP requires --allow-live-calls and a separately quota-controlled gateway')
    if args.gateway_url and args.model=='fixture-model':parser.error('specify gateway model')
    provider=HTTPChatProvider(args.gateway_url,api_key=os.environ.get("DOCUMENT_AGENT_GATEWAY_TOKEN")) if args.gateway_url else DocumentFixture()
    policy=lambda _:EligibilityDecision(not bool(args.gateway_url),'fixture_only' if not args.gateway_url else 'live_agent_context_unvalidated')
    service=GatewayService(ScopedResponseCache(eligibility=policy),provider,allowed_models={args.model})
    agent=DocumentAgent(service,DocumentTools(json.loads(args.documents.read_text())),args.database,model=args.model)
    scope={'tenant':args.tenant,'principal':args.principal}
    if args.command=='start':tid=agent.create(args.question,cache=not args.cache_off,**scope)
    else:tid=args.task_id
    if args.command in ('start','resume'):
        for _ in range(30):
            state=agent.advance(tid,**scope)
            if state['status'] in ('COMPLETED','REJECTED','FAILED','AWAITING_APPROVAL'):break
    elif args.command=='decide':state=agent.decide(tid,args.decision,modified_answer=args.answer,**scope)
    elif args.command=='delete':agent.delete(tid,**scope);state={'id':tid,'status':'DELETED'}
    else:state=agent.inspect(tid,**scope)
    print(json.dumps({'execution_mode':'live_gateway' if args.gateway_url else 'simulated_functional_fixture','task':state},indent=2))

if __name__=='__main__':main()
