"""Real cache behavior with controlled local scores; no provider network calls."""
import sys
import pytest
from scripts import demo_flagship, run_demo


def test_default_launcher_prompts_for_key_and_opens_general_chat(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import uvicorn

    apps = []
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(run_demo, "getpass", lambda prompt: "test-placeholder")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))
    monkeypatch.setattr(sys, "argv", [
        "run_demo", "--directory", str(tmp_path), "--journal", str(tmp_path / "quota.sqlite"),
    ])

    run_demo.main()

    with TestClient(apps[0]) as client:
        assert "Ask about any topic" in client.get("/").text
        assert client.get("/benchmark/").status_code == 200


class Encoder:
    identity = 'controlled'
    def __init__(self,*a,**k): pass
    def encode(self,text): return [1.,0.]
class Scorer:
    identity = 'controlled'
    def __init__(self,*a,**k): pass
    def score(self,q,a): return {'probability':.99,'logit':5.,'input_tokens':1}
def forbidden(*a,**k): raise AssertionError('Exact demo loaded neural models')
@pytest.mark.parametrize('experimental',[False,True])
def test_cli_actual_behavior(tmp_path,monkeypatch,experimental):
    monkeypatch.setattr(demo_flagship,'DenseEncoder',Encoder if experimental else forbidden)
    monkeypatch.setattr(demo_flagship,'LocalPassageScorer',Scorer if experimental else forbidden)
    receipts=[];original=demo_flagship.run_demo
    def observed(*a,**k):
        r=original(*a,**k);receipts.append(r);return r
    monkeypatch.setattr(demo_flagship,'run_demo',observed)
    monkeypatch.setattr(sys,'argv',['run_demo','--cli','--directory',str(tmp_path)]+(['--experimental-semantic'] if experimental else []))
    run_demo.main();r=receipts[0]
    assert r['serving_mode']==('guarded' if experimental else 'exact')
    assert r['steps'][2]['cache_tier']=='EXACT' and r['steps'][2]['provider_calls']==0
    assert r['steps'][3]['cache_tier']==('SEMANTIC' if experimental else 'NONE')
    assert r['steps'][3]['provider_calls']==(0 if experimental else 3)
    if not experimental: assert all(x['cache_tier']!='SEMANTIC' for x in r['steps'])
@pytest.mark.parametrize('experimental',[False,True])
def test_web_actual_behavior(tmp_path,monkeypatch,experimental):
    from src.api import demo
    from src.policy import hybrid_cache
    import uvicorn
    agents=[];apps=[];original=demo.DocumentAgent
    def capture(*a,**k):
        value=original(*a,**k);agents.append(value);return value
    monkeypatch.setattr(demo,'DocumentAgent',capture)
    monkeypatch.setattr(hybrid_cache,'DenseEncoder',Encoder if experimental else forbidden)
    monkeypatch.setattr(hybrid_cache,'LocalPassageScorer',Scorer if experimental else forbidden)
    monkeypatch.setattr(uvicorn,'run',lambda app,**k:apps.append(app))
    monkeypatch.setattr(sys,'argv',['run_demo','--whole-answer-cache','--directory',str(tmp_path)]+(['--experimental-semantic'] if experimental else []))
    run_demo.main();assert len(apps)==1
    cache=agents[0].answer_cache.cache
    body={'model':'fixture-model','temperature':0.,'messages':[{'role':'system','content':'Pinned docs'},{'role':'user','content':'Explain list conversion in Python.'}]}
    response={'model':'fixture-model','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'Use tuple(values) to construct a tuple from a list.'}}]}
    calls=[]
    def provider(r):calls.append(r);return response
    cache.process(body,tenant_id='t',provider=provider)
    assert cache.process(body,tenant_id='t',provider=provider).tier=='EXACT'
    para={**body,'messages':[body['messages'][0],{'role':'user','content':'Can you explain list conversion in Python?'}]}
    before=len(calls);result=cache.process(para,tenant_id='t',provider=provider)
    assert result.tier==('SEMANTIC' if experimental else 'PROVIDER')
    assert result.status==('HIT' if experimental else 'MISS')
    assert len(calls)-before==(0 if experimental else 1)
