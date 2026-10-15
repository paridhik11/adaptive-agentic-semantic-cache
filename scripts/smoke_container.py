"""CI smoke against an already-running fixture container; no external model calls."""
import json
import os
from urllib.request import Request, urlopen


def main():
    base = 'http://127.0.0.1:8000'
    token = os.environ['SMOKE_TENANT_TOKEN']
    body = {'model':'fixture-model','messages':[{'role':'user','content':'What is photosynthesis?'}],
            'temperature':0,'max_tokens':128}
    def complete():
        req=Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
        with urlopen(req,timeout=10) as response:
            result=json.load(response)
            assert result['choices'][0]['finish_reason']=='stop'
            return response.headers['X-Cache']
    with urlopen(base+'/health',timeout=10) as response:
        assert json.load(response)['provider_mode']=='simulated_fixture'
    assert complete()=='MISS'
    assert complete()=='HIT'
    with urlopen(base+'/dashboard',timeout=10) as response:
        assert b'Operational counters' in response.read()
    print('Container fixture MISS/HIT, health and dashboard checks passed.')


if __name__=='__main__':main()
