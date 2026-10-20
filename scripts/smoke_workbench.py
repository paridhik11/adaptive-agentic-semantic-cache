"""Exercise the integrated HTTP product; scripted reference replay, no live calls."""
import json
from pathlib import Path
import re
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from src.api.demo import create_demo_app


def main():
    with tempfile.TemporaryDirectory() as directory:
        c=TestClient(create_demo_app(directory));shell=c.get('/')
        c.headers['x-demo-session']=re.search(r'name="demo-session" content="([^"]+)"',shell.text).group(1)
        question='How do I convert between tuples and lists?'
        def run(cache):
            response=c.post('/api/tasks',json={'question':question,'cache':cache});response.raise_for_status();state=response.json()
            for _ in range(15):
                if state['status'] in ('AWAITING_APPROVAL','COMPLETED','FAILED','REJECTED'):return state
                r=c.post('/api/tasks/'+state['id']+'/advance');r.raise_for_status();state=r.json()
            raise AssertionError('Unbounded flow')
        first=run(True)
        approved=c.post('/api/tasks/'+first['id']+'/decision',json={'decision':'approve'}).json()
        repeated=run(True);off=run(False);info=c.get('/api/info').json()
        assert approved['status']=='COMPLETED' and repeated['draft_cache_hit'] and not off['draft_cache_hit']
        assert (first['observed_attempts'],repeated['observed_attempts'],off['observed_attempts'])==(3,2,3)
        assert first['draft']==repeated['draft'] and info['current_process_attempts']==8
        assert c.get('/assets/workbench.css').status_code==200 and c.get('/assets/workbench.js').status_code==200
        result={'evidence_kind':'integrated_http_scripted_reference_replay','external_model_calls':0,
                'first_task_scripted_calls':3,'repeat_task_scripted_calls':2,'cache_off_task_scripted_calls':3,
                'repeat_exact_draft_hit':True,'repeated_draft_identical':True,'human_decision_endpoint_exercised':True,
                'independent_human_reviews':0,'source_documents':len(info['reference']['documents']),
                'http_flow_verified':True,'browser_visual_qa':False,'live_savings_measured':False}
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
