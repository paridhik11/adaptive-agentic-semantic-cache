"""Subprocess restart/approval smoke. Fixture only; answer-free aggregate output."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    with tempfile.TemporaryDirectory() as directory:
        command=[sys.executable,str(ROOT/'scripts/demo_document_agent.py'),'--database',str(Path(directory)/'tasks.sqlite'),'--documents',str(ROOT/'examples/agent/documents.fixture.json')]
        def invoke(*args):
            return json.loads(subprocess.check_output([*command,*args],cwd=ROOT,text=True))['task']
        first=invoke('start','Explain tuple immutability')
        assert first['status']=='AWAITING_APPROVAL'
        assert len(first['tools'])==2 and first['provider_calls']==3
        task=first['id']
        approved=invoke('decide',task,'approve')
        resumed=invoke('resume',task)
        assert approved==resumed and resumed['status']=='COMPLETED'
        assert resumed['answer']==first['draft']
        rejected=invoke('start','Explain tuple immutability')
        rejected=invoke('decide',rejected['id'],'reject')
        assert rejected['status']=='REJECTED' and rejected['answer'] is None
        result={'schema_version':'r4-functional-smoke-v1','execution_mode':'simulated_functional_fixture',
                'external_model_calls':0,'independent_reviews':0,'quality_measured':False,'live_savings_measured':False,
                'first_task_fixture_provider_calls':first['provider_calls'],'first_task_document_tools':len(first['tools']),
                'approval_after_process_restart':True,'resume_completed_without_reexecution':True,'rejection_without_answer':True}
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
