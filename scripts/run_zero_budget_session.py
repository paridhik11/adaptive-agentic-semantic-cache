"""One foreground local job: run a small exact-cache demonstration and show progress."""
import argparse
import json
from pathlib import Path
import sys
import time
import webbrowser
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import read_json,write_new
from src.benchmark.progress import ProgressPage
from scripts.run_free_pilot import main as pilot


def workload():
    rows=[]
    topics=[('photosynthesis','What is photosynthesis?','What is photosynthesis in plants?'),
            ('binary-search-tree','What is a binary search tree?','Explain what a binary search tree is.')]
    for group,question,paraphrase in topics:
        for kind,text in [('seed',question),('repeat',question),('paraphrase',paraphrase)]:
            rows.append({'event_id':group+'-'+kind,'tenant_id':'guided-pilot','source':'authored-guided-pilot',
                'group_id':group,'split':'dev','request':{'model':'fixture-model','temperature':0,'max_tokens':512,
                'messages':[{'role':'user','content':text}]}})
    return {'schema_version':'workload-v1','name':'guided-exact-cache-demonstration','nature':'synthetic_fixture',
        'provenance':'Two authored topics, each with a seed, an exact repeat and a paraphrase. Constructed diagnostic sequence, not natural traffic or sealed data.',
        'license':'MIT','requests':rows}


def bundle(directory,evidence,run_id):
    from scripts.inspect_pilot_journal import inspect
    from src.benchmark.schema import digest
    target=directory/'results.json'
    if not target.exists():
        package={'schema_version':'pilot-result-package-v1',
            'trace':read_json(evidence/'trace.json'),'report':read_json(evidence/'report.unreviewed.json'),
            'journal_diagnostics':inspect('runs/openrouter-free.sqlite',run_id)}
        package['trace_sha256']=digest(package['trace'])
        write_new(target,package)
    print('One file to share for analysis:',target)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--no-browser',action='store_true')
    p.add_argument('--run-id',default='guided-exact-v1')
    p.add_argument('--model',default='nvidia/nemotron-3-super-120b-a12b:free')
    p.add_argument('--directory',type=Path,default=Path('runs/guided-exact-v1'))
    args=p.parse_args(argv)
    directory=args.directory;directory.mkdir(parents=True,exist_ok=True)
    binding=directory/'session.json'
    profile={'run_id':args.run_id,'model':args.model}
    if binding.exists():
        if read_json(binding)!=profile:p.error('directory belongs to another run or model')
    else:write_new(binding,profile)
    source=directory/'workload.json';expected=workload()
    if source.exists():
        if read_json(source)!=expected:p.error('existing guided workload differs')
    else:write_new(source,expected)
    evidence=directory/'evidence'
    if evidence.exists():
        print('Existing evidence preserved:',evidence)
        print(json.dumps(read_json(evidence/'report.unreviewed.json')['status_counts'],indent=2))
        bundle(directory,evidence,args.run_id);return
    page=ProgressPage(directory/'status.html',run_id=args.run_id,model=args.model,total_requests=6)
    command=['--workload',str(source),'--model',args.model,'--run-id',args.run_id,
        '--output-dir',str(evidence),'--limit','6','--max-calls','12','--max-output-tokens','512',
        '--reasoning','off','--answer-style','concise']
    if args.execute:command.append('--execute')
    print('Progress page:',page.path.resolve())
    if args.execute and not args.no_browser:webbrowser.open(page.path.resolve().as_uri())
    deadline=time.time()+180
    while True:
        result=pilot(command,observer=page)
        if not isinstance(result,dict) or result.get('reason')!='project_minute_quota':break
        retry=result.get('retry_at_utc_epoch')
        if retry is None or retry>deadline:break
        page({'kind':'waiting','message':'Minute quota reached; the same foreground job will resume from saved responses.'})
        while time.time()<retry:time.sleep(min(2,max(0,retry-time.time())))
    if isinstance(result,dict) and result.get('status')=='completed':
        bundle(directory,evidence,args.run_id)
    elif isinstance(result,dict):
        path=directory/'paused.json'
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Paused; inspect the status page. No further model calls will be made by this job.')


if __name__=='__main__':main()
