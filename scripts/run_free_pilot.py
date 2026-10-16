"""Free-only, capped OpenRouter development pilot with durable resume.

No calls run without explicit --execute. No waiting/retries are automatic.
Previously completed calls replay from the journal; resumed timing is invalidated.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.control import RunPaused
from src.benchmark.schema import canonical,digest,read_json,validate_workload,write_new
from src.benchmark.free_pilot import FreeOpenRouterProvider,PilotJournal,JournaledProvider,CompleteAnswerProvider
from src.benchmark.runner import run_workload
from src.benchmark.semantic import AnswerVerifier,LexicalEncoder
from src.benchmark.report import review_template,summarize
from src.benchmark.views import review_html


def main(argv=None,observer=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload',type=Path,required=True)
    p.add_argument('--model',required=True,help='explicit provider/model:free ID')
    p.add_argument('--verifier-model')
    p.add_argument('--policy',choices=['exact','similarity','verified'],default='exact')
    p.add_argument('--run-id',required=True)
    p.add_argument('--journal',type=Path,default=Path('runs/openrouter-free.sqlite'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--max-calls',type=int,default=30)
    p.add_argument('--max-output-tokens',type=int,help='explicit output-limit override; changing it requires a new run ID')
    p.add_argument('--reasoning',choices=['default','off'],default='default')
    p.add_argument('--answer-style',choices=['original','concise'],default='original')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if not 1<=args.limit<=10:p.error('pilot limit must be between one and ten requests')
    if not 1<=args.max_calls<=30:p.error('pilot budget must be between one and thirty attempts')
    if args.max_output_tokens is not None and not 1<=args.max_output_tokens<=4096:p.error('output limit must be between one and 4096 tokens')
    original=validate_workload(read_json(args.workload));workload=copy.deepcopy(original)
    rows=[r for r in workload['requests'] if r['split']=='dev'][:args.limit]
    if not rows:p.error('workload has no development requests')
    workload['requests']=rows
    for row in rows:
        row['request']['model']=args.model
        if args.answer_style=='concise':
            row['request']['messages'][-1]['content']+='\nAnswer in at most two concise sentences, without headings or a step-by-step explanation.'
        if args.max_output_tokens is not None:
            field='max_completion_tokens' if 'max_completion_tokens' in row['request'] else 'max_tokens'
            row['request'][field]=args.max_output_tokens
    # Entire input validation happens before creating the journal or making calls.
    from src.api.service import validate_request
    for row in rows:validate_request(row['request'],frozenset({args.model}),4096)
    models={args.model}
    if args.policy=='verified':
        if not args.verifier_model:p.error('--verifier-model is required for verified policy')
        models.add(args.verifier_model)
    # Validate the model IDs even in plan-only mode. Dummy token never leaves process.
    FreeOpenRouterProvider('validation-only',models=models)
    root=Path(__file__).resolve().parents[1]
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    # Include implementation bytes so editing uncommitted code cannot silently resume.
    code_hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ('src/api','src/policy','src/classifier','src/benchmark')
        for path in sorted((root/folder).glob('*.py'))}
    code_hashes['scripts/run_free_pilot.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config={'source_workload_sha256':digest(original),'selected_workload_sha256':digest(workload),
        'selection':'first development requests in original order; development pilot only',
        'policy':args.policy,'models':sorted(models),'limit':args.limit,'max_calls':args.max_calls,
        'reasoning':args.reasoning,'answer_style':args.answer_style,
        'code_hashes':code_hashes,'protocol':'journal-reconstruction-v1','output_dir':str(args.output_dir.resolve())}
    worst=(3 if args.policy=='verified' else 2)*len(rows)
    print(json.dumps({'requests':len(rows),'maximum_logical_calls':worst,'run_attempt_cap':args.max_calls,
        'project_daily_cap':40,'project_minute_cap':10,'execution_requested':args.execute},indent=2))
    if not args.execute:
        print('Plan only: no key read and no API calls. Add --execute after configuring OPENROUTER_API_KEY.')
        return
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:p.error('configure OPENROUTER_API_KEY in the environment; no calls were made')
    # Completed evidence is immutable, including on resume.
    if args.output_dir.exists():p.error('output directory already exists; completed evidence is not overwritten')
    args.journal.parent.mkdir(parents=True,exist_ok=True)
    journal=PilotJournal(args.journal,run_id=args.run_id,fingerprint=digest(config),max_calls=args.max_calls)
    transport=FreeOpenRouterProvider(key,models=models,**({'reasoning_enabled':False} if args.reasoning=='off' else {}))
    provider=JournaledProvider(transport,journal,observer=observer)
    guarded=CompleteAnswerProvider(provider)
    verifier=AnswerVerifier(guarded,args.verifier_model) if args.policy=='verified' else None
    try:
        trace=run_workload(workload,guarded,policy=args.policy,encoder=LexicalEncoder(),verifier=verifier,observer=observer)
    except RunPaused as pause:
        paused={'status':'paused','reason':pause.reason,'retry_at_utc_epoch':pause.retry_at,
            'new_attempts_this_invocation':provider.new_calls,'replayed_calls':provider.replayed_calls,
            'journal_states':journal.status(),'response_diagnostics':pause.diagnostics}
        print(json.dumps(paused,indent=2))
        print('Resume unchanged quota pauses; inspect incomplete responses before starting a revised run.')
        if observer is not None:observer({'kind':'paused',**paused})
        return paused
    trace.update({'code_tree':tree,'working_tree_dirty':dirty,'pilot_config':config,
        'execution_mode':'resumed_reconstruction' if provider.replayed_calls else 'live_uninterrupted',
        'new_attempts_this_invocation':provider.new_calls,'replayed_calls':provider.replayed_calls,
        'journal_states':journal.status(),'latency_valid':provider.replayed_calls==0})
    if provider.replayed_calls:
        trace['timing_scope']='Reconstructed serial run using journaled API responses; raw reconstruction timings are invalid for live latency comparisons.'
    summary=summarize(trace);template=review_template(trace)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pilot-export-',dir=args.output_dir.parent) as temporary:
        stage=Path(temporary)/'evidence'
        stage.mkdir()
        write_new(stage/'trace.json',trace)
        write_new(stage/'report.unreviewed.json',summary)
        write_new(stage/'reviews.template.json',template)
        with (stage/'review.html').open('x',encoding='utf-8') as f:f.write(review_html(trace,template,summary))
        write_new(stage/'manifest.json',{'trace_sha256':digest(trace),'fingerprint':digest(config),
            'files':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in stage.iterdir()}})
        stage.rename(args.output_dir)
    if observer is not None:observer({'kind':'completed','output_dir':str(args.output_dir),'report':summary})
    print('Completed pilot evidence:',args.output_dir)
    print('No answers are independently reviewed. Inspect execution_mode and latency_valid before making claims.')
    return {'status':'completed','output_dir':str(args.output_dir)}


if __name__=='__main__':main()
