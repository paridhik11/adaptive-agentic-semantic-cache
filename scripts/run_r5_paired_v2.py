"""Explicit revised R5 live development pilot; preserves the failed predecessor."""
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.paired_runtime import build_events,run_paired,summarize,review_packet
from src.benchmark.schema import read_json,write_new,digest
from src.benchmark.control import RunPaused
from src.benchmark.free_pilot import FreeOpenRouterProvider,PilotJournal,JournaledProvider,CompleteAnswerProvider
from src.benchmark.r5_revision import (predecessor_snapshot, revise_events, pause_guidance,
    REVISED_RUN, MAX_OUTPUT_TOKENS)
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--provider',choices=['free'],default='free')
    p.add_argument('--model',default='fixture-model')
    p.add_argument('--run-id',choices=[REVISED_RUN],default=REVISED_RUN);p.add_argument('--journal',type=Path,default=Path('runs/openrouter-free.sqlite'))
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    corpus=read_json(args.corpus)
    if digest(corpus)!='0fe8140b1d3bb1bf752821d15e863b5552db324d17ed373b89952837620a7089':p.error('corpus content differs from verified R2 preparation')
    if corpus.get('source_sha256')!='648bc2d476230c98f61571e9de969811150875996de2552d5c7656368b2df184':p.error('pinned source checksum required')
    if not (args.corpus.parent/'LICENSE').is_file():p.error('prepared corpus LICENSE is required')
    FreeOpenRouterProvider('validation-only',models={args.model})
    if not args.run_id:p.error('--run-id is required')
    # One question per development topic group; four scenario types each.
    grouped={}
    for row in corpus['items']:
        if row['split']=='dev':grouped.setdefault(row['group_id'],[]).append(row)
    selected=[r for group in sorted(grouped) for r in grouped[group][:1]]
    selected_corpus={**corpus,'items':selected}
    events=[e for e in build_events(selected_corpus,model=args.model) if e['kind'] not in ('different_tenant','different_revision')]
    events=revise_events(events)
    try: predecessor=predecessor_snapshot(args.journal,args.model)
    except (ValueError, OSError, sqlite3.Error) as exc: p.error(str(exc))
    attempt_cap=predecessor['remaining_attempt_allowance']
    files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
           for folder in ('src/agents','src/api','src/benchmark','src/policy') for path in sorted((ROOT/folder).glob('*.py'))}
    files['scripts/run_r5_paired_v2.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config={'protocol_sha256':hashlib.sha256((ROOT/'docs/R5_LIVE_V2_PROTOCOL.md').read_bytes()).hexdigest(),'protocol':'r5-live-development-v2','predecessor':predecessor,'events_sha256':digest(events),'corpus_sha256':digest(corpus),
            'code_hashes':files,'model':args.model,'reasoning_enabled':False,'max_output_tokens':MAX_OUTPUT_TOKENS,'provider':args.provider,'daily_cap':40,'minute_cap':10,
            'maximum_attempts':attempt_cap,'run_id':args.run_id,'output_dir':str(args.output_dir.resolve())}
    if not args.execute:
        print(json.dumps({'plan_only':True,'requests':len(events),'arms':2,'maximum_provider_attempts':attempt_cap,'predecessor_attempts':predecessor['attempts'],'combined_attempt_cap':16,'config_sha256':digest(config),'key_read':False,'calls_made':0},indent=2));return
    if args.output_dir.with_name(args.output_dir.name+'-evidence.zip').exists():p.error('evidence archive already exists')
    if args.output_dir.exists():p.error('output directory exists; refusing to overwrite evidence')
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:p.error('configure OPENROUTER_API_KEY; no calls made')
    args.journal.parent.mkdir(parents=True,exist_ok=True)
    journal=PilotJournal(args.journal,run_id=args.run_id,fingerprint=digest(config),max_calls=attempt_cap)
    metered=JournaledProvider(FreeOpenRouterProvider(key,models={args.model},reasoning_enabled=False),journal)
    provider=CompleteAnswerProvider(metered)
    try:trace=run_paired(events,provider,include_unsafe=False,modes=('off','exact'))
    except RunPaused as pause:
        print(json.dumps({'status':'PAUSED','reason':pause.reason,'retry_at':pause.retry_at,'diagnostics':pause.diagnostics,**pause_guidance(pause)},indent=2));return
    trace['config']=config
    trace['code_parent']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    trace['execution_mode']='resumed_reconstruction' if metered.replayed_calls else 'live_uninterrupted'
    trace['live_latency_valid']=metered.replayed_calls==0
    report=summarize(trace)
    report['collection_attempts']={'predecessor':predecessor['attempts'],
        'revised_run':sum(journal.status().values()),
        'combined':predecessor['attempts']+sum(journal.status().values()),
        'combined_cap':16,'new_this_process':metered.new_calls,
        'replayed_this_process':metered.replayed_calls}
    if metered.replayed_calls:
        report['live_latency_valid']=False
        for arm in report['arms'].values():arm['latency_is_live_model_evidence']=False
    reviews,mapping=review_packet(trace)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        stage=Path(temporary)/'evidence';stage.mkdir()
        for name,value in [('predecessor-diagnostics.json',predecessor),('trace.json',trace),('summary.json',report),('reviews.template.json',reviews),('review_mapping.private.json',mapping)]:write_new(stage/name,value)
        (stage/'LICENSE').write_bytes((args.corpus.parent/'LICENSE').read_bytes())
        write_new(stage/'source-manifest.json',read_json(ROOT/'examples/benchmark/python_faq.sources.json'))
        write_new(stage/'manifest.json',{'config_sha256':digest(config),'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in stage.iterdir()}})
        stage.rename(args.output_dir)
    import zipfile
    archive=args.output_dir.with_name(args.output_dir.name+'-evidence.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for file in args.output_dir.iterdir():bundle.write(file,file.name)
    print(json.dumps(report,indent=2));print('Evidence:',args.output_dir);print('Share evidence archive:',archive)

if __name__=='__main__':main()
