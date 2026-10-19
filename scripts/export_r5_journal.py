"""Export saved R5 responses read-only. No credentials, network, or new attempts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import digest, read_json, _constant, _unique
from src.benchmark.diagnostics import response_diagnostics
from src.benchmark.paired_runtime import build_events
from src.benchmark.r5_revision import revise_events

RUNS=('r5-live-v1','r5-live-v2')
CORPUS_HASH='0fe8140b1d3bb1bf752821d15e863b5552db324d17ed373b89952837620a7089'


def export_saved(journal, corpus_path, output, model):
    journal=Path(journal).resolve();output=Path(output).resolve()
    if not journal.is_file():raise ValueError('Existing shared journal required')
    if output.exists():raise ValueError('Output exists; refusing to overwrite evidence')
    corpus=read_json(corpus_path)
    if digest(corpus)!=CORPUS_HASH:raise ValueError('Pinned corpus digest mismatch')
    license_path=Path(corpus_path).parent/'LICENSE'
    if not license_path.is_file():raise ValueError('Prepared corpus LICENSE required')
    grouped={}
    for item in corpus['items']:
        if item['split']=='dev':grouped.setdefault(item['group_id'],[]).append(item)
    selected={**corpus,'items':[row for group in sorted(grouped) for row in grouped[group][:1]]}
    original=[e for e in build_events(selected,model=model) if e['kind'] not in ('different_tenant','different_revision')]
    profiles={RUNS[0]:original,RUNS[1]:revise_events(original)}
    db=sqlite3.connect(journal.as_uri()+'?mode=ro',uri=True)
    try:
        db.execute('BEGIN')
        metadata=db.execute('SELECT run,fingerprint,max_calls FROM pilot_runs WHERE run IN (?,?) ORDER BY run',RUNS).fetchall()
        rows=db.execute('SELECT run,ordinal,request_sha,started,state,payload FROM pilot_calls WHERE run IN (?,?) ORDER BY run,ordinal',RUNS).fetchall()
    finally:db.close()
    if not metadata or not rows:raise ValueError('No saved R5 attempts found')
    attempts=[];items={};mapping=[];counts={}
    for run,ordinal,request_sha,started,state,payload in rows:
        matches=[e for e in profiles[run] if digest(e['request'])==request_sha]
        # Repeat and seed may share a request. Never infer event order or cache arm.
        binding=matches[0] if matches else None
        record={'run_id':run,'ordinal':ordinal,'request_sha256':request_sha,'started_utc_epoch':started,
                'journal_state':state,'matching_event_ids':[e['id'] for e in matches],
                'arm':None,'response':None,'diagnostics':None,'decode_error':None}
        if state=='complete':
            try:
                response=json.loads(payload,parse_constant=_constant,object_pairs_hook=_unique)
                record['response']=response
                record['diagnostics']=response_diagnostics(response,binding['request'] if binding else {'n':1})
            except (ValueError,TypeError):record['decode_error']='invalid_saved_json'
        elif state=='error':record['decode_error']='saved_provider_error'
        else:record['decode_error']='uncertain_attempt'
        if binding and record['diagnostics'] and record['diagnostics']['admissible_complete_answer']:
            answer=record['response']['choices'][0]['message']['content']
            rid=digest({'request':binding['request'],'revision':binding['revision'],'answer':answer})
            items[rid]={'review_id':rid,'question':binding['request']['messages'][-1]['content'],
                        'reference':binding['reference'],'answer':answer,'correct':None,'supported':None,'reviewer':None,'notes':''}
            mapping.append({'run_id':run,'ordinal':ordinal,'review_id':rid,'response_sha256':record['diagnostics']['response_sha256']})
        attempts.append(record)
    for run,fingerprint,cap in metadata:
        subset=[a for a in attempts if a['run_id']==run]
        counts[run]={'registered_attempt_cap':cap,'saved_attempts':len(subset),
                    'states':dict(Counter(a['journal_state'] for a in subset)),
                    'admissible_complete_responses':sum(bool(a['diagnostics'] and a['diagnostics']['admissible_complete_answer']) for a in subset),
                    'unmatched_request_hashes':sum(not a['matching_event_ids'] for a in subset),'config_fingerprint':fingerprint}
    summary={'schema_version':'r5-partial-journal-export-v1','evidence_kind':'saved_live_responses_exported_offline',
             'collection_status':'partial_or_unverified_not_a_completed_paired_run','model_calls_made_by_export':0,
             'journal_modified':False,'live_latency_valid':False,'paired_savings':None,'answer_correctness':None,
             'combined_saved_attempts':len(attempts),'runs':counts,'unique_review_items':len(items),
             'independent_reviews':0,'source_snapshot_sha256':digest({'metadata':metadata,'rows':rows}),
             'limitations':['Arm attribution and complete event coverage are not established by this export.',
                            'Truncated, unmatched, failed and uncertain attempts are retained in accounting; only complete bound answers enter review.',
                            'Snapshot digest includes saved raw payloads; raw SQLite database and unrelated runs are not exported.']}
    reviews=list(items.values());random.Random(71).shuffle(reviews)
    values={'summary.json':summary,'saved-responses.private.json':{'attempts':attempts},
            'reviews.template.json':{'schema_version':'r5-journal-answer-review-v1','independent_review_required':True,'items':reviews},
            'review_mapping.private.json':mapping}
    files={name:(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode() for name,value in values.items()}
    files['LICENSE']=license_path.read_bytes()
    files['manifest.json']=(json.dumps({'schema_version':'r5-partial-export-manifest-v1','files':{name:hashlib.sha256(content).hexdigest() for name,content in files.items()}},indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,content in files.items():archive.writestr(name,content)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('runs/r5-partial-evidence.zip'))
    p.add_argument('--model',default='nvidia/nemotron-3-super-120b-a12b:free')
    args=p.parse_args()
    try:summary=export_saved(args.journal,args.corpus,args.output,args.model)
    except (ValueError,OSError,sqlite3.Error) as exc:p.error(str(exc))
    print(json.dumps(summary,indent=2));print('Share evidence archive:',args.output)

if __name__=='__main__':main()
