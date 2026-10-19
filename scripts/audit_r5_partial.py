"""Verify a stopped R5 export and produce aggregate evidence; no model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import digest,read_json,write_new
from src.benchmark.diagnostics import response_diagnostics
from src.benchmark.paired_runtime import build_events
from src.benchmark.r5_revision import revise_events


def audit(archive,corpus_path):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
        manifest=json.loads(z.read('manifest.json'))
        if set(manifest['files'])!=set(names)-{'manifest.json'}:raise ValueError('Manifest coverage mismatch')
        for name,h in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Manifest hash mismatch')
        summary=json.loads(z.read('summary.json'))
        attempts=json.loads(z.read('saved-responses.private.json'))['attempts']
        packet=json.loads(z.read('reviews.template.json'))
        mapping=json.loads(z.read('review_mapping.private.json'))
    corpus=read_json(corpus_path)
    if digest(corpus)!='0fe8140b1d3bb1bf752821d15e863b5552db324d17ed373b89952837620a7089':raise ValueError('Pinned corpus mismatch')
    grouped={}
    for row in corpus['items']:
        if row['split']=='dev':grouped.setdefault(row['group_id'],[]).append(row)
    selected={**corpus,'items':[row for group in sorted(grouped) for row in grouped[group][:1]]}
    model='nvidia/nemotron-3-super-120b-a12b:free'
    events=[e for e in build_events(selected,model=model) if e['kind'] not in ('different_tenant','different_revision')]
    profiles={'r5-live-v1':events,'r5-live-v2':revise_events(events)}
    seen=set();expected_items={};expected_mapping=[];counts={}
    for run in profiles:
        rows=[a for a in attempts if a['run_id']==run]
        if [a['ordinal'] for a in rows]!=list(range(len(rows))):raise ValueError('Noncontiguous attempt ordinals')
        counters=Counter();tokens=Counter()
        for a in rows:
            key=(run,a['ordinal'])
            if key in seen:raise ValueError('Duplicate attempt')
            seen.add(key)
            matches=[e for e in profiles[run] if digest(e['request'])==a['request_sha256']]
            if not matches or a['matching_event_ids']!=[e['id'] for e in matches]:raise ValueError('Request binding mismatch')
            event=matches[0]
            if a['journal_state']!='complete':raise ValueError('This audit expects received responses; retain other evidence separately')
            d=response_diagnostics(a['response'],event['request'])
            if d!=a['diagnostics']:raise ValueError('Response diagnostic mismatch')
            counters['saved_attempts']+=1
            counters['complete_responses' if d['admissible_complete_answer'] else 'incomplete_responses']+=1
            for name,value in d['usage'].items():tokens[name]+=value
            if d['admissible_complete_answer']:
                text=a['response']['choices'][0]['message']['content']
                rid=digest({'request':event['request'],'revision':event['revision'],'answer':text})
                expected_items[rid]={'review_id':rid,'question':event['request']['messages'][-1]['content'],
                    'reference':event['reference'],'answer':text,'correct':None,'supported':None,'reviewer':None,'notes':''}
                expected_mapping.append({'run_id':run,'ordinal':a['ordinal'],'review_id':rid,'response_sha256':d['response_sha256']})
        saved=summary['runs'][run]
        if saved['saved_attempts']!=len(rows) or saved['admissible_complete_responses']!=counters['complete_responses']:raise ValueError('Summary mismatch')
        counts[run]={**dict(counters),'incomplete_responses':counters['incomplete_responses'],'registered_attempt_cap':saved['registered_attempt_cap'],'observed_tokens':dict(tokens)}
    if len(seen)!=len(attempts) or summary['combined_saved_attempts']!=len(attempts):raise ValueError('Unknown run or attempt count mismatch')
    if len(packet['items'])!=len(expected_items) or {i['review_id']:i for i in packet['items']}!=expected_items:raise ValueError('Review bindings or blank labels changed')
    if mapping!=expected_mapping:raise ValueError('Private review mapping mismatch')
    return {'evidence_kind':'uploaded_live_journal_export_verified_offline','archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        'manifest_verified':True,'request_response_and_review_bindings_verified':True,
        'runs':counts,'combined_saved_attempts':len(attempts),'unique_blank_review_items':len(expected_items),
        'independent_reviews':0,'external_model_calls_during_audit':0,'live_latency_valid':False,
        'paired_comparison_complete':False,'paired_savings':None,'quality_result':None,
        'limitations':['Archive is user-supplied; manifest consistency is not independent proof of provider execution.',
                      'No raw SQLite journal was uploaded; its original snapshot digest is not independently recomputed.',
                      'No cache-arm attribution or valid live timing is inferred. Profiles must remain separate.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.archive,args.corpus);write_new(args.output,result);print(json.dumps(result,indent=2))
