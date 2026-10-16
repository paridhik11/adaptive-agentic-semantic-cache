"""Read saved pilot response metadata. No key access, network calls or DB writes."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.diagnostics import response_diagnostics
from src.benchmark.schema import _constant,_unique,write_new


def inspect(journal,run_id):
    path=Path(journal).resolve()
    if not path.is_file():raise ValueError('journal file not found')
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    try:
        rows=db.execute('SELECT ordinal,request_sha,state,payload FROM pilot_calls WHERE run=? ORDER BY ordinal',(run_id,)).fetchall()
    finally:db.close()
    if not rows:raise ValueError('run ID not found')
    calls=[]
    for ordinal,request_sha,state,payload in rows:
        item={'ordinal':ordinal,'request_sha256':request_sha,'state':state}
        if state=='complete':
            try:
                response=json.loads(payload,parse_constant=_constant,object_pairs_hook=_unique)
                item['response_diagnostics']=response_diagnostics(response,{'n':1})
            except (ValueError,TypeError):item['diagnostic_error']='invalid_saved_json'
        elif state=='error':
            # The error state already contains only a sanitized code/status.
            item['diagnostic_error']='saved_provider_error'
        else:item['diagnostic_error']='uncertain_attempt'
        calls.append(item)
    return {'schema_version':'pilot-diagnostics-v1','run_id':run_id,'model_calls_made':0,
        'note':'Read-only journal inspection; answer and reasoning text are omitted. Pilot protocol assumes n=1.',
        'calls':calls}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal',type=Path,default=Path('runs/openrouter-free.sqlite'))
    p.add_argument('--run-id',required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output and args.output.resolve()==args.journal.resolve():p.error('output must differ from journal')
    result=inspect(args.journal,args.run_id)
    if args.output:write_new(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
