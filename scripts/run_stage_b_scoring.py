"""Resume local scoring with three bounded per-model CPU workers; no new models."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','calibration'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    logs=ROOT/'artifacts/stage_b'/f'{args.split}-workers'
    logs.mkdir(parents=True,exist_ok=True)
    state={'evidence_type':'operational_scoring_run_not_accuracy','split':args.split,
           'max_concurrent_workers':3,'torch_threads_per_worker':2,'events':[],
           'policy_models_data_thresholds_changed':False,'remote_inference_calls':0}
    status=logs/f'run-{time.time_ns()}.json'
    mutex=threading.Lock()
    def event(item):
        with mutex:
            state['events'].append(item)
            temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,status)
            print(json.dumps(item),flush=True)
    base=[sys.executable,str(ROOT/'scripts/score_stage_b_candidates.py'),'--split',args.split,
          '--output-dir',str(args.output_dir.resolve()),'--resume']
    with (logs/'prepare.txt').open('a') as output:
        event({'stage':'prepare_started'})
        subprocess.run(base+['--prepare-only'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
        event({'stage':'prepare_complete'})
    def worker(key):
        with (logs/(key+'.txt')).open('a') as output:
            process=subprocess.Popen(base+['--model-key',key],cwd=ROOT,stdout=output,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            event({'stage':'model_started','key':key,'pid':process.pid})
            result=process.wait()
            event({'stage':'model_finished','key':key,'pid':process.pid,'returncode':result})
            if result:raise RuntimeError('Worker failed; preserved checkpoint and log for '+key)
            return key
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,key) for key in ['redis_reranker','quora','11','29','47']]
        for future in as_completed(futures):future.result()
    with (logs/'finalize.txt').open('a') as output:
        subprocess.run(base,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    event({'stage':'complete','receipt':str(args.output_dir/'receipt.json')})


if __name__=='__main__':main()
