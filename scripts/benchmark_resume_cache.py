"""Measure real local exact-cache overhead; no simulated LLM latency comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision
from src.policy.sqlite_store import SQLiteResponseStore


def percentile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p
    low=int(position);high=min(low+1,len(ordered)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('refusing to overwrite evidence')
    root=Path(__file__).resolve().parents[1]
    reference=json.loads((root/'src/api/demo_reference.json').read_text())
    bodies=[{'model':'reference-excerpt','temperature':0,'max_tokens':1024,
             'messages':[{'role':'user','content':d['title']}]} for d in reference['documents']]
    answers={d['title']:d['text'] for d in reference['documents']}
    def answer(body):
        return {'model':body['model'],'choices':[{'finish_reason':'stop','message':{
            'role':'assistant','content':answers[body['messages'][0]['content']]}}]}
    config={'runs':5,'hits_per_run':1000,'questions':len(bodies),'concurrent_requests':64,
            'timed_scope':'ScopedResponseCache.process including validation/serialization; excludes API/model/network',
            'provider':'deterministic bundled source-excerpt stub; no sleeps or fake LLM timing',
            'seed_calls_excluded_from_timing':True,'threshold_search':False}
    results=[]
    with tempfile.TemporaryDirectory() as temporary:
        for mode in ['memory','sqlite']:
            def build(name='latency'):
                store=SQLiteResponseStore(Path(temporary)/(mode+name+'.db'),namespace='resume-benchmark-v1') if mode=='sqlite' else None
                return ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'fixed documentation fixture'),storage=store)
            cache=build();observed=[0]
            def provider(body):observed[0]+=1;return answer(body)
            for body in bodies:assert cache.process(body,tenant_id='benchmark',provider=provider).status=='MISS'
            samples=[];runs=[]
            for run in range(config['runs']):
                if mode=='sqlite':cache=build()  # new instance, same durable source
                times=[]
                for i in range(config['hits_per_run']):
                    start=time.perf_counter_ns()
                    result=cache.process(bodies[i%len(bodies)],tenant_id='benchmark',provider=provider)
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    assert result.status=='HIT' and result.provider_calls==0
                    times.append(elapsed)
                samples+=times
                runs.append({'run':run+1,'p50_ms':percentile(times,.5),'p95_ms':percentile(times,.95),'samples_ms':times})
            assert observed[0]==len(bodies)
            # Separate cold wave: synchronize arrival and hold the counted stub
            # until all callers have entered. Timing is deliberately not reported.
            cold=build('concurrency');barrier=threading.Barrier(65);release=threading.Event()
            entered=[0];calls=[0];lock=threading.Lock();all_entered=threading.Event()
            def blocked(body):
                with lock:calls[0]+=1
                assert release.wait(20)
                return answer(body)
            def request(_):
                barrier.wait(timeout=20)
                with lock:
                    entered[0]+=1
                    if entered[0]==64:all_entered.set()
                return cold.process(bodies[0],tenant_id='benchmark',provider=blocked)
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures=[pool.submit(request,i) for i in range(64)]
                barrier.wait(timeout=20);assert all_entered.wait(20);release.set()
                completed=[f.result(timeout=20) for f in futures]
            assert calls[0]==1 and sum(r.provider_calls for r in completed)==1
            results.append({'storage':mode,'timed_hits':len(samples),'seed_stub_calls':observed[0],
                'p50_ms':percentile(samples,.5),'p95_ms':percentile(samples,.95),
                'run_p95_range_ms':[min(r['p95_ms'] for r in runs),max(r['p95_ms'] for r in runs)],
                'runs':runs,'cold_concurrent_fixture':{'requests':64,'counted_stub_invocations':calls[0],
                    'misses':sum(r.status=='MISS' for r in completed),'hits':sum(r.status=='HIT' for r in completed)}})
    result={'evidence_type':'measured local cache component latency plus constructed concurrency fixture',
        'config':config,'hardware':{'platform':platform.platform(),'logical_cpus':os.cpu_count(),
           'python':platform.python_version(),'filesystem':'temporary directory in this managed container'},
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['src/policy/response_cache.py','src/policy/sqlite_store.py','scripts/benchmark_resume_cache.py']},
        'results':results,'live_provider_calls':0,'live_cost_or_latency_savings_measured':False,
        'limitations':['Three warm exact keys only; deliberately repetitive workload.',
            'Single local process; concurrency coordination is not distributed.',
            'No actual LLM answers generated or semantic acceptance assessed.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='runs'} for row in results],indent=2))


if __name__=='__main__':main()
