"""Freeze, then execute the preregistered R6 offline engineering criterion once.

No live transport. Source-reference replay is not model inference or safety gold.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.agents.document_agent import DocumentAgent,DocumentTools
from src.api.service import GatewayService
from src.policy.response_cache import ScopedResponseCache,EligibilityDecision
from src.benchmark.paired_runtime import AgentReferenceReplay
from src.benchmark.schema import digest,read_json,write_new

ROOT=Path(__file__).resolve().parents[1]
CORPUS='0fe8140b1d3bb1bf752821d15e863b5552db324d17ed373b89952837620a7089'
PROTOCOL='docs/audit/R6_AGENT_CACHE_PREREG.md'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_hashes():
    files=[p for p in (ROOT/'src').rglob('*.py')]
    files += [ROOT/'scripts/run_r6_agent_stage.py',ROOT/'scripts/demo_document_agent.py',ROOT/'pyproject.toml']
    files += list((ROOT/'tests').glob('test_*document*.py'))+[ROOT/'tests/test_r6_criterion.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}

def environment():
    return {'python':sys.version,'platform':platform.platform(),
            'pytest':importlib.metadata.version('pytest')}

def select(corpus):
    if digest(corpus)!=CORPUS:raise ValueError('pinned corpus content mismatch')
    items=[r for r in corpus['items'] if r['split']=='dev']
    if len(items)!=22:raise ValueError('development question count changed')
    return items

def expected_texts(items,archive):
    expected=[r['reference_answer'] for r in items for _ in range(2)]
    if archive is None:
        return expected,'reconstructed from pinned source references; original raw baseline unavailable'
    recorded=read_json(ROOT/'docs/audit/R5_VALIDATION.json')['local_only_archive_sha256']
    if sha(archive)!=recorded:raise ValueError('baseline archive checksum mismatch')
    with zipfile.ZipFile(archive) as bundle:
        manifest=json.loads(bundle.read('manifest.json'))
        raw=bundle.read('agent-replay.json')
        if hashlib.sha256(raw).hexdigest()!=manifest['files']['agent-replay.json']:raise ValueError('baseline trace manifest mismatch')
        baseline=json.loads(raw)
    records=baseline['records']
    ids=[(r['faq_id'],kind) for r in items for kind in ('seed','constructed_exact_repeat')]
    if [(r['faq_id'],r['kind']) for r in records]!=ids:raise ValueError('baseline task order differs')
    for record,text in zip(records,expected):
        for mode in ('off','exact'):
            if record['arms'][mode]['status']!='COMPLETED' or record['arms'][mode]['answer']!=text:
                raise ValueError('archived baseline answer/status differs from reference replay')
    return [r['arms']['off']['answer'] for r in records],'verified raw R5 archive and source references'

def assess(records,expected,isolation_tests_passed):
    checks={'all_tasks_present':len(records)==44 and len(expected)==44,
            'complete_and_equal_answers':True,'original_baseline_text_match':True,
            'known_reconciled_provider_counts':True,'off_calls_unchanged':False,
            'strictly_fewer_provider_calls':False,'tool_work_unchanged':False,
            'isolation_and_workflow_regressions_passed':isolation_tests_passed is True}
    totals={mode:{'provider_calls':0,'tool_calls':0} for mode in ('off','exact')}
    for index,record in enumerate(records):
        a,b=record['arms']['off'],record['arms']['exact']
        if a['status']!='COMPLETED' or b['status']!='COMPLETED' or not isinstance(a['answer'],str) or not a['answer'] or a['answer']!=b['answer']:
            checks['complete_and_equal_answers']=False
        if index>=len(expected) or a['answer']!=expected[index] or b['answer']!=expected[index]:checks['original_baseline_text_match']=False
        for mode in totals:
            value=record['arms'][mode]
            if type(value['provider_calls']) is not int or value['provider_calls']<0 or value['provider_calls']!=value['observed_provider_calls']:
                checks['known_reconciled_provider_counts']=False
            totals[mode]['provider_calls']+=value['observed_provider_calls']
            totals[mode]['tool_calls']+=value['observed_tool_calls']
    checks['off_calls_unchanged']=totals['off']['provider_calls']==132
    checks['strictly_fewer_provider_calls']=totals['exact']['provider_calls']<totals['off']['provider_calls']
    checks['tool_work_unchanged']=totals['off']['tool_calls']==totals['exact']['tool_calls']==88
    return {'decision':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'totals':totals}

def freeze(args):
    corpus=read_json(args.corpus);items=select(corpus)
    _,baseline=expected_texts(items,args.baseline_archive)
    focus=args.focused_tests.read_text();full=args.full_tests.read_text()
    if ' passed' not in focus or ' passed' not in full or 'FAILED' in focus+full or 'ERROR' in focus+full:
        raise ValueError('passing regression evidence required')
    args.run_dir.mkdir(parents=True,exist_ok=False)
    value={'schema_version':'r6-stage-freeze-v1','protocol_sha256':sha(ROOT/PROTOCOL),
        'corpus_sha256':digest(corpus),'runtime_source_sha256':code_hashes(),'environment':environment(),
        'workload':[(r['faq_id'],kind) for r in items for kind in ('seed','constructed_exact_repeat')],
        'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'baseline_archive_sha256':sha(args.baseline_archive) if args.baseline_archive else None,'baseline_comparison':baseline,
        'test_evidence':{'focused':focus,'full':full,'focused_sha256':sha(args.focused_tests),'full_sha256':sha(args.full_tests)}}
    write_new(args.run_dir/'freeze.json',value)
    print('Frozen before execution:',args.run_dir/'freeze.json')

class CountedProvider:
    mode='source_reference_replay'
    def __init__(self,items):self.provider=AgentReferenceReplay(items);self.calls=0
    def __call__(self,body):self.calls+=1;return self.provider(body)

class CountedTools(DocumentTools):
    def __init__(self,documents):super().__init__(documents);self.calls=0
    def dispatch(self,*args):self.calls+=1;return super().dispatch(*args)

def execute(args):
    frozen=read_json(args.run_dir/'freeze.json');corpus=read_json(args.corpus);items=select(corpus)
    if frozen['runtime_source_sha256']!=code_hashes() or frozen['protocol_sha256']!=sha(ROOT/PROTOCOL) or frozen['environment']!=environment():
        raise ValueError('BLOCKED: code/protocol/environment differs from freeze; do not run')
    expected,comparison=expected_texts(items,args.baseline_archive)
    if frozen['baseline_archive_sha256']!=(sha(args.baseline_archive) if args.baseline_archive else None):raise ValueError('baseline selection changed')
    directory=args.run_dir/'evaluation';directory.mkdir(exist_ok=False)
    # Exclusive marker is created before any stage. No implicit retry of a run.
    write_new(directory/'started.json',{'freeze_sha256':sha(args.run_dir/'freeze.json'),'execution':'offline-source-reference-replay'})
    agents={};providers={};tools={};records=[]
    try:
        for mode in ('off','exact'):
            providers[mode]=CountedProvider(items)
            tools[mode]=CountedTools({r['faq_id']:r['reference_answer'] for r in items})
            service=GatewayService(ScopedResponseCache(eligibility=lambda _:EligibilityDecision(True,'source_replay_fixture')),providers[mode],allowed_models={'fixture-model'})
            agents[mode]=DocumentAgent(service,tools[mode],directory/(mode+'.sqlite'),model='fixture-model')
        for item in items:
            for kind in ('seed','constructed_exact_repeat'):
                row={'faq_id':item['faq_id'],'kind':kind,'arms':{}}
                for mode,agent in agents.items():
                    before=providers[mode].calls;tool_before=tools[mode].calls
                    tid=agent.create(item['question'],tenant='agent-faq',principal='benchmark-user',require_approval=False,cache=mode!='off',static_document_stage=True)
                    for _ in range(30):
                        state=agent.advance(tid,tenant='agent-faq',principal='benchmark-user')
                        if state['status'] in ('COMPLETED','AWAITING_APPROVAL','FAILED','REJECTED'):break
                    row['arms'][mode]={'status':state['status'],'answer':state['answer'],
                        'answer_sha256':digest(state['answer']),'provider_calls':state['provider_calls'],
                        'observed_provider_calls':providers[mode].calls-before,'observed_tool_calls':tools[mode].calls-tool_before,'trace':state['trace']}
                records.append(row)
        report=assess(records,expected,True)
        report.update({'schema_version':'r6-agent-stage-result-v1','evidence_kind':'source-reference-replay-development',
            'tasks_per_arm':len(records),'baseline_comparison':comparison,'freeze_sha256':sha(args.run_dir/'freeze.json'),
            'external_model_calls':0,'independent_reviews':0,'semantic_quality_measured':False,'live_savings_measured':False})
        write_new(directory/'trace.private.json',{'records':records})
        write_new(directory/'result.json',report)
    except Exception as exc:
        write_new(directory/'failure.json',{'decision':'FAIL','error':type(exc).__name__,'completed_records':len(records),'note':'Criterion run failed; preserve and do not silently rerun.'})
        raise
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','run']);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--baseline-archive',type=Path);p.add_argument('--focused-tests',type=Path);p.add_argument('--full-tests',type=Path)
    args=p.parse_args()
    if args.command=='freeze':
        if not args.focused_tests or not args.full_tests:p.error('passing test logs required before freeze')
        freeze(args)
    else:execute(args)
