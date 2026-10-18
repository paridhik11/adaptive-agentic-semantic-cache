"""Offline documentation audit: saved arithmetic, historical preservation, links.

Does not run inference, a benchmark/replay, or certify security/claim semantics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='4fbe298986427e92fb2404b0c9fd3ad770b8206f'

def original(path):
    return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT)

def checkout_original(path, *, checkout_path=None):
    """Baseline bytes as Git would check them out here (EOL/attributes honored).

    Raw Git blob bytes remain the provenance hash. We do not strip whitespace,
    reserialize JSON, or normalize arbitrary content to hide actual edits.
    """
    return subprocess.check_output(
        ['git','cat-file','--filters',f'--path={checkout_path or path}',f'{BASE}:{path}'],cwd=ROOT)

def verify():
    files={'data/load_test_telemetry.json':20,'data/scaled_load_test_telemetry_corrected.json':1,'data/phase6_telemetry.json':1}
    timing=[]
    for path,n in files.items():
        raw=(ROOT/path).read_bytes();assert raw==checkout_original(path),f'historical telemetry changed beyond Git checkout conversion: {path}'
        rows=json.loads(raw)
        auto=[r['timings_ms']['total_pipeline_ms'] for r in rows if r.get('tier')=='AUTO_REUSE']
        judge=[r['timings_ms']['total_pipeline_ms'] for r in rows if r.get('tier')=='AMBIGUOUS']
        assert len(auto)==n
        timing.append({'file':path,'sha256':hashlib.sha256(original(path)).hexdigest(),'checkout_sha256':hashlib.sha256(raw).hexdigest(),'workload_events':len(rows),'auto_reuse_n':len(auto),'ambiguous_n':len(judge),
                       'auto_total_pipeline_mean_ms':statistics.mean(auto),'ambiguous_total_pipeline_mean_ms':statistics.mean(judge),
                       'historical_component_reduction_percent':100*(1-statistics.mean(auto)/statistics.mean(judge)),
                       'baseline':'AMBIGUOUS total pipeline including remote judge; not fresh generation','evidence_kind':'historical_component_telemetry_recomputed','sampling_interval_for_reduction':None})
    preserved=[]
    for source,snapshot in [('README.md','README.before_claim_hygiene.md'),('DESIGN.md','DESIGN.before_claim_hygiene.md'),('docs/RESUME_BULLETS.md','RESUME_BULLETS.before_claim_hygiene.md')]:
        target=ROOT/'docs/archive'/snapshot
        assert target.read_bytes().endswith(checkout_original(source,checkout_path=str(target.relative_to(ROOT)).replace(chr(92),chr(47)))),f'snapshot body changed: {source}'
        preserved.append(str(target.relative_to(ROOT)))
    for name in ('STATUS.md','RECOVERY_STATUS.md','MASTER_REPORT.md','comprehensive_project_status_report.md'):
        path='docs/'+name;assert (ROOT/path).read_bytes().endswith(checkout_original(path)),f'historical report body changed: {path}'
        assert (ROOT/path).read_text(encoding='utf-8').startswith('> SUPERSEDED');preserved.append(path)
    html='docs/comprehensive_project_status_report.html'
    current=(ROOT/html).read_text(encoding='utf-8');before=original(html).decode('utf-8').replace('\r\n','\n')
    assert re.sub(r'(<body>)\n<aside.*?</aside>',r'\1',current,count=1,flags=re.S)==before
    preserved.append(html)
    # All source/runtime and R5 recorded artifacts retain baseline bytes.
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    frozen=[p for p in paths if p.startswith(('src/','data/','docs/audit/R5_'))]
    for path in frozen:assert (ROOT/path).read_bytes()==checkout_original(path),f'frozen runtime/data/artifact changed: {path}'
    targets=['README.md','DESIGN.md','AGENTS.md','docs/RESUME_BULLETS.md','docs/CURRENT_STATUS.md','docs/audit/CLAIM_HYGIENE_AUDIT.md','docs/audit/R6_AGENT_CACHE_PREREG.md']
    links=0
    for path in targets:
        content=(ROOT/path).read_text(encoding='utf-8')
        for match in re.finditer(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',content):
            target=match.group(1)
            if '://' in target or target.startswith('#'):continue
            assert ((ROOT/path).parent/target.split('#')[0]).exists(),f'broken local link in {path}'
            links+=1
    pattern=re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|sk-or-v1-[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    # Scan changed docs/scripts, including new untracked text files, not secrets/env.
    candidates=set(subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines())
    candidates.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines())
    scanned=[]
    for path in sorted(candidates):
        file=ROOT/path
        if file.is_file() and (file.suffix in ('.md','.html','.py','.json') or path=='AGENTS.md'):
            assert not pattern.search(file.read_text(encoding='utf-8')),f'targeted secret-pattern finding in {path}'
            scanned.append(path)
    return {'schema_version':'claim-hygiene-validation-v1','base_commit':BASE,'runtime_data_r5_artifacts_unchanged':True,'comparison_policy':'Baseline converted using git cat-file --filters --path; raw Git blob hashes retained; no benchmark file rewritten.',
            'frozen_files_checked':len(frozen),'historical_bodies_preserved':preserved,'local_links_checked':links,
            'timing_recomputations':timing,'external_model_calls':0,'new_benchmark_or_replay_runs':0,
            'redesign_implemented':False,'redesign_evaluation_run':False,'targeted_secret_pattern_findings':0,
            'scan_scope':scanned,'scan_limit':'Targeted token/private-key patterns, not a full secret scanner or security certification.',
            'tests':'Documentation-only change; full runtime suite not rerun. Project venv unavailable; standard-library checks run with system Python.',
            'initial_check_failures':['Project .venv executable unavailable; used system Python for standard-library audit.','Expanded numeric scan exposed missing ledger entry for historical 620-test result; added evidence row from R5_VALIDATION.json, then passed.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=verify()
    if args.output:
        with args.output.open('x',encoding='utf-8') as output:json.dump(result,output,indent=2);output.write('\n')
    else:print(json.dumps(result,indent=2))
