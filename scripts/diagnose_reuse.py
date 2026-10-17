"""Read an existing trace or reproduce an offline development diagnosis; no API calls."""
import argparse
import hashlib
import subprocess
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import read_json,write_new,digest
from src.benchmark.reuse_diagnosis import diagnose


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--trace',type=Path)
    source.add_argument('--pairs',type=Path)
    p.add_argument('--license-note')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output_dir.exists():p.error('output directory exists; evidence is immutable')
    if args.pairs and not args.license_note:p.error('--license-note is required with existing pairs')
    if args.trace:
        traces={'supplied-trace':read_json(args.trace)}
    else:
        from scripts.prepare_answer_workload import convert_pairs
        from src.api.providers import FixtureChatProvider
        from src.benchmark.runner import run_workload
        from src.benchmark.semantic import LexicalEncoder,AbstainingVerifier
        pairs=read_json(args.pairs)
        workload=convert_pairs(pairs,source_digest=digest(pairs),model='fixture-model',license_note=args.license_note)
        traces={policy:run_workload(workload,FixtureChatProvider(),policy=policy,
                    encoder=LexicalEncoder(),verifier=AbstainingVerifier() if policy=='verified' else None)
                for policy in ('exact','similarity','verified')}
    results={key:diagnose(trace) for key,trace in traces.items()}
    args.output_dir.mkdir(parents=True)
    for key,trace in traces.items():write_new(args.output_dir/(key+'.trace.json'),trace)
    root=Path(__file__).resolve().parents[1]
    implementation={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted((root/'src').rglob('*.py'))}
    implementation['scripts/diagnose_reuse.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    implementation['scripts/prepare_answer_workload.py']=hashlib.sha256((root/'scripts/prepare_answer_workload.py').read_bytes()).hexdigest()
    write_new(args.output_dir/'diagnosis.json',{'schema_version':'reuse-diagnosis-pack-v1',
        'implementation_sha256':implementation,'code_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip(),
        'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),'runs':results})
    print(json.dumps({key:{k:result[k] for k in ('requests','terminal_outcomes','eligibility_observed','semantic_terminal_stages','provider_work')} for key,result in results.items()},indent=2))
    print('No model API calls; labels and representative-traffic claims are not created.')


if __name__=='__main__':main()
