"""Inspect checkpoint progress AND active scorer processes; no model calls."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/stage_b/calibration'))
    args=parser.parse_args()
    import numpy as np
    preparation=args.output_dir/'preparation.json'
    totals=json.loads(preparation.read_text()) if preparation.exists() else {}
    progress=[]
    for path in sorted((args.output_dir/'checkpoints').glob('*.npz')):
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
        kind=metadata['binding']['kind']
        total=totals.get('unique_pairs' if kind=='pairs' else 'unique_texts')
        progress.append({'key':metadata['binding']['key'],'kind':kind,'done':metadata['done'],
            'total':total,'checkpoint_complete':metadata['done']==total if total else None,
            'seconds_since_checkpoint':round(time.time()-path.stat().st_mtime,1)})
    active=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            words=[x.decode(errors='replace') for x in path.read_bytes().split(b'\0') if x]
            if not any(x.endswith('/score_stage_b_candidates.py') or x=='scripts/score_stage_b_candidates.py' for x in words):continue
            if '--output-dir' not in words:continue
            out=Path(words[words.index('--output-dir')+1]).resolve()
            if out!=args.output_dir.resolve():continue
            key=words[words.index('--model-key')+1] if '--model-key' in words else 'prepare_or_finalize'
            active.append({'pid':int(path.parent.name),'key':key})
        except (OSError,ValueError,IndexError):pass
    print(json.dumps({'evidence_type':'operational_progress_not_model_accuracy',
        'completed_receipt_present':(args.output_dir/'receipt.json').exists(),
        'active_scorer_processes':active,'checkpoints':progress},indent=2))


if __name__=='__main__':main()
