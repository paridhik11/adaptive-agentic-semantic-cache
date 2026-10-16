"""Compare independent human reviews; disagreement is never silently majority-voted."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import read_json,write_new
from src.benchmark.review_agreement import reconcile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trace','first','second','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--consensus-output',type=Path)
    args=p.parse_args()
    if args.output.exists() or (args.consensus_output and args.consensus_output.exists()):p.error('choose new output paths')
    if args.consensus_output and args.output.resolve()==args.consensus_output.resolve():p.error('output paths must differ')
    result=reconcile(read_json(args.trace),read_json(args.first),read_json(args.second))
    write_new(args.output,result)
    if args.consensus_output:write_new(args.consensus_output,result['consensus_reviews'])
    print('Unresolved disagreements:',len(result['conflicts']))


if __name__=='__main__':main()
