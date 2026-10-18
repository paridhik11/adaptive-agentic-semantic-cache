"""Reconcile declared human R5 labels against immutable actual-answer bindings."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import read_json,write_new
from src.benchmark.paired_runtime import reconcile_reviews
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trace',type=Path,required=True);p.add_argument('--reviews',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
if __name__=='__main__':
    args=p.parse_args();write_new(args.output,reconcile_reviews(read_json(args.trace),read_json(args.reviews)))
