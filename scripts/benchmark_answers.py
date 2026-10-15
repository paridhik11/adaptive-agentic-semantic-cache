"""Run development evidence, export answer review, or recompute an audited report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.providers import FixtureChatProvider, HTTPChatProvider
from src.benchmark.schema import digest, read_json, write_new
from src.benchmark.runner import run_workload
from src.benchmark.report import review_template, summarize
from src.benchmark.views import review_html


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--workload', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True, help='new directory; never overwrite evidence')
    run.add_argument('--provider', choices=['fixture','http'], default='fixture')
    run.add_argument('--allow-live-calls', action='store_true', help='explicit opt-in: fresh baseline adds provider calls')
    run.add_argument('--max-live-calls', type=int, default=0, help='must cover two calls per development request')
    report = commands.add_parser('report')
    report.add_argument('--trace', type=Path, required=True)
    report.add_argument('--reviews', type=Path)
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'report':
        value = summarize(read_json(args.trace), read_json(args.reviews) if args.reviews else None)
        write_new(args.output, value)
        print('Saved answer-bound report:', args.output)
        return
    from src.benchmark.schema import validate_workload
    workload = validate_workload(read_json(args.workload))
    if args.provider == 'http':
        upper = 2 * sum(r['split']=='dev' for r in workload['requests'])
        if not args.allow_live_calls or args.max_live_calls < upper:
            parser.error('live run requires explicit opt-in and a call budget covering ' + str(upper) + ' calls')
        provider = HTTPChatProvider(os.environ['UPSTREAM_CHAT_URL'], api_key=os.environ.get('UPSTREAM_API_KEY'))
    else:
        provider = FixtureChatProvider()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        tree = subprocess.check_output(['git','rev-parse','HEAD^{tree}'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], cwd=Path(__file__).resolve().parents[1], text=True).strip())
    except (OSError, subprocess.SubprocessError):
        revision, tree, dirty = None, None, None
    trace = run_workload(workload, provider)
    trace['code_revision'], trace['code_tree'], trace['working_tree_dirty'] = revision, tree, dirty
    trace['python_version'] = sys.version
    trace['provider_endpoint_sha256'] = digest(provider.endpoint) if args.provider == 'http' else None
    write_new(args.output_dir/'trace.json', trace)
    template = review_template(trace)
    summary = summarize(trace)
    write_new(args.output_dir/'reviews.template.json', template)
    write_new(args.output_dir/'report.unreviewed.json', summary)
    with (args.output_dir/'review.html').open('x', encoding='utf-8') as f:
        f.write(review_html(trace, template, summary))
    write_new(args.output_dir/'manifest.json', {'workload_sha256':digest(workload),'trace_sha256':digest(trace),
        'files':{p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in args.output_dir.iterdir() if p.is_file()}})
    print('Saved', len(trace['events']), 'development requests to', args.output_dir)
    print('Mode:',trace['evaluation_mode'],'— no safety certification. Open review.html for answer review.')


if __name__ == '__main__':
    main()
