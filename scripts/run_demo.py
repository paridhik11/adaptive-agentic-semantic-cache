"""Launch the local integrated workbench; reference replay requires no key."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.api.demo import create_demo_app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--directory',type=Path,default=Path('runs/workbench'))
    p.add_argument('--live',action='store_true',help='Explicit free-provider mode; new fixed 12-attempt demo allowance.')
    p.add_argument('--journal',type=Path,help='Existing shared journal, required in live mode.')
    args=p.parse_args()
    if not 1024<=args.port<=65535:p.error('Port must be between 1024 and 65535')
    if args.live and not args.journal:p.error('Live mode requires your existing --journal path')
    key=os.environ.get('OPENROUTER_API_KEY') if args.live else None
    try:app=create_demo_app(args.directory/('live' if args.live else 'reference'),live=args.live,journal_path=args.journal,api_key=key)
    except ValueError as exc:p.error(str(exc))
    import uvicorn
    print('Open http://127.0.0.1:'+str(args.port),flush=True)
    print('LIVE FREE MODEL — shared journal, fixed 12-attempt allowance.' if args.live else 'REFERENCE REPLAY — scripted answers; real tools, cache and approval; no external model calls.',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)

if __name__=='__main__':main()
