"""Check an installed wheel through real loopback HTTP, outside the source tree.

Run with the clean environment's Python, passing a NEW JSON receipt path.
No provider calls or pretrained model loads are performed.
"""
import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new receipt path')
    with tempfile.TemporaryDirectory() as directory:
        installed = json.loads(subprocess.check_output([sys.executable, '-c',
            'import src.api.demo,json; print(json.dumps({"module":src.api.demo.__file__}))'],
            cwd=directory, text=True))
        module = Path(installed['module']).resolve()
        if not module.is_relative_to(Path(sys.prefix).resolve()) or 'site-packages' not in module.parts:
            raise RuntimeError('Expected a wheel installed in this Python environment')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        launch = ('from src.api.demo import create_demo_app; import uvicorn; '
            'app=create_demo_app("profile",whole_answer_cache=True,task_memory=True); '
            f'uvicorn.run(app,host="127.0.0.1",port={port},access_log=False)')
        with open(Path(directory)/'server.log', 'w+') as log:
            server = subprocess.Popen([sys.executable, '-c', launch], cwd=directory,
                stdout=log, stderr=log)
            base = f'http://127.0.0.1:{port}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                for _ in range(100):
                    try:
                        with opener.open(base, timeout=1) as response:
                            html = response.read().decode()
                        break
                    except OSError:
                        if server.poll() is not None:
                            log.seek(0)
                            raise RuntimeError(log.read())
                        time.sleep(.1)
                else:
                    raise RuntimeError('Demo readiness timed out')
                token = re.search(r'name="demo-session" content="([^"]+)"', html).group(1)

                def call(path, data=None, method=None):
                    request = urllib.request.Request(base+path,
                        data=None if data is None else json.dumps(data).encode(), method=method,
                        headers={'x-demo-session': token, 'Content-Type': 'application/json'})
                    with opener.open(request, timeout=10) as response:
                        return json.load(response)

                for asset in ('workbench.js', 'workbench.css'):
                    with opener.open(base+'/assets/'+asset, timeout=5) as response:
                        assert response.status == 200 and response.read()

                def task():
                    state = call('/api/tasks', {'question': 'How do I convert between tuples and lists?'})
                    for _ in range(16):
                        if state['status'] == 'AWAITING_APPROVAL':
                            return state
                        state = call('/api/tasks/'+state['id']+'/advance', {})
                    raise AssertionError('Approval state not reached')

                first = task()
                assert first['observed_attempts'] == 3
                call('/api/tasks/'+first['id']+'/decision', {'decision': 'approve'})
                repeat = task()
                assert repeat['observed_attempts'] == 0 and repeat['answer_cache_hit']
                assert repeat['answer'] is None  # fresh approval still required
                call('/api/tasks/'+repeat['id']+'/decision', {'decision': 'approve'})
                assert len(call('/api/memory')['items']) == 1
                call('/api/memory', method='DELETE')
                assert not call('/api/memory')['items']
                receipt = {'evidence_type': 'installed-wheel real HTTP offline reference fixture',
                    'installed_module': str(module), 'python': sys.version,
                    'source_tree_imported': False, 'observed_scripted_calls': [3, 0],
                    'fresh_approval_required': True, 'packaged_web_assets': True,
                    'memory_list_and_clear': True, 'browser_visual_qa': False,
                    'live_inference': False}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('x') as output:
                    output.write(json.dumps(receipt, indent=2)+'\n')
                print(json.dumps(receipt, indent=2))
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == '__main__':
    main()
