"""Offline Stage B I/O and identity checks. No final-split support."""
import hashlib
import json
import os
from pathlib import Path
import platform
import socket

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'minilm': ('sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db'),
    'quora': ('cross-encoder/quora-distilroberta-base', 'f62e7a4b20b97195c2868e53ec59126df5eac743', '73fd14ad7d08f3ef30eb25841c8f4ba89e91230f48159279233b37015ccb33fb'),
    'redis_embedding': ('redis/langcache-embed-v1', '2f6ffda359ab94a9c222947bfd87ea8f85ea7e0a', 'f4d82e69a60b9ff8af6cc7605e21b38fd1f3396e41ccf390f4e75a5a3ecd2e26'),
    'redis_reranker': ('redis/langcache-reranker-v2-bce', 'f59539f4208f9005e10e45160a379441bdb42e9b', '6fddcc709be4d434efd1b0e0865e4608b280a083cc0a6772115c6607cfa8292a'),
}
FLAG = 'possible pretraining overlap with public pair data'


def identity(path, verbose=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('STOP: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    result = {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
    if verbose:
        print(json.dumps(result), flush=True)
    return result


def config():
    path = ROOT/'configs/intent_stage_b_v1.json'
    identity(path)
    return json.loads(path.read_text())


def read_split(split):
    if split not in {'train','dev','calibration'}:
        raise ValueError('STOP: final split is not supported')
    cfg = config()
    path = ROOT/cfg['data_dir']/(split+'.jsonl')
    identity(path)
    return [json.loads(x) for x in path.read_text().splitlines()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in rows))


def offline():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    def forbidden(*args, **kwargs):
        raise RuntimeError('STOP: network connections forbidden in Stage B experiments')
    socket.socket.connect = forbidden
    socket.create_connection = forbidden


def snapshot(key):
    from huggingface_hub.constants import HF_HUB_CACHE
    repo, revision, expected = SPECS[key]
    path = Path(HF_HUB_CACHE)/('models--'+repo.replace('/','--'))/'snapshots'/revision
    for file in sorted(path.rglob('*')):
        if file.is_file():
            item = identity(file)
            if file.name == 'model.safetensors' and item['sha256'] != expected:
                raise ValueError('STOP: pinned weight hash mismatch')
    if not (path/'model.safetensors').is_file():
        raise FileNotFoundError('STOP: missing approved checkpoint '+str(path))
    return path


def hardware():
    import torch
    memory = Path('/proc/meminfo').read_text().splitlines() if Path('/proc/meminfo').exists() else []
    cpu = Path('/proc/cpuinfo').read_text().splitlines() if Path('/proc/cpuinfo').exists() else []
    return {'platform': platform.platform(), 'python': platform.python_version(),
            'torch': torch.__version__, 'logical_cpus': os.cpu_count(), 'torch_threads': torch.get_num_threads(),
            'cpu_model': next((x.split(':',1)[1].strip() for x in cpu if x.startswith('model name')), 'unknown'),
            'memory_total': next((x for x in memory if x.startswith('MemTotal:')), 'unknown'), 'device': 'cpu'}
