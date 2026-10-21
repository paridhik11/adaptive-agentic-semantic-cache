"""Atomic numeric inference checkpoints, identity-bound and pickle-free."""
import json
import os
from pathlib import Path


def load(path, binding):
    import numpy as np
    path=Path(path)
    if not path.exists():return [], {'batch_ms':[], 'load_seconds':[]}
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        data=archive['data']
    if metadata['binding']!=binding:
        raise ValueError('STOP: checkpoint inputs/model/config differ; never reuse stale scores')
    if len(data)!=metadata['done'] or not np.isfinite(data).all():
        raise ValueError('STOP: invalid checkpoint prefix')
    return data.tolist(),metadata['timings']


def save(path,binding,data,timings):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    metadata={'binding':binding,'done':len(data),'timings':timings,'format':'numeric numpy arrays, allow_pickle=False'}
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream,data=np.asarray(data,dtype=np.float32),metadata=np.asarray(json.dumps(metadata)))
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
