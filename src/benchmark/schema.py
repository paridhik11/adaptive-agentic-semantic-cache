"""Strict JSON and provenance contracts for new recovery evidence."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _constant(_):
    raise ValueError('non-finite JSON is not evidence')


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate JSON key: ' + key)
        out[key] = value
    return out


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_constant, object_pairs_hook=_unique)


def write_new(path, value):
    """Exclusive creation; existing evidence can never be silently overwritten."""
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + ' must be nonempty text')


def finite(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(field + ' must be finite and nonnegative')


def validate_workload(workload):
    if not isinstance(workload, dict) or workload.get('schema_version') != 'workload-v1':
        raise ValueError('expected workload-v1')
    for field in ('name', 'provenance', 'license'):
        nonempty(workload.get(field), field)
    if workload.get('nature') not in ('synthetic_fixture', 'natural', 'constructed_natural_queries'):
        raise ValueError('declare workload nature')
    rows = workload.get('requests')
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty requests required')
    ids, groups, texts = set(), {}, {}
    for row in rows:
        for field in ('event_id', 'tenant_id', 'source', 'group_id', 'split'):
            nonempty(row.get(field), field)
        if row['event_id'] in ids:
            raise ValueError('duplicate event_id')
        ids.add(row['event_id'])
        if row['split'] not in ('train', 'dev', 'test'):
            raise ValueError('split must be train/dev/test')
        group = row['group_id']
        if group in groups and groups[group] != row['split']:
            raise ValueError('group crosses splits')
        groups[group] = row['split']
        body = row.get('request')
        if not isinstance(body, dict) or not isinstance(body.get('messages'), list) or not body['messages']:
            raise ValueError('request messages required')
        normalized = []
        for message in body['messages']:
            nonempty(message.get('role'), 'role')
            nonempty(message.get('content'), 'content')
            normalized.append((message['role'], ' '.join(message['content'].casefold().split())))
        key = digest(normalized)
        if key in texts and texts[key] != row['split']:
            raise ValueError('normalized request crosses splits')
        texts[key] = row['split']
    return workload


def binding(event):
    """Labels bind to actual answers, not a designated pair's ground truth."""
    return {key: event[key] for key in ('event_id', 'request_sha256', 'response_sha256',
                                       'fresh_response_sha256', 'candidate_event_id')}
