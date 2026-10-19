"""Explicit R5 development-profile revision; old attempts remain budgeted."""
from pathlib import Path
import json
import sqlite3

from src.benchmark.diagnostics import response_diagnostics
from src.benchmark.schema import digest

PARENT_RUN = 'r5-live-v1'
REVISED_RUN = 'r5-live-v2'
TOTAL_ATTEMPT_CAP = 16
MAX_OUTPUT_TOKENS = 4096
CONCISE_INSTRUCTION = ('Give a concise, direct answer in at most 180 words. '
                       'Include only details needed to answer the question; do not reproduce the reference. ')


def predecessor_snapshot(path, model):
    """Read only. No answer text is exported; every predecessor attempt counts."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Existing shared journal required; do not create a new quota ledger')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        run = db.execute('SELECT fingerprint,max_calls FROM pilot_runs WHERE run=?', (PARENT_RUN,)).fetchone()
        rows = db.execute('SELECT ordinal,request_sha,started,state,payload FROM pilot_calls WHERE run=? ORDER BY ordinal', (PARENT_RUN,)).fetchall()
    finally:
        db.close()
    if not run or run[1] != TOTAL_ATTEMPT_CAP or not rows:
        raise ValueError('Expected the existing capped r5-live-v1 run with saved attempts')
    if [r[0] for r in rows] != list(range(len(rows))) or any(r[3] != 'complete' for r in rows):
        raise ValueError('Predecessor contains uncertain/failed attempts; manual inspection required')
    diagnostics = [response_diagnostics(json.loads(r[4]), {'n': 1}) for r in rows]
    if (any(not d['admissible_complete_answer'] for d in diagnostics[:-1])
            or diagnostics[-1]['rejection_reason'] != 'output_token_limit'
            or diagnostics[-1]['model'] != model):
        raise ValueError('Expected a final token-limit response from the same model; inspect predecessor')
    remaining = TOTAL_ATTEMPT_CAP - len(rows)
    if remaining < 1:
        raise ValueError('Combined development attempt allowance exhausted')
    return {'run_id': PARENT_RUN, 'fingerprint': run[0], 'snapshot_sha256': digest(rows),
            'attempts': len(rows), 'remaining_attempt_allowance': remaining,
            'combined_attempt_cap': TOTAL_ATTEMPT_CAP, 'response_diagnostics': diagnostics,
            'evidence_kind': 'saved_live_predecessor_diagnostics', 'latency_valid': False}


def revise_events(events):
    import copy
    revised = copy.deepcopy(events)
    for event in revised:
        body = event['request']
        body['max_tokens'] = MAX_OUTPUT_TOKENS
        body['messages'][0]['content'] = CONCISE_INSTRUCTION + body['messages'][0]['content']
    return revised


def pause_guidance(pause):
    resumable = pause.reason in {'project_minute_quota', 'project_daily_quota', 'upstream_rate_limit_pause'}
    return {'resume_allowed_after_retry_time': resumable,
            'note': ('Wait until retry_at, then resume this exact command; reconstructed latency is invalid.'
                     if resumable else 'Stop. Preserve the journal and share these diagnostics. '
                     'Rerunning a saved incomplete response will not repair it; do not change run IDs or reset quota.')}
