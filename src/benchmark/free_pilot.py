"""Free-only OpenRouter transport and durable, fail-closed pilot journal.

A reservation is committed BEFORE network I/O. Failed/uncertain attempts count.
No transaction spans a model call. All project runs sharing this database share
one UTC-day/rolling-minute allowance; external account usage is not visible.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import re
import sqlite3
import time

from src.api.providers import HTTPChatProvider, ProviderError
from src.benchmark.control import RunPaused
from src.benchmark.schema import canonical, digest


class FreeOpenRouterProvider:
    mode='live'
    endpoint='https://openrouter.ai/api/v1/chat/completions'
    def __init__(self,api_key,*,models,transport=None):
        if not isinstance(api_key,str) or not api_key.strip():raise ValueError('OpenRouter key required')
        self.models=frozenset(models)
        if not self.models or any(not isinstance(m,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+:free',m) for m in self.models):
            raise ValueError('explicit :free model IDs required; automatic routing is disabled')
        self.transport=transport or HTTPChatProvider(self.endpoint,api_key=api_key)
    def __call__(self,body):
        if body.get('model') not in self.models:raise ValueError('model is not in the free-only allowlist')
        if any(k in body for k in ('provider','models','route','plugins','tools','web_search_options')):
            raise ValueError('caller-controlled routing or paid auxiliary services are unsupported')
        if body.get('stream'):raise ValueError('pilot collection is non-streaming')
        from src.api.service import validate_request
        validate_request(body,self.models,4096)
        outgoing=copy.deepcopy(body)
        outgoing['provider']={'allow_fallbacks':False,'max_price':{'prompt':0,'completion':0,'request':0},'require_parameters':True}
        return self.transport(outgoing)


class PilotJournal:
    def __init__(self,path,*,run_id,fingerprint,max_calls=30,daily_limit=40,minute_limit=10,clock=time.time):
        if not isinstance(run_id,str) or not run_id.strip():raise ValueError('run ID required')
        for value,upper in ((max_calls,30),(daily_limit,40),(minute_limit,10)):
            if type(value) is not int or not 1<=value<=upper:raise ValueError('pilot limits must be positive and within free-pilot caps')
        self.path=str(path);self.run_id=run_id;self.clock=clock
        self.max_calls,self.daily_limit,self.minute_limit=max_calls,daily_limit,minute_limit
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS pilot_runs (run TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, max_calls INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS pilot_calls (run TEXT NOT NULL, ordinal INTEGER NOT NULL, request_sha TEXT NOT NULL, started REAL NOT NULL, state TEXT NOT NULL, payload TEXT, PRIMARY KEY(run,ordinal))')
            db.execute('CREATE TABLE IF NOT EXISTS pilot_control (id INTEGER PRIMARY KEY CHECK(id=1), blocked_until REAL NOT NULL)')
            db.execute('INSERT OR IGNORE INTO pilot_control VALUES (1,0)')
            row=db.execute('SELECT fingerprint,max_calls FROM pilot_runs WHERE run=?',(run_id,)).fetchone()
            if row and row!=(fingerprint,max_calls):raise ValueError('resume configuration/code/workload differs from the registered run')
            db.execute('INSERT OR IGNORE INTO pilot_runs VALUES (?,?,?)',(run_id,fingerprint,max_calls))
    @contextmanager
    def transaction(self):
        db=sqlite3.connect(self.path,timeout=5)
        try:
            db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except BaseException:
            db.rollback();raise
        finally:db.close()
    def reserve(self,ordinal,request):
        now=self.clock();request_sha=digest(request)
        midnight=datetime.fromtimestamp(now,timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
        with self.transaction() as db:
            existing=db.execute('SELECT request_sha,state,payload FROM pilot_calls WHERE run=? AND ordinal=?',(self.run_id,ordinal)).fetchone()
            if existing:
                if existing[0]!=request_sha:raise RunPaused('reconstruction_diverged')
                if existing[1]=='reserved':raise RunPaused('uncertain_attempt_requires_review')
                return existing[1],json.loads(existing[2])
            until=db.execute('SELECT blocked_until FROM pilot_control WHERE id=1').fetchone()[0]
            if until>now:raise RunPaused('upstream_rate_limit_pause',retry_at=until)
            count=db.execute('SELECT COUNT(*) FROM pilot_calls WHERE run=?',(self.run_id,)).fetchone()[0]
            if count>=self.max_calls:raise RunPaused('run_attempt_budget_exhausted')
            daily=db.execute('SELECT COUNT(*) FROM pilot_calls WHERE started>=?',(midnight,)).fetchone()[0]
            if daily>=self.daily_limit:raise RunPaused('project_daily_quota',retry_at=midnight+86400)
            recent=db.execute('SELECT started FROM pilot_calls WHERE started>? ORDER BY started',(now-60,)).fetchall()
            if len(recent)>=self.minute_limit:raise RunPaused('project_minute_quota',retry_at=recent[0][0]+60)
            db.execute('INSERT INTO pilot_calls VALUES (?,?,?,?,?,NULL)',(self.run_id,ordinal,request_sha,now,'reserved'))
            return None,None
    def finish(self,ordinal,state,payload):
        with self.transaction() as db:
            changed=db.execute("UPDATE pilot_calls SET state=?,payload=? WHERE run=? AND ordinal=? AND state='reserved'",(state,canonical(payload),self.run_id,ordinal)).rowcount
            if changed!=1:raise RunPaused('journal_state_changed')
    def block_rate_limit(self):
        now=self.clock();next_day=datetime.fromtimestamp(now,timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()+86400
        with self.transaction() as db:db.execute('UPDATE pilot_control SET blocked_until=MAX(blocked_until,?) WHERE id=1',(next_day,))
        return next_day
    def status(self):
        with self.transaction() as db:
            return dict(db.execute('SELECT state,COUNT(*) FROM pilot_calls WHERE run=? GROUP BY state',(self.run_id,)).fetchall())


class JournaledProvider:
    mode='live'
    def __init__(self,provider,journal):
        self.provider,self.journal=provider,journal
        self.ordinal=0;self.replayed_calls=0;self.new_calls=0
    @staticmethod
    def _journal_call(method,*args):
        try:return method(*args)
        except sqlite3.Error:
            raise RunPaused('quota_journal_unavailable_or_write_uncertain') from None
    def __call__(self,request):
        ordinal=self.ordinal;self.ordinal+=1
        state,payload=self._journal_call(self.journal.reserve,ordinal,request)
        if state is not None:
            self.replayed_calls+=1
            if state=='error':raise ProviderError(payload['code'],payload['status'])
            return payload
        self.new_calls+=1
        try:
            response=self.provider(request)
            canonical(response)
        except Exception as exc:
            code=exc.code if isinstance(exc,ProviderError) else 'upstream_or_processing_error'
            status=exc.status if isinstance(exc,ProviderError) else 502
            self._journal_call(self.journal.finish,ordinal,'error',{'code':code,'status':status})
            if isinstance(exc,ProviderError) and exc.code=='upstream_rate_limited':
                until=self._journal_call(self.journal.block_rate_limit)
                raise RunPaused('upstream_rate_limit_pause',retry_at=until) from None
            raise RunPaused('upstream_attempt_failed_requires_inspection') from None
        self._journal_call(self.journal.finish,ordinal,'complete',response)
        return copy.deepcopy(response)
