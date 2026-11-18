import copy
import hashlib
import json
import sqlite3
import zipfile
import pytest
from scripts import export_r5_journal as exporter
from src.benchmark.free_pilot import PilotJournal
from src.api.providers import FixtureChatProvider
from src.benchmark.paired_runtime import build_events
from src.benchmark.r5_revision import revise_events
from src.benchmark.schema import digest
from tests.test_paired_runtime import corpus

MODEL='example/test:free'

def setup(tmp_path,monkeypatch):
    source=corpus();p=tmp_path/'corpus.json';p.write_text(json.dumps(source));(tmp_path/'LICENSE').write_text('fixture')
    monkeypatch.setattr(exporter,'CORPUS_HASH',digest(source))
    events=build_events(source,model=MODEL)
    path=tmp_path/'journal.sqlite'
    for run,profile in [('r5-live-v1',events),('r5-live-v2',revise_events(events))]:
        j=PilotJournal(path,run_id=run,fingerprint=run,max_calls=16 if run=='r5-live-v1' else 15,clock=lambda:100)
        body=profile[0]['request'];response=FixtureChatProvider()(body)
        if run=='r5-live-v1':response['choices'][0]['finish_reason']='length'
        j.reserve(0,body);j.finish(0,'complete',response)
    return path,p

def test_read_only_export_complete_and_truncated_without_network(tmp_path,monkeypatch):
    import socket
    path,source=setup(tmp_path,monkeypatch);before=path.read_bytes()
    monkeypatch.setenv('OPENROUTER_API_KEY','dummy-must-not-export')
    monkeypatch.setattr(socket,'socket',lambda *a,**k:pytest.fail('Network forbidden'))
    out=tmp_path/'partial.zip';summary=exporter.export_saved(path,source,out,MODEL)
    assert path.read_bytes()==before
    assert summary['combined_saved_attempts']==2 and summary['unique_review_items']==1
    assert summary['model_calls_made_by_export']==0 and summary['paired_savings'] is None
    assert not summary['live_latency_valid']
    with zipfile.ZipFile(out) as z:
        responses=json.loads(z.read('saved-responses.private.json'))['attempts']
        assert responses[0]['diagnostics']['rejection_reason']=='output_token_limit'
        assert all(r['arm'] is None for r in responses)
        reviews=json.loads(z.read('reviews.template.json'))['items']
        assert reviews[0]['correct'] is None and 'run_id' not in reviews[0]
        manifest=json.loads(z.read('manifest.json'))
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest['files'].items())
        assert all(b'dummy-must-not-export' not in z.read(n) for n in z.namelist())
    saved=out.read_bytes()
    with pytest.raises(ValueError,match='exists'):exporter.export_saved(path,source,out,MODEL)
    assert out.read_bytes()==saved

@pytest.mark.parametrize('state',['reserved','error','complete'])
def test_uncertain_error_or_malformed_response_stays_counted(tmp_path,monkeypatch,state):
    path,source=setup(tmp_path,monkeypatch)
    with sqlite3.connect(path) as db:db.execute("UPDATE pilot_calls SET state=?,payload='bad-json' WHERE run='r5-live-v2'",(state,))
    summary=exporter.export_saved(path,source,tmp_path/'partial.zip',MODEL)
    assert summary['combined_saved_attempts']==2 and summary['unique_review_items']==0

def test_unmatched_hash_not_assigned_question_or_review(tmp_path,monkeypatch):
    path,source=setup(tmp_path,monkeypatch)
    with sqlite3.connect(path) as db:db.execute("UPDATE pilot_calls SET request_sha='unknown' WHERE run='r5-live-v2'")
    summary=exporter.export_saved(path,source,tmp_path/'partial.zip',MODEL)
    assert summary['unique_review_items']==0
    assert summary['runs']['r5-live-v2']['unmatched_request_hashes']==1

def test_missing_journal_and_changed_corpus_fail_closed(tmp_path,monkeypatch):
    path,source=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError):exporter.export_saved(tmp_path/'missing',source,tmp_path/'out.zip',MODEL)
    source.write_text('{}')
    with pytest.raises(ValueError,match='digest'):exporter.export_saved(path,source,tmp_path/'out.zip',MODEL)
    assert not (tmp_path/'out.zip').exists()

def test_export_after_combined_budget_exhaustion(tmp_path,monkeypatch):
    from src.benchmark.control import RunPaused
    path,source=setup(tmp_path,monkeypatch)
    j=PilotJournal(path,run_id='r5-live-v2',fingerprint='r5-live-v2',max_calls=15,clock=lambda:200)
    body=revise_events(build_events(corpus(),model=MODEL))[0]['request']
    for i in range(1,15):
        j.clock=lambda i=i:200+i*61
        j.reserve(i,body);j.finish(i,'complete',FixtureChatProvider()(body))
    with pytest.raises(RunPaused,match='run_attempt_budget_exhausted'):j.reserve(15,body)
    before=path.read_bytes()
    summary=exporter.export_saved(path,source,tmp_path/'partial.zip',MODEL)
    assert summary['combined_saved_attempts']==16
    assert summary['runs']['r5-live-v1']['saved_attempts']==1
    assert summary['runs']['r5-live-v2']['saved_attempts']==15
    assert path.read_bytes()==before
