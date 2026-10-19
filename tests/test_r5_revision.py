import copy
import json
import sqlite3
import pytest
from src.benchmark.r5_revision import predecessor_snapshot, revise_events, pause_guidance
from src.benchmark.free_pilot import PilotJournal, JournaledProvider, CompleteAnswerProvider, FreeOpenRouterProvider
from src.benchmark.control import RunPaused
from src.api.providers import FixtureChatProvider
from src.benchmark.schema import digest

MODEL='example/test:free'
BODY={'model':MODEL,'messages':[{'role':'user','content':'Question'}],'max_tokens':1024}

def predecessor(tmp_path, count=1):
    path=tmp_path/'shared.sqlite'
    j=PilotJournal(path,run_id='r5-live-v1',fingerprint='old',max_calls=16,clock=lambda:100)
    for i in range(count):
        j.reserve(i,BODY)
        response=FixtureChatProvider()(BODY)
        response['model']=MODEL
        response['choices'][0]['finish_reason']='length' if i==count-1 else 'stop'
        j.finish(i,'complete',response)
    return path

def test_snapshot_read_only_and_counts_all_attempts(tmp_path):
    path=predecessor(tmp_path,3); before=path.read_bytes()
    result=predecessor_snapshot(path,MODEL)
    assert result['attempts']==3 and result['remaining_attempt_allowance']==13
    assert path.read_bytes()==before
    assert 'content' not in json.dumps(result)

def test_profile_changes_real_request_without_mutating_old_events():
    events=[{'request':{'model':MODEL,'max_tokens':1024,'messages':[{'role':'system','content':'Reference'},{'role':'user','content':'Q'}]}}]
    original=copy.deepcopy(events); revised=revise_events(events)
    assert events==original and revised[0]['request']['max_tokens']==4096
    assert '180 words' in revised[0]['request']['messages'][0]['content']
    assert revised[0]['request']['messages'][1]==events[0]['request']['messages'][1]
    assert digest(revised)!=digest(events)
    seen=[]
    FreeOpenRouterProvider('fixture',models={MODEL},transport=lambda b:seen.append(b))(revised[0]['request'])
    assert seen[0]['max_tokens']==4096 and seen[0]['provider']['max_price']['completion']==0

@pytest.mark.parametrize('mutation',["UPDATE pilot_calls SET state='reserved'", "UPDATE pilot_calls SET state='error'", "DELETE FROM pilot_calls", "UPDATE pilot_runs SET max_calls=30"])
def test_bad_predecessor_fails_closed(tmp_path,mutation):
    path=predecessor(tmp_path)
    with sqlite3.connect(path) as db:db.execute(mutation)
    with pytest.raises(ValueError):predecessor_snapshot(path,MODEL)

def test_missing_journal_not_created(tmp_path):
    path=tmp_path/'missing.sqlite'
    with pytest.raises(ValueError):predecessor_snapshot(path,MODEL)
    assert not path.exists()

def test_model_change_rejected(tmp_path):
    with pytest.raises(ValueError):predecessor_snapshot(predecessor(tmp_path),'other/model:free')

def test_child_budget_includes_predecessor_and_parent_unchanged(tmp_path):
    path=predecessor(tmp_path,2); parent=predecessor_snapshot(path,MODEL)
    j=PilotJournal(path,run_id='r5-live-v2',fingerprint=digest(parent),max_calls=14,clock=lambda:200)
    for i in range(14):
        j.clock=lambda i=i:200+i*61
        j.reserve(i,BODY);j.finish(i,'complete',FixtureChatProvider()(BODY))
    with pytest.raises(RunPaused,match='run_attempt_budget_exhausted'):j.reserve(14,BODY)
    assert predecessor_snapshot(path,MODEL)==parent
    with sqlite3.connect(path) as db:assert db.execute('SELECT COUNT(*) FROM pilot_calls').fetchone()[0]==16

def test_shared_daily_cap_and_changed_parent_fingerprint(tmp_path):
    path=predecessor(tmp_path,2); parent=predecessor_snapshot(path,MODEL)
    j=PilotJournal(path,run_id='r5-live-v2',fingerprint=digest(parent),max_calls=14,daily_limit=2,clock=lambda:200)
    with pytest.raises(RunPaused,match='project_daily_quota'):j.reserve(0,BODY)
    with sqlite3.connect(path) as db:db.execute("UPDATE pilot_calls SET started=started+1 WHERE run='r5-live-v1'")
    with pytest.raises(ValueError,match='differs'):
        PilotJournal(path,run_id='r5-live-v2',fingerprint=digest(predecessor_snapshot(path,MODEL)),max_calls=14)

def test_truncation_replayed_without_new_call(tmp_path):
    path=predecessor(tmp_path);calls=[]
    provider=CompleteAnswerProvider(JournaledProvider(lambda b:calls.append(b),PilotJournal(path,run_id='r5-live-v1',fingerprint='old',max_calls=16)))
    with pytest.raises(RunPaused,match='incomplete_response'):provider(BODY)
    assert calls==[]

@pytest.mark.parametrize('reason,allowed',[('project_minute_quota',True),('project_daily_quota',True),('upstream_rate_limit_pause',True),('incomplete_response_inspect_journal',False),('uncertain_attempt_requires_review',False),('run_attempt_budget_exhausted',False)])
def test_pause_advice(reason,allowed):
    assert pause_guidance(RunPaused(reason))['resume_allowed_after_retry_time'] is allowed

def test_cli_plan_pause_resume_and_evidence_offline(tmp_path,monkeypatch,capsys):
    import sys
    import zipfile
    from scripts import run_r5_paired_v2 as runner
    from tests.test_paired_runtime import corpus
    path=predecessor(tmp_path); parent=predecessor_snapshot(path,MODEL)
    source=corpus();source['source_sha256']='648bc2d476230c98f61571e9de969811150875996de2552d5c7656368b2df184'
    source_path=tmp_path/'corpus.json';source_path.write_text(json.dumps(source))
    (tmp_path/'LICENSE').write_text('fixture license')
    real_digest=runner.digest
    monkeypatch.setattr(runner,'digest',lambda value:'0fe8140b1d3bb1bf752821d15e863b5552db324d17ed373b89952837620a7089' if value==source else real_digest(value))
    args=['runner','--corpus',str(source_path),'--output-dir',str(tmp_path/'out'),'--model',MODEL,'--journal',str(path)]
    monkeypatch.setattr(sys,'argv',args)
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    before=path.read_bytes();runner.main()
    assert '"key_read": false' in capsys.readouterr().out and path.read_bytes()==before
    now=[100.0];calls=[]
    class ClockJournal(PilotJournal):
        def __init__(self,*a,**kw):super().__init__(*a,**kw,clock=lambda:now[0])
    class FakeFree:
        mode='live'
        def __init__(self,*a,**kw):pass
        def __call__(self,body):
            calls.append(copy.deepcopy(body));return FixtureChatProvider()(body)
    monkeypatch.setattr(runner,'PilotJournal',ClockJournal)
    monkeypatch.setattr(runner,'FreeOpenRouterProvider',FakeFree)
    monkeypatch.setenv('OPENROUTER_API_KEY','fixture-value-must-not-be-exported')
    monkeypatch.setattr(sys,'argv',args+['--execute'])
    runner.main();output=capsys.readouterr().out
    assert 'project_minute_quota' in output and '"resume_allowed_after_retry_time": true' in output
    assert len(calls)==9 and not (tmp_path/'out-evidence.zip').exists()
    now[0]=161;runner.main();capsys.readouterr()
    assert len(calls)==14 and all(b['max_tokens']==4096 for b in calls)
    assert predecessor_snapshot(path,MODEL)==parent
    with zipfile.ZipFile(tmp_path/'out-evidence.zip') as archive:
        trace=json.loads(archive.read('trace.json'))
        assert trace['execution_mode']=='resumed_reconstruction' and not trace['live_latency_valid']
        assert json.loads(archive.read('predecessor-diagnostics.json'))==parent
        assert all(b'fixture-value-must-not-be-exported' not in archive.read(name) for name in archive.namelist())
    with pytest.raises(SystemExit):runner.main()
    assert len(calls)==14
