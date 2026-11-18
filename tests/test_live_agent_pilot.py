import copy
import json
import sqlite3
import zipfile
import pytest
from scripts import run_live_agent_pilot as pilot
from src.benchmark.free_pilot import PilotJournal,JournaledProvider
from src.benchmark.control import RunPaused
from src.benchmark.paired_runtime import AgentReferenceReplay

ITEM={'faq_id':'python-tuple','question':'How do I convert tuples and lists?','reference_answer':'Use tuple(seq) and list(seq).'}
BODY={'model':pilot.MODEL,'messages':[{'role':'user','content':'fixture'}]}


def seed(path,count=0):
    j=PilotJournal(path,run_id='prior',fingerprint='prior',max_calls=30,clock=lambda:100)
    for i in range(count):
        j.clock=lambda i=i:1+i*61;j.reserve(i,BODY);j.finish(i,'complete',{'choices':[]})
    return j


def test_preflight_refuses_unaffordable_zero_hit_workload_without_writes(tmp_path):
    path=tmp_path/'journal';seed(path,29);before=path.read_bytes()
    with pytest.raises(RunPaused,match='insufficient_daily_budget'):pilot.preflight(path,'config',now=3000)
    assert path.read_bytes()==before


def test_preflight_accepts_exact_daily_fit_and_requires_existing_ledger(tmp_path):
    path=tmp_path/'journal';seed(path,28)
    result=pilot.preflight(path,'config',now=3000)
    assert result['fits_daily_budget'] and result['maximum_new_attempts']==12
    assert result['project_daily_remaining']==12
    with pytest.raises(ValueError):pilot.preflight(tmp_path/'missing','config')
    assert not (tmp_path/'missing').exists()


def test_preflight_blocks_fingerprint_uncertainty_and_truncation(tmp_path):
    from src.api.providers import FixtureChatProvider
    path=tmp_path/'journal';j=PilotJournal(path,run_id=pilot.RUN_ID,fingerprint='config',max_calls=12,clock=lambda:100)
    with pytest.raises(ValueError,match='configuration'):pilot.preflight(path,'changed',now=3000)
    j.reserve(0,BODY)
    with pytest.raises(ValueError,match='uncertain'):pilot.preflight(path,'config',now=3000)
    response=FixtureChatProvider()(BODY);response['choices'][0]['finish_reason']='length';j.finish(0,'complete',response)
    with pytest.raises(ValueError,match='incomplete'):pilot.preflight(path,'config',now=3000)


def test_actual_agent_collects_both_arms_without_human_autoapproval(tmp_path):
    capture=pilot.CapturedProvider(AgentReferenceReplay([ITEM]))
    outcome=pilot.collect(ITEM,capture,tmp_path)
    assert outcome['status']=='COMPLETED_COLLECTION'
    assert len(capture.calls)==11
    assert all(r['state']['status']=='AWAITING_APPROVAL' and r['state']['answer'] is None for r in outcome['tasks'])
    assert sum(o['status']=='HIT' for o in outcome['observations']['exact'])==1
    packet,mapping=pilot.review_packet(outcome['observations'],'source-v1')
    assert packet['items'] and mapping
    for i in packet['items']:
        assert 'Keep the answer under 180 words.' in i['system_instruction']
        assert i['generation_context']['evidence'] and i['source_revision']=='source-v1'
        assert i['correct'] is None and 'arm' not in i


def test_quota_pause_export_and_reconstruction_preserve_every_attempt(tmp_path,monkeypatch):
    now=[100.0];path=tmp_path/'journal'
    raw=AgentReferenceReplay([ITEM]);actual=[]
    def transport(body):actual.append(copy.deepcopy(body));return raw(body)
    journal=PilotJournal(path,run_id=pilot.RUN_ID,fingerprint='config',max_calls=12,minute_limit=1,clock=lambda:now[0])
    outputs=[]
    monkeypatch.setenv('OPENROUTER_API_KEY','dummy-not-for-export')
    for i in range(12):
        database=tmp_path/f'tasks-{i}';database.mkdir()
        metered=JournaledProvider(transport,journal);capture=pilot.CapturedProvider(metered)
        outcome=pilot.collect(ITEM,capture,database)
        target,summary=pilot.export(tmp_path/'exports',{'source_revision':'fixed'},outcome,capture,journal,metered,ITEM,b'fixture license')
        outputs.append(target)
        with zipfile.ZipFile(target) as z:
            assert z.read('LICENSE')==b'fixture license'
            assert all(b'dummy-not-for-export' not in z.read(n) for n in z.namelist())
        if summary['status']=='COMPLETED_COLLECTION':break
        assert summary['pause']['reason']=='project_minute_quota'
        now[0]+=61
    assert len(actual)==11 and len(set(outputs))==len(outputs)
    assert summary['status']=='COMPLETED_COLLECTION' and not summary['live_latency_valid']
    assert summary['saved_attempts']==11 and summary['human_approvals']==0
    with sqlite3.connect(path) as db:assert db.execute('SELECT COUNT(*) FROM pilot_calls').fetchone()[0]==11


def test_invalid_live_plan_and_truncation_stop_with_export(tmp_path):
    from src.api.providers import FixtureChatProvider
    for name,finish in [('plan','stop'),('truncated','length')]:
        directory=tmp_path/name;directory.mkdir();path=directory/'journal'
        def bad(body):
            result=FixtureChatProvider()(body);result['choices'][0]['finish_reason']=finish;return result
        journal=PilotJournal(path,run_id=pilot.RUN_ID,fingerprint='config',max_calls=12)
        metered=JournaledProvider(bad,journal);capture=pilot.CapturedProvider(metered)
        outcome=pilot.collect(ITEM,capture,directory)
        assert outcome['status']=='PAUSED' and len(capture.calls)==1
        target,summary=pilot.export(directory/'exports',{'source_revision':'fixed'},outcome,capture,journal,metered,ITEM,b'license')
        assert target.exists() and summary['saved_attempts']==1
        assert summary['pause']['resume_allowed'] is False
