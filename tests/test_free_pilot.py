import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from src.api.providers import FixtureChatProvider,ProviderError
from src.benchmark.control import RunPaused
from src.benchmark.free_pilot import FreeOpenRouterProvider,PilotJournal,JournaledProvider
from src.benchmark.runner import run_workload
from src.benchmark.report import summarize
from src.benchmark.semantic import AnswerVerifier
from src.benchmark.schema import read_json,write_new


BODY={'model':'example/test:free','messages':[{'role':'user','content':'What is photosynthesis?'}],'temperature':0,'max_tokens':64}


def journal(path,run='test',**kwargs):
    return PilotJournal(path,run_id=run,fingerprint='fixed',**kwargs)


class Counting(FixtureChatProvider):
    def __init__(self):self.calls=0
    def __call__(self,body):self.calls+=1;return super().__call__(body)


def test_free_price_and_routing_controls_are_operator_owned():
    seen=[]
    def transport(body):seen.append(body);return FixtureChatProvider()(body)
    provider=FreeOpenRouterProvider('not-a-real-key',models={BODY['model']},transport=transport)
    provider(BODY)
    assert seen[0]['provider']=={'allow_fallbacks':False,'max_price':{'prompt':0,'completion':0,'request':0},'require_parameters':True}
    assert 'provider' not in BODY
    for field,value in [('model','example/paid'),('models',['example/paid']),('provider',{}),('plugins',[])]:
        with pytest.raises(ValueError):provider({**BODY,field:value})
    assert len(seen)==1


@pytest.mark.parametrize('model',['openrouter/free','example/paid','example/test:online:free','example/test:floor:free'])
def test_auto_and_unpriced_models_rejected(model):
    with pytest.raises(ValueError):FreeOpenRouterProvider('not-a-real-key',models={model})


def test_reopen_replays_recorded_response_without_network(tmp_path):
    path=tmp_path/'calls.sqlite';raw=Counting()
    a=JournaledProvider(raw,journal(path));answer=a(BODY)
    b=JournaledProvider(raw,journal(path))
    assert b(BODY)==answer and raw.calls==1
    assert b.replayed_calls==1 and b.new_calls==0


def test_concurrent_process_style_reservations_cannot_overspend(tmp_path):
    path=tmp_path/'calls.sqlite';journal(path,daily_limit=2)
    def reserve(i):
        try:
            return journal(path,run=str(i),daily_limit=2).reserve(0,BODY)[0]
        except RunPaused as e:return e.reason
    with ThreadPoolExecutor(max_workers=8) as pool:out=list(pool.map(reserve,range(8)))
    assert out.count(None)==2 and out.count('project_daily_quota')==6


def test_failed_calls_count_and_are_not_retried_on_resume(tmp_path):
    calls=[]
    def fail(body):calls.append(1);raise ProviderError('upstream_timeout',504)
    path=tmp_path/'calls.sqlite'
    with pytest.raises(RunPaused,match='requires_inspection'):JournaledProvider(fail,journal(path))(BODY)
    with pytest.raises(ProviderError):JournaledProvider(fail,journal(path))(BODY)
    assert len(calls)==1
    assert journal(path).status()=={'error':1}


def test_crash_after_reservation_never_automatically_retries(tmp_path):
    j=journal(tmp_path/'calls.sqlite');j.reserve(0,BODY)
    raw=Counting()
    with pytest.raises(RunPaused,match='uncertain'):JournaledProvider(raw,j)(BODY)
    assert raw.calls==0


def test_budget_minute_and_utc_day_rollover(tmp_path):
    now=[100.0];path=tmp_path/'calls.sqlite'
    a=journal(path,clock=lambda:now[0],minute_limit=1,daily_limit=2)
    a.reserve(0,BODY)
    with pytest.raises(RunPaused,match='minute'):a.reserve(1,BODY)
    now[0]+=60;a.reserve(1,BODY)
    with pytest.raises(RunPaused,match='daily'):a.reserve(2,BODY)
    now[0]=86400;a.reserve(2,BODY)
    assert a.status()=={'reserved':3}


def test_run_budget_does_not_reset_with_new_day(tmp_path):
    now=[100.0];j=journal(tmp_path/'calls.sqlite',max_calls=1,clock=lambda:now[0]);j.reserve(0,BODY)
    now[0]=86400
    with pytest.raises(RunPaused,match='run_attempt'):j.reserve(1,BODY)


def test_resume_rejects_changed_workload_or_call_sequence(tmp_path):
    path=tmp_path/'calls.sqlite';j=journal(path);j.reserve(0,BODY);j.finish(0,'complete',{})
    with pytest.raises(ValueError):PilotJournal(path,run_id='test',fingerprint='changed')
    with pytest.raises(RunPaused,match='diverged'):j.reserve(0,{**BODY,'temperature':.1})


def test_429_pauses_other_runs_until_next_local_utc_day(tmp_path):
    now=[100.0];path=tmp_path/'calls.sqlite'
    def rate(body):raise ProviderError('upstream_rate_limited',503)
    j=journal(path,clock=lambda:now[0])
    with pytest.raises(RunPaused,match='upstream_rate_limit'):JournaledProvider(rate,j)(BODY)
    other=journal(path,run='other',clock=lambda:now[0])
    with pytest.raises(RunPaused,match='upstream_rate_limit'):other.reserve(0,BODY)
    now[0]=86400;other.reserve(0,BODY)


def test_pause_propagates_through_verifier_and_runner_without_fake_errors():
    class Paused:
        mode='live'
        def __call__(self,body):raise RunPaused('quota')
    from tests.test_response_cache import request,response
    with pytest.raises(RunPaused):AnswerVerifier(Paused(),'judge')(request(),request(),response())
    w=read_json(Path(__file__).parents[1]/'examples/benchmark/workload.fixture.json')
    with pytest.raises(RunPaused):run_workload(w,Paused())


def test_reconstructed_run_hides_live_latency_comparisons(tmp_path):
    w=read_json(Path(__file__).parents[1]/'examples/benchmark/workload.fixture.json')
    w['requests']=w['requests'][:1]
    raw=Counting();path=tmp_path/'calls.sqlite'
    first=JournaledProvider(raw,journal(path));run_workload(w,first)
    second=JournaledProvider(raw,journal(path));trace=run_workload(w,second)
    trace['latency_valid']=False
    report=summarize(trace)
    assert raw.calls==2 and second.replayed_calls==2
    assert report['cache_arm']['successful_answer_latency_ms']['p50'] is None
    assert report['paired_latency']['mean_cache_minus_fresh_ms'] is None
    assert report['paired_latency']['group_bootstrap_95'] is None


def test_cli_plan_mode_needs_no_key_and_does_not_create_files(tmp_path,monkeypatch):
    from scripts.run_free_pilot import main
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    root=Path(__file__).parents[1]
    main(['--workload',str(root/'examples/benchmark/workload.fixture.json'),'--model',BODY['model'],
          '--run-id','plan','--journal',str(tmp_path/'ledger.sqlite'),'--output-dir',str(tmp_path/'out')])
    assert list(tmp_path.iterdir())==[]


def test_cli_resume_exports_once_without_repeating_completed_calls(tmp_path,monkeypatch):
    import scripts.run_free_pilot as cli
    original=PilotJournal;now=[100.0];sent=[]
    class LocalProvider(FixtureChatProvider):
        mode='live'
        def __init__(self,*args,**kwargs):pass
        def __call__(self,body):sent.append(copy.deepcopy(body));return super().__call__(body)
    monkeypatch.setattr(cli,'FreeOpenRouterProvider',LocalProvider)
    monkeypatch.setattr(cli,'PilotJournal',lambda path,**kw:original(path,**kw,minute_limit=1,clock=lambda:now[0]))
    monkeypatch.setenv('OPENROUTER_API_KEY','local-test-placeholder')
    root=Path(__file__).parents[1]
    output=tmp_path/'evidence'
    args=['--workload',str(root/'examples/benchmark/workload.fixture.json'),'--model',BODY['model'],
          '--run-id','resume','--journal',str(tmp_path/'ledger.sqlite'),'--output-dir',str(output),'--limit','1','--max-output-tokens','512','--execute']
    cli.main(args)
    assert not output.exists() and len(sent)==1
    now[0]+=60;cli.main(args)
    assert len(sent)==2 and all(body['max_tokens']==512 for body in sent)
    trace=read_json(output/'trace.json');report=read_json(output/'report.unreviewed.json')
    assert trace['execution_mode']=='resumed_reconstruction' and trace['replayed_calls']==1
    assert report['latency_valid'] is False
    assert (output/'manifest.json').exists()
    with pytest.raises(SystemExit):cli.main(args)
    assert len(sent)==2


def test_journal_failure_stops_before_sending_or_reporting_fake_api_error(tmp_path,monkeypatch):
    import sqlite3
    j=journal(tmp_path/'calls.sqlite');raw=Counting()
    def fail(*args):raise sqlite3.OperationalError('disk unavailable')
    monkeypatch.setattr(j,'reserve',fail)
    with pytest.raises(RunPaused,match='journal_unavailable'):JournaledProvider(raw,j)(BODY)
    assert raw.calls==0


def test_persistence_failure_after_send_leaves_uncertain_call_without_retry(tmp_path,monkeypatch):
    import sqlite3
    j=journal(tmp_path/'calls.sqlite');raw=Counting()
    def fail(*args):raise sqlite3.OperationalError('disk full')
    monkeypatch.setattr(j,'finish',fail)
    with pytest.raises(RunPaused,match='write_uncertain'):JournaledProvider(raw,j)(BODY)
    assert raw.calls==1 and j.status()=={'reserved':1}
    with pytest.raises(RunPaused,match='uncertain_attempt'):JournaledProvider(raw,j)(BODY)
    assert raw.calls==1
