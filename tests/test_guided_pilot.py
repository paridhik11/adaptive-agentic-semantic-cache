"""Offline checks: the guided job must make progress without wasting live quota."""
import copy
from pathlib import Path
import pytest
from src.api.providers import FixtureChatProvider
from src.benchmark.control import RunPaused
from src.benchmark.free_pilot import FreeOpenRouterProvider,CompleteAnswerProvider,JournaledProvider,PilotJournal
from src.benchmark.schema import read_json
from src.benchmark.progress import ProgressPage

BODY={'model':'example/test:free','messages':[{'role':'user','content':'What is photosynthesis?'}],'temperature':0,'max_tokens':512}


def test_reasoning_profile_is_operator_owned():
    seen=[]
    p=FreeOpenRouterProvider('dummy',models={BODY['model']},reasoning_enabled=False,
        transport=lambda body:seen.append(body) or FixtureChatProvider()(body))
    p(BODY)
    assert seen[0]['reasoning']=={'enabled':False}
    assert seen[0]['provider']['require_parameters'] is True
    assert 'reasoning' not in BODY
    with pytest.raises(ValueError):p({**BODY,'reasoning':{'enabled':True}})
    with pytest.raises(ValueError):FreeOpenRouterProvider('dummy',models={BODY['model']},reasoning_enabled='false')


def test_incomplete_answer_saved_and_blocks_followup_calls(tmp_path):
    calls=[]
    def source(body):
        calls.append(body)
        response=FixtureChatProvider()(body)
        response['choices'][0]['finish_reason']='length'
        return response
    j=PilotJournal(tmp_path/'journal.sqlite',run_id='stop',fingerprint='fixed')
    provider=CompleteAnswerProvider(JournaledProvider(source,j))
    for _ in range(1):
        with pytest.raises(RunPaused) as caught:provider(BODY)
    assert caught.value.diagnostics['rejection_reason']=='output_token_limit'
    assert len(calls)==1 and j.status()=={'complete':1}
    replay=CompleteAnswerProvider(JournaledProvider(source,j))
    with pytest.raises(RunPaused):replay(BODY)
    assert len(calls)==1


def test_progress_page_escapes_answers_and_question(tmp_path):
    page=ProgressPage(tmp_path/'status.html',run_id='test',model='free',total_requests=1)
    page({'kind':'arm_started','event_id':'x','arm':'cache','question':'<script>bad()</script>'})
    page({'kind':'arm_finished','event_id':'x','arm':'cache','status':'HIT','provider_calls':0,'latency_ms':1,
        'answer':'<img src=x onerror=bad()>'})
    text=page.path.read_text()
    assert '<script>' not in text and '<img src=x' not in text
    assert '&lt;img' in text and 'unreviewed' in text and 'HIT' in text


def local_provider(monkeypatch,truncate=False):
    import scripts.run_free_pilot as cli
    sent=[]
    class Local(FixtureChatProvider):
        mode='live'
        def __init__(self,*args,**kwargs):
            self.reasoning=kwargs.get('reasoning_enabled')
        def __call__(self,body):
            assert self.reasoning is False
            sent.append(copy.deepcopy(body))
            answer=super().__call__(body)
            if truncate:answer['choices'][0]['finish_reason']='length'
            return answer
    monkeypatch.setattr(cli,'FreeOpenRouterProvider',Local)
    monkeypatch.setenv('OPENROUTER_API_KEY','dummy-local-test')
    return sent


def test_one_session_exports_progress_and_one_bundle_without_repeating_calls(tmp_path,monkeypatch):
    from scripts.run_zero_budget_session import main
    monkeypatch.chdir(tmp_path)
    sent=local_provider(monkeypatch)
    args=['--execute','--no-browser','--model',BODY['model']]
    main(args)
    directory=tmp_path/'runs/guided-exact-v1'
    package=read_json(directory/'results.json')
    assert len(sent)==10
    assert package['report']['status_counts']['HIT']==2
    assert package['report']['latency_valid'] is True
    assert package['report']['nonexact_reuse']['accepted']==0
    assert all(x['max_tokens']==512 and 'two concise sentences' in x['messages'][-1]['content'] for x in sent)
    assert 'completed' in (directory/'status.html').read_text()
    # Recover a packaging interruption without repeating inference.
    (directory/'results.json').unlink()
    main(args)
    assert len(sent)==10 and (directory/'results.json').exists()
    with pytest.raises(SystemExit):main(args+['--run-id','different'])


def test_session_stops_after_first_truncated_answer(tmp_path,monkeypatch):
    from scripts.run_zero_budget_session import main
    monkeypatch.chdir(tmp_path)
    sent=local_provider(monkeypatch,truncate=True)
    main(['--execute','--no-browser','--model',BODY['model']])
    directory=tmp_path/'runs/guided-exact-v1'
    assert len(sent)==1
    assert read_json(directory/'paused.json')['response_diagnostics']['rejection_reason']=='output_token_limit'
    assert 'paused' in (directory/'status.html').read_text()
    assert not (directory/'evidence').exists()


def test_guided_plan_makes_no_calls_or_key_access(tmp_path,monkeypatch):
    from scripts.run_zero_budget_session import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    main(['--no-browser','--model',BODY['model']])
    assert not (tmp_path/'runs/openrouter-free.sqlite').exists()


def test_observer_failure_cannot_change_cache_decisions():
    from scripts.run_zero_budget_session import workload
    from src.benchmark.runner import run_workload
    def broken(event):raise OSError('page unavailable')
    trace=run_workload(workload(),FixtureChatProvider(),observer=broken)
    assert [e['status'] for e in trace['events']].count('HIT')==2
    assert sum(e[a]['provider_calls'] for e in trace['events'] for a in ('cache','fresh'))==10


def test_session_waits_for_local_minute_quota_without_duplicate_calls(tmp_path,monkeypatch):
    import types
    import scripts.run_zero_budget_session as session
    import scripts.run_free_pilot as cli
    monkeypatch.chdir(tmp_path)
    sent=local_provider(monkeypatch)
    now=[100.0]
    monkeypatch.setattr(session,'time',types.SimpleNamespace(time=lambda:now[0],sleep=lambda seconds:now.__setitem__(0,now[0]+seconds)))
    monkeypatch.setattr(cli,'PilotJournal',lambda path,**kw:PilotJournal(path,**kw,minute_limit=3,clock=lambda:now[0]))
    session.main(['--execute','--no-browser','--model',BODY['model']])
    result=read_json(tmp_path/'runs/guided-exact-v1/results.json')
    assert len(sent)==10 and result['report']['status_counts']['HIT']==2
    assert result['report']['latency_valid'] is False
    assert result['report']['paired_latency']['mean_cache_minus_fresh_ms'] is None
