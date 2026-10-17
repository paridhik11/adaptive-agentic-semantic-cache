import copy
from src.api.providers import FixtureChatProvider
from src.benchmark.runner import run_workload
from src.benchmark.reuse_diagnosis import diagnose
from src.benchmark.semantic import AbstainingVerifier
from scripts.run_zero_budget_session import workload
from tests.test_experimental_semantic import cache,run,changed
from tests.test_response_cache import request


def test_accounting_reconciles_and_prior_repeat_is_not_safe_opportunity():
    trace=run_workload(workload(),FixtureChatProvider())
    d=diagnose(trace)
    assert d['terminal_outcomes']=={'MISS':4,'HIT':2}
    assert d['eligible_requests']==6
    assert d['provider_work']['total_cache_attempts']==4
    assert d['provider_work']['fresh_attempts']==6
    assert d['prior_identical_request_occurrences_by_outcome']=={'HIT':2}
    assert d['semantic_expirations_observed'] is None
    assert d['safety_certified'] is False


def test_old_trace_does_not_invent_eligibility_or_candidate_data():
    trace=run_workload(workload(),FixtureChatProvider())
    for e in trace['events']:e.pop('eligibility');e.pop('decision_diagnostics')
    d=diagnose(trace)
    assert d['eligible_requests'] is None and d['eligible_hit_fraction'] is None
    assert d['eligibility_observed']=={'unrecorded':6}
    assert d['semantic_terminal_stages']=={}


def test_all_bypasses_explained_without_claiming_queries_unsafe():
    w=workload()
    for row in w['requests']:row['request']['messages'].insert(0,{'role':'system','content':'Answer concisely'})
    d=diagnose(run_workload(w,FixtureChatProvider()))
    assert d['terminal_outcomes']=={'BYPASS':6}
    assert d['eligibility_observed']=={'bypass:context_requires_review':6}
    assert d['prior_identical_request_occurrences_by_outcome']=={'BYPASS':2}
    assert d['eligible_hit_fraction'] is None


def test_candidate_and_verifier_diagnoses_are_actual_stages():
    c=cache(verifier=AbstainingVerifier())
    a=run(c,request(),'seed');b=run(c,changed(),'different')
    assert a.diagnosis['stage']=='no_candidate' and a.diagnosis['scope_candidates']==0
    assert b.diagnosis['stage']=='verifier_rejected'
    assert b.diagnosis['scope_candidates']==1 and b.diagnosis['threshold_passed'] is True
    assert b.diagnosis['verifier_approved'] is False
    assert run(c,changed(),'repeat').diagnosis['stage']=='exact_hit'
    # A result's diagnostics cannot change when a later request is processed.
    assert a.diagnosis['stage']=='no_candidate'


def test_threshold_and_expiry_distinguished():
    class Different:
        identity='test'
        def encode(self,text):return [1,0] if text==request()['messages'][-1]['content'] else [0,1]
    now=[0]
    c=cache(mode='similarity',clock=lambda:now[0],ttl_seconds=1)
    c.encoder=Different()
    run(c,request(),'seed')
    r=run(c,changed(),'rejected')
    assert r.diagnosis['stage']=='threshold_rejected'
    now[0]=2
    r=run(c,request(),'expired')
    assert r.diagnosis['stage']=='no_candidate' and r.diagnosis['expired_entries']==2


def test_diagnosis_cli_reproduces_without_model_calls(tmp_path):
    from scripts.diagnose_reuse import main
    from src.benchmark.schema import write_new,read_json
    trace=tmp_path/'trace.json';write_new(trace,run_workload(workload(),FixtureChatProvider()))
    main(['--trace',str(trace),'--output-dir',str(tmp_path/'out')])
    assert read_json(tmp_path/'out/diagnosis.json')['runs']['supplied-trace']['accounting_reconciled']


def test_policy_contract_audit_identifies_excluded_static_scope():
    from scripts.audit_policy_scope import audit
    a=audit()
    assert a['all_contracts_match']
    assert len([c for c in a['cases'] if c['case']=='unrecognized-static'])==2


def test_malformed_recorded_eligibility_is_rejected():
    import pytest
    trace=run_workload(workload(),FixtureChatProvider())
    trace['events'][0]['eligibility']['cacheable']='yes'
    with pytest.raises(ValueError,match='eligibility'):diagnose(trace)
