"""Assessment logic tests only; no 44-task agent execution occurs here."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('r6_criterion',Path(__file__).resolve().parents[1]/'scripts/run_r6_agent_stage.py')
r6=importlib.util.module_from_spec(spec);spec.loader.exec_module(r6)

def fixture():
    expected=[f'answer{i}' for i in range(44)]
    rows=[{'arms':{mode:{'status':'COMPLETED','answer':text,'provider_calls':3 if mode=='off' or i%2==0 else 2,'observed_provider_calls':3 if mode=='off' or i%2==0 else 2,'observed_tool_calls':2} for mode in ('off','exact')}} for i,text in enumerate(expected)]
    return rows,expected

def test_conjunctive_assessment_pass():
    rows,expected=fixture();assert r6.assess(rows,expected,True)['decision']=='PASS'

@pytest.mark.parametrize('failure',['answer','missing_answer','unknown_count','skipped_task','tools','off_stages_removed','tests','no_gain','baseline'])
def test_any_violation_fails(failure):
    rows,expected=fixture();tests=True
    if failure=='answer':rows[0]['arms']['exact']['answer']='wrong'
    elif failure=='missing_answer':rows[0]['arms']['exact']['answer']=None
    elif failure=='unknown_count':rows[0]['arms']['exact']['provider_calls']=None
    elif failure=='skipped_task':rows.pop()
    elif failure=='tools':rows[0]['arms']['exact']['observed_tool_calls']=1
    elif failure=='off_stages_removed':rows[0]['arms']['off'].update(provider_calls=2,observed_provider_calls=2)
    elif failure=='tests':tests=False
    elif failure=='baseline':expected[0]='other original'
    elif failure=='no_gain':
        for row in rows:row['arms']['exact'].update(provider_calls=3,observed_provider_calls=3)
    assert r6.assess(rows,expected,tests)['decision']=='FAIL'
