"""Authored boundary examples: expected policy behavior, not safety ground truth."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.api.service import ConservativeRulesPolicy
from src.policy.response_cache import ScopedResponseCache


def audit():
    policy=ScopedResponseCache(eligibility=ConservativeRulesPolicy())
    cases=[
        ('recognized-static','What is photosynthesis?',None,0,None),
        ('recognized-cs','What is a binary search tree?',None,0,None),
        ('unrecognized-static','What is binary search?',None,0,'not_rule_confirmed_static'),
        ('unrecognized-static','What is a hash table?',None,0,'not_rule_confirmed_static'),
        ('fixed-system-context','What is photosynthesis?','Answer concisely.',0,'context_requires_review'),
        ('dynamic','What is the weather right now?',None,0,'not_rule_confirmed_static'),
        ('creative','Write a poem about photosynthesis.',None,0,'personal_or_generative_request'),
        ('personal','Explain photosynthesis to person@example.test',None,0,'personal_or_generative_request'),
        ('high-temperature','What is photosynthesis?',None,1,'temperature_not_eligible'),
        ('omitted-temperature','What is photosynthesis?',None,None,'temperature_not_eligible'),
        ('long','photosynthesis '+ 'word '*128,None,0,'long_request')]
    rows=[]
    for kind,text,system,temp,expected in cases:
        body={'model':'fixture-model','max_tokens':256,'messages':[{'role':'user','content':text}]}
        if system:body['messages'].insert(0,{'role':'system','content':system})
        if temp is not None:body['temperature']=temp
        actual=policy._eligibility_reason(body)
        rows.append({'case':kind,'request':body,'expected_policy_bypass_reason':expected,
            'observed_policy_bypass_reason':actual,'contract_matches':actual==expected})
    return {'schema_version':'policy-scope-audit-v1','nature':'authored_contract_examples_not_safety_labels',
        'all_contracts_match':all(r['contract_matches'] for r in rows),'cases':rows,
        'interpretation':'Unrecognized stable topics and fixed system prompts are excluded. Recognized static wording is heuristic eligibility, not independent proof of correctness or safety.'}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
