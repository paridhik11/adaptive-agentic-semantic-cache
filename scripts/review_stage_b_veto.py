"""B2 dev-only veto diagnostics and quarantine of unverified Stage A metadata."""
from collections import Counter
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_b_common import read_split, identity, write_json, write_jsonl


def main():
    identity(ROOT/'src/evaluation/intent_stage_b_veto.py')
    from src.evaluation.intent_stage_b_veto import veto
    import json
    path=ROOT/'data/intent_stage_a/dev.jsonl'
    identity(path)
    old=[json.loads(x) for x in path.read_text().splitlines()]
    flagged=[]
    for r in old:
        sig=r.get('intent_signature') or {}
        text=json.dumps(sig).lower()
        # Most completed signatures have no verification_status at all. Their
        # unreviewed label_status is not evidence of a verified typed signature.
        not_verified=r.get('label_status') not in {'independently_verified','human_verified_signature'}
        missing=[key for key in ['direction','version','quantity','units','negated'] if key not in sig]
        if not_verified or missing or any(x in text for x in ['unverified','partial','review','unknown','as specified']):
            flagged.append({'case_id':r['case_id'],'signature_metadata':sig,
                            'label_status':r.get('label_status'),'missing_typed_fields':missing,
                            'review':'builder metadata review: not an independently verified typed contract; excluded from ALL Stage B policy inputs'})
    write_json(ROOT/'artifacts/stage_b/legacy-signature-review.json',{
        'evidence_type':'builder_metadata_audit_not_human_labeling','reviewed_split':'Stage A dev only',
        'flagged_count':len(flagged),'records':flagged,'stage_a_signatures_used_in_policy':False,
        'action':'quarantine all Stage A metadata; no labels changed; parse fresh question text only'})
    rows=read_split('dev')
    cached={r['cached_case_id']:r for r in rows if r['role']=='cached'}
    diagnostics=[]
    for r in rows:
        if r['role']!='new' or r['is_exact']:continue
        result=veto(cached[r['cached_case_id']]['question'],r['question'])
        diagnostics.append({'case_id':r['case_id'],'family':r['case_family'],
            'relation':r['relation_to_cached'],**result})
    counts={}
    for family in sorted({r['family'] for r in diagnostics}):
        rs=[r for r in diagnostics if r['family']==family]
        counts[family]={'n':len(rs),'rejected':sum(r['reject'] for r in rs),
                        'true_positives_wrongly_rejected':sum(r['reject'] and r['relation']=='same_intent' for r in rs),
                        'reasons':dict(Counter(x for r in rs for x in r['reasons']))}
    write_jsonl(ROOT/'artifacts/stage_b/veto-dev-decisions.jsonl',diagnostics)
    write_json(ROOT/'artifacts/stage_b/veto-dev-summary.json',{
        'evidence_type':'synthetic_template_held_out_dev_diagnostic_not_human_truth',
        'view':'designated pairs; classifier-conditional incremental effect reported separately after B3',
        'per_family':counts,'legacy_metadata_used':False,'rule_vocabulary':'builder-authored bounded domain, aligned with synthetic operations; not independent generalization evidence',
        'accepts_possible_from_veto_alone':False})
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
