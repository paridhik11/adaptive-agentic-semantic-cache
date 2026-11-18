"""Prepare policy-blind authoring slots; never manufacture independent labels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('refusing to overwrite a review packet')
    root=Path(__file__).resolve().parents[1]
    reference_path=root/'src/api/demo_reference.json'
    reference=json.loads(reference_path.read_text())
    sources=[{'seed_id':d['id'],'original_question':d['title'],
              'cached_answer':d['text'],'answer_origin':'verbatim bundled source excerpt; NOT new LLM output',
              'source_url':reference['source_url'],'source_revision':reference['source_revision']}
             for d in reference['documents']]
    rows=[]
    for seed in sources:
        for index in range(10):
            rows.append({'item_id':f'{len(rows)+1:03d}','seed_id':seed['seed_id'],
                'authoring_request':'equivalent intent' if index<5 else 'plausible mismatch or added requirement',
                'new_question':None,'answer_sufficient':None,'relation':None,'rationale':None,
                'author_id':None,'reviewer_id':None,'evidence_type':None})
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    (args.output_dir/'requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (args.output_dir/'SOURCE_LICENSE.txt').write_bytes((root/'src/api/demo_reference.LICENSE').read_bytes())
    instructions='''# Independent wording and answer-applicability request

Write the 30 new questions in requests.jsonl yourself, using sources.json.
Do not inspect classifier code, thresholds, fixture approvals or predictions.
Use natural wording and rearranged meaning, not just spelling changes. For each
source, aim for five equivalent requests and five plausible mismatches: changed
conversion direction, added requirement, different type, changed number base or
index/slice boundary as applicable. Authoring requests are NOT assigned labels.

Judge the actual complete cached_answer against the new question and source:
answer_sufficient=true only if it directly resolves every requested requirement,
is supported by this source, and needs no changed code, facts or extra reasoning.
Set false if a material requirement is absent or contradicted; leave null if
uncertain and explain. A source passage that merely contains related information
is not automatically sufficient. Relation: same_intent, more_specific,
more_general, different_intent or ambiguous. Keep identifiers unchanged.

Fill author_id, reviewer_id and rationale. Prefer a second person for review;
declare if author and reviewer are the same. evidence_type must be human_authored
or model_assisted. Model-assisted work is NOT independent human ground truth.
No fake human identities. Do not include personal data or API keys.

Return the completed file; keep sources unchanged. This is a small purposive
diagnostic of reference-excerpt answers, NOT a final split or representative
savings workload. No actual new LLM-generated answer was collected here. These
three source topics cannot establish broad paraphrase safety. Missing labels
remain null; builder must not fill them and claim independent evidence.
'''
    (args.output_dir/'INSTRUCTIONS.md').write_text(instructions)
    manifest={'schema':'document-review-request-v1','status':'awaiting_external_authorship_and_review',
              'slots':len(rows),'sources':len(sources),'non_null_questions':0,'non_null_labels':0,
              'evidence_type':'unlabeled authoring request; no quality observations',
              'source_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output_dir.iterdir())}}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
