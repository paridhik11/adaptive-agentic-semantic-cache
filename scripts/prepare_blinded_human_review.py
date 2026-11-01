"""Render model-authored questions for human review without model predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def prepare(sources, rows):
    shuffled=list(rows);random.Random(20261101).shuffle(shuffled)
    lines=['# Human answer review','',
        'These questions were written by a model. Its judgments are hidden here.',
        'Your review is human review of model-authored questions, not human-authored wording.','',
        'For each question judge the supplied complete cached answer: **Y** = directly answers every requirement; **N** = misses or contradicts a requirement; **?** = uncertain.',
        'Related information alone is insufficient. Judge the answer as written; do not mentally add missing code or facts.',
        'Reply in chat, e.g. `H01 Y, H02 N — missing requirement, H03 ?`. You can send one section at a time.',
        'Add a short reason for N or ?. Do not inspect the separate model review until finished.','']
    mapping=[];template=[];number=0
    for source in sources:
        lines+=['## '+source['original_question'],'','Cached answer:',
                '```text',source['cached_answer'],'```','']
        for row in shuffled:
            if row['seed_id']!=source['seed_id']:continue
            number+=1;human_id=f'H{number:02d}'
            lines+=[f"**{human_id}.** {row['new_question']}",'', 'Your judgment: ___','']
            mapping.append({'human_id':human_id,'item_id':row['item_id'],
                'question_sha256':hashlib.sha256(row['new_question'].encode()).hexdigest(),
                'answer_sha256':hashlib.sha256(source['cached_answer'].encode()).hexdigest()})
            template.append({'human_id':human_id,'item_id':row['item_id'],'seed_id':row['seed_id'],
                'new_question':row['new_question'],'question_origin':'model_authored',
                'answer_sufficient':None,'relation':None,'rationale':None,
                'author_id':row['author_id'],'reviewer_id':None,'evidence_type':'human_review_of_model_authored'})
    assert number==len(rows)
    return '\n'.join(lines),mapping,template


def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--model-responses',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('refusing to overwrite human packet')
    sources=json.loads((a.packet/'sources.json').read_text())
    rows=[json.loads(line) for line in a.model_responses.read_text().splitlines()]
    text,mapping,template=prepare(sources,rows)
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'Human_Answer_Review.md').write_text(text)
    (a.output_dir/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (a.output_dir/'human-responses.template.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in template))
    manifest={'questions':len(rows),'human_labels':0,'question_origin':'model_authored',
        'model_judgments_in_human_form':False,'blinding':'withheld in delivered form; not independently verified',
        'model_responses_sha256':hashlib.sha256(a.model_responses.read_bytes()).hexdigest(),
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output_dir.iterdir()}}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
