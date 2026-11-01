"""Real pinned-model diagnostic. No serving, fitting, downloads or threshold search."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import importlib.metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('refusing to overwrite model evidence')
    root=Path(__file__).resolve().parents[1]
    config=json.loads(a.config.read_text())
    questions=[json.loads(r) for r in (root/'data/document_product_review/human_v1/human-responses.template.jsonl').read_text().splitlines()]
    sources=json.loads((root/'data/document_product_review/v1/sources.json').read_text())
    import torch
    torch.set_num_threads(config['threads'])
    from src.policy.encoders import DenseEncoder
    from src.policy.local_models import LocalPairScorer
    encoder=DenseEncoder(config['encoder']['repo'],config['encoder']['revision'])
    pair=LocalPairScorer(config['pair_model']['repo'],config['pair_model']['revision'])
    vectors={s['seed_id']:encoder.encode(s['original_question']) for s in sources}
    results=[]
    for row in questions:
        vector=encoder.encode(row['new_question']);candidates=[]
        for source in sources:
            scored=pair.score(row['new_question'],source['original_question'])
            candidates.append({'seed_id':source['seed_id'],
                'answer_sha256':hashlib.sha256(source['cached_answer'].encode()).hexdigest(),
                'dense_cosine':sum(x*y for x,y in zip(vector,vectors[source['seed_id']])),**scored})
        dense=sorted(candidates,key=lambda c:(-c['dense_cosine'],c['seed_id']))
        pairs=sorted(candidates,key=lambda c:(-c['pair_score'],c['seed_id']))
        combined=(pairs[0]['dense_cosine']>=config['dense_threshold'] and
                  pairs[0]['pair_score']>=config['pair_threshold'] and
                  pairs[0]['pair_score']-pairs[1]['pair_score']>=config['pair_margin'])
        results.append({'item_id':row['item_id'],'human_id':row['human_id'],'question':row['new_question'],
            'candidates':candidates,'recommendations':{
                'dense_only':dense[0]['seed_id'] if dense[0]['dense_cosine']>=config['dense_threshold'] else None,
                'pair_only':pairs[0]['seed_id'] if pairs[0]['pair_score']>=config['pair_threshold'] else None,
                'combined_margin':pairs[0]['seed_id'] if combined else None}})
    a.output_dir.mkdir(parents=True)
    predictions=a.output_dir/'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(r)+'\n' for r in results))
    manifest={'evidence_type':config['study'],'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
        'rows':len(results),'model_pair_forward_passes':len(results)*len(sources),
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,
                   'sentence_transformers':importlib.metadata.version('sentence-transformers'),'threads':config['threads']},
        'human_labels_used_for_scoring':False,'serving_enabled':False,
        'warning':'Similarity recommendations do not establish actual-answer applicability.',
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['src/policy/local_models.py','src/policy/encoders.py','scripts/score_document_diagnostic.py']}}
    (a.output_dir/'prediction-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'predictions':len(results),'pair_forward_passes':manifest['model_pair_forward_passes'],
        'recommendations':{name:sum(r['recommendations'][name] is not None for r in results) for name in results[0]['recommendations']}}))


if __name__=='__main__':main()
