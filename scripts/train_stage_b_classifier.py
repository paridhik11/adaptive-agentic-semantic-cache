"""B1: train three prespecified seeds on train only; export all learned tensors."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage_b_common import config, read_split, identity, snapshot, offline, hardware, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/stage_b/models')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Refusing to overwrite trained models')
    cfg = config()
    # This executable does not accept an eval split and never loads dev/cal.
    data = read_split('train')
    cached = {r['cached_case_id']: r for r in data if r['role']=='cached'}
    rows = [r for r in data if r['role']=='new' and not r['is_exact'] and r['relation_to_cached'] is not None]
    for name in ['src/evaluation/intent_stage_b_model.py', 'scripts/stage_b_common.py', 'scripts/train_stage_b_classifier.py']:
        identity(ROOT/name)
    base = snapshot('quora')
    offline()
    import torch
    from safetensors.torch import save_file
    from src.evaluation.intent_stage_b_model import construct, learned_state, tokenize_pairs, load_adapter
    torch.set_num_threads(cfg['torch_threads'])
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True)
    counts = Counter(r['relation_to_cached'] for r in rows)
    weights = torch.tensor([len(rows)/(len(cfg['classes'])*counts[c]) for c in cfg['classes']])
    reports = []
    for seed in cfg['seeds']:
        start = time.perf_counter()
        model, tok = construct(base, cfg, seed)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        generator = torch.Generator().manual_seed(seed)
        history = []
        for epoch in range(cfg['epochs']):
            model.train()
            permutation = torch.randperm(len(rows), generator=generator).tolist()
            loss_sum, correct, total = 0., 0, 0
            for step, begin in enumerate(range(0, len(rows), cfg['batch_size'])):
                batchrows = [rows[i] for i in permutation[begin:begin+cfg['batch_size']]]
                batch = tokenize_pairs(tok, [cached[r['cached_case_id']]['question'] for r in batchrows],
                                       [r['question'] for r in batchrows], cfg['max_tokens'])
                labels = torch.tensor([cfg['classes'].index(r['relation_to_cached']) for r in batchrows])
                logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
                if not torch.isfinite(loss):
                    raise ValueError('STOP: nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, cfg['gradient_clip'])
                optimizer.step()
                loss_sum += loss.item()*len(batchrows)
                correct += int((logits.argmax(-1)==labels).sum())
                total += len(batchrows)
                if step % 20 == 0:
                    print(json.dumps({'seed': seed, 'epoch': epoch+1, 'step': step,
                                      'loss': loss.item(), 'elapsed_seconds': time.perf_counter()-start}), flush=True)
            history.append({'epoch': epoch+1, 'weighted_loss': loss_sum/total,
                            'online_train_four_class_accuracy': correct/total, 'n': total,
                            'evidence_type': 'training_diagnostic_not_generalization'})
        model.eval()
        # A fixed train-only reconstruction probe verifies exported learned state.
        probe = tokenize_pairs(tok, [cached[rows[0]['cached_case_id']]['question']], [rows[0]['question']], cfg['max_tokens'])
        with torch.inference_mode():
            expected = model(**probe).logits.clone()
        directory = args.output_dir/str(seed)
        directory.mkdir()
        path = directory/'adapter.safetensors'
        save_file(learned_state(model), str(path), metadata={'base_revision':cfg['base_revision'],'seed':str(seed),'format':'LoRA_and_four_class_output_head'})
        train_count = sum(p.numel() for p in trainable)
        total_count = sum(p.numel() for p in model.parameters())
        del optimizer, model, trainable
        gc.collect()
        restored, _ = load_adapter(base, cfg, seed, path)
        with torch.inference_mode():
            actual = restored(**probe).logits
        if not torch.equal(expected, actual):
            raise ValueError('STOP: safetensors export fails exact train-only reconstruction')
        del restored
        gc.collect()
        report = {'evidence_type':'train_only_synthetic_parameter_efficient_fine_tuning',
            'seed':seed, 'base_repo':cfg['base_repo'],'base_revision':cfg['base_revision'],
            'base_safetensors_sha256':identity(base/'model.safetensors',False)['sha256'],
            'train_data': identity(ROOT/cfg['data_dir']/'train.jsonl',False),
            'config_identity':identity(ROOT/'configs/intent_stage_b_v1.json',False),
            'train_examples':len(rows),'class_counts':dict(counts),'trainable_parameters':train_count,
            'total_parameters':total_count,'wall_seconds':time.perf_counter()-start,'hardware':hardware(),
            'history':history,'export':identity(path),'exact_reconstruction_probe_passed':True,
            'export_is_standalone':False,'loading_recipe':'src.evaluation.intent_stage_b_model.load_adapter(base_snapshot, config, seed, adapter_path)',
            'command':'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/train_stage_b_classifier.py',
            'dev_or_calibration_read':False,'final_read':False,'remote_inference_calls':0,
            'provenance_flags':cfg['provenance_flags']}
        write_json(directory/'receipt.json',report)
        reports.append(report)
        write_json(ROOT/'artifacts/stage_b/training-v1.json',{'all_seeds':reports,'config':cfg})
        print(json.dumps({'seed_completed':seed,'seconds':report['wall_seconds'],'export_sha256':report['export']['sha256']}),flush=True)


if __name__ == '__main__':
    main()
