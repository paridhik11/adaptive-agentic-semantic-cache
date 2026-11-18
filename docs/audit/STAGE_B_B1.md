# B1 training record

Evidence: synthetic TRAIN-only parameter-efficient fine-tuning. Not generalization or human accuracy.

| Seed | Train examples | Trainable / total parameters | Wall seconds (includes export/reload check) | safetensors SHA-256 |
| --- | --- | --- | --- | --- |
| 11 | 1056 | 150532 / 82268932 | 197.406 | 89685b335847eb052d7bf3ba55f9ace99cbde6e9d08277e0cb16ebeed244b348 |
| 29 | 1056 | 150532 / 82268932 | 222.955 | 1ffa69f9d4c373347b0c2dc12486a5a1929ee632dcc3b8bc6bc6ccfe51dadf73 |
| 47 | 1056 | 150532 / 82268932 | 212.832 | a64fe6ed77c3e6d721271a3c85b22b8ccdf4fe558f41623159a7b362e7e6c42b |

Base: cross-encoder/quora-distilroberta-base @ f62e7a4b20b97195c2868e53ec59126df5eac743. Exact settings: configs/intent_stage_b_v1.json. All three exports reconstructed identical train-probe outputs. No CPU infeasibility or silent task reduction: the prespecified LoRA method ran as configured.

Hardware: {"platform": "Linux-6.18.55-x86_64-with-glibc2.39", "python": "3.12.14", "torch": "2.6.0+cpu", "logical_cpus": 9, "torch_threads": 2, "cpu_model": "AMD EPYC 9V74 80-Core Processor", "memory_total": "MemTotal:       10206484 kB", "device": "cpu"}

Command: `PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/train_stage_b_classifier.py`.

Exports (605,120 bytes each): artifacts/stage_b/models/{11,29,47}/adapter.safetensors. These are all learned tensors, not standalone base models. Reconstruct using src.evaluation.intent_stage_b_model.load_adapter with the exact pinned base. Receipt copies are tracked under artifacts/stage_b/model-receipts/. Learned weights are gitignored and included in the final backup ZIP.

Possible pretraining overlap with public pair data applies to this Quora-derived model. No public data was used. Dev/calibration/final were not opened by training.

Post-training train-only classification diagnostics use scripts/diagnose_stage_b_train.py and will be compared with held-out classification in B5. No checkpoint or seed is selected using these results.
