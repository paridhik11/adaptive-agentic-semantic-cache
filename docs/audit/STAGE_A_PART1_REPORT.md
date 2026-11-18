# Stage A local publication report

## 1. Local branches and commits

```text
6cea9ec stage-a/harness research(stage-a): preserve pinned intent baseline harness and hash checks
ec77788 stage-a/data research(stage-a): record authored splits and reproducible final lock
699223a stage-a/results research(stage-a): preserve baseline evidence and lossless decision log
```

`stage-b/classifier` was created from `stage-a/results`. No push or merge.

## 2. Files

The requested harness, data and results layers are committed. `.gitattributes`
protects hashed Stage A bytes and enforces LF on FACTS. `.gitignore` excludes
final stores, backup ZIPs and the oversized raw decision-log duplicate. The
owner approved its byte-identical compressed copy. Pretrained weights, caches
and environment are absent from the commits. See the three commit file lists.

## 3. Reproduction

```bash
python scripts/verify_stage_a_regeneration.py --receipt artifacts/stage_a/final-regeneration-repeat.json
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_intent_stage_a.py
git log --oneline -1 stage-a/harness
git log --oneline -1 stage-a/data
git log --oneline -1 stage-a/results
```

Regeneration uses the original generator in an isolated temporary tree. No
random seed exists: ordered enumeration is deterministic. Only final hashes and
metadata were inspected. Both final stores remain in backups, outside Git.

## 4. Checks

| Result | Evidence type |
| --- | --- |
| Final manifest and both final stores regenerate byte-for-byte: YES | deterministic hash check |
| Manifest SHA `1d83b84584b62fc1d5662a45e0db7d5753f5871fa273dadb62f1f3a7b01e39af` unchanged | deterministic hash check |
| Required test run after results commit: 9 passed in 4.57s | engineering verification, not accuracy |
| Gzip 244,844 bytes decompresses to identical 10,380,608-byte log | lossless storage verification |

Raw test output: `artifacts/stage_b/part1-tests.txt`. Existing model results are
unchanged and remain authored/template development evidence, not human truth.

## 5. Failures and unknowns

No passing semantic quality gate follows. Stage A templates were shared across
splits; independent natural-language generalization remains untested. Public
generator reproducibility is not cryptographic secrecy.

## 6. Blockers / owner action

Oversized-file choice resolved by explicit user approval. No Part 1 blocker
remains. B0 public-data proposals and a blank human-review request are separate.

## 7. Not done

No push, merge, main rebase, force operation, provider inference, final evaluation
or serving enablement. The parsed-call comparator remains unavailable.

## 8. Backup

The pre-results snapshot `artifacts/stage_a/stage-a-local-publication-backup.zip`
preserves the excluded final stores and prior local commits. A new Stage B
closeout archive will include all branches, learned tensors and current files.
