# Stage A local commit progress — approval required for oversized artifact

## 1. Branches and commits

Base: `9cb90830aa325253a6abb2e2a69f47529d7efafa`.

```text
6cea9ec (stage-a/harness) research(stage-a): preserve pinned intent baseline harness and hash checks
ec77788 (stage-a/data) research(stage-a): record authored splits and reproducible final lock
```

`stage-a/results` currently points at the data commit. Its results commit is NOT
created. `stage-b/classifier` is NOT created. No push or merge occurred.

## 2. Files

Harness commit: `.gitattributes`, `.gitignore`, the two Stage A configs, Stage A
build/download/inventory/baseline/report/backup/regeneration scripts,
`src/evaluation/intent_stage_a.py`, `tests/test_intent_stage_a.py`.
Data commit: Stage A train/dev/calibration question and answer stores, metadata
completion receipts, aggregate manifest, final manifest, `docs/FACTS.md` and
`docs/STAGE_A_PROTOCOL.md`. Final question/answer stores are excluded.
Pending results: `artifacts/stage_a`, `docs/audit/STAGE_A*`,
`docs/CURRENT_STATUS.md`. All remain unstaged. The backup ZIP is excluded.

## 3. Exact verification commands

Run from the repository root. A new receipt path is required on repetition.

```bash
python scripts/verify_stage_a_regeneration.py --receipt artifacts/stage_a/final-regeneration-v1.json
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_intent_stage_a.py
git log --oneline --decorate -3 stage-a/harness
git log --oneline --decorate -3 stage-a/data
git log --oneline --decorate -3 stage-a/results
```

The verifier copies the exact generator into an isolated temporary source tree
and runs `python scripts/build_stage_a_intents.py --output-dir regenerated`.
It hashes final output, never prints or evaluates final records.

## 4. Checks and evidence types

| Check | Result | Evidence type |
| --- | --- | --- |
| Final manifest and both stores regenerate byte-for-byte | YES | deterministic hash check, not model evaluation |
| Manifest SHA-256 after attributes change | `1d83b84584b62fc1d5662a45e0db7d5753f5871fa273dadb62f1f3a7b01e39af` | deterministic hash check |
| Focused tests before local commits | 9 passed in 5.32s | engineering verification, not accuracy |
| Decision-log lossless gzip check | 10,380,608 bytes → 244,844 bytes; decompression equals original | deterministic storage check |

The requested post-results-commit test run is still pending because that commit
is blocked. No new accuracy, coverage, latency or model-safety result is claimed.
The original generator uses deterministic ordered enumeration, not randomness.

## 5. Failures and unknowns

The raw `artifacts/stage_a/verified/decisions.jsonl` exceeds the 5 MB commit limit.
Its SHA-256 is `642ce3477d21539134b00a52ba663470242ec6e8bc539169778f7ed24bee601e`.
It is NOT staged or committed. No other Stage A candidate file exceeds 5 MB.
The prepared gzip has SHA-256
`9b343a9e90378a998680f1335a0763716e92d1068946b56c110491dba1ef6699`.
No independent human ground truth exists. Final reproducibility does not make
the public generator cryptographically secret. Stage B feasibility is untested.

## 6. Minimum owner action

Choose whether to commit the lossless gzip copy and exclude the oversized raw
copy from Git (recommended), or explicitly authorize committing the raw file
as an exception. The user's Part 1c explicitly requires asking about any file
over 5 MB. This is the sole blocker to the results commit and then Stage B.
No public-dataset approval is requested yet; B0 has not begun.

## 7. Not done

No Stage B training or evaluation, natural-set authorship, public-data download,
final evaluation, provider/live call, UI/agent/serving change, push, merge, rebase
or force operation. General semantic serving remains disabled. No weights or
environments were staged. No final records were inspected; final hashing and
regeneration were the only permitted handling.

## 8. Backup

Download `artifacts/stage_a/stage-a-local-publication-backup.zip` before reset.
It contains a Git bundle for both committed branches and all Stage A working
files, including both excluded final stores, raw decision log, gzip candidate,
receipts and this report. A per-file SHA-256 manifest is inside. It excludes
pretrained weights and the Python environment, which remain reproducible from
the pinned download receipts and scripts. Restore the bundle with
`git clone -b stage-a/data stage-a-local.bundle restored-repo` and copy the archive working files
into that checkout; use `git switch stage-a/data` to select the data tip.
The original `stage-a-backup.zip` is preserved separately.

## Approval update

The owner approved committing the compressed log and proceeding. The raw
10,380,608-byte duplicate is now gitignored. The gzip is lossless and below the
5 MB limit. Earlier pending-status sections above are historical preparation
records; see `STAGE_A_PART1_REPORT.md` for the completed Part 1 report.

To recover the raw log:

```bash
python -c "import gzip,pathlib; p=pathlib.Path('artifacts/stage_a/verified/decisions.jsonl.gz'); p.with_suffix('').write_bytes(gzip.decompress(p.read_bytes()))"
```
