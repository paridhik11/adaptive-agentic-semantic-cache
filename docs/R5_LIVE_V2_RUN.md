> **Stopped collection:** the user reported v2 attempt-budget exhaustion. Do not
> repeat the live command below. Follow [the read-only export instructions](R5_PARTIAL_EXPORT.md)
> to recover partial evidence. The original v2 instructions are retained below.

# Revised R5 live pilot: preserve the paused run

The v1 token-limit response is failed development evidence, not a retryable quota
pause. Do not delete it, rerun v1, change its settings or reset its journal.
The original runner/protocol and historical experiment artifacts remain intact.

The v2 runner reads the predecessor without modification and deducts its attempts
from the combined 16-attempt cap. It rejects missing/uncertain predecessor records.
Same model, free-only transport, reasoning off; concise answer instruction and
4096-token allowance. This may still fail; no live success is promised.
See [the prospective revision protocol](R5_LIVE_V2_PROTOCOL.md).

In the same PowerShell session and main checkout used for the paused run:

```powershell
git pull --ff-only
$pilotFolder = "C:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache-pilot"
$py = "$pilotFolder\.venv\Scripts\python.exe"
$journal = "$pilotFolder\runs\openrouter-free.sqlite"
& $py scripts/run_r5_paired_v2.py --corpus runs/r5-corpus/corpus.json --output-dir runs/r5-live-v2 --model nvidia/nemotron-3-super-120b-a12b:free --journal $journal --execute
```

Stop if pull fails. This reuses the already-prepared corpus and existing environment
key; no key is displayed or accepted on the command line. Omit --execute for an
optional read-only plan that makes no inference calls and does not read the key.
The revised run ID is fixed. Do not substitute a different journal.

If quota/rate-limit PAUSED with resume_allowed_after_retry_time true, wait until
retry_at and repeat the exact v2 invocation. All other pauses require inspection;
share the diagnostic output and stop. Old attempts remain counted across resumes.
If another truncation occurs, retain it and stop rather than retuning repeatedly.

On completion upload runs/r5-live-v2-evidence.zip. It contains the revised answers,
trace, summary, blinded review template, private arm mapping and predecessor
metadata (including response hashes and usage, but not predecessor answer text).
The predecessor raw response stays in your local journal. The summary separately
reports prior and revised attempts. A resumed run cannot supply uninterrupted
live-latency measurements. A blinded reviewer receives only reviews.template.json.

Developer offline reproduction (not required on the user's machine):

```bash
python -m pytest tests/test_r5_revision.py tests/test_free_pilot.py tests/test_paired_runtime.py -q
python -m pytest -q
```
