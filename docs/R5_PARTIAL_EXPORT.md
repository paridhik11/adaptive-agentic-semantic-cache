# R5 stopped collection: recover evidence without more inference

On October 20 the user reported `run_attempt_budget_exhausted` from v2. The
runner's registered allowance was exhausted. The pasted pause does not establish
the predecessor/revision split, complete-answer count, cache hits or cause of the
shortfall. The shared journal is authoritative. Do not reset it, raise caps,
change run IDs or resume a live command to bypass this stop.

The separate exporter opens SQLite in read-only mode and uses one read transaction.
It selects only r5-live-v1 and r5-live-v2. It never reads an API key, instantiates
a provider, reserves attempts or makes network calls. It exports saved response
bodies, diagnostics, request hashes and registered cap/count metadata into a new
private ZIP. Existing evidence is never overwritten. The raw database and other
runs are not included. Failures, uncertain attempts and truncated responses stay
in the accounting; they are not converted into complete answers.

Using the same PowerShell variables and main checkout as the stopped pilot:

```powershell
git pull --ff-only
& $py scripts/export_r5_journal.py --journal $journal --corpus runs/r5-corpus/corpus.json
```

Stop if pull fails. Upload `runs/r5-partial-evidence.zip` for inspection. This is
an offline export, not another live run. Share only `reviews.template.json` with
a blinded reviewer, not the complete ZIP. The private packet includes answers.

Requests are bound by exact hashes to the pinned corpus's original/revised
profiles. Unknown hashes remain unmatched. Seed/repeat requests may match the
same hash, so the exporter does not invent cache-arm or event-order attribution.
Only admissible complete answers with matching request hashes enter the review
template. Labels remain null. This journal-review schema is separate from the
completed paired-trace review schema and is not input to review_r5_answers.py.

This export cannot establish a finished paired run, cache savings or valid live
latency. It recovers partial real answers for inspection/review and determines
where quota was spent. Any future collection requires a separately documented
budget and protocol; this change authorizes no further inference.

## Offline engineering validation

51 focused tests passed in 1.36 seconds, including seven new exporter tests. They
cover exhausted-budget export, byte-identical journal preservation, network-disabled
execution, complete/truncated/unmatched responses, uncertain/error/malformed rows,
missing journal/corpus mismatch, exclusive ZIP creation, manifest verification and
exclusion of a dummy environment credential. No external model calls were made.
The previous full suite result remains historical; it was not rerun for this
additive offline exporter. Existing runtime, live runners and profiles are unchanged.

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest tests/test_r5_journal_export.py tests/test_r5_revision.py tests/test_free_pilot.py tests/test_paired_runtime.py -q
python scripts/check_numeric_tokens.py
git diff --check
```
