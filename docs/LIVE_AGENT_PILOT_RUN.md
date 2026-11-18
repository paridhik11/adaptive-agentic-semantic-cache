# First live-agent development comparison

This is a separate prospective agent pilot, using the existing shared quota
journal. It does not resume the exhausted request pilots. Maximum new experiment
allowance: twelve attempts. The runner budgets for zero cache hits and checks
remaining project daily capacity before starting. No live execution occurred in
its preparation. Read [the protocol](LIVE_AGENT_PILOT_PROTOCOL.md).

Use your current main checkout on audit/claim-hygiene-r6-prereg. Reuse the existing
prepared corpus, working Python environment and environment key. Stop if pull fails.

```powershell
git pull --ff-only
$pilotFolder = "C:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache-pilot"
$py = "$pilotFolder\.venv\Scripts\python.exe"
$journal = "$pilotFolder\runs\openrouter-free.sqlite"
& $py scripts/run_live_agent_pilot.py --corpus runs/r5-corpus/corpus.json --journal $journal --execute
```

Omit --execute for an optional read-only plan. Plan does not read the API key or
register a new run. The live command itself performs the same budget preflight.
No audit or test commands are needed on your machine.

- NOT_STARTED / insufficient daily budget: no calls made. Wait until retry_at
  (UTC epoch seconds) and repeat the same command. Do not change journal/run ID.
- PAUSED / resume_allowed true: upload the exported ZIP if you want inspection;
  wait until retry_at before resuming the exact command. Timing after resume is
  marked invalid for uninterrupted live measurements.
- PAUSED / resume_allowed false or FAILED: stop and upload the ZIP. Do not retry
  or change the profile. Saved attempts stay spent.
- COMPLETED_COLLECTION: upload the ZIP and stop. Tasks should be awaiting human
  approval; model workflow review is not independent answer review.

Every collection invocation prints an archive path such as
runs/agent-live-v1/evidence-11-000.zip. Upload the exact file it prints, including
on a pause. Earlier snapshots are never overwritten. The ZIP is private and
contains the selected run's journal responses, not the entire project journal.
Only reviews.template.json goes to a blinded reviewer.

It is possible to get no draft hit if live plans/evidence differ. Preserve that
negative result. The pilot does not establish broad semantic accuracy, population
savings or an end-to-end application latency percentage.
