> **October 20 update:** the reported v1 run stopped on a truncated answer.
> Preserve it; do not rerun that command. Use [the revised v2 instructions](R5_LIVE_V2_RUN.md).
> The original procedure below is retained for history.

# R5 zero-budget foreground pilot

The offline development investigation is complete; live collection and independent
review are pending. This command is a small development pilot, not the final safety
or resume-metric evaluation. No API key is available to the hosted workspace.
Run in the updated repository on the Windows machine where you configured it.

```powershell
git pull --ff-only
[bool]$env:OPENROUTER_API_KEY
```

If the last result is false, use the same secure environment setup as before. Do
not paste the key into chat, a command argument, a file or a screenshot.

Prepare the checksummed public source (downloads documentation, not model calls):

```powershell
if (-not (Test-Path runs/r5-corpus/corpus.json)) {
  .\.venv\Scripts\python.exe scripts/prepare_python_faq.py --source-dir runs/r5-source --output-dir runs/r5-corpus --download
}
```

Plan first, without reading the key or making any inference call:

```powershell
.\.venv\Scripts\python.exe scripts/run_r5_paired.py --corpus runs/r5-corpus/corpus.json --output-dir runs/r5-live-v1 --provider free --model nvidia/nemotron-3-super-120b-a12b:free --run-id r5-live-v1
```

Expected: 8 requests, 2 arms, maximum 16 provider attempts. Use the same shared
`runs/openrouter-free.sqlite` journal as prior pilots. If your previous pilot used
a different journal path, add `--journal THAT_PATH` consistently to both commands.
Splitting project activity across independent ledgers would defeat the local cap.

Execute once:

```powershell
.\.venv\Scripts\python.exe scripts/run_r5_paired.py --corpus runs/r5-corpus/corpus.json --output-dir runs/r5-live-v1 --provider free --model nvidia/nemotron-3-super-120b-a12b:free --run-id r5-live-v1 --execute
```

No automatic model fallback, paid prices, hidden retries or waiting. The shared
caps remain 40/day and 10/minute; other account usage may reduce availability.
If PAUSED for minute/day quota, wait until the displayed retry time then run the
exact same command. Do not change the run ID or remove the journal. If PAUSED for
incomplete/failed/uncertain response, preserve the output and inspect the journal
before resuming or revising anything. A reconstruction can retain work/answer
accounting but does not produce valid uninterrupted live-latency measurements.

When completed, `runs/r5-live-v1-evidence.zip` contains answers, trace, summary,
review template and manifest; no API key or quota journal. Share that archive for
analysis. It includes a private arm mapping: a person doing blinded review should
receive only `reviews.template.json`, not the archive/mapping or summary first.
Human review is still pending even when the script says execution completed.

The model's official OpenRouter listing was checked on 2026-10-18:
https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b%3Afree
A listing is not an availability guarantee; transport errors remain evidence.
The free-only transport additionally enforces zero maximum prices.

For reviewer-completed JSON, retain question/reference/answer/ID exactly. Set
`correct` and `supported` to JSON booleans, enter the reviewer's declared name and
notes. Unknowns remain null. Reconcile without overwriting earlier evidence:

```powershell
.\.venv\Scripts\python.exe scripts/review_r5_answers.py --trace runs/r5-live-v1/trace.json --reviews PATH_TO_COMPLETED_REVIEWS.json --output runs/r5-live-v1/reviewed-development.json
```

The reviewer should judge whether the answer addresses the actual question and
is supported by the reference, not merely whether it resembles another answer.
This two-question/two-topic sample cannot establish a tight safety bound or a
representative semantic hit rate. No new dataset search or paid account is needed.
