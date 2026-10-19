# R5 live pilot repair

The reported v1 request exhausted its output allowance. Replaying its saved
response would pause again. This repair adds a separate, explicitly versioned
live-development runner; the old runner, protocol, raw datasets and historical
R5/R6 experiment results are unchanged.

## Changes

- `scripts/run_r5_paired_v2.py`: off/exact collection using the same pinned source
  and model, explicit revised profile, predecessor linkage and evidence export.
- `src/benchmark/r5_revision.py`: read-only predecessor inspection, immutable
  snapshot binding, combined attempt accounting, revised prompts, pause guidance.
- `tests/test_r5_revision.py`: isolated profile, journal and full CLI fixture tests.
- `docs/R5_LIVE_V2_PROTOCOL.md` / `docs/R5_LIVE_V2_RUN.md`: prospective revision
  and Windows instructions. Current status, facts and delivery estimates updated;
  original live instructions retained with a superseding pointer.

The failed predecessor remains in the user's journal. Its attempts reduce the
revised run's allowance so the combined cap stays at 16. Missing, altered,
uncertain or wrong-model predecessor records fail closed. Successful collection
exports predecessor response hashes/usage separately from revised results. No
private predecessor answer text or journal is published. No hidden model retry,
paid fallback, quota reset or acceptance of truncated responses is introduced.

## Validation and limits

See `R5_LIVE_V2_VALIDATION.json` for exact offline commands/results. The CLI fixture
exercises a real journal, the gateway/cache arms, quota pause, reconstruction,
ZIP export, predecessor preservation and exclusion of a dummy credential. It is
not a live model experiment. The user still needs to execute the revised pilot;
complete live answers, independent review, semantic safety and net utility remain
unproven. Another failure must be retained rather than silently retuned away.

The v1 pause message is retained in its historical runner. The superseding v2
runner offers resume advice only for quota/rate-limit pauses. Do not resume v1
under changed code: the journal's code fingerprint deliberately rejects that.

Reproduce focused checks:

```bash
python -m pytest tests/test_r5_revision.py tests/test_free_pilot.py tests/test_paired_runtime.py -q
```

Reproduce all default offline regressions:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python scripts/check_numeric_tokens.py
git diff --check
```
