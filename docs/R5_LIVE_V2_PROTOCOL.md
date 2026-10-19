# R5 revised development collection — v2

Defined October 20, 2026 before any v2 live call. This is a development repair,
not a fresh sealed evaluation or a change to a passed/failed quality gate.
The original R5 protocol, runner, replay artifacts and R6 criterion remain intact.

The user reported that r5-live-v1 paused on output_token_limit: finish_reason
length, 1024 completion tokens, 225 prompt tokens, 3869 assistant characters.
Response hash ac58df58753bd6aeb9d757d7cd2da6a813c62b0ca89a438cd47b00d700c5bbd8.
This is user-supplied diagnostic evidence; the local journal has not been uploaded
or independently inspected here. Total predecessor attempts are unknown until
read from that journal. No complete-answer or quality claim follows.

Use the same model and pinned development questions, order and off/exact arms as
v1. Add an explicit instruction to answer in at most 180 words, with only relevant
details. Raise the output allowance to 4096 tokens (an allowance, not an expected
answer length). Request reasoning off. Both arms receive exactly the revised
request. Provider free-only controls and complete-response admission stay intact.
A stop finish reason establishes transport completeness, not factual correctness;
independent review remains necessary. No automatic fallback or retry.

Run ID is fixed to r5-live-v2. Require the existing r5-live-v1 record in the same
shared journal, with a final length-truncated response from the same model and
no uncertain/error attempts. Read predecessor rows without modifying them. Bind
its fingerprint, full-row snapshot hash and all attempt counts into v2's config.
Any predecessor change prevents v2 resume. Preserve old response bytes in the
journal; export only predecessor diagnostics, hashes and usage into the v2 ZIP.

The combined v1+v2 allowance stays at 16 attempts: v2 cap is 16 minus all saved
v1 attempts. This may be insufficient to finish; then report budget exhaustion
and retain partial evidence, without increasing the cap. All runs still share
40 attempts per UTC day and 10 per minute. Do not run the old v1 command in
parallel, change run IDs or split journals. Other account usage is not visible.

Resume only quota/rate-limit pauses, at the displayed retry time. Saved incomplete,
failed or uncertain responses require inspection, not blind reruns. Reconstructing
from saved calls invalidates uninterrupted live latency even if collection finishes.
A new truncation stops v2; do not silently retune again. Completed output includes
blinded review templates; missing labels remain null. Raw answers stay private.

Offline validation must cover profile isolation, immutable predecessor accounting,
quota sharing, cap exhaustion, malformed/uncertain predecessor rejection, rejection
of truncated responses on replay, and quota-only resume advice. Fixture tests are
engineering evidence, not live success or quality results.
