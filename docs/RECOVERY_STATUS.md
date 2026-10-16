# Engineering recovery status

## Implemented

- Shared scoped exact-response runtime: complete-request and trusted tenant keys,
  explicit eligibility, admission, TTL, bounded storage, coordinated concurrent
  misses, and invalidation. Semantic matching is disabled in this runtime.
- Historical mocked load runner isolated from model downloads and API access.
- Four Antigravity Phase A commits imported from the user-supplied git bundle
  and reconciled with the exact-response runtime. Both ledger sections retained.
- Offline Phase A artifact auditor and candidate-label integrity regressions.
  See `audit/PHASE_A_SOURCE_AUDIT.md` and the separate reconciliation JSON.

- Runnable authenticated non-streaming HTTP gateway, live HTTP adapter, explicit
  simulated provider, tenant invalidation, concurrency limits, and Prometheus
  work counters. Entry and UTF-8 payload budgets bound stored responses.
- Gateway tests and a loopback HTTP smoke test with a barrier-synchronized
  50-client burst pass. These are fixture checks, not real-model savings.

- Optional SQLite persistence with restart recovery, deployment-version guards,
  transactional cross-worker invalidation, logical storage limits, and explicit
  storage-failure behavior. Cross-worker miss coalescing is not implemented.

- Answer-bound development traces, paired fresh/cache work accounting, strict
  review bindings, missing-label bounds, grouped uncertainty and offline review UI.
- Existing development-pair conversion with provenance and normalized-text groups;
  larger fixture-only exercise recorded in `audit/ANSWER_BENCHMARK_FUNCTIONAL_EVIDENCE.json`.
- Per-worker operations dashboard, gateway-only image/Compose files, container CI
  smoke job and corrected installable package layout. The remote container smoke
  job passed; local browser-render validation remains unavailable. See `deployment.md`.

- Experimental scoped similarity/verified retrieval in the development runner,
  lexical and pinned-local dense encoders, and explicit verifier-work accounting.
- Streaming text SSE bypass with bounded parsing, completion validation, slot
  cleanup, cancellation/error metrics and no partial-response admission.
- Bounded specialist/reviewer workflow with tenant/context scope and explicit
  review states; two-reviewer answer reconciliation keeps conflicts unknown.

## Evidence limits

Saved Phase A counts reconcile, but off-seed labels, repeated seed requests,
protocol/runner differences, missing checkpoint weights, and incomplete split
provenance prevent candidate-aware safety certification. No sealed inference
rerun was performed; no historical telemetry or evaluation summary was edited.
The historical Gate A failure remains on record. Fixture tests do not establish
semantic accuracy, answer quality, or measured workload savings.

## Next implementation work

- Measure streaming with a configured live provider and defined concurrent workload; the updated container fixture smoke passed.
- Collect independently reviewed live-answer evidence using the implemented evaluation contract.
- Calibrate and compare experimental semantic policies using independently reviewed real answers before a new frozen evaluation.
- Measure complete-answer latency and net generation/verification work.

## Publication

GitHub access was connected successfully and the initial three recovery commits
were published to main. Source trees from the Phase A bundle are replayed through the connector with
original-source commit trailers; see `audit/PHASE_A_IMPORT_PROVENANCE.json`.
The integration retains both Phase A and recovery code. Published commit metadata
differs from the original local history; do not equate their commit IDs.
