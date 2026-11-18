# Risk oversight evaluation protocol

## Decision and evidence

Compare exact-only, the current guarded cache, and the same guarded cache with one additional response requirement check. The candidate abstains when an expressly requested evidence source, owner, formula or exception breakdown lacks an explicit labeled section in the cached answer. Both guarded arms use the repository's pinned MiniLM models, 0.60 retrieval threshold, 0.85 verifier threshold, and default top-k. Freeze those thresholds in advance. Calibration is a wiring and data check. It cannot change thresholds for this comparison.

Use invented operational reporting examples. Each intent has a fixed seed prompt and answer. The answer itself determines whether it satisfies a probe, according to its authored requirement and annotated label. This evidence measures functional cache behavior on controlled examples. It is neither an independent review of open-ended LLM answers nor evidence about actual financial exposures.

## Design

Assign six entire intent groups to calibration and 24 to the holdout. Each group contributes a literal repeat, a wording-only paraphrase, a request for an artifact already present in the cached answer, a request for an artifact absent from it, and a material change unrelated to the proposed control. Rotate unrelated changes among entity, period, numeric threshold, scope, and negation. Exact probe duplication within a group is intentional. Reject duplicate group identifiers and normalized prompt overlap across groups, including across splits. Freeze the fixture and policy source hashes before calibration or holdout scoring. Use fresh cache namespaces by intent and policy, prime with the identical fixed answer, and prevent fallback probes from entering the cache. Compare policies on identical probe bytes. Include failures to admit a semantic candidate as part of the measured cache behavior.

Observe the returned response and verify identity against the primed answer or the explicit fallback marker. Record the actual served tier, scripted provider calls, expected reuse disposition, and guard reason. Do not grade fallback answer quality. No provider API calls. One fallback represents one scripted call in this harness.

## Outcomes

Report unsuitable accepted answers divided by semantic hits, eligible non-exact requests correctly reused divided by eligible non-exact requests, gross scripted calls avoided and calls avoided on suitable reuse. Include paired counts of wrong accepts prevented and introduced, and valid reuses lost and gained. Report group counts and exact 95% Clopper–Pearson intervals with raw denominators. Query variants share groups, so event-level binomial intervals are descriptive under the binomial assumption. With zero semantic hits, its incorrect-reuse rate and interval are undefined. Break down exceptions by scenario and show which evidence came from the candidate control. Prefer the candidate for follow-up testing only if it prevents at least one wrong acceptance, introduces zero wrong acceptances and loses zero valid reuses in the holdout. Keep exact as the serving default regardless. Local timing covers cache lookup and verification against an immediate scripted response. A separate paired fresh-generation pilot is needed for end-to-end latency claims.

Publish the frozen input hashes, machine-readable probe receipts and results. Report the holdout as spent for development after the first complete evaluation. Subsequent reproductions replay the frozen experiment.
