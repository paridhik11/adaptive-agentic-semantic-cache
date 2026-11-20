# Semantic reuse upgrade evaluation

The historical 4,000-execution campaign and evidence seals remain the reference for the four historical arms. This upgrade uses a distinct fixture replay of the live chat API. Freeze the authored fixture before evaluating. Group the examples by task and source. Use development groups for implementation. Close the implementation before first running the reserved groups. Keep both results using the same inputs and offline provider. Report the reserved groups as authored engineering cases. They do not represent independent human review or a traffic population.

For each case, prime an isolated cache with its source question and answer. Submit an exact repeat and a labeled probe. Each probe starts from the same cache state. Label answer suitability by the requested output and the stored answer, including hard negatives with unchanged keywords. Run four arms:

- No cache bypasses storage and lookup.
- Exact uses the live exact store and disables semantic lookup.
- Guarded uses the live cache routing with the default local rules. Separate targeted regressions exercise multiple candidates using deterministic fixture models.
- Shadow computes the same guarded decisions while using the exact answer or fresh fixture response for serving.

Count actual offline transport invocations and distinguish correct hits, unsuitable hits, and hypothetical shadow decisions. Seeding is excluded from reported savings. Latency uses the handler’s recorded wall time. The fixture returns in process. The small replay timings are descriptive and the runs shared a host with other checks. Record local verification time where instrumentation supplies it. Synthetic tokens describe the scripted fixture responses. Re-run the pinned ComQA protocol unchanged for comparison. Broader embeddings and alternative answer verifiers are research candidates. Enable scope-specific routing only when the evidence supports it.

The historical sealed campaign and the ComQA protocol retain their original metrics and thresholds. The upgrade leaves the documentation workbench and production gateway defaults conservative.
