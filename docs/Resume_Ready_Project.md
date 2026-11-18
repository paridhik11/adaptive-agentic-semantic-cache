# Resume-ready project entry

**LLM Caching Gateway & Agentic Documentation Assistant**  
Python · FastAPI · SQLite · SentenceTransformers

- Built a tenant-scoped response-caching gateway with TTL, invalidation and durable storage; measured **1.79 ms p95 SQLite cache-hit processing across 5,000 warm local requests**.
- Implemented a bounded documentation agent with planning, read-only source tools, review and durable human approval; validated caching and workflow behavior with **320 targeted offline regression tests**.

Optional engineering detail for a longer resume or interview:

- Stress-tested **64 concurrent identical cold requests**, producing **one counted stub-provider invocation** through process-local request coalescing.

## What these metrics mean

The latency number is a measured **local cache component**, not complete HTTP
gateway latency, LLM response time, production latency or percentage savings.
The benchmark uses three warm documentation keys, five runs of 1,000 hits each,
in a managed Linux container. Individual SQLite run p95 values ranged from
1.59 to 2.00 ms. The in-memory path measured 0.040 ms aggregate p95 over another
5,000 hits, but use the more substantial persistent-path result in the resume.
Raw samples, runtime, configuration and source hashes are retained in
`artifacts/resume/cache-benchmark-v1.json`. No artificial provider delay enters
the timed hit path. Seed calls and the separate concurrency wave are not timed.

The concurrency result is an instrumented fixture, not 64 real paid LLM calls.
It tests one local process and does not establish distributed coordination.
The test count is software validation, not 320 correctly answered questions.

## Current product limits

This is a working local engineering prototype, not a validated production
semantic-cache service. General automatic paraphrase approval has not passed:
the exposed real-model diagnostic had two sufficient and one insufficient
dense recommendations; its pair-gated setting recommended none. Do not claim
semantic accuracy, live cost reduction, end-to-end LLM latency savings, public
deployment or independent human-authored benchmark coverage.

The application contains semantic retrieval/review experiments, but the title
above emphasizes delivered engineering rather than an unsupported performance
claim. In an interview, explain the failed applicability study directly.

## Demonstrate it without a key

```bash
python -m pip install -r deploy/requirements-gateway.txt
python scripts/run_demo.py --whole-answer-cache --task-memory
```

Open the loopback URL printed by the launcher. Ask the bundled tuple/list
question, approve its answer, and ask it again. The reference demo uses scripted
responses with real tools, storage and approval; its repeat avoids new scripted
model-stage calls. The mode is labeled. This is not live-model evidence.

Reproduce the benchmark with a new output path:

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/benchmark_resume_cache.py --output NEW_BENCHMARK.json
```

On a normally installed project environment, omit the workspace-specific
PYTHONPATH prefix. Results depend on hardware/filesystem; retain your own receipt.
GitHub publication remains postponed under the user's latest instruction.
