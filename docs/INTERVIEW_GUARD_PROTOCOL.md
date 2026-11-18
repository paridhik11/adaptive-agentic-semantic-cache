# Profiling requirement experiment

**Question.** Does the existing candidate cProfile check catch an added profiling-tool requirement while allowing equivalent requests through the requirement stage?

**Scope.** The unit is a supplied candidate prompt, an incoming request and a fixed candidate answer. Compare current requirement checks with the already-written candidate check in [interview_guard_experiment.py](../scripts/interview_guard_experiment.py). Keep the known asyncio failure and its original authored controls in development. The fresh authored fixtures use distinct base intent groups and exclude asyncio and generic Python-function profiling. No retrieval, neural scoring, model calls or serving changes are part of this experiment.

**Design frozen before evaluating.** Use eight fresh intent groups. Each gets an equivalent request with a sufficient candidate answer, an added cProfile requirement with a cached generic answer, and an equivalent explicit cProfile request with an answer that omits the tool. Add four challenge cases in which a candidate mentions cProfile but gives no actionable guidance. These probe whether keyword presence is sufficient. The expected decision is to block substituted answers in all three challenge types. Expectations come from the fixture author. They represent functional requirements for the examples.

Compare known failure and fresh fixtures separately. For fresh fixtures, report correctly blocked cases and the challenge denominator, blocked adequate cases and the adequate denominator, newly prevented unsafe passes, lost adequate passes, and remaining unsafe passes by type. Disclose grouping and the known dependence between variants. Treat counts as functional evidence from authored cases. There is no statistical sampling frame or independent adjudication. Do not change the check or relabel cases after running this dataset. Publish outcomes even if the check misses a challenge.

Use the final guide for the reproduction command. The candidate remains experimental. Exact remains the release serving default.

Frozen development script SHA-256: `260585f1f5796656be9aef5c90c58c7fbc17fcf41a2cd3584b62071f689e90fe`
