# Standing project rules

User instructions take precedence. Read docs/CURRENT_STATUS.md, docs/FACTS.md,
the active phase closeout/protocol and relevant source before changing behaviour.

- Numbers must trace to FACTS.md rows or immutable raw telemetry. Fix or flag a
  document that disagrees with its evidence; never change data to fit prose.
- Label evidence measured-live, replay, simulated, projected or historical
  component measurement. Report denominators and exact confidence intervals for
  binomial error/accuracy rates, including the upper bound for zero errors.
  Disclose dependence/grouping. Latency percentages are not binomial rates;
  do not invent intervals for singleton stage timings. Deterministic fixture
  counts are functional evidence, not population safety estimates.
- Missing labels remain null. Model-assisted review is not human ground truth.
  Human identities/independence are declared unless separately verified.
- Preserve spent/sealed tests and their original labels/results. Do not edit,
  relabel or rerun them to manufacture improvement. New generalization claims
  require fresh data and a protocol defined before evaluation. A declared
  development replay can test an engineering change but is not a fresh test.
- Never change a gate or threshold to obtain a pass. Report failures plainly.
- Never print, hardcode, paste or pass secrets in command arguments. Use the
  environment. Scan each diff for key patterns and disclose scan scope.
- Live inference uses free-only transport and the shared durable quota journal.
  No paid fallback, hidden retries or quota resets. Offline fixtures need no key.
  No live calls are authorized for the claim-hygiene/preregistration task.
- Work on a task branch. Commit/push only when authorized; the user's instruction
  to push each completed task in this session supplies that authorization. Do
  not push main or merge this audit branch without further instruction. Earlier
  main-only runbooks do not override the user's current branch workflow.
- No application-level latency/cost percentages in README/resume without paired
  end-to-end evidence against fresh generation on a representative workload.
  Historical stage measurements must name their baseline and sample sizes in
  the bullet itself; they are not a headline application benefit.
- Do not publish raw answers, private review mappings or user data by default.
  Prefer aggregate artifacts and reproducible scripts; respect approval review.
- Finish each task with files changed, measurements/checks, unknowns and exact
  reproduction commands. Commit complete coherent tasks, not micro-edits.
- For R6 agent-cache work, first publish docs/audit/R6_AGENT_CACHE_PREREG.md,
  then stop for the user's explicit go. Do not implement or run its evaluation
  before that go. This is a user-requested review checkpoint, not a safety gate.
