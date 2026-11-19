-- The historical export is wide: one row per request, four matched policies.
CREATE TABLE historical (
    workload TEXT NOT NULL,
    event_id TEXT NOT NULL,
    request_type TEXT NOT NULL,
    no_cache_calls INTEGER NOT NULL,
    exact_calls INTEGER NOT NULL,
    guarded_calls INTEGER NOT NULL,
    shadow_calls INTEGER NOT NULL,
    guarded_false_accept INTEGER NOT NULL,
    PRIMARY KEY (workload, event_id)
);

CREATE VIEW historical_segments AS
SELECT workload, request_type, COUNT(*) AS requests,
       SUM(no_cache_calls) AS baseline_calls, SUM(exact_calls) AS exact_calls,
       SUM(guarded_calls) AS guarded_calls, SUM(shadow_calls) AS shadow_calls,
       SUM(no_cache_calls - exact_calls) AS exact_calls_avoided,
       SUM(exact_calls - guarded_calls) AS additional_guarded_calls_avoided,
       SUM(guarded_false_accept) AS guarded_challenge_accepts
FROM historical
GROUP BY workload, request_type;

-- Symmetric two-population decomposition in percentage points.
-- Since the outcome is calls avoided / baseline calls, use call shares as weights.
CREATE VIEW exact_gap AS
WITH rates AS (
    SELECT *, 1.0 * baseline_calls / SUM(baseline_calls) OVER (PARTITION BY workload) AS weight,
              1.0 * exact_calls_avoided / baseline_calls AS rate
    FROM historical_segments
)
SELECT a.request_type, a.requests AS repeat_heavy_n, b.requests AS broader_n,
       100 * a.weight AS repeat_heavy_baseline_share_pct,
       100 * b.weight AS broader_baseline_share_pct,
       100 * a.rate AS repeat_heavy_segment_saving_pct,
       100 * b.rate AS broader_segment_saving_pct,
       100 * (a.weight-b.weight) * (a.rate+b.rate)/2 AS mix_gap_pp,
       100 * (a.rate-b.rate) * (a.weight+b.weight)/2 AS within_segment_gap_pp
FROM rates a JOIN rates b USING (request_type)
WHERE a.workload = 'repeat_rich' AND b.workload = 'lower_repeat';

-- The risk holdout export is long: one row per intent, scenario and policy.
CREATE TABLE risk (
    intent TEXT NOT NULL,
    scenario TEXT NOT NULL,
    policy TEXT NOT NULL,
    stratum TEXT NOT NULL,
    eligible INTEGER NOT NULL,
    served INTEGER NOT NULL,
    calls INTEGER NOT NULL,
    PRIMARY KEY (intent, scenario, policy)
);
CREATE VIEW risk_segments AS
SELECT policy, stratum, COUNT(*) AS requests, SUM(eligible) AS eligible_requests,
       SUM(1-calls) AS gross_calls_avoided, SUM(eligible*served) AS suitable_calls_avoided,
       SUM((1-eligible)*served) AS unsuitable_reuses, SUM(calls) AS fallback_requests
FROM risk GROUP BY policy, stratum;

-- Scenario weights sum to one. Hold the eligible share of non-repeats at its
-- observed value. Vary exact-repeat share and trusted context among suitable
-- non-exact requests. The four strata use their observed policy outcomes.
CREATE TABLE scenario_inputs (
    scenario TEXT PRIMARY KEY,
    repeat_share REAL NOT NULL,
    resolved_share REAL NOT NULL
);
CREATE VIEW sensitivity AS
WITH mix AS (
    SELECT 1.0 * SUM(eligible_requests) / SUM(requests) AS eligible_nonexact_share
    FROM risk_segments WHERE policy = 'context' AND stratum != 'exact_repeat'
), weights AS (
    SELECT scenario, repeat_share, resolved_share, 'exact_repeat' AS stratum, repeat_share AS weight FROM scenario_inputs
    UNION ALL
    SELECT scenario, repeat_share, resolved_share, 'resolved_paraphrase',
           (1-repeat_share)*eligible_nonexact_share*resolved_share FROM scenario_inputs CROSS JOIN mix
    UNION ALL
    SELECT scenario, repeat_share, resolved_share, 'unresolved_paraphrase',
           (1-repeat_share)*eligible_nonexact_share*(1-resolved_share) FROM scenario_inputs CROSS JOIN mix
    UNION ALL
    SELECT scenario, repeat_share, resolved_share, 'challenge',
           (1-repeat_share)*(1-eligible_nonexact_share) FROM scenario_inputs CROSS JOIN mix
)
SELECT w.scenario, r.policy, w.repeat_share, w.resolved_share,
       SUM(w.weight) AS weight_sum,
       1000 * SUM(w.weight*r.gross_calls_avoided/r.requests) AS gross_avoided_per_1000,
       1000 * SUM(w.weight*r.suitable_calls_avoided/r.requests) AS suitable_avoided_per_1000,
       1000 * SUM(w.weight*r.unsuitable_reuses/r.requests) AS unsuitable_per_1000,
       1000 * SUM(w.weight*r.fallback_requests/r.requests) AS fallback_per_1000
FROM weights w JOIN risk_segments r USING (stratum)
GROUP BY w.scenario, r.policy, w.repeat_share, w.resolved_share;
