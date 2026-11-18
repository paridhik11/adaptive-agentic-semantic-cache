-- One execution per (event_id, arm), validated against the manifest before import.
-- The paired view joins executions of the same request within a single workload.
CREATE TABLE executions (
    event_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    request_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    calls INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    elapsed_ms REAL NOT NULL,
    tier TEXT NOT NULL,
    PRIMARY KEY (event_id, arm)
);

CREATE VIEW paired_events AS
SELECT b.event_id, b.request_type, b.domain,
       b.calls AS no_cache_calls,
       e.calls AS exact_calls,
       g.calls AS guarded_calls,
       s.calls AS shadow_calls,
       b.calls - e.calls AS exact_calls_avoided,
       e.calls - g.calls AS guarded_increment,
       b.tokens AS no_cache_tokens,
       e.tokens AS exact_tokens,
       g.tokens AS guarded_tokens,
       s.tokens AS shadow_tokens,
       ROUND(b.elapsed_ms, 4) AS no_cache_ms,
       ROUND(e.elapsed_ms, 4) AS exact_ms,
       ROUND(g.elapsed_ms, 4) AS guarded_ms,
       ROUND(s.elapsed_ms, 4) AS shadow_ms,
       e.tier AS exact_tier,
       g.tier AS guarded_tier,
       s.tier AS shadow_tier,
       CASE WHEN b.request_type = 'adversarial_challenge' AND g.tier != 'NONE'
            THEN 1 ELSE 0 END AS guarded_false_accept
FROM executions AS b
JOIN executions AS e ON e.event_id = b.event_id AND e.arm = 'exact_only'
JOIN executions AS g ON g.event_id = b.event_id AND g.arm = 'verified_semantic'
JOIN executions AS s ON s.event_id = b.event_id AND s.arm = 'shadow_semantic'
WHERE b.arm = 'no_cache';

CREATE VIEW segment_summary AS
WITH membership AS (
    SELECT 'workload' AS dimension, 'all' AS segment, p.* FROM paired_events AS p
    UNION ALL
    SELECT 'request_type', request_type, p.* FROM paired_events AS p
    UNION ALL
    SELECT 'domain', domain, p.* FROM paired_events AS p
)
SELECT dimension, segment, COUNT(*) AS requests,
       SUM(no_cache_calls) AS no_cache_calls,
       SUM(exact_calls) AS exact_calls,
       SUM(guarded_calls) AS guarded_calls,
       SUM(shadow_calls) AS shadow_calls,
       SUM(exact_calls_avoided) AS exact_calls_avoided,
       SUM(guarded_increment) AS guarded_increment,
       SUM(CASE WHEN guarded_increment > 0 THEN guarded_increment ELSE 0 END) AS guarded_gross_gain,
       SUM(CASE WHEN guarded_increment < 0 THEN -guarded_increment ELSE 0 END) AS guarded_extra_calls,
       SUM(guarded_increment > 0) AS guarded_fewer_requests,
       SUM(guarded_increment = 0) AS guarded_tied_requests,
       SUM(guarded_increment < 0) AS guarded_more_requests,
       SUM(request_type = 'adversarial_challenge') AS challenges,
       SUM(guarded_false_accept) AS guarded_false_accepts
FROM membership
GROUP BY dimension, segment;
