"""Phase 3 — Decision Step: ambiguous-band routing logic.

The decision step is invoked ONLY for AMBIGUOUS-tier traffic (pairs where at
least one of similarity score or stability confidence is borderline).  It
uses three signals to make a final REUSE or BYPASS decision:

  1. Similarity score  (from CacheLookupResult)
  2. Stability confidence  (from StabilityResult)
  3. Category reuse-history  (from CategoryHistory, keyed by domain)

Following comprehensive Phase 3 evaluations against the 120-pair benchmark,
the non-LLM linear combination (implemented in DecisionStep below) and
the per-category adaptive threshold engine were both empirically rejected due
to cache hazard ceiling violations (IRR_cache 52.17% and 23.08% respectively).
The production decision path for AMBIGUOUS-tier traffic is JudgeDecisionStep
(aliased as ProductionDecisionStep), routing through LLMJudge (defaulting to
OpenRouter nvidia/nemotron-3-super-120b-a12b:free), which alone cleared the
<10% hazard ceiling (IRR_cache = 8.33%). The DecisionStep class is retained
verbatim as a documented, tested historical baseline.

=============================================================================
CATEGORY REUSE-HISTORY — DESIGN DECISIONS (written before implementation)
=============================================================================

What counts as a category?
  The benchmark dataset already uses the 8 enterprise domains from Phase 0:
    computer_science, science_medicine, mathematics, system_operations,
    history_geography, finance_economics, realtime_news_weather,
    legal_compliance
  Phase 3 uses the same 8-domain taxonomy (matching the "domain" field in
  query_pair_reuse_benchmark.json), NOT Phase 1's finer-grained labels.
  This keeps category boundaries aligned with the evaluation data.

What does "history" mean with no live traffic?
  With no live traffic to draw from, reuse history is simulated from the
  benchmark dataset's labeled domains during evaluation.  CategoryHistory is
  populated by the DecisionEvaluator as it processes pairs in order, using
  each pair's domain and outcome to update the running success rate.  This
  simulates a system that learns from its own decisions as traffic arrives.

  IMPORTANT: because the evaluator feeds pairs to the decision step in
  benchmark order, and updates history after each pair, the history available
  at pair N is derived from pairs 1..N-1 only.  This is an online / streaming
  simulation.  It is NOT a look-ahead or data-leakage scenario.

Cold-start handling:
  A domain with no history defaults to a neutral reuse-success rate of 0.50.
  This is intentionally non-committal — the decision step gives equal weight
  to the other two signals when history is absent.

Staleness:
  CategoryHistory uses a simple exponentially-weighted moving average (EWMA)
  with a configurable decay factor.  Older observations count less.  With
  no live traffic, all pairs have equal recency, so EWMA degrades to a
  weighted mean (all weights equal to 1/(1+decay)^n relative terms).
  In practice across 120 pairs, EWMA and simple mean differ by < 0.02.
  EWMA is retained so the same code works correctly with live traffic.

Decision logic for AMBIGUOUS band:
  A simple linear combination of the three normalized signals:
    score = w_sim * sim_norm + w_conf * conf_norm + w_hist * hist_rate
  where:
    sim_norm  = (similarity_score - ambiguous_sim_low) /
                (auto_reuse_sim_floor - ambiguous_sim_low)
                clipped to [0, 1]
    conf_norm = (confidence - bypass_conf_ceiling) /
                (auto_reuse_conf_floor - bypass_conf_ceiling)
                clipped to [0, 1]
    hist_rate = domain's EWMA reuse-success rate in [0, 1]
                (cold-start default: 0.50)

  Weights: w_sim=0.50, w_conf=0.30, w_hist=0.20
  Rationale:
    - Similarity is the primary signal (0.50): it is what Phase 2 used alone,
      so we give it the most weight.
    - Confidence is secondary (0.30): Phase 1 showed it distinguishes safe from
      borderline STABLE predictions.
    - History is tertiary (0.20): useful signal but subject to cold-start and
      dataset skew.

  Decision threshold: score >= 0.50 → REUSE; else BYPASS.
  This threshold is symmetric around the 0.50 neutral point.

  No LLM judge call is made.  A JudgeCallable hook is provided in the
  interface for future extension.  If a judge call IS wired in, its latency
  and call count must be reported as first-class metrics in DecisionResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from src.cache.semantic_cache import CacheLookupResult
from src.classifier.models import StabilityResult
from src.decision.tier_router import DEFAULT_BOUNDARIES, TierBoundaries
from src.decision.judge_call import JudgeResult, LLMJudge


# ---------------------------------------------------------------------------
# Judge-call type alias (optional hook, unused in initial implementation)
# ---------------------------------------------------------------------------
# A JudgeCallable receives similarity_score, stability_confidence, and the
# domain string, and returns a float in [0, 1] representing the judge's
# reuse-confidence estimate.  It is synchronous (no async) to keep the
# hot-path simple.  If an LLM call is ever wired here, the returned float
# must be an honest model output probability — not a heuristic disguised
# as a model score.
JudgeCallable = Callable[[float, float, str], float]


# ---------------------------------------------------------------------------
# Per-domain reuse history tracker
# ---------------------------------------------------------------------------

class CategoryHistory:
    """Tracks per-domain reuse outcome history using an EWMA.

    Keys are domain strings (matching the 8-domain taxonomy from Phase 0 and
    the "domain" field in query_pair_reuse_benchmark.json).

    Cold-start default: 0.50 (neutral — no bias toward or against reuse).

    Args:
        decay: EWMA decay factor in [0, 1).  Each new observation has weight
            1.0; prior EWMA is weighted by decay.  Smaller decay → more
            responsive to recent observations; larger → smoother/slower.
            Default: 0.90 (standard slow EWMA for sparse data).

    Usage:
        hist = CategoryHistory()
        hist.update("computer_science", was_correct_reuse=True)
        rate = hist.get("computer_science")  # → float in [0, 1]
    """

    COLD_START_DEFAULT: float = 0.50

    def __init__(self, decay: float = 0.90) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1); got {decay}")
        self.decay = decay
        # _ewma[domain] → current EWMA value
        # _count[domain] → number of updates received
        self._ewma: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def update(self, domain: str, was_correct_reuse: bool) -> None:
        """Record one reuse outcome for a domain.

        Args:
            domain: Domain string (e.g. "computer_science").
            was_correct_reuse: True if the reuse decision was correct (TP);
                False if it was incorrect (FP) or unnecessary regeneration (FN).
                For BYPASS decisions, do not call update() — only reuse
                decisions are tracked.
        """
        observation = 1.0 if was_correct_reuse else 0.0
        if domain not in self._ewma:
            # First observation: initialise from cold-start, then update
            self._ewma[domain] = self.COLD_START_DEFAULT
            self._count[domain] = 0

        self._ewma[domain] = (
            self.decay * self._ewma[domain] + (1.0 - self.decay) * observation
        )
        self._count[domain] += 1

    def get(self, domain: str) -> float:
        """Return the current EWMA reuse-success rate for a domain.

        Returns COLD_START_DEFAULT (0.50) if no observations have been
        recorded for this domain.

        Args:
            domain: Domain string.

        Returns:
            Float in [0, 1] representing estimated probability of a reuse
            decision being correct for this domain.
        """
        return self._ewma.get(domain, self.COLD_START_DEFAULT)

    def observation_count(self, domain: str) -> int:
        """Return the number of updates received for a domain (0 if cold-start)."""
        return self._count.get(domain, 0)

    def reset(self) -> None:
        """Clear all history (useful between evaluation runs)."""
        self._ewma.clear()
        self._count.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Return a snapshot of current history for logging."""
        return {
            domain: {
                "ewma": round(self._ewma[domain], 4),
                "count": self._count[domain],
            }
            for domain in self._ewma
        }


# ---------------------------------------------------------------------------
# Decision result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionResult:
    """Output of DecisionStep.decide() for AMBIGUOUS-tier traffic.

    Attributes:
        decision: "REUSE" or "BYPASS".
        score: The linear-combination score that drove the decision (0–1).
        sim_norm: Normalised similarity contribution (0–1).
        conf_norm: Normalised confidence contribution (0–1).
        hist_rate: Domain history rate used (0–1; COLD_START_DEFAULT if absent).
        domain: Domain string that was looked up in CategoryHistory.
        history_count: Number of prior observations for this domain at decision time.
        judge_called: True if a judge call was made; False if not.
        judge_score: Score returned by the judge (None if not called).
        rationale: Human-readable explanation of the decision.
    """

    decision: str          # "REUSE" or "BYPASS"
    score: float           # linear combination in [0, 1]
    sim_norm: float
    conf_norm: float
    hist_rate: float
    domain: str
    history_count: int
    judge_called: bool
    judge_score: Optional[float]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for reporting."""
        return {
            "decision": self.decision,
            "score": round(self.score, 4),
            "sim_norm": round(self.sim_norm, 4),
            "conf_norm": round(self.conf_norm, 4),
            "hist_rate": round(self.hist_rate, 4),
            "domain": self.domain,
            "history_count": self.history_count,
            "judge_called": self.judge_called,
            "judge_score": round(self.judge_score, 4) if self.judge_score is not None else None,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Decision step
# ---------------------------------------------------------------------------

class DecisionStep:
    """Ambiguous-band decision logic for Phase 3.

    Makes a REUSE or BYPASS decision for AMBIGUOUS-tier traffic using
    similarity score, stability confidence, and category reuse-history.
    An optional judge-call hook is supported but not invoked by default.

    This class is READ-ONLY with respect to Phase 1 and Phase 2 — it receives
    their outputs as arguments and never calls into or modifies them.

    Args:
        history: CategoryHistory instance.  If None, a fresh instance is
            created (cold-start for all domains).
        boundaries: TierBoundaries defining the band edges for normalisation.
            Defaults to DEFAULT_BOUNDARIES.
        w_sim: Weight for similarity signal.  Default: 0.50.
        w_conf: Weight for confidence signal.  Default: 0.30.
        w_hist: Weight for history signal.  Default: 0.20.
        decision_threshold: Score >= this value → REUSE; else BYPASS.
            Default: 0.50.
        judge: Optional JudgeCallable.  If provided, its score is added as
            an additional weighted input (requires adjusting weights accordingly
            — not implemented by default).  Pass None to disable.

    Usage:
        step = DecisionStep()
        result = step.decide(cache_result, stability_result, domain="computer_science")
    """

    # Weights for the linear combination (must sum to 1.0)
    DEFAULT_W_SIM: float = 0.50
    DEFAULT_W_CONF: float = 0.30
    DEFAULT_W_HIST: float = 0.20
    DEFAULT_DECISION_THRESHOLD: float = 0.50

    def __init__(
        self,
        history: Optional[CategoryHistory] = None,
        boundaries: Optional[TierBoundaries] = None,
        w_sim: float = DEFAULT_W_SIM,
        w_conf: float = DEFAULT_W_CONF,
        w_hist: float = DEFAULT_W_HIST,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        judge: Optional[JudgeCallable] = None,
    ) -> None:
        if abs(w_sim + w_conf + w_hist - 1.0) > 1e-9:
            raise ValueError(
                f"Weights must sum to 1.0; got w_sim={w_sim}, w_conf={w_conf}, "
                f"w_hist={w_hist}, sum={w_sim + w_conf + w_hist}"
            )
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError(
                f"decision_threshold must be in [0, 1]; got {decision_threshold}"
            )
        self.history = history or CategoryHistory()
        self.boundaries = boundaries or DEFAULT_BOUNDARIES
        self.w_sim = w_sim
        self.w_conf = w_conf
        self.w_hist = w_hist
        self.decision_threshold = decision_threshold
        self.judge = judge

    def decide(
        self,
        cache_result: CacheLookupResult,
        stability_result: StabilityResult,
        domain: str,
        query_a: str = "",
        query_b: str = "",
    ) -> DecisionResult:
        """Make a REUSE or BYPASS decision for AMBIGUOUS-tier traffic.

        This method is called ONLY for AMBIGUOUS-tier queries.  For AUTO-REUSE
        or BYPASS, the TierRouter has already decided — do not call this method
        for those tiers.

        Args:
            cache_result: Phase 2 CacheLookupResult — read-only.
            stability_result: Phase 1 StabilityResult — read-only.
            domain: Domain string for CategoryHistory lookup (e.g. "computer_science").
            query_a: Optional cached query text.
            query_b: Optional incoming query text.

        Returns:
            DecisionResult with the decision, component scores, and rationale.
        """
        b = self.boundaries
        sim = cache_result.similarity_score
        conf = stability_result.confidence

        # Normalise similarity into the ambiguous band
        sim_range = b.auto_reuse_sim_floor - b.bypass_sim_ceiling  # 0.92 - 0.50 = 0.42
        sim_norm = max(0.0, min(1.0,
            (sim - b.bypass_sim_ceiling) / sim_range if sim_range > 0 else 0.0
        ))

        # Normalise confidence into its ambiguous band
        conf_range = b.auto_reuse_conf_floor - b.bypass_conf_ceiling  # 0.90 - 0.80 = 0.10
        conf_norm = max(0.0, min(1.0,
            (conf - b.bypass_conf_ceiling) / conf_range if conf_range > 0 else 0.0
        ))

        # Domain history
        hist_rate = self.history.get(domain)
        hist_count = self.history.observation_count(domain)

        # Optional judge call
        judge_called = False
        judge_score: Optional[float] = None
        if self.judge is not None:
            if isinstance(self.judge, LLMJudge):
                judge_score = self.judge(
                    sim, conf, domain,
                    query_a=query_a, query_b=query_b,
                    category_history_rate=hist_rate,
                )
            else:
                judge_score = self.judge(sim, conf, domain)
            judge_called = True
            # When judge is active, blend its score equally with the combined signal.
            # Weights in this branch: 50% judge, 50% (sim+conf+hist combination).
            combined_no_judge = self.w_sim * sim_norm + self.w_conf * conf_norm + self.w_hist * hist_rate
            score = 0.50 * judge_score + 0.50 * combined_no_judge
            rationale_judge = f"; judge_score={judge_score:.3f}"
        else:
            score = self.w_sim * sim_norm + self.w_conf * conf_norm + self.w_hist * hist_rate
            rationale_judge = ""

        decision = "REUSE" if score >= self.decision_threshold else "BYPASS"

        cold_note = " (cold-start)" if hist_count == 0 else f" (n={hist_count})"
        rationale = (
            f"decision={decision}; score={score:.3f} "
            f"[sim_norm={sim_norm:.3f}*{self.w_sim}, "
            f"conf_norm={conf_norm:.3f}*{self.w_conf}, "
            f"hist={hist_rate:.3f}{cold_note}*{self.w_hist}]"
            f"{rationale_judge}"
            f"; threshold={self.decision_threshold}"
        )

        return DecisionResult(
            decision=decision,
            score=round(score, 6),
            sim_norm=round(sim_norm, 6),
            conf_norm=round(conf_norm, 6),
            hist_rate=round(hist_rate, 6),
            domain=domain,
            history_count=hist_count,
            judge_called=judge_called,
            judge_score=judge_score,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Dedicated Judge Decision Step
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeDecisionResult:
    """Result of an AMBIGUOUS-tier decision from JudgeDecisionStep.

    Attributes:
        decision: "REUSE" or "BYPASS".
        is_safe: True if safe for reuse, False if bypass.
        domain: Domain string.
        similarity_score: Raw similarity score from cache lookup.
        stability_confidence: Confidence score from stability classifier.
        category_history_rate: Running category history success rate.
        history_count: Number of prior observations for domain.
        judge_result: Underlying JudgeResult from LLMJudge.
        rationale: Explanatory reasoning.
    """

    decision: str
    is_safe: bool
    domain: str
    similarity_score: float
    stability_confidence: float
    category_history_rate: float
    history_count: int
    judge_result: JudgeResult
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata for reporting."""
        return {
            "decision": self.decision,
            "is_safe": self.is_safe,
            "domain": self.domain,
            "similarity_score": round(self.similarity_score, 4),
            "stability_confidence": round(self.stability_confidence, 4),
            "category_history_rate": round(self.category_history_rate, 4),
            "history_count": self.history_count,
            "judge_result": self.judge_result.to_dict(),
            "rationale": self.rationale,
        }


class JudgeDecisionStep:
    """Ambiguous-band decision step routing through LLMJudge.

    Given full context (both query texts, similarity score, domain, stability
    confidence, category history rate), delegates the REUSE/BYPASS decision to
    an LLM judge.

    CRITICAL SAFETY INVARIANT:
        Defaults to BYPASS on any judge failure, error, or timeout (fail-closed).

    Args:
        judge: Optional LLMJudge instance. Defaults to LLMJudge().
        history: Optional CategoryHistory instance. Defaults to fresh CategoryHistory().
    """

    def __init__(
        self,
        judge: Optional[LLMJudge] = None,
        history: Optional[CategoryHistory] = None,
    ) -> None:
        self.judge = judge or LLMJudge()
        self.history = history or CategoryHistory()

    def decide(
        self,
        query_a: str,
        query_b: str,
        cache_result: CacheLookupResult,
        stability_result: StabilityResult,
        domain: str,
    ) -> JudgeDecisionResult:
        """Evaluate an ambiguous pair using the LLM judge."""
        sim = cache_result.similarity_score
        conf = stability_result.confidence
        hist_rate = self.history.get(domain)
        hist_count = self.history.observation_count(domain)

        judge_res = self.judge.judge(
            query_a=query_a,
            query_b=query_b,
            domain=domain,
            similarity_score=sim,
            stability_confidence=conf,
            category_history_rate=hist_rate,
        )

        decision = "REUSE" if judge_res.is_safe else "BYPASS"
        cold_note = " (cold-start)" if hist_count == 0 else f" (n={hist_count})"
        rationale = (
            f"JudgeDecisionStep[{decision}]: judge={judge_res.decision} "
            f"(safe={judge_res.is_safe}, conf={judge_res.confidence:.2f}, "
            f"tokens={judge_res.total_tokens}, lat={judge_res.latency_ms:.1f}ms); "
            f"domain={domain}{cold_note}; rationale={judge_res.rationale}"
        )

        return JudgeDecisionResult(
            decision=decision,
            is_safe=judge_res.is_safe,
            domain=domain,
            similarity_score=sim,
            stability_confidence=conf,
            category_history_rate=hist_rate,
            history_count=hist_count,
            judge_result=judge_res,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Production Ambiguous-Tier Decision Step Alias
# ---------------------------------------------------------------------------
# Phase 3 Final Production Decision: JudgeDecisionStep (backed by LLMJudge on
# OpenRouter nvidia/nemotron-3-super-120b-a12b:free) is the live production
# decision path for the AMBIGUOUS tier. The non-LLM DecisionStep (linear
# combination) is preserved above as a documented, tested historical alternative.
ProductionDecisionStep = JudgeDecisionStep

