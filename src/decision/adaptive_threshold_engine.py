"""Phase 3 Fix — Adaptive Threshold Policy Engine.

=============================================================================
SPECIFICATION: PER-CATEGORY ADAPTIVE THRESHOLD ENGINE (REFINED)
=============================================================================

1. BACKGROUND & ASYMMETRIC SAFETY BIAS:
   Phase 3's initial decision step used a hand-tuned linear combination that
   proved too permissive, yielding 52.17% IRR_cache. While a Bayes-optimal point
   estimate (theta at p*=0.50 on a fitted logistic curve) establishes the point
   where safety is more likely than not, it implicitly treats false positive
   cache hazard and false negative missed-reuse as equally costly.

   Throughout this project, hazard is treated as far more costly than missed reuse:
     - Phase 1: STABLE_CONFIDENCE_THRESHOLD = 0.80 overrides borderline STABLE
       queries to DYNAMIC.
     - Phase 2: IRR_cache < 10% ceiling sets a strict safety ceiling.
   
   To inject the same asymmetric-conservative bias into the adaptive policy engine,
   the OPERATING threshold used in decide() is set to the UPPER BOUND of the 95%
   confidence interval (theta_upper), rather than the point estimate theta:
     operating_threshold = ci_upper = theta + z_0.975 * SE(theta)

   Rationale & Trade-off:
     - Using ci_upper requires higher embedding similarity to justify REUSE.
     - Cost: fewer REUSE decisions (lower hit rate / ARR, higher FRR).
     - Benefit: substantially reduces cache hazard (IRR_cache) in high-overlap
       domains by requiring similarity beyond statistical uncertainty.
     - Transparency: the point estimate theta, standard error, ci_lower, and
       ci_upper remain fully reported and audited.

2. DUAL MINIMUM-SAMPLE GATES:
   The Events-Per-Variable (EPV) rule (Peduzzi et al., 1996; Harrell, Regression
   Modeling Strategies) for binary logistic regression requires at least 10–20
   events of the MINORITY class per variable to avoid severe finite-sample bias,
   quasi-complete separation, and inflated parameter variance.

   Gate 1: MINIMUM_CATEGORY_N = 20
     - Ensures overall statistical mass for the domain.
   Gate 2: MINIMUM_MINORITY_CLASS_N = 10
     - Ensures the minority class (safe or unsafe) has >= 10 observations.
     - Prevents passing severely imbalanced categories (e.g. 24 safe / 1 unsafe).

   A domain only qualifies for per-category estimation if it passes BOTH gates.
   If either gate fails, the domain strictly falls back to FALLBACK_THRESHOLD (0.85).

3. VERIFIED DOMAIN STATUS UNDER DUAL GATE (query_pair_reuse_benchmark.json, N=120):
     - computer_science:       54 pairs (26 safe / 28 unsafe) -> min=26 >= 10, N=54 >= 20 -> PER-CATEGORY
     - science_medicine:       27 pairs (17 safe / 10 unsafe) -> min=10 >= 10, N=27 >= 20 -> PER-CATEGORY
     - mathematics:            13 pairs (11 safe /  2 unsafe) -> min=2 < 10,  N=13 < 20  -> FALLBACK (0.85)
     - system_operations:      11 pairs ( 2 safe /  9 unsafe) -> min=2 < 10,  N=11 < 20  -> FALLBACK (0.85)
     - history_geography:       9 pairs ( 5 safe /  4 unsafe) -> min=4 < 10,  N=9 < 20   -> FALLBACK (0.85)
     - finance_economics:       5 pairs ( 1 safe /  4 unsafe) -> min=1 < 10,  N=5 < 20   -> FALLBACK (0.85)
     - realtime_news_weather:   1 pairs ( 0 safe /  1 unsafe) -> min=0 < 10,  N=1 < 20   -> FALLBACK (0.85)
     - legal_compliance:        0 pairs ( 0 safe /  0 unsafe) -> min=0 < 10,  N=0 < 20   -> FALLBACK (0.85)

4. ISOLATION OF STABILITY CONFIDENCE:
   Phase 1's stability confidence remains a hard gate in tier_router.py
   (bypass_conf_ceiling = 0.80, auto_reuse_conf_floor = 0.90).
   The ambiguous decision engine operates only on similarity vs category threshold.

=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MINIMUM_CATEGORY_N: int = 20
MINIMUM_MINORITY_CLASS_N: int = 10
FALLBACK_THRESHOLD: float = 0.85
DEFAULT_CONFIDENCE_LEVEL: float = 0.95


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategoryThreshold:
    """Statistical threshold and confidence metadata for a specific domain.

    Attributes:
        domain: Domain taxonomy name (e.g. "computer_science").
        sample_size: Total labeled pairs available for this domain.
        minority_count: Observations in the minority class (min(safe, unsafe)).
        threshold: Fitted point estimate theta (p*=0.50). For fallback, equals fallback_threshold.
        operating_threshold: Decision boundary used in decide(). For per-category fits,
            equals ci_upper (upper bound of 95% CI) to enforce asymmetric conservative bias;
            for fallback, equals fallback_threshold (0.85).
        is_fallback: True if threshold fell back to global default (0.85);
            False if estimated from domain data.
        fallback_reason: Explanation if is_fallback is True, else None.
        ci_lower: Lower bound of 95% confidence interval (None for fallback).
        ci_upper: Upper bound of 95% confidence interval (None for fallback).
        standard_error: Standard error of threshold estimate (None for fallback).
        b0: Logistic intercept (None for fallback).
        b1: Logistic slope (None for fallback).
    """

    domain: str
    sample_size: int
    threshold: float
    operating_threshold: float
    is_fallback: bool
    minority_count: int = 0
    fallback_reason: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    standard_error: Optional[float] = None
    b0: Optional[float] = None
    b1: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata for reporting."""
        return {
            "domain": self.domain,
            "sample_size": self.sample_size,
            "minority_count": self.minority_count,
            "threshold": round(self.threshold, 4),
            "operating_threshold": round(self.operating_threshold, 4),
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "ci_lower": round(self.ci_lower, 4) if self.ci_lower is not None else None,
            "ci_upper": round(self.ci_upper, 4) if self.ci_upper is not None else None,
            "standard_error": round(self.standard_error, 4) if self.standard_error is not None else None,
            "b0": round(self.b0, 4) if self.b0 is not None else None,
            "b1": round(self.b1, 4) if self.b1 is not None else None,
        }


@dataclass(frozen=True)
class ThresholdDecisionResult:
    """Result of an AMBIGUOUS-tier decision from AdaptiveThresholdEngine.

    Attributes:
        decision: "REUSE" or "BYPASS".
        similarity_score: The query pair's similarity score.
        threshold: The operating threshold applied for this domain.
        operating_threshold: Same as threshold (the operating decision boundary).
        point_estimate: Point estimate theta (p*=0.50), if per-category fit.
        domain: Domain evaluated.
        is_fallback: Whether the threshold was a fallback.
        ci_lower: Confidence interval lower bound.
        ci_upper: Confidence interval upper bound.
        rationale: Human-readable explanation.
    """

    decision: str
    similarity_score: float
    threshold: float
    operating_threshold: float
    domain: str
    is_fallback: bool
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    rationale: str
    point_estimate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision for evaluation reporting."""
        return {
            "decision": self.decision,
            "similarity_score": round(self.similarity_score, 4),
            "threshold": round(self.threshold, 4),
            "operating_threshold": round(self.operating_threshold, 4),
            "point_estimate": round(self.point_estimate, 4) if self.point_estimate is not None else None,
            "domain": self.domain,
            "is_fallback": self.is_fallback,
            "ci_lower": round(self.ci_lower, 4) if self.ci_lower is not None else None,
            "ci_upper": round(self.ci_upper, 4) if self.ci_upper is not None else None,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Calibration Helper: Logistic Fit with Delta Method CI
# ---------------------------------------------------------------------------

def fit_logistic_threshold(
    similarities: Sequence[float],
    safe_labels: Sequence[bool | int],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Tuple[float, float, float, float, float, float]:
    """Fit logistic regression P(safe | sim) and compute Delta-method CI on threshold.

    Returns:
        (theta, se_theta, ci_lower, ci_upper, b0, b1)
    Raises:
        ValueError if data cannot support a valid logistic fit.
    """
    sims = np.asarray(similarities, dtype=np.float64)
    ys = np.asarray(safe_labels, dtype=np.float64)

    if len(sims) < 2 or len(np.unique(ys)) < 2:
        raise ValueError("Cannot fit logistic model: needs >= 2 points and both classes represented.")

    def neg_log_likelihood(params: np.ndarray) -> float:
        b0, b1 = params
        z = np.clip(b0 + b1 * sims, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        return float(-np.sum(ys * np.log(p + eps) + (1.0 - ys) * np.log(1.0 - p + eps)))

    res = optimize.minimize(neg_log_likelihood, [0.0, 1.0], method="BFGS")
    if not res.success and res.status != 2:
        raise ValueError(f"Logistic optimization did not converge: {res.message}")

    b0, b1 = float(res.x[0]), float(res.x[1])
    if b1 <= 1e-6:
        raise ValueError(f"Slope non-positive (b1={b1:.4f}): similarity is not positively associated with safety.")

    cov = res.hess_inv
    theta = -b0 / b1

    # Gradient of theta w.r.t [b0, b1]:
    # d(theta)/d(b0) = -1 / b1
    # d(theta)/d(b1) = b0 / (b1^2)
    grad = np.array([-1.0 / b1, b0 / (b1**2)], dtype=np.float64)
    var_theta = float(grad @ cov @ grad)
    se_theta = math.sqrt(max(0.0, var_theta))

    # Two-sided confidence interval via scipy.stats
    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lower = theta - z_crit * se_theta
    ci_upper = theta + z_crit * se_theta

    return theta, se_theta, ci_lower, ci_upper, b0, b1


# ---------------------------------------------------------------------------
# Pre-calibrated Baseline Thresholds (Fit from the 120 Benchmark Pairs)
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATED_THRESHOLDS: Dict[str, CategoryThreshold] = {
    "computer_science": CategoryThreshold(
        domain="computer_science",
        sample_size=54,
        minority_count=26,
        threshold=0.6855,
        operating_threshold=0.7924,   # ci_upper: conservative safety margin
        is_fallback=False,
        fallback_reason=None,
        ci_lower=0.5785,
        ci_upper=0.7924,
        standard_error=0.0546,
        b0=-3.6696,
        b1=5.3535,
    ),
    "science_medicine": CategoryThreshold(
        domain="science_medicine",
        sample_size=27,
        minority_count=10,
        threshold=0.6171,
        operating_threshold=0.7245,   # ci_upper: conservative safety margin
        is_fallback=False,
        fallback_reason=None,
        ci_lower=0.5097,
        ci_upper=0.7245,
        standard_error=0.0548,
        b0=-9.9575,
        b1=16.1359,
    ),
    "mathematics": CategoryThreshold(
        domain="mathematics",
        sample_size=13,
        minority_count=2,
        threshold=FALLBACK_THRESHOLD,
        operating_threshold=FALLBACK_THRESHOLD,
        is_fallback=True,
        fallback_reason="Sample size 13 < MINIMUM_CATEGORY_N (20) or minority count 2 < MINIMUM_MINORITY_CLASS_N (10)",
    ),
    "system_operations": CategoryThreshold(
        domain="system_operations",
        sample_size=11,
        minority_count=2,
        threshold=FALLBACK_THRESHOLD,
        operating_threshold=FALLBACK_THRESHOLD,
        is_fallback=True,
        fallback_reason="Sample size 11 < MINIMUM_CATEGORY_N (20) or minority count 2 < MINIMUM_MINORITY_CLASS_N (10)",
    ),
    "history_geography": CategoryThreshold(
        domain="history_geography",
        sample_size=9,
        minority_count=4,
        threshold=FALLBACK_THRESHOLD,
        operating_threshold=FALLBACK_THRESHOLD,
        is_fallback=True,
        fallback_reason="Sample size 9 < MINIMUM_CATEGORY_N (20) or minority count 4 < MINIMUM_MINORITY_CLASS_N (10)",
    ),
    "finance_economics": CategoryThreshold(
        domain="finance_economics",
        sample_size=5,
        minority_count=1,
        threshold=FALLBACK_THRESHOLD,
        operating_threshold=FALLBACK_THRESHOLD,
        is_fallback=True,
        fallback_reason="Sample size 5 < MINIMUM_CATEGORY_N (20) or minority count 1 < MINIMUM_MINORITY_CLASS_N (10)",
    ),
    "realtime_news_weather": CategoryThreshold(
        domain="realtime_news_weather",
        sample_size=1,
        minority_count=0,
        threshold=FALLBACK_THRESHOLD,
        operating_threshold=FALLBACK_THRESHOLD,
        is_fallback=True,
        fallback_reason="Sample size 1 < MINIMUM_CATEGORY_N (20) or minority count 0 < MINIMUM_MINORITY_CLASS_N (10)",
    ),
    "legal_compliance": CategoryThreshold(
        domain="legal_compliance",
        sample_size=0,
        minority_count=0,
        threshold=FALLBACK_THRESHOLD,
        operating_threshold=FALLBACK_THRESHOLD,
        is_fallback=True,
        fallback_reason="Sample size 0 < MINIMUM_CATEGORY_N (20) or minority count 0 < MINIMUM_MINORITY_CLASS_N (10)",
    ),
}


# ---------------------------------------------------------------------------
# Adaptive Threshold Engine
# ---------------------------------------------------------------------------

class AdaptiveThresholdEngine:
    """Adaptive policy engine for AMBIGUOUS-tier traffic.

    Decides REUSE vs BYPASS using per-category logistic thresholds bounded
    by 95% confidence intervals (Delta method), with dual MINIMUM_CATEGORY_N
    and MINIMUM_MINORITY_CLASS_N fallback gates.

    Operating Threshold Policy:
      For per-category fits, decide() operates at ci_upper (upper bound of the
      95% CI) rather than the raw point estimate theta. This enforces an
      asymmetric conservative bias, preventing permissive reuse in borderline
      similarity ranges.

    Args:
        thresholds: Optional mapping of domain -> CategoryThreshold. If None,
            uses pre-calibrated thresholds fit on the 120 benchmark pairs.
        minimum_category_n: Total sample size cutoff. Defaults to 20.
        minimum_minority_class_n: Minority class cutoff. Defaults to 10.
        fallback_threshold: Threshold for fallback domains. Defaults to 0.85.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, CategoryThreshold]] = None,
        minimum_category_n: int = MINIMUM_CATEGORY_N,
        minimum_minority_class_n: int = MINIMUM_MINORITY_CLASS_N,
        fallback_threshold: float = FALLBACK_THRESHOLD,
    ) -> None:
        self.minimum_category_n = minimum_category_n
        self.minimum_minority_class_n = minimum_minority_class_n
        self.fallback_threshold = fallback_threshold
        self._thresholds: Dict[str, CategoryThreshold] = dict(
            thresholds if thresholds is not None else DEFAULT_CALIBRATED_THRESHOLDS
        )

    def get_threshold(self, domain: str) -> CategoryThreshold:
        """Retrieve threshold and uncertainty metadata for a domain.

        If domain fails either sample-size gate or is unknown, returns
        a CategoryThreshold configured with is_fallback=True and threshold=fallback_threshold.
        """
        if domain in self._thresholds:
            ct = self._thresholds[domain]
            if not ct.is_fallback:
                if ct.sample_size < self.minimum_category_n:
                    return CategoryThreshold(
                        domain=domain,
                        sample_size=ct.sample_size,
                        minority_count=ct.minority_count,
                        threshold=self.fallback_threshold,
                        operating_threshold=self.fallback_threshold,
                        is_fallback=True,
                        fallback_reason=f"Sample size {ct.sample_size} < minimum_n ({self.minimum_category_n})",
                    )
                if ct.minority_count < self.minimum_minority_class_n:
                    return CategoryThreshold(
                        domain=domain,
                        sample_size=ct.sample_size,
                        minority_count=ct.minority_count,
                        threshold=self.fallback_threshold,
                        operating_threshold=self.fallback_threshold,
                        is_fallback=True,
                        fallback_reason=f"Minority count {ct.minority_count} < minimum_minority_n ({self.minimum_minority_class_n})",
                    )
            return ct

        # Unseen / unknown domain
        return CategoryThreshold(
            domain=domain,
            sample_size=0,
            minority_count=0,
            threshold=self.fallback_threshold,
            operating_threshold=self.fallback_threshold,
            is_fallback=True,
            fallback_reason=f"Domain {domain!r} unknown; falling back to global reference",
        )

    def decide(self, similarity_score: float, domain: str) -> ThresholdDecisionResult:
        """Make a REUSE or BYPASS decision for an AMBIGUOUS-tier pair.

        Uses ct.operating_threshold (ci_upper for per-category fits; 0.85 for fallback).

        Args:
            similarity_score: Embedding cosine similarity in [0, 1].
            domain: Query domain string.

        Returns:
            ThresholdDecisionResult with decision, applied operating threshold, and CI.
        """
        ct = self.get_threshold(domain)
        decision = "REUSE" if similarity_score >= ct.operating_threshold else "BYPASS"

        if ct.is_fallback:
            rationale = (
                f"decision={decision}; sim={similarity_score:.4f} vs fallback_threshold={ct.operating_threshold:.4f} "
                f"({ct.fallback_reason})"
            )
        else:
            ci_str = f"[{ct.ci_lower:.4f}, {ct.ci_upper:.4f}]" if ct.ci_lower is not None else "N/A"
            rationale = (
                f"decision={decision}; sim={similarity_score:.4f} vs operating_threshold={ct.operating_threshold:.4f} "
                f"(ci_upper bound of 95% CI; point_estimate={ct.threshold:.4f}, N={ct.sample_size}, "
                f"minority_N={ct.minority_count}, 95% CI={ci_str})"
            )

        return ThresholdDecisionResult(
            decision=decision,
            similarity_score=similarity_score,
            threshold=ct.operating_threshold,
            operating_threshold=ct.operating_threshold,
            point_estimate=ct.threshold,
            domain=domain,
            is_fallback=ct.is_fallback,
            ci_lower=ct.ci_lower,
            ci_upper=ct.ci_upper,
            rationale=rationale,
        )

    @classmethod
    def fit(
        cls,
        labeled_data: Sequence[Tuple[str, float, bool]],
        minimum_category_n: int = MINIMUM_CATEGORY_N,
        minimum_minority_class_n: int = MINIMUM_MINORITY_CLASS_N,
        fallback_threshold: float = FALLBACK_THRESHOLD,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ) -> "AdaptiveThresholdEngine":
        """Fit per-category thresholds with dual-gate fallback and CI upper operating bounds.

        Args:
            labeled_data: Sequence of (domain, similarity_score, is_reuse_safe).
            minimum_category_n: Cutoff for total domain N.
            minimum_minority_class_n: Cutoff for minority class count.
            fallback_threshold: Global reference threshold for fallbacks.
            confidence_level: Confidence level for scipy.stats interval.

        Returns:
            Fitted AdaptiveThresholdEngine instance.
        """
        by_domain: Dict[str, List[Tuple[float, bool]]] = {}
        for dom, sim, safe in labeled_data:
            if dom not in by_domain:
                by_domain[dom] = []
            by_domain[dom].append((sim, safe))

        thresholds: Dict[str, CategoryThreshold] = {}
        for dom, pairs in by_domain.items():
            n = len(pairs)
            n_safe = sum(1 for p in pairs if p[1])
            n_unsafe = n - n_safe
            minority_count = min(n_safe, n_unsafe)

            if n < minimum_category_n:
                thresholds[dom] = CategoryThreshold(
                    domain=dom,
                    sample_size=n,
                    minority_count=minority_count,
                    threshold=fallback_threshold,
                    operating_threshold=fallback_threshold,
                    is_fallback=True,
                    fallback_reason=f"Sample size {n} < minimum_n ({minimum_category_n})",
                )
                continue

            if minority_count < minimum_minority_class_n:
                thresholds[dom] = CategoryThreshold(
                    domain=dom,
                    sample_size=n,
                    minority_count=minority_count,
                    threshold=fallback_threshold,
                    operating_threshold=fallback_threshold,
                    is_fallback=True,
                    fallback_reason=f"Minority class count {minority_count} < minimum_minority_n ({minimum_minority_class_n})",
                )
                continue

            sims = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]

            try:
                theta, se, ci_low, ci_high, b0, b1 = fit_logistic_threshold(
                    sims, ys, confidence_level=confidence_level
                )
                thresholds[dom] = CategoryThreshold(
                    domain=dom,
                    sample_size=n,
                    minority_count=minority_count,
                    threshold=theta,
                    operating_threshold=ci_high,  # Use upper bound of CI as operating boundary
                    is_fallback=False,
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                    standard_error=se,
                    b0=b0,
                    b1=b1,
                )
            except Exception as exc:
                thresholds[dom] = CategoryThreshold(
                    domain=dom,
                    sample_size=n,
                    minority_count=minority_count,
                    threshold=fallback_threshold,
                    operating_threshold=fallback_threshold,
                    is_fallback=True,
                    fallback_reason=f"Fit failed ({exc}); fallback applied",
                )

        return cls(
            thresholds=thresholds,
            minimum_category_n=minimum_category_n,
            minimum_minority_class_n=minimum_minority_class_n,
            fallback_threshold=fallback_threshold,
        )
