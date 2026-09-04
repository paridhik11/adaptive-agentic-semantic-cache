"""Fallback classification layer for queries uncertain under rule evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.classifier.models import StabilityLabel


@dataclass
class FallbackPrediction:
    """Prediction output from fallback classifier.

    Note on confidence: The `confidence` score and `heuristic_dynamic_score` are
    heuristic indicators calculated via hand-tuned feature weights, NOT statistically
    calibrated posterior probabilities.
    """

    label: StabilityLabel
    confidence: float
    heuristic_dynamic_score: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFallbackClassifier(ABC):
    """Abstract base class for stability fallback classifiers."""

    @abstractmethod
    def predict(self, query: str) -> FallbackPrediction:
        """Predict stability label for an incoming query."""
        raise NotImplementedError


class LightweightHeuristicClassifier(BaseFallbackClassifier):
    """Deterministic, heuristic feature-weighted scoring classifier.

    Evaluates linguistic markers, domain cues, and query framing to assign
    a heuristic dynamic score without requiring external LLM APIs.
    """

    def __init__(self, conservative_threshold: float = 0.45) -> None:
        """Initialize fallback classifier.

        Args:
            conservative_threshold: Score threshold above which a query is
                classified as DYNAMIC. Lower values are more conservative.
        """
        self.conservative_threshold = conservative_threshold
        self._init_feature_weights()

    def _init_feature_weights(self) -> None:
        # Positive weights increase dynamic score; negative weights increase stable score
        self.feature_weights: List[Tuple[re.Pattern[str], float, str]] = [
            # Dynamic Signals (+weights)
            (re.compile(r"\b(stock\s+price|price\s+of|asset\s+price|market\s+cap|valuation\s+of|p/e\s+ratio|yield\s+on|subscription\s+cost|current\s+price)\b", re.I), 2.4, "financial_valuation_cue"),
            (re.compile(r"\b(weather\s+forecast|air\s+quality|uv\s+index|temperature\s+today|humidity\s+now|weather\s+in)\b", re.I), 2.4, "weather_meteorology_cue"),
            (re.compile(r"\b(flight\s+status|traffic\s+status|wait\s+time|checkpoint\s+at|road\s+closures|transit\s+queue)\b", re.I), 2.2, "transit_operational_cue"),
            (re.compile(r"\b(live\s+score|currently\s+underway|match\s+today|super\s+bowl\s+upcoming|quarterback|roster)\b", re.I), 2.0, "sports_competition_cue"),
            (re.compile(r"\b(breaking\s+news|headline\s+story|trending\s+topics|trending\s+hashtags|right\s+now|today\b|this\s+week|\brn\b|\batm\b)\b", re.I), 2.0, "temporal_currency_cue"),
            (re.compile(r"\b(interest\s+rate|inflation\s+rate|exchange\s+rate|discount\s+rate|mortgage\s+rate|tax\s+rate)\b", re.I), 2.0, "macroeconomic_indicator_cue"),
            (re.compile(r"\b(latest\s+version|current\s+version|npm\s+version|pypi\s+version|mainline\s+branch|newest\s+model)\b", re.I), 1.8, "software_hardware_release_cue"),
            (re.compile(r"\b(tornado\s+warning|wildfire\s+danger|earthquake\s+activity|disaster\s+declaration|travel\s+advisory)\b", re.I), 2.2, "natural_hazard_security_cue"),

            # Stable Signals (-weights)
            (re.compile(r"\b(how\s+does|how\s+do|how\s+to|explain|mechanism|function|purpose)\b", re.I), -1.2, "conceptual_mechanism_cue"),
            (re.compile(r"\b(what\s+is\s+the\s+difference|difference\s+between|compare|vs\.)\b", re.I), -1.3, "comparative_distinction_cue"),
            (re.compile(r"\b(definition|defined|formula|theorem|law|proof|algorithm|principle|axiom|elements\s+of)\b", re.I), -1.6, "scientific_definition_cue"),
            (re.compile(r"\b(time\s+complexity|space\s+complexity|big-o|asymptotic|constant|linear)\b", re.I), -1.8, "complexity_invariant_cue"),
            (re.compile(r"\b(history|ancient|founded|invented|discovered|century|empire|civilization|treaty|war|battle|voyage|census)\b", re.I), -1.6, "historical_record_cue"),
            (re.compile(r"\b(statute|doctrine|amendment|regulation|article|section|compliance|precedent|tort|negligence|habeas|res\s+judicata)\b", re.I), -1.4, "legal_codification_cue"),
            (re.compile(r"\b(biology|anatomy|cell|organism|protein|enzyme|chemical|element|periodic|atomic|heart|cardiac|half-life|photosynthesis)\b", re.I), -1.6, "biological_chemical_cue"),
            (re.compile(r"\b(capital\s+city|deepest|longest|largest|mountain\s+range|ocean\s+trench)\b", re.I), -1.6, "geographical_invariant_cue"),
            (re.compile(r"\b(linux|bash|python|sql|java|c\+\+|javascript|rust|git|http|rfc|fastapi|apirouter|v8|tree|queue|stack|quicksort)\b", re.I), -0.8, "programming_standard_cue"),
        ]

        # Prior log-score (baseline prior for general informational queries is stable)
        self.prior_log_odds = -1.2

    def predict(self, query: str) -> FallbackPrediction:
        """Score query and produce heuristic stability prediction."""
        log_odds = self.prior_log_odds
        fired_signals: List[str] = []
        has_strong_dynamic_cue = False
        has_historical_guard = False

        for pattern, weight, signal_name in self.feature_weights:
            if pattern.search(query):
                log_odds += weight
                fired_signals.append(f"{signal_name}({weight:+.1f})")
                if weight >= 2.0:
                    has_strong_dynamic_cue = True
                if "historical" in signal_name:
                    has_historical_guard = True

        # Asymmetric Safety Clamp: Strong dynamic cues cannot be completely suppressed by conceptual framing
        # unless an explicit historical record context is present
        if has_strong_dynamic_cue and not has_historical_guard:
            log_odds = max(log_odds, 0.2)

        # Convert log-odds to heuristic score via logistic function
        dynamic_score = 1.0 / (1.0 + math.exp(-log_odds))

        # Classification decision based on conservative threshold
        if dynamic_score >= self.conservative_threshold:
            predicted_label = StabilityLabel.DYNAMIC
            confidence = min(0.99, max(0.51, dynamic_score))
            decision_desc = f"Likely dynamic (heuristic dynamic score: {dynamic_score:.2f} >= {self.conservative_threshold:.2f})"
        else:
            predicted_label = StabilityLabel.STABLE
            confidence = min(0.99, max(0.51, 1.0 - dynamic_score))
            decision_desc = f"Likely stable (heuristic dynamic score: {dynamic_score:.2f} < {self.conservative_threshold:.2f})"

        signals_str = ", ".join(fired_signals) if fired_signals else "baseline prior"
        rationale = f"Fallback heuristic prediction: {decision_desc}. Active cues: [{signals_str}]."

        return FallbackPrediction(
            label=predicted_label,
            confidence=confidence,
            heuristic_dynamic_score=dynamic_score,
            rationale=rationale,
            metadata={
                "heuristic_dynamic_score": round(dynamic_score, 4),
                "raw_log_odds": round(log_odds, 4),
                "active_signals": fired_signals,
                "threshold_applied": self.conservative_threshold,
                "is_statistically_calibrated": False,
            },
        )


class TrainedLexicalClassifier(BaseFallbackClassifier):
    """Machine-learned lexical fallback classifier trained on development corpora.

    Uses combined word and character n-gram TF-IDF representations paired with
    a balanced logistic regression classifier to produce smooth posterior probabilities
    for queries that bypass deterministic Stage 1 rule checks.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """Initialize trained lexical fallback classifier.

        Args:
            model_path: Optional path to serialized model artifact (.joblib).
                If None or file does not exist, uses standard artifact path or trains on dev corpus.
        """
        if model_path is None:
            self.model_path = (
                Path(__file__).resolve().parent.parent.parent
                / "models"
                / "stability_fallback_lexical.joblib"
            )
        else:
            self.model_path = Path(model_path)

        if self.model_path.exists():
            import joblib
            self.pipeline = joblib.load(self.model_path)
        else:
            self.pipeline = self.train_and_save(self.model_path)

    @classmethod
    def build_pipeline(cls) -> Any:
        """Construct the scikit-learn feature union and classifier pipeline."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        union = FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    analyzer="word",
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    ngram_range=(2, 5),
                    analyzer="char_wb",
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ])

        clf = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

        return Pipeline([
            ("features", union),
            ("classifier", clf),
        ])

    def train_and_save(self, save_path: Path) -> Any:
        """Train pipeline on the 245-query development corpus and persist artifact."""
        import json
        import joblib
        import numpy as np

        root = Path(__file__).resolve().parent.parent.parent
        bench_path = root / "data" / "raw" / "query_stability_benchmark.json"
        human_path = root / "data" / "raw" / "query_stability_human_credibility.json"

        texts: List[str] = []
        labels: List[int] = []

        for p in (bench_path, human_path):
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("queries", data) if isinstance(data, dict) else data
            for it in items:
                q = it.get("query", "").strip()
                lbl = it.get("stability_label") or it.get("expected_label") or "STABLE"
                if q:
                    texts.append(q)
                    labels.append(1 if "DYN" in str(lbl).upper() else 0)

        pipeline = self.build_pipeline()
        if texts:
            pipeline.fit(texts, np.array(labels))

        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        return pipeline

    def predict(self, query: str) -> FallbackPrediction:
        """Predict stability label with trained lexical model."""
        clean_q = query.strip()
        if not clean_q:
            return FallbackPrediction(
                label=StabilityLabel.DYNAMIC,
                confidence=1.0,
                heuristic_dynamic_score=1.0,
                rationale="Empty query defaulted to uncacheable dynamic.",
                metadata={"is_empty": True},
            )

        proba = self.pipeline.predict_proba([clean_q])[0]
        # Class 1 is DYNAMIC, Class 0 is STABLE
        p_dynamic = float(proba[1])

        if p_dynamic >= 0.50:
            predicted_label = StabilityLabel.DYNAMIC
            confidence = p_dynamic
            decision_desc = f"Likely dynamic (P(DYNAMIC)={p_dynamic:.4f} >= 0.50)"
        else:
            predicted_label = StabilityLabel.STABLE
            confidence = 1.0 - p_dynamic
            decision_desc = f"Likely stable (P(STABLE)={1.0 - p_dynamic:.4f} > 0.50)"

        rationale = f"Trained lexical fallback prediction: {decision_desc}."

        return FallbackPrediction(
            label=predicted_label,
            confidence=confidence,
            heuristic_dynamic_score=p_dynamic,
            rationale=rationale,
            metadata={
                "p_dynamic": round(p_dynamic, 4),
                "p_stable": round(1.0 - p_dynamic, 4),
                "model_type": "TrainedLexicalClassifier",
                "is_statistically_calibrated": True,
            },
        )


# Backward compatibility alias
LightweightProbabilisticClassifier = LightweightHeuristicClassifier
