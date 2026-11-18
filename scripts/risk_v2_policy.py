"""Experimental explicit reporting requirement check for the frozen evaluation."""
import re

from src.policy.hybrid_cache import ConservativeHybridCache

_ARTIFACTS = {
    "evidence source": "Source",
    "responsible team": "Owner",
    "calculation formula": "Formula",
    "exception breakdown": "Breakdown",
}

_REQUEST = re.compile(
    r"\b(?:include|provide|list|state|show)\s+(?:the\s+)?("
    + "|".join(_ARTIFACTS)
    + r")\b", re.IGNORECASE
)


class ReportingControlCache(ConservativeHybridCache):
    """Candidate policy used only in the offline comparison."""

    def _answer_requirement_reason(self, query: str, answer: str) -> str | None:
        baseline = super()._answer_requirement_reason(query, answer)
        if baseline:
            return baseline
        for artifact in _REQUEST.findall(query):
            label = _ARTIFACTS[artifact.lower()]
            if not re.search(rf"(?im)^\s*{label}:\s*\S", answer):
                return "missing_reporting_artifact:" + label.lower()
        return None
