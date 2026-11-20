"""Answer checks specific to the live chat's procedural signatures."""
import re

from src.policy.decomposed_applicability import answer_satisfies_requirements
from src.policy.procedural_evidence import profiling_reuse_reason

_CONTAINERS = {("list",), ("tuple",)}


def _shows_container_constructor(answer, source, target):
    # Accept direct instructions such as "Pass the tuple to list()" and simple
    # constructor examples. A bare mention or the opposite constructor is weak.
    destination = re.escape(target[0])
    origin = re.escape(source[0])
    if re.search(r"\b(?:do not|don't|never|avoid|instead of)\b[^.\n]{0,90}\b"
                 rf"{destination}\s*\(", answer, re.I):
        return False
    if re.search(
        rf"\b(?:use|call|pass|apply|wrap|convert|construct)\b[^.\n]{{0,120}}\b{destination}\s*\(",
        answer, re.I,
    ) and re.search(rf"\b{origin}s?\b", answer, re.I):
        return True
    return bool(re.search(rf"\b{destination}\s*\(\s*(?:[a-z_]\w*_{origin}|(?:my_|input_)?{origin}s?)\s*\)",
                          answer, re.I))


def semantic_answer_reason(original, incoming, answer, signature):
    reason = (profiling_reuse_reason(original, incoming, answer)
              or answer_satisfies_requirements(incoming, answer))
    if reason or not signature:
        return reason
    kind, words = signature
    if kind == "method" and len(words) == 3 and words[0] == "convert":
        source, target = words[1:]
        if source in _CONTAINERS and target in _CONTAINERS and source != target:
            if not _shows_container_constructor(answer, source, target):
                return "answer_lacks_conversion_evidence"
    return None
