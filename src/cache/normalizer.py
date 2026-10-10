"""Phase 8 — Query Normalization Layer.

Provides fast string canonicalization for exact-match equivalence folding:
  - Lowercase conversion.
  - Punctuation stripping (trailing ?, !, ., quotes, extraneous spaces).
  - A small, strictly defined set of interchangeable question frames:
      "what is", "what's", "whats", "what are", "what're",
      "define", "definition of", "tell me about", "explain"
  - Strips leading articles ('a', 'an', 'the') following a matched frame.

INVARIANTS & EXCLUSIONS:
  - "what is X" and "X?" fold to "X".
  - "what is X" vs "how does X work" do NOT fold (mechanism vs identity).
  - Exact canonical match routes directly to AUTO_REUSE without embedding search.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


# Small, strictly closed list of interchangeable definitional question frames.
# Notice: DOES NOT include "how does ... work", "why is", "where is", "when is".
QUESTION_FRAMES: Tuple[str, ...] = (
    "what is ",
    "what's ",
    "whats ",
    "what are ",
    "what're ",
    "define ",
    "definition of ",
    "tell me about ",
    "explain ",
)

LEADING_ARTICLES: Tuple[str, ...] = ("a ", "an ", "the ")

# Punctuation to strip from edges and normalize
_EDGE_PUNCT_RE = re.compile(r"^[\s\"'¿¡]+|[\s\"'?!.,;:]+$")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Canonicalize a query string into a normalized key.

    Args:
        query: Incoming raw query string.

    Returns:
        Canonical string for exact-match caching.
    """
    if not query:
        return ""

    text = query.strip().lower()
    # Strip leading/trailing punctuation and quotes
    text = _EDGE_PUNCT_RE.sub("", text)
    # Collapse multiple whitespaces
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Check for interchangeable definitional frames
    for frame in QUESTION_FRAMES:
        if text.startswith(frame):
            stripped = text[len(frame):].strip()
            # Strip optional leading article following the question frame
            for art in LEADING_ARTICLES:
                if stripped.startswith(art):
                    stripped = stripped[len(art):].strip()
                    break
            # Re-strip edge punctuation if any was exposed
            stripped = _EDGE_PUNCT_RE.sub("", stripped)
            if stripped:
                return stripped

    return text


class CanonicalCache:
    """Exact canonical match cache index.

    Maps canonical query strings directly to cached entries for instant
    AUTO_REUSE without running embedding models or vector searches.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Lookup by normalized canonical form."""
        key = normalize_query(query)
        if not key:
            return None
        return self._entries.get(key)

    def put(self, query: str, entry: Dict[str, Any]) -> str:
        """Store an entry under its normalized canonical form."""
        key = normalize_query(query)
        if key:
            self._entries[key] = entry
        return key

    def __contains__(self, query: str) -> bool:
        key = normalize_query(query)
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
