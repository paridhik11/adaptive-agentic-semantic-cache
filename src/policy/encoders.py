"""Local retrieval encoders; construction never downloads weights implicitly."""
from collections import Counter
import hashlib
import math
import re


def normalize(vector):
    values = [float(x) for x in vector]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("invalid embedding")
    norm = math.sqrt(sum(x * x for x in values))
    if not norm:
        raise ValueError("empty embedding")
    return [x / norm for x in values]


class LexicalEncoder:
    """Deterministic hashing baseline, not a learned semantic embedding."""

    identity = "lexical-sha256-unigram-bigram-512-v1"

    def encode(self, text):
        tokens = re.findall(r"\w+", text.casefold())
        features = tokens + [" ".join(pair) for pair in zip(tokens, tokens[1:])]
        vector = [0.0] * 512
        for feature, count in Counter(features).items():
            vector[int(hashlib.sha256(feature.encode()).hexdigest()[:8], 16) % 512] += count
        return normalize(vector)


class DenseEncoder:
    """Require an immutable revision already present in the local model cache."""

    def __init__(self, model, revision):
        if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
            raise ValueError("dense encoder requires a full immutable model revision")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, revision=revision, local_files_only=True)
        self.identity = model + "@" + revision

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("nonempty text required")
        tokens = self.model.tokenizer(text, truncation=False)["input_ids"]
        if len(tokens) > self.model.max_seq_length:
            raise ValueError("dense input exceeds model window; no silent truncation")
        return normalize(self.model.encode(text, normalize_embeddings=True).tolist())
