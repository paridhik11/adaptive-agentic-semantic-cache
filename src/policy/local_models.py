"""Pinned local models for development; never download during construction.

Pair scores describe question similarity, not answer applicability. They must
not be used alone to authorize a semantic cache hit.
"""
from __future__ import annotations

import math
import re

ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PAIR_MODEL = "cross-encoder/quora-distilroberta-base"
PAIR_REVISION = "f62e7a4b20b97195c2868e53ec59126df5eac743"


class LocalPairScorer:
    """CPU pair diagnostic with explicit logit conversion and truncation refusal."""

    def __init__(self, model=PAIR_MODEL, revision=PAIR_REVISION):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("pair model requires a full immutable revision")
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, revision=revision, local_files_only=True,
                                  device="cpu", trust_remote_code=False,
                                  activation_fn=torch.nn.Identity())
        self.identity = model + "@" + revision

    def score(self, first, second):
        if not all(isinstance(x, str) and x.strip() for x in (first, second)):
            raise ValueError("two nonempty texts required")
        tokens = self.model.tokenizer(first, second, truncation=False)["input_ids"]
        limits = [self.model.tokenizer.model_max_length]
        if self.model.max_length is not None:
            limits.append(self.model.max_length)
        if len(tokens) > min(limits):
            raise ValueError("pair exceeds model input window; no silent truncation")
        logit = float(self.model.predict([(first, second)], show_progress_bar=False)[0])
        if not math.isfinite(logit):
            raise ValueError("nonfinite pair score")
        # Stable sigmoid. The model's activation is explicitly Identity above.
        probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        return {"logit": logit, "pair_score": probability, "input_tokens": len(tokens)}
