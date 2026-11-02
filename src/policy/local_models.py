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


PASSAGE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PASSAGE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class LocalPairScorer:
    """CPU pair diagnostic with explicit logit conversion and truncation refusal."""

    def __init__(self, model=PAIR_MODEL, revision=PAIR_REVISION):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("pair model requires a full immutable revision")
        import torch
        from sentence_transformers import CrossEncoder

        kwargs = {"local_files_only": True, "device": "cpu", "trust_remote_code": False}
        import inspect
        sig = inspect.signature(CrossEncoder.__init__)
        if "activation_fn" in sig.parameters:
            kwargs["activation_fn"] = torch.nn.Identity()
            kwargs["model_kwargs"] = {"use_safetensors": True}
        else:
            kwargs["default_activation_function"] = torch.nn.Identity()
            kwargs["automodel_args"] = {"use_safetensors": True}
        self.model = CrossEncoder(model, revision=revision, **kwargs)
        self.identity = model + "@" + revision

    def score(self, first, second):
        if not all(isinstance(x, str) and x.strip() for x in (first, second)):
            raise ValueError("two nonempty texts required")
        tokens = self.model.tokenizer(first, second, truncation=False)["input_ids"]
        limits = [self.model.tokenizer.model_max_length]
        max_seq = getattr(self.model, "max_seq_length", None)
        if max_seq is None:
            max_seq = getattr(self.model, "max_length", None)
        if max_seq is not None:
            limits.append(max_seq)
        if len(tokens) > min(limits):
            raise ValueError("pair exceeds model input window; no silent truncation")
        logit = float(self.model.predict([(first, second)], show_progress_bar=False)[0])
        if not math.isfinite(logit):
            raise ValueError("nonfinite pair score")
        # Stable sigmoid. The model's activation is explicitly Identity above.
        probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        return {"logit": logit, "pair_score": probability, "input_tokens": len(tokens)}


class LocalPassageScorer:
    """Local cross-encoder scoring query relevance against actual answer content."""

    def __init__(self, model: str = PASSAGE_MODEL, revision: str = PASSAGE_REVISION):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("passage scorer requires an immutable 40-character revision")
        import threading
        import torch
        from sentence_transformers import CrossEncoder

        self.torch = torch
        self.model = CrossEncoder(
            model,
            revision=revision,
            local_files_only=True,
            device="cpu",
            trust_remote_code=False,
            activation_fn=torch.nn.Identity(),
            model_kwargs={"use_safetensors": True},
        )
        self.identity = f"{model}@{revision}"
        self._lock = threading.Lock()

    def score(self, query: str, passage: str) -> dict[str, float | int]:
        if not all(isinstance(x, str) and x.strip() for x in (query, passage)):
            raise ValueError("two nonempty strings required for passage scoring")
        tokens = self.model.tokenizer(query, passage, truncation=False)["input_ids"]
        limits = [self.model.tokenizer.model_max_length]
        max_seq = getattr(self.model, "max_seq_length", None)
        if max_seq is None:
            max_seq = getattr(self.model, "max_length", None)
        if max_seq is not None:
            limits.append(max_seq)
        if len(tokens) > min(limits):
            raise ValueError("query-passage pair exceeds model input window; no silent truncation")

        with self._lock:
            logit = float(self.model.predict([(query, passage)], show_progress_bar=False)[0])

        if not math.isfinite(logit):
            raise ValueError("nonfinite passage score logit")

        # Numerically stable sigmoid
        prob = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
        return {
            "logit": logit,
            "probability": prob,
            "input_tokens": len(tokens),
        }

