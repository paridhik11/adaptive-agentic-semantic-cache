"""Download the checkpoints used by the local live-chat semantic verifier."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

from src.policy.local_models import (
    ENCODER_MODEL, ENCODER_REVISION, PAIR_MODEL, PAIR_REVISION, PASSAGE_MODEL, PASSAGE_REVISION,
)


if __name__ == "__main__":
    for model, revision in (
        (ENCODER_MODEL, ENCODER_REVISION), (PAIR_MODEL, PAIR_REVISION), (PASSAGE_MODEL, PASSAGE_REVISION),
    ):
        print(f"Preparing {model}", flush=True)
        snapshot_download(model, revision=revision,
                          allow_patterns=["*.json", "model.safetensors"])
