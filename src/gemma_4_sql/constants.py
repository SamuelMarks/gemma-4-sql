"""Default backend, dataset, and official Gemma 4 model identifier constants.

Attributes:
    DEFAULT_BACKEND: Default execution backend identifier (`'jax'`).
    DEFAULT_PRETRAIN_DATASET: Default pretraining dataset identifier.
    DEFAULT_SFT_DATASET: Default supervised fine-tuning dataset identifier.
    DEFAULT_POSTTRAIN_DATASET: Default DPO post-training dataset identifier.
    MODEL_GEMMA4_E2B: Official Gemma 4 E2B base model identifier.
    MODEL_GEMMA4_E2B_IT: Official Gemma 4 E2B instruction-tuned model identifier.
    MODEL_GEMMA4_E4B: Official Gemma 4 E4B base model identifier.
    MODEL_GEMMA4_E4B_IT: Official Gemma 4 E4B instruction-tuned model identifier.
    MODEL_GEMMA4_26B_A4B: Official Gemma 4 26B-A4B MoE base model identifier.
    MODEL_GEMMA4_26B_A4B_IT: Official Gemma 4 26B-A4B MoE instruction-tuned model identifier.
    MODEL_GEMMA4_31B: Official Gemma 4 31B base model identifier.
    MODEL_GEMMA4_31B_IT: Official Gemma 4 31B instruction-tuned model identifier.

"""

from __future__ import annotations

DEFAULT_BACKEND: str = "jax"
DEFAULT_PRETRAIN_DATASET: str = "my-custom-dataset"
DEFAULT_SFT_DATASET: str = "my-custom-sft-dataset"
DEFAULT_POSTTRAIN_DATASET: str = "my-custom-dpo-dataset"
MODEL_GEMMA4_E2B: str = "google/gemma-4-E2B"
MODEL_GEMMA4_E2B_IT: str = "google/gemma-4-E2B-it"
MODEL_GEMMA4_E4B: str = "google/gemma-4-E4B"
MODEL_GEMMA4_E4B_IT: str = "google/gemma-4-E4B-it"
MODEL_GEMMA4_26B_A4B: str = "google/gemma-4-26B-A4B"
MODEL_GEMMA4_26B_A4B_IT: str = "google/gemma-4-26B-A4B-it"
MODEL_GEMMA4_31B: str = "google/gemma-4-31B"
MODEL_GEMMA4_31B_IT: str = "google/gemma-4-31B-it"

__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_POSTTRAIN_DATASET",
    "DEFAULT_PRETRAIN_DATASET",
    "DEFAULT_SFT_DATASET",
    "MODEL_GEMMA4_26B_A4B",
    "MODEL_GEMMA4_26B_A4B_IT",
    "MODEL_GEMMA4_31B",
    "MODEL_GEMMA4_31B_IT",
    "MODEL_GEMMA4_E2B",
    "MODEL_GEMMA4_E2B_IT",
    "MODEL_GEMMA4_E4B",
    "MODEL_GEMMA4_E4B_IT",
]
