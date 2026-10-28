"""Models module for training, pretraining, and posttraining."""

from __future__ import annotations

import copy

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import TrainingConfig, TrainResultDict

__all__ = [
    "TrainingConfig",
    "posttrain_model",
    "pretrain_model",
    "sft_model",
    "train_from_scratch",
]


def _clone_training_config(config: TrainingConfig | None, *, action: str, default_backend: str = DEFAULT_BACKEND) -> TrainingConfig:
    """Deep-copy a TrainingConfig or create a default one with the specified action.

    Args:
        config: Optional source TrainingConfig to copy.
        action: Training stage action (`'train_from_scratch'`, `'pretrain'`, `'sft'`, `'posttrain'`).
        default_backend: Default backend name when `config` is `None`.

    Returns:
        An isolated `TrainingConfig` instance whose `extra_kwargs` will not mutate the caller's dict.

    """
    cfg = copy.deepcopy(config) if config is not None else TrainingConfig(backend=default_backend)
    cfg.action = action
    return cfg


def _route_training(config: TrainingConfig) -> TrainResultDict:
    """Route training request to the appropriate backend.

    Args:
        config: The configuration parameters.

    Returns:
        A dictionary containing the results.

    """
    backend = config.backend

    return get_backend(backend).train_model(config)


def train_from_scratch(config: TrainingConfig | None = None) -> TrainResultDict:
    """Train a model from scratch.

    Args:
        config: Optional TrainingConfig object specifying training parameters.

    Returns:
        A dictionary indicating the training job status.

    """
    cfg = _clone_training_config(config, action="train_from_scratch", default_backend=DEFAULT_BACKEND)
    return _route_training(cfg)


def pretrain_model(config: TrainingConfig | None = None, *, default_backend: str = "maxtext") -> TrainResultDict:
    """Pretrains an existing model.

    Args:
        config: Optional TrainingConfig object specifying pretraining parameters.
        default_backend: Backend used when `config` is None (defaults to `'maxtext'`).

    Returns:
        A dictionary indicating the pretraining job status.

    """
    cfg = _clone_training_config(config, action="pretrain", default_backend=default_backend)
    return _route_training(cfg)


def sft_model(config: TrainingConfig | None = None) -> TrainResultDict:
    """Supervised fine-tunes (SFT) an existing model.

    Args:
        config: Optional TrainingConfig object specifying SFT parameters.

    Returns:
        A dictionary indicating the SFT job status.

    """
    cfg = _clone_training_config(config, action="sft", default_backend=DEFAULT_BACKEND)
    return _route_training(cfg)


def posttrain_model(config: TrainingConfig | None = None, *, default_backend: str = "keras") -> TrainResultDict:
    """Post-trains an existing model (e.g., RLHF, DPO).

    Args:
        config: Optional TrainingConfig object specifying posttraining parameters.
        default_backend: Backend used when `config` is None (defaults to `'keras'`).

    Returns:
        A dictionary indicating the post-training job status.

    """
    cfg = _clone_training_config(config, action="posttrain", default_backend=default_backend)
    return _route_training(cfg)
