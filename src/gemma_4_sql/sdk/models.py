"""Models module for training, pretraining, and posttraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


@dataclass
class TrainingConfig:
    """Configuration for training jobs."""

    dataset: str
    action: str = ""
    model_name: str = "gemma-4"
    epochs: int = 1
    learning_rate: float = 0.0001
    backend: str = "jax"
    distributed_strategy: str = "none"
    extra_kwargs: dict[str, object] = field(default_factory=dict)


if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def _route_training(config: TrainingConfig) -> JSONDict:
    """Route training request to the appropriate backend.

    Args:
        config: The configuration parameters.

    Returns:
        A dictionary containing the results.
    """
    backend = config.backend
    if config.extra_kwargs is None:
        config.extra_kwargs = {}
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).train_model(config)


def train_from_scratch(config: TrainingConfig | None = None) -> JSONDict:
    """Train a model from scratch.

    Args:
    ----
        config: A TrainingConfig object specifying all parameters.

    Returns:
    -------
        A dictionary indicating the training job status.

    """
    cfg = config or TrainingConfig()
    cfg.action = "train_from_scratch"
    return _route_training(cfg)


def pretrain_model(config: TrainingConfig | None = None) -> JSONDict:
    """Pretrains an existing model.

    Args:
    ----
        config: A TrainingConfig object specifying all parameters.

    Returns:
    -------
        A dictionary indicating the pretraining job status.

    """
    cfg = config or TrainingConfig(backend="maxtext")
    cfg.action = "pretrain"
    return _route_training(cfg)


def sft_model(config: TrainingConfig | None = None) -> JSONDict:
    """Supervised fine-tunes (SFT) an existing model.

    Args:
    ----
        config: A TrainingConfig object specifying all parameters.

    Returns:
    -------
        A dictionary indicating the SFT job status.

    """
    cfg = config or TrainingConfig()
    cfg.action = "sft"
    return _route_training(cfg)


def posttrain_model(config: TrainingConfig | None = None) -> JSONDict:
    """Post-trains an existing model (e.g., RLHF, DPO).

    Args:
    ----
        config: A TrainingConfig object specifying all parameters.

    Returns:
    -------
        A dictionary indicating the post-training job status.

    """
    cfg = config or TrainingConfig(backend="keras")
    cfg.action = "posttrain"
    return _route_training(cfg)
