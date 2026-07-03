"""Models module for training, pretraining, and posttraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


@dataclass
class TrainingConfig:
    """Configuration for training jobs."""

    action: str = ""
    model_name: str = "gemma-4"
    dataset: str = "dummy_dataset"
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
    ----
        config: A TrainingConfig object specifying all parameters.

    Returns:
    -------
        A dictionary indicating the training job status and metrics.

    """
    backend = config.backend
    train_kwargs = {"action": config.action, "model_name": config.model_name, "dataset": config.dataset, "epochs": config.epochs, "learning_rate": config.learning_rate}
    if backend == "pytorch" and config.distributed_strategy != "none":
        train_kwargs["distributed_strategy"] = config.distributed_strategy
    train_kwargs.update(config.extra_kwargs)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).train_model(**train_kwargs)


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
