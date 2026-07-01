"""Models module for training, pretraining, and posttraining."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def _route_training(action: str, model_name: str, dataset: str, epochs: int, learning_rate: float, **kwargs: JSONValue) -> JSONDict:
    """Route training request to the appropriate backend.

    Args:
    ----
        action: The training action to perform (e.g., 'pretrain', 'sft').
        model_name: The name of the model to train.
        dataset: The dataset to use for training.
        epochs: Number of training epochs.
        learning_rate: The learning rate for training.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        A dictionary indicating the training job status and metrics.

    """
    backend = kwargs.get("backend")
    train_kwargs = {"action": action, "model_name": model_name, "dataset": dataset, "epochs": epochs, "learning_rate": learning_rate}
    if backend == "pytorch" and "distributed_strategy" in kwargs:
        train_kwargs["distributed_strategy"] = kwargs["distributed_strategy"]
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).train_model(**train_kwargs)


def train_from_scratch(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "jax", distributed_strategy: str = "none") -> JSONDict:
    """Train a model from scratch.

    Args:
    ----
        model_name: The name of the model to train.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').
        distributed_strategy: Distributed strategy to use.

    Returns:
    -------
        A dictionary indicating the training job status.

    """
    return _route_training("train_from_scratch", model_name, dataset, epochs, learning_rate, backend=backend, distributed_strategy=distributed_strategy)


def pretrain_model(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "maxtext", distributed_strategy: str = "none") -> JSONDict:
    """Pretrains an existing model.

    Args:
    ----
        model_name: The name of the model to pretrain.
        dataset: The dataset to pretrain on.
        epochs: Number of epochs to pretrain.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').
        distributed_strategy: Distributed strategy to use.

    Returns:
    -------
        A dictionary indicating the pretraining job status.

    """
    return _route_training("pretrain", model_name, dataset, epochs, learning_rate, backend=backend, distributed_strategy=distributed_strategy)


def sft_model(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "jax", distributed_strategy: str = "none") -> JSONDict:
    """Supervised fine-tunes (SFT) an existing model.

    Args:
    ----
        model_name: The name of the model to fine-tune.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').
        distributed_strategy: Distributed strategy to use.

    Returns:
    -------
        A dictionary indicating the SFT job status.

    """
    return _route_training("sft", model_name, dataset, epochs, learning_rate, backend=backend, distributed_strategy=distributed_strategy)


def posttrain_model(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "keras", distributed_strategy: str = "none") -> JSONDict:
    """Post-trains an existing model (e.g., RLHF, DPO).

    Args:
    ----
        model_name: The name of the model to post-train.
        dataset: The dataset to post-train on.
        epochs: Number of epochs to post-train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').
        distributed_strategy: Distributed strategy to use.

    Returns:
    -------
        A dictionary indicating the post-training job status.

    """
    return _route_training("posttrain", model_name, dataset, epochs, learning_rate, backend=backend, distributed_strategy=distributed_strategy)
