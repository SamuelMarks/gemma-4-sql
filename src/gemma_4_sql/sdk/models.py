"""
Models module for training, pretraining, and posttraining.
"""

from __future__ import annotations

from typing import Any

def _route_training(
    action: str,
    model_name: str,
    dataset: str,
    epochs: int,
    learning_rate: float,
    backend: str,
) -> dict[str, Any]:
    """Route training request."""
    kwargs = {
        "action": action,
        "model_name": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }

    if backend == "jax":
        from gemma_4_sql.backends.jax.train import train_model

        return train_model(**kwargs)
    elif backend == "keras":
        from gemma_4_sql.backends.keras.train import train_model

        return train_model(**kwargs)
    elif backend == "maxtext":
        from gemma_4_sql.backends.maxtext.train import train_model

        return train_model(**kwargs)
    elif backend == "pytorch":
        from gemma_4_sql.backends.pytorch.train import train_model

        return train_model(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def train_from_scratch(
    model_name: str = "gemma-4",
    dataset: str = "dummy_dataset",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    backend: str = "jax",
) -> dict[str, Any]:
    """
    Trains a model from scratch.

    Args:
        model_name: The name of the model to train.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
        A dictionary indicating the training job status.
    """
    return _route_training(
        "train_from_scratch", model_name, dataset, epochs, learning_rate, backend
    )

def pretrain_model(
    model_name: str = "gemma-4",
    dataset: str = "dummy_dataset",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    backend: str = "maxtext",
) -> dict[str, Any]:
    """
    Pretrains an existing model.

    Args:
        model_name: The name of the model to pretrain.
        dataset: The dataset to pretrain on.
        epochs: Number of epochs to pretrain.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
        A dictionary indicating the pretraining job status.
    """
    return _route_training(
        "pretrain", model_name, dataset, epochs, learning_rate, backend
    )

def sft_model(
    model_name: str = "gemma-4",
    dataset: str = "dummy_dataset",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    backend: str = "jax",
) -> dict[str, Any]:
    """
    Supervised fine-tunes (SFT) an existing model.

    Args:
        model_name: The name of the model to fine-tune.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
        A dictionary indicating the SFT job status.
    """
    return _route_training("sft", model_name, dataset, epochs, learning_rate, backend)

def posttrain_model(
    model_name: str = "gemma-4",
    dataset: str = "dummy_dataset",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    backend: str = "keras",
) -> dict[str, Any]:
    """
    Post-trains an existing model (e.g., RLHF, DPO).

    Args:
        model_name: The name of the model to post-train.
        dataset: The dataset to post-train on.
        epochs: Number of epochs to post-train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
        A dictionary indicating the post-training job status.
    """
    return _route_training(
        "posttrain", model_name, dataset, epochs, learning_rate, backend
    )
