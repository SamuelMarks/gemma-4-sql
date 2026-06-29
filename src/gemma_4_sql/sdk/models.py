"""Models module for training, pretraining, and posttraining."""

from __future__ import annotations


def _route_training(action: str, model_name: str, dataset: str, epochs: int, learning_rate: float, **kwargs: object) -> dict[str, object]:
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
    kwargs = {"action": action, "model_name": model_name, "dataset": dataset, "epochs": epochs, "learning_rate": learning_rate}
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).train_model(**kwargs)


def train_from_scratch(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "jax") -> dict[str, object]:
    """Train a model from scratch.

    Args:
    ----
        model_name: The name of the model to train.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
    -------
        A dictionary indicating the training job status.

    """
    return _route_training("train_from_scratch", model_name, dataset, epochs, learning_rate, backend)  # type: ignore[call-arg]


def pretrain_model(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "maxtext") -> dict[str, object]:
    """Pretrains an existing model.

    Args:
    ----
        model_name: The name of the model to pretrain.
        dataset: The dataset to pretrain on.
        epochs: Number of epochs to pretrain.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
    -------
        A dictionary indicating the pretraining job status.

    """
    return _route_training("pretrain", model_name, dataset, epochs, learning_rate, backend)  # type: ignore[call-arg]


def sft_model(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "jax") -> dict[str, object]:
    """Supervised fine-tunes (SFT) an existing model.

    Args:
    ----
        model_name: The name of the model to fine-tune.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
    -------
        A dictionary indicating the SFT job status.

    """
    return _route_training("sft", model_name, dataset, epochs, learning_rate, backend)  # type: ignore[call-arg]


def posttrain_model(model_name: str = "gemma-4", dataset: str = "dummy_dataset", epochs: int = 1, learning_rate: float = 0.0001, backend: str = "keras") -> dict[str, object]:
    """Post-trains an existing model (e.g., RLHF, DPO).

    Args:
    ----
        model_name: The name of the model to post-train.
        dataset: The dataset to post-train on.
        epochs: Number of epochs to post-train.
        learning_rate: The learning rate.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
    -------
        A dictionary indicating the post-training job status.

    """
    return _route_training("posttrain", model_name, dataset, epochs, learning_rate, backend)  # type: ignore[call-arg]
