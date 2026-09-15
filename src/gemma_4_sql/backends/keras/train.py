"""Keras-specific model training/finetuning logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.type_hints import ETLConfig, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import keras as _keras
    import tensorflow as _tf

    keras: Any = _keras
    tf: Any = _tf
except (ImportError, AttributeError):
    keras = None
    tf = None


def _execute_train(model_name: str, dataset: str, epochs: int, test_mode: bool, batch_size: int = 2) -> tuple[str, float]:
    """Execute the core training loop.

    Args:
        model_name: Target model name.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        test_mode: Whether to run in test mode.
        batch_size: Training batch size.

    Returns:
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If Keras or TensorFlow dependencies are missing.
        ValueError: If model loading or dataloader fails.
    """
    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")

    model: Any = None
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        try:
            gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
            model = gemma_causal_lm_cls.from_preset(model_name)
            model.preprocessor.sequence_length = 512
            model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.AdamW(learning_rate=5e-05), metrics=["accuracy"])
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load Keras model {model_name}") from e

    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    history = model.fit(dataloader, epochs=epochs)
    final_loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
    return "completed", final_loss


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Train a Text-to-SQL model using Keras.

    Args:
        config: Training configuration object.
        **kwargs: Extra runtime options such as 'test_mode' and 'distributed_strategy'.

    Returns:
        A dictionary containing training status and final metrics.

    Raises:
        DependencyMissingError: If Keras training dependencies are missing.
    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    getattr(config, "learning_rate", 1e-05)

    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras training dependencies are missing.")

    logger.info("Starting Keras %s on %s using %s", action, model_name, dataset)
    test_mode = bool(kwargs.get("test_mode"))
    batch_size = getattr(config, "batch_size", 2)
    try:
        status, final_loss = _execute_train(model_name, dataset, epochs, test_mode, batch_size=batch_size)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Keras training error: ")
        status = f"failed: {e!s}"
        final_loss = 0.0
    return {"backend": "keras", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "status": status, "final_loss": final_loss}
