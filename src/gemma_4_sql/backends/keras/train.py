"""Keras-specific model training/finetuning logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.type_hints import ETLConfig, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
keras = None
tf = None
with catch_optional_imports():
    import keras
    import tensorflow as tf  # pragma: no cover


def _execute_train(model_name: str, dataset: str, epochs: int, test_mode: bool) -> tuple[str, float]:
    """Execute the core training loop."""
    model: keras.Model | None = None
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        try:
            gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
            model = gemma_causal_lm_cls.from_preset(model_name)
            model.preprocessor.sequence_length = 512
            model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.AdamW(learning_rate=5e-05), metrics=["accuracy"])
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load Keras model {model_name}") from e

    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))
    dataloader = data_dict.get("loader", None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    history = model.fit(dataloader, epochs=epochs)
    final_loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
    return "completed", final_loss


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Execute function.


    Args:
        **kwargs: Extra runtime options such as 'test_mode' and 'distributed_strategy'.
    Returns:
        The execution result.

    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    getattr(config, "learning_rate", 1e-05)
    """Train a Text-to-SQL model using Keras.

    Args:
    ----
        action: The training action ('pretrain', 'sft', 'posttrain').
        model_name: The name or path of the model.
        dataset: The dataset to use for training.
        **kwargs: Additional parameters (e.g., 'test_mode', 'epochs').

    Returns:
    -------
        A dictionary containing training status and final metrics.

    """
    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras training dependencies are missing.")

    logger.info("Starting Keras %s on %s using %s", action, model_name, dataset)
    test_mode = bool(kwargs.get("test_mode"))
    try:
        status, final_loss = _execute_train(model_name, dataset, epochs, test_mode)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Keras training error: ")
        status = f"failed: {e!s}"
        final_loss = 0.0
    return {"backend": "keras", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "status": status, "final_loss": final_loss}
