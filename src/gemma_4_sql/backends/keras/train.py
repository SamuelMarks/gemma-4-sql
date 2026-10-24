"""Keras-specific model training/finetuning logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import ETLConfig, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import TrainResultDict
logger = logging.getLogger(__name__)

keras = optional_import("keras")
tf = optional_import("tensorflow")
GemmaCausalLM = optional_attr("keras_nlp.models", "GemmaCausalLM")


def _execute_train(model_name: str, dataset: str, epochs: int, batch_size: int = 2) -> tuple[str, float]:
    """Execute the core training loop for Keras.

    Args:
        model_name: The target model name.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        batch_size: Batch size.

    Returns:
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If Keras or TensorFlow dependencies are missing.
        ValueError: If model loading or dataloader fails.

    """
    if keras is None or tf is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    model: Any = None
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        try:
            gemma_causal_lm_cls = GemmaCausalLM or optional_attr("keras_nlp.models", "GemmaCausalLM")
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
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


def train_model(config: TrainingConfig, **kwargs: object) -> TrainResultDict:
    """Train a Text-to-SQL model using Keras.

    Args:
        config: Training configuration object.
        **kwargs: Extra runtime options such as `'distributed_strategy'` or `'raise_on_error'`.

    Returns:
        A dictionary containing training status and final metrics.

    Raises:
        DependencyMissingError: If Keras training dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    extra = getattr(config, "extra_kwargs", {}) or {}
    raise_on_error = bool(kwargs.get("raise_on_error", extra.get("raise_on_error", False)))

    if keras is None or tf is None:
        raise DependencyMissingError("Keras training dependencies are missing.", backend="keras")

    logger.info("Starting Keras %s on %s using %s", action, model_name, dataset)
    final_loss = 0.0
    try:
        batch_size = getattr(config, "batch_size", 2)
        status, final_loss = _execute_train(model_name, dataset, epochs, batch_size=batch_size)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        logger.exception("Keras training error: ")
        if raise_on_error:
            raise TrainingError(
                f"Keras training failed: {e!s}",
                backend="keras",
                model_name=model_name,
                operation=action,
                cause=e,
            ) from e
        status = f"failed: {e!s}"
    return {"backend": "keras", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "status": status, "final_loss": final_loss}
