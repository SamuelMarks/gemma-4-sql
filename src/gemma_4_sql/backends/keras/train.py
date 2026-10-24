"""Keras-specific model training/finetuning logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_train import run_train_wrapper
from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
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
    if keras is None or tf is None:
        raise DependencyMissingError("Keras training dependencies are missing.", backend="keras")

    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    batch_size = getattr(config, "batch_size", 2)
    return run_train_wrapper(
        "keras",
        config,
        lambda: _execute_train(model_name, dataset, epochs, batch_size=batch_size),
        **kwargs,
    )
