"""Keras-specific model training/finetuning logic."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_train import run_train_wrapper
from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    ETLConfig,
    KerasCausalLMProtocol,
    TensorType,
    TfModuleProtocol,
    TfTensorProtocol,
    TrainingConfig,
    TrainResultDict,
)


@runtime_checkable
class _KerasFitHistoryProtocol(Protocol):
    """Protocol for a Keras `.fit(...)` `History` object exposing `.history`."""

    @property
    def history(self) -> Mapping[str, list[float]]:
        """Provide the dictionary of recorded training metric lists."""
        ...


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


keras = optional_import("keras")
tf = optional_import("tensorflow")
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = optional_attr(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)


def _execute_train(
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size: int = 2,
    learning_rate: float = 5e-05,
) -> tuple[str, float]:
    """Execute the core training loop for Keras.

    Args:
        model_name: The target model name.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If Keras or TensorFlow dependencies are missing.
        ValueError: If model loading or dataloader fails.

    """
    if keras is None or tf is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        try:
            gemma_causal_lm_cls = GemmaCausalLM or optional_attr("keras_nlp.models", "GemmaCausalLM", _GemmaCausalLMLoaderProtocol)
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
            model: KerasCausalLMProtocol = gemma_causal_lm_cls.from_preset(model_name)
            model.preprocessor.sequence_length = 512  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue,reportFunctionMemberAccess]  # ty: ignore[invalid-assignment]
            model.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
                metrics=["accuracy"],
            )
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load Keras model {model_name}") from e

    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    if dataloader is None:
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    def _iter_keras_batches() -> Iterator[tuple[TfTensorProtocol, TfTensorProtocol] | Mapping[str, TensorType | Sequence[int]]]:
        """Yield `(inputs, targets)` batches for `keras.Model.fit` across all dataloader items."""
        assert tf is not None
        for batch in dataloader:
            if isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
                yield (tf.constant(batch["inputs"], dtype=tf.int32), tf.constant(batch["targets"], dtype=tf.int32))
            else:
                yield batch

    history: _KerasFitHistoryProtocol = model.fit(_iter_keras_batches(), epochs=epochs)  # pyright: ignore[reportAssignmentType,reportArgumentType]  # ty: ignore[invalid-assignment, invalid-argument-type]
    final_loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
    return "completed", final_loss


def train_model(config: TrainingConfig, **kwargs: StatusValue) -> TrainResultDict:
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

    model_name = config.model_name
    dataset = config.dataset
    epochs = config.epochs
    batch_size = config.batch_size
    learning_rate = float(config.learning_rate)
    return run_train_wrapper(
        "keras",
        config,
        lambda: _execute_train(model_name, dataset, epochs, batch_size=batch_size, learning_rate=learning_rate),
        **kwargs,
    )


__all__ = [
    "GemmaCausalLM",
    "KerasCausalLMProtocol",
    "TfModuleProtocol",
    "train_model",
]
