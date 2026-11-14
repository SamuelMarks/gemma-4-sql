"""Keras-specific model training/finetuning logic."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from gemma_4_sql._types_models import KerasCompiledOptimizerProtocol
from gemma_4_sql._types_results import ETLBatchValue
from gemma_4_sql.backends.common_train import run_train_wrapper
from gemma_4_sql.backends.keras._env import resolve_and_validate_keras_backend
from gemma_4_sql.backends.keras._runtime import (
    KerasWithDistributionProtocol,
    enter_keras_distribution_scope,
    load_keras_gemma_model,
    load_preset_with_hf_fallback,
    synchronize_keras_outputs,
)
from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import (
    ModuleNamespaceProtocol,
    import_optional_symbol,
    optional_import,
)
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    DTypeProtocol,
    ETLConfig,
    KerasCausalLMProtocol,
    TensorProtocol,
    TfModuleProtocol,
    TfTensorProtocol,
    TrainingConfig,
    TrainResultDict,
)


@runtime_checkable
class _KerasFitHistoryProtocol(Protocol):
    """Protocol for a Keras `.fit(...)` `History` object exposing `.history`."""

    @property
    def history(self) -> Mapping[str, Sequence[float]]:
        """Provide the dictionary of recorded training metric lists."""
        ...


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


@runtime_checkable
class _TFStrategyProtocol(Protocol):
    """Protocol for a TensorFlow `MirroredStrategy` exposing `.scope()`."""

    def scope(self) -> AbstractContextManager[None]:
        """Return a distribution strategy scope context manager.

        Returns:
            Strategy scope context manager.

        """
        ...


@runtime_checkable
class _TFDistributeNamespaceProtocol(Protocol):
    """Protocol for `tf.distribute` exposing `MirroredStrategy`."""

    def MirroredStrategy(self) -> _TFStrategyProtocol:
        """Construct a `MirroredStrategy` instance.

        Returns:
            Initialized `_TFStrategyProtocol`.

        """
        ...


@runtime_checkable
class _TFTrainModuleProtocol(Protocol):
    """Protocol for `tensorflow` distribution strategy and integer tensor constants in Keras training."""

    @property
    def distribute(self) -> _TFDistributeNamespaceProtocol:
        """Provide `tf.distribute` namespace."""
        ...

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `tf.int32` dtype."""
        ...

    def constant(self, value: ETLBatchValue, *, dtype: DTypeProtocol) -> TfTensorProtocol:
        """Construct a constant `TfTensorProtocol` from `value`.

        Args:
            value: Batch input or target sequence/tensor.
            dtype: Target TensorFlow dtype.

        Returns:
            Constant `TfTensorProtocol` tensor.

        """
        ...


@runtime_checkable
class _KerasLossesNamespaceProtocol(Protocol):
    """Protocol for `keras.losses` exposing `SparseCategoricalCrossentropy`."""

    def SparseCategoricalCrossentropy(self, *, from_logits: bool = True) -> Callable[..., TensorProtocol]:
        """Construct a sparse categorical cross-entropy loss instance.

        Args:
            from_logits: Whether predictions are unnormalized logits.

        Returns:
            Configured cross-entropy loss callable.

        """
        ...


@runtime_checkable
class _KerasOptimizersNamespaceProtocol(Protocol):
    """Protocol for `keras.optimizers` exposing `AdamW`."""

    def AdamW(self, *, learning_rate: float) -> KerasCompiledOptimizerProtocol:
        """Construct an `AdamW` Keras optimizer.

        Args:
            learning_rate: Optimizer learning rate.

        Returns:
            Configured `KerasCompiledOptimizerProtocol` optimizer.

        """
        ...


@runtime_checkable
class _KerasTrainModuleProtocol(Protocol):
    """Protocol for `keras` module exposing `.losses` and `.optimizers`."""

    @property
    def losses(self) -> _KerasLossesNamespaceProtocol:
        """Provide `keras.losses` namespace."""
        ...

    @property
    def optimizers(self) -> _KerasOptimizersNamespaceProtocol:
        """Provide `keras.optimizers` namespace."""
        ...


@runtime_checkable
class _KerasOpsConvertProtocol(Protocol):
    """Protocol for `keras.ops` exposing `convert_to_tensor`."""

    def convert_to_tensor(self, x: ETLBatchValue, *, dtype: str = "int32") -> TfTensorProtocol:
        """Convert `x` to a backend-native Keras tensor."""
        ...


@runtime_checkable
class _KerasWithOpsProtocol(Protocol):
    """Protocol for `keras` module exposing `.ops`."""

    @property
    def ops(self) -> _KerasOpsConvertProtocol:
        """Provide `keras.ops` namespace."""
        ...


keras: _KerasTrainModuleProtocol | None = optional_import("keras", _KerasTrainModuleProtocol)
tf: _TFTrainModuleProtocol | None = optional_import("tensorflow", _TFTrainModuleProtocol)
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = import_optional_symbol(
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
    *,
    active_backend: str = "tensorflow",
    hardware: str = "cpu",
    strict_hardware: bool = False,
) -> tuple[str, float]:
    """Execute the core training loop for Keras across JAX, TensorFlow, or PyTorch backends.

    Args:
        model_name: The target model name.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        active_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`).
        hardware: Target hardware identifier (`'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).
        strict_hardware: Whether to fail fast when requested TPU/GPU hardware is unavailable.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If Keras (or TensorFlow on the TF backend) is missing.
        ImportError: Internally raised and wrapped if KerasHub/KerasNLP GemmaCausalLM is unavailable.
        ValueError: If model loading or dataloader fails.

    """
    if keras is None or (active_backend == "tensorflow" and tf is None):
        raise DependencyMissingError("Keras dependencies are missing.")

    tf_mod_ns: ModuleNamespaceProtocol | None = tf if isinstance(tf, ModuleNamespaceProtocol) else None
    dist_keras = keras if isinstance(keras, KerasWithDistributionProtocol) else None

    def _load_and_compile() -> KerasCausalLMProtocol:
        """Load preset and compile model with `SparseCategoricalCrossentropy` and `AdamW`.

        Returns:
            Compiled `KerasCausalLMProtocol` instance.

        Raises:
            ImportError: Internally raised and wrapped if KerasHub/KerasNLP is unavailable.
            ValueError: If preset loading fails.

        """
        assert keras is not None
        try:
            gemma_causal_lm_cls = load_keras_gemma_model(
                model_name,
                _GemmaCausalLMLoaderProtocol,
                explicit_cls=GemmaCausalLM,
                symbol_importer=import_optional_symbol,
            )
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_hub / keras_nlp GemmaCausalLM is not available.")
            loaded: KerasCausalLMProtocol = load_preset_with_hf_fallback(gemma_causal_lm_cls, model_name)
            loaded.preprocessor.sequence_length = 512
            loaded.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
                metrics=["accuracy"],
            )
            return loaded
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load Keras model {model_name}") from e

    resolved_dev = "cpu"
    if active_backend == "tensorflow" and tf is not None and hardware == "cpu":
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = _load_and_compile()
    else:
        with enter_keras_distribution_scope(
            active_backend,
            hardware,
            tf_mod=tf_mod_ns,
            keras_mod=dist_keras,
            strict_hardware=strict_hardware,
            backend_label=f"keras_{active_backend}",
        ) as (dev_str, _):
            resolved_dev = dev_str
            model = _load_and_compile()

    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    if dataloader is None:
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    def _to_int32_tensor(val: ETLBatchValue) -> TfTensorProtocol:
        """Convert `val` to an int32 tensor using `tf.constant` on TF or `keras.ops` on JAX/PyTorch.

        Args:
            val: Batch input or target array/sequence.

        Returns:
            Converted int32 tensor.

        Raises:
            DependencyMissingError: If neither `tf.constant` nor `keras.ops.convert_to_tensor` is available.

        """
        if active_backend == "tensorflow" and tf is not None:
            return tf.constant(val, dtype=tf.int32)
        if isinstance(keras, _KerasWithOpsProtocol):
            return keras.ops.convert_to_tensor(val, dtype="int32")
        raise DependencyMissingError("Keras tensor conversion requires keras.ops or tensorflow.")

    def _iter_keras_batches() -> Iterator[tuple[TfTensorProtocol, TfTensorProtocol] | Mapping[str, ETLBatchValue]]:
        """Yield `(inputs, targets)` batches for `keras.Model.fit` across all dataloader items.

        Yields:
            Tuple of `(inputs_tensor, targets_tensor)` or raw batch mapping.

        """
        for batch in dataloader:
            if isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
                yield (_to_int32_tensor(batch["inputs"]), _to_int32_tensor(batch["targets"]))
            else:
                yield batch
            if active_backend == "torch" and resolved_dev.startswith("xla"):
                synchronize_keras_outputs("torch", None, device=resolved_dev)

    history: _KerasFitHistoryProtocol = model.fit(_iter_keras_batches(), epochs=epochs)
    if active_backend in {"jax", "torch"}:
        synchronize_keras_outputs(active_backend, None, device=resolved_dev)
    final_loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
    return "completed", final_loss


def train_model(config: TrainingConfig, **kwargs: StatusValue) -> TrainResultDict:
    """Train a Text-to-SQL model using Keras 3 (`keras_jax`, `keras_tensorflow`, or `keras_pytorch`).

    Args:
        config: Training configuration object.
        **kwargs: Extra runtime options such as `'backend_alias'`, `'hardware'`, `'strict_hardware'`.

    Returns:
        A dictionary containing training status and final metrics.

    Raises:
        DependencyMissingError: If Keras training dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    raw_alias = (
        kwargs.get("backend_alias")
        or kwargs.get("backend")
        or (config.backend if config.backend.startswith("keras_") else "keras")
    )
    backend_alias = str(raw_alias)
    hardware = str(kwargs.get("hardware", config.extra_kwargs.get("hardware", "cpu")))
    strict_hw = bool(kwargs.get("strict_hardware", config.extra_kwargs.get("strict_hardware", False)))
    raw_kb = kwargs.get("keras_backend", config.extra_kwargs.get("keras_backend"))
    explicit_kb = str(raw_kb) if isinstance(raw_kb, str) and raw_kb else None

    if backend_alias == "keras" and explicit_kb is None and tf is not None:
        active_kb = "tensorflow"
    else:
        active_kb = resolve_and_validate_keras_backend(backend_alias, explicit_kb)

    if keras is None or (active_kb == "tensorflow" and tf is None):
        raise DependencyMissingError("Keras training dependencies are missing.", backend=backend_alias)

    model_name = config.model_name
    dataset = config.dataset
    epochs = config.epochs
    batch_size = config.batch_size
    learning_rate = float(config.learning_rate)
    res = run_train_wrapper(
        backend_alias,
        config,
        lambda: _execute_train(
            model_name,
            dataset,
            epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            active_backend=active_kb,
            hardware=hardware,
            strict_hardware=strict_hw,
        ),
        **kwargs,
    )
    res["keras_backend"] = active_kb
    return res


__all__ = [
    "GemmaCausalLM",
    "KerasCausalLMProtocol",
    "TfModuleProtocol",
    "train_model",
]
