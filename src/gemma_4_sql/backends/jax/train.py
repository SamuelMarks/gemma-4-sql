"""JAX-specific training pipeline."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, SupportsFloat, runtime_checkable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_dpo import ScalarItemProtocol
from gemma_4_sql.backends.common_inference import call_jax_causal_lm
from gemma_4_sql.backends.common_train import (
    NNXOptimizerProtocol,
    create_nnx_optimizer,
    generic_run_training_epochs,
    run_train_wrapper,
)
from gemma_4_sql.backends.common_train import (
    apply_nnx_optimizer_update as _apply_nnx_optimizer_update,
)
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.jax.inference import MODEL_CACHE as _MODEL_CACHE
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import ETLConfig, JaxCausalLMProtocol, TrainingConfig, TrainingLoopState, TrainResultDict

Gemma4Config = ModelConfig
type JaxTrainStepFn = Callable[..., Array | ScalarItemProtocol | SupportsFloat | float]


@runtime_checkable
class _NNXValueAndGradModuleProtocol(Protocol):
    """Protocol for a `flax.nnx` module exposing `value_and_grad` and `jit`."""

    def value_and_grad(self, fn: Callable[..., Array], /) -> Callable[..., tuple[Array, Array]]:
        """Create a value-and-gradient function for an NNX module."""
        ...

    def jit(self, fn: JaxTrainStepFn, /) -> JaxTrainStepFn:
        """JIT-compile an NNX function."""
        ...


def _create_nnx_optimizer(
    model: JaxCausalLMProtocol | Gemma4ForCausalLM | nnx.Module,
    tx: optax.GradientTransformation,
) -> nnx.Optimizer[nnx.Module]:
    """Create a Flax NNX Optimizer (`nnx.Optimizer(model, tx, wrt=nnx.Param)`).

    Args:
        model: Flax NNX module to optimize.
        tx: Optax gradient transformation.

    Returns:
        Configured `nnx.Optimizer` instance.

    """
    res: nnx.Optimizer[nnx.Module] = create_nnx_optimizer(nnx, model, tx)  # pyright: ignore[reportAssignmentType]
    return res


def _loss_fn(
    model: Callable[..., Array] | JaxCausalLMProtocol | Gemma4ForCausalLM,
    batch: Mapping[str, Array],
) -> Array:
    """Compute the mean cross-entropy loss for the model on a given token batch.

    Args:
        model: Trainable JAX/Flax NNX causal language model.
        batch: Batch dictionary containing `'inputs'` and `'targets'` token ID arrays.

    Returns:
        Scalar mean cross-entropy loss array.

    """
    inputs = batch["inputs"]
    seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
    positions = jnp.arange(seq_len)[None, :]
    logits: Array = call_jax_causal_lm(model, inputs, positions)  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    res: Array = jnp.mean(loss)
    return res


def _get_train_step_fn() -> JaxTrainStepFn:
    """Return a JIT-compiled Flax NNX training step function.

    Returns:
        Compiled training step callable `(model, optimizer, batch) -> loss`.

    Raises:
        DependencyMissingError: If `flax.nnx.value_and_grad` is unavailable.

    """
    if nnx is None or not isinstance(nnx, _NNXValueAndGradModuleProtocol):
        raise DependencyMissingError(
            "Flax NNX value_and_grad is required for JAX training.",
            dependency_name="flax.nnx",
            backend="jax",
        )

    def train_step(
        model: JaxCausalLMProtocol,
        optimizer: NNXOptimizerProtocol[JaxCausalLMProtocol, Array],
        batch: Mapping[str, Array],
    ) -> Array:
        """Execute a single JAX-compiled training step.

        Args:
            model: Trainable Flax NNX causal language model.
            optimizer: Flax NNX optimizer instance.
            batch: Token batch dictionary with `'inputs'` and `'targets'`.

        Returns:
            Scalar batch loss tensor.

        """
        (loss, grads) = nnx.value_and_grad(_loss_fn)(model, batch)
        _apply_nnx_optimizer_update(optimizer, model, grads)
        return loss

    return nnx.jit(train_step)


def _run_training_epochs[ModelT, OptT, BatchValT](
    state: TrainingLoopState[ModelT, OptT, Mapping[str, BatchValT]] | TrainingLoopState[ModelT, OptT, dict[str, BatchValT]],
) -> float:
    """Run supervised training epochs over the JAX dataloader.

    Args:
        state: `TrainingLoopState` holding dataloader, model, optimizer, sharding, and step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[Mapping[str, BatchValT]] = state.dataloader or ()
    train_step = state.train_step
    model = state.model
    sharding = state.sharding

    def process_batch(batch: Mapping[str, BatchValT]) -> float:
        """Place batch tensors onto target sharding and execute one training step.

        Args:
            batch: Dictionary with `inputs` and `targets` arrays.

        Returns:
            Scalar batch loss as a float.

        """
        mutable_batch = dict(batch)
        mutable_batch["inputs"] = jax.device_put(mutable_batch["inputs"], sharding)  # pyright: ignore[reportArgumentType]
        mutable_batch["targets"] = jax.device_put(mutable_batch["targets"], sharding)  # pyright: ignore[reportArgumentType]
        with contextlib.suppress(TypeError):
            batch["inputs"] = mutable_batch["inputs"]  # pyright: ignore[reportIndexIssue]  # ty: ignore[invalid-assignment]
            batch["targets"] = mutable_batch["targets"]  # pyright: ignore[reportIndexIssue]  # ty: ignore[invalid-assignment]
        assert train_step is not None
        loss = train_step(model, state.optimizer, mutable_batch)
        if isinstance(loss, ScalarItemProtocol):
            return float(loss.item())
        if isinstance(loss, (int, float, SupportsFloat)):
            return float(loss)
        return 0.0

    return generic_run_training_epochs(state.epochs, dataloader, process_batch)


def _resolve_gemma4_config(model_name: str) -> ModelConfig:
    """Resolve a `Gemma4Config` preset for the given `model_name`.

    Args:
        model_name: Model name or preset identifier.

    Returns:
        Resolved `ModelConfig` object.

    """
    return resolve_gemma4_config(Gemma4Config, model_name)


def _execute_train(
    dataset: str,
    epochs: int,
    learning_rate: float,
    batch_size: int = 2,
    model_name: str = "gemma-4",
) -> tuple[str, float]:
    """Execute the core training loop for JAX.

    Args:
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        learning_rate: Training learning rate.
        batch_size: Training batch size.
        model_name: Target model identifier used to select architectural preset.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    if jax is None or jnp is None or optax is None or Gemma4ForCausalLM is None or nnx is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX dependencies are missing for training.")

    cached_model = _MODEL_CACHE.get_model(model_name)
    model = cached_model if cached_model is not None else Gemma4ForCausalLM(_resolve_gemma4_config(model_name), rngs=nnx.Rngs(0))
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=100,
        decay_steps=max(1, epochs * 1000),
        end_value=learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule),
    )
    optimizer = _create_nnx_optimizer(model, tx)
    train_step = _get_train_step_fn()
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    has_loader = isinstance(dataloader, Iterable)

    if not has_loader:
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(
        TrainingLoopState(
            dataloader=dataloader,
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            train_step=train_step,
            sharding=sharding,
        )
    )
    _MODEL_CACHE.put_model(model_name, model)
    return "completed", final_loss


def train_model(config: TrainingConfig, **kwargs: StatusValue) -> TrainResultDict:
    """Train a Text-to-SQL model using JAX and Flax NNX.

    Args:
        config: Training configuration parameters.
        **kwargs: Extra runtime options (such as `raise_on_error=True`).

    Returns:
        Structured `TrainResultDict` containing JAX training status and final loss.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    if jax is None or jnp is None or optax is None or Gemma4ForCausalLM is None or nnx is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX dependencies are missing for training.", backend="jax")

    model_name = config.model_name
    dataset = config.dataset
    epochs = config.epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    return run_train_wrapper(
        "jax",
        config,
        lambda: _execute_train(dataset, epochs, learning_rate, batch_size=batch_size, model_name=model_name),
        **kwargs,
    )
