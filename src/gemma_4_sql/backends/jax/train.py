"""JAX-specific training pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gemma_4_sql.backends.common_inference import call_jax_causal_lm
from gemma_4_sql.backends.common_train import (
    apply_nnx_optimizer_update as _apply_nnx_optimizer_update,
)
from gemma_4_sql.backends.common_train import (
    create_nnx_optimizer,
    generic_run_training_epochs,
    run_train_wrapper,
)
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.backends.jax.gemma4.config import resolve_gemma4_config
from gemma_4_sql.backends.jax.inference import _MODEL_CACHE
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, JaxCausalLMProtocol, JSONDict, TrainerState, TrainingConfig, TrainingLoopState, TrainResultDict

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
nnx = optional_import("flax.nnx")
Gemma4Config = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4Config")
Gemma4ForCausalLM = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4ForCausalLM")


def _create_nnx_optimizer(model: Any, tx: Any) -> Any:
    """Create a Flax NNX Optimizer compatible with both Flax <0.11 and >=0.11 (`wrt=nnx.Param`).

    Args:
        model: Flax NNX module to optimize.
        tx: Optax gradient transformation.

    Returns:
        Configured `nnx.Optimizer` instance.

    """
    return create_nnx_optimizer(nnx, model, tx)


def _loss_fn(model: JaxCausalLMProtocol, batch: JSONDict) -> Any:
    """Compute the mean cross-entropy loss for the model on a given token batch.

    Args:
        model: Trainable JAX/Flax NNX causal language model.
        batch: Batch dictionary containing `'inputs'` and `'targets'` token ID arrays.

    Returns:
        Scalar mean cross-entropy loss array.

    """
    inputs: Any = batch["inputs"]
    seq_len = inputs.shape[1] if hasattr(inputs, "shape") and len(inputs.shape) > 1 else 1
    positions = jnp.arange(seq_len)[None, :] if jnp is not None and hasattr(jnp, "arange") else None
    logits = call_jax_causal_lm(model, inputs, positions)
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss) if jnp is not None else loss


def _get_train_step_fn() -> Callable[..., Any]:
    """Return a JIT-compiled Flax NNX training step function.

    Returns:
        Compiled or eager training step callable `(model, optimizer, batch) -> loss`.

    Raises:
        DependencyMissingError: If `flax.nnx.value_and_grad` is unavailable.

    """
    if nnx is None or not hasattr(nnx, "value_and_grad"):
        raise DependencyMissingError(
            "Flax NNX value_and_grad is required for JAX training.",
            dependency_name="flax.nnx",
            backend="jax",
        )

    def train_step(model: JaxCausalLMProtocol, optimizer: Any, batch: JSONDict) -> Any:
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

    if hasattr(nnx, "jit"):
        jit_fn: Callable[..., Any] = nnx.jit(train_step)
        return jit_fn
    return train_step


def _run_training_epochs(state: TrainingLoopState[Any, Any, Any] | TrainerState[Any, Any, Any, Any]) -> float:
    """Run supervised training epochs over the JAX dataloader.

    Args:
        state: `TrainingLoopState` holding dataloader, model, optimizer, sharding, and step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Any = state.dataloader or ()
    train_step = state.train_step
    model = state.model if isinstance(state, TrainingLoopState) else state.policy_model
    sharding = state.sharding if isinstance(state, TrainingLoopState) else state.params

    def process_batch(batch: dict[str, Any]) -> float:
        """Place batch tensors onto target sharding and execute one training step.

        Args:
            batch: Dictionary with `inputs` and `targets` arrays.

        Returns:
            Scalar batch loss as a float.

        """
        batch["inputs"] = jax.device_put(batch["inputs"], sharding)
        batch["targets"] = jax.device_put(batch["targets"], sharding)
        assert train_step is not None
        loss = train_step(model, state.optimizer, batch)
        return float(loss.item() if hasattr(loss, "item") else loss)

    return generic_run_training_epochs(state.epochs, dataloader, process_batch)


def _resolve_gemma4_config(model_name: str) -> Any:
    """Resolve a `Gemma4Config` preset for the given `model_name`.

    Args:
        model_name: Model name or preset identifier.

    Returns:
        Resolved `Gemma4Config` object.

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
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    if jax is None or jnp is None or optax is None or Gemma4ForCausalLM is None or nnx is None:
        raise DependencyMissingError("JAX dependencies are missing for training.")

    cached_model = _MODEL_CACHE.get_model(model_name)
    model = cached_model if cached_model is not None else Gemma4ForCausalLM(_resolve_gemma4_config(model_name), rngs=nnx.Rngs(0))
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=learning_rate, warmup_steps=100, decay_steps=max(1, epochs * 1000), end_value=learning_rate * 0.1)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule),
    )
    optimizer = _create_nnx_optimizer(model, tx)
    train_step = _get_train_step_fn()
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
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


def train_model(config: TrainingConfig, **kwargs: object) -> TrainResultDict:
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
    if jax is None or jnp is None or optax is None or Gemma4ForCausalLM is None or nnx is None:
        raise DependencyMissingError("JAX dependencies are missing for training.", backend="jax")

    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dataset")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-4)
    batch_size = getattr(config, "batch_size", 2)
    return run_train_wrapper(
        "jax",
        config,
        lambda: _execute_train(dataset, epochs, learning_rate, batch_size=batch_size, model_name=model_name),
        **kwargs,
    )
