"""JAX-specific training pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, ModelType, TrainerState, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, TrainResultDict

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
nnx = optional_import("flax.nnx")
Gemma4Config = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4Config")
Gemma4ForCausalLM = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4ForCausalLM")


def _loss_fn(model: ModelType, batch: JSONDict) -> Any:
    """Compute the cross-entropy loss for the model on a given batch.

    Args:
        model: The model.
        batch: The batch.

    Returns:
        The execution result.

    """
    logits = model(batch["inputs"])
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)


def _get_train_step_fn() -> object:
    """Return a JIT-compiled train step function.

    Returns:
        The execution result.

    """

    def train_step(model: ModelType, optimizer: Any, batch: JSONDict) -> Any:
        """Execute a single JAX-compiled training step.

        Args:
            model: The model.
            optimizer: The optimizer.
            batch: The batch.

        Returns:
            The execution result.

        """
        if nnx is not None and hasattr(nnx, "value_and_grad"):
            (loss, grads) = nnx.value_and_grad(_loss_fn)(model, batch)
        else:
            loss, grads = 0.0, None
        if optimizer is not None and hasattr(optimizer, "update"):
            optimizer.update(grads)
        return loss

    if nnx is not None and hasattr(nnx, "jit"):
        return nnx.jit(train_step)
    return train_step


def _run_training_epochs(state: TrainerState) -> float:
    """Run training loops.

    Returns:
        float: The final loss.

    """

    def process_batch(batch: dict[str, Any]) -> float:
        """Place batch tensors onto target sharding and execute one training step.

        Args:
            batch: Dictionary with `inputs` and `targets` arrays.

        Returns:
            Scalar batch loss as a float.

        """
        batch["inputs"] = jax.device_put(batch["inputs"], state.params)
        batch["targets"] = jax.device_put(batch["targets"], state.params)
        loss = state.train_step(state.policy_model, state.optimizer, batch)
        return float(loss.item() if hasattr(loss, "item") else loss)

    return generic_run_training_epochs(state.epochs, state.dataloader, process_batch)


def _resolve_gemma4_config(model_name: str) -> Any:
    """Resolve a `Gemma4Config` preset for the given `model_name`.

    Args:
        model_name: Model name or preset identifier.

    Returns:
        Resolved `Gemma4Config` object.

    """
    return Gemma4Config.from_preset(model_name) if hasattr(Gemma4Config, "from_preset") else Gemma4Config.gemma4_e2b()


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

    model = Gemma4ForCausalLM(_resolve_gemma4_config(model_name), rngs=nnx.Rngs(0))
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=learning_rate, warmup_steps=100, decay_steps=max(1, epochs * 1000), end_value=learning_rate * 0.1)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule),
    )
    optimizer = nnx.Optimizer(model, tx)
    train_step = _get_train_step_fn()
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=model, optimizer=optimizer, train_step=train_step, params=sharding))
    return "completed", float(final_loss)


def train_model(config: TrainingConfig, **kwargs: object) -> TrainResultDict:
    """Train a Text-to-SQL model using the JAX backend.

    Args:
        config: The TrainingConfig.
        **kwargs: Extra runtime options such as 'distributed_strategy'.

    Returns:
        A dictionary containing JAX training status and metrics.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.

    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)

    final_loss = 0.0
    status = "completed"
    if jax is None or jnp is None or optax is None or Gemma4ForCausalLM is None or nnx is None:
        raise DependencyMissingError("JAX dependencies are missing for training.")

    try:
        batch_size = getattr(config, "batch_size", 2)
        status, final_loss = _execute_train(dataset, epochs, learning_rate, batch_size=batch_size, model_name=model_name)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        status = f"failed: {e!s}"

    return {"backend": "jax", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "learning_rate": learning_rate, "status": status, "final_loss": float(final_loss)}
