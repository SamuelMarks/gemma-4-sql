"""Common training epoch iteration utilities across backends."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from gemma_4_sql.exceptions import (
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    GemmaSQLError,
    TrainingError,
    resolve_raise_on_error,
)
from gemma_4_sql.type_hints import TrainingConfig, TrainResultDict

T_Batch = TypeVar("T_Batch")
logger = logging.getLogger(__name__)


def create_nnx_optimizer(nnx_module: Any, model: Any, tx: Any) -> Any:
    """Create a Flax NNX Optimizer compatible with both Flax <0.11 and >=0.11 (`wrt=nnx.Param`).

    Args:
        nnx_module: Loaded `flax.nnx` module.
        model: Flax NNX module to optimize.
        tx: Optax gradient transformation.

    Returns:
        Configured `nnx.Optimizer` instance.

    """
    try:
        return nnx_module.Optimizer(model, tx, wrt=nnx_module.Param)
    except TypeError:
        return nnx_module.Optimizer(model, tx)


def apply_nnx_optimizer_update(optimizer: Any, model: Any, grads: Any) -> None:
    """Apply gradient updates on an `nnx.Optimizer` across Flax NNX versions.

    Args:
        optimizer: Flax NNX optimizer instance.
        model: Flax NNX module being optimized.
        grads: Computed gradient PyTree.

    """
    try:
        optimizer.update(model, grads)
    except TypeError:
        optimizer.update(grads)


def run_train_wrapper(
    backend_name: str,
    config: TrainingConfig,
    execute_fn: Callable[[], tuple[str, float]],
    **kwargs: object,
) -> TrainResultDict:
    """Execute a backend training loop with standardized config unpacking, error handling, and result formatting.

    Args:
        backend_name: Target backend identifier ('jax', 'keras', 'maxtext', 'mlx', 'pytorch').
        config: `TrainingConfig` dataclass instance.
        execute_fn: Zero-argument callable executing the backend training pass and returning `(status, final_loss)`.
        **kwargs: Runtime keyword overrides (including `raise_on_error`, `distributed_strategy`, `checkpoint_dir`).

    Returns:
        Structured `TrainResultDict` containing training status and final loss.

    Raises:
        TrainingError: If `execute_fn` raises an exception and `raise_on_error` is True.

    """
    action = str(getattr(config, "action", "sft") or "sft")
    model_name = str(getattr(config, "model_name", "gemma-4"))
    dataset = str(getattr(config, "dataset", "dummy"))
    epochs = int(getattr(config, "epochs", 1))
    learning_rate = float(getattr(config, "learning_rate", 1e-4))
    raise_on_error = resolve_raise_on_error(config, kwargs=dict(kwargs))

    final_loss = 0.0
    error_message: str | None = None
    try:
        status, final_loss = execute_fn()
        status_code = ExecutionStatus.FAILED.value if status.startswith("failed") else ExecutionStatus.COMPLETED.value
    except (DependencyMissingError, ConfigurationError):
        raise
    except (GemmaSQLError, RuntimeError, ValueError, TypeError, OSError, LookupError, ArithmeticError, AttributeError, ImportError) as e:
        logger.exception("%s training failed: ", backend_name)
        if raise_on_error:
            raise TrainingError(
                f"{backend_name} training failed: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation=action,
                cause=e,
            ) from e
        error_message = str(e)
        status = f"failed: {error_message}"
        status_code = ExecutionStatus.FAILED.value

    result: TrainResultDict = {
        "backend": backend_name,
        "action": action,
        "model": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "status": status,
        "status_code": status_code,
        "final_loss": float(final_loss),
    }
    if error_message is not None:
        result["error_message"] = error_message
    dist_strat = kwargs.get("distributed_strategy")
    if dist_strat is not None:
        result["distributed_strategy"] = str(dist_strat)
    extra = getattr(config, "extra_kwargs", {}) or {}
    ckpt_dir = kwargs.get("checkpoint_dir") or extra.get("checkpoint_dir")
    if ckpt_dir:
        result["checkpoint_dir"] = str(ckpt_dir)
    return result


def generic_run_training_epochs(
    epochs: int,
    dataloader: Iterable[T_Batch],
    process_batch_fn: Callable[[T_Batch], float],
) -> float:
    """Run a generic supervised training loop across epochs and batches.

    Args:
        epochs: Number of full passes over the dataset.
        dataloader: Iterable yielding training batches.
        process_batch_fn: Callback executing one optimization step and returning scalar loss.

    Returns:
        Mean batch loss from the final training epoch.

    """
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            loss = process_batch_fn(batch)
            epoch_loss += loss
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
    return final_loss
