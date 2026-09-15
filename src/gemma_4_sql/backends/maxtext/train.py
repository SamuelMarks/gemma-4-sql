"""MaxText-specific training pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.type_hints import ETLConfig, TensorType, TrainerState, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import jax as _jax
    import jax.numpy as _jnp
    import maxtext.train as _maxtext_train
    import optax as _optax
    from maxtext.models.gemma4 import Gemma4Model as _Gemma4Model

    jax: Any = _jax
    jnp: Any = _jnp
    optax: Any = _optax
    maxtext_train: Any = _maxtext_train
    Gemma4Model: Any = _Gemma4Model
except (ImportError, AttributeError):
    jax = None
    jnp = None
    optax = None
    maxtext_train = None
    Gemma4Model = None


def _loss_fn(model: Any, params: Any, batch: JSONDict) -> Any:
    """Execute logic.

    Args:
        model: The model.
        params: A mapping representing params.
        batch: The batch.

    Returns:
        The execution result.
    """
    logits = model.apply(params, batch["inputs"])
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)


def _get_train_step_fn(model: Any, optimizer: Any) -> Any:
    """Execute the get train step fn operation.

    Args:
        model: The model.
        optimizer: The optimizer.

    Returns:
        The execution result.
    """

    def train_step(params: Any, opt_state: Any, batch: JSONDict) -> Any:
        """Execute logic.

        Returns:
            object: The resulting output from the operation.

        """
        (loss, grads) = jax.value_and_grad(lambda p, b: _loss_fn(model, p, b))(params, batch)
        (updates, opt_state) = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, loss)

    if jax is not None and hasattr(jax, "jit"):
        return jax.jit(train_step)
    return train_step


def _run_training_epochs(state: TrainerState) -> tuple[TensorType, TensorType, float]:
    """Run training epochs.

    Returns:
        tuple: (params, opt_state, final_loss)

    """
    params = state.params
    opt_state = state.opt_state

    def process_batch(batch: dict[str, Any]) -> float:
        """Execute function.

        Returns:
            The execution result.

        """
        nonlocal params, opt_state
        (params, opt_state, loss) = state.train_step(params, opt_state, batch)
        return float(loss.item() if hasattr(loss, "item") else loss)

    final_loss = generic_run_training_epochs(state.epochs, state.dataloader, process_batch)
    return (params, opt_state, final_loss)


def _initialize_jax_distributed(*, test_mode: bool = False) -> None:
    """Initialize JAX distributed if not in test mode."""
    if not test_mode and jax is not None and hasattr(jax, "distributed"):  # pragma: no cover
        try:
            jax.distributed.initialize()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as init_err:
            logger.warning("jax.distributed.initialize() failed or already initialized: %s", init_err)


def _execute_train(model_name: str, dataset: str, epochs: int, learning_rate: float, test_mode: bool, batch_size: int = 2) -> tuple[str, float]:
    """Execute the core MaxText training loop.

    Args:
        model_name: Target model name.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        learning_rate: Training learning rate.
        test_mode: Whether to run in test mode.
        batch_size: Training batch size.

    Returns:
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.
        ValueError: If dataloader is invalid.
    """
    if jax is None or jnp is None or optax is None or Gemma4Model is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing for training.")

    _initialize_jax_distributed(test_mode=test_mode)
    if maxtext_train is not None and (not test_mode):
        logger.info("Connecting to MaxText training loop...")
    model = Gemma4Model(model_name)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
    params: Any = model.init(rng, dummy_input)
    optimizer = optax.adamw(learning_rate)
    opt_state: Any = optimizer.init(params)
    train_step = _get_train_step_fn(model, optimizer)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_state: tuple[Any, Any, float] = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, train_step=train_step, params=params, opt_state=opt_state))
    final_loss = final_state[2]
    return "completed", float(final_loss)


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Train a Text-to-SQL model using the MaxText backend.

    Args:
        config: The TrainingConfig.
        **kwargs: Extra parameters.

    Returns:
        A dictionary containing MaxText training status and metrics.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.
    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)

    final_loss = 0.42
    status = "completed"
    if jax is None or jnp is None or optax is None or Gemma4Model is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing.")
    try:
        batch_size = getattr(config, "batch_size", 2)
        status, final_loss = _execute_train(model_name, dataset, epochs, learning_rate, bool(kwargs.get("test_mode")), batch_size=batch_size)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        logger.exception("MaxText Train error: ")
        status = f"failed: {e!s}"
    return {"backend": "maxtext", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "learning_rate": learning_rate, "status": status, "final_loss": float(final_loss)}
