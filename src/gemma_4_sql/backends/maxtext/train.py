"""MaxText-specific training pipeline and distributed cluster orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
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
    import orbax.checkpoint as _ocp
    from maxtext.models.gemma4 import Gemma4Model as _Gemma4Model

    jax: Any = _jax
    jnp: Any = _jnp
    optax: Any = _optax
    maxtext_train: Any = _maxtext_train
    Gemma4Model: Any = _Gemma4Model
    ocp: Any = _ocp
except (ImportError, AttributeError):
    jax = None
    jnp = None
    optax = None
    maxtext_train = None
    Gemma4Model = None
    ocp = None


def _loss_fn(model: Any, params: Any, batch: JSONDict) -> Any:
    """Compute the cross-entropy loss over a training batch.

    Args:
        model: Model instance providing an apply method.
        params: Model parameter PyTree.
        batch: Dictionary containing 'inputs' and integer 'targets' tensors.

    Returns:
        Scalar mean cross-entropy loss tensor.
    """
    logits = model.apply(params, batch["inputs"])
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)


def _get_train_step_fn(model: Any, optimizer: Any) -> Any:
    """Construct a JIT-compiled or standard training step execution function.

    Args:
        model: Model instance providing forward execution.
        optimizer: Optax optimizer instance for parameter updates.

    Returns:
        A callable train_step(params, opt_state, batch) returning updated params, opt_state, and loss.
    """

    def train_step(params: Any, opt_state: Any, batch: JSONDict) -> Any:
        """Perform a single forward-backward pass and update optimizer state.

        Args:
            params: Current model parameter PyTree.
            opt_state: Current optimizer state PyTree.
            batch: Dictionary containing inputs and target tensors.

        Returns:
            Tuple of (updated_params, updated_opt_state, scalar_loss).
        """
        (loss, grads) = jax.value_and_grad(lambda p, b: _loss_fn(model, p, b))(params, batch)
        (updates, opt_state) = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, loss)

    if jax is not None and hasattr(jax, "jit"):
        return jax.jit(train_step)
    return train_step


def _run_training_epochs(state: TrainerState) -> tuple[TensorType, TensorType, float]:
    """Execute epochs over dataloader using TrainerState.

    Args:
        state: TrainerState containing dataloader, epochs, step function, and initial states.

    Returns:
        Tuple of (final_params, final_opt_state, final_loss).
    """
    params = state.params
    opt_state = state.opt_state

    def process_batch(batch: dict[str, Any]) -> float:
        """Process a single batch during epoch iteration.

        Args:
            batch: Dictionary containing model inputs and labels.

        Returns:
            Computed scalar loss value as a float.
        """
        nonlocal params, opt_state
        (params, opt_state, loss) = state.train_step(params, opt_state, batch)
        return float(loss.item() if hasattr(loss, "item") else loss)

    final_loss = generic_run_training_epochs(state.epochs, state.dataloader, process_batch)
    return (params, opt_state, final_loss)


def _initialize_jax_distributed(
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    test_mode: bool = False,
) -> bool:
    """Initialize JAX distributed multi-host coordination service.

    Args:
        coordinator_address: IP/hostname and port of the primary coordinator host.
        num_processes: Total number of participating JAX processes/hosts.
        process_id: Rank/ID of the current host process.
        test_mode: Whether to bypass distributed initialization during testing.

    Returns:
        True if initialization was performed, False if skipped.
    """
    if test_mode:
        return False
    if jax is not None and hasattr(jax, "distributed") and hasattr(jax.distributed, "initialize"):
        try:
            init_kwargs: dict[str, Any] = {}
            if coordinator_address is not None:
                init_kwargs["coordinator_address"] = coordinator_address
            if num_processes is not None:
                init_kwargs["num_processes"] = num_processes
            if process_id is not None:
                init_kwargs["process_id"] = process_id
            jax.distributed.initialize(**init_kwargs)
            return True
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as init_err:
            logger.warning("jax.distributed.initialize() failed or already initialized: %s", init_err)
    return False


def save_maxtext_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    params: Any,
    opt_state: Any | None = None,
) -> Path:
    """Save model parameters and optimizer state using Orbax CheckpointManager.

    Args:
        checkpoint_dir: Target directory for storing Orbax checkpoints.
        step: Current training step integer.
        params: Model parameters PyTree.
        opt_state: Optional optimizer state PyTree.

    Returns:
        Path pointing to the saved checkpoint directory.

    Raises:
        DependencyMissingError: If Orbax checkpoint dependency is missing.
        ExportError: If checkpoint persistence fails.
    """
    if ocp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Orbax checkpoint dependency (orbax.checkpoint) is missing.")

    from gemma_4_sql.exceptions import ExportError

    ckpt_path = Path(checkpoint_dir).resolve()
    try:
        ckpt_path.mkdir(parents=True, exist_ok=True)
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        state_to_save: dict[str, Any] = {"params": params}
        if opt_state is not None:
            state_to_save["opt_state"] = opt_state
        with ocp.CheckpointManager(ckpt_path, ocp.PyTreeCheckpointer(), options) as mngr:
            mngr.save(step, state_to_save)
        return ckpt_path
    except Exception as e:
        raise ExportError(f"Failed to persist Orbax checkpoint at '{ckpt_path}': {e}") from e


def _execute_train(
    model_name_or_config: str | TrainingConfig,
    dataset: str = "dummy",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    test_mode: bool = False,
    batch_size: int = 2,
    local_step_mode: bool = False,
    **kwargs: object,
) -> tuple[str, float]:
    """Execute the MaxText training pipeline, dispatching to cluster training or local step loop.

    Args:
        model_name_or_config: Target model identifier or complete TrainingConfig.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        learning_rate: Training learning rate.
        test_mode: Whether to run in test mode.
        batch_size: Training batch size.
        local_step_mode: Whether to force execution of the lightweight local Flax/Optax step loop.
        **kwargs: Additional parameters for MaxText cluster orchestration and checkpointing.

    Returns:
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If required MaxText dependencies are missing.
        ValueError: If dataloader is invalid or unavailable.
    """
    if isinstance(model_name_or_config, TrainingConfig):
        cfg = model_name_or_config
        model_name = cfg.model_name
        dataset = cfg.dataset
        epochs = cfg.epochs
        learning_rate = cfg.learning_rate
        batch_size = cfg.batch_size
        extra = dict(getattr(cfg, "extra_kwargs", {}) or {})
    else:
        model_name = str(model_name_or_config)
        extra = {}
        cfg = TrainingConfig(
            model_name=model_name,
            dataset=dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

    merged_kwargs: dict[str, object] = {**extra, **kwargs}
    test_mode = bool(merged_kwargs.get("test_mode", test_mode))
    local_step_mode = bool(merged_kwargs.get("local_step_mode", local_step_mode))

    if jax is None or jnp is None or optax is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing for training.")

    coord_addr = merged_kwargs.get("coordinator_address")
    num_procs = merged_kwargs.get("num_processes")
    proc_id = merged_kwargs.get("process_id")
    _initialize_jax_distributed(
        coordinator_address=str(coord_addr) if coord_addr is not None else None,
        num_processes=int(num_procs) if isinstance(num_procs, (int, str)) else None,
        process_id=int(proc_id) if isinstance(proc_id, (int, str)) else None,
        test_mode=test_mode,
    )

    etl_kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
    data_dict = build_dataloader(
        ETLConfig(
            dataset_name=dataset,
            split="train",
            batch_size=batch_size,
            distributed=(not test_mode and not local_step_mode),
        ),
        **etl_kwargs,
    )
    dataloader = data_dict.get("loader", None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
        msg = f"Invalid dataloader for dataset: {dataset}"
        raise ValueError(msg)

    if not test_mode and not local_step_mode and maxtext_train is not None:
        from gemma_4_sql.backends.maxtext.config_generator import (
            build_maxtext_cli_args,
            generate_maxtext_gin_config,
            save_maxtext_gin_config,
        )

        logger.info("Connecting to MaxText distributed training loop...")
        gin_str = generate_maxtext_gin_config(cfg, **merged_kwargs)
        explicit_gin_path = merged_kwargs.get("gin_config_path")
        gin_file_path = save_maxtext_gin_config(
            gin_str,
            output_path=Path(str(explicit_gin_path)) if explicit_gin_path else None,
        )
        cli_args = build_maxtext_cli_args(cfg, gin_config_path=gin_file_path, **merged_kwargs)
        logger.info("Invoking maxtext.train.main with CLI arguments: %s", cli_args)
        maxtext_train.main(cli_args)

        ckpt_dir = merged_kwargs.get("checkpoint_dir")
        if ckpt_dir and ocp is not None:
            save_maxtext_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                step=epochs,
                params={"status": "trained_distributed"},
                opt_state=None,
            )
        return "completed", 0.0

    if Gemma4Model is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing for training.")

    model = Gemma4Model(model_name)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
    params: Any = model.init(rng, dummy_input)
    optimizer = optax.adamw(learning_rate)
    opt_state: Any = optimizer.init(params)
    train_step = _get_train_step_fn(model, optimizer)

    final_state: tuple[Any, Any, float] = _run_training_epochs(
        TrainerState(
            dataloader=dataloader,
            epochs=epochs,
            train_step=train_step,
            params=params,
            opt_state=opt_state,
        )
    )
    final_params, final_opt_state, final_loss = final_state

    ckpt_dir = merged_kwargs.get("checkpoint_dir")
    if ckpt_dir and ocp is not None:
        save_maxtext_checkpoint(
            checkpoint_dir=str(ckpt_dir),
            step=epochs,
            params=final_params,
            opt_state=final_opt_state,
        )

    return "completed", float(final_loss)


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Train a Text-to-SQL model using the MaxText backend.

    Args:
        config: The TrainingConfig specifying hyperparameters and settings.
        **kwargs: Extra parameters passed to the training pipeline.

    Returns:
        A dictionary containing MaxText training status and metrics.

    Raises:
        DependencyMissingError: If required MaxText dependencies are missing.
    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)

    final_loss = 0.0
    status = "completed"
    if jax is None or jnp is None or optax is None or (Gemma4Model is None and maxtext_train is None):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing.")

    try:
        test_mode = bool(kwargs.get("test_mode") or getattr(config, "extra_kwargs", {}).get("test_mode", False))
        local_step_mode = bool(kwargs.get("local_step_mode") or getattr(config, "extra_kwargs", {}).get("local_step_mode", False))
        status, final_loss = _execute_train(
            config,
            dataset=dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            test_mode=test_mode,
            batch_size=getattr(config, "batch_size", 2),
            local_step_mode=local_step_mode,
            **kwargs,
        )
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        logger.exception("MaxText Train error: ")
        status = f"failed: {e!s}"

    result: JSONDict = {
        "backend": "maxtext",
        "action": action,
        "model": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "status": status,
        "final_loss": float(final_loss),
    }
    ckpt_dir = kwargs.get("checkpoint_dir") or getattr(config, "extra_kwargs", {}).get("checkpoint_dir")
    if ckpt_dir:
        result["checkpoint_dir"] = str(ckpt_dir)
    return result
