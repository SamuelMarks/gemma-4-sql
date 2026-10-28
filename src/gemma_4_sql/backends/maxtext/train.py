"""MaxText-specific training pipeline and distributed cluster orchestrator."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_train import generic_run_training_epochs, run_train_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, StatusValue
from gemma_4_sql.type_hints import (
    ETLConfig,
    JaxArrayProtocol,
    JSONValue,
    MaxTextCausalLMProtocol,
    MaxTextTrainerState,
    TensorProtocol,
    TensorType,
    TrainerState,
    TrainingConfig,
    TrainResultDict,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _StatelessApplyModelProtocol[ValT](Protocol):
    """Protocol for a stateless Flax/MaxText model exposing `.apply(params, inputs)`."""

    def apply(
        self,
        params: Mapping[str, ValT],
        inputs: ValT,
        /,
    ) -> ValT:
        """Apply model parameters to `inputs`."""
        ...


type MaxTextTrainModel[ValT] = _StatelessApplyModelProtocol[ValT] | MaxTextCausalLMProtocol


@runtime_checkable
class _OptaxOptimizerProtocol[OptInputT, OptStateT](Protocol):
    """Protocol for an Optax `GradientTransformation` supporting `.init(params)` and `.update(grads, opt_state, params)`."""

    def init(self, params: OptInputT, /) -> OptStateT:
        """Initialize optimizer state from parameters."""
        ...

    def update(
        self,
        updates: OptInputT,
        state: OptStateT,
        params: OptInputT | None = None,
        /,
    ) -> tuple[OptInputT, OptStateT]:
        """Compute parameter updates and next optimizer state."""
        ...


type MaxTextStepCallable[ParamsT, OptStateT, BatchT] = Callable[
    [ParamsT, OptStateT, BatchT],
    tuple[ParamsT, OptStateT, SupportsFloat | float],
]


jax = optional_import("jax")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
maxtext_train = optional_import("maxtext.train")
_raw_gemma4 = optional_attr("maxtext.models.gemma4", "Gemma4Model")
Gemma4Model: Callable[..., MaxTextCausalLMProtocol] | None = _raw_gemma4 if callable(_raw_gemma4) else None  # ty: ignore[invalid-assignment]
ocp = optional_import("orbax.checkpoint")


def _loss_fn[ValT: TensorType | SupportsFloat](
    model: MaxTextTrainModel[ValT],
    params: Mapping[str, ValT],
    batch: Mapping[str, ValT],
) -> JaxArrayProtocol:
    """Compute the cross-entropy loss over a training batch.

    Args:
        model: Model instance providing an `apply` method.
        params: Model parameter PyTree.
        batch: Dictionary containing `'inputs'` and integer `'targets'` tensors.

    Returns:
        Scalar mean cross-entropy loss tensor.

    """
    assert optax is not None
    assert jnp is not None
    logits = model.apply(params, batch["inputs"])  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[invalid-argument-type]
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    res: JaxArrayProtocol = jnp.mean(loss)
    return res


def _get_train_step_fn[ValT: TensorType | SupportsFloat, ParamsT, OptInputT, OptStateT, BatchT](
    model: MaxTextTrainModel[ValT],
    optimizer: _OptaxOptimizerProtocol[OptInputT, OptStateT],
) -> MaxTextStepCallable[ParamsT, OptStateT, BatchT]:
    """Construct a JIT-compiled or standard training step execution function.

    Args:
        model: Model instance providing forward execution.
        optimizer: Optax optimizer instance for parameter updates.

    Returns:
        A callable `train_step(params, opt_state, batch)` returning `(updated_params, updated_opt_state, loss)`.

    """

    def train_step(
        params: ParamsT,
        opt_state: OptStateT,
        batch: BatchT,
    ) -> tuple[ParamsT, OptStateT, SupportsFloat | float]:
        """Perform a single forward-backward pass and update optimizer state.

        Args:
            params: Current model parameter PyTree.
            opt_state: Current optimizer state PyTree.
            batch: Dictionary containing inputs and target tensors.

        Returns:
            Tuple of `(updated_params, updated_opt_state, scalar_loss)`.

        """

        def _wrapper(p: ParamsT, b: BatchT) -> JaxArrayProtocol:
            """Evaluate `_loss_fn` for JAX `value_and_grad` differentiation."""
            return _loss_fn(model, p, b)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

        assert jax is not None
        assert optax is not None
        (loss, grads) = jax.value_and_grad(_wrapper)(params, batch)
        (updates, opt_state) = optimizer.update(grads, opt_state, params)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        params = optax.apply_updates(params, updates)  # pyright: ignore[reportAssignmentType]
        return (params, opt_state, loss)  # pyright: ignore[reportReturnType]

    with contextlib.suppress(AttributeError, TypeError):
        compiled: MaxTextStepCallable[ParamsT, OptStateT, BatchT] = jax.jit(train_step)  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
        return compiled
    return train_step


def _run_training_epochs[ModelT, ParamsT, OptT, BatchT](
    state: MaxTextTrainerState[ParamsT, OptT, BatchT] | TrainerState[ModelT, ParamsT, OptT, BatchT],
) -> tuple[ParamsT | None, OptT | None, float]:
    """Execute epochs over dataloader using `MaxTextTrainerState`.

    Args:
        state: `MaxTextTrainerState` containing dataloader, epochs, step function, and initial states.

    Returns:
        Tuple of `(final_params, final_opt_state, final_loss)`.

    """
    params = state.params
    opt_state = state.opt_state
    train_step = state.train_step
    dataloader: Iterable[BatchT] = state.dataloader or ()

    def process_batch(batch: BatchT) -> float:
        """Process a single batch during epoch iteration.

        Args:
            batch: Dictionary containing model inputs and labels.

        Returns:
            Computed scalar loss value as a float.

        """
        nonlocal params, opt_state
        assert train_step is not None
        step_out: tuple[ParamsT | None, OptT | None, SupportsFloat | float] = train_step(  # ty: ignore[invalid-assignment]
            params,
            opt_state,
            batch,
        )  # pyright: ignore[reportAssignmentType]
        params, opt_state, loss = step_out
        return float(loss)

    final_loss = generic_run_training_epochs(state.epochs, dataloader, process_batch)
    return (params, opt_state, final_loss)


def _initialize_jax_distributed(
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
) -> bool:
    """Initialize JAX distributed multi-host coordination service.

    Args:
        coordinator_address: IP/hostname and port of the primary coordinator host.
        num_processes: Total number of participating JAX processes/hosts.
        process_id: Rank/ID of the current host process.

    Returns:
        `True` if initialization was performed, `False` if skipped.

    """
    if jax is not None:
        try:
            init_kwargs: dict[str, str | int] = {}
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


def save_maxtext_checkpoint[ParamValT, OptValT](
    checkpoint_dir: str | Path,
    step: int,
    params: Mapping[str, ParamValT],
    opt_state: Mapping[str, OptValT] | None = None,
) -> Path:
    """Save model parameters and optimizer state using Orbax `CheckpointManager`.

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
        raise DependencyMissingError("Orbax checkpoint dependency (orbax.checkpoint) is missing.")

    ckpt_path = Path(checkpoint_dir).resolve()
    try:
        ckpt_path.mkdir(parents=True, exist_ok=True)
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        state_to_save: dict[str, Mapping[str, ParamValT] | Mapping[str, OptValT]] = {"params": params}
        if opt_state is not None:
            state_to_save["opt_state"] = opt_state
        with ocp.CheckpointManager(ckpt_path, ocp.PyTreeCheckpointer(), options) as mngr:
            mngr.save(step, state_to_save)
        return ckpt_path
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        raise ExportError(f"Failed to persist Orbax checkpoint at '{ckpt_path}': {e}", backend="maxtext", cause=e) from e


def _execute_train(
    model_name_or_config: str | TrainingConfig,
    dataset: str = "dummy",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    local_step_mode: bool = False,
    **kwargs: JSONValue,
) -> tuple[str, float]:
    """Execute the MaxText training pipeline, dispatching to cluster training or local step loop.

    Args:
        model_name_or_config: Target model identifier or complete `TrainingConfig`.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        learning_rate: Training learning rate.
        batch_size: Training batch size.
        local_step_mode: Whether to force execution of the lightweight local Flax/Optax step loop.
        **kwargs: Additional parameters for MaxText cluster orchestration and checkpointing.

    Returns:
        A tuple of `(status, final_loss)`.

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
        extra: dict[str, JSONValue] = {k: v for k, v in (cfg.extra_kwargs or {}).items() if isinstance(v, (str, int, float, bool)) or v is None}
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

    merged_kwargs: dict[str, JSONValue] = {**extra, **kwargs}
    local_step_mode = bool(merged_kwargs.get("local_step_mode", local_step_mode))

    if jax is None or jnp is None or optax is None:
        raise DependencyMissingError("MaxText dependencies are missing for training.")

    coord_addr = merged_kwargs.get("coordinator_address")
    num_procs = merged_kwargs.get("num_processes")
    proc_id = merged_kwargs.get("process_id")
    _initialize_jax_distributed(
        coordinator_address=str(coord_addr) if coord_addr is not None else None,
        num_processes=int(num_procs) if isinstance(num_procs, (int, str)) else None,
        process_id=int(proc_id) if isinstance(proc_id, (int, str)) else None,
    )

    etl_kwargs: dict[str, JSONValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
    data_dict = build_dataloader(
        ETLConfig(
            dataset_name=dataset,
            split="train",
            batch_size=batch_size,
            distributed=(not local_step_mode),
        ),
        **etl_kwargs,
    )
    dataloader = data_dict.get("loader", None)
    has_loader = isinstance(dataloader, Iterable)
    if not has_loader:
        msg = f"Invalid dataloader for dataset: {dataset}"
        raise ValueError(msg)

    if not local_step_mode and maxtext_train is not None:
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
        dist_res = maxtext_train.main(cli_args)
        dist_loss = float(dist_res) if isinstance(dist_res, (int, float)) else 0.0
        return "completed", dist_loss

    if Gemma4Model is None:
        raise DependencyMissingError("MaxText dependencies are missing for training.")

    model = Gemma4Model(model_name)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    optimizer: _OptaxOptimizerProtocol[
        Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]],
        Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]],
    ] = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)
    train_step = _get_train_step_fn(model, optimizer)

    final_params, final_opt_state, final_loss = _run_training_epochs(
        MaxTextTrainerState(
            dataloader=dataloader,
            epochs=epochs,
            train_step=train_step,
            params=params,
            opt_state=opt_state,
        )
    )

    ckpt_dir = merged_kwargs.get("checkpoint_dir")
    if ckpt_dir and ocp is not None and isinstance(final_params, Mapping):
        save_maxtext_checkpoint(
            checkpoint_dir=str(ckpt_dir),
            step=epochs,
            params=final_params,  # pyright: ignore[reportArgumentType]
            opt_state=final_opt_state if isinstance(final_opt_state, Mapping) else None,  # pyright: ignore[reportArgumentType]
        )

    return "completed", float(final_loss)


def train_model(config: TrainingConfig, **kwargs: JSONValue) -> TrainResultDict:
    """Train a Text-to-SQL model using the MaxText backend.

    Args:
        config: The `TrainingConfig` specifying hyperparameters and settings.
        **kwargs: Extra parameters passed to the training pipeline (including `raise_on_error`).

    Returns:
        A dictionary containing MaxText training status and metrics.

    Raises:
        DependencyMissingError: If required MaxText dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    if jax is None or jnp is None or optax is None or (Gemma4Model is None and maxtext_train is None):
        raise DependencyMissingError("MaxText dependencies are missing.", backend="maxtext")

    dataset = config.dataset
    epochs = config.epochs
    learning_rate = config.learning_rate
    extra = config.extra_kwargs or {}
    local_step_mode = bool(kwargs.get("local_step_mode") or extra.get("local_step_mode", False))
    call_kwargs = dict(kwargs)
    call_kwargs.pop("raise_on_error", None)
    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}

    return run_train_wrapper(
        "maxtext",
        config,
        lambda: _execute_train(
            config,
            dataset=dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=config.batch_size,
            local_step_mode=local_step_mode,
            **call_kwargs,
        ),
        **wrapper_kwargs,
    )
