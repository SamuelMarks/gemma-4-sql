"""MaxText-specific training pipeline and distributed cluster orchestrator."""

from __future__ import annotations

import contextlib
import logging
import types
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, SupportsFloat, TypeGuard, runtime_checkable

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol, StepResult
from gemma_4_sql.backends.common_train import (
    generic_run_training_epochs,
    resolve_training_dataloader,
    run_train_wrapper,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.type_hints import (
    ETLConfig,
    JaxArrayProtocol,
    JSONValue,
    MaxTextCausalLMProtocol,
    MaxTextTrainerState,
    PyTreeNode,
    TensorType,
    TrainerState,
    TrainingConfig,
    TrainResultDict,
    extract_status_kwargs,
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


type MaxTextTrainTarget = Mapping[str, PyTreeNode] | Mapping[str, JaxArrayProtocol] | Mapping[str, TensorType] | PyTreeNode | JaxArrayProtocol | TensorType | SupportsFloat

type MaxTextStepCallable[ParamsT, OptStateT, BatchT] = Callable[
    [ParamsT, OptStateT, BatchT],
    tuple[ParamsT, OptStateT, SupportsFloat | float],
]


def _is_stateless_train_model[ValT](
    model: MaxTextTrainModel[ValT],
    _sample: ValT,
) -> TypeGuard[_StatelessApplyModelProtocol[ValT]]:
    """Narrow `model` to `_StatelessApplyModelProtocol[ValT]`.

    Args:
        model: Target stateless model instance.
        _sample: Sample value for type parameter binding.

    Returns:
        `True` if `model` is not a plain `str`.

    """
    _ = _sample
    return not isinstance(model, str)


def _is_train_val_mapping[ValT](
    obj: MaxTextTrainTarget,
    _model: MaxTextTrainModel[ValT],
) -> TypeGuard[Mapping[str, ValT]]:
    """Narrow `obj` to `Mapping[str, ValT]`.

    Args:
        obj: Parameter dictionary or batch mapping.
        _model: Model carrying type parameter `ValT`.

    Returns:
        `True` if `obj` is a `Mapping`.

    """
    _ = _model
    return isinstance(obj, Mapping)


def _is_train_opt_input[OptInputT, OptStateT](
    obj: MaxTextTrainTarget | None,
    _opt: _OptaxOptimizerProtocol[OptInputT, OptStateT],
) -> TypeGuard[OptInputT]:
    """Narrow `obj` to `OptInputT` for `optimizer.update(...)`.

    Args:
        obj: Gradient or parameter PyTree.
        _opt: Optimizer carrying type parameter `OptInputT`.

    Returns:
        `True` if `obj` is not `None`.

    """
    _ = _opt
    return obj is not None


def _is_train_params_t[ParamsT](
    obj: MaxTextTrainTarget | None,
    _sample: ParamsT,
) -> TypeGuard[ParamsT]:
    """Narrow `obj` to `ParamsT` after `optax.apply_updates`.

    Args:
        obj: Updated parameter PyTree.
        _sample: Original parameter PyTree of type `ParamsT`.

    Returns:
        `True` if `obj` is not `None`.

    """
    _ = _sample
    return obj is not None


def _is_train_step_tuple[ParamsT, OptT](
    res: StepResult | tuple[ParamsT, OptT, StepResult] | tuple[ParamsT | None, OptT | None, SupportsFloat | float],
    _p: ParamsT | None,
    _o: OptT | None,
) -> TypeGuard[tuple[ParamsT | None, OptT | None, SupportsFloat | float]]:
    """Narrow `train_step` output to `(params, opt_state, loss)`.

    Args:
        res: Raw output from `train_step`.
        _p: Parameters reference for type inference.
        _o: Optimizer state reference for type inference.

    Returns:
        `True` for 3-tuple step results.

    """
    _ = (_p, _o)
    return isinstance(res, tuple) and len(res) == 3


jax = optional_import("jax")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
maxtext_train = optional_import("maxtext.train")
Gemma4Model: MaxTextModelConstructorProtocol | None = optional_attr(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)
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
    inputs = batch["inputs"]
    assert _is_stateless_train_model(model, inputs)
    logits = model.apply(params, inputs)
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    res: JaxArrayProtocol = jnp.mean(loss)
    return res


def _is_jax_train_mod(mod: types.ModuleType | None) -> TypeGuard[types.ModuleType]:
    """Narrow `mod` to `types.ModuleType` for attribute-guarded `.jit(...)` calls.

    Args:
        mod: Optional `jax` module.

    Returns:
        `True` when `mod` is not `None`.

    """
    return mod is not None


def _get_train_step_fn[
    ValT: TensorType | SupportsFloat,
    ParamsT: MaxTextTrainTarget,
    OptInputT,
    OptStateT,
    BatchT: MaxTextTrainTarget,
](
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
            assert _is_train_val_mapping(p, model)
            assert _is_train_val_mapping(b, model)
            return _loss_fn(model, p, b)

        assert jax is not None
        assert optax is not None
        (loss, grads) = jax.value_and_grad(_wrapper)(params, batch)
        opt_params = params
        assert _is_train_opt_input(grads, optimizer)
        assert _is_train_opt_input(opt_params, optimizer)
        (updates, opt_state) = optimizer.update(grads, opt_state, opt_params)
        raw_params = optax.apply_updates(opt_params, updates)
        assert _is_train_params_t(raw_params, params)
        scalar_loss: SupportsFloat | float = loss
        return (raw_params, opt_state, scalar_loss)

    with contextlib.suppress(AttributeError, TypeError, AssertionError):
        assert _is_jax_train_mod(jax)
        compiled: MaxTextStepCallable[ParamsT, OptStateT, BatchT] = jax.jit(train_step)
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
        raw_step = train_step(
            params,
            opt_state,
            batch,
        )
        assert _is_train_step_tuple(raw_step, params, opt_state)
        params, opt_state, loss = raw_step
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
    jnp_mod = jnp

    coord_addr = merged_kwargs.get("coordinator_address")
    num_procs = merged_kwargs.get("num_processes")
    proc_id = merged_kwargs.get("process_id")
    _initialize_jax_distributed(
        coordinator_address=str(coord_addr) if coord_addr is not None else None,
        num_processes=int(num_procs) if isinstance(num_procs, (int, str)) else None,
        process_id=int(proc_id) if isinstance(proc_id, (int, str)) else None,
    )

    etl_kwargs: dict[str, JSONValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
    allow_fallback = bool(merged_kwargs.get("allow_synthetic_fallback"))
    strict_ds = bool(merged_kwargs.get("strict_dataset"))
    dataloader, _ = resolve_training_dataloader(
        dataset,
        lambda: build_dataloader(
            ETLConfig(
                dataset_name=dataset,
                split="train",
                batch_size=batch_size,
                distributed=(not local_step_mode),
            ),
            **etl_kwargs,
        ),
        lambda: [{"inputs": jnp_mod.array([[1, 2, 3, 4]], dtype=jnp_mod.int32), "targets": jnp_mod.array([[2, 3, 4, 5]], dtype=jnp_mod.int32)}],
        allow_synthetic_fallback=allow_fallback,
        strict_dataset=strict_ds,
    )

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
        dict[str, PyTreeNode],
        dict[str, PyTreeNode],
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
    if ckpt_dir and ocp is not None and final_params is not None:
        save_maxtext_checkpoint(
            checkpoint_dir=str(ckpt_dir),
            step=epochs,
            params=final_params,
            opt_state=final_opt_state,
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
    wrapper_kwargs = extract_status_kwargs(kwargs)

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
