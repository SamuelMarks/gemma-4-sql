"""MaxText-specific training pipeline and distributed cluster orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol
from gemma_4_sql._types_tensors import extract_scalar_float
from gemma_4_sql.backends.common_train import (
    generic_run_training_epochs,
    resolve_training_dataloader,
    run_train_wrapper,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.maxtext._protocols import (
    JaxDistributedAndPRNGProtocol,
    JaxDistributedHolderProtocol,
    JaxPRNGHolderProtocol,
    JnpZerosAndArrayProtocol,
    OrbaxCheckpointModuleProtocol,
)
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.type_hints import (
    ETLConfig,
    JaxArrayProtocol,
    JSONValue,
    MaxTextTrainerState,
    PyTreeNode,
    TensorType,
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
        """Apply model parameters to `inputs`.

        Args:
            params: Model parameter PyTree mapping.
            inputs: Input token ID array.

        Returns:
            Output vocabulary logits array.

        """
        ...


type MaxTextTrainModel[ValT] = _StatelessApplyModelProtocol[ValT]


@runtime_checkable
class _OptaxOptimizerProtocol[GradInT, GradOutT, OptStateT](Protocol):
    """Protocol for an Optax `GradientTransformation`.

    Supports `.init(params)` and `.update(grads, opt_state, params)`.

    """

    def init(self, params: GradInT, /) -> OptStateT:
        """Initialize optimizer state from parameters.

        Args:
            params: Model parameter PyTree.

        Returns:
            Initial optimizer state.

        """
        ...

    def update(
        self,
        updates: GradInT,
        state: OptStateT,
        params: GradInT | None = None,
        /,
    ) -> tuple[GradOutT, OptStateT]:
        """Compute parameter updates and next optimizer state.

        Args:
            updates: Gradient PyTree.
            state: Current optimizer state.
            params: Optional current parameter PyTree.

        Returns:
            Tuple of `(parameter_updates, updated_optimizer_state)`.

        """
        ...


@runtime_checkable
class _JaxJitProtocol(Protocol):
    """Protocol for a `jax` module exposing `.jit(...)`."""

    def jit[StepFnT](self, fun: StepFnT, /) -> StepFnT:
        """JIT-compile a step callable.

        Args:
            fun: Pure step function to compile.

        Returns:
            Compiled step callable with the same signature.

        """
        ...


@runtime_checkable
class _JaxValueAndGrad2Protocol(Protocol):
    """Protocol for a `jax` module exposing 2-argument `.value_and_grad(...)`."""

    def value_and_grad[P1, B1, L1](
        self,
        fun: Callable[[P1, B1], L1],
        /,
    ) -> Callable[[P1, B1], tuple[L1, P1]]:
        """Differentiate `fun(params, batch)` with respect to its first argument.

        Args:
            fun: Scalar loss callable `(params, batch) -> loss`.

        Returns:
            Callable returning `(loss, grads)`.

        """
        ...


type _OptaxUpdatesTree = (
    PyTreeNode | Iterable[_OptaxUpdatesTree] | Mapping[str, _OptaxUpdatesTree] | complex | bytes | None
)


@runtime_checkable
class _OptaxApplyUpdatesProtocol(Protocol):
    """Protocol for `optax` providing `.apply_updates(params, updates)`."""

    def apply_updates[ParamsT](self, params: ParamsT, updates: _OptaxUpdatesTree, /) -> ParamsT:
        """Add parameter updates to parameters.

        Args:
            params: Current parameter PyTree.
            updates: Optimizer update PyTree.

        Returns:
            Updated parameter PyTree.

        """
        ...


type MaxTextTrainTarget = (
    Mapping[str, PyTreeNode]
    | Mapping[str, JaxArrayProtocol]
    | Mapping[str, TensorType]
    | PyTreeNode
    | JaxArrayProtocol
    | TensorType
    | SupportsFloat
)

type MaxTextStepCallable[ParamsT, OptStateT, BatchT] = Callable[
    [ParamsT, OptStateT, BatchT],
    tuple[ParamsT, OptStateT, SupportsFloat | float],
]


@runtime_checkable
class _JnpTrainModuleProtocol(JnpZerosAndArrayProtocol, Protocol):
    """Protocol for `jax.numpy` reductions and array creation in MaxText training."""

    def maximum(self, x1: JaxArrayProtocol, x2: float, /) -> JaxArrayProtocol:
        """Compute elementwise `max(x1, x2)`.

        Args:
            x1: Mask token count sum.
            x2: Minimum denominator `1.0`.

        Returns:
            Clamped denominator `JaxArrayProtocol`.

        """
        ...

    def sum(self, a: TensorType | SupportsFloat, /) -> JaxArrayProtocol:
        """Sum all elements of `a`.

        Args:
            a: Masked token loss or mask tensor.

        Returns:
            Scalar sum `JaxArrayProtocol`.

        """
        ...

    def mean(self, a: _MultiplicativeLossProtocol | JaxArrayProtocol, /) -> JaxArrayProtocol:
        """Compute scalar mean over all elements of `a`.

        Args:
            a: Unmasked token cross-entropy loss tensor.

        Returns:
            Scalar mean `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class _MultiplicativeLossProtocol(Protocol):
    """Protocol for a per-token cross-entropy loss tensor supporting `* mask` multiplication."""

    def __mul__(self, other: TensorType | SupportsFloat, /) -> JaxArrayProtocol:
        """Multiply per-token loss by `other` mask.

        Args:
            other: Loss mask tensor or scalar.

        Returns:
            Masked loss `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class _OptaxTrainModuleProtocol(Protocol):
    """Protocol for `optax` exposing `softmax_cross_entropy_with_integer_labels` and `adamw`."""

    def softmax_cross_entropy_with_integer_labels(
        self,
        logits: TensorType | SupportsFloat,
        labels: TensorType | SupportsFloat,
        /,
    ) -> _MultiplicativeLossProtocol:
        """Compute per-token cross-entropy loss against integer token `labels`.

        Args:
            logits: Model output logits tensor.
            labels: Target integer token IDs.

        Returns:
            Per-token cross-entropy loss `JaxArrayProtocol`.

        """
        ...

    def adamw(
        self,
        learning_rate: float,
        /,
    ) -> _OptaxOptimizerProtocol[dict[str, JaxArrayProtocol], dict[str, JaxArrayProtocol], dict[str, PyTreeNode]]:
        """Construct an `adamw` gradient transformation.

        Args:
            learning_rate: Optimizer learning rate.

        Returns:
            Configured `_OptaxOptimizerProtocol`.

        """
        ...


@runtime_checkable
class _MaxTextTrainMainProtocol(Protocol):
    """Protocol for `maxtext.train` exposing `main(cli_args)`."""

    def main(self, argv: Sequence[str], /) -> float | int | None:
        """Execute distributed MaxText training loop with `argv`.

        Args:
            argv: Sequence of MaxText CLI/Gin configuration arguments.

        Returns:
            Final scalar training loss or `None`.

        """
        ...


jax: JaxDistributedAndPRNGProtocol | JaxDistributedHolderProtocol | None = optional_import(
    "jax",
    JaxDistributedAndPRNGProtocol,
)
jnp: _JnpTrainModuleProtocol | None = optional_import("jax.numpy", _JnpTrainModuleProtocol)
optax: _OptaxTrainModuleProtocol | None = optional_import("optax", _OptaxTrainModuleProtocol)
maxtext_train: _MaxTextTrainMainProtocol | None = optional_import("maxtext.train", _MaxTextTrainMainProtocol)
Gemma4Model: MaxTextModelConstructorProtocol | None = import_optional_symbol(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)
ocp: OrbaxCheckpointModuleProtocol | None = optional_import("orbax.checkpoint", OrbaxCheckpointModuleProtocol)


def _loss_fn[ArrT: TensorType](
    model: _StatelessApplyModelProtocol[ArrT],
    params: Mapping[str, ArrT],
    batch: Mapping[str, ArrT],
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
    logits = model.apply(params, inputs)
    targets = batch["targets"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    if "loss_mask" in batch:
        mask = batch["loss_mask"]
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        masked_res: JaxArrayProtocol = jnp.sum(loss * mask) / denom
        return masked_res
    res: JaxArrayProtocol = jnp.mean(loss)
    return res


def _get_train_step_fn[
    ArrT: TensorType,
    UpdatesT: _OptaxUpdatesTree,
    OptStateT,
](
    model: MaxTextTrainModel[ArrT],
    optimizer: _OptaxOptimizerProtocol[dict[str, ArrT], UpdatesT, OptStateT],
) -> MaxTextStepCallable[dict[str, ArrT], OptStateT, Mapping[str, ArrT]]:
    """Construct a JIT-compiled or standard training step execution function.

    Args:
        model: Model instance providing forward execution.
        optimizer: Optax optimizer instance for parameter updates.

    Returns:
        A callable `train_step(params, opt_state, batch)` returning `(updated_params, updated_opt_state, loss)`.

    """

    def train_step(
        params: dict[str, ArrT],
        opt_state: OptStateT,
        batch: Mapping[str, ArrT],
    ) -> tuple[dict[str, ArrT], OptStateT, SupportsFloat | float]:
        """Perform a single forward-backward pass and update optimizer state.

        Args:
            params: Current model parameter PyTree.
            opt_state: Current optimizer state PyTree.
            batch: Dictionary containing inputs and target tensors.

        Returns:
            Tuple of `(updated_params, updated_opt_state, scalar_loss)`.

        """

        def _wrapper(p: dict[str, ArrT], b: Mapping[str, ArrT]) -> JaxArrayProtocol:
            """Evaluate `_loss_fn` for JAX `value_and_grad` differentiation.

            Args:
                p: Parameter mapping.
                b: Batch mapping.

            Returns:
                Scalar cross-entropy loss array.

            """
            return _loss_fn(model, p, b)

        assert isinstance(jax, _JaxValueAndGrad2Protocol)
        assert isinstance(optax, _OptaxApplyUpdatesProtocol)
        (loss, grads) = jax.value_and_grad(_wrapper)(params, batch)
        (updates, next_opt_state) = optimizer.update(grads, opt_state, params)
        updated_params: dict[str, ArrT] = optax.apply_updates(params, updates)
        scalar_loss: SupportsFloat | float = loss
        return (updated_params, next_opt_state, scalar_loss)

    if isinstance(jax, _JaxJitProtocol):
        return jax.jit(train_step)
    return train_step


def _run_training_epochs[ParamsT, OptT, BatchT](
    state: MaxTextTrainerState[ParamsT, OptT, BatchT],
) -> tuple[ParamsT, OptT, float]:
    """Execute epochs over dataloader using `MaxTextTrainerState`.

    Args:
        state: `MaxTextTrainerState` containing dataloader, epochs, step function, and initial states.

    Returns:
        Tuple of `(final_params, final_opt_state, final_loss)`.

    """
    params = state.params
    opt_state = state.opt_state
    train_step = state.train_step
    dataloader: Iterable[BatchT] = state.dataloader

    def process_batch(batch: BatchT) -> float:
        """Process a single batch during epoch iteration.

        Args:
            batch: Dictionary containing model inputs and labels.

        Returns:
            Computed scalar loss value as a float.

        """
        nonlocal params, opt_state
        params, opt_state, loss = train_step(
            params,
            opt_state,
            batch,
        )
        return extract_scalar_float(loss)

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
        raise ExportError(
            f"Failed to persist Orbax checkpoint at '{ckpt_path}': {e}", backend="maxtext", cause=e
        ) from e


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
    from gemma_4_sql.backends.maxtext.inference import cache_maxtext_params

    if isinstance(model_name_or_config, TrainingConfig):
        cfg = model_name_or_config
        model_name = cfg.model_name
        dataset = cfg.dataset
        epochs = cfg.epochs
        learning_rate = cfg.learning_rate
        batch_size = cfg.batch_size
        extra: dict[str, JSONValue] = {
            k: v for k, v in (cfg.extra_kwargs or {}).items() if isinstance(v, (str, int, float, bool)) or v is None
        }
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

    etl_kwargs: dict[str, JSONValue] = {
        k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, dict)) or v is None
    }
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
        lambda: [
            {
                "inputs": jnp_mod.array([[1, 2, 3, 4]], dtype=jnp_mod.int32),
                "targets": jnp_mod.array([[2, 3, 4, 5]], dtype=jnp_mod.int32),
            }
        ],
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
    assert isinstance(jax, JaxPRNGHolderProtocol)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
    raw_init_params = dict(model.init(rng, dummy_input))
    params: dict[str, JaxArrayProtocol] = {
        str(k): v for k, v in raw_init_params.items() if isinstance(v, JaxArrayProtocol)
    }
    optimizer: _OptaxOptimizerProtocol[
        dict[str, JaxArrayProtocol],
        dict[str, JaxArrayProtocol],
        dict[str, PyTreeNode],
    ] = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)
    train_step: MaxTextStepCallable[
        dict[str, JaxArrayProtocol],
        dict[str, PyTreeNode],
        Mapping[str, JaxArrayProtocol],
    ] = _get_train_step_fn(model, optimizer)

    final_params, final_opt_state, final_loss = _run_training_epochs(
        MaxTextTrainerState(
            dataloader=dataloader,
            epochs=epochs,
            train_step=train_step,
            params=params,
            opt_state=opt_state,
        )
    )
    cache_maxtext_params(
        model_name,
        final_params,
        model=model,
    )

    ckpt_dir = merged_kwargs.get("checkpoint_dir")
    if ckpt_dir and ocp is not None:
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
