"""MLX-specific training pipeline without fake TypeGuards."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql._types_models import MlxLMLoadProtocol, MlxNNLossesProtocol, ScalarItemExtractorProtocol, StepResult
from gemma_4_sql._types_results import ETLBatchValue
from gemma_4_sql.backends.common_train import (
    generic_run_training_epochs,
    resolve_training_dataloader,
    run_train_wrapper,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    ETLConfig,
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    MlxTrainerState,
    TensorProtocol,
    TrainingConfig,
    TrainResultDict,
)

type MlxGradTree = Mapping[str, "MlxArrayProtocol | MlxGradTree"]


def _to_mlx_grad_tree(
    node: StepResult | MlxGradTree | Mapping[str, TensorProtocol] | JSONValue,
) -> dict[str, MlxArrayProtocol | MlxGradTree] | None:
    """Recursively extract `MlxArrayProtocol` leaves and nested submodule dictionaries from a gradient tree.

    Args:
        node: Raw gradient mapping or leaf returned by `mlx.nn.value_and_grad`.

    Returns:
        Normalized nested `MlxGradTree` dictionary, or `None` when `node` is not a `Mapping`.

    """
    if isinstance(
        node,
        (MlxArrayProtocol, TensorProtocol, ScalarItemExtractorProtocol, SupportsFloat, Sequence),
    ) or not isinstance(node, Mapping):
        return None
    out: dict[str, MlxArrayProtocol | MlxGradTree] = {}
    for k, v in node.items():
        if isinstance(v, MlxArrayProtocol):
            out[str(k)] = v
        else:
            sub = _to_mlx_grad_tree(v)
            if sub is not None:
                out[str(k)] = sub
    return out


@runtime_checkable
class _MlxParamLeafProtocol(Protocol):
    """Structural protocol for MLX parameter tensors exposing `.shape`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide tensor shape dimensions."""
        ...


@runtime_checkable
class _MlxTrainModelProtocol(Protocol):
    """Protocol for an MLX model exposing `.parameters()`."""

    def parameters(
        self,
    ) -> Mapping[
        str,
        MlxArrayProtocol | _MlxParamLeafProtocol | Mapping[str, MlxArrayProtocol] | Sequence[MlxArrayProtocol],
    ]:
        """Return parameter dictionary."""
        ...


@runtime_checkable
class _MlxOptimizerProtocol(Protocol):
    """Protocol for an `mlx.optimizers` optimizer supporting `.update(model, grads)` and `.state`."""

    @property
    def state(self) -> Mapping[str, MlxArrayProtocol] | Sequence[MlxArrayProtocol]:
        """Provide the optimizer state dictionary."""
        ...

    def update(
        self,
        model: MlxCausalLMProtocol | _MlxTrainModelProtocol,
        grads: MlxGradTree,
        /,
    ) -> None:
        """Apply gradients to `model` parameters.

        Args:
            model: Trainable MLX model.
            grads: Parameter gradient dictionary.

        """
        ...


type MlxTrainBatch = Mapping[str, ETLBatchValue | MlxArrayProtocol]
type _MlxEvalArg = (
    MlxArrayProtocol
    | Mapping[str, MlxArrayProtocol | TensorProtocol | Mapping[str, MlxArrayProtocol | TensorProtocol]]
    | Mapping[
        str,
        MlxArrayProtocol | _MlxParamLeafProtocol | Mapping[str, MlxArrayProtocol] | Sequence[MlxArrayProtocol],
    ]
    | Sequence[MlxArrayProtocol]
    | None
)


@runtime_checkable
class _MlxTrainCoreProtocol(Protocol):
    """Protocol for `mlx.core` exposing `array` and `eval` in MLX training."""

    def array(self, val: ETLBatchValue | MlxArrayProtocol, /) -> MlxArrayProtocol:
        """Convert batch sequences or arrays into an `MlxArrayProtocol`.

        Args:
            val: Batch token sequences or array.

        Returns:
            Constructed `MlxArrayProtocol`.

        """
        ...

    def eval(self, *args: _MlxEvalArg) -> None:
        """Evaluate lazy MLX parameter and optimizer state graphs.

        Args:
            *args: Parameter trees, optimizer states, or arrays to materialize.

        """
        ...


@runtime_checkable
class _MlxTrainNNProtocol(Protocol):
    """Protocol for `mlx.nn` exposing `.losses` and `.value_and_grad`."""

    @property
    def losses(self) -> MlxNNLossesProtocol:
        """Provide `mlx.nn.losses` namespace."""
        ...

    def value_and_grad[LossReturnT](
        self,
        model: MlxCausalLMProtocol | _MlxTrainModelProtocol,
        fn: Callable[..., LossReturnT],
        /,
    ) -> Callable[..., tuple[LossReturnT, dict[str, MlxArrayProtocol]]]:
        """Create a value-and-gradient callable with respect to `model`.

        Args:
            model: Trainable MLX model.
            fn: Scalar loss function.

        Returns:
            Callable returning `(loss, grads)`.

        """
        ...


@runtime_checkable
class _MlxTrainOptimModuleProtocol(Protocol):
    """Protocol for `mlx.optimizers` exposing `AdamW`."""

    def AdamW(self, *, learning_rate: float) -> _MlxOptimizerProtocol:
        """Construct an `AdamW` MLX optimizer.

        Args:
            learning_rate: Optimizer learning rate.

        Returns:
            Configured `_MlxOptimizerProtocol`.

        """
        ...


mx: _MlxTrainCoreProtocol | None = optional_import("mlx.core", _MlxTrainCoreProtocol)
nn: _MlxTrainNNProtocol | None = optional_import("mlx.nn", _MlxTrainNNProtocol)
optim: _MlxTrainOptimModuleProtocol | None = optional_import("mlx.optimizers", _MlxTrainOptimModuleProtocol)
load: MlxLMLoadProtocol | None = import_optional_symbol(
    "mlx_lm",
    "load",
    MlxLMLoadProtocol,
)


def _run_training_epochs[
    ModelT: MlxCausalLMProtocol | _MlxTrainModelProtocol,
    OptT,
    BatchT: MlxTrainBatch,
](
    state: MlxTrainerState[ModelT, OptT, BatchT],
) -> float:
    """Run MLX supervised training epochs over the dataloader.

    Args:
        state: `MlxTrainerState` holding model, optimizer, dataloader, and `loss_and_grad_fn` function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader
    model = state.model
    optimizer = state.optimizer
    loss_and_grad_fn = state.loss_and_grad_fn

    def _step(batch: BatchT) -> float:
        """Execute a single MLX supervised training step and return scalar float loss.

        Args:
            batch: Supervised training batch mapping containing `'inputs'` and `'targets'`.

        Returns:
            Scalar float cross-entropy loss for the batch.

        """
        assert mx is not None
        assert isinstance(optimizer, _MlxOptimizerProtocol)
        inputs = mx.array(batch["inputs"])
        targets = mx.array(batch["targets"])
        raw_out = loss_and_grad_fn(model, inputs, targets)
        assert isinstance(raw_out, tuple)
        assert len(raw_out) == 2
        loss, raw_grads = raw_out
        grads_map: MlxGradTree = _to_mlx_grad_tree(raw_grads) or {}
        optimizer.update(model, grads_map)
        mx.eval(model.parameters(), optimizer.state)
        if isinstance(loss, MlxArrayProtocol):
            raw_item = loss.item()
            return float(raw_item.real) if isinstance(raw_item, complex) else float(raw_item)
        if isinstance(loss, SupportsFloat):
            return float(loss)
        return 0.0

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _execute_train(
    model_name: str,
    dataset: str,
    epochs: int,
    learning_rate: float,
    batch_size: int = 2,
) -> tuple[str, float]:
    """Execute the core training loop for MLX.

    Args:
        model_name: The name of the target model.
        dataset: The name or path of the dataset.
        epochs: The integer value for epochs.
        learning_rate: The float value for learning rate.
        batch_size: Batch size for dataloader.

    Returns:
        A tuple containing `(status, final_loss)`.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    if mx is None or nn is None or optim is None or load is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    loaded = load(model_name)
    model: MlxCausalLMProtocol = loaded[0] if isinstance(loaded, (tuple, list)) else loaded

    def loss_fn(
        model_t: MlxCausalLMProtocol,
        inputs: MlxArrayProtocol,
        targets: MlxArrayProtocol,
    ) -> MlxArrayProtocol:
        """Compute cross-entropy training loss.

        Args:
            model_t: Target model.
            inputs: Inputs array.
            targets: Targets array.

        Returns:
            Mean cross-entropy loss.

        """
        assert nn is not None
        raw_logits = model_t(inputs)
        assert isinstance(raw_logits, MlxArrayProtocol)
        logits: MlxArrayProtocol = raw_logits
        if logits.ndim >= 2 and targets.ndim >= 2 and int(logits.shape[1]) != int(targets.shape[1]):
            min_len = min(int(logits.shape[1]), int(targets.shape[1]))
            logits = logits[:, :min_len]
            targets = targets[:, :min_len]
        res: MlxArrayProtocol = nn.losses.cross_entropy(logits, targets, reduction="mean")
        return res

    optimizer: _MlxOptimizerProtocol = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    dataloader, _ = resolve_training_dataloader(
        dataset,
        lambda: build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size)),
        lambda: [{"inputs": [[1, 2, 3, 4]], "targets": [[2, 3, 4, 5]]}],
    )
    final_loss = _run_training_epochs(
        MlxTrainerState(
            dataloader=dataloader,
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            loss_and_grad_fn=loss_and_grad_fn,
            learning_rate=learning_rate,
        )
    )
    return "completed", float(final_loss)


def train_model(config: TrainingConfig, **kwargs: StatusValue) -> TrainResultDict:
    """Train a Text-to-SQL model using the MLX backend.

    Args:
        config: The `TrainingConfig`.
        **kwargs: Extra parameters like `distributed_strategy` or `raise_on_error`.

    Returns:
        A dictionary containing MLX training status and metrics.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    if mx is None or nn is None or optim is None or load is None:
        raise DependencyMissingError("MLX dependencies are missing.", backend="mlx")
    model_name = config.model_name
    dataset = config.dataset
    epochs = config.epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    distributed_strategy = str(kwargs.get("distributed_strategy", "none"))
    call_kwargs: dict[str, StatusValue] = dict(kwargs)
    call_kwargs["distributed_strategy"] = distributed_strategy
    return run_train_wrapper(
        "mlx",
        config,
        lambda: _execute_train(model_name, dataset, epochs, learning_rate, batch_size=batch_size),
        **call_kwargs,
    )
