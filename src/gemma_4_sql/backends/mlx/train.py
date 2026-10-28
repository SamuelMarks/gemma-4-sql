"""MLX-specific training pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_train import generic_run_training_epochs, run_train_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    ETLConfig,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    TensorType,
    TokenizerProtocol,
    TrainingConfig,
    TrainingLoopState,
    TrainResultDict,
)


@runtime_checkable
class _MlxOptimizerProtocol(Protocol):
    """Protocol for an `mlx.optimizers` optimizer supporting `.update(model, grads)` and `.state`."""

    @property
    def state(self) -> Mapping[str, MlxArrayProtocol]:
        """Provide the optimizer state dictionary."""
        ...

    def update(
        self,
        model: MlxCausalLMProtocol,
        grads: Mapping[str, MlxArrayProtocol],
        /,
    ) -> None:
        """Apply gradients to `model` parameters."""
        ...


type MlxTrainBatch = Mapping[str, TensorType | Sequence[int] | Sequence[Sequence[int]] | MlxArrayProtocol]


mx = optional_import("mlx.core")
nn = optional_import("mlx.nn")
optim = optional_import("mlx.optimizers")
_raw_load = optional_attr("mlx_lm", "load")
load: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None] | MlxCausalLMProtocol] | None = (  # ty: ignore[invalid-assignment]
    _raw_load if callable(_raw_load) else None
)


def _run_training_epochs[ModelT, OptT, BatchT: MlxTrainBatch](
    state: TrainingLoopState[ModelT, OptT, BatchT],
) -> float:
    """Run MLX supervised training epochs over the dataloader.

    Args:
        state: `TrainingLoopState` holding model, optimizer, dataloader, and `value_and_grad` function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader or ()
    model = state.model
    optimizer = state.optimizer
    loss_and_grad_fn = state.train_step

    def _step(batch: BatchT) -> float:
        """Execute a single MLX supervised training step and return scalar float loss."""
        assert mx is not None
        assert model is not None
        assert optimizer is not None
        assert loss_and_grad_fn is not None
        inputs = mx.array(batch["inputs"])
        targets = mx.array(batch["targets"])
        (loss, grads) = loss_and_grad_fn(model, inputs, targets)  # pyright: ignore[reportGeneralTypeIssues]  # ty: ignore[not-iterable]
        optimizer.update(model, grads)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        mx.eval(model.parameters(), optimizer.state)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        try:
            return float(loss.item())  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        except (AttributeError, TypeError, ValueError):
            return float(loss)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

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
            train_step=loss_and_grad_fn,
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
