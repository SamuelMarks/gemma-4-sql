"""MLX-specific training pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import ETLConfig, ModelType, TensorType, TrainerState, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

mx = optional_import("mlx.core")
nn = optional_import("mlx.nn")
optim = optional_import("mlx.optimizers")
load = optional_attr("mlx_lm", "load")


def _run_training_epochs(state: TrainerState[Any, Any, Any, Any]) -> float:
    """Run MLX supervised training epochs over the dataloader.

    Args:
        state: `TrainerState` holding model, optimizer, dataloader, and value_and_grad function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Any = state.dataloader or ()
    epochs = state.epochs
    model: Any = state.policy_model
    optimizer: Any = state.optimizer
    loss_and_grad_fn = state.train_step
    assert loss_and_grad_fn is not None
    # Run training epochs.
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            inputs = mx.array(batch["inputs"])
            targets = mx.array(batch["targets"])
            (loss, grads) = loss_and_grad_fn(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(loss.item() if hasattr(loss, "item") else loss)
            batch_count += 1
        final_loss = epoch_loss / max(1, batch_count)
    return final_loss


def _execute_train(model_name: str, dataset: str, epochs: int, learning_rate: float, batch_size: int = 2) -> tuple[str, float]:
    """Execute the core training loop for MLX.

    Args:
        model_name: The name of the target model.
        dataset: The name or path of the dataset.
        epochs: The integer value for epochs.
        learning_rate: The float value for learning rate.
        batch_size: Batch size for dataloader.

    Returns:
        A tuple containing the results.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    if mx is None or nn is None or optim is None or load is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    loaded: Any = load(model_name)
    model: ModelType = loaded[0] if isinstance(loaded, (tuple, list)) else loaded

    def loss_fn(model_t: Any, inputs: TensorType, targets: TensorType) -> Any:
        """Compute training loss.

        Args:
            model_t: Target model.
            inputs: Inputs array.
            targets: Targets array.

        Returns:
            Cross entropy loss.

        """
        logits = model_t(inputs)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")
    final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=model, optimizer=optimizer, train_step=loss_and_grad_fn))
    return "completed", float(final_loss)


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Train a Text-to-SQL model using the MLX backend.

    Args:
        config: The TrainingConfig.
        **kwargs: Extra parameters like `distributed_strategy` or `raise_on_error`.

    Returns:
        A dictionary containing MLX training status and metrics.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    extra = getattr(config, "extra_kwargs", {}) or {}
    raise_on_error = bool(kwargs.get("raise_on_error", extra.get("raise_on_error", False)))

    distributed_strategy = str(kwargs.get("distributed_strategy", "none"))
    final_loss = 0.0
    status = "completed"
    if mx is None or nn is None or optim is None or load is None:
        raise DependencyMissingError("MLX dependencies are missing.", backend="mlx")
    try:
        batch_size = getattr(config, "batch_size", 2)
        status, final_loss = _execute_train(model_name, dataset, epochs, learning_rate, batch_size=batch_size)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        if raise_on_error:
            raise TrainingError(
                f"MLX training failed: {e!s}",
                backend="mlx",
                model_name=model_name,
                operation=action,
                cause=e,
            ) from e
        status = f"failed: {e!s}"
    return {"backend": "mlx", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "learning_rate": learning_rate, "status": status, "final_loss": final_loss, "distributed_strategy": distributed_strategy}
