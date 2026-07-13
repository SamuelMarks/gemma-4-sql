"""MLX-specific training pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.type_hints import ETLConfig, TrainerState, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
mx = None
nn = None
optim = None
with catch_optional_imports():
    import mlx.core as mx
    import mlx.optimizers as optim  # pragma: no cover
    from mlx import nn  # pragma: no cover
load = None
with catch_optional_imports():
    from mlx_lm import load


def _run_training_epochs(state: TrainerState) -> float:
    """Execute function.

    Args:
        state: The state.

    Returns:
        The computed float value.
    """
    dataloader = state.dataloader
    epochs = state.epochs
    model = state.policy_model
    optimizer = state.optimizer
    loss_and_grad_fn = state.train_step
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
            epoch_loss += loss.item()
            batch_count += 1
        final_loss = epoch_loss / max(1, batch_count)
    return final_loss


def _execute_train(model_name: str, dataset: str, epochs: int, learning_rate: float) -> tuple[str, float]:
    """Execute the core training loop for MLX.

    Args:
        model_name: The name of the target model.
        dataset: The name or path of the dataset.
        epochs: The integer value for epochs.
        learning_rate: The float value for learning rate.

    Returns:
        A tuple containing the results.
    """
    (model, _) = load(model_name)

    def loss_fn(model_t: object, inputs: object, targets: object) -> object:
        """Docstring.

        Returns:
            object: The resulting output from the operation.

        """
        logits = model_t(inputs)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))
    dataloader = data_dict.get("loader", None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")
    final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=model, optimizer=optimizer, train_step=loss_and_grad_fn))
    return "completed", float(final_loss)


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Train a Text-to-SQL model using the MLX backend.

    Args:
    ----
        config: The TrainingConfig.
        kwargs: Additional arguments.
        model_name: The name of the model to train.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        **kwargs: Extra parameters like distributed_strategy.

    Returns:
    -------
        A dictionary containing MLX training status and metrics.

    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)

    distributed_strategy = str(kwargs.get("distributed_strategy", "none"))
    final_loss = 0.5
    status = "completed"
    if mx is None or nn is None or optim is None or load is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    try:
        status, final_loss = _execute_train(model_name, dataset, epochs, learning_rate)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:  # pragma: no cover
        status = f"failed: {e!s}"  # pragma: no cover
    return {"backend": "mlx", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "learning_rate": learning_rate, "status": status, "final_loss": final_loss, "distributed_strategy": distributed_strategy}
