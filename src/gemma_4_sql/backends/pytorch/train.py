"""PyTorch-specific training pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_train import generic_run_training_epochs, run_train_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.pytorch.distributed import (
    TorchDistModuleProtocol,
    cleanup_distributed,
    setup_distributed,
    wrap_model_distributed,
)
from gemma_4_sql.backends.pytorch.etl import build_dataloader
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    ETLConfig,
    ForwardOutput,
    TorchCausalLMProtocol,
    TorchOptimizerProtocol,
    TorchTensorProtocol,
    TorchTrainerState,
    TrainingConfig,
    TrainResultDict,
)


@runtime_checkable
class _PretrainedModelInstanceProtocol(TorchCausalLMProtocol, Protocol):
    """Protocol for a loaded PyTorch model supporting `.to(device)`."""

    def to(self, device: str, /) -> TorchCausalLMProtocol:
        """Move model to `device`."""
        ...


@runtime_checkable
class _PretrainedCausalLMLoaderProtocol(Protocol):
    """Protocol for a Hugging Face or native model class exposing `.from_pretrained(...)`."""

    def from_pretrained(self, model_name: str, /) -> _PretrainedModelInstanceProtocol:
        """Load a causal language model from a pretrained identifier or path."""
        ...


@runtime_checkable
class _LogitsContainerProtocol(Protocol):
    """Protocol for model output objects exposing a `.logits` tensor attribute."""

    @property
    def logits(self) -> TorchTensorProtocol:
        """Provide the prediction logits tensor."""
        ...


@runtime_checkable
class _ReshapableLogitsProtocol(TorchTensorProtocol, Protocol):
    """Protocol for a PyTorch tensor supporting `.view(...)`, `.size(...)`, and `.backward()`."""

    def view(self, *shape: int) -> _ReshapableLogitsProtocol:
        """Return a reshaped view of the tensor."""
        ...

    def size(self, dim: int, /) -> int:
        """Return the dimension size at `dim`."""
        ...

    def backward(self) -> None:
        """Compute gradients of the scalar tensor with respect to graph leaves."""
        ...

    def to(self, device: str | None, /) -> _ReshapableLogitsProtocol:
        """Move tensor to target device."""
        ...


torch = optional_import("torch")
nn = optional_import("torch.nn")
optim = optional_import("torch.optim")
Gemma4ForCausalLM: _PretrainedCausalLMLoaderProtocol | None = optional_attr(
    "transformers.models.gemma4",
    "Gemma4ForCausalLM",
    _PretrainedCausalLMLoaderProtocol,
)


def _setup_distributed(
    distributed_strategy: str,
) -> tuple[bool, TorchDistModuleProtocol | None, str, int]:
    """Set up distributed environment.

    Args:
        distributed_strategy: The string representing the distributed strategy.

    Returns:
        A 4-tuple of `(is_distributed, dist_module, device_str, device_id)`.

    """
    return setup_distributed(distributed_strategy, torch)


def _extract_logits_tensor(
    outputs: ForwardOutput | _LogitsContainerProtocol,
) -> _ReshapableLogitsProtocol:
    """Extract the 3D logits tensor from tuple, `.logits` object, or raw tensor output.

    Args:
        outputs: Model forward output.

    Returns:
        Extracted logits tensor supporting `.view` and `.size`.

    """
    if isinstance(outputs, tuple):
        first = outputs[0]
        assert isinstance(first, _ReshapableLogitsProtocol)
        return first
    if isinstance(outputs, _LogitsContainerProtocol):
        inner = outputs.logits
        assert isinstance(inner, _ReshapableLogitsProtocol)
        return inner
    assert isinstance(outputs, _ReshapableLogitsProtocol)
    return outputs


def _run_training_epochs[ModelT, OptT, BatchValT](
    state: TorchTrainerState[ModelT, OptT, Mapping[str, BatchValT]] | TorchTrainerState[ModelT, OptT, dict[str, BatchValT]],
) -> float:
    """Run PyTorch supervised training epochs over the dataloader.

    Args:
        state: `TorchTrainerState` holding model, optimizer, criterion, dataloader, and device.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[Mapping[str, BatchValT]] = state.dataloader or ()
    model = state.model
    optimizer = state.optimizer
    criterion = state.criterion
    device = state.device

    def _step(batch: Mapping[str, BatchValT]) -> float:
        """Execute a single PyTorch supervised training step and return scalar float loss."""
        assert model is not None
        assert optimizer is not None
        assert criterion is not None
        inputs = batch["inputs"].to(device)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        targets = batch["targets"].to(device)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        optimizer.zero_grad()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        raw_outputs = model(inputs)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        logits = _extract_logits_tensor(raw_outputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(loss, _ReshapableLogitsProtocol)
        loss.backward()
        optimizer.step()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        return float(loss.item())

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _wrap_model_distributed[ModelT](
    model: ModelT,
    distributed_strategy: str,
    device_id: int,
) -> ModelT:
    """Wrap model for distributed training.

    Args:
        model: The model.
        distributed_strategy: The string representing the distributed strategy.
        device_id: The integer value for device id.

    Returns:
        Wrapped distributed model or original `model`.

    """
    return wrap_model_distributed(model, distributed_strategy, device_id, torch)


def _execute_train(
    model_name: str,
    dataset: str,
    epochs: int,
    learning_rate: float,
    distributed_strategy: str,
    batch_size: int = 2,
    backend_alias: str = "pytorch",
    *,
    model_config: Gemma4Config | None = None,
) -> tuple[str, float]:
    """Execute the core PyTorch training loop.

    Args:
        model_name: Target model identifier.
        dataset: Dataset identifier.
        epochs: Number of training epochs.
        learning_rate: Training learning rate.
        distributed_strategy: Distributed training strategy.
        batch_size: Training batch size.
        backend_alias: PyTorch backend variant.
        model_config: Optional explicit `Gemma4Config` for native initialization.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    if torch is None or nn is None or optim is None:
        raise DependencyMissingError("PyTorch dependencies are missing.")

    dist_module: TorchDistModuleProtocol | None = None
    try:
        (is_distributed, dist_module, device, device_id) = _setup_distributed(distributed_strategy)
        if backend_alias == "pytorch_native":
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4

            raw_model: TorchCausalLMProtocol = NativeGemma4.from_pretrained(model_name, config=model_config).to(device)
        else:
            assert Gemma4ForCausalLM is not None
            raw_model = Gemma4ForCausalLM.from_pretrained(model_name).to(device)
        model: TorchCausalLMProtocol = _wrap_model_distributed(raw_model, distributed_strategy, device_id)
        optimizer: TorchOptimizerProtocol = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size, distributed=is_distributed))
        dataloader = data_dict.get("loader", None)
        has_loader = isinstance(dataloader, Iterable)
        if not has_loader:
            raise ValueError(f"Invalid dataloader for dataset: {dataset}")
        model.train()
        final_loss = _run_training_epochs(
            TorchTrainerState(
                model=model,
                dataloader=dataloader,
                epochs=epochs,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )
        return "completed", float(final_loss)
    finally:
        cleanup_distributed(dist=dist_module)


def train_model(
    config: TrainingConfig,
    **kwargs: StatusValue | Gemma4Config,
) -> TrainResultDict:
    """Train a Text-to-SQL model using the PyTorch backend.

    Args:
        config: Training configuration object.
        **kwargs: Extra runtime options such as `'distributed_strategy'`, `'model_config'`, or `'raise_on_error'`.

    Returns:
        A dictionary containing PyTorch training status and metrics.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.
        TrainingError: If training fails and `raise_on_error` is enabled.

    """
    model_name = config.model_name
    dataset = config.dataset
    epochs = config.epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    raw_dist = kwargs.get("distributed_strategy")
    distributed_strategy = str(raw_dist) if isinstance(raw_dist, str) and raw_dist else config.distributed_strategy
    raw_alias = kwargs.get("backend_alias") or kwargs.get("backend") or ("pytorch_native" if config.backend == "pytorch_native" else "pytorch")
    backend_alias = str(raw_alias)

    if torch is None or optim is None or nn is None or (backend_alias != "pytorch_native" and Gemma4ForCausalLM is None):
        raise DependencyMissingError("PyTorch dependencies are missing.", backend=backend_alias)

    raw_cfg = kwargs.get("model_config") or kwargs.get("config")
    resolved_model_cfg = raw_cfg if isinstance(raw_cfg, Gemma4Config) else None
    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    wrapper_kwargs["distributed_strategy"] = distributed_strategy
    return run_train_wrapper(
        backend_alias,
        config,
        lambda: _execute_train(
            model_name,
            dataset,
            epochs,
            learning_rate,
            distributed_strategy,
            batch_size=batch_size,
            backend_alias=backend_alias,
            model_config=resolved_model_cfg,
        ),
        **wrapper_kwargs,
    )


__all__ = [
    "Gemma4ForCausalLM",
    "cleanup_distributed",
    "train_model",
]
