"""PyTorch-specific training pipeline."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.pytorch.etl import build_dataloader
from gemma_4_sql.type_hints import ETLConfig, TrainerState, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

try:
    import torch as _torch
    from torch import nn as _nn
    from torch import optim as _optim

    torch: Any = _torch
    nn: Any = _nn
    optim: Any = _optim
except (ImportError, AttributeError, RuntimeError):
    torch = None
    nn = None
    optim = None

try:
    from transformers.models.gemma4 import Gemma4ForCausalLM as _Gemma4ForCausalLM

    Gemma4ForCausalLM: Any = _Gemma4ForCausalLM
except (ImportError, AttributeError):
    Gemma4ForCausalLM = None


def _setup_distributed(distributed_strategy: str) -> tuple[bool, object, object, int]:
    """Set up distributed environment.

    Args:
        distributed_strategy: The string representing the distributed strategy.

    Returns:
        A tuple containing the results.
    """
    is_distributed = distributed_strategy in {"ddp", "fsdp"}
    dist = None
    device_id = 0
    if is_distributed:
        dist = __import__("torch.distributed", fromlist=[""])
        if not dist.is_initialized():  # pragma: no cover
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        device_id = rank % max(1, torch.cuda.device_count())
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (is_distributed, dist, device, device_id)


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
    criterion = state.criterion
    device = state.device
    """Run training epochs.

    Returns:
        object: The resulting output from the operation.

    """
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, "logits", outputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / max(1, len(dataloader))
    return final_loss


def _wrap_model_distributed(model: Any, distributed_strategy: str, device_id: int) -> object:
    """Wrap model for distributed training.

    Args:
        model: The model.
        distributed_strategy: The string representing the distributed strategy.
        device_id: The integer value for device id.

    Returns:
        The execution result.
    """
    if distributed_strategy == "ddp":
        ddp_module = importlib.import_module("torch.nn.parallel")
        ddp_class = ddp_module.DistributedDataParallel
        return ddp_class(model, device_ids=[device_id] if getattr(torch, "cuda", None) and getattr(torch.cuda, "is_available", lambda: False)() else None)
    if distributed_strategy == "fsdp":
        fsdp_module = importlib.import_module("torch.distributed.fsdp")
        fsdp_class = fsdp_module.FullyShardedDataParallel
        return fsdp_class(model)
    return model


def _cleanup_distributed(dist: Any) -> None:
    """Cleanup distributed environment.

    Args:
        dist: Distributed module object.
    """
    if dist is not None and getattr(dist, "is_initialized", lambda: False)():
        dist.destroy_process_group()


def _execute_train(
    model_name: str,
    dataset: str,
    epochs: int,
    learning_rate: float,
    distributed_strategy: str,
    batch_size: int = 2,
    backend_alias: str = "pytorch",
    **kwargs: object,
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
        **kwargs: Additional model and trainer options.

    Returns:
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.
        ValueError: If dataloader is invalid.
    """
    if torch is None or nn is None or optim is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch dependencies are missing.")

    dist_module = None
    try:
        (is_distributed, dist_module, device, device_id) = _setup_distributed(distributed_strategy)
        if backend_alias == "pytorch_native":
            from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4

            raw_model = NativeGemma4.from_pretrained(model_name, config=cast(Any, kwargs.get("model_config") or kwargs.get("config"))).to(cast(Any, device))
        else:
            raw_model = Gemma4ForCausalLM.from_pretrained(model_name).to(cast(Any, device))
        model: Any = _wrap_model_distributed(raw_model, distributed_strategy, device_id)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size, distributed=is_distributed))
        dataloader = data_dict.get("loader", None)
        if dataloader is None or not hasattr(dataloader, "__iter__"):
            raise ValueError(f"Invalid dataloader for dataset: {dataset}")
        model.train()
        final_loss = _run_training_epochs(TrainerState(policy_model=model, dataloader=dataloader, epochs=epochs, optimizer=optimizer, criterion=criterion, device=device))
        _cleanup_distributed(dist=dist_module)
        return "completed", float(final_loss)
    except Exception:
        _cleanup_distributed(dist=dist_module)
        raise


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Train a Text-to-SQL model using the PyTorch backend.

    Args:
        config: Training configuration object.
        **kwargs: Extra runtime options such as 'test_mode' and 'distributed_strategy'.

    Returns:
        A dictionary containing PyTorch training status and metrics.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.
    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    distributed_strategy = str(kwargs.get("distributed_strategy") or getattr(config, "distributed_strategy", "none"))
    backend_alias = str(kwargs.get("backend_alias") or kwargs.get("backend") or ("pytorch_native" if getattr(config, "backend", None) == "pytorch_native" else "pytorch"))

    final_loss = 0.0
    status = "completed"
    if torch is None or optim is None or (nn is None) or (backend_alias != "pytorch_native" and Gemma4ForCausalLM is None):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch dependencies are missing.")
    try:
        batch_size = getattr(config, "batch_size", 2)
        train_kwargs = dict(kwargs)
        train_kwargs.pop("backend_alias", None)
        train_kwargs.pop("backend", None)
        status, final_loss = _execute_train(
            model_name,
            dataset,
            epochs,
            learning_rate,
            str(distributed_strategy),
            batch_size=batch_size,
            backend_alias=backend_alias,
            **train_kwargs,
        )
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        status = f"failed: {e!s}"
    return {
        "backend": backend_alias,
        "action": action,
        "model": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "status": status,
        "final_loss": float(str(final_loss)) if final_loss is not None else 0.0,
        "distributed_strategy": distributed_strategy,
    }
