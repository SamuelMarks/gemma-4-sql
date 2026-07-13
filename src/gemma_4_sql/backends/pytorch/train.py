"""PyTorch-specific training pipeline."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.backends.pytorch.etl import build_dataloader
from gemma_4_sql.type_hints import ETLConfig, TrainerState, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
torch = None
nn = None
optim = None
with catch_optional_imports():
    import torch
    from torch import nn, optim
Gemma4ForCausalLM = None
with catch_optional_imports():
    from transformers.models.gemma4 import Gemma4ForCausalLM


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
            logits = getattr(outputs, "logits", outputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / max(1, len(dataloader))
    return final_loss


def _wrap_model_distributed(model: torch.nn.Module, distributed_strategy: str, device_id: int) -> object:
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


def _cleanup_distributed(dist: object) -> None:
    """Cleanup distributed environment.

    Args:
        model_name: The name of the target model.
        dataset: The name or path of the dataset.
        epochs: The integer value for epochs.
        learning_rate: The float value for learning rate.
        distributed_strategy: The string representing the distributed strategy.

    Returns:
        A tuple containing the results.
    """
    if dist is not None and getattr(dist, "is_initialized", lambda: False)():
        dist.destroy_process_group()


def _execute_train(model_name: str, dataset: str, epochs: int, learning_rate: float, distributed_strategy: str) -> tuple[str, float]:
    """Execute the core PyTorch training loop."""
    dist_module = None
    try:
        (is_distributed, dist_module, device, device_id) = _setup_distributed(distributed_strategy)
        model = Gemma4ForCausalLM.from_pretrained(model_name).to(device)
        model = _wrap_model_distributed(model, distributed_strategy, device_id)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2, distributed=is_distributed))
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
    """Execute function.


    Args:
        **kwargs: Extra runtime options such as 'test_mode' and 'distributed_strategy'.
    Returns:
        The execution result.

    """
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    distributed_strategy = kwargs.get("distributed_strategy", "none") if not hasattr(config, "distributed_strategy") else config.distributed_strategy
    """Train a Text-to-SQL model using the PyTorch backend.

    Args:
    ----
        action: The training action (e.g. 'pretrain', 'sft').
        model_name: The name of the model to train.
        dataset: The dataset to train on.
        epochs: Number of epochs to train.
        learning_rate: The learning rate.
        **kwargs: Extra parameters like distributed_strategy.

    Returns:
    -------
        A dictionary containing PyTorch training status and metrics.

    """

    final_loss = 0.5
    status = "completed"
    if torch is None or Gemma4ForCausalLM is None or optim is None or (nn is None):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch dependencies are missing.")
    try:
        status, final_loss = _execute_train(model_name, dataset, epochs, learning_rate, str(distributed_strategy))
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        status = f"failed: {e!s}"
    return {"backend": "pytorch", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "learning_rate": learning_rate, "status": status, "final_loss": final_loss, "distributed_strategy": distributed_strategy}
