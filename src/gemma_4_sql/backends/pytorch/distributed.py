"""Shared PyTorch distributed setup, model wrapping, and process-group cleanup utilities."""

from __future__ import annotations

import types
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import optional_import


@runtime_checkable
class TorchDistModuleProtocol(Protocol):
    """Protocol for `torch.distributed` process-group lifecycle operations."""

    def is_initialized(self) -> bool:
        """Return whether the default distributed process group is initialized."""
        ...

    def init_process_group(self, backend: str, /) -> None:
        """Initialize the default distributed process group."""
        ...

    def get_rank(self) -> int:
        """Return the rank of the current process in the default process group."""
        ...

    def destroy_process_group(self) -> None:
        """Destroy the default distributed process group."""
        ...


def setup_distributed(
    distributed_strategy: str,
    torch_module: types.ModuleType | None,
) -> tuple[bool, TorchDistModuleProtocol | None, str, int]:
    """Initialize PyTorch distributed process group and resolve local device.

    Args:
        distributed_strategy: Distributed strategy identifier (`'ddp'`, `'fsdp'`, or `'none'`).
        torch_module: Loaded `torch` module used to query CUDA availability and device count.

    Returns:
        A 4-tuple of `(is_distributed, dist_module, device_str, device_id)`.

    """
    assert torch_module is not None
    is_distributed = distributed_strategy in {"ddp", "fsdp"}
    dist_mod: TorchDistModuleProtocol | None = None
    device_id = 0
    cuda_available = bool(torch_module.cuda.is_available())
    device_count = max(1, int(torch_module.cuda.device_count()))

    if is_distributed:
        import torch.distributed as dist

        dist_mod = dist  # ty: ignore[invalid-assignment]
        if not dist_mod.is_initialized():  # ty: ignore[unresolved-attribute]
            dist_mod.init_process_group("nccl" if cuda_available else "gloo")  # ty: ignore[unresolved-attribute]
        rank = dist_mod.get_rank()  # ty: ignore[unresolved-attribute]
        device_id = rank % device_count
        device_str = f"cuda:{device_id}" if cuda_available else "cpu"
        if cuda_available:
            torch_module.cuda.set_device(torch_module.device(device_str))
    else:
        device_str = "cuda" if cuda_available else "cpu"
        torch_module.device(device_str)
    return (is_distributed, dist_mod, device_str, device_id)


def wrap_model_distributed[ModelT](
    model: ModelT,
    distributed_strategy: str,
    device_id: int,
    torch_module: types.ModuleType | None = None,
) -> ModelT:
    """Wrap `model` in `DistributedDataParallel` or `FullyShardedDataParallel` when configured.

    Args:
        model: Base PyTorch model instance.
        distributed_strategy: Strategy identifier (`'ddp'`, `'fsdp'`, or `'none'`).
        device_id: Local CUDA device ordinal.
        torch_module: Loaded `torch` module used to check CUDA availability for DDP `device_ids`.

    Returns:
        Wrapped distributed model or original `model` when `distributed_strategy == 'none'`.

    """
    assert torch_module is not None
    cuda_available = bool(torch_module.cuda.is_available())

    if distributed_strategy == "ddp":
        ddp_module = optional_import("torch.nn.parallel")
        assert ddp_module is not None
        ddp_class = ddp_module.DistributedDataParallel
        wrapped_ddp: ModelT = ddp_class(model, device_ids=[device_id] if cuda_available else None)
        return wrapped_ddp
    if distributed_strategy == "fsdp":
        fsdp_module = optional_import("torch.distributed.fsdp")
        assert fsdp_module is not None
        fsdp_class = fsdp_module.FullyShardedDataParallel
        wrapped_fsdp: ModelT = fsdp_class(model)
        return wrapped_fsdp
    return model


def cleanup_distributed(dist: TorchDistModuleProtocol | None) -> None:
    """Destroy the active PyTorch distributed process group if initialized.

    Args:
        dist: `torch.distributed` module or test double.

    """
    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "TorchDistModuleProtocol",
    "cleanup_distributed",
    "setup_distributed",
    "wrap_model_distributed",
]
