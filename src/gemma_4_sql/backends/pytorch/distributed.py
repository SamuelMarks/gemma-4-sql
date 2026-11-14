"""Shared PyTorch distributed setup, model wrapping, and process-group cleanup utilities."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol, optional_import


@runtime_checkable
class TorchDistModuleProtocol(Protocol):
    """Protocol for `torch.distributed` process-group lifecycle operations."""

    def is_initialized(self) -> bool:
        """Return whether the default distributed process group is initialized.

        Returns:
            `True` when the process group is initialized.

        """
        ...

    def init_process_group(self, backend: str, /) -> None:
        """Initialize the default distributed process group.

        Args:
            backend: Communication backend name (`'nccl'` or `'gloo'`).

        """
        ...

    def get_rank(self) -> int:
        """Return the rank of the current process in the default process group.

        Returns:
            Zero-based process rank.

        """
        ...

    def destroy_process_group(self) -> None:
        """Destroy the default distributed process group."""
        ...


@runtime_checkable
class TorchCudaAvailabilityProtocol(Protocol):
    """Protocol for `torch.cuda` exposing `is_available()`."""

    def is_available(self) -> bool:
        """Return whether CUDA devices are available.

        Returns:
            `True` when CUDA is available.

        """
        ...


@runtime_checkable
class TorchCudaRuntimeProtocol(TorchCudaAvailabilityProtocol, Protocol):
    """Protocol for `torch.cuda` runtime device queries and selection."""

    def device_count(self) -> int:
        """Return the number of visible CUDA devices.

        Returns:
            Visible CUDA device count.

        """
        ...

    def set_device(self, device: int, /) -> None:
        """Set the active CUDA device by integer ordinal.

        Args:
            device: Zero-based CUDA device index.

        """
        ...


@runtime_checkable
class TorchCudaHolderProtocol(Protocol):
    """Protocol for a `torch` module exposing `.cuda`."""

    @property
    def cuda(self) -> TorchCudaAvailabilityProtocol:
        """Provide `torch.cuda` availability interface."""
        ...


@runtime_checkable
class _TorchDeviceDescriptorProtocol(Protocol):
    """Structural protocol satisfied by `torch.device` instances."""

    @property
    def type(self) -> str:
        """Provide the device type string (`'cpu'`, `'cuda'`, etc.)."""
        ...


@runtime_checkable
class TorchDistributedSetupModuleProtocol(Protocol):
    """Protocol for a `torch` module exposing `.cuda` runtime controls and `.device(...)`."""

    @property
    def cuda(self) -> TorchCudaRuntimeProtocol:
        """Provide `torch.cuda` runtime interface."""
        ...

    def device(self, device_str: str, /) -> _TorchDeviceDescriptorProtocol | str:
        """Construct a PyTorch device descriptor from a device string.

        Args:
            device_str: Device identifier string such as `'cpu'` or `'cuda:0'`.

        Returns:
            Constructed device descriptor.

        """
        ...


@runtime_checkable
class _TorchDDPModuleProtocol(Protocol):
    """Protocol for `torch.nn.parallel` exposing `DistributedDataParallel`."""

    def DistributedDataParallel[ModelT](
        self,
        model: ModelT,
        *,
        device_ids: list[int] | None = None,
    ) -> ModelT:
        """Wrap `model` in `DistributedDataParallel`.

        Args:
            model: Base PyTorch model.
            device_ids: Optional list of local CUDA device ordinals.

        Returns:
            DDP-wrapped model.

        """
        ...


@runtime_checkable
class _TorchFSDPModuleProtocol(Protocol):
    """Protocol for `torch.distributed.fsdp` exposing `FullyShardedDataParallel`."""

    def FullyShardedDataParallel[ModelT](self, model: ModelT, /) -> ModelT:
        """Wrap `model` in `FullyShardedDataParallel`.

        Args:
            model: Base PyTorch model.

        Returns:
            FSDP-wrapped model.

        """
        ...


def setup_distributed(
    distributed_strategy: str,
    torch_module: TorchDistributedSetupModuleProtocol | ModuleNamespaceProtocol | None,
) -> tuple[bool, TorchDistModuleProtocol | None, str, int]:
    """Initialize PyTorch distributed process group and resolve local device.

    Args:
        distributed_strategy: Distributed strategy identifier (`'ddp'`, `'fsdp'`, or `'none'`).
        torch_module: Loaded `torch` module used to query CUDA availability and device count.

    Returns:
        A 4-tuple of `(is_distributed, dist_module, device_str, device_id)`.

    """
    assert isinstance(torch_module, TorchDistributedSetupModuleProtocol)
    is_distributed = distributed_strategy in {"ddp", "fsdp"}
    dist_mod: TorchDistModuleProtocol | None = None
    device_id = 0
    cuda_available = bool(torch_module.cuda.is_available())
    device_count = max(1, int(torch_module.cuda.device_count()))

    if is_distributed:
        raw_dist = optional_import("torch.distributed")
        assert isinstance(raw_dist, TorchDistModuleProtocol)
        dist_mod = raw_dist
        if not dist_mod.is_initialized():
            dist_mod.init_process_group("nccl" if cuda_available else "gloo")
        rank = dist_mod.get_rank()
        device_id = rank % device_count
        device_str = f"cuda:{device_id}" if cuda_available else "cpu"
        if cuda_available:
            torch_module.device(device_str)
            torch_module.cuda.set_device(device_id)
    else:
        device_str = "cuda" if cuda_available else "cpu"
        torch_module.device(device_str)
    return (is_distributed, dist_mod, device_str, device_id)


def wrap_model_distributed[ModelT](
    model: ModelT,
    distributed_strategy: str,
    device_id: int,
    torch_module: TorchCudaHolderProtocol | ModuleNamespaceProtocol | None = None,
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
    assert isinstance(torch_module, TorchCudaHolderProtocol)
    cuda_available = bool(torch_module.cuda.is_available())

    if distributed_strategy == "ddp":
        ddp_module = optional_import("torch.nn.parallel")
        assert isinstance(ddp_module, _TorchDDPModuleProtocol)
        return ddp_module.DistributedDataParallel(model, device_ids=[device_id] if cuda_available else None)
    if distributed_strategy == "fsdp":
        fsdp_module = optional_import("torch.distributed.fsdp")
        assert isinstance(fsdp_module, _TorchFSDPModuleProtocol)
        return fsdp_module.FullyShardedDataParallel(model)
    return model


def cleanup_distributed(dist: TorchDistModuleProtocol | None) -> None:
    """Destroy the active PyTorch distributed process group if initialized.

    Args:
        dist: `torch.distributed` module or test double.

    """
    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "TorchCudaAvailabilityProtocol",
    "TorchCudaHolderProtocol",
    "TorchCudaRuntimeProtocol",
    "TorchDistModuleProtocol",
    "TorchDistributedSetupModuleProtocol",
    "cleanup_distributed",
    "setup_distributed",
    "wrap_model_distributed",
]
