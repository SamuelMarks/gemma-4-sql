"""Shared PyTorch distributed setup, model wrapping, and process-group cleanup utilities."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol, optional_import
from gemma_4_sql.backends.pytorch.runtime_adapter import (
    TorchRuntimeAdapter,
    TorchXLACoreModelProtocol,
    TorchXLARuntimeProtocol,
)
from gemma_4_sql.type_hints import TorchOptimizerProtocol


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
    *,
    hardware: str = "cpu",
    strict_hardware: bool = False,
) -> tuple[bool, TorchDistModuleProtocol | None, str, int]:
    """Initialize PyTorch distributed process group and resolve local device (XLA TPU, CUDA, or CPU).

    Args:
        distributed_strategy: Strategy identifier (`'ddp'`, `'fsdp'`, `'xla'`, `'tpu'`, `'pjrt'`, or `'none'`).
        torch_module: Loaded `torch` module used to query CUDA availability and device count.
        hardware: Optional target hardware string (`'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).
        strict_hardware: Whether to raise `HardwareUnavailableError` if requested TPU/GPU is missing.

    Returns:
        A 4-tuple of `(is_distributed, dist_module, device_str, device_id)`.

    """
    if distributed_strategy in {"xla", "tpu", "pjrt"}:
        _ = optional_import("torch_xla.distributed.xla_backend")
        xla_dev = TorchRuntimeAdapter(None).get_device("tpu", strict_hardware=strict_hardware)
        raw_dist = optional_import("torch.distributed")
        dist_mod_xla: TorchDistModuleProtocol | None = (
            raw_dist if isinstance(raw_dist, TorchDistModuleProtocol) else None
        )
        if dist_mod_xla is not None and not dist_mod_xla.is_initialized():
            try:
                dist_mod_xla.init_process_group("xla")
            except (RuntimeError, ValueError):
                dist_mod_xla = None
        xla_rt = optional_import("torch_xla.runtime", TorchXLARuntimeProtocol)
        ordinal = xla_rt.global_ordinal() if xla_rt is not None else 0
        is_dist = (xla_rt is not None and xla_rt.world_size() > 1) or (
            dist_mod_xla is not None and dist_mod_xla.is_initialized()
        )
        return (is_dist, dist_mod_xla, xla_dev, ordinal)

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
        tpu_candidate = (
            TorchRuntimeAdapter(None).get_device(hardware, strict_hardware=strict_hardware)
            if hardware.strip().lower().startswith("tpu")
            else "cpu"
        )
        if tpu_candidate.startswith("xla"):
            return (False, None, tpu_candidate, 0)
        device_str = "cuda" if cuda_available else "cpu"
        torch_module.device(device_str)
    return (is_distributed, dist_mod, device_str, device_id)


def step_optimizer_xla_aware(
    optimizer: TorchOptimizerProtocol,
    device: str = "cpu",
) -> None:
    """Step `optimizer`, invoking `xm.optimizer_step(..., barrier=True)` and `sync()` on XLA TPUs.

    Args:
        optimizer: PyTorch optimizer instance.
        device: Active device string (`'xla:0'`, `'cuda'`, `'cpu'`).

    """
    if device.startswith("xla"):
        xm_mod = optional_import("torch_xla.core.xla_model", TorchXLACoreModelProtocol)
        if xm_mod is not None:
            xm_mod.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
        TorchRuntimeAdapter(None).synchronize(device)
        return
    optimizer.step()


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
    "step_optimizer_xla_aware",
    "wrap_model_distributed",
]
