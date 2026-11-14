"""Typed runtime adapter for PyTorch device, memory, RNG, and compilation APIs."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator, Mapping, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_benchmark import get_current_rss_mb, normalize_hardware_family
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import HardwareUnavailableError
from gemma_4_sql.type_hints import (
    DTypeProtocol,
    TorchCausalLMProtocol,
    TorchOptimizerProtocol,
    TorchTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _CompileCallableProtocol(Protocol):
    """Protocol for `torch.compile` preserving the compiled model type with optional `backend`."""

    def __call__[ModelT](self, model: ModelT, *, backend: str = "inductor") -> ModelT:
        """Compile `model` and return a callable of the same type.

        Args:
            model: Model instance to compile.
            backend: Optional compiler backend identifier (`'inductor'`, `'openxla'`).

        Returns:
            Compiled model instance.

        """
        ...


@runtime_checkable
class _CudaDeviceCountProtocol(Protocol):
    """Protocol for `torch.cuda` exposing `device_count()`."""

    def device_count(self) -> int:
        """Return count of visible CUDA devices.

        Returns:
            Visible CUDA device count.

        """
        ...


@runtime_checkable
class XLADeviceDescriptorProtocol(Protocol):
    """Structural protocol for `torch.device` XLA descriptors exposing `.type`."""

    @property
    def type(self) -> str:
        """Provide the device type string (`'xla'`)."""
        ...


@runtime_checkable
class TorchXLAModuleProtocol(Protocol):
    """Protocol for top-level `torch_xla` PJRT device, synchronization, and device-list APIs."""

    def device(self) -> XLADeviceDescriptorProtocol | str:
        """Return the active PJRT XLA device descriptor or string.

        Returns:
            Active XLA device descriptor or `'xla:0'` string.

        """
        ...

    def sync(self) -> None:
        """Flush and synchronize pending lazy XLA operations."""
        ...

    def devices(self) -> Sequence[XLADeviceDescriptorProtocol | str]:
        """Return all visible XLA devices on the current worker.

        Returns:
            Sequence of visible XLA device descriptors or strings.

        """
        ...


@runtime_checkable
class TorchXLACoreModelProtocol(Protocol):
    """Protocol for `torch_xla.core.xla_model` (`xm`) TPU runtime helpers."""

    def xla_device(self) -> XLADeviceDescriptorProtocol | str:
        """Return the default XLA device for the current process.

        Returns:
            Default XLA device descriptor or string.

        """
        ...

    def get_xla_supported_devices(self, devkind: str | None = None) -> Sequence[str]:
        """Return supported XLA device strings filtered by `devkind` (e.g., `'TPU'`).

        Args:
            devkind: Optional device kind filter (`'TPU'`, `'CPU'`, `'GPU'`).

        Returns:
            Sequence of supported XLA device strings.

        """
        ...

    def mark_step(self) -> None:
        """Mark an XLA execution step boundary."""
        ...

    def wait_device_ops(self) -> None:
        """Block until pending XLA device operations complete."""
        ...

    def get_memory_info(self, device: XLADeviceDescriptorProtocol | str, /) -> Mapping[str, int | float]:
        """Query XLA device HBM memory telemetry (`bytes_used`, `peak_bytes_used`, `bytes_limit`).

        Args:
            device: Target XLA device descriptor or string.

        Returns:
            Mapping of memory metric names to byte counts.

        """
        ...

    def optimizer_step(
        self,
        optimizer: TorchOptimizerProtocol,
        *,
        barrier: bool = False,
    ) -> None:
        """Perform cross-replica gradient reduction, `optimizer.step()`, and `mark_step()`.

        Args:
            optimizer: PyTorch optimizer instance.
            barrier: Whether to execute an XLA step barrier after stepping.

        """
        ...


@runtime_checkable
class TorchXLARuntimeProtocol(Protocol):
    """Protocol for `torch_xla.runtime` multi-host PJRT ordinal queries."""

    def global_ordinal(self) -> int:
        """Return global ordinal of the current process across the TPU slice.

        Returns:
            Zero-based global ordinal.

        """
        ...

    def world_size(self) -> int:
        """Return total PJRT world size across all TPU workers.

        Returns:
            Total process count.

        """
        ...

    def local_ordinal(self) -> int:
        """Return host-local ordinal on the current TPU VM worker.

        Returns:
            Zero-based local ordinal.

        """
        ...


@runtime_checkable
class _CudaNamespaceProtocol(Protocol):
    """Protocol for `torch.cuda` runtime methods."""

    def is_available(self) -> bool:
        """Return whether CUDA devices are available.

        Returns:
            `True` when CUDA is available.

        """
        ...

    def synchronize(self) -> None:
        """Synchronize active CUDA streams."""
        ...

    def max_memory_allocated(self) -> int | float:
        """Return peak allocated CUDA memory in bytes.

        Returns:
            Peak allocated CUDA bytes.

        """
        ...

    def reset_peak_memory_stats(self) -> None:
        """Reset peak CUDA memory statistics."""
        ...


@runtime_checkable
class _CudaAvailableOnlyProtocol(Protocol):
    """Protocol for minimal `torch.cuda` stubs only exposing `is_available()`."""

    def is_available(self) -> bool:
        """Return whether CUDA devices are available.

        Returns:
            `True` when CUDA is available.

        """
        ...


@runtime_checkable
class _TorchWithCudaProtocol(Protocol):
    """Protocol for `torch` module exposing `.cuda`."""

    @property
    def cuda(self) -> _CudaNamespaceProtocol | _CudaAvailableOnlyProtocol | None:
        """Provide the `torch.cuda` namespace.

        Returns:
            `torch.cuda` namespace or `None`.

        """
        ...


@runtime_checkable
class _MpsAvailabilityProtocol(Protocol):
    """Protocol for `torch.backends.mps` exposing `is_available()`."""

    def is_available(self) -> bool:
        """Return whether Apple MPS is available.

        Returns:
            `True` when MPS is available.

        """
        ...


@runtime_checkable
class _BackendsWithMpsProtocol(Protocol):
    """Protocol for `torch.backends` exposing `.mps`."""

    @property
    def mps(self) -> _MpsAvailabilityProtocol | None:
        """Provide `torch.backends.mps`.

        Returns:
            MPS backend object or `None`.

        """
        ...


@runtime_checkable
class _TorchWithBackendsProtocol(Protocol):
    """Protocol for `torch` module exposing `.backends`."""

    @property
    def backends(self) -> _BackendsWithMpsProtocol | None:
        """Provide `torch.backends`.

        Returns:
            `torch.backends` namespace or `None`.

        """
        ...


@runtime_checkable
class _MpsRuntimeNamespaceProtocol(Protocol):
    """Protocol for `torch.mps` exposing `synchronize` or `driver_allocated_memory`."""

    def synchronize(self) -> None:
        """Synchronize Apple MPS streams."""
        ...

    def driver_allocated_memory(self) -> int | float:
        """Return allocated MPS driver memory in bytes.

        Returns:
            Allocated MPS memory in bytes.

        """
        ...


@runtime_checkable
class _MpsSyncOnlyProtocol(Protocol):
    """Protocol for `torch.mps` exposing `synchronize()`."""

    def synchronize(self) -> None:
        """Synchronize Apple MPS streams."""
        ...


@runtime_checkable
class _MpsMemOnlyProtocol(Protocol):
    """Protocol for `torch.mps` exposing `driver_allocated_memory()`."""

    def driver_allocated_memory(self) -> int | float:
        """Return allocated MPS driver memory in bytes.

        Returns:
            Allocated MPS memory in bytes.

        """
        ...


@runtime_checkable
class _TorchWithMpsProtocol(Protocol):
    """Protocol for `torch` module exposing `.mps`."""

    @property
    def mps(self) -> _MpsRuntimeNamespaceProtocol | _MpsSyncOnlyProtocol | _MpsMemOnlyProtocol | None:
        """Provide `torch.mps`.

        Returns:
            `torch.mps` namespace or `None`.

        """
        ...


@runtime_checkable
class _TorchBFloat16Protocol(Protocol):
    """Protocol for `torch` exposing `.bfloat16`."""

    @property
    def bfloat16(self) -> DTypeProtocol | str | None:
        """Provide `torch.bfloat16`."""
        ...


@runtime_checkable
class _TorchFloat16Protocol(Protocol):
    """Protocol for `torch` exposing `.float16`."""

    @property
    def float16(self) -> DTypeProtocol | str | None:
        """Provide `torch.float16`."""
        ...


@runtime_checkable
class _TorchFloat32Protocol(Protocol):
    """Protocol for `torch` exposing `.float32`."""

    @property
    def float32(self) -> DTypeProtocol | str | None:
        """Provide `torch.float32`."""
        ...


@runtime_checkable
class _TorchLongProtocol(Protocol):
    """Protocol for `torch` exposing `.long`."""

    @property
    def long(self) -> DTypeProtocol | str | None:
        """Provide `torch.long`."""
        ...


@runtime_checkable
class _TorchCompileOwnerProtocol(Protocol):
    """Protocol for `torch` exposing `.compile`."""

    @property
    def compile(self) -> _CompileCallableProtocol | None:
        """Provide `torch.compile`."""
        ...


@runtime_checkable
class _TorchManualSeedProtocol(Protocol):
    """Protocol for `torch` exposing `manual_seed(seed)`."""

    def manual_seed(self, seed: int, /) -> None:
        """Seed the PyTorch RNG.

        Args:
            seed: Integer seed value.

        """
        ...


@runtime_checkable
class _TorchNoGradProtocol(Protocol):
    """Protocol for `torch` exposing `no_grad()`."""

    def no_grad(self) -> contextlib.AbstractContextManager[None]:
        """Return a no-grad context manager.

        Returns:
            Context manager disabling autograd tracking.

        """
        ...


@runtime_checkable
class TorchRandintProtocol(Protocol):
    """Protocol for `torch` exposing `randint(low, high, size, dtype=...)`."""

    def randint(
        self,
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        dtype: DTypeProtocol | str | None = None,
    ) -> TorchTensorProtocol:
        """Create a random integer tensor.

        Args:
            low: Inclusive lower bound.
            high: Exclusive upper bound.
            size: Tensor shape tuple.
            dtype: Optional integer dtype.

        Returns:
            Random integer tensor.

        """
        ...


type TorchModuleInput = (
    TorchRandintProtocol
    | _TorchWithCudaProtocol
    | _TorchWithBackendsProtocol
    | _TorchWithMpsProtocol
    | _TorchBFloat16Protocol
    | _TorchFloat16Protocol
    | _TorchFloat32Protocol
    | _TorchLongProtocol
    | _TorchCompileOwnerProtocol
    | _TorchManualSeedProtocol
    | _TorchNoGradProtocol
)


@runtime_checkable
class TorchRuntimeProtocol(Protocol):
    """Protocol for PyTorch runtime device, memory, RNG, and compilation operations."""

    def get_device(self, hardware: str) -> str:
        """Resolve the target execution device (`'cpu'`, `'cuda'`, or `'mps'`).

        Args:
            hardware: Requested hardware string.

        Returns:
            Resolved device string.

        """
        ...

    def resolve_dtype(self, dtype_name: str) -> DTypeProtocol | str | None:
        """Resolve a string dtype identifier on `torch`.

        Args:
            dtype_name: Requested dtype string.

        Returns:
            Resolved dtype descriptor or `None`.

        """
        ...

    def compile_model[ModelT](self, model: ModelT) -> ModelT:
        """Compile `model` with `torch.compile` when available.

        Args:
            model: Model to compile.

        Returns:
            Compiled or original model.

        """
        ...

    def synchronize(self, device: str) -> None:
        """Synchronize accelerator streams on `'cuda'` or `'mps'`.

        Args:
            device: Active device string.

        """
        ...

    def max_memory_mb(self, device: str, fallback_rss_mb: float) -> float:
        """Query peak allocated accelerator memory in MiB or return `fallback_rss_mb`.

        Args:
            device: Active device string.
            fallback_rss_mb: Fallback RSS memory in MiB.

        Returns:
            Peak allocated memory in MiB.

        """
        ...

    def reset_peak_memory_stats(self, device: str) -> None:
        """Reset peak memory statistics on `'cuda'` devices.

        Args:
            device: Active device string.

        """
        ...

    def manual_seed(self, seed: int) -> None:
        """Seed the PyTorch random number generator.

        Args:
            seed: Integer RNG seed.

        """
        ...


class TorchRuntimeAdapter:
    """Adapter encapsulating `torch_xla` TPU, `torch.cuda`, `torch.backends.mps`, RNG, and `compile`."""

    def __init__(
        self,
        torch_module: TorchModuleInput | None,
        *,
        xla_module: TorchXLAModuleProtocol | None = None,
        xm_module: TorchXLACoreModelProtocol | None = None,
    ) -> None:
        """Initialize the runtime adapter around `torch` and optional `torch_xla` modules.

        Args:
            torch_module: Loaded `torch` module or test double.
            xla_module: Optional `torch_xla` module or typed test double.
            xm_module: Optional `torch_xla.core.xla_model` module or typed test double.

        """
        self._torch = torch_module
        self._xla = xla_module
        self._xm = xm_module

    def _resolve_xla_modules(
        self,
    ) -> tuple[TorchXLAModuleProtocol | None, TorchXLACoreModelProtocol | None]:
        """Lazily resolve `torch_xla` and `torch_xla.core.xla_model` modules when available.

        Returns:
            2-tuple of `(xla_module, xm_module)`.

        """
        xla_mod = self._xla if self._xla is not None else optional_import("torch_xla", TorchXLAModuleProtocol)
        xm_mod = (
            self._xm if self._xm is not None else optional_import("torch_xla.core.xla_model", TorchXLACoreModelProtocol)
        )
        return (xla_mod, xm_mod)

    def get_device(self, hardware: str, *, strict_hardware: bool = False) -> str:
        """Resolve the target execution device (`'xla:0'`, `'cuda'`, `'mps'`, or `'cpu'`).

        Args:
            hardware: Requested hardware string (`'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cuda'`, `'mps'`, `'cpu'`).
            strict_hardware: If `True`, raise `HardwareUnavailableError` when TPU/GPU falls back to CPU.

        Returns:
            Resolved PyTorch device identifier string.

        Raises:
            HardwareUnavailableError: When `strict_hardware=True` and the requested TPU or GPU is unavailable.

        """
        hw_family = normalize_hardware_family(hardware)
        if hw_family == "cpu":
            return "cpu"
        if hw_family == "tpu":
            xla_mod, xm_mod = self._resolve_xla_modules()
            if xm_mod is not None:
                tpu_devs = xm_mod.get_xla_supported_devices("TPU")
                if len(tpu_devs) > 0:
                    return str(xla_mod.device()) if xla_mod is not None else str(xm_mod.xla_device())
            if xla_mod is not None and len(xla_mod.devices()) > 0:
                return str(xla_mod.device())
            if strict_hardware:
                msg = f"PyTorch/XLA TPU device unavailable for requested hardware '{hardware}'."
                raise HardwareUnavailableError(msg, hardware=hardware, backend="pytorch")
            return "cpu"
        if self._torch is None:
            if strict_hardware and hw_family == "gpu":
                msg = f"PyTorch GPU runtime unavailable for requested hardware '{hardware}'."
                raise HardwareUnavailableError(msg, hardware=hardware, backend="pytorch")
            return "cpu"
        if isinstance(self._torch, _TorchWithCudaProtocol):
            cuda_ns = self._torch.cuda
            if (
                isinstance(cuda_ns, _CudaAvailableOnlyProtocol)
                and callable(cuda_ns.is_available)
                and cuda_ns.is_available()
            ):
                return "cuda"
        if isinstance(self._torch, _TorchWithBackendsProtocol):
            backends_ns = self._torch.backends
            if isinstance(backends_ns, _BackendsWithMpsProtocol):
                mps_ns = backends_ns.mps
                if (
                    isinstance(mps_ns, _MpsAvailabilityProtocol)
                    and callable(mps_ns.is_available)
                    and mps_ns.is_available()
                ):
                    return "mps"
        if strict_hardware and hw_family == "gpu":
            msg = f"CUDA GPU device unavailable for requested hardware '{hardware}'."
            raise HardwareUnavailableError(msg, hardware=hardware, backend="pytorch")
        return "cpu"

    def get_device_count(self, hardware: str) -> int:
        """Return count of participating devices for `hardware`.

        Args:
            hardware: Target hardware identifier.

        Returns:
            Active device count (>= 1).

        """
        hw_family = normalize_hardware_family(hardware)
        if hw_family == "tpu":
            xla_mod, xm_mod = self._resolve_xla_modules()
            if xm_mod is not None:
                supported = xm_mod.get_xla_supported_devices("TPU")
                if len(supported) > 0:
                    return len(supported)
            if xla_mod is not None:
                devs = xla_mod.devices()
                if len(devs) > 0:
                    return len(devs)
            return 1
        if hw_family == "gpu" and isinstance(self._torch, _TorchWithCudaProtocol):
            cuda_ns = self._torch.cuda
            if (
                isinstance(cuda_ns, _CudaAvailableOnlyProtocol)
                and cuda_ns.is_available()
                and isinstance(cuda_ns, _CudaDeviceCountProtocol)
            ):
                return max(1, int(cuda_ns.device_count()))
        return 1

    def resolve_dtype(self, dtype_name: str) -> DTypeProtocol | str | None:
        """Resolve a string dtype identifier (`'bfloat16'`, `'float16'`, `'float32'`) on `torch`.

        Args:
            dtype_name: Name of the requested floating-point dtype.

        Returns:
            Resolved `torch.dtype` object or `None`.

        """
        if self._torch is None:
            return None
        if (
            dtype_name == "bfloat16"
            and isinstance(self._torch, _TorchBFloat16Protocol)
            and self._torch.bfloat16 is not None
        ):
            return self._torch.bfloat16
        if (
            dtype_name == "float16"
            and isinstance(self._torch, _TorchFloat16Protocol)
            and self._torch.float16 is not None
        ):
            return self._torch.float16
        if isinstance(self._torch, _TorchFloat32Protocol) and self._torch.float32 is not None:
            return self._torch.float32
        return None

    def long_dtype(self) -> DTypeProtocol | str | None:
        """Provide `torch.long` if defined on the underlying `torch` module.

        Returns:
            `torch.long` dtype or `None`.

        """
        if isinstance(self._torch, _TorchLongProtocol):
            return self._torch.long
        return None

    def compile_model[ModelT](self, model: ModelT, *, device: str = "cpu") -> ModelT:
        """Compile `model` with `torch.compile` (using `backend='openxla'` on XLA TPUs).

        Args:
            model: PyTorch model instance to compile.
            device: Resolved target device string (`'xla:0'`, `'cuda'`, `'cpu'`).

        Returns:
            Compiled model or original `model` if compilation is unsupported or fails.

        """
        if not isinstance(self._torch, _TorchCompileOwnerProtocol):
            return model
        compile_fn = self._torch.compile
        if not isinstance(compile_fn, _CompileCallableProtocol):
            return model
        try:
            if device.startswith("xla"):
                try:
                    return compile_fn(model, backend="openxla")
                except TypeError:
                    return compile_fn(model)
            return compile_fn(model)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("torch.compile failed: %s", e)
            return model

    maybe_compile = compile_model

    def synchronize(self, device: str) -> None:
        """Synchronize accelerator execution on `'xla:*'`, `'cuda'`, or `'mps'`.

        Args:
            device: Target device identifier string.

        """
        if device.startswith("xla"):
            xla_mod, xm_mod = self._resolve_xla_modules()
            if xla_mod is not None:
                xla_mod.sync()
            elif xm_mod is not None:
                xm_mod.mark_step()
                xm_mod.wait_device_ops()
            return
        if self._torch is None:
            return
        if device.startswith("cuda") and isinstance(self._torch, _TorchWithCudaProtocol):
            cuda_ns = self._torch.cuda
            if isinstance(cuda_ns, _CudaNamespaceProtocol) and callable(cuda_ns.synchronize):
                cuda_ns.synchronize()
        elif device == "mps" and isinstance(self._torch, _TorchWithMpsProtocol):
            mps_ns = self._torch.mps
            if isinstance(mps_ns, _MpsSyncOnlyProtocol) and callable(mps_ns.synchronize):
                mps_ns.synchronize()

    def max_memory_mb(self, device: str, fallback_rss_mb: float) -> float:
        """Query peak allocated accelerator HBM/VRAM in MiB or return `fallback_rss_mb`.

        Args:
            device: Target device identifier string (`'xla:0'`, `'cuda'`, `'mps'`, or `'cpu'`).
            fallback_rss_mb: Process RSS memory in MiB used when accelerator stats are unavailable.

        Returns:
            Peak allocated memory in MiB.

        """
        if device.startswith("xla"):
            _, xm_mod = self._resolve_xla_modules()
            if xm_mod is not None:
                mem_info = xm_mod.get_memory_info(device)
                raw_bytes = mem_info.get("peak_bytes_used", mem_info.get("bytes_used"))
                if isinstance(raw_bytes, (int, float)) and raw_bytes > 0:
                    return float(raw_bytes) / (1024.0 * 1024.0)
            return fallback_rss_mb
        if self._torch is None:
            return fallback_rss_mb
        if device.startswith("cuda") and isinstance(self._torch, _TorchWithCudaProtocol):
            cuda_ns = self._torch.cuda
            if isinstance(cuda_ns, _CudaNamespaceProtocol) and callable(cuda_ns.max_memory_allocated):
                return float(cuda_ns.max_memory_allocated()) / (1024.0 * 1024.0)
        elif device == "mps" and isinstance(self._torch, _TorchWithMpsProtocol):
            mps_ns = self._torch.mps
            if isinstance(mps_ns, _MpsMemOnlyProtocol) and callable(mps_ns.driver_allocated_memory):
                return float(mps_ns.driver_allocated_memory()) / (1024.0 * 1024.0)
        return fallback_rss_mb

    def get_memory_mb(self, device: str) -> float:
        """Query peak allocated memory in MiB on `device`, falling back to current process RSS.

        Args:
            device: Target device identifier string.

        Returns:
            Memory usage in MiB.

        """
        return self.max_memory_mb(device, get_current_rss_mb())

    def reset_peak_memory_stats(self, device: str) -> None:
        """Reset peak memory statistics on `'cuda'` devices (no-op on XLA/CPU/MPS).

        Args:
            device: Target device identifier string.

        """
        if device.startswith("cuda") and isinstance(self._torch, _TorchWithCudaProtocol):
            cuda_ns = self._torch.cuda
            if isinstance(cuda_ns, _CudaNamespaceProtocol) and callable(cuda_ns.reset_peak_memory_stats):
                cuda_ns.reset_peak_memory_stats()

    def manual_seed(self, seed: int) -> None:
        """Seed the PyTorch random number generator if `torch.manual_seed` is available.

        Args:
            seed: Integer random seed.

        """
        if isinstance(self._torch, _TorchManualSeedProtocol) and callable(self._torch.manual_seed):
            self._torch.manual_seed(seed)

    @contextlib.contextmanager
    def no_grad(self) -> Generator[None, None, None]:
        """Enter `torch.no_grad()` context if available on the runtime module.

        Yields:
            `None` inside the no-grad scope.

        """
        if isinstance(self._torch, _TorchNoGradProtocol) and callable(self._torch.no_grad):
            with self._torch.no_grad():
                yield
            return
        yield

    def randint(self, low: int, high: int, size: tuple[int, ...], device: str) -> TorchTensorProtocol:
        """Create a random integer tensor on `device`.

        Args:
            low: Inclusive lower bound.
            high: Exclusive upper bound.
            size: Tensor shape tuple.
            device: Target device identifier string.

        Returns:
            Random integer tensor moved to `device` when supported.

        """
        assert isinstance(self._torch, TorchRandintProtocol)
        tensor: TorchTensorProtocol = self._torch.randint(low, high, size, dtype=self.long_dtype())
        try:
            return tensor.to(device)
        except (TypeError, AssertionError, RuntimeError):
            return tensor


__all__ = [
    "TorchCausalLMProtocol",
    "TorchModuleInput",
    "TorchRandintProtocol",
    "TorchRuntimeAdapter",
    "TorchRuntimeProtocol",
    "TorchXLACoreModelProtocol",
    "TorchXLAModuleProtocol",
    "TorchXLARuntimeProtocol",
    "XLADeviceDescriptorProtocol",
]
