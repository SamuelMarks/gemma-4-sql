"""Typed runtime adapter for PyTorch device, memory, RNG, and compilation APIs."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from typing import Protocol, runtime_checkable

from gemma_4_sql.type_hints import DTypeProtocol, TorchCausalLMProtocol, TorchTensorProtocol

logger = logging.getLogger(__name__)


@runtime_checkable
class _CompileCallableProtocol(Protocol):
    """Protocol for `torch.compile` preserving the compiled model type."""

    def __call__[ModelT](self, model: ModelT) -> ModelT:
        """Compile `model` and return a callable of the same type.

        Args:
            model: Model instance to compile.

        Returns:
            Compiled model instance.

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
    """Adapter encapsulating `torch.cuda`, `torch.backends.mps`, `no_grad`, `manual_seed`, and `compile`."""

    def __init__(self, torch_module: TorchModuleInput | None) -> None:
        """Initialize the runtime adapter around a loaded `torch` module.

        Args:
            torch_module: Loaded `torch` module or test double.

        """
        self._torch = torch_module

    def get_device(self, hardware: str) -> str:
        """Resolve the target execution device (`'cpu'`, `'cuda'`, or `'mps'`).

        Args:
            hardware: Requested hardware string (`'cpu'`, `'gpu'`, `'cuda'`, `'mps'`).

        Returns:
            Resolved PyTorch device identifier string.

        """
        if hardware == "cpu" or self._torch is None:
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
        return "cpu"

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

    def compile_model[ModelT](self, model: ModelT) -> ModelT:
        """Compile `model` with `torch.compile` when available, falling back on error.

        Args:
            model: PyTorch model instance to compile.

        Returns:
            Compiled model or original `model` if compilation is unsupported or fails.

        """
        if not isinstance(self._torch, _TorchCompileOwnerProtocol):
            return model
        compile_fn = self._torch.compile
        if not isinstance(compile_fn, _CompileCallableProtocol):
            return model
        try:
            return compile_fn(model)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("torch.compile failed: %s", e)
            return model

    def synchronize(self, device: str) -> None:
        """Synchronize accelerator streams on `'cuda'` or `'mps'`.

        Args:
            device: Target device identifier string.

        """
        if self._torch is None:
            return
        if device == "cuda" and isinstance(self._torch, _TorchWithCudaProtocol):
            cuda_ns = self._torch.cuda
            if isinstance(cuda_ns, _CudaNamespaceProtocol) and callable(cuda_ns.synchronize):
                cuda_ns.synchronize()
        elif device == "mps" and isinstance(self._torch, _TorchWithMpsProtocol):
            mps_ns = self._torch.mps
            if isinstance(mps_ns, _MpsSyncOnlyProtocol) and callable(mps_ns.synchronize):
                mps_ns.synchronize()

    def max_memory_mb(self, device: str, fallback_rss_mb: float) -> float:
        """Query peak allocated accelerator memory in MiB or return `fallback_rss_mb`.

        Args:
            device: Target device identifier string (`'cuda'`, `'mps'`, or `'cpu'`).
            fallback_rss_mb: Process RSS memory in MiB used when accelerator stats are unavailable.

        Returns:
            Peak allocated memory in MiB.

        """
        if self._torch is None:
            return fallback_rss_mb
        if device == "cuda" and isinstance(self._torch, _TorchWithCudaProtocol):
            cuda_ns = self._torch.cuda
            if isinstance(cuda_ns, _CudaNamespaceProtocol) and callable(cuda_ns.max_memory_allocated):
                return float(cuda_ns.max_memory_allocated()) / (1024.0 * 1024.0)
        elif device == "mps" and isinstance(self._torch, _TorchWithMpsProtocol):
            mps_ns = self._torch.mps
            if isinstance(mps_ns, _MpsMemOnlyProtocol) and callable(mps_ns.driver_allocated_memory):
                return float(mps_ns.driver_allocated_memory()) / (1024.0 * 1024.0)
        return fallback_rss_mb

    def reset_peak_memory_stats(self, device: str) -> None:
        """Reset peak memory statistics on `'cuda'` devices.

        Args:
            device: Target device identifier string.

        """
        if device == "cuda" and isinstance(self._torch, _TorchWithCudaProtocol):
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
]
