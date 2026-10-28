"""Typed runtime adapter for PyTorch device, memory, RNG, and compilation APIs."""

from __future__ import annotations

import contextlib
import logging
import types
from collections.abc import Generator
from typing import Protocol, runtime_checkable

from gemma_4_sql.type_hints import DTypeProtocol, TorchCausalLMProtocol, TorchTensorProtocol

logger = logging.getLogger(__name__)


@runtime_checkable
class _CompileCallableProtocol(Protocol):
    """Protocol for `torch.compile` preserving the compiled model type."""

    def __call__[ModelT](self, model: ModelT) -> ModelT:
        """Compile `model` and return a callable of the same type."""
        ...


@runtime_checkable
class TorchRuntimeProtocol(Protocol):
    """Protocol for PyTorch runtime device, memory, RNG, and compilation operations."""

    def get_device(self, hardware: str) -> str:
        """Resolve the target execution device (`'cpu'`, `'cuda'`, or `'mps'`)."""
        ...

    def resolve_dtype(self, dtype_name: str) -> DTypeProtocol | str | None:
        """Resolve a string dtype identifier on `torch`."""
        ...

    def compile_model[ModelT](self, model: ModelT) -> ModelT:
        """Compile `model` with `torch.compile` when available."""
        ...

    def synchronize(self, device: str) -> None:
        """Synchronize accelerator streams on `'cuda'` or `'mps'`."""
        ...

    def max_memory_mb(self, device: str, fallback_rss_mb: float) -> float:
        """Query peak allocated accelerator memory in MiB or return `fallback_rss_mb`."""
        ...

    def reset_peak_memory_stats(self, device: str) -> None:
        """Reset peak memory statistics on `'cuda'` devices."""
        ...

    def manual_seed(self, seed: int) -> None:
        """Seed the PyTorch random number generator."""
        ...


class TorchRuntimeAdapter:
    """Adapter encapsulating `torch.cuda`, `torch.backends.mps`, `torch.no_grad`, `torch.manual_seed`, and `torch.compile`."""

    def __init__(self, torch_module: types.ModuleType | None) -> None:
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
        with contextlib.suppress(AttributeError, TypeError):
            if self._torch.cuda is not None and self._torch.cuda.is_available():
                return "cuda"
        with contextlib.suppress(AttributeError, TypeError):
            if self._torch.backends is not None and self._torch.backends.mps is not None and self._torch.backends.mps.is_available():
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
        with contextlib.suppress(AttributeError):
            if dtype_name == "bfloat16" and self._torch.bfloat16 is not None:
                return self._torch.bfloat16
            if dtype_name == "float16" and self._torch.float16 is not None:
                return self._torch.float16
            if dtype_name == "float32" and self._torch.float32 is not None:
                return self._torch.float32
        with contextlib.suppress(AttributeError):
            return self._torch.float32
        return None

    def long_dtype(self) -> DTypeProtocol | str | None:
        """Provide `torch.long` if defined on the underlying `torch` module.

        Returns:
            `torch.long` dtype or `None`.

        """
        if self._torch is not None:
            with contextlib.suppress(AttributeError):
                return self._torch.long
        return None

    def compile_model[ModelT](self, model: ModelT) -> ModelT:
        """Compile `model` with `torch.compile` when available, falling back on error.

        Args:
            model: PyTorch model instance to compile.

        Returns:
            Compiled model or original `model` if compilation is unsupported or fails.

        """
        if self._torch is None:
            return model
        try:
            compile_fn = self._torch.compile
        except AttributeError:
            return model
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
        if device == "cuda":
            with contextlib.suppress(AttributeError, TypeError):
                self._torch.cuda.synchronize()
        elif device == "mps":
            with contextlib.suppress(AttributeError, TypeError):
                self._torch.mps.synchronize()

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
        if device == "cuda":
            with contextlib.suppress(AttributeError, TypeError):
                return float(self._torch.cuda.max_memory_allocated() / (1024 * 1024))
        elif device == "mps":
            with contextlib.suppress(AttributeError, TypeError):
                return float(self._torch.mps.driver_allocated_memory() / (1024 * 1024))
        return fallback_rss_mb

    def reset_peak_memory_stats(self, device: str) -> None:
        """Reset peak memory statistics on `'cuda'` devices.

        Args:
            device: Target device identifier string.

        """
        if self._torch is not None and device == "cuda":
            with contextlib.suppress(AttributeError, TypeError):
                self._torch.cuda.reset_peak_memory_stats()

    def manual_seed(self, seed: int) -> None:
        """Seed the PyTorch random number generator if `torch.manual_seed` is available.

        Args:
            seed: Integer random seed.

        """
        if self._torch is not None:
            with contextlib.suppress(AttributeError, TypeError):
                self._torch.manual_seed(seed)

    @contextlib.contextmanager
    def no_grad(self) -> Generator[None, None, None]:
        """Enter `torch.no_grad()` context if available on the runtime module.

        Yields:
            `None` inside the no-grad scope.

        """
        if self._torch is None:
            yield
            return
        try:
            ctx = self._torch.no_grad()
        except (AttributeError, TypeError):
            yield
            return
        with ctx:
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
        assert self._torch is not None
        tensor: TorchTensorProtocol = self._torch.randint(low, high, size, dtype=self.long_dtype())
        with contextlib.suppress(AttributeError, TypeError):
            tensor = tensor.to(device)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        return tensor


__all__ = [
    "TorchCausalLMProtocol",
    "TorchRuntimeAdapter",
    "TorchRuntimeProtocol",
]
