"""MLX-specific benchmarking pipeline with native Metal profiling without contextlib.suppress."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql._types_models import MlxLMLoadProtocol
from gemma_4_sql.backends.common_benchmark import compute_throughput_and_latency_ms, run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    DTypeProtocol,
    ForwardOutput,
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _MlxSetDeviceProtocol(Protocol):
    """Protocol for `mlx.core` exposing `Device`, `cpu`, `gpu`, and `set_default_device(dev)`."""

    @property
    def cpu(self) -> int | str:
        """Provide `mx.cpu`."""
        ...

    @property
    def gpu(self) -> int | str:
        """Provide `mx.gpu`."""
        ...

    def Device(self, dev_type: int | str, /) -> int | str:
        """Construct an MLX `Device`."""
        ...

    def set_default_device(self, device: int | str, /) -> None:
        """Set active MLX default device."""
        ...


@runtime_checkable
class _MlxEvalProtocol(Protocol):
    """Protocol for `mlx.core` exposing `eval(*tensors)`."""

    def eval(self, *args: MlxArrayProtocol | ForwardOutput | Sequence[MlxArrayProtocol | str] | str) -> None:
        """Evaluate lazy MLX computation graphs."""
        ...


@runtime_checkable
class _MlxMetalAvailableProtocol(Protocol):
    """Protocol for `mx.metal` exposing `is_available()`."""

    def is_available(self) -> bool:
        """Return whether Apple Metal GPU is available."""
        ...


@runtime_checkable
class _MlxMetalPeakProtocol(Protocol):
    """Protocol for `mx.metal` exposing `get_peak_memory()`."""

    def get_peak_memory(self) -> int | float:
        """Return peak Metal memory in bytes."""
        ...


@runtime_checkable
class _MlxMetalActiveProtocol(Protocol):
    """Protocol for `mx.metal` exposing `get_active_memory()`."""

    def get_active_memory(self) -> int | float:
        """Return active Metal memory in bytes."""
        ...


@runtime_checkable
class _MlxMetalResetProtocol(Protocol):
    """Protocol for `mx.metal` exposing `reset_peak_memory()`."""

    def reset_peak_memory(self) -> None:
        """Reset peak Metal memory counter."""
        ...


@runtime_checkable
class _MlxWithMetalProtocol(Protocol):
    """Protocol for `mlx.core` exposing `.metal`."""

    @property
    def metal(self) -> _MlxMetalAvailableProtocol | _MlxMetalResetProtocol | None:
        """Provide `mx.metal` namespace."""
        ...


@runtime_checkable
class _MlxTopLevelPeakProtocol(Protocol):
    """Protocol for `mlx.core` exposing top-level `get_peak_memory()`."""

    def get_peak_memory(self) -> int | float:
        """Return peak MLX memory in bytes."""
        ...


@runtime_checkable
class _MlxBenchmarkCoreProtocol(Protocol):
    """Protocol for `mlx.core` exposing `int32` and `zeros`."""

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `mx.int32` dtype."""
        ...

    def zeros(self, shape: tuple[int, int], *, dtype: DTypeProtocol) -> MlxArrayProtocol:
        """Construct a zero-initialized 2D integer `MlxArrayProtocol`.

        Args:
            shape: `(batch_size, seq_len)` shape tuple.
            dtype: Target `mx.int32` dtype.

        Returns:
            Zero-initialized `MlxArrayProtocol`.

        """
        ...


mx: _MlxBenchmarkCoreProtocol | None = optional_import("mlx.core", _MlxBenchmarkCoreProtocol)
load: MlxLMLoadProtocol | None = import_optional_symbol(
    "mlx_lm",
    "load",
    MlxLMLoadProtocol,
)


def _load_mlx_model_and_device(model_name: str, hardware: str) -> tuple[MlxCausalLMProtocol, str]:
    """Load an MLX model and configure the target execution device.

    Args:
        model_name: The identifier or path of the target MLX model.
        hardware: Target hardware accelerator (`'gpu'`, `'metal'`, or `'cpu'`).

    Returns:
        Tuple of `(model instance, active device string)`.

    Raises:
        DependencyMissingError: If MLX or `mlx_lm` is missing.

    """
    if mx is None or load is None:
        raise DependencyMissingError("MLX dependencies (mlx.core and mlx_lm) are missing.")

    target_device = "cpu" if str(hardware).lower() == "cpu" else "gpu"
    if isinstance(mx, _MlxSetDeviceProtocol) and callable(mx.set_default_device):
        try:
            dev_arg = mx.Device(mx.cpu) if target_device == "cpu" else mx.Device(mx.gpu)
            mx.set_default_device(dev_arg)
        except (ValueError, TypeError, RuntimeError, AttributeError):
            pass

    loaded = load(model_name)
    model_obj: MlxCausalLMProtocol = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    return (model_obj, target_device)


def _sync_and_eval(tensors: MlxArrayProtocol | ForwardOutput | Sequence[MlxArrayProtocol | str] | str) -> None:
    """Force lazy evaluation and device synchronization for MLX arrays.

    Args:
        tensors: An MLX array or sequence of arrays to evaluate.

    """
    if isinstance(mx, _MlxEvalProtocol) and callable(mx.eval):
        if isinstance(tensors, (tuple, list)):
            mx.eval(*tensors)
        else:
            mx.eval(tensors)


def _get_peak_memory_mb() -> float:
    """Retrieve peak memory allocated by MLX on Metal GPU in Megabytes.

    Returns:
        Peak memory allocation in Megabytes (MB).

    """
    if mx is None:
        return 0.0

    if isinstance(mx, _MlxWithMetalProtocol):
        metal_ns = mx.metal
        if (
            isinstance(metal_ns, _MlxMetalAvailableProtocol)
            and callable(metal_ns.is_available)
            and metal_ns.is_available()
        ):
            if isinstance(metal_ns, _MlxMetalPeakProtocol) and callable(metal_ns.get_peak_memory):
                peak_bytes = float(metal_ns.get_peak_memory())
                if peak_bytes > 0.0:
                    return peak_bytes / (1024.0 * 1024.0)
            if isinstance(metal_ns, _MlxMetalActiveProtocol) and callable(metal_ns.get_active_memory):
                active_bytes = float(metal_ns.get_active_memory())
                return active_bytes / (1024.0 * 1024.0)

    if isinstance(mx, _MlxTopLevelPeakProtocol) and callable(mx.get_peak_memory):
        return float(mx.get_peak_memory()) / (1024.0 * 1024.0)

    return 0.0


def _run_benchmark_pass(
    model: MlxCausalLMProtocol | Callable[[MlxArrayProtocol], MlxArrayProtocol | str] | None,
    batch_size: int,
    num_runs: int,
    prompt_len: int = 32,
    decode_tokens: int = 16,
    mode: str = "e2e",
) -> tuple[float, float, float]:
    """Execute warm-up and timed benchmark passes using native MLX operations.

    Args:
        model: Loaded MLX model instance with callable forward pass.
        batch_size: Number of parallel sequences to evaluate.
        num_runs: Number of timed benchmark iterations.
        prompt_len: Number of prompt tokens for prefill phase.
        decode_tokens: Number of autoregressive decode tokens to generate.
        mode: Benchmark execution mode (`'prefill'`, `'decode'`, `'generate'`, or `'e2e'`).

    Returns:
        Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.

    """
    mx_mod = mx
    if mx_mod is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    prefill_inputs: MlxArrayProtocol = mx_mod.zeros((batch_size, prompt_len), dtype=mx_mod.int32)

    if model is not None:
        out = model(prefill_inputs)
        _sync_and_eval(out)

    metal_check = mx_mod
    if (
        isinstance(metal_check, _MlxWithMetalProtocol)
        and isinstance(metal_check.metal, _MlxMetalResetProtocol)
        and callable(metal_check.metal.reset_peak_memory)
    ):
        try:
            metal_check.metal.reset_peak_memory()
        except (RuntimeError, ValueError):
            pass

    start_time = time.perf_counter()
    prefill_total_time = 0.0
    decode_total_time = 0.0
    run_prefill = mode in {"prefill", "e2e"}
    run_decode = mode in {"decode", "generate", "e2e"}

    for _ in range(num_runs):
        t0 = time.perf_counter()
        if run_prefill and model is not None:
            out = model(prefill_inputs)
            _sync_and_eval(out)
        t1 = time.perf_counter()
        prefill_total_time += t1 - t0

        t2 = time.perf_counter()
        if run_decode and model is not None:
            for step_idx in range(decode_tokens):
                step_seq: MlxArrayProtocol = mx_mod.zeros(
                    (batch_size, prompt_len + step_idx + 1),
                    dtype=mx_mod.int32,
                )
                dec_out = model(step_seq)
                _sync_and_eval(dec_out)
        t3 = time.perf_counter()
        decode_total_time += t3 - t2

    total_time = time.perf_counter() - start_time
    tokens_per_iter = (prompt_len if run_prefill else 0) + (decode_tokens if run_decode else 0)
    tokens_per_sec, latency_ms = compute_throughput_and_latency_ms(
        total_time,
        tokens_per_iter=tokens_per_iter,
        batch_size=batch_size,
        num_runs=num_runs,
    )
    prefill_latency_ms = float((prefill_total_time * 1000.0) / max(1, num_runs))
    decode_latency_ms = float((decode_total_time * 1000.0) / max(1, num_runs * max(1, decode_tokens)))

    memory_mb = _get_peak_memory_mb()
    logger.info(
        "MLX Benchmark: %.2f tok/s, total latency: %.2f ms "
        "(prefill: %.2f ms, decode: %.2f ms/tok), peak memory: %.2f MB",
        tokens_per_sec,
        latency_ms,
        prefill_latency_ms,
        decode_latency_ms,
        memory_mb,
    )
    return (tokens_per_sec, latency_ms, memory_mb)


def benchmark_model(
    model_name: str,
    hardware: str = "apple-silicon",
    batch_size: int = 32,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark an MLX model on CPU or Apple Silicon Metal GPU.

    Args:
        model_name: The name or path of the model to benchmark.
        hardware: Target hardware for the benchmark (`'gpu'`, `'metal'`, or `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional options like `num_runs`, `prompt_len`, `decode_tokens`.

    Returns:
        A dictionary containing benchmark metrics and status.

    """

    def _run() -> tuple[float, float, float]:
        """Execute benchmark pass on loaded MLX model.

        Returns:
            Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

        """
        (model, _device) = _load_mlx_model_and_device(model_name, hardware)
        num_runs = int(str(kwargs.get("num_runs", 5)))
        prompt_len = int(str(kwargs.get("prompt_len", 32)))
        decode_tokens = int(str(kwargs.get("decode_tokens", kwargs.get("max_new_tokens", 16))))
        mode = str(kwargs.get("mode", "e2e"))
        return _run_benchmark_pass(
            model=model,
            batch_size=batch_size,
            num_runs=num_runs,
            prompt_len=prompt_len,
            decode_tokens=decode_tokens,
            mode=mode,
        )

    return run_benchmark_wrapper(
        backend_name="mlx",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=mx is None or load is None,
        benchmark_fn=_run,
    )
