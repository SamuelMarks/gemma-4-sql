"""MLX-specific benchmarking pipeline with native Metal profiling."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper
from gemma_4_sql.exceptions import DependencyMissingError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)

try:
    import mlx.core as _mx

    mx: Any = _mx
except (ImportError, AttributeError):
    mx = None

try:
    from mlx_lm import load as _load

    load: Any = _load
except (ImportError, AttributeError):
    load = None


def _load_mlx_model_and_device(model_name: str, hardware: str, *, test_mode: bool = False) -> tuple[Any, str]:
    """Load an MLX model and configure the target execution device.

    Args:
        model_name: The identifier or path of the target MLX model.
        hardware: Target hardware accelerator ('gpu', 'metal', or 'cpu').
        test_mode: Flag indicating whether execution is running in mock/test mode.

    Returns:
        Tuple of (model instance or None, active device string).

    Raises:
        DependencyMissingError: If MLX or mlx_lm is missing.
    """
    if test_mode:
        return (None, "cpu")

    if mx is None or load is None:
        raise DependencyMissingError("MLX dependencies (mlx.core and mlx_lm) are missing.")

    target_device = "cpu" if str(hardware).lower() == "cpu" else "gpu"
    if hasattr(mx, "set_default_device") and hasattr(mx, "Device"):
        try:
            device = mx.Device(mx.cpu) if target_device == "cpu" and hasattr(mx, "cpu") else mx.Device(getattr(mx, "gpu", 0))
            mx.set_default_device(device)
        except (ValueError, TypeError, RuntimeError, AttributeError):
            pass

    loaded = load(model_name)
    model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    return (model, target_device)


def _sync_and_eval(tensors: Any) -> None:
    """Force lazy evaluation and device synchronization for MLX arrays.

    Args:
        tensors: An MLX array or sequence of arrays to evaluate.
    """
    if mx is not None and hasattr(mx, "eval"):
        if isinstance(tensors, (tuple, list)):
            mx.eval(*tensors)
        else:
            mx.eval(tensors)


def _get_peak_memory_mb() -> float:
    """Retrieve peak memory allocated by MLX on Metal GPU in Megabytes.

    Checks mx.metal.get_peak_memory() and mx.metal.get_active_memory().
    Returns 0.0 on CPU or when Metal is unavailable.

    Returns:
        Peak memory allocation in Megabytes (MB).
    """
    if mx is None:
        return 0.0

    metal_mod = getattr(mx, "metal", None)
    if metal_mod is not None and hasattr(metal_mod, "is_available") and metal_mod.is_available():
        if hasattr(metal_mod, "get_peak_memory"):
            peak_bytes = metal_mod.get_peak_memory()
            if peak_bytes > 0:
                return float(peak_bytes / (1024.0 * 1024.0))
        if hasattr(metal_mod, "get_active_memory"):
            active_bytes = metal_mod.get_active_memory()
            return float(active_bytes / (1024.0 * 1024.0))

    if hasattr(mx, "get_peak_memory"):
        peak = mx.get_peak_memory()
        return float(peak / (1024.0 * 1024.0))

    return 0.0


def _run_benchmark_pass(
    model: Any,
    batch_size: int,
    num_runs: int,
    prompt_len: int = 32,
    decode_tokens: int = 16,
) -> tuple[float, float, float]:
    """Execute warm-up and timed benchmark passes using native MLX operations.

    Measures prefill latency, decode latency, overall token throughput,
    and peak memory usage via Apple Silicon Metal APIs.

    Args:
        model: Loaded MLX model instance with callable forward pass.
        batch_size: Number of parallel sequences to evaluate.
        num_runs: Number of timed benchmark iterations.
        prompt_len: Number of prompt tokens for prefill phase.
        decode_tokens: Number of autoregressive decode tokens to generate.

    Returns:
        Tuple of (tokens_per_sec, latency_ms, memory_mb).

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
    """
    if mx is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    prefill_inputs = mx.zeros((batch_size, prompt_len))

    # Warm-up pass
    if model is not None and callable(model):
        out = model(prefill_inputs)
        _sync_and_eval(out)

    # Reset peak memory before benchmark if supported
    metal_mod = getattr(mx, "metal", None)
    if metal_mod is not None and hasattr(metal_mod, "reset_peak_memory"):
        try:
            metal_mod.reset_peak_memory()
        except (RuntimeError, ValueError, AttributeError):
            pass

    # Timed benchmark loop using time.perf_counter()
    start_time = time.perf_counter()
    prefill_total_time = 0.0
    decode_total_time = 0.0

    for _ in range(num_runs):
        t0 = time.perf_counter()
        if model is not None and callable(model):
            out = model(prefill_inputs)
            _sync_and_eval(out)
        t1 = time.perf_counter()
        prefill_total_time += t1 - t0

        decode_step_input = prefill_inputs[:, :1]
        t2 = time.perf_counter()
        if model is not None and callable(model):
            for _ in range(decode_tokens):
                dec_out = model(decode_step_input)
                _sync_and_eval(dec_out)
        t3 = time.perf_counter()
        decode_total_time += t3 - t2

    total_time = time.perf_counter() - start_time
    total_tokens = batch_size * (prompt_len + decode_tokens) * num_runs

    tokens_per_sec = float(total_tokens / max(total_time, 1e-09))
    latency_ms = float((total_time * 1000.0) / max(1, num_runs))
    prefill_latency_ms = float((prefill_total_time * 1000.0) / max(1, num_runs))
    decode_latency_ms = float((decode_total_time * 1000.0) / max(1, num_runs * max(1, decode_tokens)))

    memory_mb = _get_peak_memory_mb()
    logger.info(
        "MLX Benchmark: %.2f tok/s, total latency: %.2f ms (prefill: %.2f ms, decode: %.2f ms/tok), peak memory: %.2f MB",
        tokens_per_sec,
        latency_ms,
        prefill_latency_ms,
        decode_latency_ms,
        memory_mb,
    )
    return (tokens_per_sec, latency_ms, memory_mb)


def benchmark_model(model_name: str, hardware: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:
    """Benchmark an MLX model on CPU or Apple Silicon Metal GPU.

    Args:
        model_name: The name or path of the model to benchmark.
        hardware: Target hardware for the benchmark ('gpu', 'metal', or 'cpu').
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional options like `num_runs`, `prompt_len`, `decode_tokens`.

    Returns:
        A dictionary containing benchmark metrics and status.
    """

    def _run() -> tuple[float, float, float]:
        """Execute benchmark pass on loaded MLX model."""
        (model, _device) = _load_mlx_model_and_device(model_name, hardware, test_mode=bool(kwargs.get("test_mode")))
        num_runs = int(str(kwargs.get("num_runs", 5)))
        prompt_len = int(str(kwargs.get("prompt_len", 32)))
        decode_tokens = int(str(kwargs.get("decode_tokens", 16)))
        return _run_benchmark_pass(
            model=model,
            batch_size=batch_size,
            num_runs=num_runs,
            prompt_len=prompt_len,
            decode_tokens=decode_tokens,
        )

    return run_benchmark_wrapper(
        backend_name="mlx",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=mx is None or load is None,
        missing_status="mocked_missing_mlx",
        benchmark_fn=_run,
    )
