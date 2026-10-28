"""MLX-specific benchmarking pipeline with native Metal profiling."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Sequence

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    ForwardOutput,
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    TokenizerProtocol,
)

logger = logging.getLogger(__name__)

mx = optional_import("mlx.core")
_raw_load = optional_attr("mlx_lm", "load")
load: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None] | MlxCausalLMProtocol] | None = (  # ty: ignore[invalid-assignment]
    _raw_load if callable(_raw_load) else None
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
    with contextlib.suppress(ValueError, TypeError, RuntimeError, AttributeError):
        device = mx.Device(mx.cpu) if target_device == "cpu" else mx.Device(mx.gpu)
        mx.set_default_device(device)

    loaded = load(model_name)
    model_obj: MlxCausalLMProtocol = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    return (model_obj, target_device)


def _sync_and_eval(tensors: MlxArrayProtocol | ForwardOutput | Sequence[MlxArrayProtocol | str] | str) -> None:
    """Force lazy evaluation and device synchronization for MLX arrays.

    Args:
        tensors: An MLX array or sequence of arrays to evaluate.

    """
    if mx is not None:
        with contextlib.suppress(AttributeError, TypeError):
            if isinstance(tensors, (tuple, list)):
                mx.eval(*tensors)
            else:
                mx.eval(tensors)


def _get_peak_memory_mb() -> float:
    """Retrieve peak memory allocated by MLX on Metal GPU in Megabytes.

    Checks `mx.metal.get_peak_memory()` and `mx.metal.get_active_memory()`.
    Returns `0.0` on CPU or when Metal is unavailable.

    Returns:
        Peak memory allocation in Megabytes (MB).

    """
    if mx is None:
        return 0.0

    with contextlib.suppress(AttributeError, TypeError):
        if mx.metal.is_available():
            with contextlib.suppress(AttributeError, TypeError):
                peak_bytes = mx.metal.get_peak_memory()
                if peak_bytes > 0:
                    return float(peak_bytes / (1024.0 * 1024.0))
            with contextlib.suppress(AttributeError, TypeError):
                active_bytes = mx.metal.get_active_memory()
                return float(active_bytes / (1024.0 * 1024.0))

    with contextlib.suppress(AttributeError, TypeError):
        peak = mx.get_peak_memory()
        return float(peak / (1024.0 * 1024.0))

    return 0.0


def _run_benchmark_pass(
    model: MlxCausalLMProtocol | None,
    batch_size: int,
    num_runs: int,
    prompt_len: int = 32,
    decode_tokens: int = 16,
    mode: str = "e2e",
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
        mode: Benchmark execution mode (`'prefill'`, `'decode'`, `'generate'`, or `'e2e'`).

    Returns:
        Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.

    """
    if mx is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    prefill_inputs: MlxArrayProtocol = mx.zeros((batch_size, prompt_len), dtype=mx.int32)

    if model is not None:
        out = model(prefill_inputs)
        _sync_and_eval(out)

    with contextlib.suppress(RuntimeError, ValueError, AttributeError, TypeError):
        mx.metal.reset_peak_memory()

    start_time = time.perf_counter()
    prefill_total_time = 0.0
    decode_total_time = 0.0
    run_prefill = mode in ("prefill", "e2e")
    run_decode = mode in ("decode", "generate", "e2e")

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
                step_seq: MlxArrayProtocol = mx.zeros(
                    (batch_size, prompt_len + step_idx + 1),
                    dtype=mx.int32,
                )
                dec_out = model(step_seq)
                _sync_and_eval(dec_out)
        t3 = time.perf_counter()
        decode_total_time += t3 - t2

    total_time = time.perf_counter() - start_time
    tokens_per_iter = (prompt_len if run_prefill else 0) + (decode_tokens if run_decode else 0)
    total_tokens = batch_size * max(1, tokens_per_iter) * num_runs

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
        """Execute benchmark pass on loaded MLX model."""
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
