"""Common benchmarking utilities."""

from __future__ import annotations

import logging
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass

from gemma_4_sql.exceptions import ConfigurationError, HardwareUnavailableError, StatusValue
from gemma_4_sql.type_hints import BenchmarkResultDict, LatencyStatsDict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetailedLoopMetrics:
    """Detailed timing breakdown returned by `execute_detailed_benchmark_loop`.

    Attributes:
        tokens_per_sec: Throughput across timed runs in tokens per second.
        latency_ms: Arithmetic mean iteration latency in milliseconds.
        compile_latency_ms: Cold compilation/warmup-0 wall-clock duration in milliseconds.
        latency_samples: Per-iteration synchronized wall-clock latencies in milliseconds.

    """

    tokens_per_sec: float
    latency_ms: float
    compile_latency_ms: float
    latency_samples: tuple[float, ...]


@dataclass(frozen=True)
class BenchmarkTelemetry:
    """Rich accelerator benchmark telemetry returned by backend `benchmark_fn` callables.

    Attributes:
        tokens_per_sec: Measured throughput in tokens per second.
        latency_ms: Mean iteration latency in milliseconds.
        memory_mb: Primary reported memory footprint in megabytes.
        resolved_device: Concrete runtime device identifier (`'tpu:0'`, `'xla:0'`, `'/TPU:0'`, `'cpu'`).
        device_count: Count of active accelerator devices participating in execution.
        compile_latency_ms: Cold XLA compilation duration recorded on warmup iteration 0 in milliseconds.
        peak_hbm_mb: Peak accelerator HBM allocation in megabytes when available.
        keras_backend: Active Keras 3 engine (`'jax'`, `'tensorflow'`, or `'torch'`) for Keras runs.
        latency_samples: Per-iteration synchronized latency samples in milliseconds.

    """

    tokens_per_sec: float
    latency_ms: float
    memory_mb: float
    resolved_device: str | None = None
    device_count: int | None = None
    compile_latency_ms: float | None = None
    peak_hbm_mb: float | None = None
    keras_backend: str | None = None
    latency_samples: tuple[float, ...] = ()


def normalize_hardware_family(hardware: str) -> str:
    """Normalize SKU-qualified hardware strings into canonical accelerator families.

    Maps `'tpu'`, `'tpu-v4-8'`, `'tpu-v5e'`, `'tpu-v5p'`, `'tpu-v6e'` to `'tpu'`; `'gpu'` and
    `'cuda'` variants to `'gpu'`; `'mps'` variants to `'mps'`; and all other labels to `'cpu'`.

    Args:
        hardware: Raw hardware target string.

    Returns:
        Canonical hardware family (`'tpu'`, `'gpu'`, `'mps'`, or `'cpu'`).

    """
    norm = hardware.strip().lower()
    if norm == "tpu" or norm.startswith(("tpu-", "tpu_", "tpu:")):
        return "tpu"
    if norm in {"gpu", "cuda"} or norm.startswith(("gpu-", "gpu_", "cuda:", "cuda-")):
        return "gpu"
    if norm == "mps" or norm.startswith("mps:"):
        return "mps"
    return "cpu"


def normalize_benchmark_mode(mode: str, *, backend: str | None = None) -> str:
    """Normalize benchmark mode aliases (`'decode'` / `'generate'`, `'prefill'`, `'end-to-end'`).

    Args:
        mode: Raw benchmark mode string.
        backend: Optional backend identifier attached to `ConfigurationError` on invalid modes.

    Returns:
        Canonical mode string (`'prefill'`, `'decode'`, or `'end-to-end'`).

    Raises:
        ConfigurationError: If `mode` is not a recognized benchmark mode.

    """
    norm = mode.strip().lower()
    if norm == "prefill":
        return "prefill"
    if norm in {"decode", "generate"}:
        return "decode"
    if norm in {"end-to-end", "end_to_end", "e2e"}:
        return "end-to-end"
    msg = f"Unsupported benchmark mode: {mode!r}. Expected 'prefill', 'decode', or 'end-to-end'."
    raise ConfigurationError(msg, backend=backend, operation="benchmark")


def execute_detailed_benchmark_loop(
    timed_step_fn: Callable[[], None],
    *,
    warmup_steps: int,
    num_runs: int,
    tokens_per_iter: int,
    batch_size: int,
    warmup_step_fn: Callable[[], None] | None = None,
    sync_fn: Callable[[], None] | None = None,
) -> DetailedLoopMetrics:
    """Execute synchronized warmup (recording cold compile latency) and per-step timed iterations.

    Args:
        timed_step_fn: Single timed iteration callable.
        warmup_steps: Number of untimed warmup iterations; iteration 0 records `compile_latency_ms`.
        num_runs: Number of timed iterations to profile.
        tokens_per_iter: Tokens processed per sequence per iteration.
        batch_size: Number of sequences per batch.
        warmup_step_fn: Optional warmup callable when warmup differs from `timed_step_fn`.
        sync_fn: Optional accelerator barrier executed after warmup and after every timed step.

    Returns:
        `DetailedLoopMetrics` containing throughput, mean latency, compile latency, and per-step samples.

    """
    active_warmup = warmup_step_fn if warmup_step_fn is not None else timed_step_fn
    compile_latency_ms = 0.0
    if warmup_steps > 0:
        compile_start = time.perf_counter()
        active_warmup()
        if sync_fn is not None:
            sync_fn()
        compile_latency_ms = float((time.perf_counter() - compile_start) * 1000.0)
        for _ in range(warmup_steps - 1):
            active_warmup()
        if warmup_steps > 1 and sync_fn is not None:
            sync_fn()

    effective_runs = max(1, num_runs)
    samples: list[float] = []
    for _ in range(effective_runs):
        step_start = time.perf_counter()
        timed_step_fn()
        if sync_fn is not None:
            sync_fn()
        samples.append(float((time.perf_counter() - step_start) * 1000.0))

    total_elapsed_s = sum(samples) / 1000.0
    tokens_per_sec, latency_ms = compute_throughput_and_latency_ms(
        total_elapsed_s,
        tokens_per_iter=tokens_per_iter,
        batch_size=batch_size,
        num_runs=effective_runs,
    )
    return DetailedLoopMetrics(
        tokens_per_sec=tokens_per_sec,
        latency_ms=latency_ms,
        compile_latency_ms=compile_latency_ms,
        latency_samples=tuple(samples),
    )


def compute_throughput_and_latency_ms(
    elapsed_s: float,
    *,
    tokens_per_iter: int,
    batch_size: int,
    num_runs: int,
) -> tuple[float, float]:
    """Compute `(tokens_per_sec, latency_ms)` from total wall-clock elapsed seconds.

    Args:
        elapsed_s: Total wall-clock seconds across `num_runs`.
        tokens_per_iter: Number of tokens processed per sequence per iteration.
        batch_size: Number of sequences per batch.
        num_runs: Number of timed benchmark iterations.

    Returns:
        Tuple of `(tokens_per_sec, latency_ms)`.

    """
    safe_elapsed = max(elapsed_s, 1e-09)
    latency_ms = float((elapsed_s * 1000.0) / max(1, num_runs))
    tokens_per_sec = float((max(1, tokens_per_iter) * batch_size * num_runs) / safe_elapsed)
    return (tokens_per_sec, latency_ms)


def execute_timed_benchmark_loop(
    timed_step_fn: Callable[[], None],
    *,
    warmup_steps: int,
    num_runs: int,
    tokens_per_iter: int,
    batch_size: int,
    warmup_step_fn: Callable[[], None] | None = None,
    post_warmup_fn: Callable[[], None] | None = None,
    post_runs_fn: Callable[[], None] | None = None,
    use_perf_counter: bool = False,
) -> tuple[float, float]:
    """Execute untimed warmup steps followed by timed benchmark runs and return `(tokens_per_sec, latency_ms)`.

    Args:
        timed_step_fn: Single timed iteration callable.
        warmup_steps: Number of untimed warmup iterations.
        num_runs: Number of timed benchmark iterations.
        tokens_per_iter: Number of tokens processed per sequence per timed run (`32` for prefill or `max_new_tokens`).
        batch_size: Number of sequences per batch.
        warmup_step_fn: Optional warmup step callable when warmup differs from `timed_step_fn`.
        post_warmup_fn: Optional synchronization barrier executed once after warmup completes.
        post_runs_fn: Optional synchronization barrier executed before stopping the timer.
        use_perf_counter: If `True`, sample `time.perf_counter()`, otherwise `time.time()`.

    Returns:
        Tuple of `(tokens_per_sec, latency_ms)`.

    """
    active_warmup = warmup_step_fn if warmup_step_fn is not None else timed_step_fn
    for _ in range(warmup_steps):
        active_warmup()
    if post_warmup_fn is not None:
        post_warmup_fn()

    start_time = time.perf_counter() if use_perf_counter else time.time()
    for _ in range(num_runs):
        timed_step_fn()
    if post_runs_fn is not None:
        post_runs_fn()
    end_time = time.perf_counter() if use_perf_counter else time.time()

    return compute_throughput_and_latency_ms(
        end_time - start_time,
        tokens_per_iter=tokens_per_iter,
        batch_size=batch_size,
        num_runs=num_runs,
    )


def get_current_rss_mb() -> float:
    """Provide current resident set size (RSS) memory in megabytes.

    Returns:
        Memory usage in MB.

    """
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except (ImportError, AttributeError):
        # Fallback to standard library resource module (ru_maxrss in KB on Linux, bytes on macOS)
        import sys

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(ru_maxrss / (1024 * 1024))
        return float(ru_maxrss / 1024)


def compute_latency_statistics(latencies_ms: list[float]) -> LatencyStatsDict:
    """Compute statistical percentiles and summaries for latency samples.

    Args:
        latencies_ms: List of observed latency measurements in milliseconds.

    Returns:
        Dictionary containing mean, median (p50), p90, p99, min, and max latencies.

    """
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    def _percentile(p: float) -> float:
        """Calculate the p-th percentile from sorted latencies.

        Args:
            p: Percentile value in `[0.0, 100.0]`.

        Returns:
            Interpolated index percentile latency in milliseconds.

        """
        idx = max(0, min(n - 1, round((p / 100.0) * (n - 1))))
        return float(sorted_lats[idx])

    return {
        "mean_ms": float(sum(sorted_lats) / n),
        "p50_ms": _percentile(50.0),
        "p90_ms": _percentile(90.0),
        "p99_ms": _percentile(99.0),
        "min_ms": float(sorted_lats[0]),
        "max_ms": float(sorted_lats[-1]),
    }


def run_benchmark_wrapper(
    backend_name: str,
    model_name: str,
    hardware: str,
    batch_size: int,
    benchmark_fn: Callable[[], tuple[float, float, float] | BenchmarkTelemetry],
    *,
    missing_deps: bool = False,
    raise_if_missing: bool = False,
    latency_samples: list[float] | None = None,
    raise_on_error: bool | None = None,
    strict_hardware: bool = False,
    **kwargs: StatusValue,
) -> BenchmarkResultDict:
    """Wrap benchmarking logic to unify exception handling, strict hardware checks, and telemetry formatting.

    Args:
        backend_name: The name of the backend.
        model_name: The name of the model.
        hardware: The target hardware.
        batch_size: The batch size.
        benchmark_fn: A callable returning `(tokens_per_sec, latency_ms, memory_mb)` or `BenchmarkTelemetry`.
        missing_deps: Whether dependencies are missing.
        raise_if_missing: If True, raises DependencyMissingError when missing_deps is True.
        latency_samples: Optional sequence of individual request latency samples for percentile stats.
        raise_on_error: If True, raises `BenchmarkError` when `benchmark_fn` fails.
        strict_hardware: If True, raises `HardwareUnavailableError` when TPU/GPU requests fall back to CPU.
        **kwargs: Additional keyword arguments inspected by `resolve_raise_on_error` and `strict_hardware`.

    Returns:
        A dictionary containing the benchmark results and optional accelerator telemetry.

    Raises:
        DependencyMissingError: If raise_if_missing is True and missing_deps is True, or if raised by `benchmark_fn`.
        HardwareUnavailableError: If `strict_hardware` is active and requested TPU/GPU hardware is unavailable.
        ConfigurationError: If raised by `benchmark_fn`.
        BenchmarkError: If `raise_on_error` is enabled and `benchmark_fn` raises an exception.

    """
    from gemma_4_sql.exceptions import (
        OPERATIONAL_EXCEPTIONS,
        BenchmarkError,
        DependencyMissingError,
        ExecutionStatus,
        resolve_raise_on_error,
    )

    enforce_strict_hw = strict_hardware or bool(kwargs.get("strict_hardware", False))
    hw_family = normalize_hardware_family(hardware)

    if missing_deps:
        if raise_if_missing:
            msg = f"Dependencies for {backend_name} benchmarking on {hardware} are missing."
            raise DependencyMissingError(msg)
        if enforce_strict_hw and hw_family in {"tpu", "gpu"}:
            msg = f"Requested hardware '{hardware}' is unavailable for backend '{backend_name}' (missing dependencies)."
            raise HardwareUnavailableError(
                msg,
                hardware=hardware,
                backend=backend_name,
                model_name=model_name,
                operation="benchmark",
            )

        return {
            "backend": backend_name,
            "model": model_name,
            "hardware": hardware,
            "batch_size": batch_size,
            "status": ExecutionStatus.BACKEND_UNAVAILABLE.value,
            "status_code": ExecutionStatus.BACKEND_UNAVAILABLE.value,
            "tokens_per_sec": 0.0,
            "latency_ms": 0.0,
            "memory_mb": 0.0,
        }

    logger.info(
        "Starting %s benchmark for %s on %s (batch size %d)", backend_name.upper(), model_name, hardware, batch_size
    )

    should_raise = resolve_raise_on_error(raise_on_error=raise_on_error, kwargs=dict(kwargs))
    rss_start = get_current_rss_mb()
    error_message: str | None = None
    telemetry: BenchmarkTelemetry | None = None
    try:
        raw_out = benchmark_fn()
        if isinstance(raw_out, BenchmarkTelemetry):
            telemetry = raw_out
            tokens_per_sec = raw_out.tokens_per_sec
            latency_ms = raw_out.latency_ms
            memory_mb = raw_out.memory_mb
        else:
            (tokens_per_sec, latency_ms, memory_mb) = raw_out
        if (
            enforce_strict_hw
            and hw_family in {"tpu", "gpu"}
            and telemetry is not None
            and telemetry.resolved_device is not None
            and telemetry.resolved_device.strip().lower() in {"cpu", "/cpu:0", "/device:cpu:0"}
        ):
            msg = (
                f"Strict hardware check failed: requested '{hardware}' on backend '{backend_name}', "
                f"but execution fell back to '{telemetry.resolved_device}'."
            )
            raise HardwareUnavailableError(
                msg,
                hardware=hardware,
                backend=backend_name,
                model_name=model_name,
                operation="benchmark",
            )
        status = ExecutionStatus.SUCCESS.value
        status_code = ExecutionStatus.SUCCESS.value
    except (DependencyMissingError, ConfigurationError):
        raise
    except OPERATIONAL_EXCEPTIONS as e:
        logger.exception("Benchmark failed: ")
        if should_raise:
            raise BenchmarkError(
                f"{backend_name} benchmark failed: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation="benchmark",
                cause=e,
            ) from e
        error_message = str(e)
        status = f"failed: {error_message}"
        status_code = ExecutionStatus.FAILED.value
        latency_ms = 0.0
        tokens_per_sec = 0.0
        memory_mb = 0.0

    rss_end = get_current_rss_mb()
    rss_mb = max(rss_end, rss_start)

    result: BenchmarkResultDict = {
        "backend": backend_name,
        "backend_variant": backend_name,
        "model": model_name,
        "hardware": hardware,
        "batch_size": batch_size,
        "tokens_per_sec": float(tokens_per_sec),
        "latency_ms": float(latency_ms),
        "memory_mb": float(memory_mb),
        "rss_memory_mb": float(rss_mb),
        "rss_mb": float(rss_mb),
        "status": status,
        "status_code": status_code,
    }
    if error_message is not None:
        result["error_message"] = error_message

    active_samples: list[float] | None = latency_samples
    if telemetry is not None:
        if telemetry.resolved_device is not None:
            result["resolved_device"] = telemetry.resolved_device
        if telemetry.device_count is not None:
            result["device_count"] = telemetry.device_count
        if telemetry.compile_latency_ms is not None:
            result["compile_latency_ms"] = float(telemetry.compile_latency_ms)
        if telemetry.peak_hbm_mb is not None:
            result["peak_hbm_mb"] = float(telemetry.peak_hbm_mb)
        if telemetry.keras_backend is not None:
            result["keras_backend"] = telemetry.keras_backend
        if active_samples is None and telemetry.latency_samples:
            active_samples = list(telemetry.latency_samples)

    if active_samples:
        stats = compute_latency_statistics(active_samples)
        result["latency_stats"] = stats

    return result


__all__ = [
    "BenchmarkTelemetry",
    "DetailedLoopMetrics",
    "compute_latency_statistics",
    "compute_throughput_and_latency_ms",
    "execute_detailed_benchmark_loop",
    "execute_timed_benchmark_loop",
    "get_current_rss_mb",
    "normalize_benchmark_mode",
    "normalize_hardware_family",
    "run_benchmark_wrapper",
]
