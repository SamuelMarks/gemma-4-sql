"""Common benchmarking utilities."""

from __future__ import annotations

import logging
import resource
from collections.abc import Callable

from gemma_4_sql.exceptions import StatusValue
from gemma_4_sql.type_hints import BenchmarkResultDict, LatencyStatsDict

logger = logging.getLogger(__name__)


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
        """Calculate the p-th percentile from sorted latencies."""
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
    benchmark_fn: Callable[[], tuple[float, float, float]],
    *,
    missing_deps: bool = False,
    raise_if_missing: bool = False,
    latency_samples: list[float] | None = None,
    raise_on_error: bool | None = None,
    **kwargs: StatusValue,
) -> BenchmarkResultDict:
    """Wrap benchmarking logic to unify exception handling and result formatting.

    Args:
        backend_name: The name of the backend.
        model_name: The name of the model.
        hardware: The target hardware.
        batch_size: The batch size.
        benchmark_fn: A callable that executes the benchmark and returns (tokens_per_sec, latency_ms, memory_mb).
        missing_deps: Whether dependencies are missing.
        raise_if_missing: If True, raises DependencyMissingError when missing_deps is True.
        latency_samples: Optional sequence of individual request latency samples for percentile stats.
        raise_on_error: If True, raises `BenchmarkError` when `benchmark_fn` fails.
        **kwargs: Additional keyword arguments inspected by `resolve_raise_on_error`.

    Returns:
        A dictionary containing the benchmark results.

    Raises:
        DependencyMissingError: If raise_if_missing is True and missing_deps is True.
        BenchmarkError: If `raise_on_error` is enabled and `benchmark_fn` raises an exception.

    """
    from gemma_4_sql.exceptions import (
        OPERATIONAL_EXCEPTIONS,
        BenchmarkError,
        ConfigurationError,
        DependencyMissingError,
        ExecutionStatus,
        resolve_raise_on_error,
    )

    if missing_deps:
        if raise_if_missing:
            msg = f"Dependencies for {backend_name} benchmarking on {hardware} are missing."
            raise DependencyMissingError(msg)

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

    logger.info("Starting %s benchmark for %s on %s (batch size %d)", backend_name.upper(), model_name, hardware, batch_size)

    should_raise = resolve_raise_on_error(raise_on_error=raise_on_error, kwargs=dict(kwargs))
    rss_start = get_current_rss_mb()
    error_message: str | None = None
    try:
        (tokens_per_sec, latency_ms, memory_mb) = benchmark_fn()
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
        "model": model_name,
        "hardware": hardware,
        "batch_size": batch_size,
        "tokens_per_sec": float(tokens_per_sec),
        "latency_ms": float(latency_ms),
        "memory_mb": float(memory_mb),
        "rss_memory_mb": float(rss_mb),
        "status": status,
        "status_code": status_code,
    }
    if error_message is not None:
        result["error_message"] = error_message

    if latency_samples:
        stats = compute_latency_statistics(latency_samples)
        result["latency_stats"] = stats

    return result
