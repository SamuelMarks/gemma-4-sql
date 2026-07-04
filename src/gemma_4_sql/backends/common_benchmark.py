# Copyright 2024
"""Common benchmarking utilities."""

from __future__ import annotations

import logging
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)


def run_benchmark_wrapper(
    backend_name: str,
    model_name: str,
    hardware: str,
    batch_size: int,
    missing_deps: bool,
    missing_status: str,
    benchmark_fn: Callable[[], tuple[float, float, float]],
) -> JSONDict:
    """Wrap benchmarking logic to unify exception handling and result formatting.

    Args:
    ----
        backend_name: The name of the backend.
        model_name: The name of the model.
        hardware: The target hardware.
        batch_size: The batch size.
        missing_deps: Whether dependencies are missing.
        missing_status: The status message if dependencies are missing.
        benchmark_fn: A callable that executes the benchmark and returns (tokens_per_sec, latency_ms, memory_mb).

    Returns:
    -------
        A dictionary containing the benchmark results.

    """
    if missing_deps:
        return {
            "backend": backend_name,
            "model": model_name,
            "hardware": hardware,
            "batch_size": batch_size,
            "status": missing_status,
            "tokens_per_sec": 0.0,
            "latency_ms": 0.0,
            "memory_mb": 0.0,
        }

    logger.info("Starting %s benchmark for %s on %s (batch size %d)", backend_name.upper(), model_name, hardware, batch_size)

    try:
        (tokens_per_sec, latency_ms, memory_mb) = benchmark_fn()
        status = "success"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Benchmark failed: ")
        status = f"failed: {e!s}"
        latency_ms = 0.0
        tokens_per_sec = 0.0
        memory_mb = 0.0

    return {
        "backend": backend_name,
        "model": model_name,
        "hardware": hardware,
        "batch_size": batch_size,
        "tokens_per_sec": float(tokens_per_sec),
        "latency_ms": float(latency_ms),
        "memory_mb": float(memory_mb),
        "status": status,
    }
