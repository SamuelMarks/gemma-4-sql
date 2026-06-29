"""Benchmarking SDK module."""

from __future__ import annotations


def benchmark(model_name: str, hardware: str, batch_size: int, backend: str) -> dict[str, object]:
    """Benchmarks a model on specific hardware.

    Args:
    ----
        model_name: Name of the model.
        hardware: Hardware to benchmark on ('gpu', 'tpu', 'cpu').
        batch_size: Batch size.
        backend: Backend to use.

    Returns:
    -------
        Benchmarking metrics.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).benchmark_model(model_name, hardware, batch_size)
