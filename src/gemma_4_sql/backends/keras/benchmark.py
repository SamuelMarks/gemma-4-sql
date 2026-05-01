"""
Keras-specific benchmarking pipeline.
"""

from __future__ import annotations

from typing import Any


def benchmark_model(
    model_name: str,
    hardware: str,
    batch_size: int,
) -> dict[str, Any]:
    """Benchmark a model using the Keras backend."""
    return {
        "backend": "keras",
        "model": model_name,
        "hardware": hardware,
        "batch_size": batch_size,
        "tokens_per_sec": 800.0,
        "latency_ms": 25.0,
        "memory_mb": 6000.0,
    }
