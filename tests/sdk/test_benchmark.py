"""Behavioral tests for the SDK Benchmark module across backends."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.benchmark import benchmark
from tests.utils.test_utils import RecordingBackendDouble


def test_benchmark_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark executes jbm.benchmark_model directly and computes LatencyStatsDict for JAX backend."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.jax.benchmark as jbm
    from gemma_4_sql.backends.common_benchmark import compute_latency_statistics

    monkeypatch.setattr(jbm, "jax", MagicMock())
    monkeypatch.setattr(jbm, "jnp", MagicMock())
    monkeypatch.setattr(jbm, "Gemma4Config", MagicMock())
    monkeypatch.setattr(jbm, "Gemma4ForCausalLM", MagicMock())
    monkeypatch.setattr(jbm, "_run_benchmark_pass", lambda *_a, **_k: (100.0, 10.0, 64.0))  # pyright: ignore[reportUnknownLambdaType]

    res = benchmark("gemma-4", "cpu", 1, "jax", warmup_steps=1, num_runs=2)
    assert res["backend"] == "jax"
    assert res["tokens_per_sec"] == 100.0
    assert res["latency_ms"] == 10.0
    assert res["memory_mb"] == 64.0
    stats = compute_latency_statistics([9.0, 10.0, 11.0])
    assert stats["mean_ms"] == pytest.approx(10.0)
    assert stats["p50_ms"] == pytest.approx(10.0)
    assert stats["p90_ms"] >= 10.0
    assert stats["p99_ms"] >= 10.0


def test_benchmark_keras_routing_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark forwards exact arguments to Keras backend and raises when TensorFlow is missing."""
    import gemma_4_sql.backends.keras as kb
    import gemma_4_sql.backends.keras.benchmark as kbm

    monkeypatch.setattr(kbm, "tf", None)
    with pytest.raises(DependencyMissingError):
        benchmark("gemma-4", "gpu", 1, "keras")

    keras_double = RecordingBackendDouble("keras")
    monkeypatch.setattr(kbm, "benchmark_model", keras_double.benchmark_model)
    monkeypatch.setattr(kb, "benchmark_model", keras_double.benchmark_model)

    res = benchmark("gemma-4", "gpu", 4, "keras", dtype="float16", mode="decode")
    assert res["backend"] == "keras"
    assert len(keras_double.benchmark_calls) == 1
    model_name, hardware, batch_size, kw = keras_double.benchmark_calls[0]
    assert model_name == "gemma-4"
    assert hardware == "gpu"
    assert batch_size == 4
    assert kw["dtype"] == "float16"
    assert kw["mode"] == "decode"


def test_benchmark_maxtext_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark forwards arguments to the MaxText backend."""
    import gemma_4_sql.backends.maxtext.benchmark as mbm

    maxtext_double = RecordingBackendDouble("maxtext")
    monkeypatch.setattr(mbm, "benchmark_model", maxtext_double.benchmark_model)
    res = benchmark("gemma-4", "tpu", 2, "maxtext")
    assert res["backend"] == "maxtext"
    assert len(maxtext_double.benchmark_calls) == 1
    assert maxtext_double.benchmark_calls[0][:3] == ("gemma-4", "tpu", 2)


def test_benchmark_pytorch_routing_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark executes real PyTorch benchmark_model on a tiny Gemma 4 model and raises when torch is missing."""
    import torch

    import gemma_4_sql.backends.pytorch.benchmark as bm

    monkeypatch.setattr(torch, "compile", lambda m: m)  # pyright: ignore[reportUnknownLambdaType]
    res = benchmark("gemma-4-tiny", "cpu", 1, "pytorch_native", warmup_steps=1, num_runs=1, max_new_tokens=2)
    assert res["backend"] == "pytorch_native"
    assert res["status"] == "success"
    assert res["tokens_per_sec"] > 0.0

    monkeypatch.setattr(bm, "torch", None)
    with pytest.raises(DependencyMissingError):
        benchmark("gemma-4", "gpu", 1, "pytorch")


def test_benchmark_unknown() -> None:
    """Verify benchmark raises ConfigurationError on an unknown backend identifier."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        benchmark("gemma-4", "gpu", 1, "unknown")
