"""Tests for SDK benchmark entry point."""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.benchmark import benchmark

try:
    import keras
except ImportError:
    keras = None

try:
    import maxtext
except ImportError:
    maxtext = None


def test_benchmark_jax() -> None:
    """Test benchmark invocation for JAX backend."""
    res = benchmark("gemma-4", "gpu", 1, "jax")
    assert res["backend"] == "jax"
    assert "tokens_per_second" in res or "latency_ms" in res or "status" in res


def test_benchmark_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test benchmark invocation for Keras backend with fallback when missing."""
    if keras is None:
        import gemma_4_sql.backends.keras.benchmark as kbm

        monkeypatch.setattr(kbm, "keras", object())
        monkeypatch.setattr(kbm, "benchmark_model", lambda *args, **kwargs: {"backend": "keras", "status": "completed"})
    res = benchmark("gemma-4", "gpu", 1, "keras")
    assert res["backend"] == "keras"


def test_benchmark_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test benchmark invocation for MaxText backend with fallback when missing."""
    if maxtext is None:
        import gemma_4_sql.backends.maxtext.benchmark as mbm

        monkeypatch.setattr(mbm, "benchmark_model", lambda *args, **kwargs: {"backend": "maxtext", "status": "completed"})
    res = benchmark("gemma-4", "gpu", 1, "maxtext")
    assert res["backend"] == "maxtext"


def test_benchmark_pytorch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test benchmark invocation for PyTorch backend successfully."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    monkeypatch.setattr(bm, "benchmark_model", lambda *args, **kwargs: {"backend": "pytorch", "status": "completed"})
    res = benchmark("gemma-4", "cpu", 1, "pytorch")
    assert res["backend"] == "pytorch"


def test_benchmark_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test benchmark for pytorch missing dependencies."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    monkeypatch.setattr(bm, "torch", None)
    with pytest.raises(DependencyMissingError):
        benchmark("gemma-4", "gpu", 1, "pytorch")


def test_benchmark_unknown() -> None:
    """Test benchmark with an unknown backend identifier raises ValueError."""
    with pytest.raises(ValueError, match=r".*"):
        benchmark("gemma-4", "gpu", 1, "unknown")


def test_benchmark_parameter_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test benchmark invocation with varied batch sizes and model names."""
    import gemma_4_sql.backends.jax.benchmark as jbm

    recorded_kwargs: dict[str, Any] = {}

    def mock_bench(*args: Any, **kwargs: Any) -> dict[str, Any]:
        recorded_kwargs.update(kwargs)
        return {"backend": "jax", "batch_size": kwargs.get("batch_size", 1)}

    monkeypatch.setattr(jbm, "benchmark_model", mock_bench)
    res = benchmark("custom_gemma", "tpu", 8, "jax")
    assert res["backend"] == "jax"
    assert res["batch_size"] == 8
