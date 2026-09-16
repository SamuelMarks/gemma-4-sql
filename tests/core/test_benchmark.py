"""Unified tests for Benchmark across all backends."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.common_benchmark import (
    compute_latency_statistics,
    get_current_rss_mb,
    run_benchmark_wrapper,
)
from gemma_4_sql.exceptions import DependencyMissingError

BACKENDS = ["jax", "keras", "maxtext", "mlx", "pytorch"]


class UniversalMock(MagicMock):
    """Mock helper for backend benchmarking tests."""

    def __call__(self, *_args: object, **kwargs: object) -> object:
        """Execute mock callable."""
        if "num_records" in kwargs:
            return self
        return UniversalMock()

    def numpy(self) -> float:
        """Return scalar 0.0."""
        return 0.0

    def numel(self) -> int:
        """Return element count."""
        return 1000

    def to(self, *_a: object, **_k: object) -> object:
        """Return self."""
        return self

    def eval(self) -> object:
        """Return self."""
        return self


@pytest.fixture
def mock_benchmark_backend(request: object, monkeypatch: object) -> object:
    """Parametrized fixture configuring mock backends for benchmarking."""
    backend = request.param
    module = importlib.import_module(f"gemma_4_sql.backends.{backend}.benchmark")
    um = UniversalMock()
    if backend == "jax":
        monkeypatch.setattr(module, "jax", um)
        monkeypatch.setattr(module, "jnp", um)
        monkeypatch.setattr(module, "nnx", um)
        monkeypatch.setattr(module, "Gemma4ForCausalLM", um)
        monkeypatch.setattr(module, "Gemma4Config", um)
    elif backend == "keras":
        monkeypatch.setattr(module, "tf", um)
        monkeypatch.setattr(module, "keras", um)
    elif backend == "maxtext":
        monkeypatch.setattr(module, "jax", um)
        monkeypatch.setattr(module, "jnp", um)
        monkeypatch.setattr(module, "Gemma4Model", um)
    elif backend == "mlx":
        monkeypatch.setattr(module, "mx", um, raising=False)
        monkeypatch.setattr(module, "mlx_lm", um, raising=False)
        monkeypatch.setattr(module, "mlx", um, raising=False)
        monkeypatch.setattr(module, "AutoModelForCausalLM", um, raising=False)
    elif backend == "pytorch":
        monkeypatch.setattr(module, "torch", um)
        monkeypatch.setattr(module, "AutoModelForCausalLM", um)
    return (backend, module)


@pytest.mark.parametrize("mock_benchmark_backend", BACKENDS, indirect=True)
def test_benchmark_model_real(mock_benchmark_backend: object) -> None:
    """Test mock backend benchmark fixture dispatch."""
    (_backend, _module) = mock_benchmark_backend


def test_compute_latency_statistics() -> None:
    """Test compute_latency_statistics with empty and non-empty samples."""
    empty_stats = compute_latency_statistics([])
    assert empty_stats["mean_ms"] == 0.0
    assert empty_stats["p50_ms"] == 0.0

    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_statistics(samples)
    assert stats["mean_ms"] == 30.0
    assert stats["p50_ms"] == 30.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 50.0
    assert stats["p90_ms"] == 50.0
    assert stats["p99_ms"] == 50.0


def test_get_current_rss_mb() -> None:
    """Test get_current_rss_mb returns positive memory float."""
    rss = get_current_rss_mb()
    assert isinstance(rss, float)
    assert rss >= 0.0


def test_run_benchmark_wrapper_paths() -> None:
    """Test run_benchmark_wrapper success, failure, missing deps, and raise_if_missing."""
    # 1. Success with latency stats
    res_succ = run_benchmark_wrapper(
        backend_name="test_backend",
        model_name="model_a",
        hardware="cpu",
        batch_size=2,
        missing_deps=False,
        missing_status="missing",
        benchmark_fn=lambda: (120.0, 15.0, 256.0),
        latency_samples=[14.0, 15.0, 16.0],
    )
    assert res_succ["status"] == "success"
    assert res_succ["tokens_per_sec"] == 120.0
    assert res_succ["latency_ms"] == 15.0
    assert res_succ["memory_mb"] == 256.0
    assert "rss_memory_mb" in res_succ
    assert res_succ["latency_stats"]["mean_ms"] == 15.0

    # 2. Execution failure branch
    def _fail() -> tuple[float, float, float]:
        raise RuntimeError("Benchmark execution crashed")

    res_fail = run_benchmark_wrapper(
        backend_name="test_backend",
        model_name="model_b",
        hardware="cuda",
        batch_size=4,
        missing_deps=False,
        missing_status="missing",
        benchmark_fn=_fail,
    )
    assert "failed: Benchmark execution crashed" in res_fail["status"]
    assert res_fail["tokens_per_sec"] == 0.0

    # 3. Missing dependencies without raise
    res_miss = run_benchmark_wrapper(
        backend_name="test_backend",
        model_name="model_c",
        hardware="tpu",
        batch_size=8,
        missing_deps=True,
        missing_status="mocked_missing_deps",
        benchmark_fn=lambda: (0.0, 0.0, 0.0),
        raise_if_missing=False,
    )
    assert res_miss["status"] == "mocked_missing_deps"

    # 4. Missing dependencies with raise
    with pytest.raises(DependencyMissingError, match="Dependencies for test_backend benchmarking on tpu are missing"):
        run_benchmark_wrapper(
            backend_name="test_backend",
            model_name="model_d",
            hardware="tpu",
            batch_size=8,
            missing_deps=True,
            missing_status="mocked_missing_deps",
            benchmark_fn=lambda: (0.0, 0.0, 0.0),
            raise_if_missing=True,
        )
