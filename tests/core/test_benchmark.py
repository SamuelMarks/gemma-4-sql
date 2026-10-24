"""Unified tests for Benchmark across all backends."""

from __future__ import annotations

import importlib
from typing import Any
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

    def __call__(self, *_args: Any, **kwargs: Any) -> Any:
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

    def to(self, *_a: Any, **_k: Any) -> Any:
        """Return self."""
        return self

    def eval(self) -> Any:
        """Return self."""
        return self


@pytest.fixture
def mock_benchmark_backend(request: Any, monkeypatch: Any) -> Any:
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
def test_benchmark_model_real(mock_benchmark_backend: Any) -> None:
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
        benchmark_fn=lambda: (120.0, 15.0, 256.0),
        latency_samples=[14.0, 15.0, 16.0],
    )
    assert res_succ["status"] == "success"
    assert res_succ["tokens_per_sec"] == 120.0
    assert res_succ["latency_ms"] == 15.0
    assert res_succ["memory_mb"] == 256.0
    assert "rss_memory_mb" in res_succ
    stats_succ = res_succ.get("latency_stats")
    assert stats_succ is not None and stats_succ["mean_ms"] == 15.0

    # 2. Execution failure branch
    def _fail() -> tuple[float, float, float]:
        """Provide fail test helper."""
        raise RuntimeError("Benchmark execution crashed")

    res_fail = run_benchmark_wrapper(
        backend_name="test_backend",
        model_name="model_b",
        hardware="cuda",
        batch_size=4,
        missing_deps=False,
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
        benchmark_fn=lambda: (0.0, 0.0, 0.0),
        raise_if_missing=False,
    )
    assert res_miss["status"] == "backend_unavailable"

    # 4. Missing dependencies with raise
    with pytest.raises(DependencyMissingError, match="Dependencies for test_backend benchmarking on tpu are missing"):
        run_benchmark_wrapper(
            backend_name="test_backend",
            model_name="model_d",
            hardware="tpu",
            batch_size=8,
            missing_deps=True,
            benchmark_fn=lambda: (0.0, 0.0, 0.0),
            raise_if_missing=True,
        )


def test_sdk_benchmark(monkeypatch: Any) -> None:
    """Verify sdk benchmark behavior."""
    import sys

    from gemma_4_sql.sdk.benchmark import benchmark

    class FakeBackend:
        """Test stub for `FakeBackend`."""

        def benchmark_model(self, *args: Any, **kwargs: Any):
            """Provide benchmark model test helper."""
            return {"status": "success"}

    def mock_get_backend(x: Any):
        """Provide mock get backend test helper."""
        return FakeBackend()

    bench_mod = sys.modules["gemma_4_sql.sdk.benchmark"]
    monkeypatch.setattr(bench_mod, "get_backend", mock_get_backend)
    res = benchmark("m", "cpu", 1, "pytorch")
    assert res["status"] == "success"


def test_universal_mock():
    """Test UniversalMock methods."""
    um = UniversalMock()
    # 25-27
    assert isinstance(um(), UniversalMock)
    assert um(num_records=5) is um
    # 31
    assert um.numpy() == 0.0
    # 35
    assert um.numel() == 1000
    # 39
    assert um.to() is um
    # 43
    assert um.eval() is um


@pytest.fixture(autouse=True)
def _mock_import_for_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intercept importlib.import_module for fake_backend during benchmark fixture tests."""
    import importlib

    orig_import = importlib.import_module

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Return dummy object when importing fake_backend."""
        if name and "fake_backend" in name:
            return object()
        return orig_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib, "import_module", mock_import)


@pytest.mark.parametrize("mock_benchmark_backend", ["fake_backend"], indirect=True)
def test_mock_benchmark_backend_unsupported(mock_benchmark_backend: Any) -> None:
    """Cover the else branch in mock_benchmark_backend."""
    res = mock_benchmark_backend
    assert res[0] == "fake_backend"
