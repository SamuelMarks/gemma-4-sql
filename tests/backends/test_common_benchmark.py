"""Unit and behavioral tests for `gemma_4_sql.backends.common_benchmark`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Never
from unittest.mock import patch

import pytest

from gemma_4_sql.backends.common_benchmark import (
    compute_latency_statistics,
    get_current_rss_mb,
    run_benchmark_wrapper,
)
from gemma_4_sql.exceptions import BenchmarkError, DependencyMissingError
from tests.utils.test_utils import FakeModuleNamespace


@dataclass
class _FakeMemoryInfo:
    """Typed memory info double exposing `rss` in bytes."""

    rss: int


class _FakePsutilProcess:
    """Typed `psutil.Process` double."""

    def memory_info(self) -> _FakeMemoryInfo:
        """Return 1 MiB RSS."""
        return _FakeMemoryInfo(rss=1048576)


class _FakePsutilModule(FakeModuleNamespace):
    """Typed `psutil` module double."""

    Process: type[_FakePsutilProcess]

    def __init__(self, name: str) -> None:
        """Initialize fake `psutil` module with `Process`.

        Args:
            name: Module name string.

        """
        super().__init__(name)
        self.Process = _FakePsutilProcess


@dataclass
class _FakeRUsage:
    """Typed `resource.struct_rusage` double exposing `ru_maxrss`."""

    ru_maxrss: int


class _FakeResourceModule:
    """Typed `resource` module double."""

    RUSAGE_SELF: int = 0

    def __init__(self, ru_maxrss: int) -> None:
        """Initialize with `ru_maxrss`."""
        self._ru_maxrss = ru_maxrss

    def getrusage(self, _who: int) -> _FakeRUsage:
        """Return `_FakeRUsage`."""
        return _FakeRUsage(ru_maxrss=self._ru_maxrss)


def test_get_current_rss_mb_psutil() -> None:
    """Verify `get_current_rss_mb` reads `psutil.Process().memory_info().rss` when `psutil` is installed."""
    fake_psutil = _FakePsutilModule("psutil")
    with patch.dict("sys.modules", {"psutil": fake_psutil}):
        assert get_current_rss_mb() == 1.0


def test_get_current_rss_mb_fallback_darwin() -> None:
    """Verify `get_current_rss_mb` converts macOS byte-granular `ru_maxrss` into MiB."""
    fake_resource = _FakeResourceModule(ru_maxrss=1048576 * 2)
    with (
        patch.dict("sys.modules", {"psutil": None}),
        patch("sys.platform", "darwin"),
        patch("gemma_4_sql.backends.common_benchmark.resource", fake_resource),
    ):
        assert get_current_rss_mb() == 2.0


def test_get_current_rss_mb_fallback_linux() -> None:
    """Verify `get_current_rss_mb` converts Linux KiB-granular `ru_maxrss` into MiB."""
    fake_resource = _FakeResourceModule(ru_maxrss=1024 * 3)
    with (
        patch.dict("sys.modules", {"psutil": None}),
        patch("sys.platform", "linux"),
        patch("gemma_4_sql.backends.common_benchmark.resource", fake_resource),
    ):
        assert get_current_rss_mb() == 3.0


def test_compute_latency_statistics_empty() -> None:
    """Verify `compute_latency_statistics` returns all-zero metrics on an empty sample sequence."""
    stats = compute_latency_statistics([])
    assert stats["mean_ms"] == 0.0
    assert stats["p50_ms"] == 0.0
    assert stats["p90_ms"] == 0.0
    assert stats["p99_ms"] == 0.0
    assert stats["min_ms"] == 0.0
    assert stats["max_ms"] == 0.0


def test_compute_latency_statistics() -> None:
    """Verify `compute_latency_statistics` computes exact mean, min, max, and percentile values."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = compute_latency_statistics(latencies)
    assert stats["mean_ms"] == 55.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 100.0
    assert stats["p50_ms"] == 50.0
    assert stats["p90_ms"] == 90.0
    assert stats["p99_ms"] == 100.0


def test_run_benchmark_wrapper_missing_deps() -> None:
    """Verify `run_benchmark_wrapper` returns `backend_unavailable` when `missing_deps=True`."""
    res = run_benchmark_wrapper("test_backend", "model", "cpu", 1, lambda: (0.0, 0.0, 0.0), missing_deps=True)
    assert res["status"] == "backend_unavailable"
    assert res["tokens_per_sec"] == 0.0


def test_run_benchmark_wrapper_missing_deps_raise() -> None:
    """Verify `run_benchmark_wrapper` raises `DependencyMissingError` when `raise_if_missing=True`."""
    with pytest.raises(
        DependencyMissingError, match=r"Dependencies for test_backend benchmarking on cpu are missing\."
    ):
        run_benchmark_wrapper(
            "test_backend",
            "model",
            "cpu",
            1,
            lambda: (0.0, 0.0, 0.0),
            missing_deps=True,
            raise_if_missing=True,
        )


def test_run_benchmark_wrapper_success() -> None:
    """Verify `run_benchmark_wrapper` populates throughput, latency, memory, RSS, and latency percentiles."""

    def mock_benchmark() -> tuple[float, float, float]:
        """Return deterministic `(tokens_per_sec, latency_ms, memory_mb)`."""
        return (10.0, 50.0, 100.0)

    with patch("gemma_4_sql.backends.common_benchmark.get_current_rss_mb", return_value=10.0):
        res = run_benchmark_wrapper("test_backend", "model", "cpu", 1, mock_benchmark, latency_samples=[10.0, 20.0])

    assert res["status"] == "success"
    assert res["tokens_per_sec"] == 10.0
    assert res["latency_ms"] == 50.0
    assert res["memory_mb"] == 100.0
    assert res.get("rss_memory_mb") == 10.0
    assert "latency_stats" in res
    assert res["latency_stats"]["mean_ms"] == 15.0


def test_run_benchmark_wrapper_failure() -> None:
    """Verify `run_benchmark_wrapper` handles operational errors, `raise_on_error`, and `DependencyMissingError`."""

    def mock_benchmark() -> Never:
        """Raise `RuntimeError` to exercise failure handling.

        Raises:
            RuntimeError: Always raised to simulate benchmark failure.

        """
        raise RuntimeError("Benchmark error")

    with patch("gemma_4_sql.backends.common_benchmark.get_current_rss_mb", return_value=10.0):
        res = run_benchmark_wrapper("test_backend", "model", "cpu", 1, mock_benchmark)

    assert res["status"] == "failed: Benchmark error"
    assert res["tokens_per_sec"] == 0.0
    assert res["latency_ms"] == 0.0
    assert res["memory_mb"] == 0.0

    with (
        patch("gemma_4_sql.backends.common_benchmark.get_current_rss_mb", return_value=10.0),
        pytest.raises(BenchmarkError, match="Benchmark error"),
    ):
        run_benchmark_wrapper("test_backend", "model", "cpu", 1, mock_benchmark, raise_on_error=True)

    with pytest.raises(DependencyMissingError, match="missing bench dep"):
        run_benchmark_wrapper(
            "test_backend",
            "model",
            "cpu",
            1,
            lambda: (_ for _ in ()).throw(DependencyMissingError("missing bench dep")),
        )
