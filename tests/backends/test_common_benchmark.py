"""Tests for common benchmark utilities."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.common_benchmark import (
    compute_latency_statistics,
    get_current_rss_mb,
    run_benchmark_wrapper,
)
from gemma_4_sql.exceptions import DependencyMissingError


def test_get_current_rss_mb_psutil():
    """Test get_current_rss_mb with psutil."""
    mock_psutil = MagicMock()
    mock_process = MagicMock()
    mock_memory_info = MagicMock()
    mock_memory_info.rss = 1048576  # 1 MB
    mock_process.memory_info.return_value = mock_memory_info
    mock_psutil.Process.return_value = mock_process

    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        assert get_current_rss_mb() == 1.0


def test_get_current_rss_mb_fallback_darwin(monkeypatch):
    """Test get_current_rss_mb fallback on darwin."""
    with patch.dict("sys.modules", {"psutil": None}), patch("sys.platform", "darwin"):
        mock_resource = MagicMock()
        mock_rusage = MagicMock()
        mock_rusage.ru_maxrss = 1048576 * 2  # 2 MB in bytes on darwin
        mock_resource.getrusage.return_value = mock_rusage
        with patch("gemma_4_sql.backends.common_benchmark.resource", mock_resource):
            assert get_current_rss_mb() == 2.0


def test_get_current_rss_mb_fallback_linux(monkeypatch):
    """Test get_current_rss_mb fallback on linux."""
    with patch.dict("sys.modules", {"psutil": None}), patch("sys.platform", "linux"):
        mock_resource = MagicMock()
        mock_rusage = MagicMock()
        mock_rusage.ru_maxrss = 1024 * 3  # 3 MB in KB on linux
        mock_resource.getrusage.return_value = mock_rusage
        with patch("gemma_4_sql.backends.common_benchmark.resource", mock_resource):
            assert get_current_rss_mb() == 3.0


def test_compute_latency_statistics_empty():
    """Test compute_latency_statistics with empty list."""
    stats = compute_latency_statistics([])
    assert stats["mean_ms"] == 0.0
    assert stats["p50_ms"] == 0.0
    assert stats["p90_ms"] == 0.0
    assert stats["p99_ms"] == 0.0
    assert stats["min_ms"] == 0.0
    assert stats["max_ms"] == 0.0


def test_compute_latency_statistics():
    """Test compute_latency_statistics."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = compute_latency_statistics(latencies)
    assert stats["mean_ms"] == 55.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 100.0
    assert stats["p50_ms"] == 50.0  # (0.5 * 9 = 4.5 -> 5)
    assert stats["p90_ms"] == 90.0
    assert stats["p99_ms"] == 100.0


def test_run_benchmark_wrapper_missing_deps():
    """Test run_benchmark_wrapper with missing dependencies."""
    res = run_benchmark_wrapper("test_backend", "model", "cpu", 1, True, "missing", lambda: (0, 0, 0))
    assert res["status"] == "missing"
    assert res["tokens_per_sec"] == 0.0


def test_run_benchmark_wrapper_missing_deps_raise():
    """Test run_benchmark_wrapper with missing dependencies and raise_if_missing."""
    with pytest.raises(DependencyMissingError, match="Dependencies for test_backend benchmarking on cpu are missing."):
        run_benchmark_wrapper("test_backend", "model", "cpu", 1, True, "missing", lambda: (0, 0, 0), raise_if_missing=True)


def test_run_benchmark_wrapper_success():
    """Test run_benchmark_wrapper success."""

    def mock_benchmark():
        return (10.0, 50.0, 100.0)

    with patch("gemma_4_sql.backends.common_benchmark.get_current_rss_mb", return_value=10.0):
        res = run_benchmark_wrapper("test_backend", "model", "cpu", 1, False, "", mock_benchmark, latency_samples=[10.0, 20.0])

    assert res["status"] == "success"
    assert res["tokens_per_sec"] == 10.0
    assert res["latency_ms"] == 50.0
    assert res["memory_mb"] == 100.0
    assert res["rss_memory_mb"] == 10.0
    assert "latency_stats" in res
    assert res["latency_stats"]["mean_ms"] == 15.0


def test_run_benchmark_wrapper_failure():
    """Test run_benchmark_wrapper failure."""

    def mock_benchmark():
        raise RuntimeError("Benchmark error")

    with patch("gemma_4_sql.backends.common_benchmark.get_current_rss_mb", return_value=10.0):
        res = run_benchmark_wrapper("test_backend", "model", "cpu", 1, False, "", mock_benchmark)

    assert res["status"] == "failed: Benchmark error"
    assert res["tokens_per_sec"] == 0.0
    assert res["latency_ms"] == 0.0
    assert res["memory_mb"] == 0.0
