"""Unit and behavioral tests for `gemma_4_sql.backends.common_benchmark`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Never
from unittest.mock import patch

import pytest

from gemma_4_sql.backends.common_benchmark import (
    BenchmarkTelemetry,
    compute_latency_statistics,
    execute_detailed_benchmark_loop,
    execute_timed_benchmark_loop,
    get_current_rss_mb,
    normalize_benchmark_mode,
    normalize_hardware_family,
    run_benchmark_wrapper,
)
from gemma_4_sql.exceptions import (
    BenchmarkError,
    ConfigurationError,
    DependencyMissingError,
    HardwareUnavailableError,
)
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


def test_normalize_hardware_and_mode() -> None:
    """Verify SKU-qualified hardware family and benchmark mode normalization including error branches."""
    assert normalize_hardware_family("tpu") == "tpu"
    assert normalize_hardware_family("TPU-v6e-8") == "tpu"
    assert normalize_hardware_family("tpu_v4_8") == "tpu"
    assert normalize_hardware_family("gpu") == "gpu"
    assert normalize_hardware_family("cuda:0") == "gpu"
    assert normalize_hardware_family("gpu-a100") == "gpu"
    assert normalize_hardware_family("mps") == "mps"
    assert normalize_hardware_family("mps:0") == "mps"
    assert normalize_hardware_family("cpu") == "cpu"
    assert normalize_hardware_family("x86_64") == "cpu"

    assert normalize_benchmark_mode("prefill") == "prefill"
    assert normalize_benchmark_mode("decode") == "decode"
    assert normalize_benchmark_mode("generate") == "decode"
    assert normalize_benchmark_mode("end-to-end") == "end-to-end"
    assert normalize_benchmark_mode("e2e") == "end-to-end"
    with pytest.raises(ConfigurationError, match="Unsupported benchmark mode"):
        normalize_benchmark_mode("invalid_mode", backend="jax_native")


def test_execute_detailed_benchmark_loop_variants() -> None:
    """Verify `execute_detailed_benchmark_loop` across warmup=0, warmup=1, and warmup=3 with sync barrier."""
    steps = 0
    warmups = 0
    syncs = 0

    def _step() -> None:
        """Increment timed step counter."""
        nonlocal steps
        steps += 1

    def _warmup() -> None:
        """Increment warmup step counter."""
        nonlocal warmups
        warmups += 1

    def _sync() -> None:
        """Increment device synchronization counter."""
        nonlocal syncs
        syncs += 1

    m0 = execute_detailed_benchmark_loop(
        _step,
        warmup_steps=0,
        num_runs=2,
        tokens_per_iter=16,
        batch_size=2,
    )
    assert m0.compile_latency_ms == 0.0
    assert len(m0.latency_samples) == 2
    assert steps == 2

    m1 = execute_detailed_benchmark_loop(
        _step,
        warmup_steps=1,
        num_runs=2,
        tokens_per_iter=16,
        batch_size=2,
        warmup_step_fn=_warmup,
        sync_fn=_sync,
    )
    assert m1.compile_latency_ms >= 0.0
    assert warmups == 1
    assert syncs == 3

    m3 = execute_detailed_benchmark_loop(
        _step,
        warmup_steps=3,
        num_runs=2,
        tokens_per_iter=16,
        batch_size=2,
        sync_fn=_sync,
    )
    assert m3.tokens_per_sec > 0.0
    assert len(m3.latency_samples) == 2

    m3_nosync = execute_detailed_benchmark_loop(
        _step,
        warmup_steps=2,
        num_runs=1,
        tokens_per_iter=16,
        batch_size=1,
    )
    assert m3_nosync.compile_latency_ms >= 0.0

    tps_legacy, lat_legacy = execute_timed_benchmark_loop(
        _step,
        warmup_steps=1,
        num_runs=1,
        tokens_per_iter=8,
        batch_size=1,
        warmup_step_fn=_warmup,
        post_warmup_fn=_sync,
        post_runs_fn=_sync,
        use_perf_counter=True,
    )
    assert tps_legacy > 0.0
    assert lat_legacy >= 0.0
    tps_plain, _ = execute_timed_benchmark_loop(
        _step,
        warmup_steps=0,
        num_runs=1,
        tokens_per_iter=8,
        batch_size=1,
    )
    assert tps_plain > 0.0


def test_run_benchmark_wrapper_telemetry_and_strict_hardware() -> None:
    """Verify `BenchmarkTelemetry` fields and `strict_hardware` enforcement in `run_benchmark_wrapper`."""
    full_tel = BenchmarkTelemetry(
        tokens_per_sec=1200.0,
        latency_ms=4.5,
        memory_mb=2048.0,
        resolved_device="tpu:0",
        device_count=8,
        compile_latency_ms=320.0,
        peak_hbm_mb=2048.0,
        keras_backend="jax",
        latency_samples=(4.0, 5.0),
    )
    res = run_benchmark_wrapper(
        "keras_jax",
        "google/gemma-4-E4B",
        "tpu-v6e",
        8,
        lambda: full_tel,
        strict_hardware=True,
    )
    assert res["status"] == "success"
    assert res.get("backend_variant") == "keras_jax"
    assert res.get("resolved_device") == "tpu:0"
    assert res.get("device_count") == 8
    assert res.get("compile_latency_ms") == 320.0
    assert res.get("peak_hbm_mb") == 2048.0
    assert res.get("keras_backend") == "jax"
    assert "latency_stats" in res
    assert res["latency_stats"]["mean_ms"] == 4.5
    assert "rss_mb" in res

    minimal_tel = BenchmarkTelemetry(tokens_per_sec=50.0, latency_ms=20.0, memory_mb=64.0)
    res_min = run_benchmark_wrapper("jax_native", "m", "cpu", 1, lambda: minimal_tel)
    assert "resolved_device" not in res_min
    assert "latency_stats" not in res_min

    with pytest.raises(HardwareUnavailableError, match="missing dependencies"):
        run_benchmark_wrapper(
            "jax_native",
            "m",
            "tpu-v5e",
            1,
            lambda: minimal_tel,
            missing_deps=True,
            strict_hardware=True,
        )

    cpu_fallback_tel = BenchmarkTelemetry(
        tokens_per_sec=10.0,
        latency_ms=99.0,
        memory_mb=128.0,
        resolved_device="/CPU:0",
    )
    with pytest.raises(HardwareUnavailableError, match="Strict hardware check failed"):
        run_benchmark_wrapper(
            "keras_tensorflow",
            "m",
            "tpu-v4-8",
            1,
            lambda: cpu_fallback_tel,
            strict_hardware=True,
        )
