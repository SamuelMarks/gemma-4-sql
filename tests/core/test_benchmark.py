"""Behavioral tests for the SDK benchmark routing and argument forwarding."""

from __future__ import annotations

import sys

import pytest

from gemma_4_sql.exceptions import ConfigurationError, HardwareUnavailableError
from gemma_4_sql.sdk.benchmark import benchmark, benchmark_matrix, format_benchmark_matrix_markdown
from gemma_4_sql.type_hints import BenchmarkResultDict, JSONValue
from tests.utils.test_utils import RecordingBackendDouble


def test_sdk_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify sdk.benchmark forwards all hardware and generation parameters to the resolved backend."""
    double = RecordingBackendDouble("pytorch_native")
    requested_backends: list[str] = []

    def mock_get_backend(name: str) -> RecordingBackendDouble:
        """Record backend name and return RecordingBackendDouble.

        Returns:
            Configured `RecordingBackendDouble` instance.

        """
        requested_backends.append(name)
        return double

    monkeypatch.setattr(sys.modules["gemma_4_sql.sdk.benchmark"], "get_backend", mock_get_backend)
    res = benchmark(
        "google/gemma-4-E2B",
        "cuda",
        16,
        "pytorch_native",
        dtype="float16",
        mode="decode",
        max_new_tokens=64,
        warmup_steps=3,
        num_runs=10,
    )
    assert requested_backends == ["pytorch_native"]
    assert len(double.benchmark_calls) == 1
    model_name, hardware, batch_size, kwargs = double.benchmark_calls[0]
    assert (model_name, hardware, batch_size) == ("google/gemma-4-E2B", "cuda", 16)
    assert kwargs["dtype"] == "float16"
    assert kwargs["mode"] == "decode"
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["warmup_steps"] == 3
    assert kwargs["backend_alias"] == "pytorch_native"
    assert kwargs["num_runs"] == 10
    assert res["status"] == "completed"
    assert res["backend"] == "pytorch_native"
    assert res["tokens_per_sec"] == 1600.0


def test_sdk_benchmark_matrix_and_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `benchmark_matrix` speedups, graceful cell failures, strict_hardware raising, and Markdown table."""

    def _fake_bench(
        model_name: str,
        hardware: str,
        batch_size: int,
        backend: str,
        dtype: str = "bfloat16",
        mode: str = "prefill",
        max_new_tokens: int = 128,
        warmup_steps: int = 5,
        **_kwargs: JSONValue,
    ) -> BenchmarkResultDict:
        """Simulate varied backend throughputs and one missing-dependency failure.

        Returns:
            Simulated `BenchmarkResultDict`.

        Raises:
            ConfigurationError: When `backend == 'keras_tensorflow'`.

        """
        _ = (dtype, mode, max_new_tokens, warmup_steps)
        if backend == "keras_tensorflow":
            raise ConfigurationError("tf missing", backend=backend)
        if backend == "maxtext":
            return {
                "backend": backend,
                "model": model_name,
                "hardware": hardware,
                "batch_size": batch_size,
                "tokens_per_sec": 0.0,
                "latency_ms": 0.0,
                "memory_mb": 128.0,
                "status": "failed: oom",
                "status_code": "failed",
            }
        tps = 100.0 if backend == "hf_transformers" else (200.0 if backend == "jax_native" else 150.0)
        out: BenchmarkResultDict = {
            "backend": backend,
            "model": model_name,
            "hardware": hardware,
            "batch_size": batch_size,
            "tokens_per_sec": tps,
            "latency_ms": 5.0,
            "memory_mb": 512.0,
            "status": "success",
        }
        if backend == "jax_native":
            out["compile_latency_ms"] = 210.5
            out["peak_hbm_mb"] = 1024.0
        return out

    monkeypatch.setattr(sys.modules["gemma_4_sql.sdk.benchmark"], "benchmark", _fake_bench)
    matrix = benchmark_matrix(
        "google/gemma-4-E4B",
        "tpu-v6e",
        backends=("hf_transformers", "pytorch_native", "keras_tensorflow", "maxtext", "jax_native"),
        modes=("prefill",),
        batch_sizes=(1,),
    )
    assert matrix["status"] == "failed"
    assert len(matrix["cells"]) == 5
    pt_cell = matrix["cells"][1]
    assert pt_cell.get("speedup_vs_hf_transformers") == 1.5
    assert pt_cell.get("speedup_vs_jax_native") == 0.75
    md = format_benchmark_matrix_markdown(matrix)
    assert "### Gemma 4 Benchmark Matrix" in md
    assert "`jax_native`" in md
    assert "210.50" in md
    assert "1.50x" in md

    # Pure success matrix using default TPU_COMPARISON_BACKENDS when all succeed
    def _all_ok_bench(
        model_name: str,
        hardware: str,
        batch_size: int,
        backend: str,
        **_kw: JSONValue,
    ) -> BenchmarkResultDict:
        """Return successful cell for every backend.

        Returns:
            Simulated `BenchmarkResultDict`.

        """
        return {
            "backend": backend,
            "model": model_name,
            "hardware": hardware,
            "batch_size": batch_size,
            "tokens_per_sec": 250.0,
            "latency_ms": 4.0,
            "memory_mb": 256.0,
            "status": "success",
        }

    monkeypatch.setattr(sys.modules["gemma_4_sql.sdk.benchmark"], "benchmark", _all_ok_bench)
    ok_matrix = benchmark_matrix("google/gemma-4-E4B", "tpu", modes=("prefill",), batch_sizes=(1,))
    assert ok_matrix["status"] == "success"
    assert len(ok_matrix["cells"]) == 7

    # strict_hardware=True propagates HardwareUnavailableError immediately
    def _raise_hw(*_a: JSONValue, **_kw: JSONValue) -> BenchmarkResultDict:
        """Raise HardwareUnavailableError.

        Raises:
            HardwareUnavailableError: Always raised to test strict_hardware propagation.

        """
        raise HardwareUnavailableError("no tpu", hardware="tpu")

    monkeypatch.setattr(sys.modules["gemma_4_sql.sdk.benchmark"], "benchmark", _raise_hw)
    with pytest.raises(HardwareUnavailableError, match="no tpu"):
        benchmark_matrix("google/gemma-4-E4B", "tpu", backends=("jax_native",), strict_hardware=True)
