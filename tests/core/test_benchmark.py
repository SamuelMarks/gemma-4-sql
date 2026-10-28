"""Behavioral tests for the SDK benchmark routing and argument forwarding."""

from __future__ import annotations

import sys

import pytest

from gemma_4_sql.sdk.benchmark import benchmark
from tests.utils.test_utils import RecordingBackendDouble


def test_sdk_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify sdk.benchmark forwards all hardware and generation parameters to the resolved backend."""
    double = RecordingBackendDouble("pytorch_native")
    requested_backends: list[str] = []

    def mock_get_backend(name: str) -> RecordingBackendDouble:
        """Record backend name and return RecordingBackendDouble."""
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
