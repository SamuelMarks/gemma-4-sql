"""Behavioral tests for the SDK benchmark routing and argument forwarding."""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.sdk.benchmark import benchmark


def test_sdk_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify sdk.benchmark forwards all hardware and generation parameters to the resolved backend."""
    captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class FakeBackend:
        """Test stub recording benchmark_model arguments."""

        def benchmark_model(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Record call arguments and return structured benchmark metrics."""
            captured_calls.append((args, kwargs))
            return {
                "backend": str(kwargs.get("backend_alias", "pytorch")),
                "model": args[0],
                "hardware": args[1],
                "batch_size": args[2],
                "tokens_per_sec": 250.0,
                "latency_ms": 8.0,
                "memory_mb": 512.0,
                "rss_memory_mb": 128.0,
                "status": "success",
            }

    requested_backends: list[str] = []

    def mock_get_backend(name: str) -> FakeBackend:
        """Record backend name and return FakeBackend."""
        requested_backends.append(name)
        return FakeBackend()

    import sys

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
    assert len(captured_calls) == 1
    args, kwargs = captured_calls[0]
    assert args == ("google/gemma-4-E2B", "cuda", 16)
    assert kwargs["dtype"] == "float16"
    assert kwargs["mode"] == "decode"
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["warmup_steps"] == 3
    assert kwargs["backend_alias"] == "pytorch_native"
    assert kwargs["num_runs"] == 10
    assert res["status"] == "success"
    assert res["backend"] == "pytorch_native"
    assert res["tokens_per_sec"] == 250.0
