"""Unit and behavioral tests for JAX benchmarking."""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.conftest import requires_jax

pytestmark = requires_jax

import pytest
from flax import nnx

import gemma_4_sql.backends.jax.benchmark as jax_benchmark
from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
from gemma_4_sql.exceptions import DependencyMissingError


def test_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model raises DependencyMissingError when jax is None."""
    monkeypatch.setattr(jax_benchmark, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing\."):
        jax_benchmark.benchmark_model("test_model", "cpu", 1)


def test_get_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_device resolves CPU/GPU/TPU devices and falls back to CPU on RuntimeError."""
    dev_cpu = jax_benchmark._get_device("cpu")
    assert dev_cpu is not None

    mock_jax = MagicMock()
    mock_jax.devices.side_effect = lambda hw: [f"mock_{hw}_device"] if hw in ("gpu", "tpu", "cpu") else []  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(jax_benchmark, "jax", mock_jax)
    assert jax_benchmark._get_device("gpu") == "mock_gpu_device"
    assert jax_benchmark._get_device("tpu") == "mock_tpu_device"

    def raise_on_accel(hw: str) -> list[str]:
        """Raise RuntimeError on tpu/gpu."""
        if hw in ("tpu", "gpu"):
            raise RuntimeError("No accelerator")
        return ["mock_cpu_device"]

    mock_jax.devices.side_effect = raise_on_accel
    assert jax_benchmark._get_device("tpu") == "mock_cpu_device"


def test_run_benchmark_pass_and_benchmark_model_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass and benchmark_model execute real JAX forward and generate passes."""
    cfg = Gemma4Config.gemma4_tiny()
    model = Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))
    cpu_device = jax_benchmark._get_device("cpu")

    tokens_prefill, latency_prefill, mem_prefill = jax_benchmark._run_benchmark_pass(
        model=model,
        batch_size=1,
        num_runs=1,
        warmup_steps=1,
        mode="prefill",
        max_new_tokens=2,
        device=cpu_device,
    )
    assert tokens_prefill > 0.0
    assert latency_prefill > 0.0
    assert mem_prefill >= 0.0

    # Device with explicit peak_bytes_in_use memory_stats and warmup in generate mode
    from contextlib import nullcontext

    class _MockJaxDevice:
        """Concrete device stub implementing `JaxDeviceProtocol`."""

        def memory_stats(self) -> dict[str, int]:
            """Return deterministic peak_bytes_in_use."""
            return {"peak_bytes_in_use": 64 * 1024 * 1024}

    mock_device_with_stats = _MockJaxDevice()
    monkeypatch.setattr(jax_benchmark.jax, "default_device", lambda _d: nullcontext())  # pyright: ignore[reportUnknownLambdaType]
    _, _, mem_stats = jax_benchmark._run_benchmark_pass(
        model=model,
        batch_size=1,
        num_runs=1,
        warmup_steps=1,
        mode="generate",
        max_new_tokens=2,
        device=mock_device_with_stats,
    )
    assert mem_stats == 64.0

    class _FailingStatsDevice:
        """Concrete device stub raising RuntimeError from `memory_stats()`."""

        def memory_stats(self) -> dict[str, int]:
            """Raise RuntimeError to exercise fallback RSS calculation."""
            raise RuntimeError("stats unavailable")

    _, _, mem_fallback = jax_benchmark._run_benchmark_pass(
        model=model,
        batch_size=1,
        num_runs=1,
        warmup_steps=0,
        mode="prefill",
        max_new_tokens=1,
        device=_FailingStatsDevice(),
    )
    assert mem_fallback >= 0.0

    # Also test when nnx is None and jax lacks block_until_ready
    monkeypatch.setattr(jax_benchmark, "nnx", None)
    monkeypatch.delattr(jax_benchmark.jax, "block_until_ready", raising=False)
    _, _, _ = jax_benchmark._run_benchmark_pass(
        model=model,
        batch_size=1,
        num_runs=1,
        warmup_steps=1,
        mode="prefill",
        max_new_tokens=1,
        device=mock_device_with_stats,
    )
    monkeypatch.setattr(jax_benchmark, "nnx", nnx)

    # End-to-end benchmark_model with tiny config
    monkeypatch.setattr(Gemma4Config, "gemma4_e2b", classmethod(lambda _cls: cfg))
    res = jax_benchmark.benchmark_model(
        "tiny_model",
        "cpu",
        1,
        dtype="float32",
        mode="prefill",
        max_new_tokens=2,
        warmup_steps=1,
        num_runs=1,
    )
    assert res["status"] == "success"
    assert res["backend"] == "jax"
    assert res["tokens_per_sec"] > 0.0
    assert res["latency_ms"] > 0.0
