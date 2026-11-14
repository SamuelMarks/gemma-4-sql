"""Unit and behavioral tests for JAX benchmarking."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

import pytest
from flax import nnx

import gemma_4_sql.backends.jax.benchmark as jax_benchmark
from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
from gemma_4_sql.exceptions import DependencyMissingError


class _FakeJaxDevicesModule:
    """Typed `jax` module double for `_get_device` hardware resolution tests."""

    def __init__(self, *, fail_accelerators: bool = False) -> None:
        """Initialize with optional accelerator failure flag."""
        self.fail_accelerators = fail_accelerators

    def devices(self, hw: str) -> list[str]:
        """Return synthetic device list or raise RuntimeError on accelerators.

        Returns:
            A single-element list containing a synthetic device identifier.

        Raises:
            RuntimeError: If `fail_accelerators` is enabled for GPU or TPU.

        """
        if self.fail_accelerators and hw in {"tpu", "gpu"}:
            raise RuntimeError("No accelerator")
        return [f"fake_{hw}_device"]


def test_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model raises DependencyMissingError and _get_device returns None when jax is None."""
    monkeypatch.setattr(jax_benchmark, "jax", None)
    assert jax_benchmark._get_device("cpu") is None
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing\."):
        jax_benchmark.benchmark_model("test_model", "cpu", 1)


def test_get_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_device resolves CPU/GPU/TPU devices and falls back to CPU on RuntimeError."""
    dev_cpu = jax_benchmark._get_device("cpu")
    assert dev_cpu is not None

    monkeypatch.setattr(jax_benchmark, "jax", _FakeJaxDevicesModule(fail_accelerators=False))
    assert jax_benchmark._get_device("gpu") == "fake_gpu_device"
    assert jax_benchmark._get_device("tpu") == "fake_tpu_device"

    monkeypatch.setattr(jax_benchmark, "jax", _FakeJaxDevicesModule(fail_accelerators=True))
    assert jax_benchmark._get_device("tpu") == "fake_cpu_device"


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
    from contextlib import AbstractContextManager, nullcontext

    class _FakeJaxDevice:
        """Concrete device stub implementing `JaxDeviceProtocol`."""

        def memory_stats(self) -> dict[str, int]:
            """Return deterministic peak_bytes_in_use."""
            return {"peak_bytes_in_use": 64 * 1024 * 1024}

    device_with_stats = _FakeJaxDevice()

    def _null_default_device(_d: _FakeJaxDevice | _FailingStatsDevice) -> AbstractContextManager[None]:
        """Return nullcontext for default_device."""
        return nullcontext()

    monkeypatch.setattr(jax_benchmark.jax, "default_device", _null_default_device)
    _, _, mem_stats = jax_benchmark._run_benchmark_pass(
        model=model,
        batch_size=1,
        num_runs=1,
        warmup_steps=1,
        mode="generate",
        max_new_tokens=2,
        device=device_with_stats,
    )
    assert mem_stats == 64.0

    class _FailingStatsDevice:
        """Concrete device stub raising RuntimeError from `memory_stats()`."""

        def memory_stats(self) -> dict[str, int]:
            """Raise RuntimeError to exercise fallback RSS calculation.

            Raises:
                RuntimeError: Always raised to simulate unavailable device stats.

            """
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
        device=device_with_stats,
    )
    monkeypatch.setattr(jax_benchmark, "nnx", nnx)

    # End-to-end benchmark_model with tiny config
    res = jax_benchmark.benchmark_model(
        "gemma-4-tiny",
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
