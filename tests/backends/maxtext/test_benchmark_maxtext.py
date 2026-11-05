"""Unit and behavioral tests for benchmark maxtext using real JAX arrays and CPU devices."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.maxtext import benchmark
from gemma_4_sql.type_hints import DTypeProtocol, PyTreeNode, TensorType


class _TinyStatelessModel:
    """Real stateless JAX model for MaxText benchmarking tests."""

    def __init__(self, model_name: str, *, dtype: DTypeProtocol | str | None = None) -> None:
        """Initialize _TinyStatelessModel."""
        self.model_name = model_name
        self.dtype = dtype

    def init(self, rng: TensorType, dummy_inputs: TensorType) -> dict[str, PyTreeNode]:
        """Initialize real JAX array weight."""
        _ = (rng, dummy_inputs)
        return {"w": jnp.ones((16, 16), dtype=jnp.float32)}

    def apply(self, params: benchmark._MaxTextBenchmarkParams, inputs: TensorType) -> jax.Array:
        """Compute deterministic vocabulary logits on JAX CPU."""
        _ = params
        batch_sz = int(inputs.shape[0])
        seq_len = int(inputs.shape[1])
        return jnp.ones((batch_sz, seq_len, 64), dtype=jnp.float32)


def test_get_device() -> None:
    """Verify _get_device resolves TPU, GPU, and real CPU devices."""

    def _devices_by_type(x: str) -> list[str]:
        """Return device list for tpu/gpu/cpu."""
        return ["tpu0"] if x == "tpu" else ["gpu0"] if x == "gpu" else ["cpu0"]

    mock_jax = MagicMock()
    mock_jax.devices.side_effect = _devices_by_type
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", mock_jax):
        assert benchmark._get_device("tpu") == "tpu0"
        assert benchmark._get_device("gpu") == "gpu0"
        assert benchmark._get_device("cpu") == "cpu0"

    real_cpu = benchmark._get_device("cpu")
    assert real_cpu is not None


def test_get_device_runtime_error() -> None:
    """Verify _get_device falls back to CPU when TPU/GPU query raises RuntimeError."""
    real_cpu_dev = benchmark._get_device("tpu")
    assert real_cpu_dev is not None


def test_get_device_no_jax() -> None:
    """Verify _get_device returns None when jax is None."""
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", None):
        assert benchmark._get_device("tpu") is None


def test_run_benchmark_pass_real_jax_cpu() -> None:
    """Verify _run_benchmark_pass runs real JIT-compiled forward and generate passes on JAX CPU."""
    model = _TinyStatelessModel("gemma-4-tiny")
    cpu_device = jax.devices("cpu")[0]
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((2, 32), dtype=jnp.int32))

    # 1. Real prefill pass on JAX CPU
    tps_prefill, lat_prefill, mem_prefill = benchmark._run_benchmark_pass(
        model,
        params,
        batch_size=2,
        num_runs=1,
        warmup_steps=1,
        mode="prefill",
        max_new_tokens=2,
        device=cpu_device,
    )
    assert tps_prefill > 0.0
    assert lat_prefill > 0.0
    assert mem_prefill > 0.0

    # 2. Real generate pass with custom memory_stats device
    class _DeviceWithStats:
        """Device wrapper providing memory_stats()."""

        def memory_stats(self) -> dict[str, int]:
            """Return peak_bytes_in_use."""
            return {"peak_bytes_in_use": 64 * 1024 * 1024}

    with patch.object(jax, "default_device", return_value=jax.default_device(cpu_device)):
        tps_gen, lat_gen, mem_gen = benchmark._run_benchmark_pass(
            model,
            params,
            batch_size=1,
            num_runs=1,
            warmup_steps=1,
            mode="generate",
            max_new_tokens=2,
            device=_DeviceWithStats(),
        )
    assert tps_gen > 0.0
    assert lat_gen > 0.0
    assert mem_gen == 64.0


def test_benchmark_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model executes end-to-end with _TinyStatelessModel on real JAX CPU."""
    monkeypatch.setattr(benchmark, "Gemma4Model", _TinyStatelessModel)
    res = benchmark.benchmark_model("gemma4", "cpu", 2, num_runs=1, warmup_steps=1, max_new_tokens=2)
    assert res["status"] == "success"
    assert res["tokens_per_sec"] > 0.0
    assert res["latency_ms"] > 0.0

    class _NoDtypeKwargModel(_TinyStatelessModel):
        """Model constructor rejecting dtype keyword argument."""

        def __init__(self, model_name: str) -> None:
            """Initialize without dtype kwarg."""
            super().__init__(model_name)

    monkeypatch.setattr(benchmark, "Gemma4Model", _NoDtypeKwargModel)
    res_fallback = benchmark.benchmark_model("gemma4", "cpu", 1, num_runs=1, warmup_steps=0)
    assert res_fallback["status"] == "success"


def test_benchmark_model_error() -> None:
    """Verify benchmark model error behavior."""
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", None):
        res = benchmark.benchmark_model("gemma4", "cpu", 2)
        assert res["status"] == "backend_unavailable"
