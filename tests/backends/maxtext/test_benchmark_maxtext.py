"""Unit and behavioral tests for benchmark maxtext."""

from typing import Any
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from gemma_4_sql.backends.maxtext import benchmark


def test_get_device() -> None:
    """Verify get device behavior."""
    mock_jax = MagicMock()
    mock_jax.devices.side_effect = lambda x: ["tpu0"] if x == "tpu" else ["gpu0"] if x == "gpu" else ["cpu0"]
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", mock_jax):
        assert benchmark._get_device("tpu") == "tpu0"
        assert benchmark._get_device("gpu") == "gpu0"
        assert benchmark._get_device("cpu") == "cpu0"


def test_get_device_runtime_error() -> None:
    """Verify get device runtime error behavior."""
    mock_jax = MagicMock()

    def mock_devices(t: Any) -> list[str]:
        """Provide mock devices test helper."""
        if t == "cpu":
            return ["cpu0"]
        raise RuntimeError()

    mock_jax.devices.side_effect = mock_devices
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", mock_jax):
        assert benchmark._get_device("tpu") == "cpu0"


def test_get_device_no_jax() -> None:
    """Verify get device no jax behavior."""
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", None):
        assert benchmark._get_device("tpu") is None


def test_run_benchmark_pass() -> None:
    """Verify run benchmark pass behavior."""
    mock_model = MagicMock()
    mock_model.apply.return_value = jnp.zeros((2, 32, 256))

    mock_jax = MagicMock()
    mock_jax.default_device.return_value.__enter__.return_value = None
    mock_jax.jit = lambda x: x
    mock_jax.random.randint.return_value = jnp.zeros((2, 32), dtype=jnp.int32)

    MagicMock()

    mock_device = MagicMock()
    mock_device.memory_stats.return_value = {"peak_bytes_in_use": 1024 * 1024 * 100}

    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", mock_jax), patch("gemma_4_sql.backends.maxtext.benchmark.jnp", jnp), patch("time.time", side_effect=[float(i) for i in range(100)]):
        # forward pass (prefill) with warmup and block_until_ready
        mock_jax.block_until_ready = MagicMock()
        tps, _mem, _mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 1, 1, "prefill", 10, mock_device)
        assert tps > 0

        # generate pass with warmup and NO block_until_ready / NO jit
        del mock_jax.block_until_ready
        del mock_jax.jit
        tps, _mem, _mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 1, 1, "generate", 1, mock_device)
        assert tps > 0

        # 0 num_runs
        tps, _mem, _mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 0, 0, "prefill", 10, mock_device)

        # exception memory_stats
        mock_device.memory_stats.side_effect = AttributeError
        tps, _mem, mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 1, 0, "prefill", 10, mock_device)
        assert mem2 > 0.0


def test_benchmark_model() -> None:
    """Verify benchmark model behavior."""
    with (
        patch("gemma_4_sql.backends.maxtext.benchmark.Gemma4Model", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.benchmark.jax", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.benchmark.jnp", MagicMock()),
        patch("gemma_4_sql.backends.maxtext.benchmark._get_device", return_value="cpu0"),
        patch("gemma_4_sql.backends.maxtext.benchmark._run_benchmark_pass", return_value=(10.0, 1.0, 2.0)),
    ):
        res = benchmark.benchmark_model("gemma4", "cpu", 2)
        assert res["status"] == "success"

        # exception
        with patch("gemma_4_sql.backends.maxtext.benchmark.Gemma4Model", side_effect=[TypeError, MagicMock()]):
            res = benchmark.benchmark_model("gemma4", "cpu", 2)
            assert res["status"] == "success"

        # jax.distributed.initialize() error
        mock_jax = MagicMock()
        mock_jax.distributed.initialize.side_effect = RuntimeError("init failed")
        with patch("gemma_4_sql.backends.maxtext.benchmark.jax", mock_jax):
            res = benchmark.benchmark_model("gemma4", "cpu", 2)
            assert res["status"] == "success"


def test_benchmark_model_error() -> None:
    """Verify benchmark model error behavior."""
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", None):
        res = benchmark.benchmark_model("gemma4", "cpu", 2)
        assert res["status"] == "backend_unavailable"
