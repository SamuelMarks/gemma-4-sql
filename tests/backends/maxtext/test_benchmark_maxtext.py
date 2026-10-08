"""Module docstring."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from gemma_4_sql.backends.maxtext import benchmark


def test_get_device():
    """Docstring for test_get_device."""
    mock_jax = MagicMock()
    mock_jax.devices.side_effect = lambda x: ["tpu0"] if x == "tpu" else ["gpu0"] if x == "gpu" else ["cpu0"]
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", mock_jax):
        assert benchmark._get_device("tpu") == "tpu0"
        assert benchmark._get_device("gpu") == "gpu0"
        assert benchmark._get_device("cpu") == "cpu0"


def test_get_device_runtime_error():
    """Docstring for test_get_device_runtime_error."""
    mock_jax = MagicMock()

    def mock_devices(t):
        """Docstring for mock_devices."""
        if t == "cpu":
            return ["cpu0"]
        raise RuntimeError()

    mock_jax.devices.side_effect = mock_devices
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", mock_jax):
        assert benchmark._get_device("tpu") == "cpu0"


def test_get_device_no_jax():
    """Docstring for test_get_device_no_jax."""
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", None):
        assert benchmark._get_device("tpu") is None


def test_run_benchmark_pass():
    """Docstring for test_run_benchmark_pass."""
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
        tps, mem, mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 1, 1, "prefill", 10, mock_device)
        assert tps > 0

        # generate pass with warmup and NO block_until_ready / NO jit
        del mock_jax.block_until_ready
        del mock_jax.jit
        tps, mem, mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 1, 1, "generate", 1, mock_device)
        assert tps > 0

        # 0 num_runs
        tps, mem, mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 0, 0, "prefill", 10, mock_device)

        # exception memory_stats
        mock_device.memory_stats.side_effect = AttributeError
        tps, _mem, mem2 = benchmark._run_benchmark_pass(mock_model, {}, 2, 1, 0, "prefill", 10, mock_device)
        assert mem2 == 16384.0


def test_benchmark_model():
    """Docstring for test_benchmark_model."""
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


def test_benchmark_model_error():
    """Docstring for test_benchmark_model_error."""
    with patch("gemma_4_sql.backends.maxtext.benchmark.jax", None):
        res = benchmark.benchmark_model("gemma4", "cpu", 2)
        assert res["status"] == "mocked_missing_maxtext"


def test_import_error():
    """Docstring for test_import_error."""
    import importlib

    with patch.dict("sys.modules", {"jax": None}):
        importlib.reload(benchmark)
        assert benchmark.jax is None
    # Restore
    importlib.reload(benchmark)


def test_import_success():
    """Docstring for test_import_success."""
    import importlib

    mock_jax = MagicMock()
    mock_jnp = MagicMock()

    mock_maxtext = MagicMock()
    mock_maxtext.models.gemma4.Gemma4Model = "FakeGemma"

    with patch.dict(
        "sys.modules",
        {
            "jax": mock_jax,
            "jax.numpy": mock_jnp,
            "maxtext": mock_maxtext,
            "maxtext.models": mock_maxtext.models,
            "maxtext.models.gemma4": mock_maxtext.models.gemma4,
        },
    ):
        importlib.reload(benchmark)
        assert benchmark.Gemma4Model == "FakeGemma"

    # Restore
    importlib.reload(benchmark)
