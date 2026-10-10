"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.keras.benchmark as mod
from gemma_4_sql.exceptions import DependencyMissingError


def test_keras_benchmark_import_error():
    """Docstring for test_keras_benchmark_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp" or name == "keras":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.keras is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_load_keras_model():
    """Docstring for test_load_keras_model."""
    mock_keras = MagicMock()
    mock_keras.config.set_floatx = MagicMock()

    with patch("gemma_4_sql.backends.keras.benchmark.keras", mock_keras):
        # Mock __import__ to return our fake GemmaCausalLM
        mock_gemma_causal_lm = MagicMock()
        mock_gemma_causal_lm.from_preset.return_value = "fake_model"

        mock_keras_nlp = MagicMock()
        mock_keras_nlp.GemmaCausalLM = mock_gemma_causal_lm

        with patch("builtins.__import__", return_value=mock_keras_nlp):
            res = mod._load_keras_model("test_model", "float32")
            assert res == "fake_model"
            mock_keras.config.set_floatx.assert_called_with("float32")

        # Test ImportError
        with patch("builtins.__import__", side_effect=ImportError("mock")):
            with pytest.raises(ValueError, match="Failed to load actual model"):
                mod._load_keras_model("test_model", "float32")


def test_load_keras_model_missing():
    """Docstring for test_load_keras_model_missing."""
    with patch("gemma_4_sql.backends.keras.benchmark.keras", None):
        with pytest.raises(DependencyMissingError):
            mod._load_keras_model("test_model", "float32")


def test_get_device_str():
    # Test when tf is None
    """Docstring for test_get_device_str."""
    with patch("gemma_4_sql.backends.keras.benchmark.tf", None):
        assert mod._get_device_str("gpu") == "/CPU:0"

    mock_tf = MagicMock()

    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf):
        # Test CPU
        assert mod._get_device_str("cpu") == "/CPU:0"

        # Test GPU
        mock_tf.config.list_physical_devices.return_value = ["GPU:0"]
        assert mod._get_device_str("gpu") == "/GPU:0"

        # Test TPU
        assert mod._get_device_str("tpu") == "/TPU:0"

        # Test fallback
        assert mod._get_device_str("unknown") == "/CPU:0"


def test_run_benchmark_pass_missing_tf():
    """Docstring for test_run_benchmark_pass_missing_tf."""
    with patch("gemma_4_sql.backends.keras.benchmark.tf", None):
        with pytest.raises(DependencyMissingError):
            mod._run_benchmark_pass(MagicMock(), 1, 1, 1, "prefill", 1, "cpu")


def test_run_benchmark_pass():
    """Docstring for test_run_benchmark_pass."""
    mock_tf = MagicMock()
    mock_model = MagicMock()

    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf):
        # Setup mocks
        mock_tf.device = MagicMock()
        mock_tf.function = lambda *args, **kwargs: lambda f: f

        # Test CPU prefill
        with patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/CPU:0"):
            tps, lat, mem = mod._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "cpu")
            assert mem == 6000.0
            assert tps > 0
            assert lat > 0

        # Test GPU generate
        with patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/GPU:0"):
            mock_tf.config.experimental.get_memory_info.return_value = {"peak": 1048576 * 10}  # 10MB
            tps, lat, mem = mod._run_benchmark_pass(mock_model, 1, 1, 1, "generate", 1, "gpu")
            assert mem == 10.0

            # Test memory error fallback
            mock_tf.config.experimental.get_memory_info.side_effect = ValueError()
            tps, lat, mem = mod._run_benchmark_pass(mock_model, 1, 1, 1, "generate", 1, "gpu")
            assert mem == 6000.0


def test_run_benchmark_pass_gpu_reset_error():
    """Docstring for test_run_benchmark_pass_gpu_reset_error."""
    mock_tf = MagicMock()
    mock_model = MagicMock()

    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf):
        mock_tf.device = MagicMock()
        mock_tf.function = lambda *args, **kwargs: lambda f: f
        mock_tf.config.experimental.reset_memory_stats.side_effect = ValueError()

        with patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/GPU:0"):
            mod._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "gpu")


def test_benchmark_model():
    """Docstring for test_benchmark_model."""
    mock_keras = MagicMock()
    mock_tf = MagicMock()

    with patch("gemma_4_sql.backends.keras.benchmark.keras", mock_keras), patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._load_keras_model"), patch("gemma_4_sql.backends.keras.benchmark._run_benchmark_pass", return_value=(1.0, 2.0, 3.0)):
        res = mod.benchmark_model("test_model", "cpu", 1)
        assert res["status"] == "success"
        assert res["tokens_per_sec"] == 1.0


def test_benchmark_model_missing():
    """Docstring for test_benchmark_model_missing."""
    with patch("gemma_4_sql.backends.keras.benchmark.keras", None):
        with pytest.raises(DependencyMissingError):
            mod.benchmark_model("test_model", "cpu", 1)
