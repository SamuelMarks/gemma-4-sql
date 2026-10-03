"""Tests for Keras benchmark."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras.benchmark import _get_device_str, _load_keras_model, _run_benchmark_pass, benchmark_model
from gemma_4_sql.exceptions import DependencyMissingError


def test_get_device_str_no_tf():
    with patch("gemma_4_sql.backends.keras.benchmark.tf", None):
        assert _get_device_str("gpu") == "/CPU:0"


def test_get_device_str_with_tf():
    mock_tf = MagicMock()
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]
    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf):
        assert _get_device_str("cpu") == "/CPU:0"
        assert _get_device_str("gpu") == "/GPU:0"
        mock_tf.config.list_physical_devices.return_value = []
        assert _get_device_str("gpu") == "/CPU:0"
        mock_tf.config.list_physical_devices.return_value = ["TPU:0"]
        assert _get_device_str("tpu") == "/TPU:0"
        assert _get_device_str("unknown") == "/CPU:0"


def test_load_keras_model_missing_deps():
    with patch("gemma_4_sql.backends.keras.benchmark.keras", None), pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        _load_keras_model("test", "bfloat16")


def test_load_keras_model_import_error():
    mock_keras = MagicMock()
    with patch("gemma_4_sql.backends.keras.benchmark.keras", mock_keras), patch("builtins.__import__", side_effect=ImportError), pytest.raises(ValueError, match="Failed to load actual model"):
        _load_keras_model("test", "bfloat16")


def test_load_keras_model_success():
    mock_keras = MagicMock()
    mock_gemma_causal_lm_cls = MagicMock()
    mock_model = MagicMock()
    mock_gemma_causal_lm_cls.GemmaCausalLM.from_preset.return_value = mock_model

    with patch("gemma_4_sql.backends.keras.benchmark.keras", mock_keras), patch("builtins.__import__", return_value=mock_gemma_causal_lm_cls):
        model = _load_keras_model("test", "bfloat16")
        assert model == mock_model
        mock_keras.config.set_floatx.assert_called_with("bfloat16")


def test_run_benchmark_pass_missing_tf():
    with patch("gemma_4_sql.backends.keras.benchmark.tf", None), pytest.raises(DependencyMissingError, match="TensorFlow dependencies are missing"):
        _run_benchmark_pass(None, 1, 1, 1, "prefill", 10, "cpu")


def test_run_benchmark_pass_prefill():
    mock_tf = MagicMock()
    mock_tf.random.uniform.return_value = MagicMock()
    mock_model = MagicMock()

    def mock_function(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    mock_tf.function = mock_function
    mock_out = MagicMock()
    mock_out.numpy.return_value = None
    mock_model.return_value = mock_out

    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/CPU:0"):
        tps, lat, mem = _run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 10, "cpu")
        assert isinstance(tps, float)
        assert isinstance(lat, float)
        assert mem == 6000.0


def test_run_benchmark_pass_generate_gpu():
    mock_tf = MagicMock()
    mock_tf.random.uniform.return_value = MagicMock()
    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 1024 * 1024}  # 1MB
    mock_model = MagicMock()

    def mock_function(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    mock_tf.function = mock_function
    mock_out = MagicMock()
    mock_out.numpy.return_value = None
    mock_model.generate.return_value = mock_out

    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/GPU:0"):
        tps, lat, mem = _run_benchmark_pass(mock_model, 1, 1, 1, "generate", 10, "gpu")
        assert isinstance(tps, float)
        assert isinstance(lat, float)
        assert mem == 1.0


def test_run_benchmark_pass_generate_gpu_value_error():
    mock_tf = MagicMock()
    mock_tf.random.uniform.return_value = MagicMock()
    mock_tf.config.experimental.get_memory_info.side_effect = ValueError
    mock_tf.config.experimental.reset_memory_stats.side_effect = ValueError
    mock_model = MagicMock()

    def mock_function(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    mock_tf.function = mock_function

    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/GPU:0"):
        _tps, _lat, mem = _run_benchmark_pass(mock_model, 1, 1, 1, "generate", 10, "gpu")
        assert mem == 6000.0


def test_run_benchmark_pass_branches():
    mock_tf = MagicMock()
    mock_tf.random.uniform.return_value = MagicMock()

    def mock_function(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    mock_tf.function = mock_function

    # Case 1: mode="prefill", out does NOT have numpy
    mock_model = MagicMock()

    class OutNoNumpy:
        pass

    mock_model.return_value = OutNoNumpy()
    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/CPU:0"):
        _run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 10, "cpu")

    # Case 2: mode="generate", model does NOT have generate
    class ModelNoGenerate:
        def __call__(self, *args, **kwargs):
            return OutNoNumpy()

    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/CPU:0"):
        _run_benchmark_pass(ModelNoGenerate(), 1, 1, 1, "generate", 10, "cpu")

    # Case 3: mode="generate", model has generate, out does NOT have numpy
    mock_model = MagicMock()
    mock_model.generate.return_value = OutNoNumpy()
    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/CPU:0"):
        _run_benchmark_pass(mock_model, 1, 1, 1, "generate", 10, "cpu")

    # Case 4: GPU without reset_memory_stats value error
    mock_tf.config.experimental.reset_memory_stats = MagicMock()
    mock_tf.config.experimental.get_memory_info.return_value = {"current": 2048 * 1024}  # 2MB
    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_out.numpy.return_value = None
    mock_model.return_value = mock_out
    with patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf), patch("gemma_4_sql.backends.keras.benchmark._get_device_str", return_value="/GPU:0"):
        _tps, _lat, mem = _run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 10, "gpu")
        assert mem == 2.0


def test_benchmark_model_missing_deps():
    with patch("gemma_4_sql.backends.keras.benchmark.keras", None), pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        benchmark_model("test", "cpu", 1)


def test_benchmark_model_success():
    mock_keras = MagicMock()
    mock_tf = MagicMock()

    def mock_run_benchmark_wrapper(backend_name, model_name, hardware, batch_size, missing_deps, missing_status, benchmark_fn):
        benchmark_fn()
        return {"status": "ok"}

    with (
        patch("gemma_4_sql.backends.keras.benchmark.keras", mock_keras),
        patch("gemma_4_sql.backends.keras.benchmark.tf", mock_tf),
        patch("gemma_4_sql.backends.keras.benchmark.run_benchmark_wrapper", mock_run_benchmark_wrapper),
        patch("gemma_4_sql.backends.keras.benchmark._load_keras_model") as mock_load,
        patch("gemma_4_sql.backends.keras.benchmark._run_benchmark_pass") as mock_pass,
    ):
        mock_load.return_value = MagicMock()
        mock_pass.return_value = (1.0, 1.0, 1.0)

        res = benchmark_model("test", "cpu", 1, dtype="float32", mode="generate", max_new_tokens=10, warmup_steps=1, num_runs=2)
        assert res == {"status": "ok"}
        mock_load.assert_called_once_with("test", dtype="float32")
        mock_pass.assert_called_once()
