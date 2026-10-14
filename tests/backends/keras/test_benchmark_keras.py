"""Unit tests for Keras hardware benchmarking, device placement, and memory telemetry."""

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.keras.benchmark as bm
from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_load_keras_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _load_keras_model sets floatx and loads GemmaCausalLM preset or raises on failure."""
    monkeypatch.setattr(bm, "keras", None)
    with pytest.raises(DependencyMissingError):
        bm._load_keras_model("name", "bfloat16")

    mock_keras = MagicMock()
    monkeypatch.setattr(bm, "keras", mock_keras)

    mock_gemma_causal_lm = MagicMock()
    mock_gemma_causal_lm.from_preset.return_value = "fake_model"
    mock_keras_nlp = MagicMock(GemmaCausalLM=mock_gemma_causal_lm)

    with patch("builtins.__import__", return_value=mock_keras_nlp):
        res = bm._load_keras_model("test_model", "float32")
        assert res == "fake_model"
        mock_keras.config.set_floatx.assert_called_with("float32")

    original_import = builtins.__import__

    def mock_import_err(name: str, *args: Any, **kwargs: Any) -> Any:
        """Raise ValueError when importing keras_nlp.models."""
        if name == "keras_nlp.models":
            raise ValueError("simulated error")
        return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", mock_import_err)
    assert __import__("math") is not None
    with pytest.raises(ValueError, match="Failed to load actual model"):
        bm._load_keras_model("name", "bfloat16")


def test_get_device_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_device_str resolves TensorFlow device strings for CPU, GPU, and TPU."""
    monkeypatch.setattr(bm, "tf", None)
    assert bm._get_device_str("gpu") == "/CPU:0"

    mock_tf = MagicMock()
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]
    monkeypatch.setattr(bm, "tf", mock_tf)

    assert bm._get_device_str("cpu") == "/CPU:0"
    assert bm._get_device_str("gpu") == "/GPU:0"
    assert bm._get_device_str("tpu") == "/TPU:0"
    assert bm._get_device_str("unknown") == "/CPU:0"


def test_run_benchmark_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass computes peak memory and latency across prefill and generate modes."""
    monkeypatch.setattr(bm, "tf", None)
    with pytest.raises(DependencyMissingError):
        bm._run_benchmark_pass(MagicMock(), 1, 1, 1, "prefill", 1, "cpu")

    mock_tf = MagicMock()
    mock_tf.function = lambda *_args, **_kwargs: lambda fn: fn
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]

    class MockDevice:
        """No-op TensorFlow device context manager."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Initialize MockDevice."""

        def __enter__(self) -> None:
            """Enter device context."""

        def __exit__(self, *_args: object) -> None:
            """Exit device context."""

    mock_tf.device = MockDevice
    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 1048576 * 10}
    monkeypatch.setattr(bm, "tf", mock_tf)

    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_out.numpy.return_value = 1
    mock_model.return_value = mock_out

    tps, lat, mem = bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "gpu")
    assert tps > 0
    assert lat > 0
    assert mem == 10.0

    mock_model_gen = MagicMock()
    mock_model_gen.generate.return_value = mock_out
    _, _, mem_cpu = bm._run_benchmark_pass(mock_model_gen, 1, 1, 1, "generate", 1, "cpu")
    assert mem_cpu == 6000.0

    # Memory stats exceptions fallback to 6000.0
    mock_tf.config.experimental.reset_memory_stats.side_effect = ValueError("reset err")
    mock_tf.config.experimental.get_memory_info.side_effect = ValueError("info err")
    _, _, mem_fallback = bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "gpu")
    assert mem_fallback == 6000.0


def test_run_benchmark_pass_missing_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass handles outputs without .numpy() and models without .generate()."""
    mock_tf = MagicMock()
    mock_tf.function = lambda *_args, **_kwargs: lambda fn: fn
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]

    class MockDevice:
        """No-op TensorFlow device context manager."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Initialize MockDevice."""

        def __enter__(self) -> None:
            """Enter device context."""

        def __exit__(self, *_args: object) -> None:
            """Exit device context."""

    mock_tf.device = MockDevice
    monkeypatch.setattr(bm, "tf", mock_tf)

    mock_model = MagicMock(return_value=object())
    res1 = bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "cpu")
    assert res1[2] == 6000.0

    res2 = bm._run_benchmark_pass(object(), 1, 1, 1, "generate", 1, "cpu")
    assert res2[2] == 6000.0

    mock_model_gen = MagicMock()
    mock_model_gen.generate.return_value = object()
    res3 = bm._run_benchmark_pass(mock_model_gen, 1, 1, 1, "generate", 1, "cpu")
    assert res3[2] == 6000.0


def test_benchmark_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model checks dependencies and invokes run_benchmark_wrapper."""
    monkeypatch.setattr(bm, "keras", None)
    with pytest.raises(DependencyMissingError):
        bm.benchmark_model("m", "h", 1)

    monkeypatch.setattr(bm, "keras", MagicMock())
    monkeypatch.setattr(bm, "tf", MagicMock())
    monkeypatch.setattr(bm, "_load_keras_model", MagicMock())
    monkeypatch.setattr(bm, "_run_benchmark_pass", MagicMock(return_value=(10.0, 5.0, 128.0)))

    res = bm.benchmark_model("m", "cpu", 1, num_runs=2)
    assert res["status"] == "success"
    assert res["tokens_per_sec"] == 10.0

    def mock_run_wrapper(*_args: Any, **kwargs: Any) -> dict[str, str]:
        """Execute benchmark_fn callback and return completed status."""
        kwargs["benchmark_fn"]()
        return {"status": "completed"}

    monkeypatch.setattr(bm, "run_benchmark_wrapper", mock_run_wrapper)
    res2 = bm.benchmark_model("m", "cpu", 1, num_runs=2)
    assert res2["status"] == "completed"
