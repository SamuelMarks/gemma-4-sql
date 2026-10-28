"""Unit tests for Keras hardware benchmarking, device placement, and memory telemetry."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.keras.benchmark as bm
from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_keras

pytestmark = requires_keras


class _DummySentinel:
    """Empty sentinel object lacking `.generate()`."""


class _MockDeviceCtx:
    """No-op TensorFlow device context manager."""

    def __init__(self, _device_name: str = "") -> None:
        """Initialize _MockDeviceCtx."""
        self.device_name = _device_name

    def __enter__(self) -> None:
        """Enter device context."""

    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_val: BaseException | None = None,
        _exc_tb: types.TracebackType | None = None,
    ) -> None:
        """Exit device context."""


def test_load_keras_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _load_keras_model sets floatx and loads GemmaCausalLM preset or raises on failure."""
    monkeypatch.setattr(bm, "keras", None)
    with pytest.raises(DependencyMissingError):
        bm._load_keras_model("name", "bfloat16")

    mock_keras = MagicMock()
    monkeypatch.setattr(bm, "keras", mock_keras)

    mock_gemma_causal_lm = MagicMock()
    mock_gemma_causal_lm.from_preset.return_value = "fake_model"

    monkeypatch.setattr(bm, "optional_attr", lambda *_a, **_k: mock_gemma_causal_lm)  # pyright: ignore[reportUnknownLambdaType]
    res = bm._load_keras_model("test_model", "float32")
    assert res == "fake_model"
    mock_keras.config.set_floatx.assert_called_with("float32")

    monkeypatch.setattr(bm, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]
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
    mock_tf.function = lambda *_args, **_kwargs: lambda fn: fn  # pyright: ignore[reportUnknownLambdaType]
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]
    mock_tf.device = _MockDeviceCtx
    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 1048576 * 10}
    monkeypatch.setattr(bm, "tf", mock_tf)

    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_model.return_value = mock_out

    tps, lat, mem = bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "gpu")
    assert tps > 0
    assert lat > 0
    assert mem == 10.0

    mock_model_gen = MagicMock()
    mock_model_gen.generate.return_value = mock_out
    _, _, mem_cpu = bm._run_benchmark_pass(mock_model_gen, 1, 1, 1, "generate", 1, "cpu")
    assert mem_cpu > 0.0

    # Memory stats exceptions fallback to RSS memory
    mock_tf.config.experimental.reset_memory_stats.side_effect = ValueError("reset err")
    mock_tf.config.experimental.get_memory_info.side_effect = ValueError("info err")
    _, _, mem_fallback = bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "gpu")
    assert mem_fallback > 0.0


def test_run_benchmark_pass_missing_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass handles models without .generate() and tf.identity failures."""
    mock_tf = MagicMock()
    mock_tf.function = lambda *_args, **_kwargs: lambda fn: fn  # pyright: ignore[reportUnknownLambdaType]
    mock_tf.identity.side_effect = TypeError("not a tensor")
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]
    mock_tf.device = _MockDeviceCtx
    monkeypatch.setattr(bm, "tf", mock_tf)

    mock_model = MagicMock(return_value=_DummySentinel())
    res1 = bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "cpu")
    assert res1[2] > 0.0

    res2 = bm._run_benchmark_pass(_DummySentinel(), 1, 1, 1, "generate", 1, "cpu")
    assert res2[2] > 0.0

    mock_model_gen = MagicMock()
    mock_model_gen.generate.return_value = _DummySentinel()
    res3 = bm._run_benchmark_pass(mock_model_gen, 1, 1, 1, "generate", 1, "cpu")
    assert res3[2] > 0.0


def test_run_benchmark_pass_real_keras_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass executes against a real Keras layer on CPU."""
    import keras

    mock_tf = MagicMock()
    mock_tf.function = lambda *_a, **_kw: lambda fn: fn  # pyright: ignore[reportUnknownLambdaType]
    mock_tf.device = _MockDeviceCtx
    mock_tf.random.uniform = lambda shape, **_kw: keras.ops.ones(shape, dtype="int32")  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(bm, "tf", mock_tf)

    emb = keras.layers.Embedding(input_dim=256000, output_dim=8)
    tps, lat, mem = bm._run_benchmark_pass(emb, 1, 1, 1, "prefill", 2, "cpu")
    assert tps > 0.0
    assert lat > 0.0
    assert mem > 0.0


def test_benchmark_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model checks dependencies and executes through run_benchmark_wrapper on CPU."""
    monkeypatch.setattr(bm, "keras", None)
    with pytest.raises(DependencyMissingError):
        bm.benchmark_model("m", "h", 1)

    mock_tf = MagicMock()
    mock_tf.function = lambda *_a, **_kw: lambda fn: fn  # pyright: ignore[reportUnknownLambdaType]
    mock_tf.device = _MockDeviceCtx
    monkeypatch.setattr(bm, "keras", MagicMock())
    monkeypatch.setattr(bm, "tf", mock_tf)
    monkeypatch.setattr(bm, "_load_keras_model", lambda *_a, **_k: lambda x: x)  # pyright: ignore[reportUnknownLambdaType]

    res = bm.benchmark_model("m", "cpu", 1, num_runs=1, warmup_steps=1, max_new_tokens=2)
    assert res["status"] == "success"
    assert res["backend"] == "keras"
    assert res["tokens_per_sec"] > 0.0
    assert res["latency_ms"] > 0.0
    assert res["memory_mb"] > 0.0
