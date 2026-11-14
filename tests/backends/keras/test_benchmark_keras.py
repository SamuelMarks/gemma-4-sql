"""Unit tests for Keras hardware benchmarking, device placement, and memory telemetry without MagicMock."""

from __future__ import annotations

import types
from collections.abc import Callable

import pytest
import torch

import gemma_4_sql.backends.keras.benchmark as bm
from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_keras

pytestmark = requires_keras


def _identity_tf_function[FnT](*_args: bool, **_kwargs: bool) -> Callable[[FnT], FnT]:
    """Return an identity decorator for `tf.function`."""

    def _dec(fn: FnT) -> FnT:
        """Return `fn` unchanged."""
        return fn

    return _dec


class _DummySentinel:
    """Empty sentinel object lacking `.generate()`."""

    def __call__(self, *_args: int | str | torch.Tensor) -> str:
        """Return sentinel string when called."""
        return "sentinel"


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


class _FakeTFExperimental:
    """Typed `tf.config.experimental` double."""

    def __init__(self, *, fail_stats: bool = False) -> None:
        """Store failure flag."""
        self.fail_stats = fail_stats

    def reset_memory_stats(self, _dev: str) -> None:
        """Reset stats or raise `ValueError`.

        Raises:
            ValueError: If `fail_stats` is enabled.

        """
        if self.fail_stats:
            raise ValueError("reset err")

    def get_memory_info(self, _dev: str) -> dict[str, int]:
        """Return peak bytes or raise `ValueError`.

        Returns:
            A dictionary mapping `'peak'` to peak memory bytes.

        Raises:
            ValueError: If `fail_stats` is enabled.

        """
        if self.fail_stats:
            raise ValueError("info err")
        return {"peak": 1048576 * 10}


class _FakeTFConfig:
    """Typed `tf.config` double."""

    def __init__(self, devices: list[str] | None = None, *, fail_stats: bool = False) -> None:
        """Store physical device list."""
        self.devices = devices if devices is not None else ["GPU:0"]
        self.experimental = _FakeTFExperimental(fail_stats=fail_stats)

    def list_physical_devices(self, dev_type: str) -> list[str]:
        """Return matching physical devices."""
        return [d for d in self.devices if dev_type in d]


class _FakeTFRandom:
    """Typed `tf.random` double."""

    @staticmethod
    def set_seed(_seed: int) -> None:
        """No-op seed setter."""

    @staticmethod
    def uniform(shape: tuple[int, int], **_kw: int | str) -> torch.Tensor:
        """Return int32 ones tensor for `shape`."""
        return torch.ones(shape, dtype=torch.int32)


class _FakeTFModule:
    """Typed `tensorflow` runtime double."""

    int32 = "int32"
    function = staticmethod(_identity_tf_function)
    device = _MockDeviceCtx
    random = _FakeTFRandom

    def __init__(
        self,
        devices: list[str] | None = None,
        *,
        fail_stats: bool = False,
        fail_identity: bool = False,
    ) -> None:
        """Initialize fake TF namespaces."""
        self.config = _FakeTFConfig(devices, fail_stats=fail_stats)
        self.fail_identity = fail_identity

    def identity(self, val: torch.Tensor | str) -> torch.Tensor | str:
        """Return `val` or raise `TypeError`.

        Returns:
            The input tensor or string unchanged.

        Raises:
            TypeError: If `fail_identity` is enabled.

        """
        if self.fail_identity:
            raise TypeError("not a tensor")
        return val


def test_load_keras_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _load_keras_model sets floatx and loads GemmaCausalLM preset or raises on failure."""
    monkeypatch.setattr(bm, "keras", None)
    with pytest.raises(DependencyMissingError):
        bm._load_keras_model("name", "bfloat16")

    floatx_calls: list[str] = []

    class _FakeKerasConfig:
        """Typed `keras.config` stub."""

        @staticmethod
        def set_floatx(dt: str) -> None:
            """Record `dt`."""
            floatx_calls.append(dt)

    class _FakeKerasMod:
        """Typed `keras` stub."""

        config = _FakeKerasConfig

    class _FakeGemmaLoader:
        """Typed `GemmaCausalLM` loader stub."""

        @staticmethod
        def from_preset(_name: str) -> str:
            """Return `'fake_model'`."""
            return "fake_model"

    def _return_gemma_cls(*_a: str | type) -> type[_FakeGemmaLoader]:
        """Return `_FakeGemmaLoader`."""
        return _FakeGemmaLoader

    def _none_attr(*_a: str | type) -> None:
        """Return None for import_optional_symbol."""

    monkeypatch.setattr(bm, "keras", _FakeKerasMod())
    monkeypatch.setattr(bm, "import_optional_symbol", _return_gemma_cls)
    res = bm._load_keras_model("test_model", "float32")
    assert res == "fake_model"
    assert floatx_calls == ["float32"]

    from gemma_4_sql.exceptions import InferenceError

    monkeypatch.setattr(bm, "import_optional_symbol", _none_attr)
    with pytest.raises(InferenceError, match="Failed to load actual model"):
        bm._load_keras_model("name", "bfloat16")


def test_get_device_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_device_str resolves TensorFlow device strings for CPU, GPU, and TPU."""
    monkeypatch.setattr(bm, "tf", None)
    assert bm._get_device_str("gpu") == "/CPU:0"

    monkeypatch.setattr(bm, "tf", _FakeTFModule(["GPU:0", "TPU:0"]))
    assert bm._get_device_str("cpu") == "/CPU:0"
    assert bm._get_device_str("gpu") == "/GPU:0"
    assert bm._get_device_str("tpu") == "/TPU:0"
    assert bm._get_device_str("unknown") == "/CPU:0"


def test_run_benchmark_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass computes peak memory and latency across prefill and generate modes."""
    monkeypatch.setattr(bm, "tf", None)
    with pytest.raises(DependencyMissingError):
        bm._run_benchmark_pass(_DummySentinel(), 1, 1, 1, "prefill", 1, "cpu")

    monkeypatch.setattr(bm, "tf", _FakeTFModule(["GPU:0"]))

    class _GenerateBenchModel:
        """Typed `_KerasGenerateModelProtocol` benchmark model."""

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """Return `x`."""
            return x

        def generate(self, *args: bm.ForwardArg | list[str], **kwargs: bm.ForwardArg) -> str:
            """Return `'SELECT 1'`."""
            _ = (args, kwargs)
            return "SELECT 1"

    gen_model = _GenerateBenchModel()
    tps, lat, mem = bm._run_benchmark_pass(gen_model, 1, 1, 1, "prefill", 1, "gpu")
    assert tps > 0
    assert lat > 0
    assert mem == 10.0

    _, _, mem_cpu = bm._run_benchmark_pass(gen_model, 1, 1, 1, "generate", 1, "cpu")
    assert mem_cpu > 0.0

    monkeypatch.setattr(bm, "tf", _FakeTFModule(["GPU:0"], fail_stats=True))
    _, _, mem_fallback = bm._run_benchmark_pass(gen_model, 1, 1, 1, "prefill", 1, "gpu")
    assert mem_fallback > 0.0


def test_run_benchmark_pass_missing_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass handles models without .generate() and tf.identity failures."""
    monkeypatch.setattr(bm, "tf", _FakeTFModule(["GPU:0"], fail_identity=True))

    res1 = bm._run_benchmark_pass(_DummySentinel(), 1, 1, 1, "prefill", 1, "cpu")
    assert res1[2] > 0.0

    res2 = bm._run_benchmark_pass(_DummySentinel(), 1, 1, 1, "generate", 1, "cpu")
    assert res2[2] > 0.0

    class _NonIdentityTF:
        """TF module without `identity` method."""

        int32 = "int32"
        function = staticmethod(_identity_tf_function)
        device = _MockDeviceCtx
        random = _FakeTFRandom
        config = _FakeTFConfig([])

    monkeypatch.setattr(bm, "tf", _NonIdentityTF())
    res3 = bm._run_benchmark_pass(_DummySentinel(), 1, 1, 1, "prefill", 1, "cpu")
    assert res3[2] > 0.0


def test_run_benchmark_pass_real_keras_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_benchmark_pass executes against a real Keras layer on CPU."""
    import keras

    monkeypatch.setattr(bm, "tf", _FakeTFModule([]))
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

    def _load_identity_model(*_a: str, **_k: str) -> _DummySentinel:
        """Return `_DummySentinel` callable."""
        return _DummySentinel()

    class _KerasMarker:
        """Non-None keras marker."""

    monkeypatch.setattr(bm, "keras", _KerasMarker())
    monkeypatch.setattr(bm, "tf", _FakeTFModule([]))

    res = bm.benchmark_keras_model(
        "m",
        "cpu",
        1,
        model_loader=_load_identity_model,
        options={"num_runs": 1, "warmup_steps": 1, "max_new_tokens": 2},
    )
    assert res["status"] == "success"
    assert res["backend"] == "keras"
    assert res["tokens_per_sec"] > 0.0
    assert res["latency_ms"] > 0.0
    assert res["memory_mb"] > 0.0
