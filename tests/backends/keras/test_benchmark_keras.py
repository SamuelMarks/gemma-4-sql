"""Tests for Keras Benchmark."""

import pytest

import gemma_4_sql.backends.keras.benchmark as bm


class MockTfTensor:
    """Provide class docstring."""

    def numpy(self) -> float:
        """Execute function."""
        return 0.0


class MockTf:
    """Provide class docstring."""

    int32 = "int32"

    def zeros(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return MockTfTensor()

    def function(self, fn: object) -> object:
        """Execute function."""
        return fn

    class MockConfig:
        """Provide class docstring."""

        class MockExperimental:
            """Provide class docstring."""

            @staticmethod
            def get_memory_info(_device: str) -> dict:
                """Execute function."""
                return {"current": 1024 * 1024 * 100}

        experimental = MockExperimental

    config = MockConfig


def test_benchmark_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(bm, "keras", None)
    res = bm.benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError


def test_benchmark_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", object())
    res = bm.benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", object())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockTf, "zeros", raise_err)
    res = bm.benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockKeras:
    """Provide class docstring."""

    def mock_input(*_args: object, **_kwargs: object) -> None:
        """Execute function."""
        return

    Input = mock_input

    class MockLayers:
        """Provide class docstring."""

        class Embedding:
            """Provide class docstring."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Execute function."""

            def __call__(self, x: object) -> object:
                """Execute function."""
                return x

        class Dense:
            """Provide class docstring."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Execute function."""

            def __call__(self, x: object) -> object:
                """Execute function."""
                return x

    layers = MockLayers

    def mock_model(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return lambda x: x

    Model = mock_model


def test_benchmark_keras_real_no_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", MockKeras())
    res = bm.benchmark_model("model", "gpu", 1)
    if res["status"] != "success":
        raise AssertionError


def test_benchmark_keras_real_mem(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    mock_tf = MockTf()
    mock_tf.config = type("MockConfig", (), {"experimental": type("MockExp", (), {"get_memory_info": lambda _x: {"current": 1024 * 1024}})})()
    monkeypatch.setattr(bm, "tf", mock_tf)
    monkeypatch.setattr(bm, "keras", MockKeras())
    res = bm.benchmark_model("model", "gpu", 1)
    if res["status"] != "success":
        raise AssertionError
