"""Tests for Keras Benchmark."""

import pytest

import gemma_4_sql.backends.keras.benchmark as bm


class MockTfTensor:
    """Provide class docstring."""

    def numpy(self) -> float:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 0.0


class MockTf:
    """Provide class docstring."""

    int32 = "int32"

    class MockRandom:
        """Provide class docstring."""

        @staticmethod
        def uniform(*_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return MockTfTensor()

    random = MockRandom

    def zeros(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTfTensor()

    def function(self, fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn

    class MockConfig:
        """Provide class docstring."""

        class MockExperimental:
            """Provide class docstring."""

            @staticmethod
            def get_memory_info(_device: str) -> dict:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return {"current": 1024 * 1024 * 100}

        experimental = MockExperimental

    config = MockConfig


def test_benchmark_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "keras", None)
    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing."):
        bm.benchmark_model("model", "gpu", 1)


def test_benchmark_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", MockKeras())
    res = bm.benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if "failed" not in res["status"]:
        raise AssertionError


def test_benchmark_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", object())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
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
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return x

        class Dense:
            """Provide class docstring."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Execute function."""

            def __call__(self, x: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return x

    layers = MockLayers

    def mock_model(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return lambda x: x

    Model = mock_model


def test_benchmark_keras_real_no_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    import sys

    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": type("MockGemma", (), {"from_preset": lambda *a, **k: MockKeras.Model()})}))

    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Test function."""
        if name == "keras_nlp.models":
            return sys.modules["keras_nlp.models"]
        return orig_import(name, _globals, _locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", MockKeras())
    bm.benchmark_model("model", "gpu", 1)


def test_benchmark_keras_real_mem(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    import sys

    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": type("MockGemma", (), {"from_preset": lambda *a, **k: MockKeras.Model()})}))

    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Test function."""
        if name == "keras_nlp.models":
            return sys.modules["keras_nlp.models"]
        return orig_import(name, _globals, _locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    mock_tf = MockTf()

    def mock_get_memory_info(_x: str) -> dict:
        """Test function."""
        msg = "No memory info"
        raise ValueError(msg)

    mock_tf.config = type("MockConfig", (), {"experimental": type("MockExp", (), {"get_memory_info": mock_get_memory_info})})()
    monkeypatch.setattr(bm, "tf", mock_tf)
    monkeypatch.setattr(bm, "keras", MockKeras())
    bm.benchmark_model("model", "gpu", 1)
