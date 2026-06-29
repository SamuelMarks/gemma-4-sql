"""Tests for SDK Inference module."""

import sys

import pytest
from gemma_4_sql.sdk.inference import generate


class MockJAX:
    """Initialize class MockJAX."""

    class Nn:
        """Initialize class nn."""

        @staticmethod
        def log_softmax(x: object, _axis: object) -> object:
            """Initialize function log_softmax.

            Args:
            ----
            x: Description of x.
            axis: Description of axis.

            """
            return x

    class Numpy:
        """Initialize class numpy."""

        @staticmethod
        def zeros(shape: object) -> object:
            """Initialize function zeros.

            Args:
            ----
            shape: Description of shape.

            """
            return type("MockJNPArray", (), {"at": type("MockAt", (), {"set": lambda _self, _x: [0.0] * shape[1]})(), "tolist": lambda _self: [[0]]})()  # type: ignore[index]

        @staticmethod
        def concatenate(_args: object, _axis: object) -> object:
            """Initialize function concatenate.

            Args:
            ----
            args: Description of args.
            axis: Description of axis.

            """
            return type("MockJNPArray", (), {"__getitem__": lambda _self, _idx: 0, "__len__": lambda _self: 1})()

        @staticmethod
        def array(_x: object, _dtype: object) -> object:
            """Initialize function array.

            Args:
            ----
            x: Description of x.
            dtype: Description of dtype.

            """
            return type("MockJNPArray", (), {"__getitem__": lambda _self, _idx: [0], "__len__": lambda _self: 1})()

        int32 = 1


@pytest.fixture()
def _mock_jax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function mock_jax_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setitem(sys.modules, "jax", type("jax", (), {"nn": type("nn", (), {"log_softmax": lambda x, _axis: x})()})())
    monkeypatch.setitem(sys.modules, "jax.numpy", type("jnp", (), {"zeros": lambda _shape: "zeros", "concatenate": lambda _args, _axis: "concat", "array": lambda _x, _dtype: "array", "int32": 1, "argsort": lambda _x: "argsort"})())


def test_generate_jax() -> None:
    """Test generate with jax."""
    res = generate("model1", "Find all users", "jax")
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError


def test_generate_keras() -> None:
    """Test generate with keras."""
    res = generate("model1", "Find all users", "keras")
    if not res["status"] == "success":
        raise AssertionError


def test_generate_maxtext() -> None:
    """Test generate with maxtext."""
    res = generate("model1", "Find all users", "maxtext")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


def test_generate_pytorch() -> None:
    """Test generate with pytorch."""
    res = generate("model1", "Find all users", "pytorch")
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError


def test_generate_invalid() -> None:
    """Test generate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        generate("model1", "Find all users", "invalid")
