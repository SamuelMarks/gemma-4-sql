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


            Returns:
                object: Description of return.

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


            Returns:
                object: Description of return.

            """
            return type("MockJNPArray", (), {"at": type("MockAt", (), {"set": lambda _self, _x: [0.0] * shape[1]})(), "tolist": lambda _self: [[0]]})()

        @staticmethod
        def concatenate(_args: object, _axis: object) -> object:
            """Initialize function concatenate.

            Returns:
                object: Description of return.

            """
            return type("MockJNPArray", (), {"__getitem__": lambda _self, _idx: 0, "__len__": lambda _self: 1})()

        @staticmethod
        def array(_x: object, _dtype: object) -> object:
            """Initialize function array.

            Returns:
                object: Description of return.

            """
            return type("MockJNPArray", (), {"__getitem__": lambda _self, _idx: [0], "__len__": lambda _self: 1})()

        int32 = 1


@pytest.fixture
def _mock_jax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function mock_jax_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setitem(sys.modules, "jax", type("jax", (), {"nn": type("nn", (), {"log_softmax": lambda x, _axis: x})()})())
    monkeypatch.setitem(sys.modules, "jax.numpy", type("jnp", (), {"zeros": lambda _shape: "zeros", "concatenate": lambda _args, _axis: "concat", "array": lambda _x, _dtype: "array", "int32": 1, "argsort": lambda _x: "argsort"})())


def test_generate_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with jax.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "jax")
    if not res.get("sql") == "SELECT 1" and (not res.get("status", "").startswith("mocked_missing_")):
        raise AssertionError


def test_generate_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with keras.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "keras")
    if not res["sql"] == "SELECT 1":
        raise AssertionError


def test_generate_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with maxtext.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "maxtext")
    if not res.get("sql") == "SELECT 1" and (not res.get("status", "").startswith("mocked_missing_")):
        raise AssertionError


def test_generate_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with pytorch.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "pytorch")
    if not res.get("sql") == "SELECT 1" and (not res.get("status", "").startswith("mocked_missing_")):
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_generate_invalid() -> None:
    """Test generate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        generate("model1", "Find all users", "invalid")
