"""
Tests for SDK Inference module.
"""

import sys

import pytest

from gemma_4_sql.sdk.inference import generate


class MockJAX:
    class nn:
        @staticmethod
        def log_softmax(x, axis):
            return x

    class numpy:
        @staticmethod
        def zeros(shape):
            return type(
                "MockJNPArray",
                (),
                {
                    "at": type(
                        "MockAt", (), {"set": lambda self, x: [0.0] * shape[1]}
                    )(),
                    "tolist": lambda self: [[0]],
                },
            )()

        @staticmethod
        def concatenate(args, axis):
            return type(
                "MockJNPArray",
                (),
                {"__getitem__": lambda self, idx: 0, "__len__": lambda self: 1},
            )()

        @staticmethod
        def array(x, dtype):
            return type(
                "MockJNPArray",
                (),
                {"__getitem__": lambda self, idx: [0], "__len__": lambda self: 1},
            )()

        int32 = 1


@pytest.fixture
def mock_jax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "jax",
        type("jax", (), {"nn": type("nn", (), {"log_softmax": lambda x, axis: x})()})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "jax.numpy",
        type(
            "jnp",
            (),
            {
                "zeros": lambda shape: "zeros",
                "concatenate": lambda args, axis: "concat",
                "array": lambda x, dtype: "array",
                "int32": 1,
                "argsort": lambda x: "argsort",
            },
        )(),
    )


def test_generate_jax() -> None:
    """Test generate with jax."""
    # Test missing JAX
    res = generate("model1", "Find all users", "jax")
    assert res["status"] == "mocked_missing_jax"


def test_generate_keras() -> None:
    """Test generate with keras."""
    res = generate("model1", "Find all users", "keras")
    assert res["status"] == "mocked_missing_keras"


def test_generate_maxtext() -> None:
    """Test generate with maxtext."""
    res = generate("model1", "Find all users", "maxtext")
    assert res["status"] == "mocked_missing_maxtext"


def test_generate_pytorch() -> None:
    """Test generate with pytorch."""
    res = generate("model1", "Find all users", "pytorch")
    assert res["status"] == "mocked_missing_torch"


def test_generate_invalid() -> None:
    """Test generate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        generate("model1", "Find all users", "invalid")
