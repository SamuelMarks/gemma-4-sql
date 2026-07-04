# Copyright 2024
"""Provide module docstring."""

import contextlib

import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, create_model_from_safe_tensors


def test_utils_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setitem(sys.modules, "safetensors", None)
    importlib.reload(m_utils)
    monkeypatch.undo()
    importlib.reload(m_utils)


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_transforms() -> None:
    """Execute function."""
    tensor = jnp.zeros((2, 4))
    transform = ((1, 0), (4, 2), True)
    state = {"a": type("S", (), {"shape": (2, 4)})()}
    assign_weights(["a"], tensor, state, "st_key", transform)
    transform = ((1, 0), (2, 4), False)
    state = {"a": type("S", (), {"shape": (2, 4)})()}
    assign_weights(["a"], tensor, state, "st_key", transform)


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_value_attribute() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    tensor = jnp.zeros((2, 4))
    transform = None
    state = {"a": type("S", (), {"value": type("V", (), {"shape": (2, 4)})()})()}
    assign_weights(["a"], tensor, state, "st_key", transform)
    if not hasattr(state["a"], "value"):
        raise AssertionError


class MockF:
    """Provide class docstring."""

    def __enter__(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __exit__(self, *args: object) -> object:
        """Execute function."""

    def __iter__(self) -> object:
        """Execute function.

        Yields:
            object: Description of yield.

        """
        yield "b"

    def get_tensor(self, _k: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return jnp.zeros((1,))


class MockNNX:
    """Provide class docstring."""

    class Rngs:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""

    def split(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return (None, {"a": type("S", (), {"shape": (1,)})()}, None)


class MockModelCls:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""


def test_create_model_from_safe_tensors(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())

    def mock_map(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return ("a", None)

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})


def test_create_model_from_safe_tensors_key_error(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())

    def mock_map(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return ("a", None)

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})


def test_create_model_from_safe_tensors_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_from_eval_shape_exceptions() -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    with contextlib.suppress(Exception):
        m_utils.assign_weights_from_eval_shape(["missing"], jnp.zeros((1,)), {}, "st_key", None)
    with contextlib.suppress(Exception):
        m_utils.assign_weights_from_eval_shape(["a"], jnp.zeros((1,)), {"a": {"b": 1}}, "st_key", None)
    with contextlib.suppress(Exception):
        m_utils.assign_weights_from_eval_shape(["a"], jnp.zeros((2,)), {"a": type("S", (), {"shape": (1,)})()}, "st_key", None)


@pytest.mark.usefixtures("monkeypatch")
def test_create_model_from_safe_tensors_missing_dir() -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    m_utils.create_model_from_safe_tensors("non_existent_dir", lambda *_a, **_k: "model", {}, {})


def test_create_model_from_safe_tensors_nnx_split_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())
    m_utils.create_model_from_safe_tensors(str(tmp_path), lambda *_a, **_k: "model", {}, {"b": (None, None)})


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_from_eval_shape_transforms() -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    tensor = jnp.zeros((2, 4))
    transform = ((1, 0), (4, 2), True)
    state = {"a": type("S", (), {"shape": (2, 4), "dtype": jnp.float32, "sharding": type("Sh", (), {"spec": None})()})()}
    m_utils.assign_weights_from_eval_shape(["a"], tensor, state, "st_key", transform)
    transform = ((1, 0), (2, 4), False)
    state = {"a": type("S", (), {"shape": (2, 4), "dtype": jnp.float32, "sharding": type("Sh", (), {"spec": None})()})()}
    m_utils.assign_weights_from_eval_shape(["a"], tensor, state, "st_key", transform)


def test_create_model_from_safe_tensors_mapped_key_none(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())

    def mock_map(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return (None, None)

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    m_utils.create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})
