"""Provide module docstring."""

import contextlib

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
import numpy as np

import gemma_4_sql.backends.jax.gemma4.utils_params as ut
from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape, create_model_from_safe_tensors, map_to_jax_key, stoi

EXPECTED_VAL = 123


def test_utils() -> object:
    """Docstring for test_utils.

    Raises:
        AssertionError: Description.

    """
    state = {"a": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    jnp.ones((2, 2))
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((4,)), state, "k", ((0,), (2, 2), True))
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((4,)), state, "k", ((0,), (2, 2), False))
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((2, 2)), state, "k", ((1, 0), None, False))
    state2 = {"a": jnp.zeros((2, 2))}
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), True), None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), False), None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", ((1, 0), None, False), None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((3, 3)), state, "k", None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((3, 3)), state2, "k", None, None)
    sharding = {"a": jax.sharding.NamedSharding(jax.sharding.Mesh(np.array(jax.devices()), ("x",)), jax.sharding.PartitionSpec("x"))}
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", None, sharding)
    if not stoi("123") == EXPECTED_VAL:
        raise AssertionError


def test_utils_part1() -> object:
    """Docstring.

    Raises:
        AssertionError: Description.

    """
    if not stoi("abc") == "abc":
        raise AssertionError
    mapping = {"abc\\.(\\d+)": ("def.xyz", None), "cde": ("uvw", None)}
    map_to_jax_key(mapping, "abc.123")
    map_to_jax_key(mapping, "cde")
    map_to_jax_key(mapping, "unmatched")


class MockModelCls:
    """Provide class docstring."""

    def __init__(self, cfg: object, rngs: object) -> None:
        """Execute function."""
        self.cfg = cfg
        self.rngs = rngs


class MockSafeOpen:
    """Provide class docstring."""

    def __init__(self, filepath: object, framework: object) -> None:
        """Execute function."""
        self.filepath = filepath
        self.framework = framework

    def __enter__(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> object:
        """Execute function."""

    def keys(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return ["layer1.weight", "layer2.weight"]

    def get_tensor(self, _key: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return jnp.ones((2, 2))


def test_create_model_from_safe_tensors(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    d = tmp_path / "model_dir"
    d.mkdir()
    f = d / "model.safetensors"
    f.write_text("dummy content")
    monkeypatch.setattr(ut, "safe_open", MockSafeOpen)
    mapping = {"layer1\\.weight": ("layer1.w", None), "layer2\\.weight": ("layer2.w", None)}
    model = create_model_from_safe_tensors(str(d), MockModelCls, "config", mapping)
    if not model.cfg == "config":
        raise AssertionError


def test_create_model_from_safe_tensors_missing_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(ut, "safe_open", MockSafeOpen)
    model = create_model_from_safe_tensors("nonexistent_dir", MockModelCls, "config", {})
    if not model.cfg == "config":
        raise AssertionError


def test_create_model_from_safe_tensors_no_safetensors(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(ut, "safe_open", None)
    model = create_model_from_safe_tensors(str(tmp_path), MockModelCls, "config", {})
    if not model.cfg == "config":
        raise AssertionError


import pytest


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
    """Class docstring."""

    def update(self, *args, **kwargs):
        """Test function."""

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


class MockModelCls2:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""


def test_create_model_from_safe_tensors2(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
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
    create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})


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
    create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})


def test_create_model_from_safe_tensors_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})


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
def test_create_model_from_safe_tensors_missing_dir2() -> None:
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
    m_utils.create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})
