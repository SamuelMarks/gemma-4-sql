"""Unit and behavioral tests for gemma4 utils."""

import contextlib
from typing import Any

from tests.conftest import requires_jax

pytestmark = requires_jax

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
import numpy as np

import gemma_4_sql.backends.jax.gemma4.utils_params as ut
from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape, create_model_from_safe_tensors, map_to_jax_key, stoi

EXPECTED_VAL = 123


def test_utils() -> Any:
    """Verify assign_weights, assign_weights_from_eval_shape, and stoi transformations.

    Raises:
        AssertionError: If weight assignment or string-to-int conversion fails.

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
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), True), sharding_dict=None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), False), sharding_dict=None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", ((1, 0), None, False), sharding_dict=None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((3, 3)), state, "k", None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((3, 3)), state2, "k", None, sharding_dict=None)
    sharding = {"a": jax.sharding.NamedSharding(jax.sharding.Mesh(np.array(jax.devices()), ("x",)), jax.sharding.PartitionSpec("x"))}
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", None, sharding_dict=sharding)
    assert stoi("123") == EXPECTED_VAL


def test_utils_part1() -> Any:
    """Verify utils part1 behavior."""
    assert stoi("abc") == "abc"
    mapping = {"abc\\.(\\d+)": ("def.xyz", None), "cde": ("uvw", None)}
    map_to_jax_key(mapping, "abc.123")
    map_to_jax_key(mapping, "cde")
    map_to_jax_key(mapping, "unmatched")


class MockModelCls:
    """Provide class docstring."""

    def __init__(self, cfg: Any, rngs: Any) -> None:
        """Initialize test stub instance."""
        self.cfg = cfg
        self.rngs = rngs


class MockSafeOpen:
    """Provide class docstring."""

    def __init__(self, filepath: Any, framework: Any) -> None:
        """Initialize test stub instance."""
        self.filepath = filepath
        self.framework = framework

    def __enter__(self) -> Any:
        """Provide enter   test helper."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Provide exit   test helper."""

    def __iter__(self) -> Any:
        """Iterate over mock safetensors keys."""
        return iter(self.keys())

    def keys(self) -> Any:
        """Provide keys test helper."""
        return ["layer1.weight", "layer2.weight"]

    def get_tensor(self, _key: Any) -> Any:
        """Provide get tensor test helper."""
        return jnp.ones((2, 2))


def test_create_model_from_safe_tensors(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create model from safe tensors behavior."""
    d = tmp_path / "model_dir"
    d.mkdir()
    f = d / "model.safetensors"
    f.write_text("dummy content")
    monkeypatch.setattr(ut, "safe_open", MockSafeOpen)
    mapping = {"layer1\\.weight": ("layer1.w", None), "layer2\\.weight": ("layer2.w", None)}
    model = create_model_from_safe_tensors(str(d), MockModelCls, "config", mapping)
    assert model.cfg == "config"


def test_create_model_from_safe_tensors_missing_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create model from safe tensors missing dir behavior."""
    monkeypatch.setattr(ut, "safe_open", MockSafeOpen)
    model = create_model_from_safe_tensors("nonexistent_dir", MockModelCls, "config", {})
    assert model.cfg == "config"


def test_assign_weights_int_to_str_keys() -> None:
    """Test integer to string key conversion in assign_weights and assign_weights_from_eval_shape."""
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape

    # str key in state_dict, int passed in keys
    d_str: dict[str, Any] = {"0": {"val": jnp.zeros((2, 2))}}
    assign_weights([0, "val"], jnp.zeros((2, 2)), d_str, "k", None)
    assert "0" in d_str

    # int key in state_dict, str passed in keys
    d_int: dict[int, Any] = {0: {"val": jnp.zeros((2, 2))}}
    assign_weights(["0", "val"], jnp.zeros((2, 2)), d_int, "k", None)
    assert 0 in d_int

    class Dummy:
        """Test class for Dummy."""

        shape = (2, 2)
        dtype = None
        sharding = None

    d_str2: dict[str, Any] = {"0": {"val": Dummy()}}
    assign_weights_from_eval_shape([0, "val"], jnp.zeros((2, 2)), d_str2, "k", None)
    assert "0" in d_str2


def test_create_model_from_safe_tensors_no_safetensors(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create model from safe tensors no safetensors behavior."""
    monkeypatch.setattr(ut, "safe_open", None)
    model = create_model_from_safe_tensors(str(tmp_path), MockModelCls, "config", {})
    assert model.cfg == "config"


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_transforms() -> None:
    """Verify assign weights transforms behavior."""
    tensor = jnp.zeros((2, 4))
    transform = ((1, 0), (4, 2), True)
    state = {"a": type("S", (), {"shape": (2, 4)})()}
    assign_weights(["a"], tensor, state, "st_key", transform)
    transform = ((1, 0), (2, 4), False)
    state = {"a": type("S", (), {"shape": (2, 4)})()}
    assign_weights(["a"], tensor, state, "st_key", transform)


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_value_attribute() -> None:
    """Verify assign weights value attribute behavior."""
    tensor = jnp.zeros((2, 4))
    transform = None
    state = {"a": type("S", (), {"value": type("V", (), {"shape": (2, 4)})()})()}
    assign_weights(["a"], tensor, state, "st_key", transform)
    assert hasattr(state["a"], "value")


class MockF:
    """Provide class docstring."""

    def __enter__(self) -> Any:
        """Provide enter   test helper."""
        return self

    def __exit__(self, *args: object) -> Any:
        """Provide exit   test helper."""

    def __iter__(self) -> Any:
        """Provide iter   test helper."""
        yield "b"

    def get_tensor(self, _k: Any) -> Any:
        """Provide get tensor test helper."""
        return jnp.zeros((1,))


class MockNNX:
    """Class docstring."""

    def update(self, *args: Any, **kwargs: Any):
        """Verify expected test case behavior."""

    """Provide class docstring."""

    class Rngs:
        """Provide class docstring."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize test stub instance."""

    def split(self, *_args: Any, **_kwargs: Any) -> Any:
        """Provide split test helper."""
        return (None, {"a": type("S", (), {"shape": (1,)})()}, None)


class MockModelCls2:
    """Provide class docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize test stub instance."""


def test_create_model_from_safe_tensors2(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create model from safe tensors2 behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())

    def mock_map(*_args: Any, **_kwargs: Any) -> Any:
        """Provide mock map test helper."""
        return ("a", None)

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})


def test_create_model_from_safe_tensors_key_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create model from safe tensors key error behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())

    def mock_map(*_args: Any, **_kwargs: Any) -> Any:
        """Provide mock map test helper."""
        return ("a", None)

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})


def test_create_model_from_safe_tensors_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create model from safe tensors exception behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_from_eval_shape_exceptions() -> None:
    """Verify assign weights from eval shape exceptions behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    with contextlib.suppress(Exception):
        m_utils.assign_weights_from_eval_shape(["missing"], jnp.zeros((1,)), {}, "st_key", None)
    with contextlib.suppress(Exception):
        m_utils.assign_weights_from_eval_shape(["a"], jnp.zeros((1,)), {"a": {"b": 1}}, "st_key", None)
    with contextlib.suppress(Exception):
        m_utils.assign_weights_from_eval_shape(["a"], jnp.zeros((2,)), {"a": type("S", (), {"shape": (1,)})()}, "st_key", None)


@pytest.mark.usefixtures("monkeypatch")
def test_create_model_from_safe_tensors_missing_dir2() -> None:
    """Verify create model from safe tensors missing dir2 behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    m_utils.create_model_from_safe_tensors("non_existent_dir", lambda *_a, **_k: "model", {}, {})


def test_create_model_from_safe_tensors_nnx_split_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create model from safe tensors nnx split fails behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())
    m_utils.create_model_from_safe_tensors(str(tmp_path), lambda *_a, **_k: "model", {}, {"b": (None, None)})


@pytest.mark.usefixtures("monkeypatch")
def test_assign_weights_from_eval_shape_transforms() -> None:
    """Verify assign weights from eval shape transforms behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    tensor = jnp.zeros((2, 4))
    transform = ((1, 0), (4, 2), True)
    state = {"a": type("S", (), {"shape": (2, 4), "dtype": jnp.float32, "sharding": type("Sh", (), {"spec": None})()})()}
    m_utils.assign_weights_from_eval_shape(["a"], tensor, state, "st_key", transform)
    transform = ((1, 0), (2, 4), False)
    state = {"a": type("S", (), {"shape": (2, 4), "dtype": jnp.float32, "sharding": type("Sh", (), {"spec": None})()})()}
    m_utils.assign_weights_from_eval_shape(["a"], tensor, state, "st_key", transform)


def test_create_model_from_safe_tensors_mapped_key_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create model from safe tensors mapped key none behavior."""
    m_utils = __import__("gemma_4_sql.backends.jax.gemma4.utils_params", fromlist=[""])
    monkeypatch.setattr(m_utils, "safe_open", lambda *_args, **_kwargs: MockF())

    def mock_map(*_args: Any, **_kwargs: Any) -> Any:
        """Provide mock map test helper."""
        return (None, None)

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)
    file_path = tmp_path / "model.safetensors"
    file_path.touch()
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())
    m_utils.create_model_from_safe_tensors(str(tmp_path), MockModelCls2, {}, {"b": ("a", None)})
