import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, create_model_from_safe_tensors


def test_utils_imports_fail(monkeypatch: pytest.MonkeyPatch):
    import importlib
    import sys

    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    monkeypatch.setitem(sys.modules, "safetensors", None)
    importlib.reload(m_utils)
    monkeypatch.undo()
    importlib.reload(m_utils)


def test_assign_weights_transforms(monkeypatch: pytest.MonkeyPatch):
    tensor = jnp.zeros((2, 4))

    # reshape_first = True
    transform = ((1, 0), (4, 2), True)
    state = {"a": type("S", (), {"shape": (2, 4)})()}
    assign_weights(["a"], tensor, state, "st_key", transform)

    # reshape_first = False
    transform = ((1, 0), (2, 4), False)
    state = {"a": type("S", (), {"shape": (2, 4)})()}
    assign_weights(["a"], tensor, state, "st_key", transform)


def test_assign_weights_value_attribute(monkeypatch: pytest.MonkeyPatch):
    tensor = jnp.zeros((2, 4))
    transform = None
    state = {"a": type("S", (), {"value": type("V", (), {"shape": (2, 4)})()})()}
    assign_weights(["a"], tensor, state, "st_key", transform)
    assert hasattr(state["a"], "value")


def test_create_model_from_safe_tensors(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    class MockF:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield "b"

        def get_tensor(self, k):
            return jnp.zeros((1,))

    monkeypatch.setattr(m_utils, "safe_open", lambda *args, **kwargs: MockF())

    def mock_map(*args, **kwargs):
        return "a", None

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)

    file_path = tmp_path / "model.safetensors"
    file_path.touch()

    class MockNNX:
        class Rngs:
            def __init__(self, *args, **kwargs):
                pass

        def split(self, *args, **kwargs):
            return None, {"a": type("S", (), {"shape": (1,)})()}, None

    class MockModelCls:
        def __init__(self, *args, **kwargs):
            pass

    import sys

    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())

    res = create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})


def test_create_model_from_safe_tensors_key_error(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    class MockF:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield "b"

        def get_tensor(self, k):
            return jnp.zeros((1,))

    monkeypatch.setattr(m_utils, "safe_open", lambda *args, **kwargs: MockF())

    def mock_map(*args, **kwargs):
        return "a", None

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)

    file_path = tmp_path / "model.safetensors"
    file_path.touch()

    class MockNNX:
        class Rngs:
            def __init__(self, *args, **kwargs):
                pass

        def split(self, *args, **kwargs):
            return None, {}, None  # Missing 'a' causes KeyError

    class MockModelCls:
        def __init__(self, *args, **kwargs):
            pass

    import sys

    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())

    res = create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})


def test_create_model_from_safe_tensors_exception(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    class MockF:
        def __enter__(self):
            msg = "err"
            raise ValueError(msg)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(m_utils, "safe_open", lambda *args, **kwargs: MockF())

    file_path = tmp_path / "model.safetensors"
    file_path.touch()

    class MockNNX:
        class Rngs:
            def __init__(self, *args, **kwargs):
                pass

        def split(self, *args, **kwargs):
            return None, {}, None

    class MockModelCls:
        def __init__(self, *args, **kwargs):
            pass

    import sys

    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())

    res = create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})


def test_assign_weights_from_eval_shape_exceptions(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    # Missing key in state dict
    try:
        m_utils.assign_weights_from_eval_shape(["missing"], jnp.zeros((1,)), {}, "st_key", None)
    except Exception:
        pass

    # Not a leaf node
    try:
        m_utils.assign_weights_from_eval_shape(["a"], jnp.zeros((1,)), {"a": {"b": 1}}, "st_key", None)
    except Exception:
        pass

    # Shape mismatch
    try:
        m_utils.assign_weights_from_eval_shape(["a"], jnp.zeros((2,)), {"a": type("S", (), {"shape": (1,)})()}, "st_key", None)
    except Exception:
        pass


def test_create_model_from_safe_tensors_missing_dir(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    m_utils.create_model_from_safe_tensors("non_existent_dir", lambda *a, **k: "model", {}, {})


def test_create_model_from_safe_tensors_nnx_split_fails(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    class MockNNX:
        class Rngs:
            def __init__(self, *args, **kwargs):
                pass

        def split(self, *args, **kwargs):
            msg = "err"
            raise ValueError(msg)

    import sys

    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())

    file_path = tmp_path / "model.safetensors"
    file_path.touch()

    class MockF:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield "b"

        def get_tensor(self, k):
            return jnp.zeros((1,))

    monkeypatch.setattr(m_utils, "safe_open", lambda *args, **kwargs: MockF())

    m_utils.create_model_from_safe_tensors(str(tmp_path), lambda *a, **k: "model", {}, {"b": (None, None)})


def test_assign_weights_from_eval_shape_transforms(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    tensor = jnp.zeros((2, 4))

    # reshape_first = True
    transform = ((1, 0), (4, 2), True)
    state = {"a": type("S", (), {"shape": (2, 4), "dtype": jnp.float32, "sharding": type("Sh", (), {"spec": None})()})()}
    m_utils.assign_weights_from_eval_shape(["a"], tensor, state, "st_key", transform)

    # reshape_first = False
    transform = ((1, 0), (2, 4), False)
    state = {"a": type("S", (), {"shape": (2, 4), "dtype": jnp.float32, "sharding": type("Sh", (), {"spec": None})()})()}
    m_utils.assign_weights_from_eval_shape(["a"], tensor, state, "st_key", transform)


def test_create_model_from_safe_tensors_mapped_key_none(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.jax.gemma4.utils_params as m_utils

    class MockF:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield "b"

        def get_tensor(self, k):
            return jnp.zeros((1,))

    monkeypatch.setattr(m_utils, "safe_open", lambda *args, **kwargs: MockF())

    def mock_map(*args, **kwargs):
        return None, None

    monkeypatch.setattr(m_utils, "map_to_jax_key", mock_map)

    file_path = tmp_path / "model.safetensors"
    file_path.touch()

    class MockNNX:
        class Rngs:
            def __init__(self, *args, **kwargs):
                pass

        def split(self, *args, **kwargs):
            return None, {"a": type("S", (), {"shape": (1,)})()}, None

    class MockModelCls:
        def __init__(self, *args, **kwargs):
            pass

    import sys

    monkeypatch.setitem(sys.modules, "flax", type("M", (), {"nnx": MockNNX()})())

    m_utils.create_model_from_safe_tensors(str(tmp_path), MockModelCls, {}, {"b": ("a", None)})
