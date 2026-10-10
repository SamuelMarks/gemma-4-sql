"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest


def test_map_to_jax_key_exceptions():
    """Docstring for test_map_to_jax_key_exceptions."""
    from gemma_4_sql.backends.jax.gemma4.utils_params import map_to_jax_key

    mapping = {r"model.*": ("b", None), r"model\.layers.*": ("d", None)}
    with pytest.raises(ValueError):
        map_to_jax_key(mapping, "model.layers.0")


def test_stoi():
    """Docstring for test_stoi."""
    from gemma_4_sql.backends.jax.gemma4.utils_params import stoi

    assert stoi("123") == 123
    assert stoi("abc") == "abc"


def test_apply_transform():
    """Docstring for test_apply_transform."""
    import numpy as np

    from gemma_4_sql.backends.jax.gemma4.utils_params import _apply_transform

    t = np.array([1, 2])
    res = _apply_transform(t, None)
    assert res is t
    with pytest.raises(ValueError):
        _apply_transform(t, "invalid")

    mock_t = MagicMock()
    mock_t.transpose.return_value = "transposed"
    res = _apply_transform(mock_t, ((1, 0), None, False))
    assert res == "transposed"

    mock_t2 = MagicMock()
    mock_t2.transpose.return_value = mock_t2
    mock_t2.reshape.return_value = "reshaped"
    res = _apply_transform(mock_t2, ((1, 0), (2, 1), False))
    assert res == "reshaped"


def test_assign_weights():
    """Docstring for test_assign_weights."""
    from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights

    mock_val = MagicMock()
    mock_val.shape = (1,)

    class MockVar:
        """Docstring for MockVar."""

    var_obj = MockVar()
    var_obj.value = mock_val
    state = {"a": {"b": var_obj}}

    import numpy as np

    t = np.array([1])

    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    # We patch device_put
    up.jax = MagicMock()
    up.jax.device_put = lambda x, shd=None: x

    assign_weights(["a", "b"], t, state, "st_k", None)


def test_assign_weights_from_eval_shape():
    """Docstring for test_assign_weights_from_eval_shape."""
    from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights_from_eval_shape

    mock_val = MagicMock()
    mock_val.shape = (2,)
    mock_val.dtype = float

    class MockVar:
        """Docstring for MockVar."""

    var_obj = MockVar()
    var_obj.value = mock_val
    var_obj.sharding = MagicMock()
    state = {"a": {"b": var_obj}}

    import numpy as np

    mock_t = np.zeros((2,))

    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    up.jax = MagicMock()
    up.jax.device_put = lambda x, s=None: x

    assign_weights_from_eval_shape(["a", "b"], mock_t, state, "st_k", None)

    mock_t2 = np.zeros((3,))
    with pytest.raises(ValueError):
        assign_weights_from_eval_shape(["a", "b"], mock_t2, state, "st_k", None)


def test_load_safetensors_weights_import(monkeypatch):
    """Docstring for test_load_safetensors_weights_import."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    class MockSafeOpen:
        """Docstring for MockSafeOpen."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def __enter__(self):
            """Docstring for __enter__."""
            raise TypeError("Simulated TypeError")

        def __exit__(self, *args):
            """Docstring for __exit__."""

    monkeypatch.setattr(up, "safe_open", MockSafeOpen)
    up._load_weights_from_safetensors_file("path", {}, {})


def test_load_safetensors_weights(monkeypatch):
    """Docstring for test_load_safetensors_weights."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    mock_f = MagicMock()
    # Mocking __iter__ to yield st_key
    mock_f.__iter__.return_value = ["model.layers.0.weight", "unmapped_key"]
    mock_f.get_tensor.return_value = "tensor"

    class MockSafeOpen:
        """Docstring for MockSafeOpen."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def __enter__(self):
            """Docstring for __enter__."""
            return mock_f

        def __exit__(self, *args):
            """Docstring for __exit__."""

    monkeypatch.setattr(up, "safe_open", MockSafeOpen)
    monkeypatch.setattr(up, "map_to_jax_key", lambda m, k: ("transformer.layer.0" if "layers" in k else None, None))
    monkeypatch.setattr(up, "assign_weights", MagicMock())

    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})
    up.assign_weights.assert_called_once()


def test_populate_state_from_files(monkeypatch):
    """Docstring for test_populate_state_from_files."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up
    from gemma_4_sql.backends.jax.gemma4.utils_params import _populate_state_from_files

    mock_path = MagicMock()
    mock_path.is_dir.return_value = True
    f1 = MagicMock()
    f1.suffix = ".safetensors"
    f1.as_posix.return_value = "f1.safetensors"

    f2 = MagicMock()
    f2.suffix = ".bin"
    f2.as_posix.return_value = "f2.bin"

    mock_path.glob.return_value = [f1, f2]
    monkeypatch.setattr(up, "Path", lambda x: mock_path)
    monkeypatch.setattr(up, "_load_weights_from_safetensors_file", MagicMock())

    _populate_state_from_files("dir", {}, {})


def test_create_model():
    """Docstring for test_create_model."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up
    from gemma_4_sql.backends.jax.gemma4.utils_params import create_model_from_safe_tensors

    with patch.object(up, "_get_model_and_state", return_value=("model", {})):
        with patch.object(up, "_populate_state_from_files"):
            with patch("gemma_4_sql.backends.jax.gemma4.utils_params.Path.is_dir", return_value=True):
                with patch("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", MagicMock()):
                    res = create_model_from_safe_tensors("dir", MagicMock(), MagicMock(), {})
                    assert res == "model"


def test_get_model_and_state():
    """Docstring for test_get_model_and_state."""
    from gemma_4_sql.backends.jax.gemma4.utils_params import _get_model_and_state

    mock_model_cls = MagicMock()
    model, state = _get_model_and_state(mock_model_cls, MagicMock())
    assert model is not None


def test_update_state_exceptions(monkeypatch):
    """Docstring for test_update_state_exceptions."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    def mock_populate(*args):
        """Docstring for mock_populate."""

    monkeypatch.setattr(up, "_populate_state_from_files", mock_populate)
    monkeypatch.setattr(up, "_get_model_and_state", lambda c, cfg: (MagicMock(), {}))
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.utils_params.Path.is_dir", lambda self: True)
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def update(self, *args):
            """Docstring for update."""
            raise ValueError("simulated error updating state")

    mock_nnx = MockNNX()

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "flax":
            mock_flax = MagicMock()
            mock_flax.nnx = mock_nnx
            return mock_flax
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    up.create_model_from_safe_tensors("dir", MagicMock(), MagicMock(), {})


def test_apply_transform_no_permute():
    """Docstring for test_apply_transform_no_permute."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    mock_tensor = MagicMock()
    # transform = (permute, reshape, reshape_first)
    # test permute = () or None, reshape = (1,), reshape_first = False
    up._apply_transform(mock_tensor, (None, (1,), False))
    up._apply_transform(mock_tensor, (None, (1,), True))


def test_assign_weights_shape_mismatch():
    """Docstring for test_assign_weights_shape_mismatch."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    tensor = MagicMock()
    tensor.shape = (1, 2)
    state = {"a": MagicMock(shape=(2, 1))}
    try:
        up.assign_weights(["a"], tensor, state, "a", None)
        assert False
    except ValueError as e:
        assert "Shape mismatch for a" in str(e)

    try:
        up.assign_weights_from_eval_shape(["a"], tensor, state, "a", None)
        assert False
    except ValueError as e:
        assert "Shape mismatch for a" in str(e)


def test_assign_weights_int_key():
    """Docstring for test_assign_weights_int_key."""
    import numpy as np

    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    tensor = np.zeros((1,))

    # Test assign_weights
    mock_val = MagicMock()
    mock_val.value.shape = (1,)
    mock_val.value.dtype = np.float32
    state = {"1": mock_val}
    # resolved_key is int 1, but state has str "1"
    up.assign_weights([1], tensor, state, "st_key", None)

    # Test assign_weights_from_eval_shape
    mock_val2 = MagicMock()
    mock_val2.shape = (1,)
    mock_val2.value.shape = (1,)
    mock_val2.value.dtype = np.float32
    mock_val2.dtype = np.float32
    state = {"1": mock_val2}
    up.assign_weights_from_eval_shape([1], tensor, state, "st_key", None)

    # resolved_key is str "1", but state has int 1
    mock_val_new = MagicMock()
    mock_val_new.value.shape = (1,)
    mock_val_new.value.dtype = np.float32
    state_int_key = {1: mock_val_new}
    up.assign_weights(["1"], tensor, state_int_key, "st_key", None)

    mock_val2_new = MagicMock()
    mock_val2_new.shape = (1,)
    mock_val2_new.value.shape = (1,)
    mock_val2_new.value.dtype = np.float32
    mock_val2_new.dtype = np.float32
    state_int_key2 = {1: mock_val2_new}
    up.assign_weights_from_eval_shape(["1"], tensor, state_int_key2, "st_key", None)

    # Test when key is int but str(key) is NOT in state_dict (to hit 114->116 branch)
    state = {"2": mock_val}
    import pytest

    with pytest.raises(KeyError):
        up.assign_weights([1], tensor, state, "st_key", None)
    with pytest.raises(KeyError):
        up.assign_weights_from_eval_shape([1], tensor, state, "st_key", None)


def test_populate_state_from_files_non_safetensors(monkeypatch):
    """Docstring for test_populate_state_from_files_non_safetensors."""
    import os

    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    def mock_walk(*args, **kwargs):
        """Docstring for mock_walk."""
        return [("root", [], ["model.safetensors", "README.md"])]

    monkeypatch.setattr(os, "walk", mock_walk)
    monkeypatch.setattr(up, "_load_weights_from_safetensors_file", MagicMock())

    up._populate_state_from_files("dir", {}, {})


def test_assign_weights_keyerror(monkeypatch):
    """Docstring for test_assign_weights_keyerror."""
    import gemma_4_sql.backends.jax.gemma4.utils_params as up

    up.jax = MagicMock()
    up.jax.device_put = lambda x, s=None: x

    # test catching key error inside _load_weights_from_safetensors_file
    mock_f = MagicMock()
    mock_f.__iter__.return_value = ["model.layers.0.weight"]
    mock_f.get_tensor.return_value = "tensor"

    class MockSafeOpen:
        """Docstring for MockSafeOpen."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def __enter__(self):
            """Docstring for __enter__."""
            return mock_f

        def __exit__(self, *args):
            """Docstring for __exit__."""

    monkeypatch.setattr(up, "safe_open", MockSafeOpen)
    monkeypatch.setattr(up, "map_to_jax_key", lambda m, k: ("mapped", None))

    def mock_assign(*args, **kwargs):
        """Docstring for mock_assign."""
        raise KeyError("Mocked key error")

    monkeypatch.setattr(up, "assign_weights", mock_assign)
    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})
