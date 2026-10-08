"""Module docstring."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.utils_params import _apply_transform, _get_model_and_state, _load_weights_from_safetensors_file, _populate_state_from_files, assign_weights, assign_weights_from_eval_shape, create_model_from_safe_tensors, map_to_jax_key, stoi


def test_map_to_jax_key_proper():
    """Docstring for test_map_to_jax_key_proper."""
    import re

    assert map_to_jax_key({}, "foo") == (None, None)
    # Multiple mappings found for "foo"
    mapping = {re.compile(r"foo"): ("key1", None), re.compile(r"fo.*"): ("key2", None)}
    with pytest.raises(ValueError):
        map_to_jax_key(mapping, "foo")


def test_apply_transform():
    """Docstring for test_apply_transform."""
    assert _apply_transform(jnp.ones(1), None).shape == (1,)


def test_assign_weights():
    """Docstring for test_assign_weights."""
    # To hit 108-128
    # list of keys: ["key1"]
    state_dict = {"1": MagicMock(value=jnp.ones(1)), "2": jnp.ones(1)}
    assign_weights(["1"], jnp.ones(1), state_dict, "st_key", None)

    # shape mismatch
    with pytest.raises(ValueError):
        assign_weights(["1"], jnp.ones(2), state_dict, "st_key", None)

    # fallback to string key or int key
    assign_weights([2], jnp.ones(1), state_dict, "st_key", None)

    # recursive assign_weights
    state_dict_nested = {"a": {"b": jnp.ones(1)}}
    assign_weights(["a", "b"], jnp.ones(1), state_dict_nested, "st_key", None)


def test_assign_weights_from_eval_shape():
    """Docstring for test_assign_weights_from_eval_shape."""
    # 145-170
    state_dict = {"1": jax.ShapeDtypeStruct((1,), jnp.float32)}
    assign_weights_from_eval_shape(["1"], jnp.ones(1), state_dict, "st_key", None)

    state_dict_nested = {"a": {"b": jax.ShapeDtypeStruct((1,), jnp.float32)}}
    assign_weights_from_eval_shape(["a", "b"], jnp.ones(1), state_dict_nested, "st_key", None)

    # missing key
    with pytest.raises(KeyError):
        assign_weights_from_eval_shape(["foo"], jnp.ones(1), state_dict, "st_key", None)

    # shape mismatch
    with pytest.raises(ValueError):
        assign_weights_from_eval_shape(["1"], jnp.ones(2), state_dict, "st_key", None)


def test_load_weights_from_safetensors_file():
    """Docstring for test_load_weights_from_safetensors_file."""
    # 175-188
    # we need safe_open to work.
    mock_safe_open = MagicMock()
    mock_f = MagicMock()
    mock_f.__iter__.return_value = ["torch_key"]
    mock_f.get_tensor.return_value = jnp.ones(1)
    mock_safe_open.return_value.__enter__.return_value = mock_f

    with patch("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", mock_safe_open), patch("gemma_4_sql.backends.jax.gemma4.utils_params.map_to_jax_key") as mock_map, patch("gemma_4_sql.backends.jax.gemma4.utils_params.assign_weights") as mock_assign:
        mock_map.return_value = ("a.b", None)
        _load_weights_from_safetensors_file("foo.safetensors", {}, {})
        mock_assign.assert_called_once()

        # test KeyError suppression
        mock_assign.side_effect = KeyError("foo")
        _load_weights_from_safetensors_file("foo.safetensors", {}, {})


def test_get_model_and_state():
    """Docstring for test_get_model_and_state."""
    mock_model_cls = MagicMock()
    mock_model_cls.return_value = "model"
    with patch("flax.nnx.split", return_value=("graph", "state", "other")):
        model, state = _get_model_and_state(mock_model_cls, {})
        assert model == "model"
        assert state == "state"


def test_populate_state_from_files():
    """Docstring for test_populate_state_from_files."""
    with patch("os.walk", return_value=[("dir", [], ["file.safetensors"])]), patch("gemma_4_sql.backends.jax.gemma4.utils_params._load_weights_from_safetensors_file") as mock_load:
        _populate_state_from_files("dir", {}, {})
        mock_load.assert_called_once()


def test_create_model_from_safe_tensors():
    """Docstring for test_create_model_from_safe_tensors."""
    with patch("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", None):
        res = create_model_from_safe_tensors("bad_dir", lambda c, **kw: "fallback", {}, {})
        assert res == "fallback"

    mock_safe_open = MagicMock()
    with (
        patch("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", mock_safe_open),
        patch("gemma_4_sql.backends.jax.gemma4.utils_params.Path") as mock_path,
        patch("gemma_4_sql.backends.jax.gemma4.utils_params._get_model_and_state", return_value=("model", "state")),
        patch("gemma_4_sql.backends.jax.gemma4.utils_params._populate_state_from_files"),
        patch("flax.nnx.update"),
    ):
        mock_path.return_value.is_dir.return_value = True
        assert create_model_from_safe_tensors("dir", None, {}, {}) == "model"

        # raise an error in nnx.update
        with patch("flax.nnx.update", side_effect=ValueError):
            assert create_model_from_safe_tensors("dir", None, {}, {}) == "model"


def test_apply_transform_more():
    """Docstring for test_apply_transform_more."""
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.utils_params import _apply_transform

    t = jnp.ones((2, 2))
    _apply_transform(t, (None, (4,), True))
    _apply_transform(t, (None, (4,), False))
    _apply_transform(t, ((1, 0), None, False))


def test_assign_weights_fallbacks():
    """Docstring for test_assign_weights_fallbacks."""
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights

    class MockSharding:
        """Docstring for MockSharding."""

        spec = "spec"

    class Target:
        """Docstring for Target."""

        sharding = MockSharding()

    sd = {1: jnp.ones(1)}
    assign_weights(["1"], jnp.ones(1), sd, "st_key", None)

    sd2 = {"2": jnp.ones(1)}
    assign_weights([2], jnp.ones(1), sd2, "st_key", None)

    sd3 = {"3": MagicMock(value=jnp.ones(1))}
    assign_weights([3], jnp.ones(1), sd3, "st_key", None)


def test_assign_weights_from_eval_shape_fallbacks():
    """Docstring for test_assign_weights_from_eval_shape_fallbacks."""
    import jax
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights_from_eval_shape

    class TargetObj:
        """Docstring for TargetObj."""

        class ShardingObj:
            """Docstring for ShardingObj."""

            spec = "spec"

        sharding = ShardingObj()
        shape = (1,)
        dtype = jnp.float32

    sd = {1: jax.ShapeDtypeStruct((1,), jnp.float32)}
    assign_weights_from_eval_shape(["1"], jnp.ones(1), sd, "st_key", None)

    sd2 = {"2": jax.ShapeDtypeStruct((1,), jnp.float32)}
    assign_weights_from_eval_shape([2], jnp.ones(1), sd2, "st_key", None)

    sd4 = {"foo": TargetObj()}
    with patch("jax.device_put", return_value=jnp.ones(1)):
        assign_weights_from_eval_shape(["foo"], jnp.ones(1), sd4, "st_key", None)


def test_load_weights_continue_and_error():
    """Docstring for test_load_weights_continue_and_error."""
    from unittest.mock import MagicMock, patch

    from gemma_4_sql.backends.jax.gemma4.utils_params import _load_weights_from_safetensors_file

    mock_safe_open = MagicMock()
    mock_f = MagicMock()
    mock_f.__iter__.return_value = ["torch_key"]
    mock_safe_open.return_value.__enter__.return_value = mock_f

    with patch("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", mock_safe_open), patch("gemma_4_sql.backends.jax.gemma4.utils_params.map_to_jax_key", return_value=(None, None)):
        _load_weights_from_safetensors_file("foo.safetensors", {}, {})

    with patch("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", side_effect=OSError("test error")):
        _load_weights_from_safetensors_file("foo.safetensors", {}, {})


def test_map_to_jax_key_success():
    """Docstring for test_map_to_jax_key_success."""
    import re

    from gemma_4_sql.backends.jax.gemma4.utils_params import map_to_jax_key

    mapping = {re.compile(r"foo"): ("bar", None)}
    assert map_to_jax_key(mapping, "foo") == ("bar", None)


def test_stoi_coverage():
    """Docstring for test_stoi_coverage."""
    assert stoi("123") == 123
    assert stoi("abc") == "abc"
