"""Unit and behavioral tests for JAX Gemma 4 safetensors parameter loading and key mapping."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gemma_4_sql.backends.jax.gemma4.utils_params as up
from gemma_4_sql.backends.jax.gemma4.utils_params import (
    _apply_transform,
    _get_model_and_state,
    _populate_state_from_files,
    assign_weights,
    assign_weights_from_eval_shape,
    create_model_from_safe_tensors,
    map_to_jax_key,
    stoi,
)
from gemma_4_sql.exceptions import ExportError


def test_map_to_jax_key_and_stoi() -> None:
    """Verify map_to_jax_key matches regex patterns, returns None on unmatched, and raises on collisions."""
    assert stoi("123") == 123
    assert stoi("abc") == "abc"

    mapping: dict[str, tuple[str, Any]] = {
        r"abc\.(\d+)": (r"def.\1", None),
        "cde": ("uvw", ((1, 0), None, False)),
    }
    assert map_to_jax_key(mapping, "abc.123") == ("def.123", None)
    assert map_to_jax_key(mapping, "cde") == ("uvw", ((1, 0), None, False))
    assert map_to_jax_key(mapping, "unmatched") == (None, None)

    ambiguous = {r"model.*": ("b", None), r"model\.layers.*": ("d", None)}
    with pytest.raises(ValueError, match="Multiple mappings found"):
        map_to_jax_key(ambiguous, "model.layers.0")


def test_apply_transform() -> None:
    """Verify _apply_transform handles None, invalid tuples, reshape_first, and permute-then-reshape."""
    t = np.zeros((2, 4), dtype=np.float32)
    assert _apply_transform(t, None) is t

    bad_transform: Any = "invalid"
    with pytest.raises(ValueError, match="too many values to unpack"):
        _apply_transform(t, bad_transform)

    # reshape_first=True: reshape (2, 4) -> (4, 2), then permute (1, 0) -> (2, 4)
    out_rf = _apply_transform(t, ((1, 0), (4, 2), True))
    assert out_rf.shape == (2, 4)

    # reshape_first=False: permute (1, 0) -> (4, 2), then reshape -> (8,)
    out_pr = _apply_transform(t, ((1, 0), (8,), False))
    assert out_pr.shape == (8,)

    # permute only
    out_p = _apply_transform(t, ((1, 0), None, False))
    assert out_p.shape == (4, 2)

    # reshape only
    out_r1 = _apply_transform(t, (None, (8,), False))
    assert out_r1.shape == (8,)
    out_r2 = _apply_transform(t, (None, (8,), True))
    assert out_r2.shape == (8,)


def test_assign_weights_and_eval_shape() -> None:
    """Verify assign_weights and assign_weights_from_eval_shape assign values, sharding, and key types."""
    tensor = jnp.ones((2, 2), dtype=jnp.float32)

    class VarHolder:
        """Parameter holder with .value attribute."""

        def __init__(self, val: Any, sharding: Any = None) -> None:
            """Initialize VarHolder."""
            self.value = val
            self.sharding = sharding
            self.shape = val.shape
            self.dtype = val.dtype

    state: dict[str, Any] = {"a": {"b": VarHolder(jnp.zeros((2, 2), dtype=jnp.float32))}}
    sharding_dict = {
        "a": {
            "b": jax.sharding.NamedSharding(
                jax.sharding.Mesh(np.array(jax.devices()), ("x",)),
                jax.sharding.PartitionSpec(),
            )
        }
    }
    assign_weights(["a", "b"], tensor, state, "st_k", None, sharding_dict=sharding_dict)
    assert jnp.allclose(state["a"]["b"].value, tensor)

    # Direct leaf without .value attribute
    state_direct: dict[str, Any] = {"a": jnp.zeros((2, 2), dtype=jnp.float32)}
    assign_weights(["a"], tensor, state_direct, "st_k", None, sharding_dict=None)
    assert jnp.allclose(state_direct["a"], tensor)

    # assign_weights_from_eval_shape with and without .value
    eval_var = VarHolder(
        jax.ShapeDtypeStruct((2, 2), jnp.float32),
        sharding=type("Sh", (), {"spec": None})(),
    )
    eval_state: dict[str, Any] = {"a": {"b": eval_var}}
    assign_weights_from_eval_shape(["a", "b"], tensor, eval_state, "st_k", None)
    assert jnp.allclose(eval_state["a"]["b"].value, tensor)

    eval_direct: dict[str, Any] = {"a": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    assign_weights_from_eval_shape(["a"], tensor, eval_direct, "st_k", None)
    assert jnp.allclose(eval_direct["a"], tensor)

    # Shape mismatch raises ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        assign_weights(["a"], jnp.ones((3, 3)), state_direct, "st_k", None)
    with pytest.raises(ValueError, match="Shape mismatch"):
        assign_weights_from_eval_shape(["a"], jnp.ones((3, 3)), eval_direct, "st_k", None)


def test_assign_weights_int_and_str_keys() -> None:
    """Verify int-to-str and str-to-int key resolution and KeyError on missing keys."""
    tensor = np.zeros((1,), dtype=np.float32)

    mock_val = MagicMock()
    mock_val.value.shape = (1,)
    mock_val.value.dtype = np.float32
    mock_val.shape = (1,)
    mock_val.dtype = np.float32

    state_str: dict[Any, Any] = {"1": mock_val}
    assign_weights([1], tensor, state_str, "st_key", None)
    assign_weights_from_eval_shape([1], tensor, state_str, "st_key", None)

    state_int: dict[Any, Any] = {1: mock_val}
    assign_weights(["1"], tensor, state_int, "st_key", None)
    assign_weights_from_eval_shape(["1"], tensor, state_int, "st_key", None)

    state_missing: dict[Any, Any] = {"2": mock_val}
    with pytest.raises(KeyError):
        assign_weights([1], tensor, state_missing, "st_key", None)
    with pytest.raises(KeyError):
        assign_weights_from_eval_shape([1], tensor, state_missing, "st_key", None)


def test_load_safetensors_and_populate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify _load_weights_from_safetensors_file and _populate_state_from_files handle files and errors."""

    class MockSafeOpenError:
        """Context manager raising TypeError on enter."""

        def __init__(self, *_a: Any, **_k: Any) -> None:
            """Initialize MockSafeOpenError."""

        def __enter__(self) -> Any:
            """Raise TypeError."""
            raise TypeError("Simulated TypeError")

        def __exit__(self, *_a: object) -> None:
            """Exit context."""

    monkeypatch.setattr(up, "safe_open", MockSafeOpenError)
    up._load_weights_from_safetensors_file("path", {}, {})

    mock_f = MagicMock()
    mock_f.__iter__.return_value = ["model.layers.0.weight", "unmapped_key"]
    mock_f.get_tensor.return_value = jnp.ones((2, 2))

    class MockSafeOpenOk:
        """Context manager yielding mock_f."""

        def __init__(self, *_a: Any, **_k: Any) -> None:
            """Initialize MockSafeOpenOk."""

        def __enter__(self) -> Any:
            """Return mock_f."""
            return mock_f

        def __exit__(self, *_a: object) -> None:
            """Exit context."""

    monkeypatch.setattr(up, "safe_open", MockSafeOpenOk)
    monkeypatch.setattr(up, "map_to_jax_key", lambda _m, k: ("transformer.layer.0" if "layers" in k else None, None))
    mock_assign = MagicMock()
    monkeypatch.setattr(up, "assign_weights", mock_assign)
    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})
    mock_assign.assert_called_once()

    # KeyError inside assign_weights is logged and caught gracefully
    monkeypatch.setattr(up, "assign_weights", MagicMock(side_effect=KeyError("Mocked key error")))
    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})

    # _populate_state_from_files directory traversal
    (tmp_path / "model.safetensors").write_text("dummy", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    mock_load_file = MagicMock()
    monkeypatch.setattr(up, "_load_weights_from_safetensors_file", mock_load_file)
    _populate_state_from_files(str(tmp_path), {}, {})
    mock_load_file.assert_called_once()


def test_create_model_from_safe_tensors_flows(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Verify create_model_from_safe_tensors handles missing dirs, missing safe_open, and strict update errors."""
    mock_model_cls = MagicMock()
    model, _ = _get_model_and_state(mock_model_cls, MagicMock())
    assert model is not None

    monkeypatch.setattr(up.nnx, "split", lambda *_a: ("graph", {"k": "v"}))
    _, state2 = _get_model_and_state(mock_model_cls, MagicMock())
    assert state2 == {"k": "v"}

    # Missing directory returns initialized model
    res_missing = create_model_from_safe_tensors(str(tmp_path / "nonexistent"), mock_model_cls, "cfg", {})
    assert res_missing is not None

    # safe_open is None returns initialized model
    monkeypatch.setattr(up, "safe_open", None)
    res_no_st = create_model_from_safe_tensors(str(tmp_path), mock_model_cls, "cfg", {})
    assert res_no_st is not None

    # Valid directory with nnx.update error in non-strict and strict modes
    monkeypatch.setattr(up, "safe_open", MagicMock())
    monkeypatch.setattr(up, "_populate_state_from_files", lambda *_a: None)
    monkeypatch.setattr(up.nnx, "update", MagicMock(side_effect=ValueError("simulated error updating state")))

    create_model_from_safe_tensors(str(tmp_path), mock_model_cls, "cfg", {}, strict=False)
    with pytest.raises(ExportError, match="simulated error updating state"):
        create_model_from_safe_tensors(str(tmp_path), mock_model_cls, "cfg", {}, strict=True)
