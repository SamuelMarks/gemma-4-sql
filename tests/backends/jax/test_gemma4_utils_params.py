"""Unit and behavioral tests for JAX Gemma 4 safetensors parameter loading and key mapping."""

from __future__ import annotations

import types
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import jax.numpy as jnp
import pytest

import gemma_4_sql.backends.jax.gemma4.utils_params as up
from gemma_4_sql.backends.jax.gemma4.utils_params import (
    KeyMapType,
    ParamValueHolderProtocol,
    SafeOpenFileProtocol,
    ShardingNode,
    StateNode,
    TransformKind,
    _apply_transform,
    _extract_leaf_shape_and_dtype,
    _get_model_and_state,
    _iter_safe_open_keys,
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

    mapping: dict[str, KeyMapType] = {
        r"abc\.(\d+)": (r"def.\1", None),
        "cde": ("uvw", ((1, 0), None, False)),
    }
    assert map_to_jax_key(mapping, "abc.123") == ("def.123", None)
    assert map_to_jax_key(mapping, "cde") == ("uvw", ((1, 0), None, False))
    assert map_to_jax_key(mapping, "unmatched") == (None, None)

    ambiguous: dict[str, KeyMapType] = {r"model.*": ("b", None), r"model\.layers.*": ("d", None)}
    with pytest.raises(ValueError, match="Multiple mappings found"):
        map_to_jax_key(ambiguous, "model.layers.0")


def test_apply_transform_and_leaf_extract() -> None:
    """Verify _apply_transform handles None, TransformKind enum, reshape_first, and permute-then-reshape."""
    t = jnp.zeros((2, 4), dtype=jnp.float32)
    assert _apply_transform(t, None) is t
    assert _apply_transform(t, TransformKind.DEFAULT) is t

    out_enum = _apply_transform(t, TransformKind.LINEAR)
    assert out_enum.shape == (4, 2)

    out_rf = _apply_transform(t, ((1, 0), (4, 2), True))
    assert out_rf.shape == (2, 4)

    out_pr = _apply_transform(t, ((1, 0), (8,), False))
    assert out_pr.shape == (8,)

    out_p = _apply_transform(t, ((1, 0), None, False))
    assert out_p.shape == (4, 2)

    out_r1 = _apply_transform(t, (None, (8,), False))
    assert out_r1.shape == (8,)
    out_r2 = _apply_transform(t, (None, (8,), True))
    assert out_r2.shape == (8,)

    assert _extract_leaf_shape_and_dtype(42) == ((), None)


def test_assign_weights_and_eval_shape() -> None:
    """Verify assign_weights and assign_weights_from_eval_shape assign values, sharding, and key types."""
    tensor = jnp.ones((2, 2), dtype=jnp.float32)

    class _ShardingSpecStub:
        """Sharding stub with .spec."""

        spec = None

    class VarHolder:
        """Parameter holder with .value attribute."""

        def __init__(self, val: jax.Array | jax.ShapeDtypeStruct, sharding: _ShardingSpecStub | None = None) -> None:
            """Initialize VarHolder."""
            self.value = val
            self.sharding = sharding
            self.shape = val.shape
            self.dtype = val.dtype

    holder = VarHolder(jnp.zeros((2, 2), dtype=jnp.float32))
    state: StateNode = {"a": {"b": holder}}
    sharding_dict: ShardingNode = {
        "a": {
            "b": jax.sharding.NamedSharding(
                jax.make_mesh((len(jax.devices()),), ("x",)),
                jax.sharding.PartitionSpec(),
            )
        }
    }
    assign_weights(["a", "b"], tensor, state, "st_k", None, sharding_dict=sharding_dict)
    assert isinstance(holder.value, jax.Array)
    assert jnp.allclose(holder.value, tensor)

    state_direct: StateNode = {"a": jnp.zeros((2, 2), dtype=jnp.float32)}
    assign_weights(["a"], tensor, state_direct, "st_k", None, sharding_dict=None)
    direct_val = state_direct["a"]
    assert isinstance(direct_val, jax.Array)
    assert jnp.allclose(direct_val, tensor)

    eval_var = VarHolder(
        jax.ShapeDtypeStruct((2, 2), jnp.float32),
        sharding=_ShardingSpecStub(),
    )
    eval_state: StateNode = {"a": {"b": eval_var}}
    assign_weights_from_eval_shape(["a", "b"], tensor, eval_state, "st_k", None)
    assert isinstance(eval_var.value, jax.Array)
    assert jnp.allclose(eval_var.value, tensor)

    eval_direct: StateNode = {"a": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    assign_weights_from_eval_shape(["a"], tensor, eval_direct, "st_k", None)
    eval_direct_val = eval_direct["a"]
    assert isinstance(eval_direct_val, jax.Array)
    assert jnp.allclose(eval_direct_val, tensor)

    with pytest.raises(ValueError, match="Shape mismatch"):
        assign_weights(["a"], jnp.ones((3, 3)), state_direct, "st_k", None)
    with pytest.raises(ValueError, match="Shape mismatch"):
        assign_weights_from_eval_shape(["a"], jnp.ones((3, 3)), eval_direct, "st_k", None)


def test_assign_weights_int_and_str_keys() -> None:
    """Verify int-to-str and str-to-int key resolution and KeyError on missing keys."""
    tensor = jnp.zeros((1,), dtype=jnp.float32)

    mock_val = MagicMock(spec=ParamValueHolderProtocol)
    mock_val.value = jnp.zeros((1,), dtype=jnp.float32)

    state_str: StateNode = {"1": mock_val}
    assign_weights([1], tensor, state_str, "st_key", None)
    assign_weights_from_eval_shape([1], tensor, state_str, "st_key", None)

    state_int: StateNode = {1: mock_val}
    assign_weights(["1"], tensor, state_int, "st_key", None)
    assign_weights_from_eval_shape(["1"], tensor, state_int, "st_key", None)

    state_missing: StateNode = {"2": mock_val}
    with pytest.raises(KeyError):
        assign_weights([1], tensor, state_missing, "st_key", None)
    with pytest.raises(KeyError):
        assign_weights_from_eval_shape([1], tensor, state_missing, "st_key", None)


def test_load_safetensors_and_populate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify _load_weights_from_safetensors_file and _populate_state_from_files handle files and errors."""

    class _KeysOnlyFile(SafeOpenFileProtocol):
        """Reader exposing .keys() without implementing Iterable directly."""

        def keys(self) -> Iterable[str]:
            """Return single key."""
            return ["k1"]

        def get_tensor(self, _key: str) -> jax.Array:
            """Return unit tensor."""
            return jnp.ones((1,))

    assert list(_iter_safe_open_keys(_KeysOnlyFile())) == ["k1"]

    class MockSafeOpenError:
        """Context manager raising TypeError on enter."""

        def __init__(self, *_a: str, **_k: str) -> None:
            """Initialize MockSafeOpenError."""

        def __enter__(self) -> SafeOpenFileProtocol:
            """Raise TypeError."""
            raise TypeError("Simulated TypeError")

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: types.TracebackType | None,
        ) -> None:
            """Exit context."""

    monkeypatch.setattr(up, "safe_open", MockSafeOpenError)
    up._load_weights_from_safetensors_file("path", {}, {})

    mock_f = MagicMock()
    mock_f.__iter__.return_value = ["model.layers.0.weight", "unmapped_key"]
    mock_f.get_tensor.return_value = jnp.ones((2, 2))

    class MockSafeOpenOk:
        """Context manager yielding mock_f."""

        def __init__(self, *_a: str, **_k: str) -> None:
            """Initialize MockSafeOpenOk."""

        def __enter__(self) -> SafeOpenFileProtocol:
            """Return mock_f."""
            return mock_f

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: types.TracebackType | None,
        ) -> None:
            """Exit context."""

    monkeypatch.setattr(up, "safe_open", MockSafeOpenOk)
    monkeypatch.setattr(up, "map_to_jax_key", lambda _m, k: ("transformer.layer.0" if "layers" in k else None, None))  # pyright: ignore[reportUnknownLambdaType]
    mock_assign = MagicMock()
    monkeypatch.setattr(up, "assign_weights", mock_assign)
    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})
    mock_assign.assert_called_once()

    monkeypatch.setattr(up, "assign_weights", MagicMock(side_effect=KeyError("Mocked key error")))
    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})

    (tmp_path / "model.safetensors").write_text("dummy", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    mock_load_file = MagicMock()
    monkeypatch.setattr(up, "_load_weights_from_safetensors_file", mock_load_file)
    _populate_state_from_files(str(tmp_path), {}, {})
    mock_load_file.assert_called_once()


def test_create_model_from_safe_tensors_flows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify create_model_from_safe_tensors handles missing dirs, missing safe_open, and strict update errors."""
    mock_model_cls = MagicMock()
    model, _ = _get_model_and_state(mock_model_cls, "cfg")
    assert model is not None

    monkeypatch.setattr(up.nnx, "split", lambda *_a: ("graph", {"k": "v"}))  # pyright: ignore[reportUnknownLambdaType]
    _, state2 = _get_model_and_state(mock_model_cls, "cfg")
    assert state2 == {"k": "v"}

    res_missing = create_model_from_safe_tensors(str(tmp_path / "nonexistent"), mock_model_cls, "cfg", {})
    assert res_missing is not None

    monkeypatch.setattr(up, "safe_open", None)
    res_no_st = create_model_from_safe_tensors(str(tmp_path), mock_model_cls, "cfg", {})
    assert res_no_st is not None

    monkeypatch.setattr(up, "safe_open", MagicMock())
    monkeypatch.setattr(up, "_populate_state_from_files", lambda *_a: None)  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(up.nnx, "update", MagicMock(side_effect=ValueError("simulated error updating state")))

    create_model_from_safe_tensors(str(tmp_path), mock_model_cls, "cfg", {}, strict=False)
    with pytest.raises(ExportError, match="simulated error updating state"):
        create_model_from_safe_tensors(str(tmp_path), mock_model_cls, "cfg", {}, strict=True)


def test_real_safetensors_roundtrip_into_jax_gemma4(tmp_path: Path) -> None:
    """Verify real safetensors file saved to disk loads into JaxGemma4ForCausalLM without mocks."""
    from safetensors.flax import save_file

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
    from gemma_4_sql.backends.jax.gemma4.params import _get_key_and_transform_mapping

    cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    st_file = tmp_path / "model.safetensors"
    emb_w = jnp.ones((32, 16), dtype=jnp.float32) * 0.25
    save_file({"model.embed_tokens.weight": emb_w}, str(st_file))

    loaded_model = create_model_from_safe_tensors(
        str(tmp_path),
        Gemma4ForCausalLM,
        cfg,
        _get_key_and_transform_mapping(),
        strict=False,
    )
    assert loaded_model is not None
    out = loaded_model(jnp.array([[1, 2]], dtype=jnp.int32), jnp.array([[0, 1]], dtype=jnp.int32))
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 2, 32)
    assert jnp.all(jnp.isfinite(logits))
