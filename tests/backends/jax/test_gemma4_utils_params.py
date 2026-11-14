"""Unit and behavioral tests for JAX Gemma 4 safetensors parameter loading and key mapping."""

from __future__ import annotations

import types
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import pytest

import gemma_4_sql.backends.jax.gemma4.utils_params as up
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.utils_params import (
    KeyMapType,
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


class _ParamHolder:
    """Typed parameter holder implementing `ParamValueHolderProtocol`."""

    def __init__(self, value: jax.Array | jax.ShapeDtypeStruct) -> None:
        """Initialize parameter holder with `value`."""
        self.value: up.ShapeDtypeLeafProtocol | up.ArrayLikeTensor = value


class _IterSafeOpenFile(SafeOpenFileProtocol):
    """Iterable safetensors reader double returning 2x2 unit arrays."""

    def keys(self) -> list[str]:
        """Return safetensors tensor key names."""
        return ["model.layers.0.weight", "unmapped_key"]

    def __iter__(self) -> Iterator[str]:
        """Iterate over safetensors tensor key names.

        Returns:
            An iterator over the safetensors key names.

        """
        return iter(self.keys())

    def get_tensor(self, _key: str) -> jax.Array:
        """Return a 2x2 float32 ones array."""
        return jnp.ones((2, 2), dtype=jnp.float32)


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

    from typing import Protocol, runtime_checkable

    from gemma_4_sql.backends.jax._protocols import TypedJaxDeviceProtocol

    @runtime_checkable
    class _JaxDevicesProtocol(Protocol):
        """Protocol for `jax` exposing `.devices()`."""

        def devices(self) -> Sequence[TypedJaxDeviceProtocol]:
            """Return available JAX devices."""
            ...

    def _jax_devices() -> Sequence[TypedJaxDeviceProtocol]:
        """Return JAX devices without narrowing outer `jax` import.

        Returns:
            Sequence of available JAX devices.

        """
        assert isinstance(jax, _JaxDevicesProtocol)
        return jax.devices()

    mesh_obj = jax.sharding.Mesh(_jax_devices(), ("x",))
    holder = VarHolder(jnp.zeros((2, 2), dtype=jnp.float32))
    state: StateNode = {"a": {"b": holder}}
    sharding_dict: ShardingNode = {
        "a": {
            "b": jax.sharding.NamedSharding(
                mesh_obj,
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
    param_val = _ParamHolder(jnp.zeros((1,), dtype=jnp.float32))

    state_str: StateNode = {"1": param_val}
    assign_weights([1], tensor, state_str, "st_key", None)
    assign_weights_from_eval_shape([1], tensor, state_str, "st_key", None)

    state_int: StateNode = {1: param_val}
    assign_weights(["1"], tensor, state_int, "st_key", None)
    assign_weights_from_eval_shape(["1"], tensor, state_int, "st_key", None)

    state_missing: StateNode = {"2": param_val}
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

    keys_only = _KeysOnlyFile()
    assert list(_iter_safe_open_keys(keys_only)) == ["k1"]
    assert keys_only.get_tensor("k1").shape == (1,)

    class _SafeOpenErrorCtx:
        """Context manager raising TypeError on enter."""

        def __init__(self, *_a: str, **_k: str) -> None:
            """Initialize `_SafeOpenErrorCtx`."""

        def __enter__(self) -> SafeOpenFileProtocol:
            """Raise TypeError.

            Raises:
                TypeError: Always raised to simulate an invalid safetensors file.

            """
            raise TypeError("Simulated TypeError")

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: types.TracebackType | None,
        ) -> None:
            """Exit context."""

    monkeypatch.setattr(up, "safe_open", _SafeOpenErrorCtx)
    up._load_weights_from_safetensors_file("path", {}, {})

    fake_reader = _IterSafeOpenFile()

    class _SafeOpenOkCtx:
        """Context manager yielding `_IterSafeOpenFile`."""

        def __init__(self, *_a: str, **_k: str) -> None:
            """Initialize `_SafeOpenOkCtx`."""

        def __enter__(self) -> SafeOpenFileProtocol:
            """Return `fake_reader`."""
            return fake_reader

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: types.TracebackType | None,
        ) -> None:
            """Exit context."""

    def _fake_map_key(_m: Mapping[str, KeyMapType], k: str) -> tuple[str | None, None]:
        """Map layer keys to transformer.layer.0.

        Returns:
            A tuple of mapped JAX key (or `None`) and `None` transform.

        """
        return ("transformer.layer.0" if "layers" in k else None, None)

    assigned_paths: list[list[up.StateKey]] = []

    def _record_assign(
        keys: Sequence[up.StateKey],
        _tensor: up.ArrayLikeTensor,
        _state_dict: StateNode,
        _st_key: str,
        _transform: up.TransformType,
    ) -> None:
        """Record mapped parameter path."""
        assigned_paths.append(list(keys))

    def _raise_key_error(
        _keys: Sequence[up.StateKey],
        _tensor: up.ArrayLikeTensor,
        _state_dict: StateNode,
        _st_key: str,
        _transform: up.TransformType,
    ) -> None:
        """Raise KeyError to verify missing state key handling.

        Raises:
            KeyError: Always raised to simulate an unmapped state dictionary key.

        """
        raise KeyError("Simulated key error")

    monkeypatch.setattr(up, "safe_open", _SafeOpenOkCtx)
    monkeypatch.setattr(up, "map_to_jax_key", _fake_map_key)
    monkeypatch.setattr(up, "assign_weights", _record_assign)
    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})
    assert assigned_paths == [["transformer", "layer", "0"]]

    monkeypatch.setattr(up, "assign_weights", _raise_key_error)
    up._load_weights_from_safetensors_file("file", {}, {"r": ("a", None)})

    (tmp_path / "model.safetensors").write_text("dummy", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    loaded_files: list[str] = []

    def _record_load_file(
        filepath: str,
        _state: StateNode,
        _mapping: Mapping[str, KeyMapType],
    ) -> None:
        """Record discovered safetensors filepath."""
        loaded_files.append(filepath)

    monkeypatch.setattr(up, "_load_weights_from_safetensors_file", _record_load_file)
    _populate_state_from_files(str(tmp_path), {}, {})
    assert len(loaded_files) == 1


def test_create_model_from_safe_tensors_flows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify create_model_from_safe_tensors handles missing dirs, missing safe_open, and strict update errors."""
    from flax import nnx

    orig_safe_open = up.safe_open

    class _StubNonNNX:
        """Non-NNX model stub testing the empty state fallback branch."""

        def __init__(self, _cfg: str, *, rngs: nnx.Rngs) -> None:
            """Initialize stub."""
            _ = rngs

    class _StubNNXModule(nnx.Module):
        """Tiny NNX module testing the `nnx.split` state extraction branch."""

        def __init__(self, _cfg: str, *, rngs: nnx.Rngs) -> None:
            """Initialize tiny NNX module."""
            _ = rngs
            self.w = nnx.Param(jnp.ones((2,), dtype=jnp.float32))

    model_non_nnx, state_empty = _get_model_and_state(_StubNonNNX, "cfg")
    assert model_non_nnx is not None
    assert state_empty == {}

    def _fake_split(*_a: _StubNNXModule) -> tuple[str, dict[str, str]]:
        """Return graph and state dict."""
        return ("graph", {"k": "v"})

    def _noop_populate(*_a: str | StateNode | Mapping[str, KeyMapType]) -> None:
        """No-op populate state."""

    monkeypatch.setattr(up.nnx, "split", _fake_split)
    _, state2 = _get_model_and_state(_StubNNXModule, "cfg")
    assert state2 == {"k": "v"}

    res_missing = create_model_from_safe_tensors(str(tmp_path / "nonexistent"), _StubNNXModule, "cfg", {})
    assert res_missing is not None

    monkeypatch.setattr(up, "safe_open", None)
    res_no_st = create_model_from_safe_tensors(str(tmp_path), _StubNNXModule, "cfg", {})
    assert res_no_st is not None

    def _raise_update(*_a: _StubNNXModule | dict[str, str]) -> None:
        """Raise ValueError on nnx.update.

        Raises:
            ValueError: Always raised to simulate an `nnx.update` failure.

        """
        raise ValueError("simulated error updating state")

    monkeypatch.setattr(up, "safe_open", orig_safe_open)
    monkeypatch.setattr(up, "_populate_state_from_files", _noop_populate)
    monkeypatch.setattr(up.nnx, "update", _raise_update)

    create_model_from_safe_tensors(str(tmp_path), _StubNNXModule, "cfg", {}, strict=False)
    with pytest.raises(ExportError, match="simulated error updating state"):
        create_model_from_safe_tensors(str(tmp_path), _StubNNXModule, "cfg", {}, strict=True)


def test_real_safetensors_roundtrip_into_jax_gemma4(tmp_path: Path) -> None:
    """Verify real safetensors file saved to disk loads into JaxGemma4ForCausalLM end-to-end."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
    from gemma_4_sql.backends.jax.gemma4.params import _get_key_and_transform_mapping
    from tests.utils.tensor_helpers import flax_save_safetensors as save_file

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
