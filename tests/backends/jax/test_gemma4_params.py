import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights


def test_assign_weights_key_error():
    state = {"a": jnp.zeros((1,))}
    with pytest.raises(KeyError):
        assign_weights(["b"], jnp.ones((1,)), state, "st_key", None)


def test_assign_weights_permute():
    state = {"a": jnp.zeros((2, 1))}
    assign_weights(["a"], jnp.ones((1, 2)), state, "st_key", transform=((1, 0), None, False))
    assert state["a"].shape == (2, 1)


from typing import NoReturn as Never

import jax

from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained


def test_assign_weight_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_params = __import__("gemma_4_sql.backends.jax.gemma4.params", fromlist=[""])

    def raise_type_error(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            TypeError: Description.

        """
        msg = "err"
        raise TypeError(msg)

    monkeypatch.setattr(m_params, "assign_weights_from_eval_shape", raise_type_error)
    with pytest.raises(TypeError):
        m_params.process_standard_tensor(type("SF", (), {"get_tensor": lambda _self, _k: 1})(), "a", {}, {"a": ("a", type("MockTransform", (), {"value": None})())})


class MockConfig:
    """Provide class docstring."""

    hidden_size = 128
    vision_config = True


class MockModelObj:
    """Provide class docstring."""

    vision_tower = type("VT", (), {"embeddings": type("E", (), {"num_patches": 10})()})()


class MockNNX:
    """Provide class docstring."""

    def eval_shape(self, _fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockModelObj()

    def split(self, _x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return (None, type("State", (), {"to_pure_dict": lambda _self: {"model": {"embed_scale": jax.ShapeDtypeStruct((1,), jnp.bfloat16)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1,), jnp.int32)}}}})())

    def merge(self, _graph_def: object, state: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return state

    class Rngs:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""


class MockSt:
    """Provide class docstring."""

    def safe_open(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return type("CM", (), {"__enter__": lambda _self: _self, "__exit__": lambda _self, *_a: None, "__iter__": lambda _self: iter([]), "keys": lambda _self: [], "get_tensor": lambda _self, _k: jnp.array(1)})()


def test_create_gemma4_from_pretrained_missing_nnx_state(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_params = __import__("gemma_4_sql.backends.jax.gemma4.params", fromlist=[""])
    monkeypatch.setattr(m_params, "nnx", MockNNX())
    monkeypatch.setattr(m_params, "safetensors", MockSt())
    (tmp_path / "model.safetensors").touch()
    res = create_gemma4_from_pretrained(str(tmp_path), MockConfig())
    if "embed_scale" not in res["model"]:
        raise AssertionError
    if "position_ids" not in res["vision_tower"]["embeddings"]:
        raise AssertionError


import re
from unittest.mock import MagicMock

from gemma_4_sql.backends.jax.gemma4.params import _process_safetensors_file
from gemma_4_sql.backends.jax.gemma4.utils_params import _load_weights_from_safetensors_file


def test_load_weights_from_safetensors_file_key_error(monkeypatch):
    class MockFile:
        def __iter__(self):
            yield "model.layer.weight"

        def get_tensor(self, key):
            return jnp.ones((1,))

    mock_safe_open = MagicMock()
    mock_safe_open.return_value.__enter__.return_value = MockFile()
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", mock_safe_open)
    state = {}
    key_mapping = {"model.layer.weight": ("layer.weight", None)}
    _load_weights_from_safetensors_file("test.safetensors", state, key_mapping)


def test_process_safetensors_file_jax_key_not_none(monkeypatch):
    class MockFile:
        def __init__(self):
            self.keys_list = ["model.layers.0.mlp.routed_experts.w1.weight"]

        def keys(self):
            return self.keys_list

        def get_tensor(self, key):
            return jnp.ones((1,))

    mock_safe_open = MagicMock()
    mock_safe_open.return_value.__enter__.return_value = MockFile()
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.params.safetensors.safe_open", mock_safe_open)

    expert_tensors = {}
    jax_state = {"model": {"layers": {"0": {"mlp": {"routed_experts": {"w1": {"weight": MagicMock()}}}}}}}
    mapping = {"model.layers.0.mlp.routed_experts.w1.weight": ("model.layers.0.mlp.routed_experts.w1.weight", None)}
    moe_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(w1|w2|w3)\.weight")
    _process_safetensors_file("test.safetensors", moe_pattern, expert_tensors, jax_state, mapping)


def test_create_gemma4_vision_pos_ids(monkeypatch):
    class MockConfig:
        vision_config = True
        audio_config = False
        hidden_size = 64
        intermediate_size = 128
        num_hidden_layers = 1
        num_attention_heads = 4
        num_key_value_heads = 2
        head_dim = 16
        shd_cfg = MagicMock()
        dtype = jnp.float32
        vocab_size = 100

    cfg = MockConfig()

    mock_model = MagicMock()
    mock_model.vision_tower.embeddings.num_patches = 14
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.params.gemma4.Gemma4ForCausalLM", MagicMock(return_value=mock_model))
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.params._populate_state_from_files", MagicMock())

    import jax

    mock_state = {"vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1, 14), jnp.int32)}}}
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.params._get_model_and_state", MagicMock(return_value=(mock_model, mock_state)))

    create_gemma4_from_pretrained("test_dir", cfg)
