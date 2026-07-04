# Copyright 2024
"""Core functionality for the test_params_gemma4_extra module."""

from unittest import mock

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM, ModelConfig
from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained
from gemma_4_sql.backends.jax.gemma4.rope import segment_ids_to_positions
from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape, map_to_jax_key


def test_map_to_jax_key_multiple() -> object:
    """Test the map_to_jax_key_multiple behavior."""
    mapping = {"a": ("b", None), ".*": ("c", None)}
    with pytest.raises(ValueError, match=r".*"):
        map_to_jax_key(mapping, "a")


def test_assign_weights_shape_mismatch() -> object:
    """Test the assign_weights_shape_mismatch behavior."""
    state = {"model": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    tensor = jnp.ones((3, 3))
    with pytest.raises(ValueError, match=r".*"):
        assign_weights_from_eval_shape(["model"], tensor, state, "src", None)
    state2 = {"model": jnp.zeros((8, 8))}
    with pytest.raises(ValueError, match=r".*"):
        assign_weights(["model"], tensor, state2, "src", None)
    state3 = {"model": {"layer": jnp.zeros((3, 3))}}
    assign_weights(["model", "layer"], tensor, state3, "src", None)
    assert jnp.array_equal(state3["model"]["layer"], tensor)


def test_assign_weights_sharding() -> object:
    """Test the assign_weights_sharding behavior."""
    state = {"model": jnp.zeros((8, 8))}
    tensor = jnp.ones((8, 8))
    sharding = {"model": jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("x",)), jax.sharding.PartitionSpec("x"))}
    assign_weights(["model"], tensor, state, "src", None, sharding_dict=sharding)


def test_segment_ids_to_positions() -> object:
    """Test the segment_ids_to_positions behavior."""
    ids = jnp.array([[1, 1, 0, 1]])
    out = segment_ids_to_positions(ids)
    assert out.shape == (1, 4)


def test_gemma4_from_pretrained(tmp_path: object) -> object:
    """Test the gemma4 from pretrained behavior.

    Args:
    ----
    tmp_path: The tmp_path parameter required for this operation.

    """
    np = __import__("numpy", fromlist=[""])
    st_np = __import__("safetensors.numpy", fromlist=[""])
    cfg = ModelConfig(vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_experts=2, num_experts_per_tok=1, vision_config=None)
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 16), dtype=np.float32),
        "invalid.key": np.zeros((10,), dtype=np.float32),
        "model.layers.0.per_layer_projection.weight": np.zeros((1, 1), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((16,), dtype=np.float32),
    }
    file_path = tmp_path / "model.safetensors"
    st_np.save_file(tensors, str(file_path))
    create_gemma4_from_pretrained(str(tmp_path), cfg)


@mock.patch("huggingface_hub.snapshot_download")
def test_gemma4_causal_from_pretrained(mock_download: object, tmp_path: object) -> object:
    """Test the gemma4 causal from pretrained behavior.

    Args:
    ----
    mock_download: The mock_download parameter required for this operation.
    tmp_path: The tmp_path parameter required for this operation.



    """
    mock_download.return_value = str(tmp_path)
    np = __import__("numpy", fromlist=[""])
    st_np = __import__("safetensors.numpy", fromlist=[""])
    st_np.save_file({"model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32)}, str(tmp_path) + "/model.safetensors")
    with pytest.raises(ValueError, match=r".*"):
        Gemma4ForCausalLM.from_pretrained("unknown_model")
    cfg = ModelConfig(vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_experts=2, num_experts_per_tok=1, vision_config=None)
    model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B", config=cfg)
    assert model is not None


def test_gemma4_causal_from_pretrained_no_config(tmp_path: object) -> object:
    """Test the gemma4 causal from pretrained no config behavior.

    Args:
    ----
    tmp_path: The tmp_path parameter required for this operation.



    """
    with mock.patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = str(tmp_path)
        np = __import__("numpy", fromlist=[""])
        st_np = __import__("safetensors.numpy", fromlist=[""])
        st_np.save_file({"model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32)}, str(tmp_path) + "/model.safetensors")
        model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B")
        assert model is not None
