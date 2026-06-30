"""Module docstring."""

from unittest import mock

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM, ModelConfig
from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained
from gemma_4_sql.backends.jax.gemma4.rope import segment_ids_to_positions
from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape, map_to_jax_key


def test_map_to_jax_key_multiple() -> object:  # type: ignore[return]
    """Initialize function test_map_to_jax_key_multiple."""
    mapping = {"a": ("b", None), ".*": ("c", None)}
    with pytest.raises(ValueError, match=r".*"):
        map_to_jax_key(mapping, "a")


def test_assign_weights_shape_mismatch() -> object:  # type: ignore[return]
    """Initialize function test_assign_weights_shape_mismatch."""
    state = {"model": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    tensor = jnp.ones((3, 3))
    with pytest.raises(ValueError, match=r".*"):
        assign_weights_from_eval_shape(["model"], tensor, state, "src", None)
    state2 = {"model": jnp.zeros((8, 8))}
    with pytest.raises(ValueError, match=r".*"):
        assign_weights(["model"], tensor, state2, "src", None)

    state3 = {"model": {"layer": jnp.zeros((3, 3))}}
    assign_weights(["model", "layer"], tensor, state3, "src", None)
    if not jnp.array_equal(state3["model"]["layer"], tensor):
        raise AssertionError


def test_assign_weights_sharding() -> object:  # type: ignore[return]
    """Initialize function test_assign_weights_sharding."""
    state = {"model": jnp.zeros((8, 8))}
    tensor = jnp.ones((8, 8))
    sharding = {"model": jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("x",)), jax.sharding.PartitionSpec("x"))}
    assign_weights(["model"], tensor, state, "src", None, sharding_dict=sharding)


def test_segment_ids_to_positions() -> object:  # type: ignore[return]
    """Initialize function test_segment_ids_to_positions."""
    ids = jnp.array([[1, 1, 0, 1]])
    out = segment_ids_to_positions(ids)
    if not out.shape == (1, 4):
        raise AssertionError


def test_gemma4_from_pretrained(tmp_path: object) -> object:  # type: ignore[return]
    """Initialize function test_gemma4_from_pretrained.

    Args:
    ----
    tmp_path: Description of tmp_path.

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
    file_path = tmp_path / "model.safetensors"  # type: ignore[operator]
    st_np.save_file(tensors, str(file_path))
    create_gemma4_from_pretrained(str(tmp_path), cfg)


@mock.patch("huggingface_hub.snapshot_download")
def test_gemma4_causal_from_pretrained(mock_download: object, tmp_path: object) -> object:  # type: ignore[return]
    """Initialize function test_gemma4_causal_from_pretrained.

    Args:
    ----
    mock_download: Description of mock_download.
    tmp_path: Description of tmp_path.

    """
    mock_download.return_value = str(tmp_path)  # type: ignore[attr-defined]
    np = __import__("numpy", fromlist=[""])
    st_np = __import__("safetensors.numpy", fromlist=[""])
    st_np.save_file({"model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32)}, str(tmp_path) + "/model.safetensors")
    with pytest.raises(ValueError, match=r".*"):
        Gemma4ForCausalLM.from_pretrained("unknown_model")
    cfg = ModelConfig(vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_experts=2, num_experts_per_tok=1, vision_config=None)
    model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B", config=cfg)
    if not model is not None:
        raise AssertionError


def test_gemma4_causal_from_pretrained_no_config(tmp_path: object) -> object:  # type: ignore[return]
    """Initialize function test_gemma4_causal_from_pretrained_no_config.

    Args:
    ----
    tmp_path: Description of tmp_path.

    """
    with mock.patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = str(tmp_path)
        np = __import__("numpy", fromlist=[""])
        st_np = __import__("safetensors.numpy", fromlist=[""])
        st_np.save_file({"model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32)}, str(tmp_path) + "/model.safetensors")
        model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B")
        if not model is not None:
            raise AssertionError
