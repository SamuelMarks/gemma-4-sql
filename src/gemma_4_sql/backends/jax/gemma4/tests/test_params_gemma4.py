"""Module docstring."""

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.modeling import ModelConfig
from gemma_4_sql.backends.jax.gemma4.params import _get_key_and_transform_mapping, create_gemma4_from_pretrained
from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights_from_eval_shape, map_to_jax_key, stoi


def test_stoi() -> object:  # type: ignore[return]
    """Initialize function test_stoi."""
    expected_val = 123
    if stoi("123") != expected_val:
        raise AssertionError
    if stoi("abc") != "abc":
        raise AssertionError


def test_map_to_jax_key() -> object:  # type: ignore[return]
    """Initialize function test_map_to_jax_key."""
    mapping = _get_key_and_transform_mapping()
    (jax_key, _transform) = map_to_jax_key(mapping, "model.embed_tokens.weight")
    if jax_key != "model\\.embed_tokens\\.embedding":
        raise AssertionError
    (jax_key, _transform) = map_to_jax_key(mapping, "invalid.key")
    if jax_key is not None:
        raise AssertionError
    (jax_key, _transform) = map_to_jax_key(mapping, "model.layers.5.per_layer_projection.weight")
    if jax_key != "model\\.layers\\.5\\.per_layer_projection\\.kernel":
        raise AssertionError


def test_assign_weights_from_eval_shape() -> object:  # type: ignore[return]
    """Initialize function test_assign_weights_from_eval_shape."""
    state = {"model": {"layer": {"scale": jax.ShapeDtypeStruct((2, 2), jnp.float32)}}}
    tensor = jnp.ones((2, 2))
    assign_weights_from_eval_shape(["model", "layer", "scale"], tensor, state, "src", None)
    if not (jnp.array_equal(state["model"]["layer"]["scale"], tensor)):
        raise AssertionError
    state = {"kernel": jax.ShapeDtypeStruct((2, 3), jnp.float32)}
    tensor = jnp.ones((3, 2))
    assign_weights_from_eval_shape(["kernel"], tensor, state, "src", ((1, 0), None, False))
    if state["kernel"].shape != (2, 3):
        raise AssertionError


def test_create_gemma4_from_pretrained(tmp_path: object) -> object:  # type: ignore[return]
    """Initialize function test_create_gemma4_from_pretrained.

    Args:
    ----
    tmp_path: Description of tmp_path.

    """
    np = __import__("numpy", fromlist=[""])
    st_np = __import__("safetensors.numpy", fromlist=[""])
    cfg = ModelConfig(vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_experts=2, num_experts_per_tok=1, vision_config=ModelConfig.gemma4_base().vision_config)  # type: ignore[attr-defined]
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": np.zeros((32, 16), dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight": np.zeros((32, 16), dtype=np.float32),
        "model.layers.0.per_layer_projection.weight": np.zeros((16, 16), dtype=np.float32),
    }
    tmp_path / "model.safetensors"  # type: ignore[operator]
    st_np.save_file(tensors, str(tmp_path) + "/model.safetensors")
    model = create_gemma4_from_pretrained(str(tmp_path), cfg)
    if model is None:
        raise AssertionError
    with pytest.raises(ValueError, match=r".*"):
        create_gemma4_from_pretrained(str(tmp_path / "empty"), cfg)  # type: ignore[operator]
