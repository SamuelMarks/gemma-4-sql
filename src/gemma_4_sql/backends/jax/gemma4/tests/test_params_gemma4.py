import jax
import jax.numpy as jnp
import pytest

from ..modeling import ModelConfig
from ..params import _get_key_and_transform_mapping, create_gemma4_from_pretrained
from ..utils_params import assign_weights_from_eval_shape, map_to_jax_key, stoi


def test_stoi():
    assert stoi("123") == 123
    assert stoi("abc") == "abc"


def test_map_to_jax_key():
    mapping = _get_key_and_transform_mapping()
    # Test valid
    jax_key, transform = map_to_jax_key(mapping, "model.embed_tokens.weight")
    assert jax_key == r"model\.embed_tokens\.embedding"

    # Test invalid
    jax_key, transform = map_to_jax_key(mapping, "invalid.key")
    assert jax_key is None

    # Test regex group
    jax_key, transform = map_to_jax_key(
        mapping, "model.layers.5.per_layer_projection.weight"
    )
    assert jax_key == r"model\.layers\.5\.per_layer_projection\.kernel"


def test_assign_weights_from_eval_shape():
    state = {"model": {"layer": {"scale": jax.ShapeDtypeStruct((2, 2), jnp.float32)}}}
    tensor = jnp.ones((2, 2))
    assign_weights_from_eval_shape(
        ["model", "layer", "scale"], tensor, state, "src", None
    )
    assert jnp.array_equal(state["model"]["layer"]["scale"], tensor)

    # Transpose
    state = {"kernel": jax.ShapeDtypeStruct((2, 3), jnp.float32)}
    tensor = jnp.ones((3, 2))
    assign_weights_from_eval_shape(
        ["kernel"], tensor, state, "src", ((1, 0), None, False)
    )
    assert state["kernel"].shape == (2, 3)


def test_create_gemma4_from_pretrained(tmp_path):
    # Create empty safetensors
    import numpy as np
    import safetensors.numpy as st_np

    cfg = ModelConfig(
        vocab_size=10,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        vision_config=ModelConfig.gemma4_base().vision_config,
    )

    # Create fake tensors that match some patterns
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": np.zeros(
            (32, 16), dtype=np.float32
        ),
        "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight": np.zeros(
            (32, 16), dtype=np.float32
        ),
        "model.layers.0.per_layer_projection.weight": np.zeros(
            (16, 16), dtype=np.float32
        ),
    }

    file_path = tmp_path / "model.safetensors"
    st_np.save_file(tensors, str(file_path))

    # Should run without error
    model = create_gemma4_from_pretrained(str(tmp_path), cfg)
    assert model is not None

    # Error on no files
    with pytest.raises(ValueError):
        create_gemma4_from_pretrained(str(tmp_path / "empty"), cfg)
