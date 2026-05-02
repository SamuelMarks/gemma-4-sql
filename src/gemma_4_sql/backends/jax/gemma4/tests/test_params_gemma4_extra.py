import pytest
import jax.numpy as jnp
import jax
from unittest import mock
import safetensors.flax as safetensors
import os
from flax import nnx

from ..params import _get_key_and_transform_mapping, create_gemma4_from_pretrained
from ..utils_params import stoi, map_to_jax_key, assign_weights_from_eval_shape, assign_weights
from ..modeling import ModelConfig, Gemma4ForCausalLM
from ..rope import segment_ids_to_positions

def test_map_to_jax_key_multiple():
    # Force multiple matches
    mapping = {"a": ("b", None), ".*": ("c", None)}
    with pytest.raises(ValueError):
        map_to_jax_key(mapping, "a")

def test_assign_weights_shape_mismatch():
    state = {"model": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    tensor = jnp.ones((3, 3))
    with pytest.raises(ValueError):
        assign_weights_from_eval_shape(["model"], tensor, state, "src", None)
        
    state2 = {"model": jnp.zeros((8, 8))}
    with pytest.raises(ValueError):
        assign_weights(["model"], tensor, state2, "src", None, None)

def test_assign_weights():
    state = {"model": {"layer": jnp.zeros((8, 8))}}
    tensor = jnp.ones((8, 8))
    assign_weights(["model", "layer"], tensor, state, "src", None, None)
    assert jnp.array_equal(state["model"]["layer"], tensor)

def test_assign_weights_sharding():
    # just cover the branches
    state = {"model": jnp.zeros((8, 8))}
    tensor = jnp.ones((8, 8))
    sharding = {"model": jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('x',)), jax.sharding.PartitionSpec('x'))}
    assign_weights(["model"], tensor, state, "src", None, sharding)

def test_segment_ids_to_positions():
    ids = jnp.array([[1, 1, 0, 1]])
    out = segment_ids_to_positions(ids)
    assert out.shape == (1, 4)

def test_gemma4_from_pretrained(tmp_path):
    import safetensors.numpy as st_np
    import numpy as np
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
        vision_config=None,
    )
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 16), dtype=np.float32),
        "invalid.key": np.zeros((10,), dtype=np.float32),
        "model.layers.0.per_layer_projection.weight": np.zeros((1, 1), dtype=np.float32), # trigger shape mismatch exception handling
        "model.layers.0.input_layernorm.weight": np.zeros((16,), dtype=np.float32),
    }
    file_path = tmp_path / "model.safetensors"
    st_np.save_file(tensors, str(file_path))
    
    # Should catch errors in try-except block
    model = create_gemma4_from_pretrained(str(tmp_path), cfg)
    
@mock.patch("huggingface_hub.snapshot_download")
def test_gemma4_causal_from_pretrained(mock_download, tmp_path):
    mock_download.return_value = str(tmp_path)
    import safetensors.numpy as st_np
    import numpy as np
    st_np.save_file({"model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32)}, str(tmp_path / "model.safetensors"))
    
    with pytest.raises(ValueError):
        Gemma4ForCausalLM.from_pretrained("unknown_model")
        
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
        vision_config=None,
    )
    model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B", config=cfg)
    assert model is not None

def test_gemma4_causal_from_pretrained_no_config(tmp_path):
    with mock.patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = str(tmp_path)
        import safetensors.numpy as st_np
        import numpy as np
        st_np.save_file({"model.embed_tokens.weight": np.zeros((10, 16), dtype=np.float32)}, str(tmp_path / "model.safetensors"))
        model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B")
        assert model is not None

