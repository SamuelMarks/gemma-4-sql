"""Provide module docstring."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM, init_cache, modeling

orig_jit = jax.jit
jax.jit = lambda f, *_args, **_kwargs: f


def test_modeling_coverage() -> None:
    """Execute function."""
    config = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=1, num_experts_per_tok=1)
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(config, rngs=rngs)
    cache = init_cache(config, 1, 10)
    input_ids = jnp.array([[1, 2]])
    positions = jnp.array([[0, 1]])
    model(input_ids, positions, cache=cache)
    vision_config = modeling.VisionConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4, image_size=14, patch_size=14)
    config = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vision_config=vision_config, hidden_size_per_layer_input=64)
    model_v = Gemma4ForCausalLM(config, rngs=rngs)
    pixel_values = jnp.zeros((1, 14, 14, 3))
    image_token_mask = jnp.array([[False, True]])
    model_v(input_ids, positions, pixel_values=pixel_values, image_token_mask=image_token_mask)
    model_v(input_ids, positions, pixel_values=pixel_values)
    model_v.model(input_ids, positions, per_layer_inputs=None)
    config = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2)
    model_moe = Gemma4ForCausalLM(config, rngs=rngs)
    model_moe(input_ids, positions)
    audio_config = modeling.AudioConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4, use_clipped_linears=True)
    config = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16, audio_config=audio_config)
    model_a = Gemma4ForCausalLM(config, rngs=rngs)
    input_features = jnp.zeros((1, 48, 128))
    input_features_mask = jnp.ones((1, 48))
    audio_token_mask = jnp.array([[False, True]])
    model_a(input_ids, positions, input_features=input_features, input_features_mask=input_features_mask, audio_token_mask=audio_token_mask)
    model_a(input_ids, positions, input_features=input_features, input_features_mask=None, audio_token_mask=audio_token_mask)
    config = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16, final_logit_softcapping=1.0)
    model_s = Gemma4ForCausalLM(config, rngs=rngs)
    cache = init_cache(config, 1, 10)
    modeling.forward(model_s, cache, input_ids, positions)
    config_g = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=7, num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_global_key_value_heads=1, global_head_dim=32)
    init_cache(config_g, 1, 10)
    config_s = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=7, num_attention_heads=4, num_key_value_heads=2, head_dim=16, share_kv_projections=True)
    Gemma4ForCausalLM(config_s, rngs=rngs)
    jax.jit = orig_jit


def test_preset_configs_and_sharding() -> None:
    """Execute function."""
    model_config_cls = __import__("gemma_4_sql.backends.jax.gemma4.modeling", fromlist=["ModelConfig"]).ModelConfig
    model_config_cls.gemma4_base(use_fsdp=True, use_tp=True)
    model_config_cls.gemma4_e2b(use_fsdp=True, use_tp=True)
    model_config_cls.gemma4_e4b(use_fsdp=True, use_tp=True)
    model_config_cls.gemma4_26b_a4b(use_fsdp=True, use_tp=True)
    model_config_cls.gemma4_31b(use_fsdp=True, use_tp=True)


def test_mlp_attention_sharding() -> None:
    """Execute function."""
    gemma4_attention_cls = __import__("gemma_4_sql.backends.jax.gemma4.modeling", fromlist=["Gemma4Attention"]).Gemma4Attention
    gemma4_mlp_cls = __import__("gemma_4_sql.backends.jax.gemma4.modeling", fromlist=["Gemma4MLP"]).Gemma4MLP
    rngs = nnx.Rngs(0)
    gemma4_mlp_cls(hidden_size=64, intermediate_size=128, dtype=jnp.float32, shd=None, rngs=rngs)
    config = Gemma4Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    gemma4_attention_cls(config, "local", rngs=rngs)


def test_download_and_load_pretrained_error() -> None:
    """Execute function."""
    import pytest

    from gemma_4_sql.backends.jax.gemma4.modeling import _download_and_load_pretrained

    with pytest.raises(ValueError, match="is unknown, please provide config argument"):
        _download_and_load_pretrained("unknown_model_name")


def test_from_pretrained() -> None:
    """Execute function."""
    from unittest.mock import patch

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM

    config = Gemma4Config(vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)

    with patch("gemma_4_sql.backends.jax.gemma4.modeling._download_and_load_pretrained") as mock_dl:
        Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B", config)
        mock_dl.assert_called_once_with("google/gemma-4-E2B", config)

    with patch("huggingface_hub.snapshot_download") as mock_snap, patch("gemma_4_sql.backends.jax.gemma4.modeling.create_gemma4_from_pretrained") as mock_create:
        Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B", None)
        mock_snap.assert_called_once()
        mock_create.assert_called_once()
