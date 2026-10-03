import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.config import AttentionType, ModelConfig
from gemma_4_sql.backends.jax.gemma4.decoder_layer import Gemma4DecoderLayer


def test_decoder_layer_mlp():
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=1,
    )
    layer = Gemma4DecoderLayer(config, AttentionType.GLOBAL, rngs=rngs)

    x = jnp.ones((2, 10, 64))
    positions = jnp.arange(10)[None, :]

    # Test with cache and mask
    class DummyCache:
        def __init__(self):
            import jax.numpy as jnp
            from flax import nnx

            self.k_cache = nnx.Variable(jnp.zeros((2, 10, 2, 16)))
            self.v_cache = nnx.Variable(jnp.zeros((2, 10, 2, 16)))
            self.cur_ind = nnx.Variable(jnp.array(0))

    layer_cache = DummyCache()
    attention_mask = jnp.ones((2, 1, 10, 10))
    out = layer(x, positions, layer_cache=layer_cache, attention_mask=attention_mask)
    assert out.shape == (2, 10, 64)

    # Test without cache and mask
    out2 = layer(x, positions)
    assert out2.shape == (2, 10, 64)


def test_decoder_layer_moe():
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=2,
    )
    layer = Gemma4DecoderLayer(config, AttentionType.GLOBAL, rngs=rngs)

    x = jnp.ones((2, 10, 64))
    positions = jnp.arange(10)[None, :]
    out = layer(x, positions)
    assert out.shape == (2, 10, 64)


def test_decoder_layer_per_layer_input():
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=1,
        hidden_size_per_layer_input=32,
    )
    layer = Gemma4DecoderLayer(config, AttentionType.GLOBAL, rngs=rngs)

    x = jnp.ones((2, 10, 64))
    positions = jnp.arange(10)[None, :]
    per_layer_input = jnp.ones((2, 10, 32))

    out = layer(x, positions, per_layer_input=per_layer_input)
    assert out.shape == (2, 10, 64)

    # Trigger branch where per_layer_input is None but hidden_size_per_layer_input is truthy
    out2 = layer(x, positions, per_layer_input=None)
    assert out2.shape == (2, 10, 64)
