import jax
import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.config import ShardConfig
from gemma_4_sql.backends.jax.gemma4.layers import (
    ConstVar,
    Gemma4ClippableLinear,
    Gemma4MLP,
    Gemma4RMSNorm,
    StatVar,
    make_embed,
    make_linear,
)


def test_make_linear():
    rngs = nnx.Rngs(0)
    linear = make_linear(10, 20, kernel_metadata={"foo": "bar"}, bias_metadata={"baz": "qux"}, rngs=rngs)
    assert isinstance(linear, nnx.Linear)
    assert linear.in_features == 10
    assert linear.out_features == 20


def test_make_embed():
    rngs = nnx.Rngs(0)
    embed = make_embed(100, 32, embedding_metadata={"foo": "bar"}, rngs=rngs)
    assert isinstance(embed, nnx.Embed)
    assert embed.num_embeddings == 100
    assert embed.features == 32


def test_gemma4_rms_norm():
    rngs = nnx.Rngs(0)
    norm = Gemma4RMSNorm(16, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16))
    out = norm(x)
    assert out.shape == (2, 16)

    # test without scale
    norm_no_scale = Gemma4RMSNorm(16, with_scale=False, rngs=rngs)
    out_no_scale = norm_no_scale(x)
    assert out_no_scale.shape == (2, 16)


def test_const_var():
    v = ConstVar(jnp.array(1.0))
    assert v.value == 1.0


def test_stat_var():
    v = StatVar(jnp.array(1.0))
    assert v.value == 1.0


def test_gemma4_clippable_linear():
    rngs = nnx.Rngs(0)
    linear = Gemma4ClippableLinear(10, 20, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10))
    out = linear(x)
    assert out.shape == (2, 20)

    linear_no_clip = Gemma4ClippableLinear(10, 20, use_clipped_linears=False, rngs=rngs)
    out_no_clip = linear_no_clip(x)
    assert out_no_clip.shape == (2, 20)


def test_gemma4_mlp():
    rngs = nnx.Rngs(0)
    mlp = Gemma4MLP(16, 64, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16))
    out = mlp(x)
    assert out.shape == (2, 16)

    # Test with ShardConfig
    shd = ShardConfig.no_sharding()
    mlp2 = Gemma4MLP(16, 64, rngs=rngs, shd=shd)
    out2 = mlp2(x)
    assert out2.shape == (2, 16)


def test_gemma4_rms_norm_with_scale():
    import jax.numpy as jnp
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4.layers import Gemma4RMSNorm

    rngs = nnx.Rngs(0)
    # Using float32 for dummy test
    norm = Gemma4RMSNorm(64, with_scale=True, eps=1e-6, dtype=jnp.float32, rngs=rngs)
    x = jnp.ones((2, 10, 64))
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_stat_var_and_make_embed():
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4.layers import StatVar, make_embed

    rngs = nnx.Rngs(0)
    var = StatVar(1.0)
    assert var.value == 1.0

    # cover line 42 in layers.py
    # make_embed uses None for _shd default
    emb = make_embed(10, 16, embedding_metadata={}, rngs=rngs)
    assert emb.embedding.value.shape == (10, 16)
