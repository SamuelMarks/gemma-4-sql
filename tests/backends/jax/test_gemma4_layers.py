"""Unit and behavioral tests for gemma4 layers."""

import jax
import jax.numpy as jnp
from flax import nnx

from tests.conftest import requires_jax

pytestmark = requires_jax

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


def test_make_linear() -> None:
    """Verify make linear behavior."""
    rngs = nnx.Rngs(0)
    linear = make_linear(10, 20, kernel_metadata={"foo": "bar"}, bias_metadata={"baz": "qux"}, rngs=rngs)
    assert isinstance(linear, nnx.Linear)
    assert linear.in_features == 10
    assert linear.out_features == 20


def test_make_embed() -> None:
    """Verify make embed behavior."""
    rngs = nnx.Rngs(0)
    embed = make_embed(100, 32, embedding_metadata={"foo": "bar"}, rngs=rngs)
    assert isinstance(embed, nnx.Embed)
    assert embed.num_embeddings == 100
    assert embed.features == 32


def test_gemma4_rms_norm() -> None:
    """Verify gemma4 rms norm behavior."""
    rngs = nnx.Rngs(0)
    norm = Gemma4RMSNorm(16, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16))
    out = norm(x)
    assert out.shape == (2, 16)

    # test without scale
    norm_no_scale = Gemma4RMSNorm(16, with_scale=False, rngs=rngs)
    out_no_scale = norm_no_scale(x)
    assert out_no_scale.shape == (2, 16)


def test_const_var() -> None:
    """Verify const var behavior."""
    v = ConstVar(jnp.array(1.0))
    assert v.value == 1.0


def test_stat_var() -> None:
    """Verify stat var behavior."""
    v = StatVar(jnp.array(1.0))
    assert v.value == 1.0


def test_gemma4_clippable_linear() -> None:
    """Verify gemma4 clippable linear behavior."""
    rngs = nnx.Rngs(0)
    linear = Gemma4ClippableLinear(10, 20, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10))
    out = linear(x)
    assert out.shape == (2, 20)

    linear_no_clip = Gemma4ClippableLinear(10, 20, use_clipped_linears=False, rngs=rngs)
    out_no_clip = linear_no_clip(x)
    assert out_no_clip.shape == (2, 20)


def test_gemma4_mlp() -> None:
    """Verify gemma4 mlp behavior."""
    rngs = nnx.Rngs(0)
    mlp = Gemma4MLP(16, 64, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16))
    out = mlp(x)
    assert out.shape == (2, 16)

    # Test with ShardConfig (both unsharded and sharded with bias specs)
    shd = ShardConfig.no_sharding()
    mlp2 = Gemma4MLP(16, 64, rngs=rngs, shd=shd)
    out2 = mlp2(x)
    assert out2.shape == (2, 16)

    shd_full = ShardConfig.default(use_fsdp=True, use_tp=True)
    mlp3 = Gemma4MLP(16, 64, rngs=rngs, shd=shd_full)
    out3 = mlp3(x, original_x=x)
    assert out3.shape == (2, 16)
