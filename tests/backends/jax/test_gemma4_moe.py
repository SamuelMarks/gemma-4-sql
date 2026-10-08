"""Module docstring."""

import jax
import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
from gemma_4_sql.backends.jax.gemma4.moe import Gemma4MoE, Gemma4RoutedExperts


def test_gemma4_routed_experts():
    """Docstring for test_gemma4_routed_experts."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
    )
    routed = Gemma4RoutedExperts(config, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 64))
    topk_indices = jnp.array([[[0, 1]] * 10] * 2)
    topk_weights = jnp.array([[[0.6, 0.4]] * 10] * 2)

    out = routed(x, topk_indices, topk_weights)
    assert out.shape == (2, 10, 64)


def test_gemma4_routed_experts_moe_intermediate():
    """Docstring for test_gemma4_routed_experts_moe_intermediate."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=256,
        num_experts=4,
    )
    routed = Gemma4RoutedExperts(config, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 64))
    topk_indices = jnp.array([[[0, 1]] * 10] * 2)
    topk_weights = jnp.array([[[0.6, 0.4]] * 10] * 2)

    out = routed(x, topk_indices, topk_weights)
    assert out.shape == (2, 10, 64)


def test_gemma4_moe():
    """Docstring for test_gemma4_moe."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    moe = Gemma4MoE(config, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 64))
    original_x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 64))
    out = moe(x, original_x)
    assert out.shape == (2, 10, 64)


def test_gemma4_moe_no_gate_logits():
    """Docstring for test_gemma4_moe_no_gate_logits."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        float32_gate_logits=False,
    )
    moe = Gemma4MoE(config, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 64))
    original_x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 64))
    out = moe(x, original_x)
    assert out.shape == (2, 10, 64)


def test_gemma4_moe_no_gate_attr():
    """Docstring for test_gemma4_moe_no_gate_attr."""
    # Simulate if gate is somehow missing or None
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    moe = Gemma4MoE(config, rngs=rngs)
    moe.gate = None

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 64))
    original_x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 64))
    out = moe(x, original_x)
    assert out.shape == (2, 10, 64)
