"""Unit and behavioral tests for gemma4 moe."""

import jax
import jax.numpy as jnp
from flax import nnx

from tests.conftest import requires_jax

pytestmark = requires_jax

from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
from gemma_4_sql.backends.jax.gemma4.moe import Gemma4MoE, Gemma4RoutedExperts


def test_gemma4_routed_experts():
    """Execute and verify `test_gemma4_routed_experts` behavior and invariants."""
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
    """Execute and verify `test_gemma4_routed_experts_moe_intermediate` behavior and invariants."""
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
    """Execute and verify `test_gemma4_moe` behavior and invariants."""
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
    """Execute and verify `test_gemma4_moe_no_gate_logits` behavior and invariants."""
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
    """Execute and verify `test_gemma4_moe_no_gate_attr` behavior and invariants."""
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
