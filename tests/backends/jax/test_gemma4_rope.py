"""Unit and behavioral tests for gemma4 rope."""

import pytest

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from tests.conftest import requires_jax

pytestmark = requires_jax

from gemma_4_sql.backends.jax.gemma4.rope import RoPE, apply_rope, default_rope_params, segment_ids_to_positions


def test_segment_ids_to_positions() -> None:
    """Verify segment ids to positions behavior."""
    segment_ids = jnp.array([[1, 1, 1], [1, 0, 1]])
    positions = segment_ids_to_positions(segment_ids)
    assert jnp.allclose(positions, jnp.array([[1, 2, 3], [1, 1, 2]]))


def test_default_rope_params() -> None:
    """Verify default rope params behavior including linear factor and proportional partial rotation."""
    positions = jnp.array([1, 2, 3])
    freq, factor = default_rope_params(positions, head_dim=4, rope_theta=10000, factor=2.0)
    assert jnp.allclose(freq, jnp.array([0.5, 0.005]))
    assert factor == 1.0

    partial_freq, _ = default_rope_params(positions, head_dim=8, rope_theta=10000, factor=1.0, proportion=0.5)
    assert partial_freq.shape == (4,)
    assert jnp.allclose(partial_freq[2:], jnp.zeros((2,)))
    assert bool(jnp.all(partial_freq[:2] > 0.0))

    zero_freq, _ = default_rope_params(positions, head_dim=8, rope_theta=10000, factor=1.0, proportion=0.0)
    assert jnp.allclose(zero_freq, jnp.zeros((4,)))


def test_apply_rope() -> None:
    """Verify apply rope behavior."""
    x = jnp.ones((2, 3, 2, 4))
    sin = jnp.zeros((2, 3, 2))
    cos = jnp.ones((2, 3, 2))

    out = apply_rope(x, sin, cos)
    assert out.shape == x.shape
    assert jnp.allclose(out, x)

    with pytest.raises(ValueError, match=r"Expected \(x\.ndim, sin\.ndim, cos\.ndim\) == \(4, 3, 3\)"):
        apply_rope(jnp.zeros((2, 3, 4)), sin, cos)
    with pytest.raises(ValueError, match=r"Expected \(x\.ndim, sin\.ndim, cos\.ndim\) == \(4, 3, 3\)"):
        apply_rope(x, jnp.zeros((2, 3)), cos)
    with pytest.raises(ValueError, match=r"Expected \(x\.ndim, sin\.ndim, cos\.ndim\) == \(4, 3, 3\)"):
        apply_rope(x, sin, jnp.zeros((2, 3)))


def test_rope_module() -> None:
    """Verify rope module behavior."""
    rope = RoPE(rope_type="default", head_dim=4, rope_theta=10000, factor=2.0)
    positions = jnp.array([[1, 2]])
    sin, cos = rope(positions)
    assert sin.shape == (1, 2, 2)
    assert cos.shape == (1, 2, 2)
