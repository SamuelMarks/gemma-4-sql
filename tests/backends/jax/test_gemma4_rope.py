"""Module docstring."""

import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.rope import RoPE, apply_rope, default_rope_params, segment_ids_to_positions


def test_segment_ids_to_positions():
    """Docstring for test_segment_ids_to_positions."""
    segment_ids = jnp.array([[1, 1, 1], [1, 0, 1]])
    positions = segment_ids_to_positions(segment_ids)
    assert jnp.allclose(positions, jnp.array([[1, 2, 3], [1, 1, 2]]))


def test_default_rope_params():
    """Docstring for test_default_rope_params."""
    positions = jnp.array([1, 2, 3])
    freq, factor = default_rope_params(positions, head_dim=4, rope_theta=10000, factor=2.0)
    assert jnp.allclose(freq, jnp.array([0.5, 0.005]))
    assert factor == 1.0


def test_apply_rope():
    """Docstring for test_apply_rope."""
    x = jnp.ones((2, 3, 2, 4))
    sin = jnp.zeros((2, 3, 2))
    cos = jnp.ones((2, 3, 2))

    out = apply_rope(x, sin, cos)
    assert out.shape == x.shape
    assert jnp.allclose(out, x)

    with pytest.raises(AssertionError):
        apply_rope(jnp.zeros((2, 3, 4)), sin, cos)
    with pytest.raises(AssertionError):
        apply_rope(x, jnp.zeros((2, 3)), cos)
    with pytest.raises(AssertionError):
        apply_rope(x, sin, jnp.zeros((2, 3)))


def test_rope_module():
    """Docstring for test_rope_module."""
    rope = RoPE(rope_type="default", head_dim=4, rope_theta=10000, factor=2.0)
    positions = jnp.array([[1, 2]])
    sin, cos = rope(positions)
    assert sin.shape == (1, 2, 2)
    assert cos.shape == (1, 2, 2)
