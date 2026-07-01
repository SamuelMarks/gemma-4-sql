import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.rope import apply_rope


def test_apply_rope_ndim_assertion():
    x = jnp.ones((2, 2))
    sin = jnp.ones((2, 2))
    cos = jnp.ones((2, 2))
    with pytest.raises(AssertionError):
        apply_rope(x, sin, cos)
