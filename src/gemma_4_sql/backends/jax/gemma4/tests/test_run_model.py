"""Core functionality for the test_run_model module."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM


def run_model(_path_root: object = None) -> object:
    """Run the Gemma 4 model.

    Args:
        _path_root: The  path root.

    Returns:
        The execution result.
    """
    config = Gemma4Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_experts=2, num_experts_per_tok=1)
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(config, rngs=rngs)
    input_ids = jnp.array([[1, 2, 3, 4]])
    positions = jnp.array([[0, 1, 2, 3]])
    model(input_ids, positions)


if __name__ == "__main__":
    run_model()


def test_run_model() -> object:
    """Test the run_model behavior."""
    run_model()
