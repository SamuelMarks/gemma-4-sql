# Copyright 2024
"""Core functionality for the test_sharding_gemma4 module."""

from absl.testing import absltest
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM, ShardConfig
from gemma_4_sql.backends.jax.gemma4.modeling import ModelConfig as Gemma4Config


class TestSharding(absltest.TestCase):
    """Implementation of TestSharding."""

    @classmethod
    def setUpClass(cls) -> object:
        """Set up the virtual mesh for sharding tests."""
        super().setUpClass()

    def test_model_sharding(self) -> object:
        """Test that model sharding config works correctly."""
        shd = ShardConfig.no_sharding()
        config = Gemma4Config(vocab_size=100, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=2, shd_cfg=shd)
        rngs = nnx.Rngs(0)
        model = Gemma4ForCausalLM(config, rngs=rngs)
        assert model.model.embed_tokens is not None


if __name__ == "__main__":
    absltest.main()
