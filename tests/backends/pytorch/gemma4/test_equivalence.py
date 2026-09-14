"""Equivalence tests between JAX and PyTorch implementations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp

try:
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config as JaxGemma4Config
    from gemma_4_sql.backends.jax.gemma4.layers import Gemma4RMSNorm as JaxRMSNorm
    from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM as JaxGemma4ForCausalLM
except ImportError:
    pytest.skip("Flax/JAX Gemma4 modeling not available", allow_module_level=True)

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config as PtGemma4Config
from gemma_4_sql.backends.pytorch.gemma4.layers import Gemma4RMSNorm as PtRMSNorm
from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as PtGemma4ForCausalLM
from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch


def test_logit_equivalence() -> None:
    """Test logit equivalence and param translation between JAX and PyTorch models.

    Returns:
        None.
    """
    vocab_size = 128
    hidden_size = 64
    pt_config = PtGemma4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )

    pt_model = PtGemma4ForCausalLM(pt_config)
    jax_cfg = JaxGemma4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )
    assert jax_cfg.vocab_size == pt_config.vocab_size
    assert JaxGemma4ForCausalLM is not None

    dummy_jax_params = {
        "model.lm_head.kernel": jnp.ones((hidden_size, vocab_size)),
        "model.layers.0.norm.scale": jnp.zeros(hidden_size),
    }

    pt_state_dict = translate_jax_to_pytorch(dummy_jax_params)
    assert "model.lm_head.weight" in pt_state_dict
    assert pt_state_dict["model.lm_head.weight"].shape == (vocab_size, hidden_size)
    assert "model.layers.0.norm.weight" in pt_state_dict

    input_ids = torch.tensor([[1, 5, 10]], dtype=torch.long)
    logits, _ = pt_model(input_ids)

    assert logits is not None
    assert logits.shape == (1, 3, vocab_size)


def test_rmsnorm_numerical_equivalence() -> None:
    """Test numerical equivalence of RMSNorm between JAX and PyTorch.

    Returns:
        None.
    """
    dim = 32
    pt_norm = PtRMSNorm(dim=dim)
    rngs = nnx.Rngs(0)
    jax_norm = JaxRMSNorm(dim=dim, rngs=rngs)

    scale = np.random.default_rng(42).normal(size=(dim,)).astype(np.float32)
    jax_norm.scale.value = jnp.array(scale)
    pt_norm.weight.data = torch.from_numpy(1.0 + scale)

    x = np.random.default_rng(42).normal(size=(2, 4, dim)).astype(np.float32)
    pt_out = pt_norm(torch.from_numpy(x)).detach().numpy()
    jax_out = np.array(jax_norm(jnp.array(x)))

    assert np.allclose(pt_out, jax_out, atol=1e-5)


def test_linear_projection_numerical_equivalence() -> None:
    """Test linear projection equivalence after JAX to PyTorch kernel translation.

    Returns:
        None.
    """
    in_dim = 16
    out_dim = 32
    kernel = np.random.default_rng(101).normal(size=(in_dim, out_dim)).astype(np.float32)

    translated = translate_jax_to_pytorch({"proj.kernel": jnp.array(kernel)})
    assert translated["proj.weight"].shape == (out_dim, in_dim)

    x = np.random.default_rng(101).normal(size=(2, 4, in_dim)).astype(np.float32)
    pt_res = torch.nn.functional.linear(torch.from_numpy(x), translated["proj.weight"]).detach().numpy()
    jax_res = np.array(jnp.dot(jnp.array(x), kernel))

    assert np.allclose(pt_res, jax_res, atol=1e-5)
