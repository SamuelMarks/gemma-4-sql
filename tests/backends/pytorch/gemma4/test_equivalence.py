"""Equivalence tests between JAX and PyTorch implementations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config as PtGemma4Config
from gemma_4_sql.backends.pytorch.gemma4.layers import Gemma4RMSNorm as PtRMSNorm
from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as PtGemma4ForCausalLM
from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch

jax = pytest.importorskip("jax")

from flax import nnx

from gemma_4_sql.backends.jax.gemma4 import Gemma4Config as JaxGemma4Config
from gemma_4_sql.backends.jax.gemma4.layers import Gemma4RMSNorm as JaxRMSNorm
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM as JaxGemma4ForCausalLM


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
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )

    pt_model = PtGemma4ForCausalLM(pt_config)
    # Untie weights to match JAX independent initialization
    from torch import nn

    pt_model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    jax_cfg = JaxGemma4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        num_experts=1,
    )

    rngs = nnx.Rngs(0)
    jax_model = JaxGemma4ForCausalLM(jax_cfg, rngs=rngs)

    # Get JAX params and translate to PyTorch
    _, jax_state = nnx.split(jax_model)

    def _flatten_state(state: Any, prefix: str = "") -> dict[str, Any]:
        """Recursively flatten nested Flax NNX state dictionary."""
        flat: dict[str, Any] = {}
        for k, v in state.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if hasattr(v, "items"):
                flat.update(_flatten_state(v, new_key))
            else:
                flat[new_key] = getattr(v, "value", v)
        return flat

    jax_params_flat = _flatten_state(jax_state)
    pt_state_dict = translate_jax_to_pytorch(jax_params_flat)

    # Load into PyTorch model (strict=False)
    pt_model.load_state_dict(pt_state_dict, strict=False)
    pt_model.embed_tokens.padding_idx = None

    input_ids_np = np.array([[1]])

    # JAX forward
    jax_input_ids = jnp.array(input_ids_np)
    jax_positions = jnp.arange(1)[None, :]
    jax_logits = np.array(jax_model(jax_input_ids, positions=jax_positions))

    # PyTorch forward
    pt_input_ids = torch.from_numpy(input_ids_np)
    pt_model.eval()
    pt_mask = torch.full((1, 1, 1, 1), float("-inf"))
    pt_mask = torch.triu(pt_mask, diagonal=1)

    with torch.no_grad():
        pt_logits, _ = pt_model(pt_input_ids, attention_mask=pt_mask)
        pt_logits_np = pt_logits.numpy()

    assert np.allclose(pt_logits_np, jax_logits, atol=1e-4)


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
    assert jax_norm.scale is not None
    assert pt_norm.weight is not None
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


def test_attention_and_moe_parity_jax_vs_pytorch() -> None:
    """Verify Q/K/V RMSNorm attention and shared+routed MoE structural parity between JAX and PyTorch."""
    from gemma_4_sql.backends.jax.gemma4.attention import Gemma4Attention as JaxAttention
    from gemma_4_sql.backends.jax.gemma4.config import AttentionType
    from gemma_4_sql.backends.jax.gemma4.moe import Gemma4MoE as JaxMoE
    from gemma_4_sql.backends.pytorch.gemma4.attention import Gemma4Attention as PtAttention
    from gemma_4_sql.backends.pytorch.gemma4.moe import Gemma4MoE as PtMoE

    rng = np.random.default_rng(123)
    x_np = rng.normal(size=(1, 3, 16)).astype(np.float32)

    jax_cfg = JaxGemma4Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        share_kv_projections=True,
        attn_logits_soft_cap=50.0,
        num_experts=2,
        num_experts_per_tok=1,
    )
    pt_cfg = PtGemma4Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        share_kv_projections=True,
        attn_logits_soft_cap=50.0,
        global_attn_layers=[0],
        sliding_window=None,
        num_experts=2,
        num_experts_per_tok=1,
    )

    jax_attn = JaxAttention(jax_cfg, AttentionType.GLOBAL, rngs=nnx.Rngs(0))
    pt_attn = PtAttention(pt_cfg, layer_idx=0)
    assert pt_attn.v_proj is None

    # Copy Q, K, O projections and Q/K norms from JAX to PyTorch
    pt_attn.q_proj.weight.data = torch.from_numpy(np.array(jax_attn.q_proj.kernel.value).T.copy())
    pt_attn.k_proj.weight.data = torch.from_numpy(np.array(jax_attn.k_proj.kernel.value).T.copy())
    pt_attn.o_proj.weight.data = torch.from_numpy(np.array(jax_attn.o_proj.kernel.value).T.copy())
    assert jax_attn.q_norm.scale is not None
    assert jax_attn.k_norm.scale is not None
    assert pt_attn.q_norm.weight is not None
    assert pt_attn.k_norm.weight is not None
    pt_attn.q_norm.weight.data = torch.from_numpy(1.0 + np.array(jax_attn.q_norm.scale.value).copy())
    pt_attn.k_norm.weight.data = torch.from_numpy(1.0 + np.array(jax_attn.k_norm.scale.value).copy())

    # At position 0 for single token, RoPE is identity (cos=1, sin=0) in both JAX and PyTorch
    x_single = x_np[:, :1, :]
    jax_out = np.array(jax_attn(jnp.array(x_single), jnp.array([[0]])))
    pt_out, _ = pt_attn(torch.from_numpy(x_single), position_ids=torch.tensor([[0]], dtype=torch.long))
    assert np.allclose(pt_out.detach().numpy(), jax_out, atol=1e-4)

    # Verify MoE router and shared + routed structure
    jax_moe = JaxMoE(jax_cfg, rngs=nnx.Rngs(0))
    pt_moe = PtMoE(pt_cfg)
    jax_moe_out = np.array(jax_moe(jnp.array(x_np), jnp.array(x_np)))
    pt_moe_out, pt_router_logits = pt_moe(torch.from_numpy(x_np), original_hidden_states=torch.from_numpy(x_np))
    assert jax_moe_out.shape == pt_moe_out.shape == (1, 3, 16)
    assert pt_router_logits.shape == (3, 2)
