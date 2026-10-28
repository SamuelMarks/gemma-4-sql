"""Equivalence tests between JAX and PyTorch implementations."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import torch

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config as PtGemma4Config
from gemma_4_sql.backends.pytorch.gemma4.layers import Gemma4RMSNorm as PtRMSNorm
from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as PtGemma4ForCausalLM
from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch
from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
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

    def _flatten_state(
        state: Mapping[str | int, nnx.Variable[jax.Array] | jax.Array],
        prefix: str = "",
    ) -> dict[str, jax.Array]:
        """Recursively flatten nested Flax NNX state dictionary."""
        flat: dict[str, jax.Array] = {}
        for k, v in state.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, Mapping):
                flat.update(_flatten_state(v, new_key))
            elif isinstance(v, nnx.Variable):
                flat[new_key] = v.value
            else:
                flat[new_key] = v
        return flat

    jax_params_flat = _flatten_state(jax_state)
    pt_state_dict = translate_jax_to_pytorch(jax_params_flat)

    # Load into PyTorch model (strict=False)
    pt_model.load_state_dict(pt_state_dict, strict=False)
    pt_model.embed_tokens.padding_idx = None

    # JAX forward
    jax_input_ids = jnp.array([[1]], dtype=jnp.int32)
    jax_positions = jnp.arange(1)[None, :]
    jax_logits = jax_model(jax_input_ids, positions=jax_positions)

    # PyTorch forward
    pt_input_ids = torch.tensor([[1]], dtype=torch.long)
    pt_model.eval()
    pt_mask = torch.full((1, 1, 1, 1), float("-inf"))
    pt_mask = torch.triu(pt_mask, diagonal=1)

    with torch.no_grad():
        pt_logits, _ = pt_model(pt_input_ids, attention_mask=pt_mask)

    assert jnp.allclose(jnp.array(pt_logits.tolist()), jax_logits, atol=1e-4)


def test_rmsnorm_numerical_equivalence() -> None:
    """Test numerical equivalence of RMSNorm between JAX and PyTorch.

    Returns:
        None.

    """
    dim = 32
    pt_norm = PtRMSNorm(dim=dim)
    rngs = nnx.Rngs(0)
    jax_norm = JaxRMSNorm(dim=dim, rngs=rngs)

    k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)
    scale = jax.random.normal(k1, shape=(dim,), dtype=jnp.float32)
    assert jax_norm.scale is not None
    assert pt_norm.weight is not None
    jax_norm.scale.value = scale
    pt_norm.weight.data = torch.tensor((1.0 + scale).tolist(), dtype=torch.float32)

    x_jax = jax.random.normal(k2, shape=(2, 4, dim), dtype=jnp.float32)
    x_pt = torch.tensor(x_jax.tolist(), dtype=torch.float32)
    pt_out = pt_norm(x_pt).detach()
    jax_out = jax_norm(x_jax)

    assert jnp.allclose(jnp.array(pt_out.tolist()), jax_out, atol=1e-5)


def test_linear_projection_numerical_equivalence() -> None:
    """Test linear projection equivalence after JAX to PyTorch kernel translation.

    Returns:
        None.

    """
    in_dim = 16
    out_dim = 32
    k1, k2 = jax.random.split(jax.random.PRNGKey(101), 2)
    kernel = jax.random.normal(k1, shape=(in_dim, out_dim), dtype=jnp.float32)

    translated = translate_jax_to_pytorch({"proj.kernel": kernel})
    assert translated["proj.weight"].shape == (out_dim, in_dim)

    x_jax = jax.random.normal(k2, shape=(2, 4, in_dim), dtype=jnp.float32)
    x_pt = torch.tensor(x_jax.tolist(), dtype=torch.float32)
    pt_res = torch.nn.functional.linear(x_pt, translated["proj.weight"]).detach()
    jax_res = jnp.dot(x_jax, kernel)

    assert jnp.allclose(jnp.array(pt_res.tolist()), jax_res, atol=1e-5)


def test_attention_and_moe_parity_jax_vs_pytorch() -> None:
    """Verify Q/K/V RMSNorm attention and shared+routed MoE structural parity between JAX and PyTorch."""
    from gemma_4_sql.backends.jax.gemma4.attention import Gemma4Attention as JaxAttention
    from gemma_4_sql.backends.jax.gemma4.config import AttentionType
    from gemma_4_sql.backends.jax.gemma4.moe import Gemma4MoE as JaxMoE
    from gemma_4_sql.backends.pytorch.gemma4.attention import Gemma4Attention as PtAttention
    from gemma_4_sql.backends.pytorch.gemma4.moe import Gemma4MoE as PtMoE

    x_jax = jax.random.normal(jax.random.PRNGKey(123), shape=(1, 3, 16), dtype=jnp.float32)
    x_pt = torch.tensor(x_jax.tolist(), dtype=torch.float32)

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
    pt_attn.q_proj.weight.data = torch.tensor(jax_attn.q_proj.kernel.value.T.tolist(), dtype=torch.float32)
    pt_attn.k_proj.weight.data = torch.tensor(jax_attn.k_proj.kernel.value.T.tolist(), dtype=torch.float32)
    pt_attn.o_proj.weight.data = torch.tensor(jax_attn.o_proj.kernel.value.T.tolist(), dtype=torch.float32)
    assert jax_attn.q_norm.scale is not None
    assert jax_attn.k_norm.scale is not None
    assert pt_attn.q_norm.weight is not None
    assert pt_attn.k_norm.weight is not None
    pt_attn.q_norm.weight.data = torch.tensor((1.0 + jax_attn.q_norm.scale.value).tolist(), dtype=torch.float32)
    pt_attn.k_norm.weight.data = torch.tensor((1.0 + jax_attn.k_norm.scale.value).tolist(), dtype=torch.float32)

    # At position 0 for single token, RoPE is identity (cos=1, sin=0) in both JAX and PyTorch
    x_single_jax = x_jax[:, :1, :]
    x_single_pt = x_pt[:, :1, :]
    jax_out = jax_attn(x_single_jax, jnp.array([[0]]))
    pt_out, _ = pt_attn(x_single_pt, position_ids=torch.tensor([[0]], dtype=torch.long))
    assert jnp.allclose(jnp.array(pt_out.detach().tolist()), jax_out, atol=1e-4)

    # Verify MoE router and shared + routed structure
    jax_moe = JaxMoE(jax_cfg, rngs=nnx.Rngs(0))
    pt_moe = PtMoE(pt_cfg)
    jax_moe_out = jax_moe(x_jax, x_jax)
    pt_moe_out, pt_router_logits = pt_moe(x_pt, original_hidden_states=x_pt)
    assert jax_moe_out.shape == tuple(pt_moe_out.shape) == (1, 3, 16)
    assert pt_router_logits.shape == (3, 2)


def test_vision_and_audio_encoder_parity_jax_vs_pytorch() -> None:
    """Verify JAX and PyTorch Vision and Audio encoders share defaults and produce aligned output shapes."""
    from gemma_4_sql.backends.jax.gemma4.audio import Gemma4AudioModel as JaxAudioModel
    from gemma_4_sql.backends.jax.gemma4.audio_layers import Gemma4AudioLightConv1d as JaxLightConv1d
    from gemma_4_sql.backends.jax.gemma4.config import AudioConfig as JaxAudioConfig
    from gemma_4_sql.backends.jax.gemma4.config import VisionConfig as JaxVisionConfig
    from gemma_4_sql.backends.jax.gemma4.vision import SiglipVisionTransformer as JaxVisionModel
    from gemma_4_sql.backends.pytorch.gemma4.audio import Gemma4AudioModel as PtAudioModel
    from gemma_4_sql.backends.pytorch.gemma4.audio_layers import Gemma4ConformerLightConv1d as PtLightConv1d
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4AudioConfig as PtAudioConfig
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4VisionConfig as PtVisionConfig
    from gemma_4_sql.backends.pytorch.gemma4.vision import Gemma4VisionModel as PtVisionModel

    # 1. Shared architectural defaults
    assert JaxVisionConfig().hidden_size == PtVisionConfig().hidden_size == 1152
    assert JaxAudioConfig().hidden_size == PtAudioConfig().hidden_size == 1024

    # 2. LightConv1d numerical equivalence between JAX and PyTorch
    jax_a_cfg = JaxAudioConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        subsampling_conv_channels=(8, 8),
        conv_kernel_size=3,
        use_clipped_linears=False,
        output_proj_dims=16,
    )
    jax_lconv = JaxLightConv1d(jax_a_cfg, rngs=nnx.Rngs(0))
    pt_lconv = PtLightConv1d(hidden_size=16, kernel_size=3, eps=jax_a_cfg.rms_norm_eps)

    assert jax_lconv.pre_layer_norm.scale is not None
    assert jax_lconv.conv_norm.scale is not None
    assert pt_lconv.pre_layer_norm.weight is not None
    assert pt_lconv.conv_norm.weight is not None
    pt_lconv.pre_layer_norm.weight.data = torch.tensor((1.0 + jax_lconv.pre_layer_norm.scale.value).tolist(), dtype=torch.float32)
    pt_lconv.conv_norm.weight.data = torch.tensor((1.0 + jax_lconv.conv_norm.scale.value).tolist(), dtype=torch.float32)
    pt_lconv.linear_start.weight.data = torch.tensor(jax_lconv.linear_start.linear.kernel.value.T.tolist(), dtype=torch.float32)
    pt_lconv.linear_end.weight.data = torch.tensor(jax_lconv.linear_end.linear.kernel.value.T.tolist(), dtype=torch.float32)
    # JAX depthwise_conv1d.conv.kernel is (K, 1, C); PyTorch Conv1d depthwise weight is (C, 1, K)
    dw_k = jnp.transpose(jax_lconv.depthwise_conv1d.conv.kernel.value, (2, 1, 0))
    pt_lconv.depthwise_conv1d.weight.data = torch.tensor(dw_k.tolist(), dtype=torch.float32)

    x_aud_jax = jax.random.normal(jax.random.PRNGKey(77), shape=(2, 6, 16), dtype=jnp.float32)
    x_aud_pt = torch.tensor(x_aud_jax.tolist(), dtype=torch.float32)
    jax_lconv_out = jax_lconv(x_aud_jax)
    pt_lconv_out = pt_lconv(x_aud_pt).detach()
    assert jnp.allclose(jnp.array(pt_lconv_out.tolist()), jax_lconv_out, atol=1e-4)

    # 3. Vision & Audio encoder structural parity on 3D spectrograms and images
    jax_v_cfg = JaxVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14)
    pt_v_cfg = PtVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14)
    jax_v_out = JaxVisionModel(jax_v_cfg, rngs=nnx.Rngs(0))(jnp.ones((2, 28, 28, 3)))
    pt_v_out = PtVisionModel(pt_v_cfg)(torch.ones((2, 3, 28, 28)))
    assert jax_v_out.shape == tuple(pt_v_out.shape) == (2, 4, 16)

    pt_full_cfg = PtGemma4Config(
        hidden_size=16,
        audio_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "conv_kernel_size": 3},
    )
    jax_aud_out = JaxAudioModel(jax_a_cfg, rngs=nnx.Rngs(0))(jnp.ones((2, 16, 8)))
    pt_aud_out = PtAudioModel(pt_full_cfg)(torch.ones((2, 16, 8)))
    assert jax_aud_out.shape == tuple(pt_aud_out.shape) == (2, 4, 16)
