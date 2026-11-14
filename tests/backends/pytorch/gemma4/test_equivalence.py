"""Multi-layer, cached-decoding, and MoE numerical equivalence tests between JAX and PyTorch Gemma 4."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import torch
from flax import nnx

from gemma_4_sql.backends.jax.gemma4 import Gemma4Config as JaxGemma4Config
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.cache import init_cache as init_jax_cache
from gemma_4_sql.backends.jax.gemma4.layers import Gemma4RMSNorm as JaxRMSNorm
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM as JaxGemma4ForCausalLM
from gemma_4_sql.backends.pytorch.gemma4.cache import DynamicCache, StaticCache
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config as PtGemma4Config
from gemma_4_sql.backends.pytorch.gemma4.layers import Gemma4RMSNorm as PtRMSNorm
from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as PtGemma4ForCausalLM
from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch
from gemma_4_sql.type_hints import ToListArrayProtocol
from tests.conftest import requires_jax

pytestmark = requires_jax

type _NNXStateTree = Mapping[str | int, "nnx.Variable[jax.Array] | jax.Array | _NNXStateTree"]


def _flatten_nnx_state(
    state: _NNXStateTree,
    prefix: str = "",
) -> dict[str, jax.Array]:
    """Recursively flatten a nested Flax NNX state tree into dotted parameter paths.

    Args:
        state: Nested Flax NNX state mapping.
        prefix: Current dotted key prefix.

    Returns:
        Flat dictionary mapping dotted parameter names to `jax.Array` values.

    """
    flat: dict[str, jax.Array] = {}
    for k, v in state.items():
        new_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            flat.update(_flatten_nnx_state(v, new_key))
        else:
            flat[new_key] = v.value if isinstance(v, nnx.Variable) else v
    return flat


def _split_flat_state(mod: nnx.Module) -> dict[str, jax.Array]:
    """Split `mod` via `nnx.split` and return flattened `dict[str, jax.Array]`.

    Args:
        mod: Target Flax NNX module.

    Returns:
        Flattened parameter dictionary.

    """
    split_fn: Callable[[nnx.Module], tuple[nnx.GraphDef[nnx.Module], _NNXStateTree]] = vars(nnx)["split"]
    _, state = split_fn(mod)
    return _flatten_nnx_state(state)


def _pt_to_jax_arr(tensor: ToListArrayProtocol) -> jax.Array:
    """Convert a PyTorch tensor to `jax.Array` via `ToListArrayProtocol`.

    Args:
        tensor: PyTorch tensor implementing `ToListArrayProtocol`.

    Returns:
        Converted `jax.Array`.

    """
    return jnp.array(tensor.tolist())


def test_multilayer_moe_ple_kv_shared_logit_equivalence_strict() -> None:
    """Verify 6-layer (sliding + global RoPE + MoE + PLE + KV-shared) JAX vs PyTorch strict parity."""
    vocab_size = 64
    hidden_size = 16
    num_layers = 6  # GEMMA4_ATTENTION_PATTERN: layers 0..4 LOCAL_SLIDING, layer 5 GLOBAL

    jax_cfg = JaxGemma4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        hidden_size_per_layer_input=8,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=16,
        sliding_window_size=3,
        num_experts=2,
        num_experts_per_tok=1,
        num_kv_shared_layers=1,
        use_post_norms=True,
        tie_word_embeddings=True,
        global_rope_proportion=0.5,
        local_rope_proportion=1.0,
        attn_logits_soft_cap=50.0,
        final_logit_softcapping=30.0,
    )
    pt_cfg = PtGemma4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        hidden_size_per_layer_input=8,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=16,
        sliding_window=3,
        num_experts=2,
        num_experts_per_tok=1,
        num_kv_shared_layers=1,
        use_post_norms=True,
        tie_word_embeddings=True,
        enable_multimodal=False,
        global_rope_proportion=0.5,
        local_rope_proportion=1.0,
        attn_logits_soft_cap=50.0,
        final_logit_softcapping=30.0,
    )

    jax_model = JaxGemma4ForCausalLM(jax_cfg, rngs=nnx.Rngs(7))
    pt_model = PtGemma4ForCausalLM(pt_cfg)
    pt_model.embed_tokens.padding_idx = None

    pt_state_dict = translate_jax_to_pytorch(_split_flat_state(jax_model))

    # strict=True proves every single parameter across all 6 layers, MoE routers/experts, PLE, and norms maps 1-to-1
    pt_model.load_state_dict(pt_state_dict, strict=True)
    pt_model.eval()

    tokens = [[2, 5, 9, 14, 21, 3]]
    jax_ids = jnp.array(tokens, dtype=jnp.int32)
    jax_positions = jnp.arange(len(tokens[0]))[None, :]
    jax_logits = jax_model(jax_ids, positions=jax_positions)

    pt_ids = torch.tensor(tokens, dtype=torch.long)
    with torch.no_grad():
        pt_logits, _ = pt_model(pt_ids)

    assert jnp.allclose(jnp.array(pt_logits.tolist()), jax_logits, atol=1e-4)

    # Multi-step cached autoregressive decoding (exceeding sliding_window=3) must match full-sequence logits
    dyn_cache = DynamicCache()
    static_cache = StaticCache(pt_cfg, max_batch_size=1, max_cache_len=16, device=torch.device("cpu"))
    jax_cache = init_jax_cache(jax_cfg, batch_size=1, max_seq_len=16)

    with torch.no_grad():
        # Prefill first 2 tokens, then decode tokens 2..5 one token at a time (q_len=1, kv_seq_len=3..6 > sw=3)
        pt_prefill, _ = pt_model(pt_ids[:, :2], past_key_values=dyn_cache)
        pt_prefill_st, _ = pt_model(pt_ids[:, :2], past_key_values=static_cache)
        jax_prefill = jax_model(jax_ids[:, :2], positions=jnp.array([[0, 1]]), cache=jax_cache)
        assert torch.allclose(pt_prefill, pt_logits[:, :2, :], atol=1e-4)
        assert torch.allclose(pt_prefill_st, pt_logits[:, :2, :], atol=1e-4)
        assert jnp.allclose(jax_prefill, jax_logits[:, :2, :], atol=1e-4)

        for step in range(2, len(tokens[0])):
            pt_step_dyn, _ = pt_model(pt_ids[:, step : step + 1], past_key_values=dyn_cache)
            pt_step_st, _ = pt_model(pt_ids[:, step : step + 1], past_key_values=static_cache)
            jax_step = jax_model(jax_ids[:, step : step + 1], positions=jnp.array([[step]]), cache=jax_cache)

            assert torch.allclose(pt_step_dyn[:, 0, :], pt_logits[:, step, :], atol=1e-4)
            assert torch.allclose(pt_step_st[:, 0, :], pt_logits[:, step, :], atol=1e-4)
            assert jnp.allclose(jax_step[:, 0, :], jax_logits[:, step, :], atol=1e-4)

    # Verify anchor layer cache index only advanced to 6 (never double-stepped to 12 by KV-shared layer 5!)
    assert int(jax_cache[0].cur_ind.value) == 6
    assert dyn_cache.get_seq_length(0) == 6
    assert static_cache.get_seq_length(0) == 6


def test_rmsnorm_and_linear_numerical_equivalence() -> None:
    """Verify numerical equivalence of `Gemma4RMSNorm` and linear kernel transposition."""
    dim = 32
    pt_norm = PtRMSNorm(dim=dim)
    jax_norm = JaxRMSNorm(dim=dim, rngs=nnx.Rngs(0))

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
    assert jnp.allclose(_pt_to_jax_arr(pt_out), jax_out, atol=1e-5)

    in_dim, out_dim = 16, 32
    k3, k4 = jax.random.split(jax.random.PRNGKey(101), 2)
    kernel = jax.random.normal(k3, shape=(in_dim, out_dim), dtype=jnp.float32)
    translated = translate_jax_to_pytorch({"proj.kernel": kernel})
    assert translated["proj.weight"].shape == (out_dim, in_dim)
    x2_jax = jax.random.normal(k4, shape=(2, 4, in_dim), dtype=jnp.float32)
    x2_pt = torch.tensor(x2_jax.tolist(), dtype=torch.float32)
    pt_res = torch.nn.functional.linear(x2_pt, translated["proj.weight"]).detach()
    jax_res = jnp.dot(x2_jax, kernel)
    assert jnp.allclose(_pt_to_jax_arr(pt_res), jax_res, atol=1e-5)


def test_attention_multi_pos_and_moe_numerical_parity_jax_vs_pytorch() -> None:
    """Verify multi-position proportional-RoPE attention AND translated MoE outputs match numerically."""
    from gemma_4_sql.backends.jax.gemma4.attention import Gemma4Attention as JaxAttention
    from gemma_4_sql.backends.jax.gemma4.config import AttentionType
    from gemma_4_sql.backends.jax.gemma4.moe import Gemma4MoE as JaxMoE
    from gemma_4_sql.backends.pytorch.gemma4.attention import Gemma4Attention as PtAttention
    from gemma_4_sql.backends.pytorch.gemma4.moe import Gemma4MoE as PtMoE

    x_jax = jax.random.normal(jax.random.PRNGKey(123), shape=(1, 4, 16), dtype=jnp.float32)
    x_pt = torch.tensor(x_jax.tolist(), dtype=torch.float32)

    jax_cfg = JaxGemma4Config(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        share_kv_projections=True,
        global_rope_proportion=0.5,
        attn_logits_soft_cap=50.0,
        num_experts=3,
        num_experts_per_tok=2,
    )
    pt_cfg = PtGemma4Config(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        share_kv_projections=True,
        global_rope_proportion=0.5,
        attn_logits_soft_cap=50.0,
        global_attn_layers=[0],
        sliding_window=None,
        num_experts=3,
        num_experts_per_tok=2,
        enable_multimodal=False,
    )

    jax_attn = JaxAttention(jax_cfg, AttentionType.GLOBAL, rngs=nnx.Rngs(0))
    pt_attn = PtAttention(pt_cfg, layer_idx=0)
    pt_attn_sd = translate_jax_to_pytorch(_split_flat_state(jax_attn))
    pt_attn.load_state_dict(pt_attn_sd, strict=True)

    # Test across positions [0, 1, 2, 3] (non-zero RoPE angles + partial rotary dims!)
    # AND shared_kv reuse with position_ids=None.
    jax_out = jax_attn(x_jax, jnp.array([[0, 1, 2, 3]]))
    pt_out, present_kv = pt_attn(x_pt, position_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long))
    assert jnp.allclose(_pt_to_jax_arr(pt_out.detach()), jax_out, atol=1e-4)
    pt_shared_out, _ = pt_attn(x_pt[:, -1:, :], position_ids=None, shared_kv=present_kv)
    assert pt_shared_out.shape == (1, 1, 16)

    # Copy full MoE weights via translate_jax_to_pytorch with strict=True and assert numerical parity
    jax_moe = JaxMoE(jax_cfg, rngs=nnx.Rngs(11))
    # Give per_expert_scale and pre_forward_scale_2 non-unit values to catch any +1.0 / scale corruption
    jax_moe.per_expert_scale.value = jnp.array([0.7, 1.3, 0.9], dtype=jnp.float32)
    jax_moe.pre_forward_scale_2.value = jnp.linspace(0.5, 1.5, 16, dtype=jnp.float32)
    moe_prefixed = {f"layers.0.mlp.{k}": v for k, v in _split_flat_state(jax_moe).items()}
    pt_moe_sd = {k[len("layers.0.mlp.") :]: v for k, v in translate_jax_to_pytorch(moe_prefixed).items()}
    pt_moe = PtMoE(pt_cfg)
    pt_moe.load_state_dict(pt_moe_sd, strict=True)
    pt_moe.eval()

    jax_moe_out = jax_moe(x_jax, x_jax)
    with torch.no_grad():
        pt_moe_out, pt_router_logits = pt_moe(x_pt, original_hidden_states=x_pt)
    assert pt_router_logits.shape == (4, 3)
    assert jnp.allclose(jnp.array(pt_moe_out.tolist()), jax_moe_out, atol=1e-4)


def test_vision_projector_and_audio_encoder_parity_jax_vs_pytorch() -> None:
    """Verify JAX and PyTorch multimodal projectors, audio LightConv1d, and token splicing match numerically."""
    from gemma_4_sql.backends.jax.gemma4.audio import Gemma4AudioModel as JaxAudioModel
    from gemma_4_sql.backends.jax.gemma4.audio_layers import Gemma4AudioLightConv1d as JaxLightConv1d
    from gemma_4_sql.backends.jax.gemma4.config import AudioConfig as JaxAudioConfig
    from gemma_4_sql.backends.jax.gemma4.config import VisionConfig as JaxVisionConfig
    from gemma_4_sql.backends.jax.gemma4.multimodal import batched_merge_modalities
    from gemma_4_sql.backends.jax.gemma4.vision import Gemma4MultiModalProjector as JaxProjector
    from gemma_4_sql.backends.jax.gemma4.vision import SiglipVisionTransformer as JaxVisionModel
    from gemma_4_sql.backends.pytorch.gemma4.audio import Gemma4AudioModel as PtAudioModel
    from gemma_4_sql.backends.pytorch.gemma4.audio_layers import Gemma4ConformerLightConv1d as PtLightConv1d
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4AudioConfig as PtAudioConfig
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4VisionConfig as PtVisionConfig
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4MultiModalProjector as PtProjector
    from gemma_4_sql.backends.pytorch.gemma4.modeling import merge_modality_embeddings
    from gemma_4_sql.backends.pytorch.gemma4.vision import Gemma4VisionModel as PtVisionModel

    assert JaxVisionConfig().hidden_size == PtVisionConfig().hidden_size == 1152
    assert JaxAudioConfig().hidden_size == PtAudioConfig().hidden_size == 1024

    # 1. Multimodal projector numerical parity with 2x2 spatial patch pooling (16 patches -> 4 tokens)
    jax_v_cfg = JaxVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4, image_size=56, patch_size=14
    )
    jax_mm_cfg = JaxGemma4Config(hidden_size=24, mm_tokens_per_image=4, vision_config=jax_v_cfg)
    pt_mm_cfg = PtGemma4Config(
        hidden_size=24,
        mm_tokens_per_image=4,
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 56,
            "patch_size": 14,
        },
    )
    jax_proj = JaxProjector(jax_mm_cfg, rngs=nnx.Rngs(3))
    jax_proj.mm_input_projection_weight.value = jax.random.normal(jax.random.PRNGKey(9), (16, 24), dtype=jnp.float32)
    pt_proj = PtProjector(pt_mm_cfg)
    pt_proj.load_state_dict(translate_jax_to_pytorch(_split_flat_state(jax_proj)), strict=True)

    vis_patches_jax = jax.random.normal(jax.random.PRNGKey(10), (2, 16, 16), dtype=jnp.float32)
    vis_patches_pt = torch.tensor(vis_patches_jax.tolist(), dtype=torch.float32)
    jax_proj_out = jax_proj(vis_patches_jax)
    pt_proj_out = pt_proj(vis_patches_pt).detach()
    assert jax_proj_out.shape == (2, 4, 24)
    assert jnp.allclose(_pt_to_jax_arr(pt_proj_out), jax_proj_out, atol=1e-5)

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
    pt_lconv.pre_layer_norm.weight.data = torch.tensor(
        (1.0 + jax_lconv.pre_layer_norm.scale.value).tolist(), dtype=torch.float32
    )
    pt_lconv.conv_norm.weight.data = torch.tensor((1.0 + jax_lconv.conv_norm.scale.value).tolist(), dtype=torch.float32)
    pt_lconv.linear_start.weight.data = torch.tensor(
        jax_lconv.linear_start.linear.kernel.value.T.tolist(), dtype=torch.float32
    )
    pt_lconv.linear_end.weight.data = torch.tensor(
        jax_lconv.linear_end.linear.kernel.value.T.tolist(), dtype=torch.float32
    )
    dw_k = jnp.transpose(jax_lconv.depthwise_conv1d.conv.kernel.value, (2, 1, 0))
    pt_lconv.depthwise_conv1d.weight.data = torch.tensor(dw_k.tolist(), dtype=torch.float32)

    x_aud_jax = jax.random.normal(jax.random.PRNGKey(77), shape=(2, 6, 16), dtype=jnp.float32)
    x_aud_pt = torch.tensor(x_aud_jax.tolist(), dtype=torch.float32)
    jax_lconv_out = jax_lconv(x_aud_jax)
    pt_lconv_out = pt_lconv(x_aud_pt).detach()
    assert jnp.allclose(_pt_to_jax_arr(pt_lconv_out), jax_lconv_out, atol=1e-4)

    # 3. Vision & Audio encoder structural parity + multimodal token splicing
    jax_v_out = JaxVisionModel(jax_v_cfg, rngs=nnx.Rngs(0))(jnp.ones((2, 56, 56, 3)))
    pt_v_out = PtVisionModel(pt_mm_cfg.vision_config)(torch.ones((2, 3, 56, 56)))
    assert jax_v_out.shape == tuple(pt_v_out.shape) == (2, 16, 16)

    pt_full_cfg = PtGemma4Config(
        hidden_size=16,
        audio_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "conv_kernel_size": 3},
    )
    jax_aud_out = JaxAudioModel(jax_a_cfg, rngs=nnx.Rngs(0))(jnp.ones((2, 16, 8)))
    pt_aud_out = PtAudioModel(pt_full_cfg)(torch.ones((2, 16, 8)))
    assert jax_aud_out.shape == tuple(pt_aud_out.shape) == (2, 4, 16)

    text_jax = jnp.ones((2, 4, 16), dtype=jnp.float32)
    mod_jax = jnp.full((2, 2, 16), 5.0, dtype=jnp.float32)
    mask_jax = jnp.array([[False, True, False, True], [True, False, True, False]])
    merged_jax = batched_merge_modalities(mod_jax, text_jax, mask_jax)

    text_pt = torch.ones((2, 4, 16), dtype=torch.float32)
    mod_pt = torch.full((2, 2, 16), 5.0, dtype=torch.float32)
    mask_pt = torch.tensor([[False, True, False, True], [True, False, True, False]], dtype=torch.bool)
    merged_pt = merge_modality_embeddings(mod_pt, text_pt, mask_pt)
    assert jnp.allclose(_pt_to_jax_arr(merged_pt), merged_jax, atol=1e-5)


def test_untied_embeddings_rope_factors_and_edge_branches() -> None:
    """Verify untied embeddings (`tie_word_embeddings=False`), bare `scale` translation, and RoPE edge cases."""
    from gemma_4_sql.backends.pytorch.gemma4.rope import Gemma4RotaryEmbedding

    # Bare "scale" key maps to "weight" with +1.0 RMSNorm offset
    bare = translate_jax_to_pytorch({"scale": jnp.array([0.5, -0.5], dtype=jnp.float32)})
    assert torch.allclose(bare["weight"], torch.tensor([1.5, 0.5]))

    # PyTorch Gemma4RotaryEmbedding with proportion=0.0 and factor!=1.0
    rope_zero = Gemma4RotaryEmbedding(dim=8, proportion=0.0)
    cos_z, sin_z = rope_zero(torch.zeros((1, 4, 8)), seq_len=4)
    assert torch.allclose(cos_z, torch.ones_like(cos_z))
    assert torch.allclose(sin_z, torch.zeros_like(sin_z))

    rope_scaled = Gemma4RotaryEmbedding(dim=8, factor=2.0, proportion=0.5)
    cos_s, sin_s = rope_scaled(torch.zeros((1, 4, 8)), seq_len=4)
    assert cos_s.shape == (4, 8)
    assert sin_s.shape == (4, 8)

    # Untied embeddings in both JAX and PyTorch (`tie_word_embeddings=False`)
    # plus empty tuple past_key_values and oversized attention_mask.
    jax_untied_cfg = JaxGemma4Config(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        tie_word_embeddings=False,
    )
    pt_untied_cfg = PtGemma4Config(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        tie_word_embeddings=False,
        enable_multimodal=False,
    )
    jax_untied = JaxGemma4ForCausalLM(jax_untied_cfg, rngs=nnx.Rngs(5))
    pt_untied = PtGemma4ForCausalLM(pt_untied_cfg)
    assert pt_untied.embed_tokens.weight is not pt_untied.lm_head.weight
    pt_untied.embed_tokens.padding_idx = None
    pt_untied.load_state_dict(translate_jax_to_pytorch(_split_flat_state(jax_untied)), strict=True)

    ids_pt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits_empty_past, _ = pt_untied(ids_pt, past_key_values=(), attention_mask=torch.ones((1, 6), dtype=torch.long))
    logits_jax_untied = jax_untied(jnp.array([[1, 2, 3]], dtype=jnp.int32))
    assert jnp.allclose(_pt_to_jax_arr(logits_empty_past.detach()), logits_jax_untied, atol=1e-5)
