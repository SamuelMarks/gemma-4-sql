"""Tests for native PyTorch Gemma 4 modeling."""

import torch

from gemma_4_sql.backends.pytorch.gemma4 import (
    DynamicCache,
    Gemma4Attention,
    Gemma4Config,
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4MLP,
    Gemma4RMSNorm,
    Gemma4RotaryEmbedding,
    Gemma4RotaryEmbedding2D,
    apply_rotary_pos_emb,
)


def test_gemma4_config():
    """Test Gemma4Config."""
    config = Gemma4Config(hidden_size=512, test_kwarg=True, vision_config={"hidden_size": 128, "extra": 1}, audio_config={"hidden_size": 128, "extra": 1})
    assert config.hidden_size == 512
    assert config.vocab_size == 256000
    assert hasattr(config, "test_kwarg")
    assert config.vision_config.hidden_size == 128
    assert hasattr(config.vision_config, "extra")
    assert config.audio_config.hidden_size == 128
    assert hasattr(config.audio_config, "extra")

    config_defaults = Gemma4Config()
    assert config_defaults.vision_config.hidden_size == 1152
    assert config_defaults.audio_config.hidden_size == 768


def test_gemma4_rmsnorm():
    """Test Gemma4RMSNorm."""
    norm = Gemma4RMSNorm(dim=512)
    x = torch.randn(2, 4, 512)
    out = norm(x)
    assert out.shape == x.shape


def test_gemma4_mlp():
    """Test Gemma4MLP."""
    config = Gemma4Config(hidden_size=256, intermediate_size=512)
    mlp = Gemma4MLP(config)
    x = torch.randn(2, 4, 256)
    out = mlp(x)
    assert out.shape == (2, 4, 256)


def test_apply_rotary_pos_emb():
    """Test apply_rotary_pos_emb."""
    q = torch.randn(1, 1, 2)
    k = torch.randn(1, 1, 2)
    cos = torch.ones(1, 2)
    sin = torch.zeros(1, 2)
    position_ids = torch.zeros(1, dtype=torch.long)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    assert q_out is not None
    assert k_out is not None

    emb = Gemma4RotaryEmbedding(dim=64, max_position_embeddings=10)
    cos, sin = emb(torch.randn(1, 10, 64), seq_len=15)
    assert cos.shape == (15, 64)
    assert emb.max_seq_len_cached == 15

    emb2d = Gemma4RotaryEmbedding2D(dim=64)
    cos2d, _sin2d = emb2d(torch.randn(1, 10, 64), 14, 14)
    assert cos2d.shape == (196, 64)


def test_gemma4_attention():
    """Test Gemma4Attention."""
    config = Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64)
    attn = Gemma4Attention(config, layer_idx=0)
    x = torch.randn(2, 4, 256)
    out, past = attn(x)
    assert out.shape == (2, 4, 256)
    assert past is not None

    # Test with past
    out2, past2 = attn(x, past_key_value=past)
    assert out2.shape == (2, 4, 256)
    assert past2 is not None

    # Test with tuple past
    tuple_past = (torch.randn(2, 2, 4, 64), torch.randn(2, 2, 4, 64))
    out4, _ = attn(x, past_key_value=tuple_past)
    assert out4.shape == (2, 4, 256)

    # Test with mask
    mask = torch.randn(2, 1, 4, 4)
    out3, _ = attn(x, attention_mask=mask)
    assert out3.shape == (2, 4, 256)

    # Test SDPA path
    attn_sdpa = Gemma4Attention(Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, sliding_window=None), layer_idx=0)
    out5, _ = attn_sdpa(x)
    assert out5.shape == (2, 4, 256)


def test_gemma4_decoder_layer():
    """Test Gemma4DecoderLayer."""
    config = Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512, num_experts=1)
    layer = Gemma4DecoderLayer(config, layer_idx=0)
    x = torch.randn(2, 4, 256)
    out, past, router_logits = layer(x)
    assert out.shape == (2, 4, 256)
    assert past is not None
    assert router_logits is None

    config_moe = Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512, num_experts=4)
    layer_moe = Gemma4DecoderLayer(config_moe, layer_idx=0)
    out_moe, past_moe, router_logits_moe = layer_moe(x)
    assert out_moe.shape == (2, 4, 256)
    assert past_moe is not None
    assert router_logits_moe is not None
    assert router_logits_moe.shape == (8, 4)


def test_gemma4_for_causal_lm():
    """Test Gemma4ForCausalLM."""
    config = Gemma4Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        head_dim=64,
    )
    model = Gemma4ForCausalLM(config)
    input_ids = torch.randint(0, 1000, (2, 4))
    logits, cache = model(input_ids)
    assert logits.shape == (2, 4, 1000)
    assert cache is not None

    logits2, _cache2 = model(input_ids, past_key_values=cache)
    assert logits2.shape == (2, 4, 1000)

    gen = model.generate(input_ids, max_new_tokens=5)
    assert gen.shape == (2, 4 + 5)

    # Test multimodal and cache
    pixel_values = torch.randn(2, 3, 224, 224)
    audio_values = torch.randn(2, 1000)
    logits_mm, cache_mm = model(input_ids, pixel_values=pixel_values, audio_values=audio_values)
    assert logits_mm.shape[0] == 2
    assert cache_mm is not None

    dyn_cache = DynamicCache()
    logits_dyn, dyn_cache = model(input_ids, past_key_values=dyn_cache)
    assert logits_dyn.shape == (2, 4, 1000)
    assert dyn_cache.get_seq_length(0) == 4
