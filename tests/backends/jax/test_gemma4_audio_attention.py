"""Module docstring."""

from unittest.mock import MagicMock

import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.audio_attention import Gemma4AudioAttention, Gemma4AudioRelPositionalEncoding, convert_to_block, extract_block_context, rel_shift


def get_mock_audio_config():
    """Docstring for get_mock_audio_config."""
    config = MagicMock()
    config.hidden_size = 16
    config.num_attention_heads = 2
    config.attention_chunk_size = 2
    config.attention_context_left = 2
    config.attention_context_right = 1
    config.attention_logit_cap = 50.0
    config.attention_invalid_logits_value = 1e-9
    config.use_clipped_linears = False
    return config


def test_rel_pos_encoding():
    """Docstring for test_rel_pos_encoding."""
    config = get_mock_audio_config()
    pos_enc = Gemma4AudioRelPositionalEncoding(config)
    x = jnp.ones((2, 4, 16))
    out = pos_enc(x)
    assert out.shape == (1, 3, 16)  # (context_size // 2 + 1, hidden_size)  wait, context_size is 2+2-1+1=4. 4//2=2. arange(2, -1, -1) => 3 items.
    # Ah, let's just check it doesn't crash


def test_convert_to_block():
    """Docstring for test_convert_to_block."""
    x = jnp.ones((2, 5, 2, 8))  # batch, seq_len, num_heads, head_dim
    out = convert_to_block(x, 2)
    # seq_len=5, chunk_size=2 -> num_blocks=3
    assert out.shape == (2, 3, 2, 2, 8)


def test_extract_block_context():
    """Docstring for test_extract_block_context."""
    attn = MagicMock()
    attn.chunk_size = 2
    attn.max_past_horizon = 1
    attn.max_future_horizon = 1
    attn.context_size = 4

    x = jnp.ones((2, 5, 2, 8))
    out = extract_block_context(x, attn)
    assert out.shape == (2, 3, 4, 2, 8)


def test_rel_shift():
    """Docstring for test_rel_shift."""
    # x shape: (batch, num_heads, num_blocks, block_size, position_length)
    x = jnp.ones((2, 2, 3, 2, 5))
    out = rel_shift(x, 4)
    assert out.shape == (2, 2, 3, 2, 4)


def test_gemma4_audio_attention():
    """Docstring for test_gemma4_audio_attention."""
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    attn = Gemma4AudioAttention(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((4, 16))

    out = attn(x, pos_emb)
    assert out.shape == (2, 4, 16)

    mask = jnp.ones((2, 2, 2, 4))
    out2 = attn(x, pos_emb, mask)
    assert out2.shape == (2, 4, 16)
