"""Unit and behavioral tests for gemma4 audio attention."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

from flax import nnx

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.audio_attention import (
    Gemma4AudioAttention,
    Gemma4AudioRelPositionalEncoding,
    convert_to_block,
    extract_block_context,
    rel_shift,
)
from gemma_4_sql.backends.jax.gemma4.config import AudioConfig


def make_audio_config() -> AudioConfig:
    """Create a miniature `AudioConfig` instance for audio attention tests.

    Returns:
        A small `AudioConfig` configured for attention block testing.

    """
    return AudioConfig(
        hidden_size=16,
        num_attention_heads=2,
        attention_chunk_size=2,
        attention_context_left=2,
        attention_context_right=1,
        attention_logit_cap=50.0,
        attention_invalid_logits_value=1e-9,
        use_clipped_linears=False,
    )


def test_rel_pos_encoding() -> None:
    """Verify rel pos encoding behavior."""
    config = make_audio_config()
    pos_enc = Gemma4AudioRelPositionalEncoding(config)
    x = jnp.ones((2, 4, 16))
    out = pos_enc(x)
    assert out.shape == (1, 3, 16)


def test_convert_to_block() -> None:
    """Verify convert to block behavior."""
    x = jnp.ones((2, 5, 2, 8))
    out = convert_to_block(x, 2)
    assert out.shape == (2, 3, 2, 2, 8)


def test_extract_block_context() -> None:
    """Verify extract block context behavior."""
    attn = Gemma4AudioAttention(make_audio_config(), rngs=nnx.Rngs(0))
    x = jnp.ones((2, 5, 2, 8))
    out = extract_block_context(x, attn)
    assert out.shape == (2, 3, 4, 2, 8)


def test_rel_shift() -> None:
    """Verify rel shift behavior."""
    x = jnp.ones((2, 2, 3, 2, 5))
    out = rel_shift(x, 4)
    assert out.shape == (2, 2, 3, 2, 4)


def test_gemma4_audio_attention() -> None:
    """Verify gemma4 audio attention behavior."""
    config = make_audio_config()
    rngs = nnx.Rngs(0)
    attn = Gemma4AudioAttention(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((4, 16))

    out = attn(x, pos_emb)
    assert out.shape == (2, 4, 16)

    mask = jnp.ones((2, 2, 2, 4))
    out2 = attn(x, pos_emb, mask)
    assert out2.shape == (2, 4, 16)
