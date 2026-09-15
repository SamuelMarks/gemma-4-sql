"""Tests for test gemma4 vision module."""

from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
from gemma_4_sql.backends.jax.gemma4.vision import Gemma4MultimodalEmbedder


def test_multimodal_embedder_no_audio(monkeypatch):
    """Test multimodal embedder no audio functionality."""
    from flax import nnx

    config = ModelConfig()
    config.audio_config = None

    rngs = nnx.Rngs(0)
    embedder = Gemma4MultimodalEmbedder(config, rngs=rngs)

    assert embedder.embedding_projection is not None


def test_vision_projector_no_vision_config():
    """Test Gemma4MultiModalProjector raises ValueError when vision_config is None."""
    import pytest
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4.vision import Gemma4MultiModalProjector

    config = ModelConfig()
    config.vision_config = None
    rngs = nnx.Rngs(0)
    with pytest.raises(ValueError, match="Vision config is required for Gemma4VisionProjector"):
        Gemma4MultiModalProjector(config, rngs=rngs)
