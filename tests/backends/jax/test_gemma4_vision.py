from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
from gemma_4_sql.backends.jax.gemma4.vision import Gemma4MultimodalEmbedder


def test_multimodal_embedder_no_audio(monkeypatch):
    from flax import nnx

    config = ModelConfig()
    config.audio_config = None

    rngs = nnx.Rngs(0)
    embedder = Gemma4MultimodalEmbedder(config, rngs=rngs)

    assert embedder.embedding_projection is not None
